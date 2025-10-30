#!/bin/bash

# Singularity Manager Script for PromptMR+ Task R2
# Usage: ./singularity_manager.sh --build|--run [options]

set -e

# Default values
DOCKER_IMAGE=""
DOCKER_TAR=""
INPUT_DIR=""
OUTPUT_DIR=""
SINGULARITY_IMAGE=""
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [--build|--run] [options]"
    echo ""
    echo "Commands:"
    echo "  --build    Build a Singularity image from Docker image or tar file"
    echo "  --run      Run the Singularity container"
    echo ""
    echo "Options:"
    echo "  --docker-image <IMAGE>      Full Docker image name (e.g., docker.synapse.org/syn12345678/task-r2:latest)"
    echo "                             Required for build command (when not using --docker-tar)"
    echo "  --docker-tar <TAR_FILE>     Path to Docker image tar file"
    echo "                             Alternative to --docker-image for build command"
    echo "  --input <DIR>               Input directory for run (required for run)"
    echo "  --output <DIR>              Output directory for run (required for run)"
    echo "  --singularity-image <PATH>  Path to Singularity image file"
    echo "                             For build: output path (auto-generated if not specified)"
    echo "                             For run: input path (required)"
    echo ""
    echo "Examples:"
    echo "  Build from Docker daemon: $0 --build --docker-image docker.synapse.org/syn12345678/task-r2:latest"
    echo "  Build from tar file: $0 --build --docker-tar ./my-docker-image.tar"
    echo "  Build with custom output: $0 --build --docker-tar ./image.tar --singularity-image ./my-image.sif"
    echo "  Run:   $0 --run --singularity-image ./my-image.sif --input /path/to/input --output /path/to/output"
}

build_singularity() {
    # Check if singularity is installed
    if ! command -v singularity &> /dev/null; then
        echo -e "${RED}Error: Singularity is not installed or not in PATH${NC}"
        echo "Please install Singularity first: https://sylabs.io/guides/3.0/user-guide/installation.html"
        exit 1
    fi

    # Check that either docker image or docker tar is provided, but not both
    if [ -z "$DOCKER_IMAGE" ] && [ -z "$DOCKER_TAR" ]; then
        echo -e "${RED}Error: Either Docker image or Docker tar file is required for building Singularity image${NC}"
        echo "Use --docker-image <full_docker_image_name> OR --docker-tar <path_to_tar_file>"
        echo "Example: --docker-image docker.synapse.org/syn12345678/task-r2:latest"
        echo "Example: --docker-tar ./my-docker-image.tar"
        exit 1
    fi
    
    if [ -n "$DOCKER_IMAGE" ] && [ -n "$DOCKER_TAR" ]; then
        echo -e "${RED}Error: Cannot specify both --docker-image and --docker-tar. Choose one.${NC}"
        exit 1
    fi
    
    # Validate tar file exists if specified
    if [ -n "$DOCKER_TAR" ] && [ ! -f "$DOCKER_TAR" ]; then
        echo -e "${RED}Error: Docker tar file does not exist: ${DOCKER_TAR}${NC}"
        exit 1
    fi
    
    # Auto-generate singularity image name if not provided
    if [ -z "$SINGULARITY_IMAGE" ]; then
        if [ -n "$DOCKER_IMAGE" ]; then
            # Extract meaningful parts from docker image name for the filename
            # Convert docker.synapse.org/syn12345678/task-r2:latest -> task-r2-latest.sif
            local image_base=$(echo "$DOCKER_IMAGE" | sed 's|.*/||' | tr ':' '-')
            SINGULARITY_IMAGE="./${image_base}.sif"
        else
            # Generate name from tar file
            local tar_base=$(basename "$DOCKER_TAR" .tar)
            SINGULARITY_IMAGE="./${tar_base}.sif"
        fi
    fi
    
    if [ -n "$DOCKER_TAR" ]; then
        echo -e "${GREEN}Building Singularity image from Docker tar file: ${DOCKER_TAR}${NC}"
        echo -e "${YELLOW}Output: ${SINGULARITY_IMAGE}${NC}"
        echo -e "${YELLOW}This may take several minutes...${NC}"
        
        # Build Singularity image from Docker tar file
        singularity build "${SINGULARITY_IMAGE}" "docker-archive://${DOCKER_TAR}"
    else
        echo -e "${GREEN}Building Singularity image from Docker image: ${DOCKER_IMAGE}${NC}"
        echo -e "${YELLOW}Output: ${SINGULARITY_IMAGE}${NC}"
        echo -e "${YELLOW}This may take several minutes...${NC}"
        
        # Check if Docker image exists locally
        if ! docker image inspect "${DOCKER_IMAGE}" &> /dev/null; then
            echo -e "${YELLOW}Docker image not found locally. Attempting to pull...${NC}"
            if ! docker pull "${DOCKER_IMAGE}"; then
                echo -e "${RED}Error: Could not pull Docker image: ${DOCKER_IMAGE}${NC}"
                echo -e "${YELLOW}Please ensure the image exists and you have access to it${NC}"
                exit 1
            fi
        fi
        
        # Build Singularity image from Docker image
        singularity build "${SINGULARITY_IMAGE}" "docker-daemon://${DOCKER_IMAGE}"
    fi
    
    echo -e "${GREEN}Successfully built Singularity image: ${SINGULARITY_IMAGE}${NC}"
    echo -e "${YELLOW}You can run it with:${NC}"
    echo "./singularity_manager.sh --run --singularity-image ${SINGULARITY_IMAGE} --input /path/to/input --output /path/to/output"
}

run_singularity() {
    # Check if singularity is installed
    if ! command -v singularity &> /dev/null; then
        echo -e "${RED}Error: Singularity is not installed or not in PATH${NC}"
        echo "Please install Singularity first: https://sylabs.io/guides/3.0/user-guide/installation.html"
        exit 1
    fi

    if [ -z "$SINGULARITY_IMAGE" ]; then
        echo -e "${RED}Error: Singularity image path is required for running${NC}"
        echo "Use --singularity-image /path/to/your/image.sif"
        exit 1
    fi

    if [ ! -f "$SINGULARITY_IMAGE" ]; then
        echo -e "${RED}Error: Singularity image file does not exist: ${SINGULARITY_IMAGE}${NC}"
        exit 1
    fi
    
    if [ -z "$INPUT_DIR" ]; then
        echo -e "${RED}Error: Input directory is required for running${NC}"
        echo "Use --input /path/to/your/input/folder"
        exit 1
    fi
    
    if [ -z "$OUTPUT_DIR" ]; then
        echo -e "${RED}Error: Output directory is required for running${NC}"
        echo "Use --output /path/to/your/output/folder"
        exit 1
    fi
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo -e "${RED}Error: Input directory does not exist: ${INPUT_DIR}${NC}"
        exit 1
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    echo -e "${GREEN}Running Singularity container with GPU support${NC}"
    echo -e "${YELLOW}Image:  ${SINGULARITY_IMAGE}${NC}"
    echo -e "${YELLOW}Input:  ${INPUT_DIR}${NC}"
    echo -e "${YELLOW}Output: ${OUTPUT_DIR}${NC}"
    
    # Run Singularity container with GPU support and volume bindings
    singularity run --nv \
        --pwd /app \
        -B "${INPUT_DIR}:/input" \
        -B "${OUTPUT_DIR}:/output" \
        "${SINGULARITY_IMAGE}"
    
    echo -e "${GREEN}Singularity container finished successfully${NC}"
    echo -e "${YELLOW}Results saved to: ${OUTPUT_DIR}${NC}"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            COMMAND="build"
            shift
            ;;
        --run)
            COMMAND="run"
            shift
            ;;
        --docker-image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --docker-tar)
            DOCKER_TAR="$2"
            shift 2
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --singularity-image)
            SINGULARITY_IMAGE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Check that --docker-image and --docker-tar cannot be specified at the same time
if [ -n "$DOCKER_IMAGE" ] && [ -n "$DOCKER_TAR" ]; then
    echo -e "${RED}Error: Cannot specify both --docker-image and --docker-tar. Choose one.${NC}"
    echo "Use either --docker-image <image_name> OR --docker-tar <tar_file_path>"
    print_usage
    exit 1
fi

case $COMMAND in
    build)
        build_singularity
        ;;
    run)
        run_singularity
        ;;
    *)
        echo -e "${RED}Error: Must specify either --build or --run${NC}"
        print_usage
        exit 1
        ;;
esac