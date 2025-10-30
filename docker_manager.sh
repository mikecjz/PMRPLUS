#!/bin/bash

# Docker Manager Script for PromptMR+ Task R2
# Usage: ./docker_manager.sh --build|--run [options]

set -e

# Default values
PROJECT_ID="syn68754167"
TAG="latest"
INPUT_DIR=""
OUTPUT_DIR=""
TASK="task-s2"
CKPT="$HOME/HPC-LIDXXLAB/Yi/augmented_data_training_results1025_intensity_FOV_combined1/16cascade_with_data_augmentation_continue/cmr2024_2025_augmented/wp6m2z1q/restored_ckpts/best-epochepoch=01-valvalidation_loss=0.0145.ckpt"
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [--build|--run] [options]"
    echo ""
    echo "Commands:"
    echo "  --build    Build the Docker image"
    echo "  --run      Run the Docker container"
    echo ""
    echo "Options:"
    echo "  --project-id <ID>     Synapse project ID (required for build)"
    echo "  --tag <TAG>           Docker image tag (default: latest)"
    echo "  --task <TASK>         Task name (default: task-s2)"
    echo "  --ckpt <PATH>         Checkpoint file used for building"
    echo "  --input <DIR>         Input directory for run (required for run)"
    echo "  --output <DIR>        Output directory for run (required for run)"
    echo ""
    echo "Examples:"
    echo "  Build: $0 --build --project-id syn12345678 --tag v1"
    echo "  Run:   $0 --run --input /path/to/input --output /path/to/output --project-id syn12345678 --tag v1"
}

build_docker() {
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: Project ID is required for building${NC}"
        echo "Use --project-id <your_synapse_project_id>"
        exit 1
    fi

    #Copy checkpoint file to current directory
    echo -e "${GREEN}Copying checkpoint file: ${CKPT}${NC}"
    cp $CKPT ./last.ckpt

    local image_name="docker.synapse.org/${PROJECT_ID}/${TASK}:${TAG}"
    
    echo -e "${GREEN}Building Docker image: ${image_name}${NC}"
    echo -e "${YELLOW}This may take several minutes...${NC}"

    if [ "$TASK" == "task-r1" ]; then
        CONFIG_FILE="configs/inference/pmr-plus/cmr25-task1-docker.yaml"
    elif [ "$TASK" == "task-r2" ]; then
        CONFIG_FILE="configs/inference/pmr-plus/cmr25-task2-docker.yaml"
    elif [ "$TASK" == "task-s1" ]; then
        CONFIG_FILE="configs/inference/pmr-plus/cmr25-tasks1-docker.yaml"
    elif [ "$TASK" == "task-s2" ]; then
        CONFIG_FILE="configs/inference/pmr-plus/cmr25-tasks2-docker.yaml"
    else
        echo -e "${RED}Error: Invalid task: ${TASK}${NC}"
        exit 1
    fi
    
    docker build -t "${image_name}" --build-arg CONFIG_FILE="${CONFIG_FILE}" .
    
    echo -e "${GREEN}Successfully built Docker image: ${image_name}${NC}"
    echo -e "${YELLOW}You can now push it to Synapse with:${NC}"
    echo "docker push ${image_name}"
}

run_docker() {
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
    
    local image_name
    if [ -n "$PROJECT_ID" ]; then
        image_name="docker.synapse.org/${PROJECT_ID}/${TASK}:${TAG}"
    else
        # If no project ID provided for run, assume local build
        image_name="promptmr-plus-${TASK}:${TAG}"
    fi
    
    echo -e "${GREEN}Running Docker container with GPU support${NC}"
    echo -e "${YELLOW}Input:  ${INPUT_DIR}${NC}"
    echo -e "${YELLOW}Output: ${OUTPUT_DIR}${NC}"
    echo -e "${YELLOW}Image:  ${image_name}${NC}"
    
    # Always use all GPUs, but CUDA_VISIBLE_DEVICES will limit visibility inside container
    GPU_ARG="--gpus all"
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        echo -e "${YELLOW}Using GPUs: ${CUDA_VISIBLE_DEVICES} (via CUDA_VISIBLE_DEVICES)${NC}"
    else
        echo -e "${YELLOW}Using all available GPUs${NC}"
    fi
    
    docker run -it --rm \
        --shm-size=16g \
        ${GPU_ARG} \
        -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        -v "${INPUT_DIR}:/input" \
        -v "${OUTPUT_DIR}:/output" \
        "${image_name}"
    
    echo -e "${GREEN}Docker container finished successfully${NC}"
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
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
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
        --ckpt)
            CKPT="$2"
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

case $COMMAND in
    build)
        build_docker
        ;;
    run)
        run_docker
        ;;
    *)
        echo -e "${RED}Error: Must specify either --build or --run${NC}"
        print_usage
        exit 1
        ;;
esac
