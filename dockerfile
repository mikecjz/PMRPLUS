# start from nvidia pytorch image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

## Set workdir in Docker Container
# set default workdir(app) in your docker container
# In other words your scripts will run from this directory
RUN mkdir /app
WORKDIR /app

## Install python in Docker image
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip

## Copy only requirements.txt first for better caching
COPY requirements.txt /app/requirements.txt

## Install requirements (this layer will be cached if requirements.txt doesn't change)
RUN pip3 install -r /app/requirements.txt

## Copy the rest of the application files
COPY . /app

RUN chmod 777 /app/last.ckpt


# Define build argument with default value
ARG CONFIG_FILE=configs/inference/pmr-plus/cmr25-task2-docker.yaml
ENV CONFIG_FILE=${CONFIG_FILE}

RUN echo "CONFIG_FILE is set to: ${CONFIG_FILE}"

## Entrypoint
ENTRYPOINT python main.py predict --config ${CONFIG_FILE}