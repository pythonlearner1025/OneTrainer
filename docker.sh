#!/bin/bash
# Build the Docker image
sudo docker build -t trainer .
# Run the Docker container with GPU access, volume mount, and environment variables
sudo docker run --rm -it --gpus all --mount type=bind,source=/home/minjune/Desktop/workspace,target=/data --env-file .deploy-envs --name new trainer