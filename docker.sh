#!/bin/bash

# Build the Docker image
sudo docker build -t deploy-ot .

# Check and remove existing containers
if sudo docker ps -a | grep -q trainerct; then
  sudo docker remove trainerct -f
fi

default_fs="--rm --gpus all"

sudo docker run $default_fs $mount_ot --env-file .deploy-envs --name trainerct trainer &

