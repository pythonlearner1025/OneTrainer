#!/bin/bash

# Build the Docker image
sudo docker build -t trainer .
sudo docker build -t comfy ComfyUI

# Check and remove existing containers
if sudo docker ps -a | grep -q comfyct; then
  sudo docker remove comfyct -f
fi

if sudo docker ps -a | grep -q trainerct; then
  sudo docker remove trainerct -f
fi

SCRIPT_PATH=$(realpath "$0")
ROOTDIR=$(dirname "$SCRIPT_PATH")
VOLUME_PATH="$ROOTDIR/datavolume"
mount_ot="--mount type=bind,source=$VOLUME_PATH,target=/$VOLUME_PATH"
mount_comfy="--mount type=bind,source=$VOLUME_PATH,target=/$VOLUME_PATH"
default_fs="--rm --gpus all"

sudo docker run $default_fs $mount_comfy -p 127.0.0.1:8188:8188 --env-file ComfyUI/.deploy-envs --name comfyct comfy &
#sudo docker remove new -f
sudo docker run $default_fs $mount_ot --env-file .deploy-envs --name trainerct trainer &

