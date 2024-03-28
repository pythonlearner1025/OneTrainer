#!/bin/bash

# Set the output log file path

# Start the ComfyUI server in the background and redirect stdout and stderr to the log file
exec python3 ComfyUI/main.py --listen &

sleep 30
# Start the training script
exec python3 scripts/train.py &

wait -n

exit $?