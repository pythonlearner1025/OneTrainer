#!/bin/bash

# Set the output log file path

exec python3 main.py --listen &

sleep 30
# Start the training script
exec python3 scripts/train.py &

wait -n

exit $?