#!/bin/bash

# Define the local and remote directories
LOCAL_DIR="/Users/mark/Programming/LAR/"
REMOTE_DIR="horpymar@192.168.65.29:/home.nfs/horpymar/control/"

# Sync the directories
rsync -avz --delete $LOCAL_DIR $REMOTE_DIR

echo "Sync complete."
