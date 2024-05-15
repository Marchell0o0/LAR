#!/bin/bash

# Define the local and remote directories
LOCAL_DIR="/Users/mark/Programming/LAR/"
REMOTE_DIR="horpymar@192.168.65.32:/home.nfs/horpymar/control/"

# Sync the directories
rsync -avz $LOCAL_DIR $REMOTE_DIR

echo "Sync complete."