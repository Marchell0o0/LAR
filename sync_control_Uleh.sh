#!/bin/bash

# Define the local and remote directories
LOCAL_DIR="/mnt/d/CVUT/4_semester/LAR_Mark/LAR/"
REMOTE_DIR="borysole@192.168.65.30:/home.nfs/borysole/control/"

# Sync the directories
rsync -avz $LOCAL_DIR $REMOTE_DIR

echo "Sync complete."
