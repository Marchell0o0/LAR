#!/bin/bash

# Define the local and remote directories
LOCAL_DIR="/mnt/d/CVUT/4_semester/LAR_Mark/LAR/"
REMOTE_DIR="borysole@192.168.65.32:/home.nfs/borysole/control/"

# Sync local to remote
rsync -avz --delete "$LOCAL_DIR" "$REMOTE_DIR"

# Sync remote to local
rsync -avz --delete "$REMOTE_DIR" "$LOCAL_DIR"

echo "Sync complete."
