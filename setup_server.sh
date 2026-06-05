#!/bin/bash
# Server Environment Setup Script
# This script configures the symbolic links and environment for the Docker container.

echo "🔧 Setting up server environment..."

# 1. Fix the Isaac Sim symbolic link
if [ -L "_isaac_sim" ]; then
    echo "🔄 Recreating symbolic link for Isaac Sim..."
    rm -rf _isaac_sim
fi
ln -s /isaac-sim _isaac_sim

# 2. Grant execution permissions to scripts
chmod +x *.sh

echo "✅ Setup complete! You can now run 'bash train_go2.sh'"
