#!/usr/bin/env bash
# Script to tune NVIDIA H100 GPU on GCP
# To stablize performance

set -ex

# Enable persistent mode
sudo nvidia-smi -pm 1
# Lock power limit to 650W
sudo nvidia-smi -pl 650

# Default Memory Frequency: 2619 MHz
# Default Graphics Frequency: 1980 MHz
sudo nvidia-smi -lgc 1980,1980
sudo nvidia-smi -lmc 2619,2619
sudo nvidia-smi -ac 2619,1980
