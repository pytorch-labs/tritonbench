#!/usr/bin/env bash
# Script to tune NVIDIA H100 GPU on GCP
# To reset GPU status

# Reset GPU and Memory clocks
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc

# Restore the default power limit (500W)
sudo nvidia-smi -pl 500

# Disable persistent mode
sudo nvidia-smi -pm 0
