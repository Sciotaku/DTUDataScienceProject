#!/bin/sh

### General LSF options ###
# Request one GPU (change as needed)
# - We specify the type of GPU, if needed (e.g., gpu_v100)
# - We request 4 hours of time (you can adjust based on your need)
#BSUB -n 1                # Number of CPU cores
#BSUB -R "rusage[mem=8GB]" # Memory request
#BSUB -q gpuv100           # Queue for GPUs
#BSUB -o job_output.log    # Standard output file
#BSUB -e job_error.log     # Standard error file

# Load environment variables
source .env   # Make sure you have a .env file with necessary variables like REPO, ENV1, etc.

# Set the repository path (this is your working directory)
REPO=/zhome/bb/9/212485/DTUDataScienceProject/myproject

# Create output directory if it doesn't exist
if [[ ! -d ${REPO}/job_out ]]; then
    mkdir -p ${REPO}/job_out
fi

# Load necessary modules
module load python3/3.10.12  # Load the Python module (adjust version as needed)
module load cuda/11.3         # Load CUDA for GPU support

# Activate your virtual environment (now inside the project folder)
source ${REPO}/.venv/bin/activate  # Activate your 'myenv' virtual environment inside myproject

# Print the Python version and CUDA version to ensure everything is set correctly
python --version
nvidia-smi

# Run your sentiment analysis script
python3 reccengine.py  # Ensure that the script is in the working directory

# The script should process your dataset, and the results will be saved to a CSV file