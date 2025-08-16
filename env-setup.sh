#!/bin/bash

# Prompt the user for environment name
echo "Please enter environment name:"
read ENV_NAME

# Prompt the user for Python version
echo "Please enter Python version (e.g., 3.9, 3.10, 3.11) or press Enter for latest:"
read PYTHON_VERSION

# Set default Python version if none specified
if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_SPEC="python"
    echo "Creating: $ENV_NAME with latest Python version"
else
    PYTHON_SPEC="python=$PYTHON_VERSION"
    echo "Creating: $ENV_NAME with Python $PYTHON_VERSION"
fi

# Create a data directory
mkdir -p data

# Create data subdirectories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/interim

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Create the conda environment
echo "Creating conda environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" "$PYTHON_SPEC" -y

# Activate the conda environment
echo "Activating conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Setup environment
echo "Setting up environment..."
pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name="$ENV_NAME"

# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found, skipping package installation"
fi



echo "Conda environment '$ENV_NAME' created and activated."

# Update requirements.txt
echo "Updating requirements.txt..."
pip freeze > requirements.txt

echo "Process Complete. Press any key to exit..."
read -n 1 -s
exit 0