#!/bin/bash
# Setup Python environment for backgammon training on HPC cluster

echo "=========================================="
echo "Setting up Backgammon Training Environment"
echo "=========================================="

# Load required modules
module purge
module load Python/3.11.3
module load matplotlib/3.7.2-python-3.11.3

echo "✓ Modules loaded"

# Create virtual environment
VENV_DIR="$HOME/backgammon_env"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, run: rm -rf $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR ..."
    python -m venv $VENV_DIR
    echo "✓ Virtual environment created"
fi

# Activate environment
source $VENV_DIR/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version - no GPU needed for this)
echo "Installing PyTorch (CPU version)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other required packages
echo "Installing other dependencies..."
pip install numpy matplotlib

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate this environment in the future:"
echo "  module load Python/3.11.3"
echo "  module load matplotlib/3.7.2-python-3.11.3"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Environment location: $VENV_DIR"
