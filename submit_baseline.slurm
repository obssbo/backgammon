#!/bin/bash
#SBATCH --job-name=baseline_ac
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --time=30:00:00              # 30 hours (24h training + buffer)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # 4 CPU cores
#SBATCH --mem=8G                     # 8GB memory
#SBATCH --partition=compute          # Adjust based on your cluster partitions

echo "=========================================="
echo "BASELINE AGENT TRAINING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module purge
module load Python/3.11.3
module load matplotlib/3.7.2-python-3.11.3

# Activate virtual environment
VENV_DIR="$HOME/backgammon_env"
source $VENV_DIR/bin/activate

# Navigate to working directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Print environment info
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# Run training
echo "Starting baseline agent training..."
python train_baseline.py

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
