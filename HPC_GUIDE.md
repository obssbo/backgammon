# HPC Cluster Setup and Execution Guide

Complete guide for running backgammon agent training on the HPC cluster.

---

## Step 1: Initial Setup (One-time)

### 1.1 Transfer Code to Cluster

```bash
# From your local machine:
scp -r backgammon/ obs5@elja-irhpc.rhi.hi.is:~/

# Or use git:
ssh obs5@elja-irhpc.rhi.hi.is
cd ~
git clone <your-repo-url>
cd backgammon
```

### 1.2 Setup Python Environment

```bash
# On the cluster:
cd ~/backgammon

# Make setup script executable
chmod +x setup_hpc_env.sh

# Run setup (takes ~5 minutes)
./setup_hpc_env.sh
```

**This will:**
- Load Python 3.11.3 and matplotlib modules
- Create a virtual environment at `~/backgammon_env`
- Install PyTorch (CPU version)
- Install NumPy and other dependencies

**You only need to do this once!**

---

## Step 2: Check Cluster Configuration

### 2.1 Check Available Partitions

```bash
sinfo
```

Look for partition names (e.g., `compute`, `batch`, `long`). Update the `#SBATCH --partition=` line in the `.slurm` files if needed.

### 2.2 Check Resource Limits

```bash
# Check partition time limits
sinfo -o "%P %.10l"

# Check your account limits
sacctmgr show user $USER
```

If the time limit is less than 30 hours, you may need to:
- Request a longer partition: `#SBATCH --partition=long`
- Or reduce training games in the scripts

### 2.3 Verify Modules

```bash
module load Python/3.11.3
module load matplotlib/3.7.2-python-3.11.3
python --version  # Should show 3.11.3
```

---

## Step 3: Submit Training Jobs

### 3.1 Create Logs Directory

```bash
cd ~/backgammon
mkdir -p logs
```

### 3.2 Make Scripts Executable

```bash
chmod +x submit_baseline.slurm
chmod +x submit_micro.slurm
```

### 3.3 Submit Both Jobs (Parallel Training)

```bash
# Submit baseline agent
sbatch submit_baseline.slurm

# Submit micro agent
sbatch submit_micro.slurm
```

**Expected output:**
```
Submitted batch job 123456
Submitted batch job 123457
```

**Note the job IDs!** You'll need them to monitor progress.

---

## Step 4: Monitor Jobs

### 4.1 Check Job Status

```bash
# Check your running jobs
squeue -u $USER

# Detailed info for a specific job
scontrol show job <JOB_ID>
```

**Job states:**
- `PD` (Pending): Waiting for resources
- `R` (Running): Currently executing
- `CG` (Completing): Finishing up
- `CD` (Completed): Finished successfully
- `F` (Failed): Job failed

### 4.2 Monitor Training Progress

**While jobs are running:**

```bash
# Watch baseline progress
tail -f logs/baseline_<JOB_ID>.out

# Watch micro progress
tail -f logs/micro_<JOB_ID>.out

# Exit with Ctrl+C
```

You'll see output like:
```
==============================================================
TRAINING BASELINE AGENT (agent_ac_adv.py)
==============================================================
[  5000 games] Win rate:  45.20%
[ 10000 games] Win rate:  48.60%
  → New best: 48.60%
...
```

### 4.3 Check Learning Curves During Training

```bash
# Load interactive session (for plotting)
srun --pty --time=00:30:00 --mem=4G bash

# Load modules and activate environment
module load Python/3.11.3
module load matplotlib/3.7.2-python-3.11.3
source ~/backgammon_env/bin/activate

# Navigate and plot
cd ~/backgammon
python plot_learning_curves.py

# Exit interactive session
exit
```

### 4.4 Check Errors

```bash
# If job fails, check error log
tail -100 logs/baseline_<JOB_ID>.err
tail -100 logs/micro_<JOB_ID>.err
```

---

## Step 5: After Training Completes

### 5.1 Verify Training Completed

```bash
cd ~/backgammon

# Check that checkpoints were saved
ls -lh checkpoints/baseline/
ls -lh checkpoints/micro/

# Check that learning curves exist
ls -lh results/
```

**Expected files:**
```
checkpoints/baseline/best.pt
checkpoints/baseline/epoch_5000.pt
...
checkpoints/micro/best.pt
...
results/baseline_learning_curve.json
results/micro_learning_curve.json
```

### 5.2 Generate Comparison Plots

```bash
# Start interactive session
srun --pty --time=01:00:00 --mem=4G bash

# Load environment
module load Python/3.11.3
module load matplotlib/3.7.2-python-3.11.3
source ~/backgammon_env/bin/activate

cd ~/backgammon

# Generate learning curve comparison
python plot_learning_curves.py

# Run qualitative analysis (takes ~5 minutes)
python qualitative_analysis.py

# Exit
exit
```

### 5.3 Download Results to Local Machine

```bash
# From your local machine:
scp -r obs5@elja-irhpc.rhi.hi.is:~/backgammon/results ./
scp -r obs5@elja-irhpc.rhi.hi.is:~/backgammon/checkpoints ./
```

---

## Troubleshooting

### Job Won't Start (Stuck in PD)

**Possible causes:**
1. Cluster is busy → Wait
2. Requested too many resources → Reduce `--mem` or `--cpus-per-task`
3. Wrong partition → Check with `sinfo` and update scripts

```bash
# Check why job is pending
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.20R"
```

### Job Failed Immediately

```bash
# Check error log
cat logs/baseline_<JOB_ID>.err

# Common issues:
# - Module not found → Check module availability with `ml avail`
# - Virtual env missing → Re-run setup_hpc_env.sh
# - Permission denied → chmod +x on scripts
```

### Out of Memory

If job fails with OOM (Out of Memory):

1. Edit `.slurm` file: Increase `--mem=16G`
2. Or reduce training: Edit `train_*.py` and set `n_games=100_000`

### Out of Time

If job hits time limit before finishing:

**Option 1: Request more time**
```bash
# Edit .slurm file:
#SBATCH --time=48:00:00
```

**Option 2: Resume from checkpoint**

The training scripts already save checkpoints. To resume (you'd need to modify the training scripts to support this):
- Start from the last saved checkpoint
- Continue training for remaining games

### PyTorch Not Found

```bash
# Reinstall in environment
source ~/backgammon_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Resource Usage Estimates

Based on 200k games of training:

| Resource | Estimated Usage |
|----------|----------------|
| **Time** | ~24 hours per agent |
| **Memory** | ~4-6 GB |
| **CPUs** | 4 cores recommended |
| **Storage** | ~500 MB (checkpoints + results) |

---

## Useful SLURM Commands

```bash
# Submit job
sbatch submit_baseline.slurm

# Check queue
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Job history (after completion)
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# Check job efficiency
seff <JOB_ID>

# Interactive session (for testing)
srun --pty --time=01:00:00 --mem=4G bash
```

---

## Quick Reference: Full Workflow

```bash
# 1. Setup (one-time)
cd ~/backgammon
./setup_hpc_env.sh

# 2. Submit both training jobs
sbatch submit_baseline.slurm
sbatch submit_micro.slurm

# 3. Monitor
squeue -u $USER
tail -f logs/baseline_*.out

# 4. After training (interactive session)
srun --pty --time=01:00:00 --mem=4G bash
module load Python/3.11.3 matplotlib/3.7.2-python-3.11.3
source ~/backgammon_env/bin/activate
cd ~/backgammon
python plot_learning_curves.py
python qualitative_analysis.py
exit

# 5. Download results
# (from local machine)
scp -r obs5@elja-irhpc.rhi.hi.is:~/backgammon/results ./
```

---

## Advanced: Customizing Training

### Reduce Training Time (for testing)

Edit `train_baseline.py` and `train_micro.py`:

```python
# Change from:
train(n_games=200_000, n_epochs=5_000, n_eval=500)

# To (faster, less accurate):
train(n_games=50_000, n_epochs=5_000, n_eval=100)
```

### Change Evaluation Frequency

```python
# Evaluate more often (more data points in learning curve):
train(n_games=200_000, n_epochs=2_500, n_eval=500)

# Evaluate less often (faster training):
train(n_games=200_000, n_epochs=10_000, n_eval=500)
```

### Adjust Resources in SLURM Scripts

```bash
# Edit submit_*.slurm:
#SBATCH --time=48:00:00      # More time
#SBATCH --mem=16G            # More memory
#SBATCH --cpus-per-task=8    # More CPUs
```

---

## Getting Help

1. **Cluster-specific docs**: Check your cluster documentation for partition names and policies
2. **Module issues**: `module spider <name>` to search for modules
3. **SLURM help**: `man sbatch` or `sbatch --help`
4. **Python environment**: Make sure you activated the venv before running Python commands

---

## Contact

For cluster-specific issues, contact your HPC support team.
For code issues, check the main `EXPERIMENT_GUIDE.md`.
