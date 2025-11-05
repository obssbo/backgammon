# Quick Start Guide for HPC Cluster

**Goal**: Train both agents in parallel on the HPC cluster (~24 hours)

---

## Prerequisites

You should be logged into the HPC cluster:
```bash
ssh obs5@elja-irhpc.rhi.hi.is
```

---

## Step 1: One-Time Setup (5 minutes)

```bash
# Navigate to your backgammon directory
cd ~/backgammon

# Make setup script executable
chmod +x setup_hpc_env.sh

# Run setup - this installs PyTorch and dependencies
./setup_hpc_env.sh
```

**Wait for it to finish.** You'll see "✓ Setup complete!" when done.

---

## Step 2: Verify Setup (1 minute)

```bash
# Load modules
module load Python/3.11.3
module load matplotlib/3.7.2-python-3.11.3

# Activate environment
source ~/backgammon_env/bin/activate

# Run test
python test_hpc_setup.py
```

**Expected output:** "✓ All tests passed!"

If tests fail, check `HPC_GUIDE.md` troubleshooting section.

---

## Step 3: Check Cluster Partitions

```bash
# See available partitions
sinfo
```

Look for partition names. If you see something other than `compute`, edit the SLURM scripts:

```bash
# Edit both files
nano submit_baseline.slurm
nano submit_micro.slurm

# Change this line to match your partition:
#SBATCH --partition=<your_partition_name>
```

Common partition names: `batch`, `compute`, `long`, `normal`

---

## Step 4: Submit Training Jobs

```bash
# Make sure you're in the backgammon directory
cd ~/backgammon

# Create logs directory
mkdir -p logs

# Make scripts executable
chmod +x submit_baseline.slurm
chmod +x submit_micro.slurm

# Submit both jobs
sbatch submit_baseline.slurm
sbatch submit_micro.slurm
```

**You'll see:**
```
Submitted batch job 123456
Submitted batch job 123457
```

**Write down these job IDs!**

---

## Step 5: Monitor Progress

### Check if jobs are running:

```bash
squeue -u $USER
```

**Status codes:**
- `R` = Running (good!)
- `PD` = Pending (waiting for resources - this is normal)
- `CG` = Completing
- `CD` = Completed

### Watch training progress:

```bash
# Replace <JOB_ID> with your actual job ID from Step 4
tail -f logs/baseline_<JOB_ID>.out

# Press Ctrl+C to exit
```

You should see:
```
[  5000 games] Win rate:  45.20%
[ 10000 games] Win rate:  48.60%
  → New best: 48.60%
```

### Check for errors:

```bash
tail -f logs/baseline_<JOB_ID>.err
```

If there's nothing in the error file, that's good!

---

## Step 6: Wait for Completion (~24 hours)

The jobs will run for approximately 24 hours. You can:

1. **Check periodically:**
   ```bash
   squeue -u $USER
   ```

2. **Check progress:**
   ```bash
   tail -30 logs/baseline_<JOB_ID>.out
   tail -30 logs/micro_<JOB_ID>.out
   ```

3. **Log out and come back later** - Jobs will keep running!

---

## Step 7: After Training Completes

### Verify completion:

```bash
cd ~/backgammon

# Check that both jobs finished
sacct -u $USER --format=JobID,JobName,State,Elapsed | grep -E "baseline|micro"

# Should show "COMPLETED" status

# Check that checkpoints exist
ls -lh checkpoints/baseline/best.pt
ls -lh checkpoints/micro/best.pt

# Check results
ls -lh results/
```

---

## Step 8: Generate Comparison Plots

```bash
# Start interactive session (needed for plotting)
srun --pty --time=01:00:00 --mem=4G bash

# Load environment
module load Python/3.11.3
module load matplotlib/3.7.2-python-3.11.3
source ~/backgammon_env/bin/activate

cd ~/backgammon

# Generate learning curve comparison
python plot_learning_curves.py

# Run qualitative analysis (~5 minutes)
python qualitative_analysis.py

# Exit interactive session
exit
```

---

## Step 9: Download Results to Your Computer

**From your local machine** (not the cluster):

```bash
# Download results
scp -r obs5@elja-irhpc.rhi.hi.is:~/backgammon/results ~/Desktop/

# Download plot
scp obs5@elja-irhpc.rhi.hi.is:~/backgammon/results/learning_curves_comparison.png ~/Desktop/
```

Now you have:
- `learning_curves_comparison.png` - Learning curve plot for your report
- `qualitative_analysis.json` - Behavioral statistics
- `baseline_learning_curve.json` - Raw data
- `micro_learning_curve.json` - Raw data

---

## Troubleshooting

### "Job won't start (stuck in PD)"
- **Cause**: Cluster is busy or you requested too many resources
- **Solution**: Wait, or reduce resources in `.slurm` files

### "Module not found"
- **Cause**: Module name different on your cluster
- **Solution**: Run `ml avail` and find the correct Python/matplotlib module names

### "PyTorch not installed"
- **Solution**: Re-run `./setup_hpc_env.sh`

### "Out of memory" or "Out of time"
- **Solution**: Edit `.slurm` files:
  ```bash
  #SBATCH --time=48:00:00  # Increase time
  #SBATCH --mem=16G        # Increase memory
  ```

### "Permission denied"
- **Solution**: `chmod +x` on all scripts

---

## Need More Help?

- **Cluster-specific**: Check your institution's HPC documentation
- **Detailed guide**: See `HPC_GUIDE.md`
- **Experiment details**: See `EXPERIMENT_GUIDE.md`

---

## Quick Reference Card

```bash
# Setup (once)
./setup_hpc_env.sh

# Submit jobs
sbatch submit_baseline.slurm
sbatch submit_micro.slurm

# Monitor
squeue -u $USER
tail -f logs/baseline_*.out

# After completion
srun --pty --time=01:00:00 bash
module load Python/3.11.3 matplotlib/3.7.2-python-3.11.3
source ~/backgammon_env/bin/activate
python plot_learning_curves.py
python qualitative_analysis.py
exit
```

---

**Expected Total Time:**
- Setup: 5 minutes
- Training: 24 hours (both agents in parallel)
- Analysis: 10 minutes
- **Total: ~24 hours**

Good luck! 🚀
