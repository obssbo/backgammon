# Experimental Comparison: Baseline vs Micro-Action Agent

This guide explains how to run experiments comparing the baseline actor-critic agent (macro-action, after-state) with the new micro-action agent (state-value).

## Overview

- **Baseline Agent**: `agent_ac_adv.py` - Uses macro-actions (full dice roll) with after-state critic
- **Micro Agent**: `agent_ac_adv_micro.py` - Uses micro-actions (single-die moves) with state-value critic

## Directory Structure

```
backgammon/
├── train_baseline.py          # Train baseline agent
├── train_micro.py             # Train micro-action agent
├── plot_learning_curves.py    # Compare learning curves
├── qualitative_analysis.py    # Behavioral comparison
├── checkpoints/
│   ├── baseline/              # Baseline checkpoints
│   └── micro/                 # Micro agent checkpoints
└── results/
    ├── baseline_learning_curve.json
    ├── micro_learning_curve.json
    ├── learning_curves_comparison.png
    └── qualitative_analysis.json
```

## Step-by-Step Execution

### Option A: Parallel Training (Recommended - 24 hours)

If you have the computational resources, train both agents simultaneously:

```bash
# Terminal 1: Train baseline
python train_baseline.py

# Terminal 2: Train micro agent (run concurrently)
python train_micro.py
```

### Option B: Sequential Training (48 hours)

Train one after the other:

```bash
# First: Train baseline (24 hours)
python train_baseline.py

# Then: Train micro agent (24 hours)
python train_micro.py
```

### Training Parameters

Both scripts use the same parameters:
- **Total games**: 200,000
- **Evaluation interval**: Every 5,000 games
- **Evaluation games**: 500 games vs pubeval
- **Total evaluations**: 40 checkpoints

You can modify these in the scripts if needed.

---

## After Training: Generate Results

### 1. Plot Learning Curves

```bash
python plot_learning_curves.py
```

**Output**:
- `results/learning_curves_comparison.png` - Side-by-side learning curves
- Console summary with final/best win rates

**What to look for**:
- Which agent learns faster (steeper curve)?
- Which achieves higher final performance?
- Is one more stable (less variance)?

---

### 2. Qualitative Behavioral Analysis

```bash
python qualitative_analysis.py
```

**Output**:
- `results/qualitative_analysis.json` - Detailed statistics
- Console report with behavioral comparisons

**Metrics analyzed**:
1. **Game length** - Average moves per game
2. **Blot exposure** - Risk-taking behavior (exposed checkers)
3. **Bar escape** - How effectively they escape from bar
4. **Bearing off** - Endgame efficiency
5. **Move efficiency** - How often they use all dice

**What to look for**:
- Does micro-agent play more conservatively or aggressively?
- Are there differences in tactical vs strategic behavior?
- How do they handle key situations (bar, bearing off)?

---

## Monitoring Training Progress

During training, you'll see output like:

```
==============================================================
TRAINING BASELINE AGENT (agent_ac_adv.py)
Macro-action, after-state critic
==============================================================
[  5000 games] Win rate:  45.20%
[  10000 games] Win rate:  48.60%
  → New best: 48.60%
[  15000 games] Win rate:  51.40%
  → New best: 51.40%
...
```

**Checkpoints saved**:
- `checkpoints/{agent}/epoch_5000.pt`, `epoch_10000.pt`, etc.
- `checkpoints/{agent}/best.pt` (best performing checkpoint)

**Learning curves saved**:
- `results/{agent}_learning_curve.json` (updated every 5000 games)

You can monitor progress without waiting for full training:
```bash
# While training is running:
python plot_learning_curves.py  # Plot partial results
```

---

## Expected Results

### Learning Curves

**Hypothesis 1**: Micro-action agent might show:
- ✅ Faster initial learning (finer credit assignment)
- ✅ More stable convergence (updates every micro-step)
- ⚠️ Or slower learning (more complex action space)

**Hypothesis 2**: Baseline agent might:
- ✅ More sample-efficient (fewer decisions per turn)
- ✅ Better strategic coherence (plans full turns)

### Qualitative Differences

**Expected behavioral differences**:

1. **Bar Escape**
   - Micro-agent: More flexible, can adapt mid-turn
   - Baseline: Commits to full-turn strategy

2. **Bearing Off**
   - Micro-agent: Potentially more optimal die-by-die
   - Baseline: May be more efficient overall

3. **Blot Exposure**
   - Micro-agent: Risk tolerance may vary per micro-step
   - Baseline: More consistent risk profile

4. **Game Length**
   - If micro-agent is more conservative: Longer games
   - If micro-agent is more tactical: Could be shorter

---

## Troubleshooting

### Training crashes or errors

```bash
# Check agent implementations
python -c "import agent_ac_adv; print('Baseline OK')"
python -c "import agent_ac_adv_micro; print('Micro OK')"
```

### No learning progress (stuck at ~50%)

- This is expected early in training
- Should improve after 10,000-20,000 games
- If stuck after 50,000 games, check learning rates

### Out of memory

Reduce batch size or use CPU:
- Both agents use CPU by default
- If still issues, reduce `n_eval` from 500 to 100

---

## Report Template

After running experiments, report:

### 1. Learning Curves

```
Include: results/learning_curves_comparison.png

Discussion:
- Which agent converged faster?
- Final win rates: Baseline X%, Micro Y%
- Stability: Which had less variance?
```

### 2. Qualitative Comparison

```
Key observations from qualitative_analysis.py:

Game Length:
- Baseline: X moves/game
- Micro: Y moves/game
- Interpretation: ...

Blot Exposure:
- Baseline: X blots/move
- Micro: Y blots/move
- Interpretation: Micro is more [aggressive/conservative] because...

Bar Escape:
- Baseline: X% success rate
- Micro: Y% success rate
- Interpretation: ...

Bearing Off:
- Observed differences in endgame efficiency...

Strategic Differences:
- Micro-action optimizes at micro-step level, leading to...
- Baseline plans full turns, resulting in...
```

### 3. Example Game Analysis

Pick 2-3 interesting positions and show:
- Board state
- Dice roll
- Baseline choice
- Micro choice
- Explanation of why they differ

---

## Quick Reference

```bash
# Full workflow
python train_baseline.py          # 24 hrs
python train_micro.py             # 24 hrs (can run in parallel)
python plot_learning_curves.py    # Instant
python qualitative_analysis.py    # ~5 mins

# Check progress during training
ls -lh checkpoints/baseline/      # See saved checkpoints
ls -lh checkpoints/micro/
python plot_learning_curves.py    # Plot partial results
```

---

## Notes

- Both agents use the same hyperparameters (α=0.1, λ=0.7, γ=1.0)
- Evaluation always uses greedy policy (no exploration)
- Pubeval is a strong classical baseline (~intermediate human level)
- Random baseline would show ~90-95% win rate when converged
