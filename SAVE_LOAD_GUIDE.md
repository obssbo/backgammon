# agent_mcts Save/Load Guide

## Overview

agent_mcts now includes full checkpoint save/load functionality, allowing you to:
- ✅ Save training progress at regular intervals
- ✅ Resume training if interrupted
- ✅ Load best-performing models for evaluation

## Added Methods

### 1. `save(path=None)`
Saves the current model state to a checkpoint file.

```python
import agent_mcts

# Save to default location
agent_mcts.save()  # Saves to "checkpoints/mcts_best.pt"

# Save to specific path
agent_mcts.save("checkpoints/epoch_5000.pt")
```

**What's saved:**
- Model state dict (neural network weights)
- MCTS simulation count

### 2. `load(path=None, map_location=None)`
Loads a saved checkpoint.

```python
import agent_mcts

# Load from default location
agent_mcts.load()  # Loads from "checkpoints/mcts_best.pt"

# Load from specific path
agent_mcts.load("checkpoints/epoch_5000.pt")

# Load with specific device mapping
agent_mcts.load("checkpoints/best.pt", map_location="cpu")
```

### 3. `set_eval_mode(is_eval)`
Controls whether the model is in evaluation or training mode.

```python
# Set to evaluation mode (for testing/playing)
agent_mcts.set_eval_mode(True)

# Set to training mode
agent_mcts.set_eval_mode(False)
```

### 4. Episode Hooks (Compatibility)
These are implemented as no-op stubs for train.py compatibility:

```python
agent_mcts.episode_start()
agent_mcts.end_episode(outcome, final_board, perspective)
agent_mcts.game_over_update(board, reward)
```

## Integration with train.py

When you use `train.py` or `train_mcts.py`, checkpoints are automatically saved:

```python
# train.py automatically calls agent.save() when:
# 1. Each evaluation checkpoint (e.g., every 5000 games)
# 2. When a new best win-rate is achieved

# Example output:
# [agent_mcts] Saved checkpoint to checkpoints/epoch_5000.pt
# [Checkpoint] Saved checkpoints/epoch_5000.pt
# [agent_mcts] Saved checkpoint to checkpoints/best.pt
# [Best] New best: 65.000% — saved checkpoints/best.pt
```

## Checkpoint Format

Checkpoints are saved as PyTorch `.pt` files with the following structure:

```python
{
    "model": state_dict,      # Neural network weights
    "simulations": int,       # MCTS simulation count
}
```

## Example: Resume Training After Interruption

```python
import agent_mcts

# Load the last saved checkpoint
agent_mcts.load("checkpoints/epoch_10000.pt")

# Continue training from where you left off
# (run train.py or train_mcts.py)
```

## Example: Evaluate Best Model

```python
import agent_mcts
import backgammon

# Load the best checkpoint
agent_mcts.load("checkpoints/best.pt")

# Model is automatically set to eval mode after loading
# Now you can use it for games
board = backgammon.init_board()
dice = backgammon.roll_dice()
move = agent_mcts.action(board, dice, player=1, i=0)
```

## Checkpoint Management Tips

1. **Disk Space**: Each checkpoint is relatively small (a few MB), but they add up over long training runs.

2. **Naming Convention**:
   - `epoch_N.pt`: Checkpoints at regular intervals
   - `best.pt`: Best performing model so far

3. **Backup Important Checkpoints**: Copy `best.pt` to a safe location periodically.

4. **Clean Old Checkpoints**: Remove intermediate checkpoints you don't need to save disk space.

## Troubleshooting

### "Checkpoint not found" message
```python
[agent_mcts] Checkpoint checkpoints/mcts_best.pt not found, skipping load
```
This is normal if you haven't saved any checkpoints yet. The agent will start with randomly initialized weights.

### Loading checkpoint from different device
If you saved on GPU but want to load on CPU:
```python
agent_mcts.load("checkpoints/best.pt", map_location="cpu")
```

### Verifying checkpoint was saved
```bash
ls -lh checkpoints/
# Should show .pt files with sizes
```

## Summary

With these new save/load features, agent_mcts is now fully compatible with train.py's checkpoint system. You can safely interrupt and resume training without losing progress!
