#!/usr/bin/env python3
"""
Test script to verify agent_mcts save/load functionality
"""
import sys
import os
from pathlib import Path

print("Testing agent_mcts save/load functionality...\n")

# Test 1: Import agent_mcts
print("1. Testing import...")
try:
    import agent_mcts
    print("   ✓ agent_mcts imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check for save method
print("\n2. Checking for save() method...")
if hasattr(agent_mcts, 'save'):
    print("   ✓ save() method found")
else:
    print("   ✗ save() method not found")
    sys.exit(1)

# Test 3: Check for load method
print("\n3. Checking for load() method...")
if hasattr(agent_mcts, 'load'):
    print("   ✓ load() method found")
else:
    print("   ✗ load() method not found")
    sys.exit(1)

# Test 4: Check other compatibility methods
print("\n4. Checking optional compatibility methods...")
methods = ['set_eval_mode', 'episode_start', 'end_episode', 'game_over_update']
for method in methods:
    if hasattr(agent_mcts, method):
        print(f"   ✓ {method}() found")
    else:
        print(f"   ✗ {method}() not found")

# Test 5: Test save functionality
print("\n5. Testing save() functionality...")
test_path = "test_checkpoint_temp.pt"
try:
    agent_mcts.save(test_path)
    if Path(test_path).exists():
        print(f"   ✓ Checkpoint saved to {test_path}")
        print(f"   ✓ File size: {Path(test_path).stat().st_size} bytes")
    else:
        print("   ✗ Checkpoint file not created")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Save failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test load functionality
print("\n6. Testing load() functionality...")
try:
    agent_mcts.load(test_path)
    print(f"   ✓ Checkpoint loaded from {test_path}")
except Exception as e:
    print(f"   ✗ Load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test set_eval_mode
print("\n7. Testing set_eval_mode()...")
try:
    agent_mcts.set_eval_mode(True)
    print("   ✓ set_eval_mode(True) succeeded")
    agent_mcts.set_eval_mode(False)
    print("   ✓ set_eval_mode(False) succeeded")
except Exception as e:
    print(f"   ✗ set_eval_mode failed: {e}")
    sys.exit(1)

# Test 8: Test episode hooks
print("\n8. Testing episode hooks...")
try:
    agent_mcts.episode_start()
    print("   ✓ episode_start() succeeded")
    import numpy as np
    dummy_board = np.zeros(29)
    agent_mcts.end_episode(1, dummy_board, 1)
    print("   ✓ end_episode() succeeded")
    agent_mcts.game_over_update(dummy_board, 1)
    print("   ✓ game_over_update() succeeded")
except Exception as e:
    print(f"   ✗ Episode hooks failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
print("\n9. Cleaning up test files...")
try:
    if Path(test_path).exists():
        os.remove(test_path)
        print(f"   ✓ Removed {test_path}")
except Exception as e:
    print(f"   ⚠ Cleanup warning: {e}")

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("="*50)
print("\nSummary:")
print("  - agent_mcts now has save() and load() methods")
print("  - Checkpoints can be saved during training")
print("  - All train.py compatibility hooks are present")
print("  - You can now safely stop and resume training")
