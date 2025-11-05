#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to verify HPC setup is working correctly.
Run this before submitting long training jobs.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False

    return True

def test_agent_imports():
    """Test that agent modules can be imported."""
    print("\nTesting agent imports...")

    try:
        import backgammon
        print("✓ backgammon module")
    except ImportError as e:
        print(f"✗ backgammon import failed: {e}")
        return False

    try:
        import agent_ac_adv
        print("✓ agent_ac_adv (baseline)")
    except ImportError as e:
        print(f"✗ agent_ac_adv import failed: {e}")
        return False

    try:
        import agent_ac_adv_micro
        print("✓ agent_ac_adv_micro")
    except ImportError as e:
        print(f"✗ agent_ac_adv_micro import failed: {e}")
        return False

    try:
        import pubeval_player
        print("✓ pubeval_player")
    except ImportError as e:
        print(f"✗ pubeval_player import failed: {e}")
        return False

    return True

def test_directories():
    """Test that required directories exist or can be created."""
    print("\nTesting directory setup...")

    dirs = ['checkpoints', 'checkpoints/baseline', 'checkpoints/micro',
            'results', 'logs']

    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            print(f"✓ {d}/")
        except Exception as e:
            print(f"✗ Failed to create {d}/: {e}")
            return False

    return True

def test_training_scripts():
    """Test that training scripts exist and are readable."""
    print("\nTesting training scripts...")

    scripts = ['train_baseline.py', 'train_micro.py',
               'plot_learning_curves.py', 'qualitative_analysis.py']

    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {script}")
        else:
            print(f"✗ {script} not found")
            return False

    return True

def quick_training_test():
    """Run a quick training test (1 game) to verify everything works."""
    print("\nRunning quick training test (1 game)...")

    try:
        import numpy as np
        import backgammon
        import agent_ac_adv_micro as agent

        # Initialize agent
        agent.episode_start()

        # Play one game
        board = backgammon.init_board()
        player = 1
        moves = 0

        while not backgammon.game_over(board) and moves < 100:
            dice = backgammon.roll_dice()
            board_copy = board.copy()

            move = agent.action(board_copy, dice, player, i=0, train=True)

            if move is not None and len(move) > 0:
                mv = np.asarray(move, dtype=np.int32)
                board = backgammon.update_board(board, mv, player)

            player = -player
            moves += 1

        winner = -player
        agent.end_episode(+1 if winner == 1 else -1, board, perspective=+1)

        print(f"✓ Successfully played {moves} moves")
        return True

    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("HPC Setup Verification Test")
    print("=" * 60)
    print()

    results = []

    results.append(("Package imports", test_imports()))
    results.append(("Agent imports", test_agent_imports()))
    results.append(("Directory setup", test_directories()))
    results.append(("Training scripts", test_training_scripts()))
    results.append(("Quick training", quick_training_test()))

    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! You're ready to submit training jobs.")
        print("\nNext steps:")
        print("  sbatch submit_baseline.slurm")
        print("  sbatch submit_micro.slurm")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before submitting jobs.")
        print("\nTroubleshooting:")
        print("  - Make sure you ran: ./setup_hpc_env.sh")
        print("  - Activate environment: source ~/backgammon_env/bin/activate")
        print("  - Load modules: module load Python/3.11.3 matplotlib/3.7.2-python-3.11.3")
        return 1

if __name__ == "__main__":
    sys.exit(main())
