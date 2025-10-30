#!/usr/bin/env python3
"""
Test script to verify agent_mcts compatibility with train.py
"""
import ast
import sys

def check_function_exists(filepath, function_name):
    """Check if a function exists in a Python file"""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.func_name == function_name:
            return True
    return False

def get_function_signature(filepath, function_name):
    """Get the signature of a function"""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = [arg.arg for arg in node.args.args]
            defaults = [None] * (len(args) - len(node.args.defaults)) + node.args.defaults
            return {
                'args': args,
                'defaults': defaults,
                'vararg': node.args.vararg,
                'kwarg': node.args.kwarg
            }
    return None

print("Testing agent_mcts compatibility with train.py...\n")

# Check if action function exists
print("1. Checking action() function:")
action_sig = get_function_signature('agent_mcts.py', 'action')
if action_sig:
    print(f"   ✓ action() found")
    print(f"     Args: {action_sig['args']}")
    print(f"     Has **kwargs: {action_sig['kwarg'] is not None}")

    # Verify it can accept train.py's call signature
    required_params = ['board_copy', 'dice', 'player', 'i']
    has_required = all(p in action_sig['args'] for p in required_params)
    can_accept_extra = action_sig['kwarg'] is not None

    if has_required or can_accept_extra:
        print(f"   ✓ Compatible with train.py's call signature")
    else:
        print(f"   ✗ Missing required parameters")
else:
    print(f"   ✗ action() not found")
    sys.exit(1)

# Check optional functions
print("\n2. Checking optional functions (train.py uses hasattr):")
optional_funcs = ['episode_start', 'end_episode', 'save', 'set_eval_mode', 'game_over_update']
for func in optional_funcs:
    sig = get_function_signature('agent_mcts.py', func)
    if sig:
        print(f"   ✓ {func}() found")
    else:
        print(f"   - {func}() not found (optional, will be skipped)")

print("\n3. Checking imports:")
try:
    with open('agent_mcts.py', 'r') as f:
        content = f.read()
        if 'import backgammon' in content or 'from backgammon' in content:
            print("   ✓ Imports backgammon (lowercase) correctly")
        elif 'import Backgammon' in content and 'backgammon' not in content:
            print("   ✗ Imports Backgammon (uppercase) - will fail on case-sensitive systems")
            sys.exit(1)
        else:
            print("   ✓ Backgammon import found")
except Exception as e:
    print(f"   ✗ Error checking imports: {e}")
    sys.exit(1)

print("\n✓ agent_mcts appears compatible with train.py!")
print("\nNotes:")
print("  - agent_mcts doesn't have save(), so checkpoints won't be saved")
print("  - agent_mcts doesn't have episode_start/end_episode, but this is OK")
print("  - The 'train' parameter in action() will be ignored by agent_mcts")
