#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative comparison of baseline vs micro-action agents.
Analyzes behavioral differences in key scenarios.
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict

import backgammon
import agent_ac_adv as baseline_agent
import agent_ac_adv_micro as micro_agent
import flipped_agent

def _is_empty_move(move):
    if move is None: return True
    if isinstance(move, (list, tuple)): return len(move) == 0
    if isinstance(move, np.ndarray): return move.size == 0
    return False

def _apply_move_sequence(board, move_seq, player):
    mv = np.asarray(move_seq, dtype=np.int32)
    return backgammon.update_board(board, mv, player)

def count_blots(board, player):
    """Count exposed checkers (blots) for a player."""
    blots = 0
    if player == 1:
        for pos in range(1, 25):
            if board[pos] == 1:  # Single checker
                blots += 1
    else:
        for pos in range(1, 25):
            if board[pos] == -1:  # Single checker
                blots += 1
    return blots

def is_on_bar(board, player):
    """Check if player has checkers on the bar."""
    if player == 1:
        return board[25] > 0
    else:
        return board[26] < 0

def is_bearing_off(board, player):
    """Check if player is in bearing-off phase."""
    if player == 1:
        # All checkers in home board (19-24) or already borne off
        for pos in range(1, 19):
            if board[pos] > 0:
                return False
        return True
    else:
        for pos in range(7, 25):
            if board[pos] < 0:
                return False
        return True

def analyze_game(agent, agent_name, n_games=100):
    """Analyze agent behavior over multiple games."""

    stats = {
        'total_games': n_games,
        'total_moves': 0,
        'empty_moves': 0,
        'bar_situations': 0,
        'bar_escapes': 0,
        'bearing_off_situations': 0,
        'blots_created': [],
        'game_lengths': [],
    }

    agent.set_eval_mode(True)

    for game_idx in range(n_games):
        board = backgammon.init_board()
        player = 1  # Agent always plays as +1
        move_count = 0

        if hasattr(agent, "episode_start"):
            agent.episode_start()

        while not backgammon.game_over(board) and not backgammon.check_for_error(board):
            dice = backgammon.roll_dice()

            for r in range(1 + int(dice[0] == dice[1])):
                board_copy = board.copy()

                # Track pre-move state
                was_on_bar = is_on_bar(board, player)
                was_bearing_off = is_bearing_off(board, player)
                blots_before = count_blots(board, player)

                # Get move
                move = agent.action(board_copy, dice, player, i=r, train=False)

                if _is_empty_move(move):
                    stats['empty_moves'] += 1
                    continue

                # Apply move
                board = _apply_move_sequence(board, move, player)
                move_count += 1
                stats['total_moves'] += 1

                # Track post-move state
                blots_after = count_blots(board, player)

                # Analyze behavior
                if was_on_bar:
                    stats['bar_situations'] += 1
                    if not is_on_bar(board, player):
                        stats['bar_escapes'] += 1

                if was_bearing_off:
                    stats['bearing_off_situations'] += 1

                # Track blot creation/elimination
                stats['blots_created'].append(blots_after - blots_before)

            player = -player

        stats['game_lengths'].append(move_count)

        if hasattr(agent, "end_episode"):
            winner = -player
            agent.end_episode(+1 if winner == 1 else -1, board, perspective=+1)

    # Compute summary statistics
    stats['avg_game_length'] = np.mean(stats['game_lengths'])
    stats['std_game_length'] = np.std(stats['game_lengths'])
    stats['avg_blots_change'] = np.mean(stats['blots_created'])
    stats['bar_escape_rate'] = (stats['bar_escapes'] / stats['bar_situations'] * 100
                                  if stats['bar_situations'] > 0 else 0)
    stats['empty_move_rate'] = (stats['empty_moves'] / stats['total_moves'] * 100
                                 if stats['total_moves'] > 0 else 0)

    return stats

def compare_on_position(board, dice, player):
    """Compare both agents' moves on a specific position."""

    baseline_agent.set_eval_mode(True)
    micro_agent.set_eval_mode(True)

    board_baseline = board.copy()
    board_micro = board.copy()

    baseline_move = baseline_agent.action(board_baseline, dice, player, i=0, train=False)
    micro_move = micro_agent.action(board_micro, dice, player, i=0, train=False)

    return {
        'baseline_move': baseline_move,
        'micro_move': micro_move,
        'same_move': np.array_equal(baseline_move, micro_move) if not (_is_empty_move(baseline_move) and _is_empty_move(micro_move)) else True
    }

def print_comparison_report(baseline_stats, micro_stats):
    """Print formatted comparison report."""

    print("\n" + "=" * 80)
    print("QUALITATIVE BEHAVIORAL COMPARISON")
    print("=" * 80)

    print(f"\nGames analyzed: {baseline_stats['total_games']}")

    print("\n" + "-" * 80)
    print("1. GAME LENGTH")
    print("-" * 80)
    print(f"{'Metric':<40} {'Baseline':<20} {'Micro':<20}")
    print(f"{'Average moves per game':<40} {baseline_stats['avg_game_length']:>18.2f} {micro_stats['avg_game_length']:>18.2f}")
    print(f"{'Std dev':<40} {baseline_stats['std_game_length']:>18.2f} {micro_stats['std_game_length']:>18.2f}")

    diff_length = micro_stats['avg_game_length'] - baseline_stats['avg_game_length']
    print(f"\n  → Micro agent games are {abs(diff_length):.2f} moves {'longer' if diff_length > 0 else 'shorter'} on average")

    print("\n" + "-" * 80)
    print("2. BLOT EXPOSURE (Risk-taking behavior)")
    print("-" * 80)
    print(f"{'Average blot change per move':<40} {baseline_stats['avg_blots_change']:>18.3f} {micro_stats['avg_blots_change']:>18.3f}")

    if baseline_stats['avg_blots_change'] > micro_stats['avg_blots_change']:
        print("  → Baseline agent is MORE aggressive (creates more blots)")
    elif baseline_stats['avg_blots_change'] < micro_stats['avg_blots_change']:
        print("  → Micro agent is MORE aggressive (creates more blots)")
    else:
        print("  → Similar risk profiles")

    print("\n" + "-" * 80)
    print("3. BAR ESCAPE BEHAVIOR")
    print("-" * 80)
    print(f"{'Bar situations encountered':<40} {baseline_stats['bar_situations']:>18d} {micro_stats['bar_situations']:>18d}")
    print(f"{'Successful bar escapes':<40} {baseline_stats['bar_escapes']:>18d} {micro_stats['bar_escapes']:>18d}")
    print(f"{'Bar escape rate (%)':<40} {baseline_stats['bar_escape_rate']:>18.2f} {micro_stats['bar_escape_rate']:>18.2f}")

    print("\n" + "-" * 80)
    print("4. BEARING OFF")
    print("-" * 80)
    print(f"{'Bearing-off moves':<40} {baseline_stats['bearing_off_situations']:>18d} {micro_stats['bearing_off_situations']:>18d}")

    print("\n" + "-" * 80)
    print("5. MOVE EFFICIENCY")
    print("-" * 80)
    print(f"{'Total moves made':<40} {baseline_stats['total_moves']:>18d} {micro_stats['total_moves']:>18d}")
    print(f"{'Empty/pass moves':<40} {baseline_stats['empty_moves']:>18d} {micro_stats['empty_moves']:>18d}")
    print(f"{'Empty move rate (%)':<40} {baseline_stats['empty_move_rate']:>18.2f} {micro_stats['empty_move_rate']:>18.2f}")

    print("\n" + "=" * 80)

def main():
    """Run qualitative analysis."""

    # Load best checkpoints
    baseline_ckpt = Path("checkpoints/baseline/best.pt")
    micro_ckpt = Path("checkpoints/micro/best.pt")

    if not baseline_ckpt.exists():
        print(f"ERROR: Baseline checkpoint not found at {baseline_ckpt}")
        print("Train baseline agent first with: python train_baseline.py")
        return

    if not micro_ckpt.exists():
        print(f"ERROR: Micro checkpoint not found at {micro_ckpt}")
        print("Train micro agent first with: python train_micro.py")
        return

    # Load checkpoints
    print("Loading trained agents...")
    baseline_agent.load(str(baseline_ckpt))
    micro_agent.load(str(micro_ckpt))
    print("✓ Agents loaded")

    # Run analysis
    print("\nAnalyzing baseline agent behavior...")
    baseline_stats = analyze_game(baseline_agent, "Baseline", n_games=100)

    print("Analyzing micro agent behavior...")
    micro_stats = analyze_game(micro_agent, "Micro", n_games=100)

    # Print comparison
    print_comparison_report(baseline_stats, micro_stats)

    # Save results
    results = {
        'baseline': baseline_stats,
        'micro': micro_stats
    }

    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    results = convert_types(results)

    output_file = Path("results/qualitative_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Full results saved to: {output_file}")

if __name__ == "__main__":
    main()
