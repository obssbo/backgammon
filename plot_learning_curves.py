#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot and compare learning curves for baseline vs micro-action agents.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_learning_curve(filepath):
    """Load learning curve from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    games = [d['game'] for d in data]
    win_rates = [d['win_rate'] for d in data]
    return games, win_rates

def plot_comparison():
    """Create comparison plot of learning curves."""

    baseline_file = Path("results/baseline_learning_curve.json")
    micro_file = Path("results/micro_learning_curve.json")

    # Check files exist
    if not baseline_file.exists():
        print(f"ERROR: {baseline_file} not found. Train baseline agent first.")
        return
    if not micro_file.exists():
        print(f"ERROR: {micro_file} not found. Train micro agent first.")
        return

    # Load data
    baseline_games, baseline_wr = load_learning_curve(baseline_file)
    micro_games, micro_wr = load_learning_curve(micro_file)

    # Create figure
    plt.figure(figsize=(12, 7))

    # Plot both curves
    plt.plot(baseline_games, baseline_wr,
             label='Baseline (Macro-action, After-state)',
             marker='o', linewidth=2, markersize=4, color='#1f77b4')
    plt.plot(micro_games, micro_wr,
             label='Micro-action (State-value)',
             marker='s', linewidth=2, markersize=4, color='#ff7f0e')

    # Add 50% reference line
    plt.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random play (50%)')

    # Formatting
    plt.xlabel('Training Games', fontsize=12)
    plt.ylabel('Win Rate vs Pubeval (%)', fontsize=12)
    plt.title('Learning Curves: Baseline vs Micro-Action Actor-Critic', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)

    # Set reasonable axis limits
    plt.xlim(0, max(max(baseline_games), max(micro_games)))
    plt.ylim(0, 100)

    # Add statistics text box
    baseline_final = baseline_wr[-1]
    micro_final = micro_wr[-1]
    baseline_best = max(baseline_wr)
    micro_best = max(micro_wr)

    stats_text = (
        f"Final Win Rates:\n"
        f"  Baseline: {baseline_final:.2f}%\n"
        f"  Micro:    {micro_final:.2f}%\n"
        f"\n"
        f"Best Win Rates:\n"
        f"  Baseline: {baseline_best:.2f}%\n"
        f"  Micro:    {micro_best:.2f}%"
    )

    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    output_file = Path("results/learning_curves_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {output_file}")

    # Display
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("LEARNING CURVE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nBaseline (Macro-action, After-state):")
    print(f"  Final win rate: {baseline_final:.2f}%")
    print(f"  Best win rate:  {baseline_best:.2f}%")
    print(f"  Total games:    {baseline_games[-1]:,}")

    print(f"\nMicro-action (State-value):")
    print(f"  Final win rate: {micro_final:.2f}%")
    print(f"  Best win rate:  {micro_best:.2f}%")
    print(f"  Total games:    {micro_games[-1]:,}")

    print(f"\nDifference (Micro - Baseline):")
    print(f"  Final: {micro_final - baseline_final:+.2f}%")
    print(f"  Best:  {micro_best - baseline_best:+.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    plot_comparison()
