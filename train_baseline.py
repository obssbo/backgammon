#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for BASELINE agent (agent_ac_adv.py - macro-action after-state)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

import backgammon
import pubeval_player as pubeval
import random_player as randomAgent
import flipped_agent as flipped_util
import agent_ac_adv as agent  # BASELINE: macro-action, after-state critic

CKPT_DIR = Path("checkpoints/baseline")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = Path("results/baseline_learning_curve.json")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _is_empty_move(move):
    if move is None: return True
    if isinstance(move, (list, tuple)): return len(move) == 0
    if isinstance(move, np.ndarray): return move.size == 0
    return False

def _apply_move_sequence(board, move_seq, player):
    mv = np.asarray(move_seq, dtype=np.int32)
    return backgammon.update_board(board, mv, player)

def play_one_game(agent1, agent2, training=False):
    board = backgammon.init_board()
    player = np.random.randint(2) * 2 - 1

    if hasattr(agent1, "episode_start"): agent1.episode_start()
    if hasattr(agent2, "episode_start"): agent2.episode_start()

    while not backgammon.game_over(board) and not backgammon.check_for_error(board):
        dice = backgammon.roll_dice()

        for r in range(1 + int(dice[0] == dice[1])):
            board_copy = board.copy()
            if player == 1:
                move = agent1.action(board_copy, dice, player, i=r, train=training)
            else:
                move = agent2.action(board_copy, dice, player, i=r, train=training)

            if _is_empty_move(move):
                continue

            board = _apply_move_sequence(board, move, player)

        player = -player

    winner = -player
    final_board = board

    if hasattr(agent1, "end_episode"):
        agent1.end_episode(+1 if winner == 1 else -1, final_board, perspective=+1)
    if hasattr(agent2, "end_episode"):
        agent2.end_episode(+1 if winner == -1 else -1, final_board, perspective=-1)

    return winner, final_board

def evaluate(agent_mod, evaluation_agent, n_eval):
    wins = 0
    for i in range(n_eval):
        if i % 2 == 0:
            w, _ = play_one_game(agent_mod, evaluation_agent, training=False)
        else:
            w, _ = play_one_game(evaluation_agent, agent_mod, training=False)
            w = -w
        wins += int(w == 1)
    winrate = round(wins / n_eval * 100.0, 3)
    return winrate

def train(n_games=200_000, n_epochs=5_000, n_eval=500):
    baseline = pubeval
    best_wr = -1.0
    learning_curve = []  # Store (game_num, win_rate) tuples

    print("=" * 60)
    print("TRAINING BASELINE AGENT (agent_ac_adv.py)")
    print("Macro-action, after-state critic")
    print("=" * 60)

    for g in range(1, n_games + 1):
        winner, final_board = play_one_game(agent, agent, training=True)

        if hasattr(agent, "game_over_update"):
            agent.game_over_update(final_board, int(winner == 1))
            flipped_final = flipped_util.flip_board(final_board)
            agent.game_over_update(flipped_final, int(winner == -1))

        if (g % n_epochs) == 0:
            if hasattr(agent, "set_eval_mode"): agent.set_eval_mode(True)

            wr = evaluate(agent, baseline, n_eval)
            learning_curve.append({"game": g, "win_rate": wr})

            print(f"[{g:7d} games] Win rate: {wr:6.2f}%")

            # Save checkpoint
            if hasattr(agent, "save"):
                epoch_ckpt = CKPT_DIR / f"epoch_{g}.pt"
                agent.save(str(epoch_ckpt))

                if wr > best_wr:
                    best_wr = wr
                    best_ckpt = CKPT_DIR / "best.pt"
                    agent.save(str(best_ckpt))
                    print(f"  → New best: {best_wr:.2f}%")

            # Save learning curve to JSON
            with open(LOG_FILE, 'w') as f:
                json.dump(learning_curve, f, indent=2)

            if hasattr(agent, "set_eval_mode"): agent.set_eval_mode(False)

    print("\n" + "=" * 60)
    print(f"BASELINE TRAINING COMPLETE")
    print(f"Final win rate: {learning_curve[-1]['win_rate']:.2f}%")
    print(f"Best win rate: {best_wr:.2f}%")
    print(f"Learning curve saved to: {LOG_FILE}")
    print("=" * 60)

    return learning_curve

if __name__ == "__main__":
    train(n_games=200_000, n_epochs=5_000, n_eval=500)
