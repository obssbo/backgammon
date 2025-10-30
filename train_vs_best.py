#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train MCTS agent against the best checkpoint
This prevents regression from circular self-play
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

import backgammon
import pubeval_player as pubeval
import flipped_agent as flipped_util

from pathlib import Path
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def plot_perf(perf, title="Training progress (win-rate vs baseline)"):
    if not perf:
        return
    xs = np.arange(len(perf))
    plt.plot(xs, perf)
    plt.xlabel("Evaluation checkpoints")
    plt.ylabel("Win rate (%)")
    plt.title(title)
    plt.grid(True)
    plt.savefig("training_progress.png")
    print("[Plot] Saved training_progress.png")

def _is_empty_move(move):
    if move is None: return True
    if isinstance(move, (list, tuple)): return len(move) == 0
    if isinstance(move, np.ndarray): return move.size == 0
    return False

def _apply_move_sequence(board, move_seq, player):
    mv = np.asarray(move_seq, dtype=np.int32)
    return backgammon.update_board(board, mv, player)

def play_one_game(agent1, agent2, training=False, commentary=False):
    board = backgammon.init_board()
    player = np.random.randint(2) * 2 - 1  # +1 or -1

    if hasattr(agent1, "episode_start"): agent1.episode_start()
    if hasattr(agent2, "episode_start"): agent2.episode_start()

    while not backgammon.game_over(board) and not backgammon.check_for_error(board):
        dice = backgammon.roll_dice()
        if commentary:
            print(f"player {player}, dice {dice}")

        for r in range(1 + int(dice[0] == dice[1])):  # doubles -> two applications
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

    if hasattr(agent1, "end_episode"): agent1.end_episode(+1 if winner == 1 else -1, final_board, perspective=+1)
    if hasattr(agent2, "end_episode"): agent2.end_episode(+1 if winner == -1 else -1, final_board, perspective=-1)

    return winner, final_board

def evaluate(agent_mod, evaluation_agent, n_eval, label=""):
    wins = 0
    # alternate who starts to reduce bias
    for i in range(n_eval):
        if i % 2 == 0:
            w, _ = play_one_game(agent_mod, evaluation_agent, training=False, commentary=False)
        else:
            w, _ = play_one_game(evaluation_agent, agent_mod, training=False, commentary=False)
            w = -w
        wins += int(w == 1)
    winrate = round(wins / n_eval * 100.0, 3)
    print(f"[Eval] {label or 'checkpoint'}: win-rate = {winrate}% over {n_eval} games")
    return winrate

class BestOpponent:
    """Wrapper to load and use the best checkpoint as opponent"""
    def __init__(self, checkpoint_path):
        from agent_mcts import PolicyValueNet, MCTSAgent, MCTS_SIMULATIONS

        self.model = PolicyValueNet()
        self.model.eval()

        # Load best checkpoint
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state["model"])
            print(f"[BestOpponent] Loaded from {checkpoint_path}")
        except Exception as e:
            print(f"[BestOpponent] Warning: Could not load {checkpoint_path}: {e}")

        # Import helper functions
        import agent_mcts as mcts_module
        self._flip_board = mcts_module._flip_board
        self._flip_move = mcts_module._flip_move
        self.agent = MCTSAgent(self.model, simulations=MCTS_SIMULATIONS)

    def action(self, board_copy, dice, player, i=0, train=False, **_):
        board_pov = self._flip_board(board_copy) if player == -1 else board_copy
        move = self.agent.search(board_pov, dice)
        return self._flip_move(move) if player == -1 else move

    def episode_start(self):
        pass

    def end_episode(self, outcome, final_board, perspective):
        pass  # Frozen opponent doesn't learn

def train(n_games=200_000, n_epochs=5_000, n_eval=500, eval_vs="pubeval",
          opponent_type="best", best_checkpoint="checkpoints/best.pt"):
    """
    Train MCTS agent

    Args:
        n_games: Total number of games to play
        n_epochs: Evaluate every n_epochs games
        n_eval: Number of evaluation games
        eval_vs: "pubeval" or "random"
        opponent_type: "best" (frozen best checkpoint) or "self" (traditional self-play)
        best_checkpoint: Path to best checkpoint
    """
    import agent_mcts as agent

    baseline = pubeval if eval_vs == "pubeval" else None

    # Setup opponent
    if opponent_type == "best":
        if not Path(best_checkpoint).exists():
            print(f"[Warning] Best checkpoint not found at {best_checkpoint}")
            print(f"[Warning] Falling back to self-play")
            opponent = agent
        else:
            opponent = BestOpponent(best_checkpoint)
            print(f"[Training] Playing against frozen best checkpoint: {best_checkpoint}")
    else:
        opponent = agent
        print(f"[Training] Traditional self-play (agent vs itself)")

    best_wr = -1.0
    winrates = []

    print("Training agent with self-play...")
    print(f"Baseline for eval: {baseline.__name__ if hasattr(baseline, '__name__') else baseline}")

    for g in range(1, n_games + 1):
        # Play training game
        # Alternate who plays as agent1 vs agent2 to balance learning
        if g % 2 == 0:
            winner, final_board = play_one_game(agent, opponent, training=True, commentary=False)
        else:
            winner, final_board = play_one_game(opponent, agent, training=True, commentary=False)

        if (g % n_epochs) == 0:
            if hasattr(agent, "set_eval_mode"): agent.set_eval_mode(True)
            wr = evaluate(agent, baseline, n_eval, label=f"after {g} games")
            winrates.append(wr)

            # Always save this eval checkpoint
            if hasattr(agent, "save"):
                epoch_ckpt = CKPT_DIR / f"epoch_{g}.pt"
                agent.save(str(epoch_ckpt))
                print(f"[Checkpoint] Saved {epoch_ckpt}")

                # Update best.pt if improved
                if wr > best_wr:
                    best_wr = wr
                    best_ckpt = CKPT_DIR / "best.pt"
                    agent.save(str(best_ckpt))
                    print(f"[Best] New best: {best_wr:.3f}% — saved {best_ckpt}")

                    # If using best opponent, reload it with new best
                    if opponent_type == "best" and isinstance(opponent, BestOpponent):
                        print(f"[BestOpponent] Reloading opponent from new best checkpoint")
                        opponent = BestOpponent(str(best_ckpt))

            if hasattr(agent, "set_eval_mode"): agent.set_eval_mode(False)

    plot_perf(winrates)

if __name__ == "__main__":
    # Resume from best checkpoint (15k games, 50% win rate)
    # and train against it
    train(
        n_games=100_000,      # Additional 100k games
        n_epochs=5_000,       # Evaluate every 5k games
        n_eval=100,           # 100 evaluation games
        eval_vs="pubeval",
        opponent_type="best", # Train against best checkpoint!
        best_checkpoint="checkpoints/best.pt"  # Your 50% win-rate checkpoint
    )
