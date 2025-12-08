#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy-only PyTorch Policy agent to estimate (after-state evaluation).

API:
  - action(board, dice, player, i, train=False, train_config=None)
  - game_over_update(board, reward)
  - set_eval_mode(is_eval)
  - save(path="checkpoints/best.pt")
  - load(path="checkpoints/best.pt")

Key points:
- Policy network (_pnet) estimates Q(after-state) from +1 POV.
- Computing the TD(lambda) eligibility traces manually.
- During training, opponent's turn is simulated to get next after-state.
"""

from collections import deque
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backgammon  # engine

# ---------------- Config ----------------
class Config:
    state_dim = 24 + 4 + 1      # 24 points + (bar_self, bar_opp, off_self, off_opp) + moves_left
    gamma = 0.99
    lr = 1e-3
    epsilon = 0.0
    lam = 0.9                     # TD(lambda) trace decay
    batch_size = 256
    buffer_size = 100_000
    start_learning_after = 5_00
    target_update_every = 2_000
    train_every = 1
    hidden1 = 256                # <- same width as your agent.py
    hidden2 = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

# ------------- Flip helpers (29-length boards) -------------
_FLIP_IDX = np.array(
    [0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
     12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 25, 28, 27],
    dtype=np.int32
)

def _flip_board(board):
    out = np.empty(29, dtype=board.dtype)
    out[:] = -board[_FLIP_IDX]
    return out

def _flip_move(move):
    if len(move) == 0:
        return move
    mv = np.asarray(move, dtype=np.int32).copy()
    for r in range(mv.shape[0]):
        mv[r, 0] = _FLIP_IDX[mv[r, 0]]
        mv[r, 1] = _FLIP_IDX[mv[r, 1]]
    return mv

class PolicyNet(nn.Module):
    def __init__(self, in_dim=CFG.state_dim, hid1=CFG.hidden1, hid2=CFG.hidden2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid1), nn.ReLU(),
            nn.Linear(hid1, hid2), nn.ReLU(),
            nn.Linear(hid2, 1),
        )
    def forward(self, x):
        scores = self.net(x).squeeze(-1)
        return scores

_pnet = PolicyNet().to(CFG.device)
_tpnet = PolicyNet().to(CFG.device)
_tpnet.load_state_dict(_pnet.state_dict())
_opt = torch.optim.Adam(_pnet.parameters(), lr=CFG.lr)

_traces = {name: torch.zeros_like(param) for name, param in _pnet.named_parameters()}
_episode_trajectory = []  # flat list

_steps = 0
_eval_mode = False

# ------------- Save / Load -------------
CHECKPOINT_PATH = Path("checkpoints/best.pt")
_loaded_from_disk = False

def save(path: str = str(CHECKPOINT_PATH)):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pnet": _pnet.state_dict()}, p)

def load(path: str = str(CHECKPOINT_PATH), map_location: str | torch.device = "cpu"):
    global _loaded_from_disk
    state = torch.load(path, map_location=map_location)
    _pnet.load_state_dict(state["pnet"])
    _tpnet.load_state_dict(_pnet.state_dict())
    set_eval_mode(True)
    _loaded_from_disk = True

def _lazy_load_if_available():
    global _loaded_from_disk
    if _loaded_from_disk:
        return
    if CHECKPOINT_PATH.exists():
        try:
            load(str(CHECKPOINT_PATH), map_location="cpu")
        except Exception:
            pass
        _loaded_from_disk = True
        

# ------------- Features -------------
def _moves_left(dice, i):
    # doubles -> two applications; otherwise one
    return 1 + int(dice[0] == dice[1]) - i

def _encode_state(board_plus_one, moves_left):
    """
    +1 POV features: 24 points (scaled), bar_self, bar_opp, off_self, off_opp, moves_left
    """
    x = np.zeros(CFG.state_dim, dtype=np.float32)
    x[:24] = board_plus_one[1:25] * 0.2
    x[24]  = board_plus_one[25] * 0.2
    x[25]  = board_plus_one[26] * 0.2
    x[26]  = board_plus_one[27] / 15.0
    x[27]  = board_plus_one[28] / 15.0
    x[28]  = float(moves_left)
    return x

def _is_terminal_plus_one(board_plus_one):
    return board_plus_one[27] == 15

# ------------- Hooks -------------
def set_eval_mode(is_eval=False):
    global _eval_mode
    _eval_mode = bool(is_eval)
    if _eval_mode: _pnet.eval()
    else:          _pnet.train()

def episode_start():
    global _episode_trajectory, _traces
    _episode_trajectory = []
    for name, param in _pnet.named_parameters():
        _traces[name].zero_()

def end_episode(outcome, final_board, perspective):
    if _eval_mode or len(_episode_trajectory) == 0:
        return

    final_reward = 1.0 if outcome == perspective else 0.0

    # Set the final reward for the last after-state
    last_state, last_value, _, _ = _episode_trajectory[-1]
    _episode_trajectory[-1] = (last_state, last_value, final_reward, 0.0)

    # Forward TD(lambda) update
    for t, (state_features, _, reward, next_v) in enumerate(_episode_trajectory):
        state_t = torch.tensor(state_features, dtype=torch.float32, device=CFG.device).unsqueeze(0)
        _opt.zero_grad()
        v = _pnet(state_t).squeeze(0)

        # TD error
        td_error = reward + CFG.gamma * next_v - v

        #td_error = torch.clamp(td_error, -1.0,1.0)

        # Policy gradient-like update
        v.backward()  # compute grad of predicted value w.r.t params

        with torch.no_grad():
            for name, param in _pnet.named_parameters():
                if param.grad is not None:
                    grad = param.grad * td_error.item()
                    _traces[name] = CFG.gamma * CFG.lam * _traces[name] + grad
                    param.add_(_traces[name], alpha=CFG.lr)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(_pnet.parameters(), max_norm=0.5)


def game_over_update(board, reward):
    pass

# ------------- Opponent turn → next after-state features -------------
def _s_next_after_opponent(chosen_board_plus_one: np.ndarray) -> np.ndarray:
    """
    From our chosen after-state (+1 POV), roll opponent dice, enumerate their
    legal after-states (player=-1), flip to +1 POV, encode with moves_left=1
    (our next first move), pick best-for-+1 using the TARGET net.
    """
    opp_dice = backgammon.roll_dice()
    opp_moves, opp_boards = backgammon.legal_moves(chosen_board_plus_one, opp_dice, player=-1)

    if len(opp_boards) == 0:
        return _encode_state(chosen_board_plus_one, moves_left=1)

    feats = np.stack([_encode_state(_flip_board(b), moves_left=1) for b in opp_boards], axis=0)
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=CFG.device)
    with torch.no_grad():
        vals = _tpnet(feats_t)
        idx = int(torch.argmin(vals).item())
    return feats[idx]

# ------------- Policy -------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    """
    Returns [] if no legal moves, else an array of shape (k,2) of [start,end] moves.
    Selection is ALWAYS greedy (no epsilon). If train=True, learn with after-state tuples.
    """
    global _steps
    if not train:
        _lazy_load_if_available()

    # Work in +1 POV
    board_pov = _flip_board(board_copy) if player == -1 else board_copy

    # Enumerate legal after-states from current dice
    possible_moves, possible_boards = backgammon.legal_moves(board_pov, dice, player=1)
    nA = len(possible_moves)
    if nA == 0:
        return []

    moves_left_now = _moves_left(dice, i)
    moves_left_after = max(0, moves_left_now - 1)
    Sp = np.stack([_encode_state(b, moves_left_after) for b in possible_boards], axis=0)
    Sp_t = torch.as_tensor(Sp, dtype=torch.float32, device=CFG.device)

    with torch.no_grad():
        policy_scores = _pnet(Sp_t)
    # Epsilon-greedy action selection
    epsilon = CFG.epsilon
    if train:
        probs = torch.softmax(policy_scores, dim=0)
        a_idx = int(torch.multinomial(probs, 1).item())
    else:
        a_idx = int(torch.argmax(policy_scores).item())

    chosen_move = possible_moves[a_idx]
    chosen_board_plus_one = possible_boards[a_idx]
    s_after = Sp[a_idx]

    # Terminal & reward from +1 POV
    done = _is_terminal_plus_one(chosen_board_plus_one)
    r = 1.0 if done else 0.0

    if train and (not _eval_mode):
    
        _steps += 1
        if _steps % CFG.target_update_every == 0:
            _tpnet.load_state_dict(_pnet.state_dict())
        
        # Boostrap next after-state value via opponent simulation
        if not done:
            next_s_after = _s_next_after_opponent(chosen_board_plus_one)
            with torch.no_grad():
              next_v = float(_tpnet(
                  torch.tensor(next_s_after, dtype=torch.float32, device=CFG.device).unsqueeze(0)
              ))
        else:
            next_s_after = s_after
            next_v = 0.0
        #if train and _steps % 100 == 0:
        #    print(f"[DEBUG] Step {_steps}: value range [{policy_scores.min().item():.3f}, {policy_scores.max().item():.3f}], std={policy_scores.std().item():.3f}")  
        #if _steps % 100 == 0:
        #    print(r)
        # Append to flat trajectory
        _episode_trajectory.append((s_after, float(policy_scores[a_idx]), r, next_v))


    # Return move in ORIGINAL POV
    if player == -1:
        chosen_move = _flip_move(chosen_move)
    return chosen_move
