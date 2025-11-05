#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faithful Actor–Critic TD(λ) for Backgammon (aligned to original script),
adapted to train.py (online updates inside action()).

- Shared trunk (w1,b1) for both actor and critic.
- Critic head (w2,b2) for V(after-state) via sigmoid.
- Actor head is a single row vector theta (1 x nh) over shared h_tanh.
- Two sets of eligibility traces: normal POV (+1) and flipped (-1).
- Same one-hot features (nx = 11*24 + 4 + 1) and update logic/ordering.

API: episode_start(), action(...), set_eval_mode(), save(), load(), end_episode()
"""

from pathlib import Path
import numpy as np
import torch
from torch.autograd import Variable

# ---- engine imports (lower-case backgammon preferred) ----
import backgammon as Backgammon
import flipped_agent as flipped_agent  # your existing flip helpers

# -------------------- Device --------------------
# Original used CPU by default for stability/faithfulness
device = torch.device("cpu")

# -------------------- Hyperparameters (faithful) --------------------
alpha  = 0.1    # actor step size (theta)
alpha1 = 0.001  # critic layer-1 step size (w1,b1)
alpha2 = 0.001  # critic layer-2 step size (w2,b2)
lam    = 0.7    # TD(λ)
gamma  = 1.0    # episodic undiscounted

# -------------------- Features --------------------
dice_dim = 6
nx = 11 * 24 + 4 + 1 + dice_dim  # matches your working script
nh = int(nx / 2)

def one_hot_encoding(board, nSecondRoll):
    oneHot = np.zeros(nx, dtype=np.float32)
    # mark where zeros are
    zero_idx = np.where(board[1:25] == 0)[0]
    if zero_idx.size > 0:
        oneHot[zero_idx] = 1.0
    # +1 piles
    for i in range(0, 4):
        idx = np.where(board[1:25] == (i + 1))[0]
        if idx.size > 0:
            oneHot[24 + i * 24 + idx] = 1.0
    # +1 5+ piles (store count-4)
    idx = np.where(board[1:25] >= 5)[0]
    if idx.size > 0:
        oneHot[24 + 4 * 24 + idx] = board[idx + 1] - 4
    # -1 piles
    for i in range(0, 4):
        idx = np.where(board[1:25] == -(i + 1))[0]
        if idx.size > 0:
            oneHot[6 * 24 + i * 24 + idx] = 1.0
    # -1 5+ piles (store count-4)
    idx = np.where(board[1:25] <= -5)[0]
    if idx.size > 0:
        oneHot[6 * 24 + 4 * 24 + idx] = -board[idx + 1] - 4
    # jail/home + second-roll
    oneHot[11 * 24 + 0] = board[25]
    oneHot[11 * 24 + 1] = board[26]
    oneHot[11 * 24 + 2] = board[27]
    oneHot[11 * 24 + 3] = board[28]
    oneHot[11 * 24 + 4] = float(nSecondRoll)
    return oneHot

# -------------------- Parameters (faithful init) --------------------
w1 = Variable(0.1 * torch.randn(nh, nx, device=device, dtype=torch.float), requires_grad=True)
b1 = Variable(torch.zeros((nh, 1), device=device, dtype=torch.float), requires_grad=True)
w2 = Variable(0.1 * torch.randn(1, nh, device=device, dtype=torch.float), requires_grad=True)
b2 = Variable(torch.zeros((1, 1), device=device, dtype=torch.float), requires_grad=True)
theta = Variable(0.1 * torch.randn(156, nh, device=device, dtype=torch.float), requires_grad=True) # -> 156 states fixed

# -------------------- Per-episode state --------------------
# critic traces
Z_w1 = Z_b1 = Z_w2 = Z_b2 = None
Zf_w1 = Zf_b1 = Zf_w2 = Zf_b2 = None
# actor traces (theta)
Z_theta = Zf_theta = None

# caches for previous step
xold = xold_flipped = None
gradlnpi = gradlnpi_flipped = None
advantage = 0.0
advantage_flipped = 0.0
I = 1.0
If = 1.0
moveNumber = 0
dice_dim = 6

_eval_mode = False
_loaded_once = False

# -------------------- Save / Load --------------------
CKPT_DEFAULT = Path("checkpoints/td_lambda_ac.pt")

def save(path: str | None = None):
    p = Path(path) if path else CKPT_DEFAULT
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"w1": w1, "w2": w2, "b1": b1, "b2": b2, "theta": theta}, p)

def load(path: str | None = None, map_location: str | torch.device = "cpu"):
    p = Path(path) if path else CKPT_DEFAULT
    state = torch.load(p, map_location=map_location)
    w1.data.copy_(state["w1"].data);  w2.data.copy_(state["w2"].data)
    b1.data.copy_(state["b1"].data);  b2.data.copy_(state["b2"].data)
    theta.data.copy_(state["theta"].data)
    set_eval_mode(True)
    
def _lazy_load():
    global _loaded_once
    if _loaded_once:
        return
    if CKPT_DEFAULT.exists():
        try:
            load(str(CKPT_DEFAULT), map_location=device)
        except Exception as e:
            print(f"[agent-submit] Warning: failed to load '{CKPT_DEFAULT}': {e}")
    else:
        print(f"[agent-submit] Warning: checkpoint not found: {CKPT_DEFAULT}")
    _loaded_once = True

def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)
    
def _expand_dice_micro(dice: np.darray) -> list[int]:
    d1, d2 = int(dice[0]), int(dice[1])
    return [d1, d1] if d1 == d2 else [d1, d2]

def encoding_remaining_dice(remaining_dice):
    counts = np.zeros(dice_dim, dtype=np.float32)
    for d in remaining_dice:
        counts[d-1] += 1.0
    return counts

def build_mask(board, remaining_dice, oplayer):
    mask = np.zeros(156, dtype=np.float32)
    
    for die in remaining_dice:
        legal_moves = Backgammon.legal_move(board, die, oplayer)
        for move in legal_moves:
            src = int(move[0])
            idx = 6 * src + die
            mask[idx] = 1.0
    
    if mask.sum() == 0:
        for die in set(remaining_dice):
            src = 0
            mask[6 * src + die] = 1.0
    
    return mask

def encode_state_micro(board, remaining_dice, nSecondRoll):
    # s_t = (b_t, D_t)
    board_features = one_hot_encoding(board, nSecondRoll)[:269]
    dice_features = encoding_remaining_dice(remaining_dice)
    return np.concatenate([board_features,dice_features])

def action_idx_2_src_die(idx):
    die = ((idx - 1) % 6) + 1
    src = (idx - die) // 6
    return src,die

@torch.no_grad()
def greedy_action_micro(board, dice, oplayer):
    
    is_first_step = True
    flipped_flag = (oplayer == -1)
    if flipped_flag:
        board_eff = flipped_agent.flip_board(np.copy(board))
        player_eff = +1
    else:
        board_eff = board
        player_eff = oplayer
    
    dice_steps = _expand_dice_micro(dice)
    remaining_dice = dice_steps.copy()
    chosen_moves_eff = []
    
    while remaining_dice:
        # Enumerate single-die legal moves
        all_legals = []
        for die in remaining_dice:
            legals = Backgammon.legal_move(board_eff, die, player_eff)  # list of [start,end]
            for mv in legals:
                all_legals.append((mv,die))
        
        
        # nSecondRoll flag: True only on first micro-step of a double
        nSecondRoll = bool((dice_steps[0] == dice_steps[-1]) and len(remaining_dice) == len(dice_steps) and is_first_step)
                
        if len(all_legals) == 0:
            # cannot move on this die ⇒ pass this micro-step
            break
        # Encode state
        values = []
        for (mv, die) in all_legals:
            bb = Backgammon.update_board(board_eff,np.asarray(mv, dtype=np.int32),player_eff)
            state_enc = encode_state_micro(bb, [d for d in remaining_dice if d != die], nSecondRoll)
            
            #Evaluate
            x = torch.from_numpy(state_enc).to(device).view(-1,1)
            h = torch.tanh(torch.mm(w1, x) + b1)
            v = torch.sigmoid(torch.mm(w2, h) + b2)
            values.append(v.item())
            
        best_idx = int(np.argmax(values))
        best_move, best_die = all_legals[best_idx]
        
        chosen_moves_eff.append(np.array(best_move, dtype=np.int32))
        board_eff = Backgammon.update_board(board_eff, best_move, player_eff)
        remaining_dice.remove(best_die)
        is_first_step = False
        
    if len(chosen_moves_eff) == 0:
        return []
    
    moves_arr = np.stack(chosen_moves_eff, axis=0).astype(np.int32)
    if flipped_flag:
        moves_arr = flipped_agent.flip_move(moves_arr)
    return moves_arr


def softmax_policy_micro(board, remaining_dice, oplayer, nSecondRoll):
    """
    Sample ONE micro-action from 156-action masked policy.
    Called ONCE per micro-step by action().
    
    Returns ONE action, does NOT execute it!
    """
    flippedplayer = -1
    flipped_flag = (flippedplayer == oplayer)
    
    if flipped_flag:
        board_eff = flipped_agent.flip_board(np.copy(board))
        player_eff = -oplayer
    else:
        board_eff = board
        player_eff = oplayer
    
    # 1. Encode CURRENT state
    state_features = encode_state_micro(board_eff, remaining_dice, nSecondRoll)
    x = torch.from_numpy(state_features).to(device, dtype=torch.float).view(-1, 1)
    
    # 2. Forward pass
    h = torch.mm(w1, x) + b1
    h_tanh = h.tanh()
    
    # 3. Actor: 156 logits
    logits = torch.mm(theta, h_tanh).squeeze(1)  # (156,)
    
    # 4. Build mask
    mask = build_mask(board_eff, remaining_dice, player_eff)
    mask_tensor = torch.from_numpy(mask).to(device, dtype=torch.float)
    
    # Check no legal moves
    if mask.sum() == 0:
        return None, None, None, None, None, flipped_flag
    
    # 5. Masked softmax
    logits_masked = torch.where(mask_tensor > 0, logits, torch.tensor(-1e9, device=device))
    pi = torch.softmax(logits_masked, dim=0)
    
    # 6. Sample ONE action
    action_idx = int(torch.multinomial(pi, 1).item())
    src, die = action_idx_2_src_die(action_idx)
    
    # 7. Critic: state value V(s_t)
    V_t = torch.sigmoid(torch.mm(w2, h_tanh) + b2)
    value_scalar = V_t.data[0, 0]
    
    # 8. Policy gradient: ∇ log π(a_t | s_t)
    pi_expanded = pi.view(-1, 1)  # (156, 1)
    expected_feature = torch.sum(pi_expanded * h_tanh, dim=0, keepdim=True).T  # (nh, 1)
    grad_ln_pi = (h_tanh - expected_feature).detach()  # (nh, 1)
    
    # Return ONE action (not executed!)
    return src, die, x, value_scalar, grad_ln_pi, flipped_flag


    
def softmax_policy(board, dice, oplayer, nRoll):
    """
    Returns:
      action (list of moves),
      x_selected (nx,1) tensor for chosen after-state (not strictly required for update),
      target (scalar tensor value prediction of chosen after-state),
      advantage (float),
      chosen_after_eff (numpy 1D of after-state in +1 POV),
      flipped_flag (bool)
    """
    flippedplayer = -1
    nSecondRoll = bool((dice[0] == dice[1]) and (nRoll == 0))
    flipped_flag = (flippedplayer == oplayer)

    if flipped_flag:
        board_eff = flipped_agent.flip_board(np.copy(board))
        player_eff = -oplayer
    else:
        board_eff = board
        player_eff = oplayer

    possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
    na = len(possible_moves)
    if na == 0:
        return [], None, None, None, 0.0, None, flipped_flag

    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
    x = Variable(torch.tensor(xa.T, dtype=torch.float, device=device))

    # shared trunk
    h = torch.mm(w1, x) + b1
    h_tanh = h.tanh()

    # actor logits via single-row theta (1 x nh) -> (1 x na)
    logits = torch.mm(theta, h_tanh)
    pi = logits.softmax(dim=1)

    # sample action
    m = torch.multinomial(pi, 1)              # shape (1,1)
    m_idx = int(m.item())
    action = possible_moves[m_idx]
    if flipped_flag:
        action = flipped_agent.flip_move(action)

    # critic on after-states
    y = torch.mm(w2, h_tanh) + b2
    va = y.sigmoid()
    target = va.data[0, m_idx]                # scalar tensor

    # advantage as in original
    advantage = (target - torch.sum(pi * va)).item()

    # grad ln pi(a) wrt theta (1 x nh), faithful construction:
    # grad = h_tanh[:,a]^T - sum_j pi_j * h_tanh[:,j]^T
    xtheta_mean = torch.sum(h_tanh * pi, dim=1)                  # (nh,)
    h_a = h_tanh[:, m_idx]                                       # (nh,)
    grad_ln_pi = (h_a - xtheta_mean).view(1, -1).detach()        # (1 x nh)

    x_selected = Variable(torch.tensor(xa[m_idx, :], dtype=torch.float, device=device)).view(nx, 1)
    chosen_after_eff = possible_boards[m_idx].reshape(-1)

    return action, x_selected, target, grad_ln_pi, advantage, chosen_after_eff, flipped_flag
# -------------------- Episode hooks --------------------
def episode_start():
    global Z_w1, Z_b1, Z_w2, Z_b2, Zf_w1, Zf_b1, Zf_w2, Zf_b2
    global Z_theta, Zf_theta
    global xold, xold_flipped, gradlnpi, gradlnpi_flipped
    global advantage, advantage_flipped, I, If, moveNumber

    Z_w1 = torch.zeros_like(w1.data)
    Z_b1 = torch.zeros_like(b1.data)
    Z_w2 = torch.zeros_like(w2.data)
    Z_b2 = torch.zeros_like(b2.data)

    Zf_w1 = torch.zeros_like(w1.data)
    Zf_b1 = torch.zeros_like(b1.data)
    Zf_w2 = torch.zeros_like(w2.data)
    Zf_b2 = torch.zeros_like(b2.data)

    Z_theta = torch.zeros_like(theta.data)
    Zf_theta = torch.zeros_like(theta.data)

    xold = None
    xold_flipped = None
    gradlnpi = None
    gradlnpi_flipped = None

    advantage = 0.0
    advantage_flipped = 0.0

    I = 1.0
    If = 1.0
    moveNumber = 0

def end_episode(outcome, final_board, perspective):
    # original resets per episode; no extra terminal bookkeeping needed here
    pass

def game_over_update(board, reward):
    # compatibility hook (not used in this faithful port)
    pass


def action_micro(board_copy, dice, player, i, train=False, train_config=None):
    global Z_w1, Z_b1, Z_w2, Z_b2, Zf_w1, Zf_b1, Zf_w2, Zf_b2
    global Z_theta, Zf_theta
    global xold, xold_flipped, gradlnpi, gradlnpi_flipped
    global moveNumber
    
    nSecondRoll_flag = bool((dice[0] == dice[1]) and (i == 0))
    flippedplayer = -1
    
    # Greedy during eval
    if (not train) or _eval_mode:
        return greedy_action_micro(board_copy, dice, player)
    
    # Training mode: micro-rollout with policy sampling
    dice_steps = _expand_dice_micro(dice)
    remaining_dice = dice_steps.copy()
    current_board = board_copy.copy()
    moves_taken = []
    
    # Flip board if needed (do once at start)
    flipped_flag = (flippedplayer == player)
    if flipped_flag:
        current_board = flipped_agent.flip_board(current_board)
        player_eff = -player
    else:
        player_eff = player
    
    # Track previous micro-step for TD updates
    V_old = None
    x_old = None
    grad_old = None
    
    # MICRO-ROLLOUT LOOP
    while remaining_dice:
        # Call policy to sample ONE micro-action
        src, die, x, V_t, grad_ln_pi, _ = softmax_policy_micro(
            current_board, 
            remaining_dice, 
            player_eff,
            nSecondRoll_flag
        )
        
        if src is None:  # No legal moves
            break
        
        # Find actual legal move
        legal_moves = Backgammon.legal_move(current_board, die, player_eff)
        
        move = None
        for mv in legal_moves:
            if int(mv[0]) == src:
                move = np.array(mv, dtype=np.int32)
                break
        
        if move is None:
            print(f"Policy selected src={src}, die={die} but no matching legal move!")
            break
        
        # Execute micro-action
        current_board = Backgammon.update_board(current_board, move, player_eff)
        remaining_dice.remove(die)
        moves_taken.append(move)
        
        # Check if game ended
        is_terminal = (current_board[27] == 15) if player_eff == 1 else (current_board[28] == -15)
        
        # ========== TD UPDATE ==========
        if V_old is not None:  # Have previous step to update from
            # Compute TD error: δ = r + γ * V_t - V_old
            if is_terminal:
                # Terminal reward (from agent's perspective)
                reward = 1.0
                delta = reward + 0 - V_old  # V_terminal = 0
            else:
                # Non-terminal: r = 0 for micro-steps within turn
                delta = gamma * V_t - V_old
            
            # Convert to tensor
            delta_tensor = torch.tensor(delta, device=device, dtype=torch.float)
            
            # Compute gradient of V_old for critic traces
            # (Re-run forward pass on x_old with requires_grad=True)
            x_old_var = Variable(x_old, requires_grad=True)
            h_old = torch.mm(w1, x_old_var) + b1
            h_old_tanh = h_old.tanh()
            V_old_recompute = torch.sigmoid(torch.mm(w2, h_old_tanh) + b2)
            V_old_recompute.backward()
            
            # Update critic eligibility traces
            if flipped_flag:
                # Update flipped traces
                Zf_w1 = gamma * lam * Zf_w1 + w1.grad.data
                Zf_b1 = gamma * lam * Zf_b1 + b1.grad.data
                Zf_w2 = gamma * lam * Zf_w2 + w2.grad.data
                Zf_b2 = gamma * lam * Zf_b2 + b2.grad.data
                
                # Update actor traces
                Zf_theta = gamma * lam * Zf_theta + grad_old
                
                # Apply updates
                w1.data = w1.data + alpha1 * delta_tensor * Zf_w1
                b1.data = b1.data + alpha1 * delta_tensor * Zf_b1
                w2.data = w2.data + alpha2 * delta_tensor * Zf_w2
                b2.data = b2.data + alpha2 * delta_tensor * Zf_b2
                theta.data = theta.data + alpha * delta_tensor * Zf_theta
            else:
                # Update normal traces
                Z_w1 = gamma * lam * Z_w1 + w1.grad.data
                Z_b1 = gamma * lam * Z_b1 + b1.grad.data
                Z_w2 = gamma * lam * Z_w2 + w2.grad.data
                Z_b2 = gamma * lam * Z_b2 + b2.grad.data
                
                # Update actor traces
                Z_theta = gamma * lam * Z_theta + grad_old
                
                # Apply updates
                w1.data = w1.data + alpha1 * delta_tensor * Z_w1
                b1.data = b1.data + alpha1 * delta_tensor * Z_b1
                w2.data = w2.data + alpha2 * delta_tensor * Z_w2
                b2.data = b2.data + alpha2 * delta_tensor * Z_b2
                theta.data = theta.data + alpha * delta_tensor * Z_theta
            
            # Zero gradients
            w1.grad.data.zero_()
            b1.grad.data.zero_()
            w2.grad.data.zero_()
            b2.grad.data.zero_()
        
        # Cache current values for next micro-step
        V_old = V_t
        x_old = x
        grad_old = grad_ln_pi
        
        if is_terminal:
            break
    
    # Update move counter
    if not nSecondRoll_flag:
        moveNumber += 1
    
    # Convert back if flipped
    if flipped_flag and len(moves_taken) > 0:
        moves_arr = np.stack(moves_taken, axis=0)
        moves_arr = flipped_agent.flip_move(moves_arr)
        return moves_arr
    
    return np.stack(moves_taken, axis=0) if moves_taken else []

# -------------------- Policy helpers (faithful) --------------------
def greedy_action(board, dice, oplayer, nSecondRoll):
    flippedplayer = -1
    if flippedplayer == oplayer:
        board = flipped_agent.flip_board(np.copy(board))
        player = -oplayer
    else:
        player = oplayer

    possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
    na = len(possible_boards)
    if na == 0:
        return []

    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
    x = Variable(torch.tensor(xa.T, dtype=torch.float, device=device))

    h = torch.mm(w1, x) + b1
    h_tanh = h.tanh()
    y = torch.mm(w2, h_tanh) + b2
    va = y.sigmoid().detach().cpu().numpy()
    action = possible_moves[int(np.argmax(va))]
    if flippedplayer == oplayer:
        action = flipped_agent.flip_move(action)
    return action

# -------------------- Main action (called by train.py) --------------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    return action_micro(board_copy,dice,player,i)
#    global Z_w1, Z_b1, Z_w2, Z_b2, Zf_w1, Zf_b1, Zf_w2, Zf_b2
#    global Z_theta, Zf_theta
#    global xold, xold_flipped, gradlnpi, gradlnpi_flipped
#    global advantage, advantage_flipped, I, If, moveNumber

#    nSecondRoll_flag = bool((dice[0] == dice[1]) and (i == 0))
#    flippedplayer = -1

    # Greedy during eval
#    if (not train) or _eval_mode:
#        return greedy_action(np.copy(board_copy), dice, player, nSecondRoll_flag)

#    # Sample action + get targets/grad
#    out = softmax_policy(np.copy(board_copy), dice, player, nRoll=i)
#    act, x, target_val, grad_ln_pi, A, chosen_after_eff, flipped_flag = out
#    if isinstance(act, list) and len(act) == 0:
#        # no legal moves
#        if not nSecondRoll_flag:
#            moveNumber += 1
#        return []

    # Terminal check using chosen after-state in +1 POV
#    is_terminal = (chosen_after_eff[27] == 15)

    # Rewards exactly as in original
#    if is_terminal:
#        reward  = 1.0 if (player != flippedplayer) else 0.0
#        rewardf = 1.0 - reward
#        tgt = torch.tensor(0.0, device=device, dtype=torch.float)
#    else:
#        reward  = 0.0
#        rewardf = 0.0
#        tgt = target_val  # scalar tensor

#    # Start updates after at least one full turn and a move happened
#    if (moveNumber > 1) and (len(act) > 0):
#        # ----- flipped branch OR terminal -----
#        if (flippedplayer == player) or is_terminal:
#            if xold_flipped is not None:
#                # critic forward/backward on previous flipped after-state
#                h = torch.mm(w1, xold_flipped) + b1
#                h_tanh = h.tanh()
#                y = torch.mm(w2, h_tanh) + b2
#                y_sigmoid = y.sigmoid()
#                y_sigmoid.backward()

                # update critic traces
#                Zf_w1 = gamma * lam * Zf_w1 + w1.grad.data
#                Zf_b1 = gamma * lam * Zf_b1 + b1.grad.data
#                Zf_w2 = gamma * lam * Zf_w2 + w2.grad.data
#                Zf_b2 = gamma * lam * Zf_b2 + b2.grad.data
                # zero grads
#                w1.grad.data.zero_(); b1.grad.data.zero_()
#                w2.grad.data.zero_(); b2.grad.data.zero_()

                # actor traces with stored grad ln pi from previous flipped step
#                if gradlnpi_flipped is not None:
#                    Zf_theta = gamma * lam * Zf_theta + If * gradlnpi_flipped

                # TD error (scalar tensor broadcast over params)
#                delta = torch.tensor(rewardf, device=device) + gamma * tgt - y_sigmoid.detach()
                # critic updates
#                w1.data = w1.data + alpha1 * delta * Zf_w1
#                b1.data = b1.data + alpha1 * delta * Zf_b1
#                w2.data = w2.data + alpha2 * delta * Zf_w2
#                b2.data = b2.data + alpha2 * delta * Zf_b2
                # actor update (faithful: uses advantage from previous flipped step)
#                theta.data = theta.data + alpha * advantage_flipped * Zf_theta

#                If = If * gamma

        # ----- non-flipped branch OR terminal -----
#        if (flippedplayer != player) or is_terminal:
#            if xold is not None:
#                h = torch.mm(w1, xold) + b1
#                h_tanh = h.tanh()
#                y = torch.mm(w2, h_tanh) + b2
#                y_sigmoid = y.sigmoid()
#                y_sigmoid.backward()

#                Z_w1 = gamma * lam * Z_w1 + w1.grad.data
#                Z_b1 = gamma * lam * Z_b1 + b1.grad.data
#                Z_w2 = gamma * lam * Z_w2 + w2.grad.data
#                Z_b2 = gamma * lam * Z_b2 + b2.grad.data
#                w1.grad.data.zero_(); b1.grad.data.zero_()
#                w2.grad.data.zero_(); b2.grad.data.zero_()

#                if gradlnpi is not None:
#                    Z_theta = gamma * lam * Z_theta + I * gradlnpi

#                delta = torch.tensor(reward, device=device) + gamma * tgt - y_sigmoid.detach()
#                w1.data = w1.data + alpha1 * delta * Z_w1
#                b1.data = b1.data + alpha1 * delta * Z_b1
#                w2.data = w2.data + alpha2 * delta * Z_w2
#                b2.data = b2.data + alpha2 * delta * Z_b2

#                theta.data = theta.data + alpha * advantage * Z_theta

#                I = gamma * I

    # cache current side’s features & actor grad ln pi
#    if x is not None and len(act) > 0:
#        if player == -1:
#            xold_flipped = x
#            gradlnpi_flipped = grad_ln_pi
#            advantage_flipped = A
#        else:
#            xold = x
#            gradlnpi = grad_ln_pi
#            advantage = A

#    if not nSecondRoll_flag:
#        moveNumber += 1

#    return act
