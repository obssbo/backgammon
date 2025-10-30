"""
MCTS agent for Backgammon
"""
import numpy as np
import torch
import torch.nn as nn
import backgammon as Backgammon
import math
from collections import defaultdict

# -------- Constants --------
STATE_DIM = 29
HID = 256
MCTS_SIMULATIONS = 100  # Adjust based on time constraints
C_PUCT = 1.0  # Exploration constant


# -------- Neural Network --------
class PolicyValueNet(nn.Module):
    def __init__(self, in_dim=STATE_DIM, hid=HID):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hid, 1)  # Prior probability for this state
        self.value_head = nn.Linear(hid, 1)  # State value

    def forward(self, x):
        features = self.shared(x)
        policy = self.policy_head(features)
        value = torch.tanh(self.value_head(features))  # [-1, 1]
        return policy, value


# -------- MCTS Node --------
class MCTSNode:
    def __init__(self, state, parent=None, prior_p=1.0):
        self.state = state  # Board position
        self.parent = parent
        self.children = {}  # move -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_p = prior_p
        self.is_expanded = False

    def Q(self):
        """Average value"""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def U(self, parent_visits):
        """Exploration bonus (UCB)"""
        return C_PUCT * self.prior_p * math.sqrt(parent_visits) / (1 + self.visit_count)

    def best_child(self):
        """Select child with highest Q + U"""
        return max(self.children.items(),
                   key=lambda item: item[1].Q() + item[1].U(self.visit_count))

    def update(self, value):
        """Backpropagate value"""
        self.visit_count += 1
        self.total_value += value


# -------- Helper Functions --------
_FLIP_IDX = np.array([0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
                      12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 25, 28, 27], dtype=np.int32)


def _flip_board(b):
    return -b[_FLIP_IDX]


def _flip_move(m):
    if len(m) == 0:
        return m
    m = np.asarray(m, dtype=np.int32).copy()
    for r in range(m.shape[0]):
        m[r, 0] = _FLIP_IDX[m[r, 0]]
        m[r, 1] = _FLIP_IDX[m[r, 1]]
    return m


def _encode_state(board_flipped, moves_left=0):
    x = np.zeros(STATE_DIM, dtype=np.float32)
    x[:24] = board_flipped[1:25] * 0.2
    x[24] = board_flipped[25] * 0.2
    x[25] = board_flipped[26] * 0.2
    x[26] = board_flipped[27] / 15.0
    x[27] = board_flipped[28] / 15.0
    x[28] = float(moves_left)
    return x


def _board_to_key(board):
    """Convert board to hashable key for node lookup"""
    return tuple(board.tolist())


# -------- MCTS Class --------
class MCTSAgent:
    def __init__(self, model, simulations=MCTS_SIMULATIONS):
        self.model = model
        self.model.eval()
        self.simulations = simulations

    @torch.no_grad()
    def evaluate(self, board):
        """Get policy prior and value from network"""
        state = _encode_state(board)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        policy, value = self.model(state_t)
        return policy.item(), value.item()

    def expand(self, node, dice):
        """Expand node with all legal moves"""
        possible_moves, possible_boards = Backgammon.legal_moves(
            node.state, dice, player=1
        )

        if not possible_moves:
            node.is_expanded = True
            return

        # Get priors for each after-state
        for move, board in zip(possible_moves, possible_boards):
            prior, _ = self.evaluate(board)
            move_key = tuple(map(tuple, move)) if len(move) > 0 else ()
            node.children[move_key] = MCTSNode(board, parent=node, prior_p=abs(prior))

        node.is_expanded = True

    def simulate(self, root, dice):
        """Run one MCTS simulation"""
        node = root

        # 1. Selection: traverse tree using UCB
        while node.is_expanded and node.children:
            move_key, node = node.best_child()

        # 2. Expansion
        if not node.is_expanded:
            self.expand(node, dice)
            # If we just expanded, pick a child
            if node.children:
                move_key, node = node.best_child()

        # 3. Evaluation (using neural network instead of rollout)
        _, value = self.evaluate(node.state)

        # 4. Backpropagation
        while node is not None:
            node.update(value)
            value = -value  # Flip value for opponent
            node = node.parent

    def search(self, board, dice):
        """Run MCTS and return best move"""
        root = MCTSNode(board)

        # Run simulations
        for _ in range(self.simulations):
            self.simulate(root, dice)

        # Return most visited move (more robust than highest value)
        if not root.children:
            return []

        best_move_key = max(root.children.items(),
                            key=lambda item: item[1].visit_count)[0]

        # Convert back to array format
        if len(best_move_key) == 0:
            return []
        return np.array(best_move_key)


# -------- Main Action Function --------
_model = PolicyValueNet()
_model.eval()

# Load weights
try:
    state = torch.load("checkpoints/mcts_best.pt", map_location="cpu")
    _model.load_state_dict(state["model"])
except Exception:
    pass

_agent = MCTSAgent(_model, simulations=MCTS_SIMULATIONS)


# -------- Checkpoint Management --------
def save(path=None):
    """Save model checkpoint compatible with train.py"""
    from pathlib import Path
    if path is None:
        path = "checkpoints/mcts_best.pt"

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": _model.state_dict(),
        "simulations": MCTS_SIMULATIONS,
    }, save_path)
    print(f"[agent_mcts] Saved checkpoint to {save_path}")


def load(path=None, map_location=None):
    """Load model checkpoint"""
    from pathlib import Path
    if path is None:
        path = "checkpoints/mcts_best.pt"

    load_path = Path(path)
    if not load_path.exists():
        print(f"[agent_mcts] Checkpoint {load_path} not found, skipping load")
        return

    ml = map_location or "cpu"
    state = torch.load(load_path, map_location=ml)
    _model.load_state_dict(state["model"])
    print(f"[agent_mcts] Loaded checkpoint from {load_path}")
    set_eval_mode(True)


def set_eval_mode(is_eval):
    """Set model to eval or train mode"""
    if is_eval:
        _model.eval()
    else:
        _model.train()


# -------- Episode Hooks (for train.py compatibility) --------
def episode_start():
    """Called at the start of each episode (no-op for MCTS)"""
    pass


def end_episode(outcome, final_board, perspective):
    """Called at the end of each episode (no-op for MCTS)"""
    pass


def game_over_update(board, reward):
    """Legacy compatibility hook (no-op for MCTS)"""
    pass


# -------- Main Action Function --------
def action(board_copy, dice, player, i=0, **_):
    # Flip to player +1 perspective
    board_pov = _flip_board(board_copy) if player == -1 else board_copy

    # Run MCTS
    move = _agent.search(board_pov, dice)

    # Flip back if needed
    return _flip_move(move) if player == -1 else move