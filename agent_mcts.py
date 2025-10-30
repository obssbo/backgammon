"""
MCTS agent for Backgammon
"""
import numpy as np
import torch
import torch.nn as nn
import Backgammon
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


# -------- Training Infrastructure --------
class MCTSTrainer:
    """Wrapper for training MCTS agent with self-play"""
    def __init__(self, model, simulations=MCTS_SIMULATIONS, lr=1e-3):
        self.model = model
        self.agent = MCTSAgent(model, simulations)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.is_training = True
        self.memory = []  # Store (state, value_target) pairs
        self.batch_size = 64

    def collect_training_data(self, board, outcome):
        """Store position and outcome for training"""
        state = _encode_state(board)
        # outcome: +1 if we won, -1 if we lost
        self.memory.append((state, outcome))

    def train_step(self):
        """Train on collected experience"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample random batch
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        states = torch.tensor([s for s, _ in batch], dtype=torch.float32)
        values_target = torch.tensor([[v] for _, v in batch], dtype=torch.float32)

        # Forward pass
        self.model.train()
        _, values_pred = self.model(states)

        # Loss: only train value head (policy head gets supervision from MCTS visit counts)
        loss = torch.nn.functional.mse_loss(values_pred, values_target)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

        return loss.item()

    def clear_memory(self):
        """Clear replay buffer (e.g., after checkpoint)"""
        self.memory = []


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
_trainer = MCTSTrainer(_model)


def action(board_copy, dice, player, i=0, train=False, **_):
    # Flip to player +1 perspective
    board_pov = _flip_board(board_copy) if player == -1 else board_copy

    # Run MCTS
    move = _agent.search(board_pov, dice)

    # Flip back if needed
    return _flip_move(move) if player == -1 else move


def episode_start():
    """Called at start of each game"""
    pass


def end_episode(outcome, final_board, perspective):
    """Called at end of game to collect training data

    Args:
        outcome: +1 if won, -1 if lost (from this agent's perspective)
        final_board: final board state
        perspective: +1 or -1 (which player we are)
    """
    if _trainer.is_training:
        # Store the final position with outcome
        board_pov = _flip_board(final_board) if perspective == -1 else final_board
        _trainer.collect_training_data(board_pov, outcome)

        # Periodically train
        if len(_trainer.memory) >= _trainer.batch_size:
            for _ in range(4):  # 4 gradient steps
                _trainer.train_step()


def save(path: str):
    """Save model checkpoint"""
    torch.save({
        "model": _model.state_dict(),
        "optimizer": _trainer.optimizer.state_dict(),
    }, path)


def load(path: str):
    """Load model checkpoint"""
    state = torch.load(path, map_location="cpu")
    _model.load_state_dict(state["model"])
    if "optimizer" in state:
        _trainer.optimizer.load_state_dict(state["optimizer"])


def set_eval_mode(is_eval: bool):
    """Switch between training and evaluation mode"""
    _trainer.is_training = not is_eval
    if is_eval:
        _model.eval()
    else:
        _model.train()