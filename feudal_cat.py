import torch
import torch.nn as nn
import torch.nn.functional as F

from dilated_lstm import DilatedLSTM
from preprocess import Preprocessor
from storage import Storage


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor along feature dimension to unit norm (L2)."""
    norm = tensor.norm(p=2, dim=-1, keepdim=True) + 1e-8
    return tensor / norm


def cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two tensors along last dimension."""
    return F.cosine_similarity(t1, t2, dim=-1)


class Percept(nn.Module):
    """
    Shared perceptual representation network.
    For Atari, takes 84x84x3 frames (preserving color channels to distinguish items).
    """
    def __init__(self, input_shape: tuple, d: int, mlp: bool = False):
        super().__init__()
        self.mlp = mlp

        if mlp:
            input_dim = input_shape[-1]
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, d),
                nn.ReLU()
            )
        else:
            in_channels = input_shape[2] if len(input_shape) == 3 else input_shape[0]
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 9 * 9, d),
                nn.ReLU()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Manager(nn.Module):
    """
    Manager module operating at temporal horizon c.
    Outputs unit-norm direction goal vectors g_t and estimates state value V_m.
    """
    def __init__(self, c: int, d: int, r: int, eps: float, device: torch.device):
        super().__init__()
        self.c = c
        self.d = d
        self.r = r
        self.eps = eps
        self.device = device

        self.mspace = nn.Linear(self.d, self.d)
        self.mrnn = DilatedLSTM(self.d, self.d, radius=self.r)
        self.critic = nn.Linear(self.d, 1)

    def forward(self, z: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor, save: bool = True):
        state = F.relu(self.mspace(z))
        mask_expanded = mask.unsqueeze(1)  # [B, 1, 1]
        hidden = (mask_expanded * hidden[0], mask_expanded * hidden[1])

        goal_hat, hidden = self.mrnn(state, hidden, save=save)
        value_est = self.critic(goal_hat)

        goal = normalize(goal_hat)

        if self.training and torch.rand(1).item() < self.eps:
            random_goal = torch.randn_like(goal)
            goal = normalize(random_goal)

        return goal, hidden, state.detach(), value_est

    def state_goal_cosine(self, states: list[torch.Tensor], goals: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculates cosine similarity for Manager Transition Policy Gradient:
            cos(s_{t+c} - s_t, g_t(theta_M))
        states, goals, masks are lists of length 2*c + 1.
        At rollout step tau = t + c:
            states[2*c] is s_{t+c}, states[c] is s_t, goals[c] is g_t.
        """
        t = self.c
        mask = torch.stack(masks[t : t + self.c]).prod(dim=0)
        s_diff = states[t + self.c] - states[t]
        cos_sim = cosine_similarity(s_diff, goals[t]).unsqueeze(-1)
        return mask * cos_sim


class WorkerCat(nn.Module):
    """
    WorkerCat: FeUdal Network Worker variant with Direct Goal Concatenation.
    Instead of passing recent goals through a linear projection layer (phi) to modulate
    action weights via matrix product (U^T phi(g)), WorkerCat DIRECTLY CONCATENATES
    the goal vector sum [recent_goals] with sensory representation [z_t] into the Worker's recurrent state.
    """
    def __init__(self, c: int, d: int, hidden_dim: int, num_actions: int, device: torch.device):
        super().__init__()
        self.c = c
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.device = device

        # Worker LSTM takes concatenated [z_t, recent_goals] of dimension (d + d)
        self.wrnn = nn.LSTMCell(2 * d, hidden_dim)

        # Actor head predicting action logits directly from Worker hidden state
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        # Worker value critic V_w
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z: torch.Tensor, goals: list[torch.Tensor], hidden: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
        hidden = (mask * hidden[0], mask * hidden[1])

        # Detach gradient from Worker back to Manager goals (paper requirement)
        recent_goals = torch.stack(goals[-self.c:]).detach().sum(dim=0)

        # DIRECT CONCATENATION: [z_t, recent_goals] -> shape: [B, 2*d]
        worker_input = torch.cat([z, recent_goals], dim=-1)

        u, cx = self.wrnn(worker_input, hidden)
        hidden = (u, cx)

        logits = self.actor(u)
        action_probs = F.softmax(logits, dim=-1)
        value_est = self.critic(u)

        return action_probs, hidden, value_est

    def intrinsic_reward(self, states: list[torch.Tensor], goals: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        """
        Intrinsic reward for the Worker (Eq. 8 in paper):
            r^I_t = 1/c sum_{i=1}^c cos(s_t - s_{t-i}, g_{t-i})
        Evaluated at the latest state in the sliding window (t = 2 * c).
        """
        t = 2 * self.c
        b = states[0].shape[0]
        r_i = torch.zeros(b, 1, device=self.device)
        mask_cum = torch.ones(b, 1, device=self.device)

        for i in range(1, self.c + 1):
            mask_cum = mask_cum * masks[t - i]
            s_diff = states[t] - states[t - i]
            cos_sim = cosine_similarity(s_diff, goals[t - i]).unsqueeze(-1)
            r_i += mask_cum * cos_sim

        return (r_i / self.c).detach()


class FeUdalCatNetwork(nn.Module):
    """
    FeUdalCatNetwork:
    Hierarchical Architecture combining Percept, Dilated LSTM Manager,
    and Direct Goal-Concatenating Worker (WorkerCat).
    """
    def __init__(
        self,
        num_workers: int,
        input_shape: tuple,
        n_actions: int,
        hidden_dim_manager: int = 256,
        hidden_dim_worker: int = 128,
        time_horizon: int = 10,
        dilation: int = 10,
        eps: float = 1e-4,
        device: torch.device = torch.device("cpu"),
        mlp: bool = False
    ):
        super().__init__()
        self.num_workers = num_workers
        self.c = time_horizon
        self.d = hidden_dim_manager
        self.k = hidden_dim_worker
        self.n_actions = n_actions
        self.device = device
        self.mlp = mlp

        self.percept = Percept(input_shape, self.d, mlp=mlp)
        self.manager = Manager(self.c, self.d, dilation, eps, device)
        self.worker = WorkerCat(self.c, self.d, self.k, self.n_actions, device)

        self.preprocessor = Preprocessor(input_shape, device, mlp=mlp)

        # Worker recurrent state: (u, cx)
        self.worker_hidden = (
            torch.zeros(num_workers, self.k, device=device),
            torch.zeros(num_workers, self.k, device=device)
        )
        # Manager recurrent state: Dilated LSTM
        self.manager_hidden = self.manager.mrnn.init_hidden(num_workers, device)

    def init_obj(self):
        """Initialize sliding window buffers for goals, states, and masks."""
        window_size = 2 * self.c + 1
        goals = [torch.zeros(self.num_workers, self.d, device=self.device, requires_grad=True) for _ in range(window_size)]
        states = [torch.zeros(self.num_workers, self.d, device=self.device) for _ in range(window_size)]
        masks = [torch.ones(self.num_workers, 1, device=self.device) for _ in range(window_size)]
        return goals, states, masks

    def repackage_hidden(self):
        """Detach hidden states to truncate BPTT across n-step batches."""
        self.worker_hidden = (self.worker_hidden[0].detach(), self.worker_hidden[1].detach())
        self.manager_hidden = (self.manager_hidden[0].detach(), self.manager_hidden[1].detach())

    def forward(self, x, goals: list[torch.Tensor], states: list[torch.Tensor], mask: torch.Tensor, save: bool = True):
        # Observation preprocessing
        x_tensor = self.preprocessor(x)
        z = self.percept(x_tensor)

        # Manager forward
        goal, manager_hidden, state, value_m = self.manager(z, self.manager_hidden, mask, save=save)

        if save:
            self.manager_hidden = manager_hidden
            goals.pop(0)
            goals.append(goal)
            states.pop(0)
            states.append(state)
            worker_goals = goals
        else:
            worker_goals = goals[1:] + [goal]

        # Worker forward with direct goal concatenation
        action_probs, worker_hidden, value_w = self.worker(z, worker_goals, self.worker_hidden, mask)
        if save:
            self.worker_hidden = worker_hidden

        return action_probs, goals, states, value_m, value_w

    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)


def feudal_cat_loss(storage: Storage, next_v_m: torch.Tensor, next_v_w: torch.Tensor, args) -> tuple[torch.Tensor, dict]:
    """
    Computes total FuN loss for FeUdalCat.
    Exact same formulation as paper FuN to ensure 1:1 fairness in experimental comparison.
    """
    ret_m = next_v_m
    ret_w = next_v_w

    storage.placeholder(['ret_m', 'ret_w'])
    for i in reversed(range(args.num_steps)):
        ret_m = storage.r[i] + args.gamma_m * ret_m * storage.m[i]
        r_i_step = storage.r_i[i + 1] if i + 1 < args.num_steps else storage.r_i[i]
        ret_w = (storage.r[i] + args.alpha * r_i_step) + args.gamma_w * ret_w * storage.m[i]
        storage.ret_m[i] = ret_m
        storage.ret_w[i] = ret_w

    rewards_intrinsic, value_m, value_w, ret_w, ret_m, logps, entropy, state_goal_cosines = storage.stack(
        ['r_i', 'v_m', 'v_w', 'ret_w', 'ret_m', 'logp', 'entropy', 's_goal_cos']
    )

    advantage_w = ret_w - value_w
    advantage_m = ret_m - value_m

    loss_worker = (logps * advantage_w.detach()).mean()

    c = args.time_horizon
    if args.num_steps > c:
        aligned_cosines = state_goal_cosines[c:]
        aligned_adv_m = advantage_m[:-c].detach()
    else:
        aligned_cosines = state_goal_cosines
        aligned_adv_m = advantage_m.detach()

    loss_manager = (aligned_cosines * aligned_adv_m).mean()

    value_w_loss = 0.5 * advantage_w.pow(2).mean()
    value_m_loss = 0.5 * advantage_m.pow(2).mean()

    mean_entropy = entropy.mean()

    loss = -loss_worker - loss_manager + value_w_loss + value_m_loss - args.entropy_coef * mean_entropy

    loss_dict = {
        'loss/total': loss.item(),
        'loss/worker': loss_worker.item(),
        'loss/manager': loss_manager.item(),
        'loss/value_worker': value_w_loss.item(),
        'loss/value_manager': value_m_loss.item(),
        'policy/entropy': mean_entropy.item(),
        'policy/worker_advantage': advantage_w.mean().item(),
        'policy/worker_intrinsic_reward': rewards_intrinsic.mean().item(),
        'policy/manager_cosines': aligned_cosines.mean().item(),
        'policy/manager_advantage': advantage_m.mean().item(),
    }

    return loss, loss_dict
