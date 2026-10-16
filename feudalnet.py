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
            # 3 channels (RGB) x 84 x 84
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

    def forward(self, z: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
        state = F.relu(self.mspace(z))
        # Mask out hidden state on episode boundary
        # hidden[0] and hidden[1] are [B, radius, d]
        mask_expanded = mask.unsqueeze(1)  # [B, 1, 1]
        hidden = (mask_expanded * hidden[0], mask_expanded * hidden[1])

        goal_hat, hidden = self.mrnn(state, hidden)
        value_est = self.critic(goal_hat)

        # Normalize goal_hat to unit vector g
        goal = normalize(goal_hat)

        # Exploration epsilon: occasionally emit random Gaussian directional goal
        if self.training and torch.rand(1).item() < self.eps:
            random_goal = torch.randn_like(goal)
            goal = normalize(random_goal)

        return goal, hidden, state.detach(), value_est

    def state_goal_cosine(self, states: list[torch.Tensor], goals: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculates cosine similarity for Manager Transition Policy Gradient:
            cos(s_{t+c} - s_t, g_t(theta_M))
        states, goals, masks are lists of length 2*c + 1 centered at t = c.
        """
        t = self.c
        # Cumulative mask from t to t+c-1
        mask = torch.stack(masks[t : t + self.c]).prod(dim=0)
        s_diff = states[t + self.c] - states[t]
        cos_sim = cosine_similarity(s_diff, goals[t]).unsqueeze(-1)
        return mask * cos_sim


class Worker(nn.Module):
    """
    Worker module operating at every environment step.
    Combines sensory representation z_t with goal projection w_t = phi(sum(g_{t-i}))
    to compute policy action distribution and Worker value V_w.
    """
    def __init__(self, c: int, d: int, k: int, num_actions: int, device: torch.device):
        super().__init__()
        self.c = c
        self.d = d
        self.k = k
        self.num_actions = num_actions
        self.device = device

        self.wrnn = nn.LSTMCell(d, k * num_actions)
        self.phi = nn.Linear(d, k, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(k * num_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z: torch.Tensor, goals: list[torch.Tensor], hidden: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
        # Mask out hidden state on episode boundary
        hidden = (mask * hidden[0], mask * hidden[1])
        u, cx = self.wrnn(z, hidden)
        hidden = (u, cx)

        # Cut gradient from Worker back to Manager goals
        recent_goals = torch.stack(goals[-self.c:]).detach().sum(dim=0)
        w = self.phi(recent_goals)  # [B, k]
        value_est = self.critic(u)

        # Reshape u to [B, k, num_actions]
        u_matrix = u.view(u.shape[0], self.k, self.num_actions)
        # Action logits: a = einsum("bk, bka -> ba", w, u)
        logits = torch.einsum("bk, bka -> ba", w, u_matrix)
        action_probs = F.softmax(logits, dim=-1)

        return action_probs, hidden, value_est

    def intrinsic_reward(self, states: list[torch.Tensor], goals: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        """
        Intrinsic reward for the Worker (Eq. 8 in paper):
            r^I_t = 1/c sum_{i=1}^c cos(s_t - s_{t-i}, g_{t-i})
        """
        t = self.c
        b = states[0].shape[0]
        r_i = torch.zeros(b, 1, device=self.device)
        mask_cum = torch.ones(b, 1, device=self.device)

        for i in range(1, self.c + 1):
            s_diff = states[t] - states[t - i]
            cos_sim = cosine_similarity(s_diff, goals[t - i]).unsqueeze(-1)
            r_i += mask_cum * cos_sim
            mask_cum = mask_cum * masks[t - i]

        return (r_i / self.c).detach()


class FeUdalNetwork(nn.Module):
    """
    Complete FeUdal Network (FuN) integrating Percept, Manager, and Worker.
    """
    def __init__(
        self,
        num_workers: int,
        input_shape: tuple,
        n_actions: int,
        hidden_dim_manager: int = 256,
        hidden_dim_worker: int = 16,
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
        self.worker = Worker(self.c, self.d, self.k, self.n_actions, device)

        self.preprocessor = Preprocessor(input_shape, device, mlp=mlp)

        # Worker recurrent state: (u, cx)
        self.worker_hidden = (
            torch.zeros(num_workers, self.k * self.n_actions, device=device),
            torch.zeros(num_workers, self.k * self.n_actions, device=device)
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
        goal, self.manager_hidden, state, value_m = self.manager(z, self.manager_hidden, mask)

        if save:
            goals.pop(0)
            goals.append(goal)
            states.pop(0)
            states.append(state)

        # Worker forward
        action_probs, self.worker_hidden, value_w = self.worker(z, goals, self.worker_hidden, mask)

        return action_probs, goals, states, value_m, value_w

    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)


def feudal_loss(storage: Storage, next_v_m: torch.Tensor, next_v_w: torch.Tensor, args) -> tuple[torch.Tensor, dict]:
    """
    Computes total FuN loss combining Manager loss, Worker loss, value losses, and entropy bonus.
    """
    ret_m = next_v_m
    ret_w = next_v_w

    storage.placeholder(['ret_m', 'ret_w'])
    for i in reversed(range(args.num_steps)):
        ret_m = storage.r[i] + args.gamma_m * ret_m * storage.m[i]
        ret_w = (storage.r[i] + args.alpha * storage.r_i[i]) + args.gamma_w * ret_w * storage.m[i]
        storage.ret_m[i] = ret_m
        storage.ret_w[i] = ret_w

    rewards_intrinsic, value_m, value_w, ret_w, ret_m, logps, entropy, state_goal_cosines = storage.stack(
        ['r_i', 'v_m', 'v_w', 'ret_w', 'ret_m', 'logp', 'entropy', 's_goal_cos']
    )

    advantage_w = ret_w - value_w
    advantage_m = ret_m - value_m

    loss_worker = (logps * advantage_w.detach()).mean()
    loss_manager = (state_goal_cosines * advantage_m.detach()).mean()

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
        'policy/manager_cosines': state_goal_cosines.mean().item(),
        'policy/manager_advantage': advantage_m.mean().item(),
    }

    return loss, loss_dict
