import torch
import torch.nn as nn
from torch.distributions import Normal

from feudalnet_continuous import ContinuousPercept, ValueNormalizer
from storage import Storage


class DummyMRNN:
    """Dummy MRNN interface for compatibility with FuN training and evaluation loops."""
    def init_hidden(self, num_workers: int, device: torch.device):
        return (torch.zeros(num_workers, 1, device=device), torch.zeros(num_workers, 1, device=device))


class DummyManager(nn.Module):
    """Dummy Manager module for Worker-only agent compatibility."""
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.mrnn = DummyMRNN()

    def forward(self, *args, **kwargs):
        pass


class WorkerCore(nn.Module):
    """
    Standard Recurrent Worker for continuous control.
    LSTM recurrent backbone + Gaussian Actor Head + Value Critic Head.
    """
    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int, device: torch.device):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.device = device

        self.wrnn = nn.LSTMCell(in_dim, hidden_dim)

        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * -0.5)

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
        hidden = (mask * hidden[0], mask * hidden[1])
        u, cx = self.wrnn(z, hidden)
        new_hidden = (u, cx)

        raw_mu = self.actor_mu(u)
        mu = torch.tanh(raw_mu)

        std = torch.exp(torch.clamp(self.log_std, -2.0, 1.0)).expand_as(mu)
        dist = Normal(mu, std)

        value_est = self.critic(u)
        return dist, new_hidden, value_est


class ContinuousWorkerOnlyNetwork(nn.Module):
    """
    Continuous Worker-Only Agent (Flat Recurrent Actor-Critic Baseline).
    Removes Manager, Dilated LSTM, high-level goal vectors, and intrinsic rewards.
    Learns purely from external environmental rewards.
    """
    def __init__(
        self,
        num_workers: int,
        obs_dim: int,
        act_dim: int,
        hidden_dim_manager: int = 256,
        hidden_dim_worker: int = 128,
        time_horizon: int = 10,
        dilation: int = 10,
        eps: float = 0.0,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.num_workers = num_workers
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d = hidden_dim_manager
        # Ensure sufficient capacity for flat control (default to 128 if 16 is passed)
        self.hidden_dim = 128 if hidden_dim_worker <= 16 else hidden_dim_worker
        self.c = time_horizon
        self.device = device

        self.percept = ContinuousPercept(obs_dim, self.d)
        self.worker = WorkerCore(self.d, self.hidden_dim, act_dim, device)
        self.manager = DummyManager(device)

        self.worker_hidden = (
            torch.zeros(num_workers, self.hidden_dim, device=device),
            torch.zeros(num_workers, self.hidden_dim, device=device)
        )
        self.manager_hidden = self.manager.mrnn.init_hidden(num_workers, device)

        # Online observation normalization buffers
        self.register_buffer("obs_mean", torch.zeros(1, obs_dim, device=device))
        self.register_buffer("obs_var", torch.ones(1, obs_dim, device=device))
        self.register_buffer("obs_count", torch.tensor(1e-4, dtype=torch.float32, device=device))

        # PopArt-lite Value Normalizer for Worker Critic
        self.val_norm_w = ValueNormalizer(device)
        self.val_norm_m = ValueNormalizer(device)

    def _normalize_obs(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        if update and self.training:
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(dim=0, unbiased=False, keepdim=True)
            batch_count = float(x.shape[0])

            delta = batch_mean - self.obs_mean
            tot_count = self.obs_count + batch_count

            new_mean = self.obs_mean + delta * batch_count / tot_count
            m_a = self.obs_var * self.obs_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * self.obs_count * batch_count / tot_count
            new_var = m2 / tot_count

            self.obs_mean.copy_(new_mean)
            self.obs_var.copy_(new_var)
            self.obs_count.copy_(tot_count)

        x_norm = (x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        return torch.clamp(x_norm, -10.0, 10.0)

    def init_obj(self):
        window_size = 2 * self.c + 1
        goals = [torch.zeros(self.num_workers, self.d, device=self.device) for _ in range(window_size)]
        states = [torch.zeros(self.num_workers, self.d, device=self.device) for _ in range(window_size)]
        masks = [torch.ones(self.num_workers, 1, device=self.device) for _ in range(window_size)]
        return goals, states, masks

    def repackage_hidden(self):
        self.worker_hidden = (self.worker_hidden[0].detach(), self.worker_hidden[1].detach())
        self.manager_hidden = (self.manager_hidden[0].detach(), self.manager_hidden[1].detach())

    def reset_workers(self, reset_mask: torch.Tensor, goals: list[torch.Tensor] = None, states: list[torch.Tensor] = None):
        if goals is not None:
            for i in range(len(goals)):
                goals[i] = goals[i] * reset_mask
        if states is not None:
            for i in range(len(states)):
                states[i] = states[i] * reset_mask
        self.worker_hidden = (
            self.worker_hidden[0] * reset_mask,
            self.worker_hidden[1] * reset_mask
        )

    def forward(self, x: torch.Tensor, goals=None, states=None, mask=None, save: bool = True, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float32)

        if mask is None:
            mask = torch.ones(x.shape[0], 1, device=self.device)

        x = self._normalize_obs(x, update=save)
        z = self.percept(x)

        action_dist, worker_hidden, value_w = self.worker(z, self.worker_hidden, mask)
        if save:
            self.worker_hidden = worker_hidden

        value_m = torch.zeros_like(value_w)
        return action_dist, goals, states, value_m, value_w

    def intrinsic_reward(self, states, goals, masks):
        b = self.num_workers if not states else states[0].shape[0]
        return torch.zeros(b, 1, device=self.device)

    def state_goal_cosine(self, states, goals, masks):
        b = self.num_workers if not states else states[0].shape[0]
        return torch.zeros(b, 1, device=self.device)


def continuous_worker_only_loss(storage: Storage, next_v_m: torch.Tensor, next_v_w: torch.Tensor, model, args) -> tuple[torch.Tensor, dict]:
    """
    A2C / Policy Gradient loss for Worker-Only baseline.
    Trained strictly on environmental returns with PopArt-lite Value Normalization.
    """
    ret_w = next_v_w
    # Time-limit truncation bootstraps (gamma * V(final_obs), already in true scale), 0 elsewhere
    has_boot = 'boot_w' in storage.keys
    storage.placeholder(['ret_w'])
    for i in reversed(range(args.num_steps)):
        boot_w = storage.boot_w[i] if has_boot else 0.0
        ret_w = storage.r[i] + boot_w + args.gamma_w * ret_w * storage.m[i]
        storage.ret_w[i] = ret_w

    value_w, ret_w, logps, entropy = storage.stack(['v_w', 'ret_w', 'logp', 'entropy'])

    undiluted = getattr(args, "undiluted_reward", True)

    # Update Value Normalizer with unnormalized returns from all workers
    model.val_norm_w.update(ret_w.detach())

    # Normalized value targets for numerical stability (O(1) MSE)
    targets_w = model.val_norm_w.normalize(ret_w, center=(not undiluted)).detach()
    loss_value_w = 0.5 * (value_w - targets_w).pow(2).mean()

    advantage_w = targets_w - value_w.detach()
    if undiluted:
        adv_w = advantage_w * float(getattr(args, "num_workers", getattr(model, "num_workers", 1)))
    else:
        adv_w = (advantage_w - advantage_w.mean()) / (advantage_w.std() + 1e-8)
    loss_policy_w = -(adv_w * logps).mean()
    loss_entropy = -args.entropy_coef * entropy.mean()

    total_loss = loss_policy_w + loss_value_w + loss_entropy

    loss_dict = {
        "loss/total": total_loss.item(),
        "loss/worker": loss_policy_w.item(),
        "loss/value_worker": loss_value_w.item(),
        "policy/entropy": entropy.mean().item(),
        "stats/return_w_mean": model.val_norm_w.mean.item(),
        "stats/return_w_std": torch.sqrt(torch.clamp(model.val_norm_w.var, min=1.0)).item(),
    }

    return total_loss, loss_dict
