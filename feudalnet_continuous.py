import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from dilated_lstm import DilatedLSTM
from storage import Storage


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor along feature dimension to unit norm (L2)."""
    norm = tensor.norm(p=2, dim=-1, keepdim=True) + 1e-8
    return tensor / norm


def cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two tensors along last dimension."""
    return F.cosine_similarity(t1, t2, dim=-1)


class ContinuousPercept(nn.Module):
    """
    MLP perceptual module for vector-based continuous control (e.g. MuJoCo).
    Maps observation vector to latent state dimension d.
    """
    def __init__(self, obs_dim: int, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, d),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousManager(nn.Module):
    """
    Manager module for continuous domain.
    Produces unit direction goal vectors g_t via Dilated LSTM and estimates V_m.
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
        state = F.elu(self.mspace(z))
        mask_expanded = mask.unsqueeze(1)
        hidden = (mask_expanded * hidden[0], mask_expanded * hidden[1])

        goal_hat, hidden = self.mrnn(state, hidden, save=save)
        value_est = self.critic(goal_hat)
        goal = normalize(goal_hat)

        if self.training and torch.rand(1).item() < self.eps:
            random_goal = torch.randn_like(goal)
            goal = normalize(random_goal)

        return goal, hidden, state.detach(), value_est

    def state_goal_cosine(self, states: list[torch.Tensor], goals: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        t = self.c
        mask = torch.stack(masks[t : t + self.c]).prod(dim=0)
        s_diff = states[t + self.c] - states[t]
        cos_sim = cosine_similarity(s_diff, goals[t]).unsqueeze(-1)
        return mask * cos_sim


class ContinuousWorker(nn.Module):
    """
    Continuous Worker module.
    Modulates continuous actions through Manager goal:
        mu_t = tanh(U_t^T phi(sum g))
        a_t ~ Normal(mu_t, sigma_t^2)
    """
    def __init__(self, c: int, d: int, k: int, act_dim: int, device: torch.device):
        super().__init__()
        self.c = c
        self.d = d
        self.k = k
        self.act_dim = act_dim
        self.device = device

        self.wrnn = nn.LSTMCell(d, k * act_dim)
        self.phi = nn.Linear(d, k, bias=False)
        self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * -0.5)

        self.critic = nn.Sequential(
            nn.Linear(k * act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z: torch.Tensor, goals: list[torch.Tensor], hidden: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
        hidden = (mask * hidden[0], mask * hidden[1])
        u, cx = self.wrnn(z, hidden)
        hidden = (u, cx)

        recent_goals = torch.stack(goals[-self.c:]).detach().sum(dim=0)
        w = self.phi(recent_goals)

        u_matrix = u.view(u.shape[0], self.k, self.act_dim)
        raw_mu = torch.einsum("bk, bka -> ba", w, u_matrix)
        mu = torch.tanh(raw_mu)  # Continuous actions bounded in [-1, 1]

        std = torch.exp(torch.clamp(self.log_std, -2.0, 1.0)).expand_as(mu)
        dist = Normal(mu, std)

        value_est = self.critic(u)
        return dist, hidden, value_est

    def intrinsic_reward(self, states: list[torch.Tensor], goals: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
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


class ContinuousFeUdalNetwork(nn.Module):
    """
    Continuous FeUdal Network (FuN) architecture for continuous control environments.
    """
    def __init__(
        self,
        num_workers: int,
        obs_dim: int,
        act_dim: int,
        hidden_dim_manager: int = 256,
        hidden_dim_worker: int = 16,
        time_horizon: int = 10,
        dilation: int = 10,
        eps: float = 1e-4,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.num_workers = num_workers
        self.c = time_horizon
        self.d = hidden_dim_manager
        self.k = hidden_dim_worker
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.percept = ContinuousPercept(obs_dim, self.d)
        self.manager = ContinuousManager(self.c, self.d, dilation, eps, device)
        self.worker = ContinuousWorker(self.c, self.d, self.k, self.act_dim, device)

        self.worker_hidden = (
            torch.zeros(num_workers, self.k * self.act_dim, device=device),
            torch.zeros(num_workers, self.k * self.act_dim, device=device)
        )
        self.manager_hidden = self.manager.mrnn.init_hidden(num_workers, device)

        # Online observation normalization buffers (saved automatically in state_dict)
        self.register_buffer("obs_mean", torch.zeros(1, obs_dim, device=device))
        self.register_buffer("obs_var", torch.ones(1, obs_dim, device=device))
        self.register_buffer("obs_count", torch.tensor(1e-4, dtype=torch.float32, device=device))

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
        goals = [torch.zeros(self.num_workers, self.d, device=self.device, requires_grad=True) for _ in range(window_size)]
        states = [torch.zeros(self.num_workers, self.d, device=self.device) for _ in range(window_size)]
        masks = [torch.ones(self.num_workers, 1, device=self.device) for _ in range(window_size)]
        return goals, states, masks

    def repackage_hidden(self):
        self.worker_hidden = (self.worker_hidden[0].detach(), self.worker_hidden[1].detach())
        self.manager_hidden = (self.manager_hidden[0].detach(), self.manager_hidden[1].detach())

    def forward(self, x: torch.Tensor, goals: list[torch.Tensor], states: list[torch.Tensor], mask: torch.Tensor, save: bool = True):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float32)

        x = self._normalize_obs(x, update=save)
        z = self.percept(x)
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

        action_dist, worker_hidden, value_w = self.worker(z, worker_goals, self.worker_hidden, mask)
        if save:
            self.worker_hidden = worker_hidden

        return action_dist, goals, states, value_m, value_w

    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)


def continuous_feudal_loss(storage: Storage, next_v_m: torch.Tensor, next_v_w: torch.Tensor, args) -> tuple[torch.Tensor, dict]:
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

    # Critic learns true unnormalized returns (preserves value scale across rollout boundaries)
    advantage_w = ret_w - value_w
    advantage_m = ret_m - value_m

    value_w_loss = 0.5 * advantage_w.pow(2).mean()
    value_m_loss = 0.5 * advantage_m.pow(2).mean()

    # Standardize advantages before policy gradient updates to prevent velocity-induced gradient spikes
    adv_w_norm = (advantage_w.detach() - advantage_w.detach().mean()) / (advantage_w.detach().std() + 1e-8)
    adv_m_norm = (advantage_m.detach() - advantage_m.detach().mean()) / (advantage_m.detach().std() + 1e-8)

    loss_worker = (logps * adv_w_norm).mean()

    c = args.time_horizon
    if args.num_steps > c:
        aligned_cosines = state_goal_cosines[c:]
        aligned_adv_m = adv_m_norm[:-c]
    else:
        aligned_cosines = state_goal_cosines
        aligned_adv_m = adv_m_norm

    loss_manager = (aligned_cosines * aligned_adv_m).mean()

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
