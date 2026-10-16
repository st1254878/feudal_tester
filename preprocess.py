import numpy as np
import torch


class RunningMeanStd:
    """
    Tracks running mean and variance of observations for online normalization.
    As described in OpenAI Baselines and the FuN paper.
    """
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0, dtype=np.float32)
        batch_var = np.var(x, axis=0, dtype=np.float32)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class Preprocessor:
    """
    Observation preprocessor that permutes channel dimensions to PyTorch format (B, C, H, W)
    and centers observations around zero on GPU.
    """
    def __init__(self, obs_shape: tuple, device: torch.device, mlp: bool = False):
        self.mlp = mlp
        self.device = device

        if mlp:
            self.shape = (obs_shape[-1],)
        else:
            # Gym/Gymnasium Atari obs are (H, W, C) -> PyTorch expects (C, H, W)
            if len(obs_shape) == 3:
                self.shape = (obs_shape[2], obs_shape[0], obs_shape[1])
            else:
                self.shape = obs_shape

        self.rms = RunningMeanStd(shape=(1,) + self.shape)

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """
        Processes a batch of observations from vectorized envs.
        x: [B, H, W, C] or [B, dim]
        """
        x = np.asarray(x, dtype=np.float32)
        if not self.mlp and x.ndim == 4:
            # Transpose [B, H, W, C] to [B, C, H, W]
            x = np.transpose(x, (0, 3, 1, 2))

        self.rms.update(x)
        # Fast GPU subtraction with zero CPU heap allocation
        x_tensor = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        mean_tensor = torch.as_tensor(self.rms.mean, device=self.device, dtype=torch.float32)
        return x_tensor - mean_tensor
