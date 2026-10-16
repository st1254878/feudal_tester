import torch
import torch.nn as nn


class DilatedLSTM(nn.Module):
    """
    Dilated LSTM (dLSTM) module as proposed in FeUdal Networks (Vezhnevets et al., 2017).
    
    The recurrent state is partitioned into `radius` cores.
    At timestep t, only core (t % radius) is updated using state s_t.
    The output is pooled across all previous radius cores:
        y_t = (h^{t % r}_t + sum_{i != t % r} detached(h^i_t)) / radius
    This enables the Manager to remember over extended time horizons while maintaining
    a smooth gradient flow for the active core.
    """
    def __init__(self, input_size: int, hidden_size: int, radius: int = 10):
        super().__init__()
        self.radius = radius
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.dilation = 0

    def forward(self, state: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            state: [batch_size, input_size]
            hidden: tuple of (hx, cx), each of shape [batch_size, radius, hidden_size]
        Returns:
            y: pooled output [batch_size, hidden_size]
            (hx_next, cx_next): updated hidden states [batch_size, radius, hidden_size]
        """
        hx, cx = hidden
        idx = self.dilation

        # Current core at dilation index
        h_curr = hx[:, idx]
        c_curr = cx[:, idx]

        # Update current core with LSTMCell
        h_next, c_next = self.rnn(state, (h_curr, c_curr))

        # Build new hidden tensors without in-place mutation
        # Stack cores to keep autograd graph intact
        cores_h = [hx[:, i] if i != idx else h_next for i in range(self.radius)]
        cores_c = [cx[:, i] if i != idx else c_next for i in range(self.radius)]

        hx_new = torch.stack(cores_h, dim=1)
        cx_new = torch.stack(cores_c, dim=1)

        # Pooled output: active core (has gradient) + sum of other detached cores
        other_cores_sum = sum(hx[:, i].detach() for i in range(self.radius) if i != idx)
        y = (h_next + other_cores_sum) / self.radius

        self.dilation = (self.dilation + 1) % self.radius
        return y, (hx_new, cx_new)

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize Dilated LSTM states to zeros."""
        hx = torch.zeros(batch_size, self.radius, self.hidden_size, device=device)
        cx = torch.zeros(batch_size, self.radius, self.hidden_size, device=device)
        self.dilation = 0
        return (hx, cx)
