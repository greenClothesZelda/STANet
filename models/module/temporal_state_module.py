import torch.nn as nn
import torch


class TemporalStateUpdater(nn.Module):
    def __init__(self, GRU_configs, input_size, **kwargs):
        super().__init__()
        self.hidden_size = input_size
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid(),
        )
        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_size))

    def forward(self, x):
        """Build gated fused representation s_{r,t} from e_{r,t} and h_{r,t-1}."""
        B, N, T, _ = x.size()
        h_prev = self.h0.view(1, 1, self.hidden_size).expand(B, N, self.hidden_size).contiguous()
        fused_seq = []
        gate_seq = []
        for t in range(T):
            e_t = x[:, :, t, :]  # (B, N, D)
            gate_input = torch.cat([e_t, h_prev], dim=-1)  # (B, N, 2D)
            g_t = self.gate_layer(gate_input)  # (B, N, D)
            s_t = g_t * h_prev + (1.0 - g_t) * e_t  # (B, N, D)
            fused_seq.append(s_t)
            gate_seq.append(g_t)
            h_prev = s_t
        fused_state = torch.stack(fused_seq, dim=2)  # (B, N, T, D)
        gate = torch.stack(gate_seq, dim=2)  # (B, N, T, D)
        return fused_state, fused_state, gate


# Legacy alias kept for backward compatibility.
TemporalStateModule = TemporalStateUpdater


__all__ = ["TemporalStateUpdater", "TemporalStateModule"]
