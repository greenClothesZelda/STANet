import torch.nn as nn
import torch


class TemporalStateUpdater(nn.Module):
    def __init__(self, GRU_configs, input_size, **kwargs):
        super().__init__()
        self.hidden_size = input_size  # hidden_size == input_size
        self.gru = nn.GRU(
            batch_first=True,
            bidirectional=False,
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            **GRU_configs,
        )
        self.gru_num_layers = GRU_configs.get("num_layers", 1)
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid(),
        )
        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_size))  # Initial hidden state

    def forward(self, x):
        """Update per-region temporal state with GRU and gated fusion."""
        B, N, T, _ = x.size()
        x_origin = x
        x = x.view(-1, x.size(2), x.size(3))  # (B*N, T, Embedding_Dim)
        h0 = self.h0.unsqueeze(0).expand(self.gru_num_layers, B * N, self.hidden_size).contiguous()  # (1, B*N, Hidden_Size)
        gru_out, _ = self.gru(x, h0)  # (B*N, T, Hidden_Size)
        gru_out = gru_out.view(B, N, T, self.hidden_size)  # (B, N, T, Hidden_Size)
        gate_factor = torch.cat([x_origin, gru_out], dim=-1)  # (B, N, T, Embedding_Dim*2)
        gate = self.gate_layer(gate_factor)  # (B, N, T, Hidden_Size)
        aggregated_state = gate * gru_out + (1 - gate) * x_origin
        return aggregated_state, gru_out, gate


# Legacy alias kept for backward compatibility.
TemporalStateModule = TemporalStateUpdater


__all__ = ["TemporalStateUpdater", "TemporalStateModule"]
