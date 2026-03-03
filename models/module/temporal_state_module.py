import torch.nn as nn
import torch


class TemporalStateUpdater(nn.Module):
    def __init__(self, GRU_configs, input_size, **kwargs):
        super().__init__()
        gru_configs = dict(GRU_configs)
        if gru_configs.get("num_layers", 1) == 1 and gru_configs.get("dropout", 0.0) > 0:
            gru_configs["dropout"] = 0.0
        self.hidden_size = input_size
        self.gru_num_layers = gru_configs.get("num_layers", 1)
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid(),
        )
        self.gru = nn.GRU(
            batch_first=True,
            bidirectional=False,
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            **gru_configs,
        )
        self.h0 = nn.Parameter(torch.zeros(self.gru_num_layers, 1, self.hidden_size))

    def init_hidden(self, batch_size, num_nodes, device, dtype):
        return self.h0.to(device=device, dtype=dtype).expand(
            self.gru_num_layers, batch_size * num_nodes, self.hidden_size
        ).contiguous()

    def gated_fusion(self, e_t, h_prev_top):
        gate_input = torch.cat([e_t, h_prev_top], dim=-1)  # (B, N, 2D)
        g_t = self.gate_layer(gate_input)  # (B, N, D)
        s_t = g_t * h_prev_top + (1.0 - g_t) * e_t  # (B, N, D)
        return s_t, g_t

    def gru_update(self, z_t, h_prev):
        B, N, D = z_t.size()
        z_step = z_t.reshape(B * N, 1, D)  # (B*N, 1, D)
        gru_out, h_new = self.gru(z_step, h_prev)
        h_top = gru_out.reshape(B, N, D)  # (B, N, D)
        return h_new, h_top

    def forward(self, x):
        """Fallback full-sequence update (without spatial attention injection)."""
        B, N, T, _ = x.size()
        h_prev = self.init_hidden(B, N, x.device, x.dtype)
        s_seq = []
        h_seq = []
        g_seq = []
        for t in range(T):
            e_t = x[:, :, t, :]  # (B, N, D)
            h_prev_top = h_prev[-1].reshape(B, N, self.hidden_size)
            s_t, g_t = self.gated_fusion(e_t, h_prev_top)
            h_prev, h_t = self.gru_update(s_t, h_prev)
            s_seq.append(s_t)
            h_seq.append(h_t)
            g_seq.append(g_t)
        s_seq = torch.stack(s_seq, dim=2)  # (B, N, T, D)
        h_seq = torch.stack(h_seq, dim=2)  # (B, N, T, D)
        g_seq = torch.stack(g_seq, dim=2)  # (B, N, T, D)
        return s_seq, h_seq, g_seq


# Legacy alias kept for backward compatibility.
TemporalStateModule = TemporalStateUpdater


__all__ = ["TemporalStateUpdater", "TemporalStateModule"]
