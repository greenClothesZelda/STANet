import torch.nn as nn
import torch


class TemporalWindowAggregator(nn.Module):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__()
        self.tanh = nn.Tanh()
        self.score_fc = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.value_fc = nn.Linear(embedding_dim, 1, bias=False)
        self.output_dim = embedding_dim

    def forward(self, state):
        """Temporal attention over hidden sequence {h_{r,tau}}."""
        B, N, T, _ = state.size()
        state = state.contiguous().view(-1, state.size(2), state.size(3))  # (B*N, T, D)
        score = self.tanh(self.score_fc(state))  # (B*N, T, D)
        score = self.value_fc(score).squeeze(-1)  # (B*N, T)
        attn_weights = torch.softmax(score, dim=-1)  # (B*N, T)
        attn_weights = attn_weights.unsqueeze(-1)  # (B*N, T, 1)
        aggregated_state = attn_weights * state  # (B*N, T, D)
        aggregated_state = aggregated_state.sum(dim=1)  # (B*N, D)
        aggregated_state = aggregated_state.view(B, N, -1)  # (B, N, D)
        return aggregated_state


# Legacy alias kept for backward compatibility.
TemporalAggregationModule = TemporalWindowAggregator


__all__ = ["TemporalWindowAggregator", "TemporalAggregationModule"]
