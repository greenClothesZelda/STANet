import torch.nn as nn
import torch


class TemporalWindowAggregator(nn.Module):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.tanh = nn.Tanh()
        self.score_fc = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_fc = nn.Linear(embedding_dim, 1, bias=False)
        self.output_dim = embedding_dim

    def forward(self, state):
        """Aggregate hidden states over the temporal window."""
        B, N, T, _ = state.size()
        state = state.contiguous().view(-1, state.size(2), state.size(3))  # (B*N, T, Hidden_Size)
        gru_out, _ = self.gru(state)  # (B*N, T, Hidden_Size)
        score = self.tanh(self.score_fc(gru_out))  # (B*N, T, Hidden_Size)
        score = self.value_fc(score).squeeze(-1)  # (B*N, T)
        attn_weights = torch.softmax(score, dim=-1)  # (B*N, T)
        attn_weights = attn_weights.unsqueeze(-1)  # (B*N, T, 1)
        aggregated_state = attn_weights * gru_out  # (B*N, T, Hidden_Size)
        aggregated_state = aggregated_state.sum(dim=1)  # (B*N, Hidden_Size)
        aggregated_state = aggregated_state.view(B, N, -1)  # (B, N, Hidden_Size)
        return aggregated_state  # (B, N, Hidden_Size)


# Legacy alias kept for backward compatibility.
TemporalAggregationModule = TemporalWindowAggregator


__all__ = ["TemporalWindowAggregator", "TemporalAggregationModule"]
