import torch
import torch.nn as nn


class SnapshotGlobalAttention(nn.Module):
    """Integrate region interactions using snapshot self-attention."""

    def __init__(self, embedding_dim, nhead=None, attn_module=None, attn_configs=None, **kwargs):
        super().__init__()
        if nhead is None:
            nhead = kwargs.pop("num_heads", None)
        if nhead is None:
            raise ValueError("nhead or num_heads must be provided.")
        if attn_module is None:
            raise ValueError("attn_module must be provided.")
        if attn_configs is None:
            attn_configs = {}

        self.nhead = nhead
        self.attn = attn_module(d_model=embedding_dim,
                                n_heads=nhead, **attn_configs)
        self.layer_norm = nn.LayerNorm(
            embedding_dim, eps=kwargs.get("layer_norm_eps", 1e-5))
        self.output_dim = embedding_dim

    def forward(self, state, OD=None):
        """Apply attention over regions for each snapshot t."""
        B, N, T, D = state.size()
        state_reshaped = state.permute(0, 2, 1, 3).reshape(B * T, N, D)
        attn_output = self.attn(state_reshaped)  # (B*T, N, D)

        attn_output = self.layer_norm(
            attn_output + state_reshaped)  # (B*T, N, D)

        attn_output = attn_output.view(
            B, T, N, D).permute(0, 2, 1, 3)  # (B, N, T, D)
        return attn_output


# Legacy alias kept for backward compatibility.
SnapshotGlobalAttn = SnapshotGlobalAttention


__all__ = ["SnapshotGlobalAttention", "SnapshotGlobalAttn"]
