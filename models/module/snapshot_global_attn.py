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

    def _apply_snapshot_attention(self, snapshot_state):
        """Apply spatial attention to a single snapshot (B, N, D)."""
        attn_output = self.attn(snapshot_state)  # (B, N, D)
        return self.layer_norm(attn_output + snapshot_state)  # (B, N, D)

    def forward_snapshot(self, snapshot_state, OD=None):
        # OD is intentionally unused in model2-1.pdf formulation.
        return self._apply_snapshot_attention(snapshot_state)

    def forward(self, state, OD=None):
        """Apply attention over regions for each snapshot t."""
        if state.dim() == 3:
            return self.forward_snapshot(state, OD=OD)

        B, N, T, _ = state.size()
        outputs = []
        for t in range(T):
            outputs.append(self.forward_snapshot(state[:, :, t, :], OD=None))
        return torch.stack(outputs, dim=2)  # (B, N, T, D)


# Legacy alias kept for backward compatibility.
SnapshotGlobalAttn = SnapshotGlobalAttention


__all__ = ["SnapshotGlobalAttention", "SnapshotGlobalAttn"]
