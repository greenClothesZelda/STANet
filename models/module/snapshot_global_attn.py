import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.use_od_bias = kwargs.get("use_od_bias", True)
        self.od_eps = kwargs.get("od_eps", 1e-6)
        self.theta_od = nn.Parameter(torch.tensor(kwargs.get("lambda_od_init", 0.0)))

    def forward(self, state, OD=None):
        """Apply attention over regions for each snapshot t."""
        B, N, T, D = state.size()
        state_reshaped = state.permute(0, 2, 1, 3).reshape(B * T, N, D)

        attn_bias = None
        if self.use_od_bias and OD is not None:
            if OD.dim() != 4:
                raise ValueError(f"OD must have shape (B, T, N, N), got {tuple(OD.shape)}.")
            if OD.size(0) != B or OD.size(1) != T or OD.size(2) != N or OD.size(3) != N:
                raise ValueError(
                    f"OD shape mismatch: expected ({B}, {T}, {N}, {N}), got {tuple(OD.shape)}."
                )
            od_log = torch.log1p(OD.float())  # x_{rj,t} = log(1 + OD_{r->j,t})
            od_mean = od_log.mean(dim=(-2, -1), keepdim=True)
            od_std = od_log.std(dim=(-2, -1), keepdim=True, unbiased=False)
            od_norm = (od_log - od_mean) / (od_std + self.od_eps)
            lambda_od = F.softplus(self.theta_od)
            attn_bias = (lambda_od * od_norm).reshape(B * T, N, N)

        attn_output = self.attn(state_reshaped, attn_bias=attn_bias)  # (B*T, N, D)

        attn_output = self.layer_norm(
            attn_output + state_reshaped)  # (B*T, N, D)

        attn_output = attn_output.view(
            B, T, N, D).permute(0, 2, 1, 3)  # (B, N, T, D)
        return attn_output


# Legacy alias kept for backward compatibility.
SnapshotGlobalAttn = SnapshotGlobalAttention


__all__ = ["SnapshotGlobalAttention", "SnapshotGlobalAttn"]
