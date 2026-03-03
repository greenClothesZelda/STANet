import torch.nn as nn
import torch


class DynamicDemandEncoder(nn.Module):
    def __init__(self, lag_window=None, embedding_dim=32, time_step=None, **kwargs):
        super().__init__()
        self.activate = kwargs.get("activate", False)
        if lag_window is None:
            lag_window = time_step
        if lag_window is None:
            raise ValueError("lag_window or time_step must be provided.")
        self.lag_window = lag_window
        # Per-time-step dynamic feature projection:
        # [y_{r,t-k}, xi_{r,t,k}, c_{r,t}, Delta t_last] -> D_d
        self.dynamic_linear = nn.Linear(4, embedding_dim)
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        self.delta_max = kwargs.get("delta_max", 24)

    def forward(
        self,
        y_lag=None,
        m_lag=None,
        delta_t_last=None,
        demand_series=None,
        valid_mask=None,
        deactivation_period=None,
    ):
        demand_values = y_lag if y_lag is not None else demand_series
        mask_values = m_lag if m_lag is not None else valid_mask
        recency_values = delta_t_last if delta_t_last is not None else deactivation_period
        if demand_values is None or mask_values is None or recency_values is None:
            raise ValueError("Dynamic demand inputs require y_lag/m_lag/delta_t_last (or legacy aliases).")

        if demand_values.dim() != 3 or mask_values.dim() != 3 or recency_values.dim() != 3:
            raise ValueError("Expected y_lag/m_lag with shape (B, N, T) and delta_t_last with shape (B, N, 1).")
        if recency_values.size(-1) != 1:
            raise ValueError("delta_t_last must have shape (B, N, 1).")
        if demand_values.size(-1) != self.lag_window:
            raise ValueError(
                f"y_lag last dim must match lag_window={self.lag_window}, got {demand_values.size(-1)}."
            )
        demand_values = demand_values.float()
        mask_values = mask_values.float()
        demand_count = (demand_values > 0).float().sum(dim=-1, keepdim=True)  # (B, N, 1)
        recency_values = torch.clamp(recency_values.float(), max=self.delta_max)
        demand_count_per_timestep = demand_count.expand(-1, -1, demand_values.size(-1))
        recency_per_timestep = recency_values.expand(-1, -1, demand_values.size(-1))
        dynamic_feat = torch.stack(
            [demand_values, mask_values, demand_count_per_timestep, recency_per_timestep],
            dim=-1,
        )  # (B, N, T, 4)
        dynamic_feat = self.dynamic_linear(dynamic_feat)  # (B, N, T, D_d)
        return dynamic_feat


# Legacy alias kept for backward compatibility.
LocalLagEncoder = DynamicDemandEncoder


__all__ = ["DynamicDemandEncoder", "LocalLagEncoder"]
