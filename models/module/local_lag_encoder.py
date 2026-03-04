import torch.nn as nn
import torch


class DynamicDemandEncoder(nn.Module):
    def __init__(self, lag_window=None, embedding_dim=32, time_step=None, **kwargs):
        super().__init__()
        self.activate = kwargs.get("activate", False)
        self.use_near_demand = kwargs.get("use_near_demand", False)
        if lag_window is None:
            lag_window = time_step
        if lag_window is None:
            raise ValueError("lag_window or time_step must be provided.")
        self.lag_window = lag_window
        # Dynamic descriptor per step:
        # [y_{r,t}^{(l)} || xi_{r,t}^{(l)} || c_{r,t} || Delta t_last_{r,t}]
        dynamic_input_dim = 2 * self.lag_window + 2
        if self.use_near_demand:
            dynamic_input_dim += self.lag_window
        self.dynamic_linear = nn.Linear(dynamic_input_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        self.delta_max = kwargs.get("delta_max", 24)

    def _build_lag_windows(self, values):
        B, N, T = values.shape
        L = self.lag_window
        padded = torch.cat(
            [torch.zeros(B, N, L - 1, device=values.device, dtype=values.dtype), values],
            dim=-1,
        )  # (B, N, T + L - 1)
        windows = [padded[:, :, t:t + L] for t in range(T)]
        return torch.stack(windows, dim=2)  # (B, N, T, L)

    def forward(
        self,
        y_lag=None,
        m_lag=None,
        delta_t_last=None,
        near_y_lag=None,
        near_demand_series=None,
        demand_series=None,
        valid_mask=None,
        deactivation_period=None,
    ):
        demand_values = y_lag if y_lag is not None else demand_series
        mask_values = m_lag if m_lag is not None else valid_mask
        recency_values = delta_t_last if delta_t_last is not None else deactivation_period
        if demand_values is None or mask_values is None or recency_values is None:
            raise ValueError("Dynamic demand inputs require y_lag/m_lag/delta_t_last (or legacy aliases).")

        if demand_values.dim() not in (3, 4) or mask_values.dim() != demand_values.dim():
            raise ValueError(
                "Expected y_lag and m_lag with shape (B, N, T) or (B, N, T, L)."
            )
        if recency_values.dim() != 3:
            raise ValueError("delta_t_last must have shape (B, N, 1) or (B, N, T).")

        demand_values = demand_values.float()
        mask_values = mask_values.float()
        recency_values = torch.clamp(recency_values.float(), max=self.delta_max)

        if demand_values.dim() == 4:
            y_windows = demand_values
            mask_windows = mask_values
            _, _, T, L = y_windows.shape
            if L != self.lag_window:
                raise ValueError(
                    f"y_lag last dim must match lag_window={self.lag_window}, got {L}."
                )
        else:
            y_windows = self._build_lag_windows(demand_values)  # (B, N, T, L)
            mask_windows = self._build_lag_windows(mask_values)  # (B, N, T, L)
            T = demand_values.size(-1)

        demand_count = (y_windows > 0).float().sum(dim=-1, keepdim=True)  # (B, N, T, 1)
        if recency_values.size(-1) == 1:
            recency_seq = recency_values.expand(-1, -1, T)
        elif recency_values.size(-1) == T:
            recency_seq = recency_values
        else:
            raise ValueError("delta_t_last last dim must be 1 or match T.")
        recency_seq = recency_seq.unsqueeze(-1)  # (B, N, T, 1)

        dynamic_parts = [y_windows, mask_windows, demand_count, recency_seq]
        if self.use_near_demand:
            near_values = near_y_lag if near_y_lag is not None else near_demand_series
            if near_values is None:
                raise ValueError(
                    "use_near_demand=True requires near_y_lag or near_demand_series."
                )
            if near_values.dim() == 4:
                near_windows = near_values.float()
                if near_windows.size(-1) != self.lag_window:
                    raise ValueError(
                        f"near_y_lag last dim must match lag_window={self.lag_window}, got {near_windows.size(-1)}."
                    )
                if near_windows.size(2) != T:
                    raise ValueError(
                        f"near_y_lag T dim must match demand T={T}, got {near_windows.size(2)}."
                    )
            elif near_values.dim() == 3:
                if near_values.size(-1) != T:
                    raise ValueError(
                        f"near_demand_series last dim must match T={T}, got {near_values.size(-1)}."
                    )
                near_windows = self._build_lag_windows(near_values.float())  # (B, N, T, L)
            else:
                raise ValueError(
                    "near demand input must be (B, N, T) or (B, N, T, L)."
                )
            dynamic_parts.append(near_windows)

        dynamic_feat = torch.cat(dynamic_parts, dim=-1)  # (B, N, T, *)
        return self.dynamic_linear(dynamic_feat)  # (B, N, T, D_d)


# Legacy alias kept for backward compatibility.
LocalLagEncoder = DynamicDemandEncoder


__all__ = ["DynamicDemandEncoder", "LocalLagEncoder"]
