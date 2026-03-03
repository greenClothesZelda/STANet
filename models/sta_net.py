import torch
import torch.nn as nn
from models.module import (
    DynamicDemandEncoder,
    POIEncoder,
    SnapshotGlobalAttention,
    StaticRegionEncoder,
    TemporalContextEncoder,
    TemporalStateUpdater,
    TemporalWindowAggregator,
)


class PTReLU(nn.Module):
    def __init__(self, alpha=1.0, n=3.0):
        super().__init__()
        self.alpha = alpha
        self.n = n

    def forward(self, x):
        out = torch.zeros_like(x)
        mask1 = (x > 0) & (x <= self.alpha)
        mask2 = (x > self.alpha)
        out[mask1] = (1.0 / (self.alpha ** (self.n - 1))) * \
            (x[mask1] ** self.n)
        out[mask2] = x[mask2]
        return out


class STANet(nn.Module):
    def __init__(
        self,
        embedding_dim,
        POIEncoder_configs=None,
        StaticRegionEncoder_configs=None,
        TemporalContextEncoder_configs=None,
        TemporalStateUpdater_configs=None,
        DynamicDemandEncoder_configs=None,
        SnapshotGlobalAttention_configs=None,
        TemporalWindowAggregator_configs=None,
        PTReLU_configs=None,
        attn_module=None,
        attn_configs=None,
        **kwargs,
    ):
        super().__init__()
        poi_encoder_configs = POIEncoder_configs or {}
        static_region_encoder_configs = StaticRegionEncoder_configs
        if static_region_encoder_configs is None:
            static_region_encoder_configs = kwargs.pop("RegeonEncoder_configs", {})
        temporal_context_encoder_configs = TemporalContextEncoder_configs
        if temporal_context_encoder_configs is None:
            temporal_context_encoder_configs = kwargs.pop("TemporalEncoder_configs", {})
        temporal_state_updater_configs = TemporalStateUpdater_configs
        if temporal_state_updater_configs is None:
            temporal_state_updater_configs = kwargs.pop("TemporalStateModule_configs", {})
        dynamic_demand_encoder_configs = DynamicDemandEncoder_configs
        if dynamic_demand_encoder_configs is None:
            dynamic_demand_encoder_configs = kwargs.pop("LocalLagEncoder_configs", {})
        snapshot_global_attention_configs = SnapshotGlobalAttention_configs
        if snapshot_global_attention_configs is None:
            snapshot_global_attention_configs = kwargs.pop("SnapshotGlobalAttn_configs", {})
        temporal_window_aggregator_configs = TemporalWindowAggregator_configs
        if temporal_window_aggregator_configs is None:
            temporal_window_aggregator_configs = kwargs.pop("TemporalAggregationModule_configs", {})
        pt_relu_configs = PTReLU_configs or {}
        attn_configs = attn_configs or {}
        nhead = snapshot_global_attention_configs.get(
            "nhead",
            snapshot_global_attention_configs.get("num_heads"),
        )
        if nhead is None:
            raise KeyError("SnapshotGlobalAttention config requires 'nhead' or 'num_heads'.")

        poi_encoder = POIEncoder(**poi_encoder_configs)
        self.static_region_encoder = StaticRegionEncoder(
            **static_region_encoder_configs,
            poi_encoder_model=poi_encoder,
        )
        self.D_region = self.static_region_encoder.output_dim
        self.temporal_context_encoder = TemporalContextEncoder(**temporal_context_encoder_configs)
        self.D_temporal = self.temporal_context_encoder.output_dim
        self.dynamic_demand_encoder = DynamicDemandEncoder(**dynamic_demand_encoder_configs)
        self.D_dynamic = self.dynamic_demand_encoder.output_dim

        self.embedding_dim = embedding_dim
        self.initial_linear = nn.Linear(
            self.D_region + self.D_temporal + self.D_dynamic,
            embedding_dim,
        )
        self.initial_activation = nn.ReLU()

        self.temporal_state_updater = TemporalStateUpdater(
            **temporal_state_updater_configs,
            input_size=embedding_dim,
        )
        self.spatial_module = SnapshotGlobalAttention(
            **snapshot_global_attention_configs,
            embedding_dim=embedding_dim,
            attn_module=attn_module,
            attn_configs=attn_configs,
        )
        self.temporal_window_aggregator = TemporalWindowAggregator(
            **temporal_window_aggregator_configs,
            embedding_dim=embedding_dim,
            nhead=nhead,
            attn_module=attn_module,
            attn_configs=attn_configs,
        )
        self.event_head = nn.Linear(embedding_dim, 1)
        self.magnitude_head = nn.Linear(embedding_dim, 1)

        self.pt_relu = PTReLU(
            alpha=pt_relu_configs.get("alpha", 0.5),
            n=pt_relu_configs.get("n", 5.0),
        )

    def forward(self, demand_features, temporal_features, OD_matrix=None, od_matrix=None):
        if OD_matrix is None:
            OD_matrix = od_matrix
        u_time = self.temporal_context_encoder(**temporal_features)  # (B, T, D_t)
        B, T, _ = u_time.size()
        u_reg = self.static_region_encoder(u_time)  # (B, N, T, D_r)
        _, N, _, _ = u_reg.size()

        u_dyn = self.dynamic_demand_encoder(**demand_features)  # (B, N, T, D_d)

        u_time = u_time.unsqueeze(1).expand(
            B, N, T, self.D_temporal)  # (B, N, T, D_temporal)
        # (B, N, T, D_r + D_t + D_d)
        combined_features = torch.cat([u_reg, u_time, u_dyn], dim=-1)
        e_seq = self.initial_activation(
            self.initial_linear(combined_features)
        )  # (B, N, T, embedding_dim)

        h_prev = self.temporal_state_updater.init_hidden(
            batch_size=B,
            num_nodes=N,
            device=e_seq.device,
            dtype=e_seq.dtype,
        )
        h_seq = []
        for t in range(T):
            e_t = e_seq[:, :, t, :]  # (B, N, D)
            h_prev_top = h_prev[-1].reshape(B, N, self.embedding_dim)
            s_t, _ = self.temporal_state_updater.gated_fusion(e_t, h_prev_top)  # (B, N, D)
            z_t = self.spatial_module(s_t)  # (B, N, D), includes residual + layer norm
            h_prev, h_t = self.temporal_state_updater.gru_update(z_t, h_prev)  # (L, B*N, D), (B, N, D)
            h_seq.append(h_t)

        h_seq = torch.stack(h_seq, dim=2)  # (B, N, T, D)
        state = self.temporal_window_aggregator(h_seq)  # (B, N, D)
        p_event = torch.sigmoid(
            self.event_head(state)).squeeze(-1)  # (B, N)
        y_hat_pos = torch.nn.functional.softplus(
            self.magnitude_head(state)).squeeze(-1)  # (B, N)
        y_hat = p_event * y_hat_pos  # (B, N)
        return {
            "event_prob": p_event,
            "magnitude": y_hat_pos,
            "prediction": y_hat,
            "p_event": p_event,
            "y_hat_pos": y_hat_pos,
            "y_hat": y_hat,
        }

    # Legacy attribute aliases (read-only properties for backward compatibility).
    @property
    def regeion_encoder(self):
        return self.static_region_encoder

    @property
    def temporal_encoder(self):
        return self.temporal_context_encoder

    @property
    def dynamic_encoder(self):
        return self.dynamic_demand_encoder

    @property
    def temporal_state_module(self):
        return self.temporal_state_updater

    @property
    def temporal_aggregation_module(self):
        return self.temporal_window_aggregator

    @property
    def ptRelu(self):
        return self.pt_relu
