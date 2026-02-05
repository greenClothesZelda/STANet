import torch
import torch.nn as nn
from models.module import *


class STANet(nn.Module):
    def __init__(self,
                 POIEncoder_configs, RegeonEncoder_configs, TemporalEncoder_configs,
                 TemporalStateModule_configs, DynamicEncoder_configs, embedding_dim,
                 SnapshotGlobalAttn_configs, TemporalAggregationModule_configs, **kwargs):
        super().__init__()
        # 입력데이터 embedding 모듈들
        poi_encoder = POIEncoder(**POIEncoder_configs)
        self.regeion_encoder = RegeonEncoder(
            **RegeonEncoder_configs, poi_encoder_model=poi_encoder)
        self.D_region = self.regeion_encoder.output_dim
        self.temporal_encoder = TemporalEncoder(**TemporalEncoder_configs)
        self.D_temporal = self.temporal_encoder.output_dim
        self.dynamic_encoder = DynamicEncoder(**DynamicEncoder_configs)
        self.D_dynamic = self.dynamic_encoder.output_dim

        self.embedding_dim = embedding_dim
        self.initial_linear = nn.Linear(
            self.D_region + self.D_temporal + self.D_dynamic, embedding_dim)

        # 데이터 처리 모듈 (시공간 attention)
        self.temporal_state_module = TemporalStateModule(
            **TemporalStateModule_configs, input_size=embedding_dim)
        self.spatial_module = SnapshotGlobalAttn(**SnapshotGlobalAttn_configs, embedding_dim=embedding_dim)
        self.temporal_aggregation_module = TemporalAggregationModule(
            **TemporalAggregationModule_configs, embedding_dim=embedding_dim)
        self.event_head = nn.Linear(embedding_dim, 1)
        self.magnitude_head = nn.Linear(embedding_dim, 1)

    def forward(self, demand_features, temporal_features):
        region_features = self.regeion_encoder()  # (N, D_region)
        N, _ = region_features.size()

        temporal_emb = self.temporal_encoder(
            **temporal_features)  # (B, T, D_temporal)
        B, T, _ = temporal_emb.size()

        dynamic_emb = self.dynamic_encoder(
            **demand_features)  # (B, N, D_dynamic)

        temporal_emb = temporal_emb.unsqueeze(1).expand(
            B, N, T, self.D_temporal)  # (B, N, T, D_temporal)
        region_features = region_features.unsqueeze(0).unsqueeze(
            2).expand(B, N, T, self.D_region)  # (B, N, T, D_region)
        dynamic_emb = dynamic_emb.unsqueeze(2).expand(
            B, N, T, self.D_dynamic)  # (B, N, T, D_dynamic)

        # (B, N, T, D_region + D_temporal + D_dynamic)
        combinded_features = torch.cat(
            [region_features, temporal_emb, dynamic_emb], dim=-1)
        combined_features = self.initial_linear(
            combinded_features)  # (B, N, T, Embedding_Dim)
        state, gru_out, gate = self.temporal_state_module(
            combined_features)  # (B, N, T, Embedding_Dim)
        state = self.spatial_module(state)  # (B, N, T, Embedding_Dim)
        state = self.temporal_aggregation_module(
            state)  # (B, N, Embedding_Dim)
        event_prob = torch.sigmoid(
            self.event_head(state)).squeeze(-1)  # (B, N)
        magnitude = torch.nn.functional.softplus(
            self.magnitude_head(state)).squeeze(-1)  # (B, N)
        prediction = event_prob * magnitude  # (B, N)
        return {
            'event_prob': event_prob,
            'magnitude': magnitude,
            'prediction': prediction,
        }
