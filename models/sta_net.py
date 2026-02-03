import torch
import torch.nn as nn
from models.module import *

class STANet(nn.Module):
    def __init__(self, POIEncoder_configs, RegeonEncoder_configs, TemporalEncoder_configs, DynamicEncoder_configs, **kwargs):
        super().__init__()
        self.poi_encoder = POIEncoder(**POIEncoder_configs)
        self.regeion_encoder = RegeonEncoder(**RegeonEncoder_configs)
        self.temporal_encoder = TemporalEncoder(**TemporalEncoder_configs)
        self.dynamic_encoder = DynamicEncoder(**DynamicEncoder_configs)
        
    def forward(self, demands_inputs, temporal_inputs):
        region_features = self.regeion_encoder() # (N, D_region)
        N, D_region = region_features.size()
        
        temporal_features = self.temporal_encoder(**temporal_inputs) # (B, D_temporal)
        B, D_temporal = temporal_features.size()
        
        dynamic_features = self.dynamic_encoder(**demands_inputs) # (B, N, D_dynamic)
        _, _, D_dynamic = dynamic_features.size()
        
        temporal_features = temporal_features.unsqueeze(1).expand(B, N, D_temporal) # (B, N, D_temporal)
        region_features = region_features.unsqueeze(0).expand(B, N, D_region) # (B, N, D_region)
        
        combinded_features = torch.cat([region_features, temporal_features, dynamic_features], dim=-1) # (B, N, D_region + D_temporal + D_dynamic)

        