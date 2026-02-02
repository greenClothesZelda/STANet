import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.day_embed = nn.Embedding(7, embedding_dim)
        self.hour_embed = nn.Embedding(24, embedding_dim)
        self.holiday_embed = nn.Embedding(2, embedding_dim)
    def forward(self, day_of_week, hour_of_day, is_holiday):
        day_feat = self.day_embed(day_of_week)
        hour_feat = self.hour_embed(hour_of_day)
        holiday_feat = self.holiday_embed(is_holiday)
        temporal_feat = day_feat + hour_feat + holiday_feat
        return temporal_feat # (Batch_Size, Embedding_Dim)