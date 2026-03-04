import torch
import torch.nn as nn


class TemporalContextEncoder(nn.Module):
    def __init__(self, embedding_dim, weather_dim=0):
        super().__init__()
        self.day_embed = nn.Embedding(7, embedding_dim)
        self.hour_embed = nn.Embedding(24, embedding_dim)
        self.holiday_vector = nn.Parameter(torch.zeros(embedding_dim))
        self.weather_dim = int(weather_dim)
        self.weather_proj = None
        if self.weather_dim > 0:
            self.weather_proj = nn.Linear(self.weather_dim, embedding_dim)
        self.output_dim = embedding_dim

    def forward(
        self,
        dow=None,
        hod=None,
        holiday=None,
        weather=None,
        day_of_week=None,
        hour_of_day=None,
        is_holiday=None,
    ):
        day_index = dow if dow is not None else day_of_week
        hour_index = hod if hod is not None else hour_of_day
        holiday_index = holiday if holiday is not None else is_holiday
        if day_index is None or hour_index is None or holiday_index is None:
            raise ValueError("Temporal inputs require dow/hod/holiday (or legacy aliases).")

        day_feat = self.day_embed(day_index)
        hour_feat = self.hour_embed(hour_index)
        holiday_feat = holiday_index.float().unsqueeze(-1) * self.holiday_vector
        temporal_feat = day_feat + hour_feat + holiday_feat
        if self.weather_proj is not None:
            if weather is None:
                raise ValueError(
                    f"TemporalContextEncoder expects weather input with last dim {self.weather_dim}."
                )
            if weather.size(-1) != self.weather_dim:
                raise ValueError(
                    f"weather last dim must be {self.weather_dim}, got {weather.size(-1)}."
                )
            temporal_feat = temporal_feat + self.weather_proj(weather.float())
        return temporal_feat  # (B, T, D_t)


# Legacy alias kept for backward compatibility.
TemporalEncoder = TemporalContextEncoder


__all__ = ["TemporalContextEncoder", "TemporalEncoder"]
