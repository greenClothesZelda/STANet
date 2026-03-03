import torch
import torch.nn as nn


class TemporalContextEncoder(nn.Module):
    def __init__(self, embedding_dim, weather_input_dim=4):
        super().__init__()
        self.day_embed = nn.Embedding(7, embedding_dim)
        self.hour_embed = nn.Embedding(24, embedding_dim)
        self.holiday_embed = nn.Embedding(2, embedding_dim)
        self.weather_input_dim = weather_input_dim
        self.weather_linear = nn.Linear(weather_input_dim, embedding_dim)
        self.output_dim = embedding_dim

    def forward(
        self,
        dow=None,
        hod=None,
        holiday=None,
        weather_features=None,
        meteo_features=None,
        day_of_week=None,
        hour_of_day=None,
        is_holiday=None,
    ):
        day_index = dow if dow is not None else day_of_week
        hour_index = hod if hod is not None else hour_of_day
        holiday_index = holiday if holiday is not None else is_holiday
        weather_input = weather_features if weather_features is not None else meteo_features
        if day_index is None or hour_index is None or holiday_index is None:
            raise ValueError("Temporal inputs require dow/hod/holiday (or legacy aliases).")

        day_feat = self.day_embed(day_index)
        hour_feat = self.hour_embed(hour_index)
        holiday_feat = self.holiday_embed(holiday_index)
        temporal_feat = day_feat + hour_feat + holiday_feat
        if weather_input is not None:
            if weather_input.dim() != 3:
                raise ValueError(
                    f"weather_features must have shape (B, T, {self.weather_input_dim}), "
                    f"got {tuple(weather_input.shape)}."
                )
            if weather_input.size(-1) != self.weather_input_dim:
                raise ValueError(
                    f"weather_features last dim must be {self.weather_input_dim}, "
                    f"got {weather_input.size(-1)}."
                )
            weather_feat = self.weather_linear(weather_input.float())
            temporal_feat = temporal_feat + weather_feat
        return temporal_feat  # (B, T, D_t)


# Legacy alias kept for backward compatibility.
TemporalEncoder = TemporalContextEncoder


__all__ = ["TemporalContextEncoder", "TemporalEncoder"]
