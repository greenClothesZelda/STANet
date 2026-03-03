import torch
import torch.nn as nn
import torch.nn.functional as F


class POIEncoder(nn.Module):
    def __init__(self, num_poi_categories, x_poi, **kwargs):
        """Encode POI counts into weighted presence/magnitude features."""
        super().__init__()
        self.activate = kwargs.get("activate", False)
        self.num_cats = num_poi_categories
        self.temporal_dim = kwargs.get("temporal_dim", 16)
        self.theta_z = nn.Parameter(torch.randn(num_poi_categories))
        self.theta_s = nn.Parameter(torch.randn(num_poi_categories))
        self.theta_gamma_w = nn.Parameter(torch.randn(num_poi_categories, self.temporal_dim))
        self.theta_gamma_b = nn.Parameter(torch.zeros(num_poi_categories))
        self.register_buffer("x_poi", x_poi)  # (N, C)
        self.output_dim = num_poi_categories

    def forward(self, temporal_context=None):
        if not self.activate:
            if temporal_context is None:
                return torch.zeros_like(self.x_poi)
            B, T, _ = temporal_context.shape
            return torch.zeros(
                (B, self.x_poi.size(0), T, self.num_cats),
                dtype=self.x_poi.dtype,
                device=self.x_poi.device,
            )
        z = (self.x_poi > 0).float()
        s = torch.log1p(self.x_poi)
        w_z = F.softplus(self.theta_z)  # (C,)
        w_s = F.softplus(self.theta_s)  # (C,)
        poi_static = w_z * z + w_s * s  # (N, C)
        if temporal_context is None:
            return poi_static

        if temporal_context.size(-1) != self.temporal_dim:
            raise ValueError(
                f"temporal_context last dim must be {self.temporal_dim}, "
                f"got {temporal_context.size(-1)}."
            )
        gamma = torch.sigmoid(
            torch.einsum("btd,cd->btc", temporal_context, self.theta_gamma_w) + self.theta_gamma_b
        )  # (B, T, C)
        poi_feat = gamma.unsqueeze(1) * poi_static.unsqueeze(0).unsqueeze(2)  # (B, N, T, C)
        return poi_feat


class StaticRegionEncoder(nn.Module):
    """Encode static region features into u_stat."""

    def __init__(
        self,
        land_composition=None,
        poi_encoder_model=None,
        x_geo=None,
        x_land=None,
        **kwargs,
    ):
        super().__init__()
        self.activate = kwargs.get("activate", False)
        if x_land is not None and land_composition is None:
            land_composition = x_land
        if land_composition is None:
            raise ValueError("land_composition or x_land must be provided.")
        if poi_encoder_model is None:
            raise ValueError("poi_encoder_model must be provided.")
        if x_geo is None:
            raise ValueError("x_geo must be provided.")

        self.register_buffer("x_land", land_composition)  # (N, C_land)
        self.poi_encoder = poi_encoder_model
        self.register_buffer("x_geo", x_geo)  # (N, 3)
        self.land_dim = land_composition.size(1)
        self.poi_dim = getattr(self.poi_encoder, "output_dim", poi_encoder_model.num_cats)
        self.geo_dim = x_geo.size(1)
        self.output_dim = kwargs.get("output_dim", 32)
        self.output_layer = nn.Linear(self.land_dim + self.poi_dim + self.geo_dim, self.output_dim)

    def forward(self, temporal_context=None):
        if temporal_context is None:
            land_feat = self.x_land
            poi_feat = self.poi_encoder()
            geo_feat = self.x_geo
            region_feat = torch.cat([land_feat, poi_feat, geo_feat], dim=1)
            return self.output_layer(region_feat)

        B, T, _ = temporal_context.shape
        N = self.x_land.size(0)
        land_feat = self.x_land.unsqueeze(0).unsqueeze(2).expand(B, N, T, self.land_dim)
        poi_feat = self.poi_encoder(temporal_context)  # (B, N, T, C_poi)
        geo_feat = self.x_geo.unsqueeze(0).unsqueeze(2).expand(B, N, T, self.geo_dim)
        region_feat = torch.cat([land_feat, poi_feat, geo_feat], dim=-1)  # (B, N, T, *)
        return self.output_layer(region_feat)  # (B, N, T, D_region)


# Legacy aliases kept for backward compatibility.
class RegeonEncoder(StaticRegionEncoder):
    pass


RegionEncoder = StaticRegionEncoder


__all__ = ["POIEncoder", "StaticRegionEncoder", "RegionEncoder", "RegeonEncoder"]
