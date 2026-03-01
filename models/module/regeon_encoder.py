import torch
import torch.nn as nn
import torch.nn.functional as F


class POIEncoder(nn.Module):
    def __init__(self, num_poi_categories, x_poi, **kwargs):
        """Encode POI counts into weighted presence/magnitude features."""
        super().__init__()
        self.activate = kwargs.get("activate", False)
        self.num_cats = num_poi_categories
        self.theta_z = nn.Parameter(torch.randn(num_poi_categories))
        self.theta_s = nn.Parameter(torch.randn(num_poi_categories))
        self.register_buffer("x_poi", x_poi)  # (N, C)
        self.output_dim = num_poi_categories

    def forward(self):
        if not self.activate:
            return torch.zeros_like(self.x_poi)
        z = (self.x_poi > 0).float()
        s = torch.log1p(self.x_poi)
        w_z = F.softplus(self.theta_z)  # (C,)
        w_s = F.softplus(self.theta_s)  # (C,)
        feat = w_z * z + w_s * s  # (N, C)
        return feat


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

        land_dim = land_composition.size(1)
        poi_dim = getattr(self.poi_encoder, "output_dim", poi_encoder_model.num_cats)
        geo_dim = x_geo.size(1)
        self.output_dim = kwargs.get("output_dim", land_dim + poi_dim + geo_dim)
        self.output_layer = nn.Linear(land_dim + poi_dim + geo_dim, self.output_dim)

    def forward(self):
        land_feat = self.x_land
        poi_feat = self.poi_encoder()
        geo_feat = self.x_geo
        region_feat = torch.cat([land_feat, poi_feat, geo_feat], dim=1)
        region_feat = self.output_layer(region_feat)
        return region_feat


# Legacy aliases kept for backward compatibility.
class RegeonEncoder(StaticRegionEncoder):
    pass


RegionEncoder = StaticRegionEncoder


__all__ = ["POIEncoder", "StaticRegionEncoder", "RegionEncoder", "RegeonEncoder"]
