import torch
import torch.nn as nn
import torch.nn.functional as F

class POIEncoder(nn.Module):
    def __init__(self, num_poi_categories, x_poi, **kwargs):
        '''
        Docstring for __init__
        
        :param num_poi_categories: 주요 건물 카테고리 수
        :param x_poi: 노드내 주요 건물 정보 (N, C)
        :param kwargs: Description
        '''
        super().__init__()
        self.activate = kwargs.get('activate', False)
        self.num_cats = num_poi_categories
        self.theta_z = nn.Parameter(torch.randn(num_poi_categories))
        self.theta_s = nn.Parameter(torch.randn(num_poi_categories))
        self.register_buffer('x_poi', x_poi) # (N, C)
        self.output_dim = num_poi_categories

    def forward(self):
        if not self.activate:
            return torch.zeros_like(self.x_poi)
        # 1. 분해 (Decomposition)
        z = (self.x_poi > 0).float()              
        s = torch.log(1 + self.x_poi)                
        
        # 2. 가중치 적용 (Softplus)
        w_z = F.softplus(self.theta_z)         #(C,)
        w_s = F.softplus(self.theta_s)         #(C,)
        
        # 3. 결합
        # 브로드캐스팅을 위해 차원 맞춤
        feat = w_z * z + w_s * s               #(N, C)
        return feat
    
class RegeonEncoder(nn.Module):
    def __init__(self, land_composition, poi_encoder_model, x_geo, **kwargs):
        '''
        Docstring for __init__
        
        :param land_composition: 노드 구성 정보 (N, C) eg. 상업지구, 주거지구
        :param POIEncoder: 노드내 주요 건물 정보 인코더
        :param x_geo: 노드 지리 정보 (N, 3) lat, lon, log(1+ area)
        '''
        super().__init__()
        self.activate = kwargs.get('activate', False)
        self.register_buffer('land_x', land_composition) # N, C
        self.poi_encoder = poi_encoder_model
        self.register_buffer('x_geo', x_geo) # (N, 3) lat, lon, log(1+ area)
        land_dim = land_composition.size(1)
        poi_dim = getattr(self.poi_encoder, 'output_dim', poi_encoder_model.num_cats)
        geo_dim = x_geo.size(1)
        self.output_dim = land_dim + poi_dim + geo_dim
        
    def forward(self):
        land_feat = self.land_x
        poi_feat = self.poi_encoder()
        geo_feat = self.x_geo
        region_feat = torch.cat([land_feat, poi_feat, geo_feat], dim=1)
        return region_feat # (N, land_C + poi_C + 3)