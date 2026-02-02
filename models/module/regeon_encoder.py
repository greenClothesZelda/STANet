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
    def __init__(self, land_composition, POIEncoder, x_geo, **kwargs):
        '''
        Docstring for __init__
        
        :param land_composition: 노드 구성 정보 (N, C) eg. 상업지구, 주거지구
        :param POIEncoder: 노드내 주요 건물 정보 인코더
        :param x_geo: 노드 지리 정보 (N, 3) lat, lon, log(1+ area)
        '''
        super().__init__()
        self.activate = kwargs.get('activate', False)
        self.register_buffer('land_x', torch.tensor(land_composition, dtype=torch.float32)) # N, C
        self.poi_encoder = POIEncoder(**POIEncoder)
        self.register_buffer('x_geo', x_geo) # (N, 3) lat, lon, log(1+ area)
        
    def forward(self):
        if not self.activate:
            return torch.zeros_like(self.land_x.shape[0], self.land_x.shape[1] + self.x_geo.shape[1] + self.poi_encoder().shape[1])
        land_feat = self.land_x
        poi_feat = self.poi_encoder()
        geo_feat = self.x_geo
        region_feat = torch.cat([land_feat, poi_feat, geo_feat], dim=1)
        return region_feat # (N, land_C + poi_C + 3)