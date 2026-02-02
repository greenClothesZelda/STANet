import torch.nn as nn
import torch
class DynamicEncoder(nn.Module):
    def __init__(self, time_step, embedding_dim,**kwargs):
        super().__init__()
        self.activate = kwargs.get('activate', False)
        self.dynamic_linear = nn.Linear(time_step * 3 + 1, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, demand_series, vaild_mask, deactivation_period): 
        '''
        Docstring for forward
        
        :param demand_series: 수요 시계열 데이터 (B, N, T)
        :param vaild_mask: demand_series가 진짜인지 (B, N, T)
        :param deactivation_period: 비활성화 기간 (B, N, 1)
        '''
        if not self.activate:
            return torch.zeros_like(demand_series.shape[0], demand_series.shape[1], self.embedding_dim)
        sparse_mask = (demand_series > 0).float()
        dynamic_feat = torch.cat([demand_series, vaild_mask, sparse_mask, deactivation_period], dim=-1)  # (B, N, T*3 + 1)
        dynamic_feat = self.dynamic_linear(dynamic_feat)  # (B, N, Embedding_Dim)
        return dynamic_feat  # (B, N, Embedding_Dim)
        