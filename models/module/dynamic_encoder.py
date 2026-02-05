import torch.nn as nn
import torch
class DynamicEncoder(nn.Module):
    def __init__(self, time_step, embedding_dim,**kwargs):
        super().__init__()
        self.activate = kwargs.get('activate', False)
        self.dynamic_linear = nn.Linear(time_step * 3 + 1, embedding_dim)
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        self.delta_max = kwargs.get('delta_max', 24) # 최대 비활성화 기간 클리핑 값

    def forward(self, demand_series, valid_mask, deactivation_period): 
        '''
        Docstring for forward
        
        :param demand_series: 수요 시계열 데이터 (B, N, T)
        :param valid_mask: demand_series가 진짜인지 (B, N, T)
        :param deactivation_period: 비활성화 기간 (B, N, 1)
        '''
        #print(  "demand_series shape:", demand_series.shape)
        sparse_mask = (demand_series > 0).float()
        #print(f'deactivation_period shape:', deactivation_period.shape)
        deactivation_period = torch.clamp(deactivation_period, max=self.delta_max) 
        dynamic_feat = torch.cat([demand_series, valid_mask, sparse_mask, deactivation_period], dim=-1)  # (B, N, T*3 + 1) #LLM 오류탐지(Section 2.3/Step3: sparsity descriptor와 전역 recency 클리핑이 누락되어 explain과 불일치)
        #print(  "dynamic_feat shape:", dynamic_feat.shape)
        dynamic_feat = self.dynamic_linear(dynamic_feat)  # (B, N, Embedding_Dim)
        return dynamic_feat  # (B, N, Embedding_Dim)
        