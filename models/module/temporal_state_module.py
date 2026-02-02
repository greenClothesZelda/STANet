import torch.nn as nn
import torch
class TemporalStateModule(nn.Module):
    def __init__(self, GRU_configs, **kwargs):
        super().__init__()
        self.hidden_size = GRU_configs['hidden_size'] #hidden_size == input_size
        self.gru = nn.GRU(
            batch_first=True,
            bidirectional=False,
            **GRU_configs
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        '''
        Docstring for forward
        
        :param x: 임베딩된 시계열 데이터 (B, N, T, Embedding_Dim)
        '''
        B, N, T, _ = x.size()
        x = x.view(-1, x.size(2), x.size(3))  # (B*N, T, Embedding_Dim)
        gru_out, _ = self.gru(x)  # (B*N, T, Hidden_Size)
        gru_out = gru_out.view(B, N, T, self.hidden_size)  # (B, N, T, Hidden_Size)
        
        gate = self.gate_layer(gru_out)  # (B, N, T, Hidden_Size)
        aggregated_state = gate * gru_out + (1 - gate) * x # (B, N, T, Hidden_Size)
        
        return aggregated_state, gru_out, gate # (B, N, T, Hidden_Size)
        