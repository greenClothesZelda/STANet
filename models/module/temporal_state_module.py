import torch.nn as nn
import torch
class TemporalStateModule(nn.Module):
    def __init__(self, GRU_configs, input_size, **kwargs):
        super().__init__()
        self.hidden_size = input_size #hidden_size == input_size
        self.gru = nn.GRU(
            batch_first=True,
            bidirectional=False,
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            **GRU_configs
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        '''
        Docstring for forward
        
        :param x: 임베딩된 시계열 데이터 (B, N, T, Embedding_Dim)
        '''
        B, N, T, _ = x.size()
        x_origin = x
        x = x.view(-1, x.size(2), x.size(3))  # (B*N, T, Embedding_Dim)
        gru_out, _ = self.gru(x)  # (B*N, T, Hidden_Size)
        gru_out = gru_out.view(B, N, T, self.hidden_size)  # (B, N, T, Hidden_Size)
        #print(  "gru_out shape:", gru_out.shape)
        #print(  "x shape:", x.shape)
        gate_factor = torch.cat([x_origin, gru_out], dim=-1)  # (B, N, T, Embedding_Dim*2)
        gate = self.gate_layer(gate_factor)  # (B, N, T, Hidden_Size)
        aggregated_state = gate * gru_out + (1 - gate) * x_origin # (B, N, T, Hidden_Size) #LLM 오류탐지(Step 2: gate가 e||h 대신 gru_out만 사용해 설명의 gated fusion과 다름)
        #print(  "aggregated_state shape:", aggregated_state.shape)
        return aggregated_state, gru_out, gate # (B, N, T, Hidden_Size)
        