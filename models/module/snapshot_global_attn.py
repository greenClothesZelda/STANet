import torch
import torch.nn as nn

class SnapshotGlobalAttn(nn.Module):
    '''
    지역간의 정보를 attention 메커니즘을 통해 통합하는 모듈
    '''
    def __init__(self, embedding_dim, nhead,**kwargs):
        super().__init__()
        self.nhead = nhead
        assert embedding_dim % nhead == 0, "embedding_dim must be divisible by nhead"
        self.head_dim = embedding_dim // nhead
        self.Q_fc = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K_fc = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V_fc = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.scale = torch.sqrt(torch.FloatTensor([embedding_dim]))
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=kwargs.get('layer_norm_eps', 1e-5))
    
    def forward(self, state, OD=None):
        '''
        Docstring for forward
        :param state: temporal state 모듈의 출력 aggregated_state (B, N, T, Hidden_Size)
        '''
        B, N, T, D = state.size()
        state = state.permute(0, 2, 1, 3)  # (B, T, N, D)
        Q = self.Q_fc(state).view(B, T, N, self.nhead, self.head_dim).permute(0, 1, 3, 2, 4)  # (B, T, nhead, N, head_dim)
        K = self.K_fc(state).view(B, T, N, self.nhead, self.head_dim).permute(0, 1, 3, 2, 4)  # (B, T, nhead, N, head_dim)
        V = self.V_fc(state).view(B, T, N, self.nhead, self.head_dim).permute(0, 1, 3, 2, 4)  # (B, T, nhead, N, head_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(state.device)  # (B, T, nhead, N, N)
        #TODO OD bias 추가
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T, nhead, N, N)
        attn_output = torch.matmul(attn_weights, V)  # (B, T, nhead, N, head_dim)
        attn_output = attn_output.permute(0, 1, 3, 2, 4).contiguous().view(B, T, N, D) # (B, T, N, D)
        attn_output = self.layer_norm(attn_output + state)  # (B, T, N, D)
        attn_output = attn_output.permute(0, 2, 1, 3)  # (B, N, T, D)
        return attn_output
        