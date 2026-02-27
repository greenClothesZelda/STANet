import torch
import torch.nn as nn


class SnapshotGlobalAttn(nn.Module):
    '''
    지역간의 정보를 attention 메커니즘을 통해 통합하는 모듈
    '''

    def __init__(self, embedding_dim, nhead, attn_module, attn_configs, **kwargs):
        super().__init__()
        # The attn_module is expected to handle projections and multi-head logic.
        self.attn = attn_module(d_model=embedding_dim,
                                n_heads=nhead, **attn_configs)
        self.layer_norm = nn.LayerNorm(
            embedding_dim, eps=kwargs.get('layer_norm_eps', 1e-5))
        self.output_dim = embedding_dim

    def forward(self, state, OD=None):
        '''
        Docstring for forward
        :param state: temporal state 모듈의 출력 (B, N, T, Hidden_Size)
        '''
        B, N, T, D = state.size()
        # Reshape for snapshot attention: treat each time step as a batch item.
        # (B, N, T, D) -> (B, T, N, D) -> (B*T, N, D)
        state_reshaped = state.permute(0, 2, 1, 3).reshape(B * T, N, D)

        # The injected attention module performs self-attention over the N dimension.
        # TODO: OD bias is not handled by the generic attention interface.
        attn_output = self.attn(state_reshaped)  # (B*T, N, D)

        # Residual connection and layer norm
        attn_output = self.layer_norm(
            attn_output + state_reshaped)  # (B*T, N, D)

        # Reshape back to original format
        attn_output = attn_output.view(
            B, T, N, D).permute(0, 2, 1, 3)  # (B, N, T, D)
        return attn_output
