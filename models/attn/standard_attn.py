import torch
import torch.nn as nn


class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, **kwargs)

    def forward(self, x, attn_mask=None):
        # In MHA, Q, K, V are the same for self-attention
        attn_output, _ = self.mha(
            x, x, x, attn_mask=attn_mask, need_weights=False)
        return attn_output
