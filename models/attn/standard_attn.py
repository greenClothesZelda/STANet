import torch
import torch.nn as nn


class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, **kwargs)

    def _expand_bias_for_mha(self, attn_bias):
        # MultiheadAttention expects 3D attn_mask of shape:
        # (batch_size * num_heads, target_len, source_len)
        if attn_bias.dim() == 3:
            bsz, tgt_len, src_len = attn_bias.shape
            return attn_bias.unsqueeze(1).expand(
                bsz, self.mha.num_heads, tgt_len, src_len
            ).reshape(bsz * self.mha.num_heads, tgt_len, src_len)
        if attn_bias.dim() == 4:
            bsz, n_heads, tgt_len, src_len = attn_bias.shape
            if n_heads != self.mha.num_heads:
                raise ValueError(
                    f"attn_bias head dim mismatch: expected {self.mha.num_heads}, got {n_heads}."
                )
            return attn_bias.reshape(bsz * n_heads, tgt_len, src_len)
        raise ValueError(f"attn_bias must be 3D or 4D, got shape {tuple(attn_bias.shape)}.")

    def forward(self, x, attn_mask=None, attn_bias=None):
        mask = attn_mask
        if attn_bias is not None:
            bias_mask = self._expand_bias_for_mha(attn_bias).to(dtype=x.dtype, device=x.device)
            if mask is None:
                mask = bias_mask
            else:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).expand(bias_mask.size(0), -1, -1)
                if mask.dtype == torch.bool:
                    bool_mask = torch.zeros_like(mask, dtype=x.dtype)
                    bool_mask = bool_mask.masked_fill(mask, float("-inf"))
                    mask = bool_mask
                mask = mask.to(dtype=x.dtype, device=x.device) + bias_mask

        attn_output, _ = self.mha(
            x, x, x, attn_mask=mask, need_weights=False)
        return attn_output
