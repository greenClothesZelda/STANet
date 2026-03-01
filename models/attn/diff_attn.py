import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DifferentialAttention(nn.Module):
    def __init__(self, d_model, n_heads, lambda_init=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K는 split을 위해 d_model * 2의 출력을 가짐
        self.q_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.k_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.v_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.o_proj = nn.Linear(d_model * 2, d_model, bias=False)

        # 학습 가능한 lambda 파라미터 (논문에 따라 초기값 설정)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_init = lambda_init

    def forward(self, x, attn_mask=None, attn_bias=None):
        b, n, _ = x.shape
        h = self.n_heads
        d = self.d_head  # d in pseudocode (Q1, Q2 dimension)

        # 1. Projections & Split (Q1, Q2, K1, K2)
        # q, k: [b, n, 2 * d_model] -> [b, n, 2, h, d]
        q = self.q_proj(x).view(b, n, 2, h, d)
        k = self.k_proj(x).view(b, n, 2, h, d)
        v = self.v_proj(x).view(b, n, h, 2 * d)  # V는 2d 차원

        q1, q2 = q[:, :, 0], q[:, :, 1]  # [b, n, h, d]
        k1, k2 = k[:, :, 0], k[:, :, 1]  # [b, n, h, d]

        # Transpose for attention: [b, h, n, d]
        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v = v.transpose(1, 2)  # [b, h, n, 2d]

        # 2. Attention Scores (A1, A2)
        s = 1.0 / math.sqrt(d)
        attn1 = (q1 @ k1.transpose(-1, -2)) * s
        attn2 = (q2 @ k2.transpose(-1, -2)) * s

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                # [b, n, n] -> [b, 1, n, n]
                attn_bias = attn_bias.unsqueeze(1)
            elif attn_bias.dim() == 4:
                # already [b, h|1, n, n]
                pass
            else:
                raise ValueError(f"attn_bias must be 3D or 4D, got shape {tuple(attn_bias.shape)}.")
            attn_bias = attn_bias.to(dtype=attn1.dtype, device=attn1.device)
            attn1 = attn1 + attn_bias
            attn2 = attn2 + attn_bias

        if attn_mask is not None:
            # attn_mask should be broadcastable to [b, h, n, n]
            attn1 = attn1.masked_fill(attn_mask == 0, -float('inf'))
            attn2 = attn2.masked_fill(attn_mask == 0, -float('inf'))

        # 3. Differential Attention calculation
        # softmax(A1) - lambda * softmax(A2)
        diff_attn = F.softmax(attn1, dim=-1) - \
            self.lambda_param * F.softmax(attn2, dim=-1)

        # 4. Multi-head output
        out = diff_attn @ v  # [b, h, n, 2d]

        # 5. Reshape & Scaling
        # [b, h, n, 2d] -> [b, n, h, 2d] -> [b, n, 2 * d_model]
        out = out.transpose(1, 2).reshape(b, n, -1)

        # Scaling by (1 - lambda_init)
        out = out * (1 - self.lambda_init)

        # 6. Final Output Projection
        return self.o_proj(out)
