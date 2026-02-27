import torch.nn as nn
from .diff_attn import DifferentialAttention
from .standard_attn import StandardAttention

ATTN_REGISTRY = {
    "standard": StandardAttention,
    "differential": DifferentialAttention,
}


def get_attn_module(attn_name: str) -> nn.Module:
    if attn_name not in ATTN_REGISTRY:
        raise ValueError(
            f"Unknown attention module: {attn_name}. Available: {list(ATTN_REGISTRY.keys())}")
    return ATTN_REGISTRY[attn_name]
