import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .sta_net import STANet


class STANetForTrainer(nn.Module):
    """Wrapper that adds loss computation for Hugging Face Trainer."""

    def __init__(self, stanet: STANet, lambda_mag: float = 1.0):
        super().__init__()
        self.stanet = stanet
        self.lambda_mag = lambda_mag

    def forward(
        self,
        demand_features: Dict[str, torch.Tensor],
        temporal_features: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.stanet(demand_features=demand_features,
                              temporal_features=temporal_features)
        logits = outputs['prediction']  # (B, N)

        loss = None
        if labels is not None:
            labels = labels.float()
            event_target = (labels > 0).float()
            event_prob = outputs['event_prob']
            magnitude = outputs['magnitude']

            event_loss = F.binary_cross_entropy(event_prob, event_target)
            pos_mask = event_target > 0
            if pos_mask.any():
                mag_loss = F.l1_loss(
                    magnitude[pos_mask], labels[pos_mask], reduction='mean')
            else:
                mag_loss = torch.tensor(0.0, device=logits.device)
            loss = event_loss + self.lambda_mag * mag_loss

        return {
            'loss': loss,
            'logits': logits,
            'event_prob': outputs['event_prob'],
            'magnitude': outputs['magnitude'],
            'prediction': outputs['prediction'],
        }
