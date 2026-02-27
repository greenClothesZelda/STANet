import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .sta_net import STANet

class DMVSTLoss(nn.Module):
    def __init__(self, gamma=1.0, eps=1.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        term1 = diff ** 2

        relative_diff = diff / (y_true + self.eps)
        term2 = torch.abs(relative_diff)
        loss = term1 + (self.gamma * term2)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    


class STANetForTrainer(nn.Module):
    """Wrapper that adds loss computation for Hugging Face Trainer."""

    def __init__(self, stanet: STANet, lambda_mag: float = 1.0, **kwargs):
        super().__init__()
        self.stanet = stanet
        self.lambda_mag = lambda_mag
        self.magnitude_loss_fn = DMVSTLoss(gamma=kwargs.get('gamma', 1.0), eps=kwargs.get('eps', 1.0), reduction=kwargs.get('reduction', 'mean'))

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
        event_prob = outputs['event_prob']
        magnitude = outputs['magnitude']
        if labels is not None:
            labels = labels.float()
            event_target = (labels > 0).float()

            event_loss = F.binary_cross_entropy(event_prob, event_target)
            pos_mask = event_target > 0
            if pos_mask.any():
                mag_loss = self.magnitude_loss_fn(magnitude[pos_mask], labels[pos_mask])
            else:
                mag_loss = torch.tensor(0.0, device=logits.device)
            loss = (1.0 - self.lambda_mag)* event_loss + self.lambda_mag * mag_loss

        return {
            'loss': loss,
            'logits': logits,
            'event_prob': outputs['event_prob'],
            'magnitude': outputs['magnitude'],
            'prediction': outputs['prediction'],
        }
