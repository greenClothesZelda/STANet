import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .sta_net import STANet


class DMVSTLoss(nn.Module):
    def __init__(self, lambda_rel=1.0, reduction="mean"):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        abs_diff = torch.abs(diff)
        loss = abs_diff / (1.0 + y_true) + self.lambda_rel * abs_diff
        if self.reduction == "sum":
            loss = loss.sum()
        return loss


class STANetForTrainer(nn.Module):
    """Wrapper that adds loss computation for Hugging Face Trainer."""

    def __init__(self, stanet: STANet, lambda_mag: float = 1.0, **kwargs):
        super().__init__()
        self.stanet = stanet
        self.lambda_mag = lambda_mag
        lambda_rel = kwargs.get("lambda_rel", kwargs.get("gamma", 1.0))
        self.magnitude_loss_fn = DMVSTLoss(
            lambda_rel=lambda_rel,
            reduction=kwargs.get("reduction", "mean"),
        )

    def forward(
        self,
        demand_features: Dict[str, torch.Tensor],
        temporal_features: Dict[str, torch.Tensor],
        OD_matrix: Optional[torch.Tensor] = None,
        od_matrix: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if OD_matrix is None:
            OD_matrix = od_matrix
        outputs = self.stanet(
            demand_features=demand_features,
            temporal_features=temporal_features,
            OD_matrix=OD_matrix,
        )
        y_hat = outputs.get("y_hat", outputs.get("prediction"))  # (B, N)
        p_event = outputs.get("p_event", outputs.get("event_prob"))  # (B, N)
        y_hat_pos = outputs.get("y_hat_pos", outputs.get("magnitude"))  # (B, N)
        if y_hat is None or p_event is None or y_hat_pos is None:
            raise KeyError("Model outputs must include y_hat/p_event/y_hat_pos or their legacy aliases.")

        loss = None
        if labels is not None:
            labels = labels.float()
            event_target = (labels > 0).float()
            event_loss = F.binary_cross_entropy(
                p_event, event_target, reduction="none").sum(dim=-1).mean()
            mag_term = self.magnitude_loss_fn(y_hat_pos, labels)
            mag_loss = (event_target * mag_term).sum(dim=-1).mean()
            loss = event_loss + mag_loss

        return {
            "loss": loss,
            "logits": y_hat,
            "event_prob": p_event,
            "magnitude": y_hat_pos,
            "prediction": y_hat,
            "p_event": p_event,
            "y_hat_pos": y_hat_pos,
            "y_hat": y_hat,
        }
