from .losses import ACTLossHead
from .mse_loss_head import MSELossHead
from .combined_loss_head import CombinedLossHead

IGNORE_LABEL_ID = -100

__all__ = [
    "ACTLossHead",
    "MSELossHead",
    "CombinedLossHead",
    "IGNORE_LABEL_ID",
]
