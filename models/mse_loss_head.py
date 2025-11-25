import torch
import torch.nn as nn

class MSELossHead(nn.Module):
    def __init__(self, key="prediction", target_key="targets", reduction="mean"):
        super().__init__()
        self.key = key
        self.target_key = target_key
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, outputs, batch):
        pred = outputs[self.key]              # [B]
        target = batch[self.target_key]       # [B]
        return self.criterion(pred, target)
