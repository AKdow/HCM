import torch
import torch.nn as nn

class CombinedLossHead(nn.Module):
    def __init__(self, losses):
        """
        losses: list of instantiated loss objects
        """
        super().__init__()
        # 将配置的 loss 子项全部实例化
        self.losses = nn.ModuleList(losses)

    def forward(self, outputs, batch):
        total_loss = 0.0
        detail = {}

        for i, loss_fn in enumerate(self.losses):
            loss_value = loss_fn(outputs, batch)
            total_loss = total_loss + loss_value
            detail[f"loss_{i}"] = loss_value.detach()

        return total_loss, detail
