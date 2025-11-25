import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, Tuple


class RegressionLossHead(nn.Module):
    """
    Loss head for regression tasks in HRM.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ):
        new_carry, outputs = self.model(**model_kwargs)

        if "prediction" not in outputs:
            raise KeyError("RegressionLossHead expects outputs['prediction'].")

        preds = outputs["prediction"].to(torch.float32)

        if "targets" in new_carry.current_data:
            targets = new_carry.current_data["targets"]
        elif "labels" in new_carry.current_data:
            targets = new_carry.current_data["labels"]
        else:
            raise KeyError("No targets found in new_carry.current_data.")

        targets = targets.to(torch.float32)

        loss = F.mse_loss(preds, targets, reduction="mean")

        mae = torch.mean(torch.abs(preds - targets)).detach()

        metrics = {
            "mse": loss.detach(),
            "mae": mae,
            "count": torch.tensor(preds.shape[0], dtype=torch.long),
        }

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        stopped_flag = new_carry.halted.all() if hasattr(new_carry, "halted") else torch.tensor(True)

        return new_carry, loss, metrics, detached_outputs, stopped_flag
