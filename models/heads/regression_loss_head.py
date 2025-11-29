import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, Tuple


class RegressionLossHead(nn.Module):
    """
    Correct regression loss head
    - Supports per-sample prediction
    - No broadcasting
    - Handles targets correctly
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
        # Forward HRM model
        new_carry, outputs = self.model(**model_kwargs)

        # -------------------------
        # 1️⃣ 取 prediction（必须是 shape [B]）
        # -------------------------
        if "prediction" not in outputs:
            raise KeyError("RegressionLossHead expects outputs['prediction'].")

        preds = outputs["prediction"]    # could be (B,) or (B,1)
        preds = preds.to(torch.float32)

        # Ensure final shape = (B,)
        if preds.dim() == 2 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)
        elif preds.dim() != 1:
            raise RuntimeError(f"preds dim error: expected (B,) or (B,1), got {preds.shape}")

        # -------------------------
        # 2️⃣ 取 targets（必须是 [B]）
        # -------------------------
        if "targets" in new_carry.current_data:
            targets = new_carry.current_data["targets"]
        elif "labels" in new_carry.current_data:
            targets = new_carry.current_data["labels"]
        else:
            raise KeyError("No targets found in new_carry.current_data.")

        targets = targets.to(torch.float32)

        # Ensure final shape = (B,)
        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        elif targets.dim() != 1:
            raise RuntimeError(f"targets dim error: expected (B,) or (B,1), got {targets.shape}")

        # -------------------------
        # 3️⃣ 计算 Loss
        # -------------------------
        loss = F.mse_loss(preds, targets, reduction="mean")
        mae = torch.mean(torch.abs(preds - targets)).detach()

        metrics = {
            "mse": loss.detach(),
            "mae": mae,
            "count": torch.tensor(preds.size(0), dtype=torch.long),
        }

        # -------------------------
        # 4️⃣ detached 输出（防止梯度泄漏）
        # -------------------------
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        stopped_flag = (
            new_carry.halted.all()
            if hasattr(new_carry, "halted")
            else torch.tensor(True)
        )

        return new_carry, loss, metrics, detached_outputs, stopped_flag
