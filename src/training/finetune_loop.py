"""图像输入版逐折训练循环（M2 部分微调，plan.md §8 / §9）。

与 `src/training/train_loop.py:train_one_fold` 同一套训练控制，只把输入从
（已缓存的）特征矩阵换成图像张量：

    - early stopping 只看 val AUROC（硬约束 6），patience 用尽后恢复 val 最优权重；
    - batch 顺序按 `seed + epoch` 的 CPU Generator 打乱（与 train_one_fold 约定一致）；
    - 每个 epoch 在 val 上评估一次，history 落盘供检查收敛形态。

调用约定（吸取 2026-11-04 init 未固定的教训）：**模型必须在调用本函数前构造，且构造前
已 `torch.manual_seed(seed)`** —— 本函数只负责训练，不重新 seed 整体（内部的 Generator
只控制 batch 顺序）。理由见 src/models/fer_finetune.py 与 memory/mlp-init-determinism-fix。

标准化不在本函数内：图像管线的归一化由预处理（ImageNet mean/std）完成，无 train-only
拟合统计量（硬约束 5 对图像输入不适用）。
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def train_one_fold_images(
    model: nn.Module,
    Xtr: torch.Tensor,
    ytr: np.ndarray,
    Xva: torch.Tensor,
    yva: np.ndarray,
    seed: int = 2027,
    max_epochs: int = 30,
    patience: int = 5,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 32,
    device: str = "cuda",
    amp: bool = False,
) -> tuple[dict, dict]:
    """训练一折（图像输入），返回 (val 最优权重 state_dict, 训练历史)。

    Args:
        model: 需已构造好的模型（先 seed 后构造）。若实现了 `param_groups()`
            （FinetuneFER），用它做判别式学习率；否则退化为单一 lr=lr_head 的参数组。
        Xtr/Xva: (N, 3, H, W) float32 张量，已在 CPU 上（按 index 取 batch 后再上卡）。
        amp: 是否混合精度。默认 False —— 主口径要求跨进程逐位可复现，AMP 的算子选择
            可能随显存压力变化。
    """
    device = torch.device(device)
    model = model.to(device)
    if hasattr(model, "param_groups"):
        groups = model.param_groups(lr_head=lr_head, lr_backbone=lr_backbone,
                                    weight_decay=weight_decay)
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": lr_head, "weight_decay": weight_decay}]
    opt = torch.optim.Adam(groups)
    loss_fn = nn.BCEWithLogitsLoss()

    yt = torch.as_tensor(np.asarray(ytr), dtype=torch.float32, device=device)
    yv = np.asarray(yva)
    Xv = Xva.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp) and device.type == "cuda")

    n = len(Xtr)
    best_state, best_val, best_epoch, best_loss = None, -np.inf, 0, float("nan")
    history: list[dict] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        g = torch.Generator().manual_seed(int(seed) + epoch)  # 与 train_one_fold 同约定
        perm = torch.randperm(n, generator=g)
        run_loss, seen = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = yt[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                out = model(xb)
                loss = loss_fn(out, yb)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            run_loss += float(loss.detach()) * len(idx)
            seen += len(idx)

        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(Xv).float()).cpu().numpy()
        val_auc = float(roc_auc_score(yv, p_val)) if len(np.unique(yv)) == 2 else 0.5
        history.append({"epoch": epoch, "loss": run_loss / max(seen, 1), "val_auroc": val_auc})

        if val_auc > best_val:
            best_val, best_epoch, best_loss = val_auc, epoch, history[-1]["loss"]
            best_state = copy.deepcopy({k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()})
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:  # patience=0 等极端情况
        model.load_state_dict(best_state)
        model.to(device)
    return best_state, {"best_epoch": best_epoch, "best_val_auroc": float(best_val),
                        "best_train_loss": best_loss, "n_epochs_run": len(history),
                        "history": history}


@torch.no_grad()
def logits_images(model: nn.Module, X: torch.Tensor, device: str = "cuda",
                  batch_size: int = 128) -> np.ndarray:
    """图像张量 → 原始 logits（sigmoid 之前）。分批前向，避免 val/test 一次性上卡。

    供 M4 融合用：logit 空间加权要求读到 sigmoid 之前的量，概率无法反推（数值上可逆但
    会抹掉尺度，而尺度正是融合层要处理的对象）。预测值本身仍走 predict_images。
    """
    device = torch.device(device)
    model = model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        out.append(model(xb).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


@torch.no_grad()
def predict_images(model: nn.Module, X: torch.Tensor, device: str = "cuda",
                   batch_size: int = 128) -> np.ndarray:
    """图像张量 → goal 概率。分批前向，避免 val/test 一次性上卡。"""
    device = torch.device(device)
    model = model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)
