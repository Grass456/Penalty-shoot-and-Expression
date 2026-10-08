"""逐折训练循环，供 Context MLP（B2）与后续 FER 分支/M3 融合共用（plan.md §8）。

所有模型必须经由此处的同一协议训练，保证 folds/early stopping/指标口径一致：
    - 逐折从 folds/match_disjoint.json 取 train/val/test 的 event_ids；
    - early stopping 只看 val AUROC（硬约束 6），patron 用尽后恢复 val 最优权重；
    - 每折返回 val/test 预测概率，指标计算与落盘由调用方（run_baselines 风格）完成。
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def train_one_fold(
    model: nn.Module,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    seed: int = 2026,
    max_epochs: int = 200,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int | None = None,  # None = 全批量（pilot 每折 train 仅 ~96 行）
    device: str = "cpu",
) -> tuple[dict, dict]:
    """训练一折，返回 (最优权重 state_dict, 训练历史)。

    X* 应已按 train 统计量标准化（由调用方保证，硬约束 5）。
    """
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)

    best_state, best_val, best_epoch = None, -np.inf, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        if batch_size is None:
            opt.zero_grad()
            out = model(Xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()
        else:
            g = torch.Generator().manual_seed(seed + epoch)
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(Xv)).cpu().numpy()
        val_auc = roc_auc_score(yva, p_val) if len(np.unique(yva)) == 2 else 0.5
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_auroc": float(val_auc)})

        if val_auc > best_val:
            best_val, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:  # patience=0 等极端情况下可能无记录
        model.load_state_dict(best_state)
    return best_state, {"best_epoch": best_epoch, "best_val_auroc": float(best_val), "history": history}


def predict(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """标准化后的 X → goal 概率。"""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32, device=device))
    return torch.sigmoid(out).cpu().numpy()