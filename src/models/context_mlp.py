"""B2 Context MLP（plan.md §7.3、§9）。

低容量设计：1 个隐藏层（默认 16 单元）+ dropout + weight decay，
输入为 run_baselines.build_context_matrix 的同一 6 维 context 向量。

训练控制（plan.md §8）：
    - early stopping 只看 val AUROC（硬约束 6，禁止用 test 选模型）；
    - 标准化器（均值/方差）只来自 train（硬约束 5）；
    - 固定种子，逐折独立训练。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextMLP(nn.Module):
    """6 维 context → 隐藏层（ReLU+dropout）→ 单 logit。

    Args:
        in_dim: 输入维度（CONTEXT_NUMERIC 长度，默认 6）。
        hidden: 隐藏单元数。保持低容量防小数据过拟合（plan §7.3）。
        dropout: 隐藏层 dropout 率。
    """

    def __init__(self, in_dim: int = 6, hidden: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)