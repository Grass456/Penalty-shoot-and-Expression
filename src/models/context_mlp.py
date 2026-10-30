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


class GroupedContextMLP(nn.Module):
    """分组双隐层 MLP：情景组（4 维）与历史组（2 维）各自一个隐层再汇合。

    结构（配置驱动，见 context_v2.yaml 的 model.grouped）:
        情景组: Linear(4→h_s) → ReLU → Dropout
        历史组: Linear(2→h_h) → ReLU → Dropout
        汇合:  cat([h_s, h_h]) → Linear(·→1)

    归纳偏置：组内允许非线性交互，组间只通过汇合层线性混合——
    比全连接 MLP 更贴合"比赛情景 / 球员历史"两组信息的先验结构。

    Args:
        hidden_situ: 情景组隐层单元数（默认 2）。
        hidden_hist: 历史组隐层单元数（默认 1）。
        dropout: 两组隐层共用同一 dropout 率。
    """

    def __init__(self, hidden_situ: int = 2, hidden_hist: int = 1, dropout: float = 0.3):
        super().__init__()
        self.situ = nn.Sequential(nn.Linear(4, hidden_situ), nn.ReLU(), nn.Dropout(dropout))
        self.hist = nn.Sequential(nn.Linear(2, hidden_hist), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_situ + hidden_hist, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 列顺序 = CONTEXT_V2：前 4 列情景组，后 2 列历史组
        h = torch.cat([self.situ(x[:, :4]), self.hist(x[:, 4:])], dim=1)
        return self.head(h).squeeze(-1)