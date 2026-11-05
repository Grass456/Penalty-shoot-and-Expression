"""M2 部分微调 FER 分支（plan.md §9 M2 / §12 A3；判读线 H4 = M2 vs M1）。

预注册：results/reports/m2_finetune_criteria_preregistered.txt（2026-11-05 跑前写定）。

结构：
    - backbone = EmotiEffLib 的 timm EfficientNet（classifier 已 Identity 化，直接输出 (N, D) embedding）；
    - 解冻范围由 `unfreeze_depth` 控制：
        1 = 最后一个 stage（blocks[-1]，torchvision 记法的 layer4，717,232 参数）+ 新建线性头；
        0 = 只训线性头（backbone 全冻结，作为退化检查，信息性不判定）；
    - BN 统计量默认冻结（training 时层内 BN 走 eval 分支）：n=385、batch 32 下用 ImageNet 预训练
      统计量比用每折 ~250 个样本重估更可信。`bn_train=True` 是预留的鲁棒性变体，主口径不用。
    - 前向把冻结前缀（conv_stem/bn1/blocks[:-d]）包在 no_grad 内：省显存与时间，数值上与
      整网带梯度前向一致（冻结参数不需要梯度，BN 又走 eval 分支）。前缀输出必须 detach，
      否则会跨 batch 保留计算图。

调用约定（吸取 2026-11-04 init 未固定的教训，见 memory/mlp-init-determinism-fix）：
    **构造本模型前必须先 torch.manual_seed(seed)** —— 头部 Linear 的 init 取自全局 RNG，
    不先 seed 则同 seed 跨进程不可复现。backbone 权重本身来自预训练 ckpt，不受影响。

用法：
    from cache_fer_features import build_fer
    from src.models.fer_finetune import FinetuneFER

    backbone, _, emb_dim, _, _ = build_fer("enet_b0_8_best_vgaf", "cuda")
    pristine = copy.deepcopy(backbone)          # 每折从 pristine 拷贝，禁止跨折复用被改过的副本
    torch.manual_seed(seed)
    model = FinetuneFER(pristine, emb_dim=emb_dim, unfreeze_depth=1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FinetuneFER(nn.Module):
    """FER backbone + 线性头；只解冻最后 unfreeze_depth 个 stage。

    Args:
        backbone: EmotiEffLib 的 timm EfficientNet（forward 输出 (N, D) embedding）。
            调用方负责 deepcopy —— 本模块会就地修改 requires_grad 并训练其中的参数。
        emb_dim: embedding 维度（enet_b0_* 为 1280）。
        unfreeze_depth: 解冻最后几个 stage；0 = 只训头。
        bn_train: True 时允许解冻层内的 BN 统计量随 batch 更新（主口径 False）。
        head: 可选自定义头；默认 nn.Linear(emb_dim, 1)。
    """

    def __init__(self, backbone: nn.Module, emb_dim: int = 1280,
                 unfreeze_depth: int = 1, bn_train: bool = False,
                 head: nn.Module | None = None):
        super().__init__()
        stages = list(backbone.blocks)
        if not 0 <= unfreeze_depth <= len(stages):
            raise ValueError(f"unfreeze_depth 越界: {unfreeze_depth} (stages={len(stages)})")
        self.backbone = backbone
        self.unfreeze_depth = int(unfreeze_depth)
        self.bn_train = bool(bn_train)
        self.head = nn.Linear(emb_dim, 1) if head is None else head
        self._configure_requires_grad()
        self._bn_eval()  # 初始即冻结 BN 统计量（主口径）

    # ------------------------------------------------------------ 配置

    def _configure_requires_grad(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        stages = list(self.backbone.blocks)
        for stage in stages[len(stages) - self.unfreeze_depth:]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def _bn_eval(self) -> None:
        """把 backbone 内所有 BN 切到 eval（只影响统计量更新，不影响 requires_grad）。"""
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def train(self, mode: bool = True) -> "FinetuneFER":
        super().train(mode)
        if mode and not self.bn_train:
            self._bn_eval()
        return self

    # ------------------------------------------------------------ 分组

    def param_groups(self, lr_head: float = 1e-3, lr_backbone: float = 1e-4,
                     weight_decay: float = 1e-2) -> list[dict]:
        """判别式学习率的 Adam 参数组（头用大 lr，解冻层用小 lr）。"""
        head_p = [p for p in self.head.parameters() if p.requires_grad]
        bb_p = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = []
        if bb_p:
            groups.append({"params": bb_p, "lr": lr_backbone, "weight_decay": weight_decay})
        groups.append({"params": head_p, "lr": lr_head, "weight_decay": weight_decay})
        return groups

    def n_trainable(self) -> tuple[int, int]:
        """返回 (可训练参数量, 总参数量)。"""
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        return tr, tot

    # ------------------------------------------------------------ 前向

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) → (N,) logits。"""
        bb = self.backbone
        stages = list(bb.blocks)
        k = len(stages) - self.unfreeze_depth
        if self.unfreeze_depth == 0:
            with torch.no_grad():
                z = bb(x)
            return self.head(z).squeeze(-1)
        with torch.no_grad():
            z = bb.bn1(bb.conv_stem(x))
            for stage in stages[:k]:
                z = stage(z)
        z = z.detach()  # 冻结前缀输出当常量，避免跨 batch 保留计算图
        for stage in stages[k:]:
            z = stage(z)
        z = bb.bn2(bb.conv_head(z))
        z = bb.forward_head(z)  # global_pool (+ Identity classifier)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """整网冻结前向，返回 (N, D) embedding；仅用于图像管线一致性检查。"""
        return self.backbone(x)
