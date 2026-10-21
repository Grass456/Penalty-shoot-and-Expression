# Action 3 Pilot Smoke Test 汇总报告

**日期**：2026-10-21
**数据**：14 场世界杯点球大战 pilot（可建模 114/124，goal 77 / miss 37）
**协议**：LOMO 14 折（`folds/match_disjoint.json`，test=1 场、val=下一场，确定性无种子）；所有拟合参数只来自 train（plan.md 硬约束 5/6）；指标 = AUROC / AUPRC / Balanced Acc / Brier（硬约束 7）。
**结果文件**：`results/reports/baselines_majority_context_lr_emotion_lr_fer_lr_fused_lr_imagenet_lr_20261021_142732.json`（逐折 CSV 同前缀）

pilot 结果只用于发现实现问题和估算信号强度，不作最终论文结论（plan.md §10.1）。

## 汇总（test split，均值±逐折标准差）

| 模型 | 输入 | AUROC | AUPRC | BalAcc | Brier |
|---|---|---|---|---|---|
| B0 Majority | 常数 p=train goal 率 | 0.500 | 0.670±0.143 | — | **0.225±0.053** |
| B1 Context LR | 6 维 context | 0.573±0.203 | 0.761±0.178 | 0.483±0.043 | 0.236±0.066 |
| B2 Context MLP | 同 B1 | 0.512±0.178 | 0.754±0.176 | 0.507±0.060 | 0.255±0.075 |
| B3 Emotion Prob LR | 8 维 emo_probs | 0.484±0.265 | 0.769±0.151 | 0.480±0.054 | 0.254±0.079 |
| **M1 FER LR** | 1280 维 FER embedding | **0.602±0.204** | **0.807±0.158** | 0.539±0.126 | 0.237±0.076 |
| M3 Fused LR | 6+1280=1286 维 | 0.596±0.193 | 0.805±0.155 | 0.532±0.127 | 0.238±0.075 |
| B4 ImageNet LR | 1280 维 efficientnet_b0 | 0.588±0.224 | 0.800±0.129 | 0.535±0.139 | 0.255±0.088 |

正则策略：M1/M3/B4 统一固定 C=0.01 强 L2（rationale 记录在 report JSON：val=单场 6-10 样本已被 B2 证明 early stopping 不可靠，故不做逐折 C 扫描、不引入 PCA；LR 退化为近似 ridge，确定性可复现）。

## 关键发现

1. **M1 最高（0.602），FER embedding 是目前唯一明显高于 B0 的特征臂**；AUPRC 0.807 亦最高。
2. **B4 对照：M1 (0.602) vs B4 (0.588) 只差 0.014**，远小于逐折 std（~0.2）。pilot 规模下**无法区分"FER 情绪域预训练"与"通用 ImageNet 表征"**——H1 的"额外信号"判据（M1−B4>0）在 pilot 上不成立也不被否证，需要扩数据。
3. **B3 无信号（0.484）**：8 维情绪概率不携带判别信息。注意与 M1 对照——embedding（表征）有微弱信号迹象而情绪概率（argmax 层面的解读）没有，符合"用 facial visual representation 而非真实心理状态"的定位（硬约束 9）。
4. **M3 无融合增益（0.596 < 0.602）**：强 C 均匀收缩下 context 6 维进不了 top10 系数，预期内。双通道/late fusion 结构按 plan.md §7.4 属于 M4 阶段（扩数据后 val 可靠时再上）。
5. **B2 教训**：14 折中 7 折 best_epoch=1，val=单场时神经网络的 val-early-stopping 形同虚设。smoke test 主角改为确定性 LR 系；扩数据后问题自然缓解。
6. **校准**：所有臂 Brier ≥ 0.225（B0），无一臂校准更好；LR 输出未做 Platt 调整（各臂 share 同一协议，pilot 不调校准）。

## 局限

- LOMO 单折 test 仅 6-10 样本，逐折 AUROC 噪声极大（0.25–0.89），**聚合均值是唯一可靠读数**；std 0.2 下 0.01-0.1 的臂间差异无意义。
- 114 样本 vs 1280 维：强正则下信号方向存在，但效应量估计不可靠。
- 360p 人脸（30-50px）质量上限；3 张脸 <32px 上采样。
- B0 的 Brier 0.225 = 类不平衡下的校准下界参照，所有模型的概率输出都未超越它。

## 下一步

1. **扩数据**（plan.md 下一阶段）：60-80 场，val 规模上来后 early stopping 可靠、M4（Fine-tuned FER + Context MLP，plan §7.4 late fusion）的训练前提成立。
2. 扩数据后先复跑 B1/M1/B4：若 M1−B4 的差距仍不拉开，H1 主张（FER 优于通用表征）应降级或重述为"视觉表征有信号、情绪域预训练非必要"。
3. ImageNet 臂权重加载已支持本地 .pth 缓存（`cache_fer_features.py` 的 `build_imagenet`），服务器/离线环境均可复现。
