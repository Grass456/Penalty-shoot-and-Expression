# 归档：40 折 LOMO 时期结果（2026-10-08 至 11-02，已作废）

这些结果全部基于旧划分 `folds/match_disjoint_lomo40.json`（40 折 leave-one-match-out，val=下一场；14 场 pilot 时期为 14 折）。2026-11-04 划分换成 5 折 StratifiedGroupKFold，所有实验已重跑（见 summary_5fold_20261104.md），**本文件里的数字不再作为结论引用**。

原始逐折表与 JSON 报告已从工作区删除。需要时可从 git 提交 `3ed0a7b`（New fold distribution，最后一个仍含这些文件的提交）检出 `results/` 目录，用 `git show 3ed0a7b:results/reports/<文件名>` 或 `git checkout 3ed0a7b -- results/`。

保留它的理由（可能写进论文的方法论部分）：
- LOMO 下 fold_mean 的置换零分布约 0.50±0.04、p95≈0.567，单折 AUROC 不能直接和 0.5 比较；
- multi_va2 在 LOMO 下 0.287（置换 p<0.005），换 5 折后回到 0.47：整场为 test 的折结构与场级特征耦合会制造假结果；
- B1v2 在 LOMO 上 p=0.046，换划分后不显著：特征是看过 test 后设计的。

## A. 40 场 LOMO 主臂（test，fold_mean）

| 臂 | fold_mean | 备注 |
|---|---:|---|
| B0 majority | 0.500 | |
| B1v1 context LR | 0.315 | 方向反 |
| B2v1 context MLP | 0.454 | |
| B1v2 context LR | 0.577±0.251 | pooled_rank 0.557，置换 p=0.046 |
| B1v2-int | 0.583 | |
| B2v2 MLP / grouped | 0.493 / 0.527 | |
| B3 emotion vgaf | 0.509 | b2_8 0.594 / b2_7 0.567 / afew 0.576 |
| M1 FER emb | 0.475 | pilot 0.602 未复现 |
| B4 ImageNet | 0.471 | |
| M3 / M3v2 fused | 0.469 / 0.468 | |

---

## B. Action 3 Pilot Smoke Test 汇总报告

**日期**：2026-10-21
**数据**：14 场世界杯点球大战 pilot（可建模 114/124，goal 77 / miss 37）
**协议**：LOMO 14 折（`folds/match_disjoint.json`，test=1 场、val=下一场，确定性无种子）；所有拟合参数只来自 train（plan.md 硬约束 5/6）；指标 = AUROC / AUPRC / Balanced Acc / Brier（硬约束 7）。
**结果文件**：`results/reports/baselines_majority_context_lr_emotion_lr_fer_lr_fused_lr_imagenet_lr_20261021_142732.json`（逐折 CSV 同前缀）

pilot 结果只用于发现实现问题和估算信号强度，不作最终论文结论（plan.md §10.1）。

### 汇总（test split，均值±逐折标准差）

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

### 关键发现

1. **M1 最高（0.602），FER embedding 是目前唯一明显高于 B0 的特征臂**；AUPRC 0.807 亦最高。
2. **B4 对照：M1 (0.602) vs B4 (0.588) 只差 0.014**，远小于逐折 std（~0.2）。pilot 规模下**无法区分"FER 情绪域预训练"与"通用 ImageNet 表征"**——H1 的"额外信号"判据（M1−B4>0）在 pilot 上不成立也不被否证，需要扩数据。
3. **B3 无信号（0.484）**：8 维情绪概率不携带判别信息。注意与 M1 对照——embedding（表征）有微弱信号迹象而情绪概率（argmax 层面的解读）没有，符合"用 facial visual representation 而非真实心理状态"的定位（硬约束 9）。
4. **M3 无融合增益（0.596 < 0.602）**：强 C 均匀收缩下 context 6 维进不了 top10 系数，预期内。双通道/late fusion 结构按 plan.md §7.4 属于 M4 阶段（扩数据后 val 可靠时再上）。
5. **B2 教训**：14 折中 7 折 best_epoch=1，val=单场时神经网络的 val-early-stopping 形同虚设。smoke test 主角改为确定性 LR 系；扩数据后问题自然缓解。
6. **校准**：所有臂 Brier ≥ 0.225（B0），无一臂校准更好；LR 输出未做 Platt 调整（各臂 share 同一协议，pilot 不调校准）。

### 局限

- LOMO 单折 test 仅 6-10 样本，逐折 AUROC 噪声极大（0.25–0.89），**聚合均值是唯一可靠读数**；std 0.2 下 0.01-0.1 的臂间差异无意义。
- 114 样本 vs 1280 维：强正则下信号方向存在，但效应量估计不可靠。
- 360p 人脸（30-50px）质量上限；3 张脸 <32px 上采样。
- B0 的 Brier 0.225 = 类不平衡下的校准下界参照，所有模型的概率输出都未超越它。

### 下一步

1. **扩数据**（plan.md 下一阶段）：60-80 场，val 规模上来后 early stopping 可靠、M4（Fine-tuned FER + Context MLP，plan §7.4 late fusion）的训练前提成立。
2. 扩数据后先复跑 B1/M1/B4：若 M1−B4 的差距仍不拉开，H1 主张（FER 优于通用表征）应降级或重述为"视觉表征有信号、情绪域预训练非必要"。
3. ImageNet 臂权重加载已支持本地 .pth 缓存（`cache_fer_features.py` 的 `build_imagenet`），服务器/离线环境均可复现。

---

## C. FER 侧模型现状总结

**日期**：2026-11-02 | **数据**：40 场 / 412 events / 385 有效（goal 278 / miss 107，基率 72.2%）| **协议**：40 折 LOMO（match-disjoint），所有拟合参数只来自 train

本文件汇总 2026-10-31 至 11-02 的 FER 侧系统性排查结果。所有实验均遵守 plan.md 硬约束（触球前帧、train-only 拟合、不按 test 选模型），判读标准在实验前预注册（见 combined_arm_criteria_preregistered.txt）。

---

### 1. 总结论

**Frozen 预训练 FER 表征（单帧或多帧、任何标签空间、任何 backbone、任何正则强度）在 40 场数据上均无点球结果预测信号。** 全部 12+ 条路线经对照实验逐一排除，负面结论互证完整，可直接支撑论文 H1/H3 的反证论述。

### 2. 排除路线清单（按检验顺序）

| # | 路线 | 检验方法 | 结果（test fold_mean / pooled_rank） | 判定 |
|---|---|---|---|---|
| 1 | C 正则强度 | sweep_fer_c.py：C∈[3e-5,1] 12 点 × M1/B4 | 全程平坦 0.46-0.50，无 U 型 | C=0.01 维持，非正则问题 |
| 2 | Backbone 选择 | compare_fer_backbones.py：4 FER + ImageNet，3 个 C | embedding 臂全 0.42-0.53 无方向 | 换 backbone 无效 |
| 3 | 情绪概率头（B3） | probe_emo_head_signal.py：真实头 vs 随机投影 vs 错配头 | b2 系 0.57-0.61 但错配头 0.586 几乎不降 | 与情绪语义无关 |
| 4 | 降维瓶颈效应 | bottleneck_dim_sweep.py：7 维度 × 60 随机投影 | 随机 8 维投影 pooled_rank 0.507±0.038 | B3 的 0.6 判为降维伪影 |
| 5 | 融合（M3/M3v2） | run_baselines + sweep_fused_c.py：8 个 C 判别实验 | Δ(fused−context) 在所有 C 上均为负（−0.046~−0.098） | H3 干净阴性：融合原始 embedding 纯添噪 |
| 6 | 假设 1：8 类空间错配 | eval_va_arms.py：3 个 VA-MTL backbone，三预言 | P1 方向无强度、P2 配对差 +0.03 噪声内、P3 VA 头正常（与 8 类相关 +0.7） | 标签空间错配版本否定 |
| 7 | 方向 2：单帧测量噪声 | cache_multiframe_features.py + eval_multiframe.py：多帧聚合 3 模式 | multi − single 配对差 +0.01~+0.04（20-24/40 折） | 方向一致但噪声内，无叠加 |
| 8 | 合并（多帧 × VA） | eval_multiframe.py 合并臂（预注册标准） | multi_va_emb 最高 0.511，全部低于判定线 0.567 | 合并无叠加效应 |

**方法论收获**（对所有后续实验有效）：
- fold_mean 的零假设不是 0.5（置换零分布 0.50±0.04，p95≈0.567）；推荐口径为 pooled_rank（折内转秩后合并，evaluate_oof_stats.py）。
- 高维 embedding（1280+）+ n≈370 时，任何降到 k≤32 的投影都会产生 ~0.51-0.52 的 fold_mean 假阳性基线；评估必须与同维随机投影分布比较。

### 3. 反常发现（记录在案，不构成主张）

**multi_va2 臂 fold_mean = 0.287-0.30，显著低于 0.5**（完整置换 200 次 p<0.005）。排查：非 C 伪影、非对齐 bug、非退化预测器。线索：train AUROC≈0.499（样本内无信号）但 test 系统性 0.287；场均 valence 与场 miss 率相关 +0.23（与样本级 +0.018 反向）；VA 场内 ICC 0.10-0.14。机制不明，判读为 LOMO 整场折结构与 VA 场级读数耦合的统计反常。**方向与"紧张→miss"假设相反，不可解释为信号**；扩数据后可用同协议（eval_multiframe.py）复查。

### 4. 资产清单

### 特征缓存（features/fer/，gitignored）

| 缓存 | 规模 | 说明 |
|---|---|---|
| enet_b0_8_best_vgaf / _afew | 385×1280 | 8 类 B0（视频微调 VGAF/AFEW），单帧 |
| enet_b2_8 / enet_b2_7 | 385×1408 | 8/7 类 B2（AffectNet 静态），单帧 |
| enet_b0_8_va_mtl / mbf_va_mtl / mobilevit_va_mtl | 385×{1280,512,768} +VA | VA-MTL（valence/arousal 连续头），单帧 |
| imagenet_b0 | 385×1280 | H3 对照臂 |
| multiframe_{mean,median,last_k_k8}_vgaf | 356×1280 | 多帧聚合（vgaf） |
| multiframe_mean_va_mtl | 356×1280 +VA | 多帧聚合（VA，含 va 列均值） |

多帧提取参数（冻结）：窗口 [face_closeup_s, contact_time_s−0.5s]、2fps、YuNet conf≥0.6 + 脸宽≥48px、SFace 身份锚验证（阈值 0.363）。

### 脚本（scripts/）

cache_fer_features.py（单帧提取，8 backbone 通用）、cache_va_features.py（VA 单帧）、cache_multiframe_features.py（多帧提取 + VA 扩展）、compare_fer_backbones.py、eval_va_arms.py、eval_multiframe.py、sweep_fer_c.py、sweep_fused_c.py、bottleneck_dim_sweep.py、probe_emo_head_signal.py、evaluate_oof_stats.py（统计口径诊断）。

### 报告（results/reports/，可入库）

sweep_fer_c、backbone_compare、emo_head_probe、oof_stats、bottleneck_sweep、sweep_fused_c、va_arms、multiframe_eval（×4）、combined_arm_criteria_preregistered。历史：summary_action3_pilot.md（14 场 pilot，其 0.602 已被 40 场证伪）。

### 已知数据局限（写论文时须披露）

- 360p 转播，人脸 crop 中位 119px 上采样；
- SFace 官方同人阈值 0.363 在本数据上不可用（场内同人对相似度中位 0.074），多帧身份验证实际起"近景 vs 远景"的镜头质量过滤作用；
- 29/385 踢次无球员帧（空踢次 goal 率 0.759 vs 保留集 0.719，选择偏差不大）；
- 8 类情绪 argmax 高度集中（Anger 55%），是模型对紧绷表情的先验偏置。

### 5. 下一步

1. **扩数据 60-80 场**（plan.md 主线）：val 规模上来后 early stopping 可靠；
2. **部分微调 M2**（plan §8 Stage 2）：frozen 全线阴性是微调实验的动机与对照基线；只解冻最后 block，encoder lr 1e-5 / head lr 1e-3；
3. 扩数据后同协议复查 multi_va2 反常（代码就绪）；
4. Neutral 方向一致性（4 backbone 下 goal 率均高于基率 0.743-0.793）可作探索性假设复查。

---

## D. 当时的预注册标准（combined_arm_criteria_preregistered.txt 原文）

```
# 合并臂判读标准（2026-11-02 预注册，跑实验前写定）
#
# 「明显效果」的判定（两者须同时满足）：
#   1. fold_mean > 0.567（40 折置换零分布 p95，来源 oof_stats_20261031_132848.json）
#   2. pooled_rank > 0.55
#
# 「方向性提示」（不构成发现，只作扩数据复查依据）：
#   fold_mean 在 0.52-0.567 且 multi_VA2 > multi_vgaf-emb（紧张假设的方向特异性）
#
# 预期（基于已有证据）：三臂全部 ≈0.5，合并无叠加空间（同一 encoder 的信号源）。
# 若 multi_VA2 抬头而 multi_vgaf-emb 不抬头 → 时序平均修好了 VA 读数噪声，
# 这是唯一无法从现有数据排除的情形。
```
