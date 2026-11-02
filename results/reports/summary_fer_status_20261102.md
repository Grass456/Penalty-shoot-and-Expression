# FER 侧模型现状总结

**日期**：2026-11-02 | **数据**：40 场 / 412 events / 385 有效（goal 278 / miss 107，基率 72.2%）| **协议**：40 折 LOMO（match-disjoint），所有拟合参数只来自 train

本文件汇总 2026-10-31 至 11-02 的 FER 侧系统性排查结果。所有实验均遵守 plan.md 硬约束（触球前帧、train-only 拟合、不按 test 选模型），判读标准在实验前预注册（见 combined_arm_criteria_preregistered.txt）。

---

## 1. 总结论

**Frozen 预训练 FER 表征（单帧或多帧、任何标签空间、任何 backbone、任何正则强度）在 40 场数据上均无点球结果预测信号。** 全部 12+ 条路线经对照实验逐一排除，负面结论互证完整，可直接支撑论文 H1/H3 的反证论述。

## 2. 排除路线清单（按检验顺序）

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

## 3. 反常发现（记录在案，不构成主张）

**multi_va2 臂 fold_mean = 0.287-0.30，显著低于 0.5**（完整置换 200 次 p<0.005）。排查：非 C 伪影、非对齐 bug、非退化预测器。线索：train AUROC≈0.499（样本内无信号）但 test 系统性 0.287；场均 valence 与场 miss 率相关 +0.23（与样本级 +0.018 反向）；VA 场内 ICC 0.10-0.14。机制不明，判读为 LOMO 整场折结构与 VA 场级读数耦合的统计反常。**方向与"紧张→miss"假设相反，不可解释为信号**；扩数据后可用同协议（eval_multiframe.py）复查。

## 4. 资产清单

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

## 5. 下一步

1. **扩数据 60-80 场**（plan.md 主线）：val 规模上来后 early stopping 可靠；
2. **部分微调 M2**（plan §8 Stage 2）：frozen 全线阴性是微调实验的动机与对照基线；只解冻最后 block，encoder lr 1e-5 / head lr 1e-3；
3. 扩数据后同协议复查 multi_va2 反常（代码就绪）；
4. Neutral 方向一致性（4 backbone 下 goal 率均高于基率 0.743-0.793）可作探索性假设复查。
