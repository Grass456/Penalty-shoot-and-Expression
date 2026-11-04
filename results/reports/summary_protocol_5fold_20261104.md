# 划分协议更换：40 折 LOMO → 5 折 StratifiedGroupKFold（全部实验重跑）

**日期**：2026-11-04 | **数据**：40 场 / 385 有效（goal 278 / miss 107）| **folds**：`folds/match_disjoint.json`（schema v2，seed=2026）

## 协议

- 外层 StratifiedGroupKFold(5, shuffle, seed=2026)，group=`match_id`，按 result 分层：test 每折 7-9 场 / 53-98 踢 / 14-25 个 miss。
- 内层：其余比赛再跑一次 StratifiedGroupKFold(5, 同 seed) 取第一折作 val，6-7 场 / 50-76 踢 / 12-21 个 miss。
- train 24-27 场 / 211-281 踢。
- 旧 40 折 LOMO 存档在 `folds/match_disjoint_lomo40.json`。11-04 前的所有报告都用旧 folds（报告里 folds.sha256_16 = `1d4a2239a85cfbd9`）。
- 只跑了一组划分、单种子。5 个 test 点方差大，见末节。

## 主臂 test 对比（旧 LOMO fold_mean → 新 5 折 fold_mean）

| 臂 | 旧 | 新 | 新 pooled_rank（置换 p） |
|---|---:|---:|---:|
| B0 majority | 0.500 | 0.500 | — |
| B1v1 context LR | 0.315 | 0.448 | — |
| B2v1 context MLP | 0.454 | 0.482 | — |
| **B1v2 context LR（主臂）** | **0.577** | **0.516** | 0.519（p=0.246） |
| B1v2-int | 0.583 | 0.522 | — |
| B2v2 context MLP | 0.493 | 0.553 | —（单种子） |
| B2v2-grouped | 0.527 | 0.471 | — |
| B3 emotion (vgaf) | 0.509 | 0.502 | 0.490（p=0.617） |
| B3 emotion (b2_7) | 0.567 | 0.534 | 0.529（p=0.189） |
| B3 emotion (b2_8) | 0.594 | 0.510 | 0.502（p=0.482） |
| M1 FER emb (vgaf) | 0.475 | 0.477 | 0.485（p=0.682） |
| B4 ImageNet emb | 0.471 | 0.502 | 0.494（p=0.569） |
| M3 fused v1 | 0.469 | 0.478 | — |
| M3v2 fused v2 | 0.468 | 0.478 | 0.485（p=0.670） |

## 结论变化

1. **B1v2 的边缘显著消失**：旧 pooled_rank 0.557（p=0.046）→ 0.519（p=0.246）。v2 特征是看过旧 test 结果后设计的，换划分后不再超出零分布。context 侧目前没有可以主张的信号。
2. **B3 情绪臂 0.57-0.61 回落到 0.50-0.53**，随机投影的 pooled_rank 也降到约 0.50-0.52，与"降维伪影"判读一致。情绪头 vs 随机投影：所有 backbone 的 p≥0.64（emo_head_probe_20261104_134626）。
3. **multi_va2 反常消失**：旧 0.287（置换 p<0.005）→ 新 0.47。这支持旧判读：反常来自 LOMO 整场折结构与场级 VA 的耦合，不是信号。
4. **FER embedding、ImageNet、融合仍在 0.48-0.50**，H1/H3 阴性结论不变。C 扫描（3e-5 到 1.0）全程平坦，在 0.474-0.493 之间。
5. **sweep_fused_c**：小 C 下 Δ(fused−context) 变成正值（+0.019），但这是因为 context-only 被强 L2 压到 0.465，fused 本身从未超过 0.491。H3 判读不变。
6. **需要留意、不构成主张**：
   - VA-MTL emb1280 pooled_rank 0.538，MTL−8cls 配对差 +0.061，5/5 折为正。
   - B2v2 MLP 0.553。
   - 多帧 multi_mean C=1.0 0.526，multi−single +0.068，4/5 折为正。

   以上都只有 5 个 test 点、单划分，也没有置换零分布。要先用重复划分复查，才能判读。

## 下一步

- 重复划分：至少 5 组 seed × 5 折，报告均值 ± 标准差。把第 6 条里的 3 个臂写进预注册判读线，再复查。
- 在新 folds 上做 M2 部分微调（val 规模足够做 early stopping）。

## 本次报告文件

baselines_*_20261104_134419 / _134437、sweep_fer_c_20261104_134517、backbone_compare_20261104_134528、sweep_fused_c_20261104_134541、emo_head_probe_20261104_134626、va_arms_20261104_134633、bottleneck_sweep_20261104_134857 / _134920、multiframe_eval_20261104_134933、oof_stats_20261104_135155。
