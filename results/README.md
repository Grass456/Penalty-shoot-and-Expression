# results/

先读 [reports/summary_5fold_20261104.md](reports/summary_5fold_20261104.md)，那是当前全部结论的汇总。

| 文件 | 内容 |
|---|---|
| reports/summary_5fold_20261104.md | 当前结论：重复划分主表 + A6 随机划分对照 + A5 特征组消融 + M2 部分微调 + 单划分诊断 + 数据局限 |
| reports/repeated_splits_criteria_preregistered.txt | 重复划分的预注册判读标准 |
| reports/random_split_criteria_preregistered.txt | A6/H7 随机划分对照的预注册判读标准 |
| reports/context_ablation_criteria_preregistered.txt | A5 context 特征组消融的预注册判读标准 |
| reports/m2_finetune_criteria_preregistered.txt | M2 部分微调的预注册判读标准（含置换降级口径与全部超参） |
| reports/archive_lomo40.md | 40 折 LOMO 时期（已作废）的结果摘要 + 方法论教训；原始文件在 git 提交 3ed0a7b |
| reports/*.json + tables/*.csv | 各脚本的原始输出，同名成对（`<脚本前缀>_<时间戳>`），对应关系见 summary 第 1–5 节 |

约定：
- 新实验沿用 `<前缀>_<YYYYMMDD_HHMMSS>` 命名。同一实验重跑后，旧版本若已被取代就删掉（git 里有）。
- 每完成一阶段，把结论并入 summary，不再单独新建零散的 md。
- **可复现性**：报告 JSON 里带 `determinism.mlp_init_seeded = true` 的才是逐位可复现的输出
  （2026-11-04 前 MLP 臂建模型前未固定 init，见 summary 顶部的勘误）。旧报告保留用于审计对照。
