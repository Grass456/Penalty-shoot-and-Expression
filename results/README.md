# results/

先读 [reports/summary_5fold_20261104.md](reports/summary_5fold_20261104.md)，那是当前全部结论的汇总。

| 文件 | 内容 |
|---|---|
| reports/summary_5fold_20261104.md | 当前结论：重复划分主表 + 单划分诊断 + 数据局限 |
| reports/repeated_splits_criteria_preregistered.txt | 重复划分的预注册判读标准 |
| reports/archive_lomo40.md | 40 折 LOMO 时期（已作废）的结果摘要 + 方法论教训；原始文件在 git 提交 3ed0a7b |
| reports/*.json + tables/*.csv | 各脚本的原始输出，同名成对（`<脚本前缀>_<时间戳>`），对应关系见 summary 第 1、2 节 |

约定：
- 新实验沿用 `<前缀>_<YYYYMMDD_HHMMSS>` 命名。同一实验重跑后，旧版本若已被取代就删掉（git 里有）。
- 每完成一阶段，把结论并入 summary，不再单独新建零散的 md。
