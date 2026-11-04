# CLAUDE.md

本文件为 Claude Code 在本仓库工作时的指引。每次会话开始时自动加载。

## 项目身份

- **名称**：PenalShoot-FER — 面部表情与比赛上下文对点球大战结果的预测研究
- **类型**：课程研究项目（深度学习），目标产出 NeurIPS 四页正文
- **核心问题**：预训练 FER 表征能否在历史点球表现与比赛上下文之外，为点球大战单次罚球结果（Goal/Miss）提供额外预测信号
- **完整计划书**：[docs/plan.md](docs/plan.md) —— 任何涉及研究设计、数据定义、评估协议的决策都以该文件为权威来源。改动研究设计前先读它。

## 技术栈（已定）

- 语言：Python 3.10+（conda 环境 `penalty-fer`，版本锁定见 requirements.txt）
- 深度学习：PyTorch（MLP、后续部分微调）
- 视频处理：OpenCV（本机无 ffmpeg，切片与截帧都走 cv2）
- 人脸检测：OpenCV YuNet（`models/detection/`）；多帧身份过滤用 SFace
- FER 模型：EmotiEffLib，主 backbone `enet_b0_8_best_vgaf`（1280 维 embedding + 8 类情绪概率）；
  对比用 `enet_b0_8_best_afew` / `enet_b2_8` / `enet_b2_7`，以及 3 个 VA-MTL backbone
- ImageNet 对照：timm `efficientnet_b0`（与主 FER backbone 同骨干）
- 评估：scikit-learn（StratifiedGroupKFold、AUROC、AUPRC、Brier、Log Loss）；
  主判读口径为 pooled_rank + 全流程 refit 置换零分布（见 plan.md §11 修订）
- 元数据：pandas / CSV / Parquet
- 实验跟踪：建议配置 + checkpoint + 指标落盘到 `results/`，不依赖外部服务

依赖最终写入 `requirements.txt`。实现前若对库选择不确定，参照 plan.md 或询问用户。

## 目录约定

```
data/raw        原始视频          （gitignored，版权+体积）
data/clips      触球前切片         （gitignored）
data/faces      选定单帧人脸       （gitignored，个人数据）
data/metadata   事件元数据/字典    （可入库，不含 PII）
data/external   外部数据集         （gitignored）
src/data        数据 pipeline
src/models      模型定义
src/training    训练循环
src/evaluation  划分/指标/消融
src/config      配置文件
scripts         可执行脚本
notebooks       探索性（ipynb，不入库输出）
checkpoints     模型权重            （gitignored）
results/tables  结果表（可入库）
results/figures 图（可入库）
results/reports 报告（可入库）
```

新增文件时遵循上述归位；`.gitignore` 已按媒体类型与目录屏蔽大文件和敏感数据。

## 硬性约束（不可违反）

以下来自 plan.md，违反即破坏研究有效性，写代码时必须保证：

1. **触球前帧**：人脸图像必须严格来自触球前。禁止使用罚球后表情（会直接泄漏结果）。
2. **时间截断**：历史点球统计以本次罚球日期为截止点，禁止使用未来数据（temporal leakage）。
3. **罚球前比分**：所有比分/轮次派生变量只按罚球前状态计算，不能含当前罚球结果。
4. **Match-disjoint**：同一场点球大战的所有罚球只能出现在 train/val/test 之一；GroupKFold 的 group=`match_id`。所有模型共享相同 folds。
5. **拟合参数来源**：标准化器、缺失值填充、Beta 平滑参数、类别编码**只能由训练集确定**，再 apply 到 val/test。
6. **选模型**：用验证集 early stopping，**禁止用测试集选模型**。
7. **类不平衡**：进球多于未进球，不能只报告 accuracy；至少报 AUROC / AUPRC / Balanced Acc / Brier。
8. **不重分发版权视频与人脸图像**：公开发布只发处理代码与事件元数据。
9. **不解释为真实心理状态**：用 "facial visual representation" 描述输入，只谈预测相关性。
10. **元数据完整性**：被排除样本及原因必须保留（`exclusion_reason`），避免选择性保留。

## 编码规范

- 代码风格匹配现有文件（PEP 8 + 类型提示；脚本头部 docstring 写用法、协议与输出路径，参照 `scripts/run_baselines.py`）。
- Windows 下读写 json/csv 一律显式 `encoding="utf-8"`（默认 GBK 会炸）；运行脚本前设 `PYTHONIOENCODING=utf-8`。
- 配置驱动：模型/数据/训练参数放进 `src/config/*.yaml`，不在脚本里硬编码路径与超参。
- 可复现：固定并报告随机种子；主实验跑多种子；每次实验保存配置 + checkpoint + 指标。
- 数据 pipeline 与模型解耦：FER 特征预提取缓存到 `features/`，下游分类器读缓存，避免重复前向。
- 路径用 `pathlib.Path`；跨平台（Windows bash 环境）注意正斜杠。
- 中文注释可接受（plan.md 为中文），但函数/变量名用英文。

## 常用命令

```bash
# 数据标注（人工）
python scripts/marker.py --match m004        # 打特写(f)/触球(空格)时间点
python scripts/capture_face.py --match m004  # 通看视频，人工截主罚球员触球前人脸
python scripts/fill_events.py --validate     # 由 result + 踢序推导轮次/比分/突然死亡（--interactive 填写）

# 特征与划分
python scripts/cache_fer_features.py [--model <backbone>] [--imagenet]  # 单帧特征缓存 → features/fer/<name>/
python scripts/build_folds.py [--validate]   # 5 折 SGKF + 内层 val → folds/match_disjoint.json

# 实验
python scripts/run_baselines.py --models majority,context_lr_v2,fer_lr,imagenet_lr,fused_lr_v2 \
       --config src/config/context_v2.yaml   # 单划分基线（seed 2026）
python scripts/eval_repeated_splits.py       # 10 seed × 5 折 + 200 次置换（主结论来源）
python scripts/evaluate_oof_stats.py         # 单划分统计口径 + 置换
```

events.csv 变动后，必须先重跑 `cache_*_features.py --force` 和 `build_folds.py`。下游脚本按 event_id 对齐，发现不一致会报错。

## 当前阶段（2026-11-04）

详细进度、假设检验现状和与原计划的偏离见 [docs/plan.md](docs/plan.md) §0；当前全部结论数字见 [results/reports/summary_5fold_20261104.md](results/reports/summary_5fold_20261104.md)。

- **数据已定稿**：40 场 / 385 有效样本，不再扩数据。
- **划分已定稿**：5 折 StratifiedGroupKFold + 内层 val（`folds/match_disjoint.json` schema v2）。
  正式结论用重复划分（`folds/repeats/`，seed 2027–2036）。旧 40 折 LOMO 已作废，不要引用其数字。
- **已完成**：B0–B4、M1、M3 及多项 FER 侧诊断。重复划分下没有任何臂达到显著（B1v2 最接近，p=0.060）。
- **下一步**：M2 部分微调 → M4（先预注册判读线），再做 A6 random split、A5 context 消融。
- **待用户决定**：主客场与赛事类别两类 context 是补标还是记为偏离（plan.md §6.6 修订）。

新实验先写预注册判读标准，再跑；结果落盘 `results/`，并把结论并入 summary（见 `results/README.md`），不另建零散报告。

## 与用户协作

- 用户为项目作者，研究设计决策（数据定义、评估协议、假设）已在 plan.md 定稿，**不要重新提议推翻**，除非发现实现上不可行。
- 涉及研究设计变更时，先指向 plan.md 对应章节并确认，再动手。
- 负面结果（FER 无效、融合无提升）在本项目是**有效结论**，不要为追求"提升"而调整评估协议。
- 每次会话结束前，若产生了跨会话有用的事实（如选定的 FER backbone、固定的人脸阈值），写入 `memory/`。
