# CLAUDE.md

本文件为 Claude Code 在本仓库工作时的指引。每次会话开始时自动加载。

## 项目身份

- **名称**：PenalShoot-FER — 面部表情与比赛上下文对点球大战结果的预测研究
- **类型**：课程研究项目（深度学习），目标产出 NeurIPS 四页正文
- **核心问题**：预训练 FER 表征能否在历史点球表现与比赛上下文之外，为点球大战单次罚球结果（Goal/Miss）提供额外预测信号
- **完整计划书**：[docs/plan.md](docs/plan.md) —— 任何涉及研究设计、数据定义、评估协议的决策都以该文件为权威来源。改动研究设计前先读它。

## 技术栈（计划）

- 语言：Python 3.10+
- 深度学习：PyTorch（FER encoder、MLP、部分微调）
- 视频处理：ffmpeg / ffmpeg-python（触球前切片）
- 人脸检测：待定（候选：MediaPipe / MTCNN / RetinaFace）
- FER 模型：预训练（待选定 backbone，如 FER+ / DDAMFN / HSE-ResNet）
- ImageNet 对照：torchvision 预训练 ResNet
- 评估：scikit-learn（GroupKFold / StratifiedGroupKFold、AUROC、AUPRC、Brier）
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

- 代码风格匹配现有文件（当前仓库尚无源码，新建时遵循 PEP 8 + 类型提示）。
- 配置驱动：模型/数据/训练参数放进 `src/config/*.yaml`，不在脚本里硬编码路径与超参。
- 可复现：固定并报告随机种子；主实验跑多种子；每次实验保存配置 + checkpoint + 指标。
- 数据 pipeline 与模型解耦：FER 特征预提取缓存到 `features/`，下游分类器读缓存，避免重复前向。
- 路径用 `pathlib.Path`；跨平台（Windows bash 环境）注意正斜杠。
- 中文注释可接受（plan.md 为中文），但函数/变量名用英文。

## 常用命令（待实现）

当前 `src/` 与 `scripts/` 为空，以下为 plan.md 规划的入口，随 Phase 推进实现：

```bash
python scripts/build_metadata.py        # 构建事件元数据表
python scripts/extract_clips.py         # 切触球前视频片段
python scripts/select_face_frames.py    # 选定合格单帧人脸
python scripts/cache_fer_features.py   # 缓存 FER embedding/情绪概率
python scripts/run_experiment.py --config src/config/<name>.yaml
python scripts/evaluate.py --folds folds/match_disjoint.json
```

实现脚本时，保持入口与 README "快速开始" 一致。

## 当前阶段

仓库刚初始化，无任何提交。按 plan.md 第 18、22 节，下一步是 **Phase 0 / Action 1：固定标注规范**（触球时刻定义、单帧选择规则、context 编码、历史表现截断规则、赛事类别映射），随后进入 20 场 Pilot。

在用户给出具体编码任务前，不要擅自实现研究逻辑；优先协助固定规范文档或搭建脚手架。

## 与用户协作

- 用户为项目作者，研究设计决策（数据定义、评估协议、假设）已在 plan.md 定稿，**不要重新提议推翻**，除非发现实现上不可行。
- 涉及研究设计变更时，先指向 plan.md 对应章节并确认，再动手。
- 负面结果（FER 无效、融合无提升）在本项目是**有效结论**，不要为追求"提升"而调整评估协议。
- 每次会话结束前，若产生了跨会话有用的事实（如选定的 FER backbone、固定的人脸阈值），写入 `memory/`。
