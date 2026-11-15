# PenalShoot-FER

> Evaluating Facial Expression and Match Context for Penalty Shootout Outcome Prediction
>
> 面部表情与比赛上下文对点球大战结果的预测研究

研究球员在点球大战主罚前的面部视觉信息，能否在历史点球表现和比赛上下文之外，为该次罚球结果（`Goal` / `Miss`）提供额外的预测信号。

> 核心研究问题：**Do pre-trained facial expression representations provide predictive information beyond historical performance and match context in penalty shootouts?**

---

## 项目动机

点球大战是高压力决策场景。本课程项目评估预训练 Facial Expression Recognition（FER）表征在体育转播画面这一存在领域偏移的场景中，是否具有稳定且可证伪的预测价值。

项目**不**声称读取球员真实心理状态，只讨论面部视觉表征与点球结果之间的**统计相关性**。

---

## 研究范围

### 包含

- 点球大战（penalty shootout）中的罚球事件；
- 每次罚球触球前的**单帧**面部图像；
- 预训练 FER encoder 的情绪概率与中间层 embedding；
- 基于比赛上下文的 Context LR / MLP；
- FER-only / Context-only / FER+Context 三类模型；
- 冻结 vs. 部分微调 FER 的比较；
- 以比赛为分组单位的跨比赛评估（match-disjoint）；
- 面部质量、身份信息与数据泄漏分析。

### 不包含

- 常规时间或加时赛中的单次点球；
- 罚球后的表情、射门动作或守门员反应；
- 多帧视频建模（GRU / LSTM / Temporal Transformer）；
- 将 FER 输出解释为球员真实心理状态；
- 使用罚球结果发生后才能获得的信息。

---

## 方法概览

两条独立分支 + late feature fusion：

```
点球大战视频 ──► 触球前切片 ──► 合格单帧人脸 ──► Pre-trained FER Encoder
                                                    ├──► Emotion Probabilities ──► Face Classifier (FER-only)
                                                    └──► FER Embedding         ──┐
                                                                                 ├──► Feature Fusion ──► Goal/Miss 概率
历史表现/轮次/罚球前比分 ──► Context LR / MLP ───────────────────────────────┘    (FER + Context)
```

- **FER 分支**：冻结 encoder → 预提取 embedding / 情绪概率；下游使用低容量分类器。
- **Context 分支**：最终使用六维 context v2，比较 LR 与 MLP。
- **融合**：比较冻结 FER + context 与部分微调 FER + context；融合实验使用预注册的低容量 logit 融合。

---

## 模型与基线

| 编号 | 模型 | 输入 | 用途 |
|---|---|---|---|
| B0 | Majority Class | — | 下界 |
| B1 | Context Logistic Regression | 六维 context v2 | 线性上下文基线 |
| B2 | Context MLP | 六维 context v2 | 主要 Context-only |
| B3 | Emotion Probability Classifier | FER 情绪概率 | 粗粒度情绪是否有用 |
| B4 | ImageNet Embedding Classifier | ImageNet embedding | FER 预训练的特殊价值对照 |
| M1 | Frozen FER Embedding Classifier | FER embedding | 主要 FER-only |
| M2 | Partially Fine-tuned FER | 单帧人脸 | 领域适配收益 |
| M3 | Frozen FER + Context MLP | 两分支特征 | FER 增量价值 |
| M4 | Fine-tuned FER + Context MLP | 两分支特征 | 最终融合模型 |

最低必做：B0、B1、B2、B3、B4、M1、M3。M2 / M4 已在最终数据集上完成。

---

## 数据计划

数据建设已完成：

1. **Pipeline Pilot**：14 场、123 次罚球、114 个有效人脸，用于验证处理流程。
2. **最终数据集**：40 场、412 次罚球、385 个有效样本（goal 278 / miss 107）；27 个样本因 `no_face` 排除。

### Context 特征

1. 历史点球表现（按罚球日期严格时间截断）；
2. 点球大战状态（轮次内顺序、是否突然死亡）；
3. 罚球前比分（本队/对手已进球、比分差）；
4. `history_missing` 与 `has_history` 指示变量。

主客场和赛事类别在最终版本中未建模，论文中作为与原计划的偏离披露。

### 结果标签

`1 = Goal`（进球）；`0 = Miss`（被扑出 / 射偏 / 击中门框）。二分类为主。

---

## 评估协议

- **主**：StratifiedGroupKFold 5-fold（`match_id` 为 group），同场罚球不跨集合；使用 10 组 split seed。
- **辅**：Random event split，仅用于测量同场信息共享造成的性能高估。
- **主指标**：折内排序后合并的 `pooled_rank`，并以全流程标签置换零分布判读。
- **辅助指标**：AUROC、AUPRC、goal-class F1（阈值 0.5）、Balanced Accuracy、Brier Score、Log Loss。
- **不确定性**：`scripts/eval_f1_bootstrap.py` 提供按 `match_id` 聚类的 percentile bootstrap 95% CI；split seed 的均值 ± 标准差单独报告。
- 进球通常多于未进球，**禁止只报告 accuracy**。

---

## 目录结构

```
Penalty-shoot-and-Expression/
├── data/                  # 数据（视频/人脸图像不入库，见 .gitignore）
│   ├── raw/               # 原始视频（gitignored）
│   ├── clips/             # 触球前切片（gitignored）
│   ├── faces/             # 选定单帧人脸（gitignored）
│   ├── metadata/          # 事件元数据 / 数据字典（可入库）
│   └── external/          # 外部数据集（gitignored）
├── src/                   # 源码模块
│   ├── data/              #   数据 pipeline（切片、人脸选择、context 构建）
│   ├── models/            #   FER / Context / Fusion 模型定义
│   ├── training/          #   训练循环
│   └── config/            #   配置
├── scripts/               # 可执行脚本（切片、标注辅助、跑实验）
├── docs/                  # 文档
│   └── plan.md            #   完整项目计划书
├── notebooks/             # 探索性分析
├── checkpoints/           # 模型权重（gitignored）
├── results/               # 实验产物（表格/图/报告可入库）
└── requirements.txt       # 依赖
```

---

## 快速开始

```bash
# 本地 Windows 工作机（RTX 4060 Laptop）
conda create -n penalty-fer python=3.10 -y
conda activate penalty-fer
# 先装 CUDA 版 torch，再装其余依赖（顺序很重要，避免被 PyPI CPU 版覆盖）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 训练服务器（Linux, 2×A6000 48GB）：PyPI 的 torch 默认即 CUDA 版，直接
#   conda create -n penalty-fer python=3.10 -y && conda activate penalty-fer
#   pip install -r requirements.txt
# 先用 nvidia-smi 确认驱动 ≥ 550（cu124 要求）；驱动较老则按 requirements.txt 头部注释换 cu121

# 已完成的数据与特征产物位于 data/metadata、data/faces、features/fer。
# 若要从原始视频重建 pipeline：
# python scripts/extract_clips.py
# python scripts/select_face_frames.py
# python scripts/cache_fer_features.py

# 验证最终数据与折划分
python scripts/build_folds.py --validate

# 主重复划分实验（已有结果见 results/reports/）
python scripts/eval_repeated_splits.py --n-perm 200

# 补算 goal-class F1 与比赛级 bootstrap 95% CI
python scripts/eval_f1_bootstrap.py --n-bootstrap 2000

# 如需同时补算 M2 / M4（需要 PyTorch 和本地人脸图像，运行更久）
python scripts/eval_f1_bootstrap.py --include-finetune --n-bootstrap 2000
```

> 原始视频和人脸图像不随仓库分发；从头重建 pipeline 需要本地准备合法的视频源和检测模型。

---

## 实施阶段

- **Phase 0**：标注规范与工具准备
- **Phase 1**：完成，14 场 pipeline pilot
- **Phase 2**：完成，最终规模 40 场
- **Phase 3**：完成，单分支基线 B0–B4、M1
- **Phase 4**：完成，融合与部分微调 M2、M3、M4
- **Phase 5**：完成，正式评估、消融、随机划分和质量诊断
- **Phase 6**：下一步，整理结果并进入论文写作

详见 [docs/plan.md](docs/plan.md)。

---

## 当前结果摘要

主结果见 [results/reports/summary_5fold_20261104.md](results/reports/summary_5fold_20261104.md)。在 10 组重复划分下：

- Context LR（B1v2）：`pooled_rank = 0.553 ± 0.017`，未通过预注册显著性判定；
- 冻结 FER embedding（M1）：`0.467 ± 0.028`，置换 `p = 0.821`；
- ImageNet embedding（B4）：`0.464 ± 0.021`，M1 − B4 的配对差为 `+0.003`，`p = 0.483`；
- 冻结 FER + context（M3v2）：低于 context-only，配对差为 `−0.086`；
- 部分微调和 logit 融合均未产生可测增量。

这些结果支持的结论是：在当前样本量、360p 转播画质和检测下限内，未检出 FER 表征对点球结果的稳定增量预测价值；这不是对因果关系或真实心理状态的判断。

---

## 伦理与使用边界

1. 仅使用合法获得的公开比赛视频并记录来源；
2. **不重新分发受版权保护的视频**——公开发布时优先发布处理代码与事件元数据；
3. 不将面部数据用于身份识别或课程之外的个人分析；
4. **不把 FER 输出解释为球员真实心理状态**；
5. 不将模型用于博彩或针对个人的高风险决策；
6. 讨论肤色、年龄、姿态、光照和视频质量可能造成的偏差；
7. 明确区分统计相关性与因果关系。

> The model estimates statistical associations between pre-kick facial representations, match context, and penalty outcomes. It neither observes a player's true emotional state nor establishes that facial behavior causally determines the outcome.

---

## 项目成功标准

项目成功**不要求** FER+Context 取得高准确率。以下任一结果均可形成有效课程结论：

- pilot 证明 pipeline 可可靠扩展并量化成本与有效率；
- FER-only 在 match-disjoint 下稳定优于基线 / 不优于基线；
- FER embedding 优于 / 不优于基本情绪概率；
- FER encoder 优于 / 不优于 ImageNet encoder；
- FER+Context 优于 / 不优于 Context-only；
- random split 与 match-disjoint 差异揭示同场信息泄漏；
- 部分微调造成过拟合说明当前规模下冻结更合适。

最终目标不是预设"面部表情一定能预测点球"，而是**严谨评估** FER 表征与有限比赛上下文在点球大战结果预测中的相对作用与互补性。

---

## License

课程项目，仅用于学术与教学用途。视频与面部图像版权属原版权方，不随本仓库分发。
