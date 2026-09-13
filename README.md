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
- 基于五类上下文变量的 Context MLP；
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
历史表现/主客场/轮次/罚球前比分/赛事类别 ──► Context MLP ───────────────────────────┘    (FER + Context)
```

- **FER 分支**：冻结 encoder → 预提取缓存 embedding / 情绪概率；下游用低容量 MLP。
- **Context 分支**：五类变量 → 1–2 层 MLP + dropout + weight decay。
- **融合**：两分支投影到相近维度后拼接，避免高维 FER 压倒低维 context。

---

## 模型与基线

| 编号 | 模型 | 输入 | 用途 |
|---|---|---|---|
| B0 | Majority Class | — | 下界 |
| B1 | Context Logistic Regression | 五类 context | 线性上下文基线 |
| B2 | Context MLP | 五类 context | 主要 Context-only |
| B3 | Emotion Probability Classifier | FER 情绪概率 | 粗粒度情绪是否有用 |
| B4 | ImageNet Embedding Classifier | ImageNet embedding | FER 预训练的特殊价值对照 |
| M1 | Frozen FER Embedding Classifier | FER embedding | 主要 FER-only |
| M2 | Partially Fine-tuned FER | 单帧人脸 | 领域适配收益 |
| M3 | Frozen FER + Context MLP | 两分支特征 | FER 增量价值 |
| M4 | Fine-tuned FER + Context MLP | 两分支特征 | 最终融合模型 |

最低必做：B0、B1、B2、B3、B4、M1、M3。M2 / M4 在正式数据集与冻结模型稳定后进行。

---

## 数据计划

两阶段建设：

1. **阶段一 — 20 场 Pipeline Pilot**：验证视频可得性、人脸切片质量、FER 可用性、context 完整性与处理成本，**不作最终结论**。
2. **阶段二 — 60–80 场正式数据集**：基于 pilot 实测有效率扩展，构建 match-disjoint folds。

### Context 五类特征

1. 历史点球表现（Beta smoothing，按罚球日期严格时间截断）；
2. 主客场（`home` / `away` / `neutral`）；
3. 点球大战轮次（轮次号、本轮先/后罚、是否突然死亡）；
4. 罚球前比分（本队/对手已进球、净胜球差、已完成罚球数）；
5. 赛事类别（合并为少量稳定类别）。

### 结果标签

`1 = Goal`（进球）；`0 = Miss`（被扑出 / 射偏 / 击中门框）。二分类为主。

---

## 评估协议

- **主**：Match-disjoint GroupKFold（`match_id` 为 group，建议 5-fold），同场罚球不跨集合。
- **辅**：Random event split，仅用于测量同场信息共享造成的性能高估。
- **指标**：AUROC、AUPRC、Balanced Accuracy、F1、Brier Score、Log Loss；多随机种子 + bootstrap CI。
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
│   ├── evaluation/        #   划分、指标、消融
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

# 2.（待实现）准备数据元数据
# python scripts/build_metadata.py --matches data/metadata/matches.csv

# 3.（待实现）提取触球前切片与单帧人脸
# python scripts/extract_clips.py
# python scripts/select_face_frames.py

# 4.（待实现）缓存 FER 特征
# python scripts/cache_fer_features.py

# 5.（待实现）跑基线与融合
# python scripts/run_experiment.py --config src/config/baselines.yaml
```

> 以上命令为规划中的入口，具体实现随 Phase 推进补全。

---

## 实施阶段

- **Phase 0**：标注规范与工具准备
- **Phase 1**：20 场 Pipeline Pilot + 端到端 smoke test
- **Phase 2**：扩展至 60–80 场
- **Phase 3**：单分支基线（B0–B4、M1）
- **Phase 4**：融合与微调（M3、M2、M4）
- **Phase 5**：正式评估与消融（match-disjoint、随机种子、特征消融、泄漏分析）
- **Phase 6**：写作与展示（NeurIPS 四页正文）

详见 [docs/plan.md](docs/plan.md)。

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
