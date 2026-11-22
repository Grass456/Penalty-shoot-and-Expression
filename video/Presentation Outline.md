# Penalty Shootout and Facial Expression

## 视频 presentation outline 与 scripts 初稿

### 0. 使用说明

本笔记把 `video/Video Guideline.md` 的交付要求与 `report/report.tex` 的研究内容合并成一套约 9 分 25 秒的 workshop-level 成果汇报方案。建议制作 14 张内容页，另加 1 张 references 页，内容页数量保持在 guideline 的 15 页以内。

### 1. 两类信息的边界

#### 视频 guideline 提供的硬要求

- 视频输出为 MP4，16:9，建议 1080p，最低 720p。
- 时长严格控制在 8–10 分钟，目标时长约 9 分 25 秒。
- 音频清晰，避免背景噪声。讲者出镜可选，建议在开场和结尾短暂出镜。
- 内容需要覆盖标题与研究问题、问题背景、方法、结果与分析、结论与未来工作。
- slides 输出为 PDF，保持统一字体、色彩和版式。单页正文控制在 5–7 个 bullet 以内，优先图表和流程图。
- 图像和关键思想需要标注来源，技术细节保持简洁并保证公式可读。

#### report 提供的研究事实

- 研究问题是预踢面部表征在 match context 之外，能否为 penalty shootout 的 goal/miss 提供增量预测信息。
- 数据包含 40 场 shootout、412 次记录 kick。预处理后保留 385 个有效面部样本，其中 278 个 goal、107 个 miss，27 个 kick 因缺少合格面部帧被排除。
- 视频主要为 360p broadcast。面部候选帧必须落在 ball contact 之前的时间窗口内。
- context 分支编码历史 penalty performance、round 与 kick order、sudden-death 状态、踢球前比分以及缺失历史指示变量。
- visual 分支比较 AffectNet 预训练 FER、ImageNet control、emotion probabilities、frozen embedding、partial fine-tuning 与 late logit fusion。
- 评估采用 match-disjoint stratified grouped five-fold cross-validation、10 个 split seeds、200 次完整标签 permutation refit，以及 2,000 次 match-level bootstrap。
- 主指标是 pooled-rank。context LR 为 `.553 ± .017, p=.060`，frozen FER 为 `.467 ± .028, p=.821`，ImageNet control 为 `.464 ± .021, p=.831`，fused model 为 `.467 ± .028, p=.811`，partial fine-tuning 为 `.488 ± .024, p=.475`。
- FER 与 ImageNet 的差异只有 `+0.003, p=.483`。加入 FER 后，context ranking 变化为 `-0.086`。
- valence–arousal embedding 相比 eight-emotion FER 的差异为 `+0.024, p=.164`。同一子集上 multi-frame mean 相比 single-frame 的差异为 `+0.045, p=.134`。
- 结论是当前样本、画质和建模范围内没有稳定的 FER 增量证据。该结果属于受限的 feasibility finding，不涉及对个人内在情绪状态的推断。

### 2. 整体视觉与讲述策略

- **画幅与版式**：16:9，1080p，单页一个核心信息。标题 32 pt 左右，正文 20–24 pt，避免缩小字号堆叠细节。
- **字体**：Inter。标题使用 SemiBold，正文使用 Regular，数字和指标使用 Medium。
- **色彩**：深海军蓝 `#0B1F3A` 作为背景或标题色，青绿色 `#2D8C91` 表示 FER 分支，金色 `#F5A623` 表示 goal，珊瑚红 `#D95F59` 表示 miss，米白 `#F7F4EE` 作为底色。
- **图形语言**：使用球场线、预踢时间窗、面部 crop、模型分支和结果点图。图表尽量使用大数字与短标签，避免 dashboard 式卡片堆叠。
- **口播节奏**：每页先说结论，再解释图中证据。方法页用一句话解释术语。结果页固定使用过渡句，例如“Now let’s look at the results”。
- **出镜安排**：第一幕人物出镜开场白，最后一幕人物出镜总结。其余页面以 slides 为主，保证图表可读，右上角挂一个人物的小镜头，增强主讲人的全程参与感。

### 3. Slide-by-slide outline

#### Slide 0 — What is penalty shootout and why facial expression?

- **时间**：0:20
- **核心信息**：点球大战是足球比赛的一个特殊阶段，人们往往通过赛前球员的面部特写来猜测罚球的结果。
- **版式设计**：中间写着“Penalty Shootout", 文字变小并移动到正上方，下方出现一段点球大战球员前往罚球点的视频，停在球员踢球前的面部特写。
- **屏幕文字**：
  - `Penalty Shootout`
  - `2022 FIFA World Cup Final: Argentina vs France`
- **讲稿初稿**：
  > 点球大战是足球比赛中的一个特殊阶段，球员需要在罚球点上独自与门将博弈。当转播视角给球员面部特写时，观众往往喜欢通过球员的表情来猜测球员的情绪，进而判断罚球结果。
- **来源/备注**：底部注明视频来源，确保使用授权素材。

#### Slide 1 — Can facial representations predict penalty outcomes?

- **时间**：0:20
- **核心信息**：研究测试的是面部表征能否在 match context 之外提供 goal/miss 的增量信息。
- **版式设计**：左侧放研究问题，右侧放一张真实的face crop并用箭头指向goal。
- **屏幕文字**：
  - `Can Pre-trained Facial Expression Representations Predict Penalty Shootout Outcomes Beyond Match Context?`
- **讲稿初稿**：
  > 转播中的赛前特写让我们产生一个直接的问题：一个预训练的 facial representation，能否在已知比赛情境之外，提供关于下一次罚球结果的额外信息？今天我会介绍这个问题的一个受控可行性研究。
- **来源/备注**：面部 still 必须来自项目数据或获得授权的素材。若暂时没有可用 still，可使用第 5 页的 face-crop 示意图。

#### Slide 2 — Why this is a difficult transfer problem

- **时间**：0:45
- **核心信息**：FER 的 source task 与 penalty outcome 的 target task 存在分布和标签差异。
- **版式设计**：左侧是一条“broadcast face → expression representation → goal/miss”的路径，右侧列出三个可视化困难：低分辨率、遮挡/姿态变化、目标标签不同。页底放小字引用 AffectNet、EfficientNet 与 facial-expression interpretation 文献。
- **屏幕文字**：
  - `Source task: expression representation`
  - `Target task: goal / miss before ball contact`
  - `Low-resolution broadcast footage`
- **讲稿初稿**：
  > 这个迁移问题并不简单。FER 模型通常从带有情绪标签的 in-the-wild 人脸学习，而本项目使用的是低分辨率 broadcast 中的短暂面部帧，最终标签却是一次罚球的 goal 或 miss。姿态、遮挡、压缩和标签定义都会改变迁移效果。因此，研究重点是检验可重复的增量信息，而不是把面部外观直接解释成私人情绪。
- **页脚引用**：Mollahosseini et al., 2019；Tan & Le, 2019；Barrett et al., 2019。

#### Slide 3 — Research question and contribution

- **时间**：0:40
- **核心信息**：研究问题、比较对象和贡献在一页内对齐。
- **版式设计**：中央放研究问题，下面用三个短句排成横向路径。每句配一个简洁图标或小型线稿。
- **屏幕文字**：
  - `Question: Does FER add information beyond context?`
  - `Dataset: penalty-shootout kicks with pre-kick face crops`
  - `Benchmark: match-disjoint comparison with permutation tests`
  - `Outcome: a bounded negative feasibility result`
- **讲稿初稿**：
  > 本项目围绕一个增量问题展开：在历史表现、回合顺序和比分已经可用的情况下，FER 是否还能帮助区分 goal 与 miss？为此，我们构建了带有预踢面部帧和比赛情境的 penalty-shootout 数据集，并在 match-disjoint 的评估中比较 context、FER、ImageNet control、fine-tuning 和 fusion。最终贡献是一项边界清楚的 feasibility benchmark，以及一个稳定性不足的负结果。

#### Slide 4 — Dataset and the pre-kick boundary

- **时间**：0:55
- **核心信息**：样本规模、标签比例和时间边界决定了结论的适用范围。
- **版式设计**：左侧使用数据漏斗 `412 recorded kicks → 385 valid face samples`，右侧使用一条从 face frame 到 ball contact 的时间轴。数字用大号字体，细节控制在 4 个 bullet 内。
- **屏幕文字**：
  - `40 shootouts`
  - `412 recorded kicks`
  - `385 valid face samples`
  - `278 goals / 107 misses`
  - `27 excluded: no qualified face frame`
  - `Mostly 360p broadcast video`
- **讲稿初稿**：
  > 数据集包含 40 场 shootout 和 412 次记录 kick。经过面部质量和时间窗口筛选后，保留 385 个有效样本，其中 goal 278 个、miss 107 个，另有 27 次因为没有合格的预踢面部帧而排除。候选帧必须出现在 ball contact 之前，这条边界保证模型只使用罚球发生前的信息。
- **制作备注**：可在时间轴上明确标注 `pre-kick window` 和 `ball contact`，不要展示会泄漏结果的踢球后画面。

#### Slide 5 — Two information branches

- **时间**：0:50
- **核心信息**：context 和 visual branch 分别编码两类可用信息。
- **版式设计**：使用报告中的 `fusion_pipeline.png` 作为主图，保留两条输入路径。左下角补充 context 特征，右下角补充 FER embedding 和 emotion probabilities。
- **屏幕文字**：
  - `Context: history · order · score`
  - `Visual: face crop → FER embedding / emotion probabilities`
  - `Control: same-backbone ImageNet representation`
- **讲稿初稿**：
  > 模型输入分成两条分支。context 分支使用历史 penalty performance、round 与 kick order、sudden-death 状态、踢球前比分和缺失历史指示变量。visual 分支从 pre-kick face crop 提取 FER 的中间 embedding 和八类 emotion probabilities。为了区分表达导向的预训练和一般视觉表征，我们加入同一 EfficientNet 家族的 ImageNet control。
- **来源**：`report/figures/fusion_pipeline.png`。图中所有输入都应标注为 ball contact 前可获得的信息。

#### Slide 6 — Model families and the attribution logic

- **时间**：0:50
- **核心信息**：模型比较围绕“context 是否有用”和“FER 是否增加信息”展开。
- **版式设计**：用一条从 baseline 到 fusion 的横向阶梯。只保留观众需要理解的 6 个标签，完整 arm 名称放 speaker notes 或 appendix。
- **屏幕文字**：
  - `Context LR / MLP`
  - `Emotion-probability classifiers`
  - `Frozen FER embedding`
  - `ImageNet control`
  - `Partial FER fine-tuning`
  - `Late logit fusion`
- **讲稿初稿**：
  > 我们先建立 context-only baseline，再逐步加入视觉表征。Frozen FER 检验预训练特征本身，ImageNet control 检验表达预训练是否超出一般视觉信息，partial fine-tuning 检验下游适配，late logit fusion 则把 FER 和 context 的两个分数在低维层面合并。这样的设计让每个比较都对应一个明确问题。

#### Slide 7 — Evaluation that respects match structure

- **时间**：0:55
- **核心信息**：match-disjoint、fold-local preprocessing 和 permutation null 是结果可信度的关键。
- **版式设计**：中央放 5-fold 圆环，旁边放三层标签：`match-disjoint`、`10 split seeds`、`200 permutations`。底部用小字写 `2,000 match-bootstrap resamples`。
- **屏幕文字**：
  - `Stratified grouped 5-fold CV`
  - `No match appears in both train and test`
  - `10 split seeds`
  - `200 full-label permutation refits`
  - `Primary metric: pooled-rank`
- **讲稿初稿**：
  > 评估按 match 分组，任何一场比赛的 kick 都不会同时出现在训练和测试中。我们使用 stratified grouped five-fold cross-validation，并用 10 个 split seeds 观察分割变化。主指标 pooled-rank 先在每个 held-out fold 内转换成相对排名，再合并计算 goal-versus-miss 的 ranking AUC。每个模型还进行 200 次完整标签 permutation refit，构造与随机事件相比较的 null 分布。
- **页脚备注**：2,000 次 match-level bootstrap 用于辅助指标的区间估计。所有 scaling、history shrinkage、early stopping 和 model selection 都在对应训练或验证分区内完成。

#### Slide 8 — Main result: no arm clears the permutation test

- **时间**：1:10
- **核心信息**：context 只有弱的 ranking ability，FER、ImageNet、fusion 和 fine-tuning 均未显示稳定证据。
- **版式设计**：主图使用横向 dot plot。横轴为 pooled-rank，点旁直接标出 `mean ± SD`，右侧标出 permutation `p`。用金色突出 context LR，青绿色突出 FER，灰色表示 ImageNet control。不要把辅助 F1 与主指标混在同一坐标轴。
- **建议绘制的数据**：

  | Arm | pooled-rank mean ± SD | test p |
  | --- | ---: | ---: |
  | Context LR | .553 ± .017 | .060 |
  | Frozen FER | .467 ± .028 | .821 |
  | ImageNet control | .464 ± .021 | .831 |
  | Fused | .467 ± .028 | .811 |
  | Partial FT | .488 ± .024 | .475 |

- **屏幕结论**：`Context is weak; FER-based arms remain inside the null comparison.`
- **讲稿初稿**：
  > 现在看主结果。context logistic regression 的 pooled-rank 均值是 .553，接近预设检测边界，但 permutation p 值为 .060。frozen FER 为 .467，ImageNet control 为 .464，late fusion 仍为 .467，partial fine-tuning 为 .488。所有这些视觉相关模型都没有通过对应的 permutation test。这里的主结论来自 pooled-rank，F1 只作为辅助指标。
- **制作备注**：报告引用的 `fig1_main_results.pdf` 目前不在工作区，建议依据上表重绘一张干净的 dot plot，并把 null range 作为浅灰背景带。

#### Slide 9 — Incremental value of FER

- **时间**：0:50
- **核心信息**：表达导向预训练没有超过一般视觉表征，加入 FER 也没有改善 context ranking。
- **版式设计**：两组大号 delta 对比。左侧 `FER − ImageNet = +.003, p=.483`，右侧 `Context + FER − Context = −.086`。每组下面放一句解释，不再放完整模型表。
- **屏幕文字**：
  - `FER vs ImageNet: +0.003, p=.483`
  - `Adding FER to context: −0.086 ranking change`
  - `No stable incremental value in this sample`
- **讲稿初稿**：
  > 关键问题是 FER 有没有带来额外信息。答案在两个对比里都很清楚。FER 与 ImageNet control 的差异只有 .003，配对 p 值为 .483。把 FER 加到 context 后，ranking 反而下降 .086。这个结果把解释重点放在迁移对齐问题上，而不是继续寻找一个更复杂的下游分类器。

#### Slide 10 — Robustness and temporal ablations

- **时间**：0:55
- **核心信息**：更换 affect label space 或增加邻近帧，方向上有变化，但统计稳定性不足。
- **版式设计**：左右两个小型 dumbbell chart。左侧比较 eight-emotion FER 与 VA embedding，右侧比较 single-frame 与 multi-frame mean。每个图只显示均值、差值和 p 值。
- **屏幕文字**：
  - `VA embedding: .490 vs .467, Δ=.024, p=.164`
  - `Multi-frame mean: .494 vs .449, Δ=.045, p=.134`
  - `Random-event split shows no inflation`
  - `Context ablations and quality subset preserve the conclusion`
- **讲稿初稿**：
  > 我们还检查了几个简单解释。把八类 emotion supervision 换成 valence–arousal embedding，均值从 .467 变为 .490，差值 .024，p=.164。对同一子集做 multi-frame mean，均值从 .449 变为 .494，差值 .045，p=.134。两个方向都值得在更大数据上继续测试，但当前样本没有提供稳定改进。随机事件划分、context ablation 和 quality subset 也没有改变主结论。

#### Slide 11 — What the negative result says about transfer

- **时间**：0:50
- **核心信息**：模型能编码 source-task 的面部外观，却没有显示与 goal/miss 标签稳定对齐的 target-task 信息。
- **版式设计**：左侧是“source task → representation → target event”的简化链路，右侧突出一句解释性结论。使用灰色断点或低饱和连接线表示证据弱，不使用失败或崩溃式视觉。
- **屏幕文字**：
  - `Expression-oriented pretraining ≈ generic visual pretraining`
  - `The transfer bottleneck is the target alignment`
  - `Downstream optimization alone did not recover an increment`
- **讲稿初稿**：
  > 综合这些比较，FER 与 ImageNet 的表现几乎相同，fine-tuning 和 logit fusion 也没有恢复稳定增量。一个合理解释是，FER 表征确实能编码与 source task 相关的面部外观，但这些外观在低分辨率 broadcast 中没有和 goal/miss 事件形成足够稳定的对应关系。结果把后续工作指向数据规模、时间信息和目标对齐。

#### Slide 12 — Scope and limitations

- **时间**：0:40
- **核心信息**：结论范围由样本量、画质、分组方式和 fusion 设计共同限定。
- **版式设计**：左侧用四个简短限制条件，右侧用一句 scope statement。保持语气中性，避免把限制条件视觉化成警报。
- **屏幕文字**：
  - `40 matches / 385 valid samples`
  - `Mostly low-resolution broadcast footage`
  - `Split is match-disjoint, not player-disjoint`
  - `Fusion uses restricted logit design`
  - `Facial features are statistical inputs for an observed outcome`
- **讲稿初稿**：
  > 这项研究的结论有明确边界。样本来自 40 场比赛和 385 个有效面部样本，视频主要是低分辨率 broadcast。评估按 match 分组，但目前还没有做到 player-disjoint。fusion 使用的是受限的 logit design，报告也没有执行 attribution analysis。面部特征在这里是对观测结果的统计输入，研究不据此判断个人的私人情绪或心理状态。

#### Slide 13 — Conclusion and future work

- **时间**：0:45
- **核心信息**：用三句话收束发现，再给出与证据直接对应的下一步。
- **版式设计**：上半区放三条结论，下半区放一条从数据规模到模型设计的 future-work 路径。结论数字加粗，文字保持短。
- **屏幕文字**：
  - `Context: weak, non-significant ranking ability`
  - `FER: no advantage over ImageNet control`
  - `Fine-tuning, fusion, VA, and multi-frame: no stable gain`
  - `Next: larger scale · higher-quality video · player-disjoint tests · richer temporal fusion`
- **讲稿初稿**：
  > 最后总结三点。context 提供了弱而未达显著的 ranking ability。FER 没有超过同骨干的 ImageNet control。fine-tuning、fusion、valence–arousal supervision 和 multi-frame averaging 都没有显示稳定增益。下一步应扩大比赛与球员覆盖，使用更高质量的多帧视频，并在 player-disjoint 的设置下检验更丰富的 temporal fusion。当前结果是一项范围清晰的 feasibility benchmark。

#### Slide 14 — Repository and closing

- **时间**：0:20
- **核心信息**：结束时再次回答研究问题，并给出可复现入口。
- **版式设计**：左侧是一句最终答案，右侧放 GitHub repository 的短链接或二维码。讲者出镜 10 秒，背景保持干净。
- **屏幕文字**：
  - `Answer: no stable FER increment appears in the present sample`
  - `Code and project materials: github.com/Grass456/Penalty-shoot-and-Expression`
  - `Thank you`
- **讲稿初稿**：
  > 回到开场的问题：在目前的数据和评估范围内，我们没有观察到稳定的 FER 增量信息。这个结果同时给出了一个可复现的 benchmark 和一个清楚的下一步方向。代码、数据说明和实验材料都整理在项目仓库中。谢谢。
- **来源备注**：仓库链接来自报告正文，正式录制前确认链接和二维码可访问。

#### Slide 15 — References（不计入内容页上限）

- **版式设计**：两列小字号参考文献，保留与 slides 直接相关的 5–7 条。方法页页脚可用短引文，完整信息集中到本页。
- **建议保留**：Mollahosseini et al. 2019（AffectNet）、Tan & Le 2019（EfficientNet）、Roberts et al. 2017（structured cross-validation）、Varma & Simon 2006（model-selection bias）、Good 2005（permutation tests）、Efron 1979（bootstrap）、Barrett et al. 2019（facial movement and emotion inference）。
- **备注**：ImageNet control 的原始来源为 Deng et al. 2009。若空间允许，将其加入本页。

### 4. 录制与制作检查表

- [ ] 内容页共 14 张，references 单独成页。
- [ ] 成片时长保持在 8–10 分钟，目标约 9 分 25 秒。
- [ ] 导出 PDF slides 与 MP4 视频，画面为 16:9，优先 1080p。
- [ ] 主结果图明确标注 pooled-rank、mean ± SD 和 permutation p 值。
- [ ] 只把 F1 放在辅助位置，避免观众把高 goal-rate 带来的 F1 误读为模型发现。
- [ ] 所有面部 still、图标和外部方法思想都写明来源。优先使用项目自有图件。
- [ ] 录音前删除未授权的比赛画面，确认面部图像的展示权限。
- [ ] 音频检查峰值、背景噪声和口播清晰度。每页切换时加入短 verbal cue。
- [ ] 补齐报告引用但当前目录缺失的 `fig1_main_results.pdf`、`fig2_seed_stability.pdf`、`fig3_signal_map.pdf`，或按本笔记中的数据重绘。

### 5. 录制时的过渡句库

- 从动机到方法：`To test this question without leakage, we first define the pre-kick boundary and the two information branches.`
- 从方法到结果：`With the split and null model fixed, now let’s look at the main results.`
- 从主结果到稳健性：`The main comparison is negative, so we next test whether simple representation or temporal changes rescue the signal.`
- 从结果到结论：`Taken together, these comparisons define what the current data can and cannot support.`

