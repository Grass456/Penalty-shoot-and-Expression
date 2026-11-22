# Penalty Shootout and Facial Expression

## 成片规格

- 目标时长 9 分 25 秒，16:9，1080p 优先，最低 720p。
- **时间码校验**：outline 中 Slide 0–14 的单页时长相加为 11 分 05 秒。当前分镜保留 outline 的逐页时长，正式录制前需要压缩 1 分 40 秒，或将目标总时长改为 11 分 05 秒。
- Slides 共 16 页。第 0–14 页为视频内容，第 15 页为 references，录制时可作为片尾静帧。
- 视觉基调使用深海军蓝、米白、青绿色、金色和珊瑚红。字体为 Inter。
- 出镜安排在开场和结尾各约 10 秒。其余页面以 slides 为主，可在右上角叠加讲者小窗。

## 分镜表

| 时间 | 页码 | 画面与动作 | 屏幕文字 | 口播 | 音频与转场 | 素材状态 |
|---|---:|---|---|---|---|---|
| 00:00–00:20 | 0 | 讲者出镜 10 秒后切入赛前特写。标题保持在左侧，右侧停在踢球前 face crop。 | `Penalty Shootout and Facial Expression`<br>`Feasibility study · 40 shootouts · 385 valid face samples` | 点球大战是足球比赛中的特殊阶段，球员需要在罚球点上独自与门将博弈。当转播给出球员面部特写时，观众常通过表情猜测情绪，进而猜测罚球结果。 | 开场环境声淡入。讲者说完后用短淡化切到研究问题。 | **占位**：当前 deck 使用项目 face crop。需要获得授权的比赛视频或开场人物镜头。 |
| 00:20–00:40 | 1 | 左侧逐行出现研究问题，右侧 face crop 通过短线指向 goal/miss。 | `Can facial representations predict penalty outcomes?` | 转播中的赛前特写让我们产生一个直接的问题：预训练的 facial representation，能否在已知比赛情境之外，提供关于下一次罚球结果的额外信息？ | 语音承接上一页，页面保持静止，避免复杂动画。 | 面部 still 来自项目 metadata，录制前确认展示权限。 |
| 00:40–01:25 | 2 | 用三段链路表现 source task、representation 和 target event。底部依次出现低分辨率、姿态遮挡、标签差异。 | `Expression representation`<br>`goal / miss before contact` | 这个迁移问题并不简单。FER 从带情绪标签的人脸学习，而本项目使用低分辨率 broadcast 中的短暂面部帧，最终标签却是一次罚球的 goal 或 miss。研究重点是检验可重复的增量信息，不把面部外观解释成私人情绪。 | 说到每个困难时出现对应短语。切换到 study design。 | 方法观点引用写在页脚和 references 页。 |
| 01:25–02:05 | 3 | 四个横向模块依次高亮 question、dataset、benchmark、outcome。 | `Does FER add information beyond context?`<br>`pre-kick face crops`<br>`match-disjoint comparison`<br>`bounded negative feasibility result` | 本项目围绕一个增量问题展开。在历史表现、回合顺序和比分已经可用的情况下，FER 是否还能帮助区分 goal 与 miss？我们比较 context、FER、ImageNet control、fine-tuning 和 fusion。 | 每个模块之间保留约 0.5 秒停顿。 | 无缺失素材。 |
| 02:05–03:00 | 4 | 左侧展示 40、412、385 三个大数字。右侧时间轴从 face frame 走到 ball contact，ball contact 之后不显示任何画面。 | `40 shootouts`<br>`412 recorded kicks`<br>`385 valid face samples`<br>`278 goals / 107 misses`<br>`pre-kick window` | 数据包含 40 场 shootout 和 412 次记录 kick。筛选后保留 385 个有效样本，其中 goal 278 个、miss 107 个，27 次因没有合格的预踢面部帧而排除。候选帧必须出现在 ball contact 之前。 | 数字出现时加入轻微 click。时间轴用单次横向擦入。 | 数据来自 `events.csv` 和 `face_frames.csv`。 |
| 03:00–03:50 | 5 | 显示项目 `fusion_pipeline.png`。上方为 visual branch，下方为 context branch，右端落到 goal/miss。 | `Context: history · order · score`<br>`Visual: face crop · FER embedding · emotion probabilities` | 模型输入分成两条分支。context 使用历史表现、回合顺序、sudden-death 状态和比分。visual 从 pre-kick face crop 提取 FER embedding 和八类 emotion probabilities，并加入 ImageNet control。 | 讲者用手势分别指向两条路径。用短 verbal cue 进入下一页。 | 使用项目自有图件。 |
| 03:50–04:40 | 6 | 从 context baseline 开始，横向出现 emotion probabilities、frozen FER、ImageNet control、partial FT 和 fusion。 | `Context LR / MLP`<br>`Frozen FER embedding`<br>`ImageNet control`<br>`Partial fine-tuning`<br>`Late logit fusion` | 我们先建立 context-only baseline，再逐步加入视觉表征。Frozen FER 检验预训练特征本身，ImageNet control 检验表达预训练是否超出一般视觉信息，fine-tuning 和 late fusion 分别检验下游适配与分数融合。 | 每个模型名出现时只保留一个简短强调音。 | 无缺失素材。 |
| 04:40–05:35 | 7 | 左侧为 5-fold 圆环，右侧依次显示 match-disjoint、10 split seeds、200 permutations，底部补充 bootstrap。 | `Stratified grouped 5-fold CV`<br>`No match appears in both train and test`<br>`Primary metric: pooled-rank` | 评估按 match 分组，任何一场比赛的 kick 都不会同时出现在训练和测试中。我们使用 stratified grouped five-fold cross-validation、10 个 split seeds 和 200 次完整标签 permutation refit。 | 使用过渡句：`To test this question without leakage, we first define the pre-kick boundary and the two information branches.` | 无缺失素材。 |
| 05:35–06:45 | 8 | 左侧放项目主结果图，右侧用金色卡片突出 Context LR，用青绿色卡片突出 Frozen FER。 | `Context LR .553 ± .017 | p=.060`<br>`Frozen FER .467 ± .028 | p=.821`<br>`Visual arms remain inside the permutation comparison.` | 现在看主结果。context logistic regression 的 pooled-rank 是 .553，p=.060。frozen FER 是 .467，ImageNet control 是 .464，late fusion 是 .467，partial fine-tuning 是 .488。视觉相关模型都没有通过 permutation test。 | 使用固定过渡句：`With the split and null model fixed, now let’s look at the main results.` | 使用 `results/figures/fig1_main_results.png`。 |
| 06:45–07:35 | 9 | 两个大号 delta 并列。左侧 FER–ImageNet 为 +.003，右侧加入 FER 后 context ranking 为 −.086。 | `FER − ImageNet = +0.003, p=.483`<br>`Context + FER − Context = −0.086` | FER 与 ImageNet control 的差异只有 .003，配对 p 值为 .483。把 FER 加到 context 后，ranking 反而下降 .086。当前样本没有显示稳定的 FER 增量。 | 主结果后停顿 1 秒，随后进入 robustness。 | 无缺失素材。 |
| 07:35–08:30 | 10 | 左右两组 dumbbell chart。左侧比较 8-class FER 与 VA embedding，右侧比较 single-frame 与 multi-frame mean。 | `VA embedding: .490 vs .467, Δ=.024, p=.164`<br>`Multi-frame mean: .494 vs .449, Δ=.045, p=.134` | 把八类 emotion supervision 换成 valence–arousal embedding，差值是 .024，p=.164。multi-frame mean 的差值是 .045，p=.134。方向有变化，但当前样本没有稳定改进。 | 使用过渡句：`The main comparison is negative, so we next test whether simple representation or temporal changes rescue the signal.` | 数值来自 outline 与项目结果图。 |
| 08:30–09:20 | 11 | 左侧链路为 source task、representation、target event。中部深蓝框突出 target alignment。 | `Expression-oriented pretraining ≈ generic visual pretraining`<br>`The bottleneck is target alignment.` | FER 与 ImageNet 的表现几乎相同，fine-tuning 和 logit fusion 也没有恢复稳定增量。一个合理解释是，source-task 的面部外观没有和 goal/miss 事件形成足够稳定的对应关系。 | 画面保持克制，避免失败或崩溃式视觉。 | 无缺失素材。 |
| 09:20–10:00 | 12 | 米白页展示五条限制条件，右侧深蓝框给出 scope statement。 | `40 matches / 385 valid samples`<br>`Mostly low-resolution broadcast footage`<br>`Player-disjoint testing remains open`<br>`Restricted late-logit fusion design` | 结论范围由样本量、画质、分组方式和 fusion 设计共同限定。评估按 match 分组，但目前还没有做到 player-disjoint。面部特征在这里是对观测结果的统计输入，研究不据此判断私人情绪或心理状态。 | 逐条出现，每条停留约 3 秒。 | 无缺失素材。 |
| 10:00–10:45 | 13 | 深色页用三行结论收束，下半区出现 future-work 路径。 | `Context: weak, non-significant ranking ability`<br>`FER: no advantage over ImageNet control`<br>`Next: larger scale · higher-quality video · player-disjoint tests · richer temporal fusion` | 最后总结三点。context 提供弱而未达显著的 ranking ability。FER 没有超过 ImageNet control。fine-tuning、fusion、VA 和 multi-frame 都没有显示稳定增益。下一步是更大规模、更高质量视频和 player-disjoint 测试。 | 进入结尾前使用 `Taken together, these comparisons define what the current data can and cannot support.` | 无缺失素材。 |
| 10:45–11:05 | 14 | 讲者出镜 10 秒。左侧给最终答案和仓库链接，右侧保留二维码白色占位框。 | `No stable FER increment appears in the present sample.`<br>`github.com/Grass456/Penalty-shoot-and-Expression`<br>`Thank you` | 回到开场的问题：在目前的数据和评估范围内，我们没有观察到稳定的 FER 增量信息。代码、数据说明和实验材料整理在项目仓库中。谢谢。 | 音乐轻微回升，最后停在 Thank you。 | **占位**：需要讲者结尾镜头和可访问仓库二维码。 |
| 11:05–11:05+ | 15 | References 静帧，可作为录制后片尾或导出 PDF 的最后一页。 | 8 条精选参考文献与项目材料路径 | 不口播，或只说一句 `References and project materials are on screen.` | 片尾静帧保持 3–5 秒。 | 文献来源与项目路径已列出。 |

## 录制前需要补齐的素材

1. **开场比赛视频或讲者开场镜头**。需获得授权，建议使用 5–8 秒、球员踢球前的片段，避免展示 ball contact 之后的结果画面。
2. **面部 still 的展示权限**。当前 deck 使用项目 metadata 中的 face crop。确认项目数据许可后再录制。
3. **结尾讲者镜头**。建议干净背景、约 10 秒，画面比例 16:9。
4. **仓库二维码**。确认仓库链接可访问后生成高清二维码，替换第 14 页的 `QR PLACEHOLDER`。
5. **外部素材来源字幕**。若使用比赛画面或外部图标，在对应镜头和 references 页标注来源。

## 音频与剪辑提示

- 口播目标约 125–135 wpm，总时长保持在 8–10 分钟。
- 每页先说结论，再解释图中证据。结果页不要把 F1 和 pooled-rank 放在同一坐标轴上。
- 录音前检查峰值、背景噪声和口播清晰度。页面切换时保留短 verbal cue。
- 最终导出 MP4 之前确认所有占位框已替换或保留清晰说明。
