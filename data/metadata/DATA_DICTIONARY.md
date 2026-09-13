# 数据字典与标注规范（Data Dictionary & Annotation Guide）

> 本文档对应 plan.md §6.6–6.8 与 §22 Action 1。**在批量处理 20 场 pilot 前冻结本规范**；
> 建模开始后如需修改任何字段定义，必须在 `CHANGELOG` 部分记录修改日期和理由。

两张表配合使用：

- `matches_template.csv` — 每场比赛一行（点球大战粒度），用于 pilot 统计与 match-disjoint 分组；
- `events_template.csv` — 每次罚球一行（plan.md §6.7 元数据表），建模的主表。

`event_id` / `match_id` / `player_id` 用自增整数或 `m001` / `m001-k03` 风格编号均可，
但**一经分配不得复用或重排**；真实球员姓名不出现在任何入库文件中（伦理约束，plan.md §16）。

---

## 1. matches 表（比赛级）

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `match_id` | str | ✅ | 比赛唯一编号，如 `m001`。match-disjoint 划分的 group 键 |
| `match_date` | date `YYYY-MM-DD` | ✅ | 比赛日期（当地日期）。**历史统计时间截断的依据**，必须准确 |
| `competition` | str | ✅ | 赛事原名，如 `UEFA Euro 2020`、`FA Cup 2018-19` |
| `competition_category` | enum | ✅ | 四类之一：`international_nt`（国际国家队）/ `domestic_cup`（国内杯赛）/ `continental_club`（洲际俱乐部）/ `other`。**建模前冻结映射，不得过细**（plan.md §6.6.5） |
| `home_team` | str | ✅ | 主队名（官方赛程身份） |
| `away_team` | str | ✅ | 客队名 |
| `venue` | str | ✅ | 实际比赛场地（城市/球场名）。用于核验主客场（plan.md §6.6.2：以实际场地状态为准） |
| `is_neutral_venue` | bool `0/1` | ✅ | 是否中立场地。`1` 时 `venue_status` 必须为 `neutral` |
| `video_source` | str | ✅ | 视频来源（如 `YouTube/<channel>`）。只记录来源，**视频本身不入库** |
| `video_quality` | enum | ✅ | `hd`（≥720p）/ `sd`（<720p）。用于画质分层分析 |
| `n_raw_penalties` | int | ✅ | 该场原始罚球事件数（pilot 统计指标，plan.md §6.2） |
| `n_valid_faces` | int | ✅ | 通过质量筛选、进入数据集的样本数 |
| `n_manual_corrections` | int | ✅ | 人脸检测需人工修正的次数（估算自动化程度） |
| `processing_time_min` | float | ✅ | 单场人工处理分钟数（估算扩展成本） |
| `processing_status` | enum | ✅ | `todo` / `clipped`（已切片）/ `annotated`（已标注）/ `qc_done`（已质检） |
| `notes` | str | | 自由备注（如：转播切换频繁、加时后点球等特殊情况） |

---

## 2. events 表（罚球级，主表）

### 2.1 标识与文件

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `event_id` | str | ✅ | 罚球唯一编号，建议 `m001-k03`（m001 场第 3 踢）。同一次罚球**不得重复入库**（plan.md §6.8.7） |
| `match_id` | str | ✅ | 外键 → matches 表。同场所有罚球只能进 train/val/test 之一 |
| `player_id` | str | ✅ | 球员唯一编号（如 `p0001`），姓名不入库。用于跨折重复球员统计与 player-disjoint 检查 |
| `match_date` | date | ✅ | 冗余自 matches 表，便于独立校验时间截断 |
| `clip_path` | str | ✅ | 触球前视频切片相对路径（相对 `data/clips/`），如 `m001/k03.mp4` |
| `face_path` | str | ✅ | 选定单帧人脸相对路径（相对 `data/faces/`），如 `m001/k03.jpg`。被排除的样本留空并填 `exclusion_reason` |
| `face_timestamp_s` | float | | 选定帧在**原视频**中的时间戳（秒）。用于质检"严格触球前"（plan.md §6.8.2） |
| `source` | str | ✅ | 该行数据的来源：`manual`（人工标注）/ `auto`（脚本提取）/ `auto+manual_fix` |

### 2.2 标签（plan.md §6.4）

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `result` | enum | ✅ | 主任务二分类标签：`goal` / `miss`。**以官方比赛记录为准**（plan.md §6.8.4） |
| `result_detail` | enum | | 细分：`saved`（被扑出）/ `off_target`（射偏）/ `woodwork`（击中门框）。`result=goal` 时留空。仅作探索性分析，**不参与主任务** |

> 判定边界约定：球被守门员扑出后补射进门——点球大战中不存在补射，`result=miss`（`saved`）。
> 击中门框弹入 → `goal`；弹出 → `miss`（`woodwork`）。

### 2.3 历史点球表现（plan.md §6.6.1）

**时间截断规则（硬性约束）**：所有历史字段只统计 `match_date` **严格早于**本场日期的比赛。
同日比赛无法确定先后时，按"无历史"处理并记 `history_missing=1`。**禁止使用未来数据。**

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `historical_attempts` | int | ✅ | 该球员此前（严格早于本场）点球大战主罚总次数。无记录填 `0` |
| `historical_goals` | int | ✅ | 此前历史进球数。无记录填 `0` |
| `historical_rate` | float | ✅ | **平滑后**历史成功率：`(goals+α)/(attempts+α+β)`。α/β 只由训练集确定（plan.md 硬性约束 5）；**标注阶段此列留空**，由 `build_metadata.py` 在训练时计算并回填 |
| `history_missing` | bool `0/1` | ✅ | 是否缺少历史记录（`attempts==0` 即为 `1`）。缺失值填充规则同样只由训练集确定 |

### 2.4 主客场（plan.md §6.6.2）

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `venue_status` | enum | ✅ | `home` / `away` / `neutral`。**以实际场地状态为准**（官方赛程主队 ≠ 实际主场时，以场地归属判断；决赛等中立场地一律 `neutral`） |

### 2.5 点球大战轮次（plan.md §6.6.3）

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `round_number` | int | ✅ | 当前第几轮，从 1 开始。前 5 轮每轮各队 1 踢；进入突然死亡后继续递增（第 6 轮、第 7 轮……） |
| `kick_order_in_round` | enum | ✅ | `first`（本轮先罚方）/ `second`（本轮后罚方）。以该队在本轮的先后为准 |
| `sudden_death` | bool `0/1` | ✅ | 是否突然死亡阶段（前 5 轮结束后双方仍平局开始的额外轮次）。`round_number>=6` 时通常为 `1`，但以实际赛况为准（若前 5 轮后未平则本场无突然死亡） |

### 2.6 罚球前比分（plan.md §6.6.4）

**硬性约束**：以下所有字段只反映**本次罚球发生前**的状态，不得包含本次罚球结果（plan.md §6.8.5）。

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `team_score_before` | int | ✅ | 本队在本场点球大战中已进球数（不含本次） |
| `opponent_score_before` | int | ✅ | 对手已进球数（不含本次） |
| `team_kicks_before` | int | ✅ | 本队此前已完成罚球次数（不含本次） |
| `opponent_kicks_before` | int | ✅ | 对手已完成罚球次数。注意：后罚方踢第 1 轮时对手可能已完成 1 踢，先罚方为 0，如实记录 |

> 派生变量（净胜球差 `goal_diff = team_score_before - opponent_score_before`、
> "罚进即胜/罚丢即负"压力状态等）**不单独入库**，由训练代码从这四列即时派生，
> 避免标注阶段手工计算出错（plan.md §6.6.4：所有派生变量只按罚球前状态计算）。

### 2.7 人脸质量（plan.md §6.5、§6.6.7）

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `face_detection_score` | float 0–1 | ✅ | 自动人脸检测置信度。无检测时留空 |
| `face_quality` | enum | ✅ | `good` / `ok` / `bad`。判定标准见 §3.2。被排除样本也必须填写 |

### 2.8 排除记录（plan.md §6.8.9）

| 字段 | 类型 | 必填 | 定义与允许值 |
|---|---|---|---|
| `exclusion_reason` | enum | 排除时必填 | 留空 = 样本有效。允许值见下 |

**`exclusion_reason` 允许值（冻结，不得自创）**：

| 值 | 含义 |
|---|---|
| `no_face` | 整个触球前切片中无可用人脸 |
| `face_too_small` | 人脸低于最小像素尺寸 |
| `occluded` | 眉/嘴严重遮挡，无合格帧 |
| `blurry` | 全部候选帧运动模糊/低分辨率 |
| `after_kick_only` | 只有触球后画面（**此类样本绝不入库为有效样本**） |
| `wrong_person` | 画面中不是当前主罚球员（plan.md §6.8.1） |
| `result_leakage_risk` | 画面含比分更新/庆祝/失望反应（plan.md §6.8.3） |
| `label_uncertain` | 结果标签无法与官方记录核对 |
| `duplicate` | 与已有事件重复 |
| `video_unavailable` | 视频缺失或无法解码 |

> 被排除样本**保留整行并填写原因**，避免选择性保留（plan.md 硬性约束 10）。

---

## 3. 冻结的判定规则（Phase 0 定稿，批量标注前不再改动）

### 3.1 触球时刻与"触球前"

- **触球时刻** = 支撑脚落定后、踢球脚首次接触球的帧；
- **触球前切片** = 从主罚球员进入画面起，至触球时刻前**至少 0.5s** 截止；
- 选帧窗口 = 切片内、且严格早于触球时刻。

### 3.2 人脸质量阈值（`face_quality` 判定）

| 等级 | 判定标准 |
|---|---|
| `good` | 人脸长边 ≥ 64px；检测置信度 ≥ 0.9；双眼、眉毛、嘴清晰可见；无明显运动模糊 |
| `ok` | 人脸长边 ≥ 48px；检测置信度 ≥ 0.7；眉/嘴部分可见但不影响表情判断；轻度模糊 |
| `bad` | 低于 `ok` 任一标准。**不得入选**（plan.md §6.5 规则 2–3） |

单帧选择规则（plan.md §6.5）：在合格帧中选**最接近触球时刻**的一张；
最后一帧不合格则按时间**向前**回退找最近的合格帧；整段无合格帧则排除并记原因。

### 3.3 QC 检查单（每场标注完成后过一遍，plan.md §6.8）

1. [ ] 每行 `face_path` 指向的确实是当前主罚球员（非门将/裁判/观众）；
2. [ ] 抽查 `face_timestamp_s` < 触球时刻；
3. [ ] 图像不含比分字幕、庆祝或失望反应；
4. [ ] `result` 与官方比赛记录一致；
5. [ ] `*_score_before` / `*_kicks_before` 不含本次结果；
6. [ ] 历史统计截止日 < 本场 `match_date`；
7. [ ] 无重复 `event_id`；
8. [ ] 被排除样本均有 `exclusion_reason`。

---

## 4. 最小填表示例

`matches_template.csv`：

```csv
match_id,match_date,competition,competition_category,home_team,away_team,venue,is_neutral_venue,video_source,video_quality,n_raw_penalties,n_valid_faces,n_manual_corrections,processing_time_min,processing_status,notes
m001,2021-07-03,UEFA Euro 2020,international_nt,Switzerland,Spain,Saint Petersburg,0,YouTube/UEFA,hd,10,8,3,45,qc_done,round 4 sudden death
```

`events_template.csv`：

```csv
event_id,match_id,player_id,match_date,clip_path,face_path,result,result_detail,historical_attempts,historical_goals,historical_rate,history_missing,venue_status,round_number,kick_order_in_round,sudden_death,team_score_before,opponent_score_before,team_kicks_before,opponent_kicks_before,competition_category,face_detection_score,face_quality,face_timestamp_s,exclusion_reason,source
m001-k03,p0007,2021-07-03,m001/k03.mp4,m001/k03.jpg,goal,,12,9,,0,away,4,second,0,2,3,3,3,international_nt,0.97,good,7312.4,,auto+manual_fix
m001-k05,p0012,2021-07-03,m001/k05.mp4,,miss,saved,0,0,,1,away,5,second,0,3,3,4,4,international_nt,0.62,bad,9150.1,face_too_small,auto
```

> 注意示例第二行：`historical_rate` 留空（训练时计算）；`history_missing=1`；
> 该样本 `face_quality=bad` 且无 `face_path`，保留行并记 `exclusion_reason=face_too_small`。

---

## CHANGELOG

| 日期 | 修改 | 理由 |
|---|---|---|
| 2026-09-13 | 初版创建 | Phase 0 / Action 1 |
