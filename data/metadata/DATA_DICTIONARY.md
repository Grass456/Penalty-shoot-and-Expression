# 数据字典（简化版）

> 对应 plan.md §6.6–6.8，2026-09-14 按课程项目规模简化。
> 简化原则：只保留**易获取 + 高价值**的字段；plan 中其余字段（venue、competition_category、
> result_detail、player_id 等）如后续需要可在扩展阶段加回，列定义见 git 历史。
> 修改字段定义必须记入文末 CHANGELOG。

两张表：

- `matches_template.csv` — 每场比赛一行；
- `events_template.csv` — 每次罚球一行（建模主表）。

ID 风格：`m001`（比赛）、`m001-k03`（m001 场第 3 踢）。分配后不复用。
球员姓名不入库（伦理约束）；本版本**不设 player_id 列**，历史表现在按"球员姓名→本地映射表"人工核对
（映射表放 `data/external/`，gitignored），player-disjoint 分析留到扩展阶段再引入。

---

## 1. matches 表（比赛级，11 列）

| 字段 | 类型 | 必填 | 定义 |
|---|---|---|---|
| `match_id` | str | ✅ | 比赛唯一编号 `m001`…。match-disjoint 划分的 group 键 |
| `match_date` | date `YYYY-MM-DD` | ✅ | 比赛日期。**历史统计时间截断依据** |
| `competition` | str | ✅ | 赛事原名，如 `FIFA World Cup 2022`。pilot 全是世界杯，暂不做类别映射 |
| `home_team` / `away_team` | str | ✅ | 对阵双方（用于查历史和核对记录） |
| `video_source` | str | ✅ | 来源，如 `YouTube/<channel>`。视频本身不入库 |
| `video_quality` | enum | ✅ | `hd`（≥720p）/ `sd`。之后对比人脸可用率用 |
| `n_raw_penalties` | int | ✅ | 该场罚球数（= events 表行数，脚本可校验） |
| `n_valid_faces` | int | ✅ | 有人脸且未排除的样本数 |
| `processing_time_min` | float | ✅ | 该场人工处理分钟数（估算扩展成本） |
| `notes` | str | | 自由备注 |

**已删**（相比初版）：`venue`、`is_neutral_venue`、`home/away_team` 保留但不再派生
主客场编码、`competition_category`、`n_manual_corrections`、`processing_status`。

---

## 2. events 表（罚球级，15 列）

### 2.1 标识与文件

| 字段 | 类型 | 必填 | 定义 |
|---|---|---|---|
| `event_id` | str | ✅ | `m001-k03`。同一次罚球不得重复入库 |
| `match_id` | str | ✅ | 外键 → matches 表 |
| `contact_time_s` | float | ✅ | **触球时刻**（原视频秒）。防泄漏硬上界：切片终点永不越过它前 0.5s |
| `face_closeup_s` | float | 建议 | **特写时刻**（原视频秒，marker.py 的 f 键打点）。切片窗口锚点：切片 = [特写−8s, min(特写+6s, 触球−0.5s)]；漏打时脚本回退 45s 旧窗口 |
| `face_path` | str | 选帧后填 | 相对 `data/faces/`，如 `m001/k03.jpg`。被排除样本留空 |
| `exclusion_reason` | enum | 排除时必填 | 见 §2.4 |

> `clip_path`、`face_timestamp_s`、`face_detection_score`、`face_quality`、`source` 均已删：
> 切片路径由 `match_id/event_id` 约定俗成，无需入库；质量字段脚本打在**独立审计文件**
> `data/metadata/face_audit.csv`（脚本生成，可随时重建），不进人工维护的主表。

### 2.2 标签

| 字段 | 类型 | 必填 | 定义 |
|---|---|---|---|
| `result` | enum | ✅ | `goal` / `miss`（含被扑、射偏、门框）。以官方记录为准 |
| `result_detail` | **已删** | | 探索性分析用，课程项目不必要 |

### 2.3 Context（五类 → 精简为四类，主客场整类暂缓）

**比分状态**（plan §6.6.4，罚球前，不含本次结果）：

| 字段 | 类型 | 必填 | 定义 |
|---|---|---|---|
| `team_score_before` | int | ✅ | 本队本场点球大战已进球数 |
| `opponent_score_before` | int | ✅ | 对手已进球数 |
| `kicks_taken_before` | int | ✅ | 本队已完成罚球数（不区分对手侧，简化） |

**轮次**（plan §6.6.3）：

| 字段 | 类型 | 必填 | 定义 |
|---|---|---|---|
| `round_number` | int | ✅ | 第几轮，从 1 起；突然死亡继续递增 |
| `kick_order_in_round` | enum | ✅ | `first` / `second`（本轮先罚/后罚方） |
| `sudden_death` | bool 0/1 | ✅ | 是否突然死亡轮次 |

**历史点球表现**（plan §6.6.1，pilot 简化：只统计世界杯决赛圈点球大战，可查到几场算几场）：

| 字段 | 类型 | 必填 | 定义 |
|---|---|---|---|
| `history_attempts` | int | ✅ | 此前（严格早于本场日期）世界杯点球大战主罚次数；查不到填 0 |
| `history_goals` | int | ✅ | 此前世界杯点球大战进球数；查不到填 0 |
| `history_missing` | bool 0/1 | ✅ | 是否缺少可靠历史（`attempts==0` 即 1）。Beta 平滑率训练时由脚本从这两列计算，**不入库** |

> 注意：世界杯范围内查不到 ≠ 球员没踢过点球（俱乐部历史缺失），这正是 `history_missing`
> 要如实标记的原因（plan §6.6.1）。主客场（plan §6.6.2）整类**暂缓**：世界杯多为中立场，
> 该特征方差极低，扩展阶段引入非中立场数据时再加回。

### 2.4 排除记录

| 值 | 含义 |
|---|---|
| `no_face` | 触球前切片无可用人脸 |
| `face_too_small` | 人脸低于最小像素尺寸 |
| `occluded` | 眉/嘴严重遮挡 |
| `blurry` | 候选帧全部模糊 |
| `after_kick_only` | 只有触球后画面 |
| `wrong_person` | 画面中不是主罚球员 |
| `leakage_risk` | 画面含比分牌/庆祝/失望反应 |
| `video_issue` | 视频缺失、掐头去尾导致该踢不完整 |

被排除样本**保留整行并填原因**。

---

## 3. 保留的硬性约束（简化不豁免）

1. `contact_time_s` 必须准确——所有防泄漏规则的锚点；
2. 比分/轮次字段只含**罚球前**状态；
3. 历史统计截止日严格早于 `match_date`；
4. 人脸图像严格来自触球前；排除样本保留行；
5. 同场所有罚球只能在 train/val/test 之一（match_id 为 group）。

---

## 4. 单场标注流程（简化后）

1. 看视频，每次罚球记一行：`event_id`、`contact_time_s`、`result`、轮次/比分状态（~15-20 min）；
2. 跑脚本自动切片段 + 检测人脸 + 选帧（写 `face_audit.csv`）；
3. 快速过一遍选出的图，确认是主罚球员、无泄漏画面（~1 min）；
4. 查 Wikipedia 补历史表现列（~10 min）；
5. 填 matches 表的统计列。

---

## CHANGELOG

| 日期 | 修改 | 理由 |
|---|---|---|
| 2026-09-13 | 初版（26+16 列，五类 context 全量） | Phase 0 |
| 2026-09-14 | **简化**：events 26→15 列，matches 16→11 列；删 venue/主客场、competition_category、player_id、result_detail、clip_path 等质量与流程字段；历史统计限定世界杯范围；质量字段移至脚本生成的 face_audit.csv | 课程项目规模；只保留易获取且高价值的字段 |
| 2026-09-16 | **双打点方案**：marker.py 增加 f 键（特写点），每踢记录 face_closeup_s + contact_time_s 两个时间戳；切片改为 [特写−8s, min(特写+6s, 触球−0.5s)]（~14s/踢），漏打特写点回退 45s 旧窗口；events 表加 face_closeup_s 列 | 转播特写时刻因球员而异（有的准备 >45s），人工打点直接锁定特写比固定窗口可靠；触球点保留为防泄漏硬上界 |
| 2026-09-15 | 切片窗口标定：`pad_before=45s`（起点 = max(上一踢触球, 本次触球−45s)，终点 = 触球前 0.5s）。实测 m001/k02：15s 窗口 0 帧合格脸，45s 窗口 97 帧 | 转播面部特写多在踢前 20-40s（走回罚球点/放球阶段），15s 只覆盖助跑远景 |
