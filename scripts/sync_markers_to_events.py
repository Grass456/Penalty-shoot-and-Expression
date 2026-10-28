#!/usr/bin/env python
"""kick_markers -> events.csv 同步脚本：为已有 marker 文件的比赛补建 events 行。

用法（项目根目录）:
    python scripts/sync_markers_to_events.py            # 全量同步（幂等）
    python scripts/sync_markers_to_events.py --match m015
    python scripts/sync_markers_to_events.py --dry-run  # 只预览，不写回

对每个 data/metadata/kick_markers/<match_id>_markers.csv:
  - events.csv 中已有该场踢 -> 跳过已有 kick_seq，只追加缺失的行；
    若某踢 event_id 已存在但时间不一致则报错（markers 被改动，人工处理）
  - 该场无任何行 -> 全部追加

追加行的 event_id 为 <match_id>-<kick_seq 小写>（如 m015-k01），只填
match_id / contact_time_s / face_closeup_s，其余字段（result、推导列、
history_*、exclusion_reason）留空，交给 fill_events.py 与人工填写。
不触碰 events.csv 中已存在且 marker 里没有的行（如被排除的踢）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
MARKERS_DIR = PROJECT_ROOT / "data" / "metadata" / "kick_markers"

EVENT_COLUMNS = ["event_id", "match_id", "contact_time_s", "face_closeup_s", "face_path",
                 "result", "round_number", "kick_order_in_round", "sudden_death",
                 "team_score_before", "opponent_score_before", "kicks_taken_before",
                 "history_attempts", "history_goals", "history_missing", "exclusion_reason"]

TIME_TOL = 1e-6  # 秒级时间戳比较容差


def load_events() -> tuple[list[dict], dict[str, dict]]:
    if not EVENTS_PATH.exists():
        print(f"缺 events 表: {EVENTS_PATH}", file=sys.stderr)
        raise SystemExit(1)
    with open(EVENTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EVENT_COLUMNS:
            print(f"events.csv 列顺序与预期不符: {reader.fieldnames}", file=sys.stderr)
            raise SystemExit(1)
        rows = list(reader)
    return rows, {r["event_id"]: r for r in rows}


def load_markers(match_id: str) -> list[dict]:
    path = MARKERS_DIR / f"{match_id}_markers.csv"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"kick_seq", "contact_time_s", "face_closeup_s"}
        if not required <= set(reader.fieldnames or ()):
            print(f"{path.name}: 缺必需列 {sorted(required)}", file=sys.stderr)
            raise SystemExit(1)
        return [
            {
                "kick_seq": r["kick_seq"].strip(),
                "contact_time_s": r["contact_time_s"].strip(),
                "face_closeup_s": r["face_closeup_s"].strip(),
            }
            for r in reader
        ]


def same_time(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) <= TIME_TOL
    except ValueError:
        return a == b  # 非数值则按字面比较


def new_row(match_id: str, m: dict) -> dict:
    return {c: "" for c in EVENT_COLUMNS} | {
        "event_id": f"{match_id}-{m['kick_seq'].lower()}",
        "match_id": match_id,
        "contact_time_s": m["contact_time_s"],
        "face_closeup_s": m["face_closeup_s"],
    }


def sync_match(match_id: str, markers: list[dict], existing: dict[str, dict],
               match_rows: list[dict], dry_run: bool) -> int:
    """为一场同步 markers -> events。返回追加行数，冲突时退出。"""
    added = 0
    for m in markers:
        eid = f"{match_id}-{m['kick_seq'].lower()}"
        if eid in existing:
            row = existing[eid]
            if not same_time(row["contact_time_s"], m["contact_time_s"]):
                print(f"  !! {match_id} {eid}: events contact_time={row['contact_time_s']} "
                      f"!= markers {m['contact_time_s']}（markers 被改动？请人工处理）", file=sys.stderr)
                raise SystemExit(1)
            if row["face_closeup_s"] and not same_time(row["face_closeup_s"], m["face_closeup_s"]):
                print(f"  !! {match_id} {eid}: face_closeup 不一致，跳过不改", file=sys.stderr)
            continue
        if dry_run:
            print(f"  + {eid} @ {m['contact_time_s']}s (face @ {m['face_closeup_s']}s)")
        else:
            match_rows.append(new_row(match_id, m))
            existing[eid] = match_rows[-1]
        added += 1
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", help="只处理指定比赛（默认全部 marker 文件）")
    ap.add_argument("--dry-run", action="store_true", help="只预览将要追加的行，不写回")
    args = ap.parse_args()

    if args.match:
        marker_files = [MARKERS_DIR / f"{args.match}_markers.csv"]
        if not marker_files[0].exists():
            print(f"无 marker 文件: {marker_files[0]}", file=sys.stderr)
            return 1
    else:
        marker_files = sorted(MARKERS_DIR.glob("*_markers.csv"))
        if not marker_files:
            print(f"目录无 marker 文件: {MARKERS_DIR}", file=sys.stderr)
            return 1

    rows, by_event_id = load_events()
    by_match: dict[str, list[dict]] = {}
    for r in rows:
        by_match.setdefault(r["match_id"], []).append(r)

    total_added = 0
    for path in marker_files:
        match_id = path.name.removesuffix("_markers.csv")
        markers = load_markers(match_id)
        added = sync_match(match_id, markers, by_event_id, by_match.setdefault(match_id, []), args.dry_run)
        total_added += added
        if added and not args.dry_run:
            by_match[match_id].sort(key=lambda r: r["event_id"])  # 保持 event_id 排序
        print(f"{match_id}: markers {len(markers)} 踢, {'将' if args.dry_run else ''}追加 {added}")

    if args.dry_run:
        print(f"\n--dry-run：未修改文件，共 {total_added} 行待追加")
        return 0
    if total_added == 0:
        print("events.csv 已同步，无需改动")
        return 0

    tmp = EVENTS_PATH.with_suffix(".csv.tmp")
    # 从 by_match 重建（sync_match 把新行追加到了 by_match 的组列表中）
    all_rows = [r for mid in sorted(by_match) for r in by_match[mid]]
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    tmp.replace(EVENTS_PATH)
    print(f"\n已写回 {EVENTS_PATH.relative_to(PROJECT_ROOT)}（追加 {total_added} 行）")
    print("提醒: 新行其余字段留空，请先运行 fill_events.py 填 result 与推导列，"
          "再人工填 history_* 与 face_path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
