#!/usr/bin/env python
"""events.csv 加 player 列（一次性迁移脚本）。

背景: 2026-09-18 history 定义变更后必须按球员统计，events 表需要 player 列。
原设计无 player 列是伦理决定（姓名不入库）；现改为 player 列写真实姓名，但
data/metadata/ 本就不入库（gitignored），与 matches 表存对阵双方的处理一致，
公开发布时只发处理代码与非 PII 元数据（硬约束 8）。

用法（项目根目录）:
    python scripts/add_player_column.py            # 插入 player 列（幂等）
    python scripts/add_player_column.py --remove   # 撤销（发布前如需抹名可用）

player 列插入在 face_path 之后、result 之前。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"

OLD_COLUMNS = ["event_id", "match_id", "contact_time_s", "face_closeup_s", "face_path",
               "result", "round_number", "kick_order_in_round", "sudden_death",
               "team_score_before", "opponent_score_before", "kicks_taken_before",
               "history_attempts", "history_goals", "history_missing", "exclusion_reason"]
NEW_COLUMNS = ["event_id", "match_id", "contact_time_s", "face_closeup_s", "face_path",
               "player", "result", "round_number", "kick_order_in_round", "sudden_death",
               "team_score_before", "opponent_score_before", "kicks_taken_before",
               "history_attempts", "history_goals", "history_missing", "exclusion_reason"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remove", action="store_true", help="删除 player 列（恢复 16 列）")
    args = ap.parse_args()

    if not EVENTS_PATH.exists():
        print(f"缺 {EVENTS_PATH}", file=sys.stderr)
        return 1
    with open(EVENTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)

    if args.remove:
        if "player" not in cols:
            print("player 列不存在，无需操作")
            return 0
        new_cols = [c for c in cols if c != "player"]
        new_rows = [{k: v for k, v in r.items() if k != "player"} for r in rows]
    else:
        if "player" in cols:
            print("player 列已存在，无需操作")
            return 0
        if cols != OLD_COLUMNS:
            print(f"events.csv 列不符预期: {cols}", file=sys.stderr)
            return 1
        new_cols = NEW_COLUMNS
        new_rows = [{**r, "player": ""} for r in rows]

    tmp = EVENTS_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_cols)
        w.writeheader()
        w.writerows(new_rows)
    tmp.replace(EVENTS_PATH)
    print(f"完成: {len(new_rows)} 行, {len(new_cols)} 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
