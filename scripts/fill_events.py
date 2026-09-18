#!/usr/bin/env python
"""events.csv 自动填充脚本：输入每踢的 result，推导其余全部 context 字段。

用法（项目根目录）:
    python scripts/fill_events.py --interactive                    # 逐场逐踢询问 result
    python scripts/fill_events.py --interactive --match m004       # 只处理一场
    python scripts/fill_events.py --results m001:gmmsgmm           # 已整理好结果串时用
    python scripts/fill_events.py --validate                       # 只做一致性检查

--results 结果串格式: 每踢一个字符，按踢序:
    g = goal   m = miss   . = 跳过（该踢保持未填）
    例: --results m001:gmmsgmm, m004:mgmgmgg

推导规则（与 DATA_DICTIONARY.md 一致，严格交替 A→B→A→B）:
    kick_order_in_round   全局踢序 i 奇数 -> first（先罚方），偶数 -> second
    round_number          ceil((i + first_kick_offset) / 2)，突然死亡从 6 递增
    sudden_death          round >= 6
    kicks_taken_before    (i - 1) // 2
    team_score_before     本队此前所有 result=g 的个数
    opponent_score_before 对手此前所有 result=g 的个数
    history_*             本脚本不填（留空待人工调查后填），仅做校验不覆盖

先罚方通过 --interactive 的询问确定（每场问一次），或默认 events.csv 中
该场第一踢已有的 kick_order_in_round；两者都无则按第一踢 = first。

写回 events.csv 只动推导列与 result 列；exclusion_reason 等既有内容不覆盖。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
MATCHES_PATH = PROJECT_ROOT / "data" / "metadata" / "matches.csv"

EVENT_COLUMNS = ["event_id", "match_id", "contact_time_s", "face_closeup_s", "face_path",
                 "player", "result", "round_number", "kick_order_in_round", "sudden_death",
                 "team_score_before", "opponent_score_before", "kicks_taken_before",
                 "history_attempts", "history_goals", "history_missing", "exclusion_reason"]

RESULT_CHAR = {"g": "goal", "m": "miss"}


def load_events() -> list[dict]:
    if not EVENTS_PATH.exists():
        print(f"缺 events 表: {EVENTS_PATH}", file=sys.stderr)
        raise SystemExit(1)
    with open(EVENTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EVENT_COLUMNS:
            print(f"events.csv 列顺序与预期不符: {reader.fieldnames}", file=sys.stderr)
            raise SystemExit(1)
        return list(reader)


def group_by_match(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:  # events.csv 已按 match_id + 踢序排列
        groups.setdefault(r["match_id"], []).append(r)
    return groups


def ask_results(match_id: str, kicks: list[dict]) -> list[str] | None:
    """交互式询问一场的各踢结果。返回与 kicks 等长的 result 列表（''=跳过），None=放弃本场。"""
    print(f"\n=== {match_id} ({len(kicks)} 踢) ===")
    print("逐踢输入结果: g=进球  m=未进  回车=跳过该踢  s=本场剩余全跳  a=放弃本场  x=退出脚本\n")
    out: list[str] = []
    for r in kicks:
        excl = f"（排除: {r['exclusion_reason']}）" if r["exclusion_reason"] else ""
        while True:
            raw = input(f"  {r['event_id']} @ {float(r['contact_time_s']):7.2f}s{excl} -> ").strip().lower()
            if raw in ("g", "m"):
                out.append(RESULT_CHAR[raw])
                break
            if raw == "":  # 跳过该踢
                out.append("")
                break
            if raw == "s":
                out.extend([""] * (len(kicks) - len(out)))
                return out
            if raw == "a":
                return None
            if raw == "x":
                raise SystemExit(0)
            print("    无效输入，请输入 g / m / 回车 / s / a / x")
    return out


def parse_results_arg(spec: str, groups: dict[str, list[dict]]) -> dict[str, list[str]]:
    """解析 --results 的 m001:gmmsgmm 列表，校验长度。"""
    plans: dict[str, list[str]] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        match_id, _, seq = item.partition(":")
        match_id = match_id.strip()
        if match_id not in groups:
            print(f"--results 中出现未知比赛: {match_id}", file=sys.stderr)
            raise SystemExit(1)
        seq = seq.strip().lower()
        if len(seq) != len(groups[match_id]):
            print(f"{match_id}: 结果串长度 {len(seq)} != 踢数 {len(groups[match_id])}", file=sys.stderr)
            raise SystemExit(1)
        if any(ch not in ".gm" for ch in seq):
            print(f"{match_id}: 结果串含非法字符（只允许 g m .）", file=sys.stderr)
            raise SystemExit(1)
        plans[match_id] = [RESULT_CHAR[c] if c in RESULT_CHAR else "" for c in seq]
    return plans


def derive(results: list[str], first_kick_offset: int) -> list[dict]:
    """按踢序推导轮次/比分字段。results[i] in {goal, miss, ''}。"""
    derived: list[dict] = []
    first_goals = second_goals = 0
    n_taken_first = n_taken_second = 0
    for i, res in enumerate(results, 1):
        idx0 = i - 1  # 0-based
        order = "first" if i % 2 == 1 else "second"
        rnd = math.ceil((i + first_kick_offset) / 2)
        team_score = first_goals if order == "first" else second_goals
        opp_score = second_goals if order == "first" else first_goals
        derived.append({
            "round_number": rnd,
            "kick_order_in_round": order,
            "sudden_death": 1 if rnd >= 6 else 0,
            "team_score_before": team_score,
            "opponent_score_before": opp_score,
            "kicks_taken_before": n_taken_first if order == "first" else n_taken_second,
        })
        if res:  # 跳过的踢不计入比分/踢数（要求本场要么全填要么明确跳过）
            if order == "first":
                first_goals += res == "goal"
                n_taken_first += 1
            else:
                second_goals += res == "goal"
                n_taken_second += 1
    return derived


def write_back(all_rows: list[dict]) -> None:
    tmp = EVENTS_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    tmp.replace(EVENTS_PATH)


def validate(rows: list[dict], matches: dict[str, dict] | None = None) -> int:
    """一致性检查（不修改文件）。返回问题数。"""
    problems = 0
    for match_id, kicks in group_by_match(rows).items():
        n = len(kicks)
        # 对照 matches.csv 的官方罚球总数——漏标一踢会让全部推导错位，必须先发现
        if matches and match_id in matches:
            n_official = matches[match_id].get("n_raw_penalties", "").strip()
            if n_official and n_official.isdigit() and int(n_official) != n:
                print(f"  !! {match_id}: markers {n} 踢 != matches.csv n_raw_penalties={n_official}（疑漏标/多标）")
                problems += 1
        filled = [k["result"] for k in kicks]
        if any(filled) and not all(filled):
            print(f"  !! {match_id}: result 部分填写（{sum(1 for x in filled if x)}/{n}）")
            problems += 1
        for i, k in enumerate(kicks, 1):
            order = "first" if i % 2 == 1 else "second"
            if k["kick_order_in_round"] and k["kick_order_in_round"] != order:
                print(f"  !! {match_id} {k['event_id']}: kick_order={k['kick_order_in_round']} 与踢序矛盾")
                problems += 1
            if k["round_number"] and int(k["round_number"]) != math.ceil(i / 2):
                print(f"  !! {match_id} {k['event_id']}: round={k['round_number']} 与踢序矛盾")
                problems += 1
        # 比分一致性重演
        fg = sg = 0
        for i, k in enumerate(kicks, 1):
            order = "first" if i % 2 == 1 else "second"
            ts = fg if order == "first" else sg
            os_ = sg if order == "first" else fg
            if k["team_score_before"] and int(k["team_score_before"]) != ts:
                print(f"  !! {match_id} {k['event_id']}: team_score={k['team_score_before']} != 推导 {ts}")
                problems += 1
            if k["opponent_score_before"] and int(k["opponent_score_before"]) != os_:
                print(f"  !! {match_id} {k['event_id']}: opponent_score={k['opponent_score_before']} != 推导 {os_}")
                problems += 1
            if k["result"]:
                if order == "first":
                    fg += k["result"] == "goal"
                else:
                    sg += k["result"] == "goal"
        # 无 result 时比分/踢数不应有值（会被推导覆盖，但提示）
        if not any(filled) and any(k["team_score_before"] or k["kicks_taken_before"] for k in kicks):
            print(f"  !! {match_id}: 无 result 却已有比分/踢数（建议重新推导）")
            problems += 1
        # n_raw_penalties 对照
        print(f"  {match_id}: {n} 踢, result 已填 {sum(1 for x in filled if x)}")
    return problems


def load_matches() -> dict[str, dict]:
    """读 matches.csv（日期格式不统一，此处只取 id/罚球数，不做日期解析）。"""
    if not MATCHES_PATH.exists():
        return {}
    with open(MATCHES_PATH, newline="", encoding="utf-8") as f:
        return {r["match_id"]: r for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interactive", action="store_true", help="逐场逐踢询问 result")
    ap.add_argument("--results", help="结果串 m001:gmmsgmm,m004:mgmgmgg（g/m/.）")
    ap.add_argument("--match", help="--interactive 时只处理指定比赛")
    ap.add_argument("--validate", action="store_true", help="只做一致性检查，不修改文件")
    args = ap.parse_args()

    rows = load_events()
    groups = group_by_match(rows)
    print(f"载入 {len(rows)} 行，{len(groups)} 场")

    if args.validate:
        problems = validate(rows, load_matches())
        print(f"\n{'通过' if problems == 0 else f'{problems} 个问题'}")
        return 0 if problems == 0 else 1

    plans: dict[str, list[str]] = {}
    if args.results:
        plans = parse_results_arg(args.results, groups)
    elif args.interactive:
        if not sys.stdin.isatty():
            print("--interactive 需要在终端中直接运行（python scripts/fill_events.py）", file=sys.stderr)
            return 1
        targets = [args.match] if args.match else list(groups)
        for mid in targets:
            if mid not in groups:
                print(f"未知比赛: {mid}", file=sys.stderr)
                return 1
            if all(k["result"] for k in groups[mid]):
                print(f"{mid}: 已全部填写，跳过")
                continue
            got = ask_results(mid, groups[mid])
            if got is not None and any(got):
                plans[mid] = got
    else:
        print("请指定 --interactive 或 --results 或 --validate", file=sys.stderr)
        return 1

    # 应用计划：重演推导并写回
    n_changed = 0
    for match_id, results in plans.items():
        kicks = groups[match_id]
        # 先罚偏移：若第一踢已标 second，则本场以第二罚开表（罕见，需人工确认）
        pre = kicks[0].get("kick_order_in_round", "")
        first_kick_offset = 1 if pre == "second" else 0
        if first_kick_offset:
            print(f"  !! {match_id}: 第一踢标为 second，按第二轮开局推导（请人工确认）")
        derived = derive(results, first_kick_offset)
        for k, res, d in zip(kicks, results, derived):
            new_vals = dict(d)
            if res:
                new_vals["result"] = res
            elif not res and k["result"]:
                new_vals["result"] = k["result"]  # 跳过的踢保留既有 result
            changed = any(str(new_vals[c] if new_vals[c] != "" else "") != k[c] for c in new_vals)
            if changed:
                n_changed += 1
            k.update({c: str(v) if v != "" else "" for c, v in new_vals.items()})
        print(f"  {match_id}: 已推导 {sum(1 for r in results if r)}/{len(results)} 踢")

    if n_changed or plans:
        write_back(rows)
        print(f"\n已写回 {EVENTS_PATH.relative_to(PROJECT_ROOT)}（改动 {n_changed} 行）")
        print("提醒: history_attempts/history_goals/history_missing 仍需人工调查后填写")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
