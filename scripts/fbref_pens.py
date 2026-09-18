#!/usr/bin/env python
"""FBref 点球历史工作流：球员页 HTML（人工浏览器保存）→ pens 收录 → 历史列自动填充。

背景（2026-09-18 定义变更，已记入 DATA_DICTIONARY CHANGELOG）:
    history_attempts/goals 的定义从"世界杯点球大战"改为"最近一个已完成的
    俱乐部赛季的联赛点球"（FBref pens_att/pens_made）。赛季映射（严格早于
    match_date 的最近完整赛季，防 temporal leakage）:
        2018 世界杯场次 -> 2017-18 赛季
        2014 -> 2013-14；2010 -> 2009-10；2006 -> 2005-06
        2022 世界杯（11-12 月，赛季中）-> 2021-22 赛季
    口径 = FBref Dom League 标准表（联赛，不含杯赛/欧战/国家队）。

为何两段式（本地无法直连 FBref）:
    本机直连外网 403/超时。工作流:
    用户浏览器打开球员 FBref 页 -> Ctrl+S 存 HTML 到 data/external/fbref_pages/ ->
    跑本脚本 --ingest 解析。HTML 快照落 gitignored 目录，不入库。

用法（项目根目录）:
    python scripts/fbref_pens.py --ingest    # 解析 fbref_pages/*.html -> pens_data.csv
    python scripts/fbref_pens.py --list      # 查看已收录球员/赛季与覆盖情况
    python scripts/fbref_pens.py --todo      # 生成待查球员 x 赛季清单
    python scripts/fbref_pens.py --fill      # 按 player+match_date 填 history_* 三列

    --ingest 按 <Player>_<season>.html 命名解析，如 "Lionel Messi_2021-2022.html"。
    --fill 需要 events.csv 有 player 列（events.csv 现无此列，--todo 会提示先生成）。

输出:
    data/external/pens_data.csv   每行: player, season, pens_att, pens_made,
                                  league, source_file, captured_date
    data/external/pens_todo.csv   待查清单: player, seasons_needed

球员名单从 events.csv 的 player 列读取（需先填）。

页面解析说明:
    打开的应是球员【主导航】页（fbref.com/en/players/.../<Name>），其默认
    Standard Stats 表每行一个 (season, squad, country) 组合：season 列形如
    "2021-2022"（俱乐部行）与 "2021 Argentina"（国家��行）。解析规则:
    season 文本精确匹配 SEASONS 映射值；line 上的 squad 限 Dom League 俱乐部行。
    FBref 的表格常被 HTML 注释包裹（转播延迟反爬），沿用社区通用的
    "剥注释再解析" 处理。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = PROJECT_ROOT / "data" / "external" / "fbref_pages"
PENS_CSV = PROJECT_ROOT / "data" / "external" / "pens_data.csv"
TODO_CSV = PROJECT_ROOT / "data" / "external" / "pens_todo.csv"
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
MATCHES_PATH = PROJECT_ROOT / "data" / "metadata" / "matches.csv"

# 世界杯年份 -> 对应"严格早于 match_date 的最近完整赛季"（FBref season 文本）
SEASONS = {2006: "2005-2006", 2010: "2009-2010", 2014: "2013-2014", 2018: "2017-2018",
           2022: "2021-2022"}

EVENT_COLUMNS = ["event_id", "match_id", "contact_time_s", "face_closeup_s", "face_path",
                 "player", "result", "round_number", "kick_order_in_round", "sudden_death",
                 "team_score_before", "opponent_score_before", "kicks_taken_before",
                 "history_attempts", "history_goals", "history_missing", "exclusion_reason"]

PENS_COLUMNS = ["player", "season", "pens_att", "pens_made", "league",
                "source_file", "captured_date"]


def season_for_date(d: date) -> str:
    """match_date -> 最近一个已结束赛季的 FBref 文本（严格早于比赛日期）。"""
    wc_year = d.year if d.month >= 6 else d.year - 1
    if wc_year not in SEASONS:
        print(f"  !! 未映射的世界杯年份 {wc_year}，SEASONS 需扩充", file=sys.stderr)
        return ""
    return SEASONS[wc_year]


def load_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        print(f"缺 {EVENTS_PATH}", file=sys.stderr)
        raise SystemExit(1)
    return pd.read_csv(EVENTS_PATH, dtype=str).fillna("")


def strip_fbref_comments(html: str) -> str:
    """FBref 把表格藏在 HTML 注释里（性能反爬），剥掉即可见。"""
    return re.sub(r"<!--|-->", "", html)


def parse_player_page(html_path: Path, expect_season: str) -> dict | None:
    """从球员页 HTML 提取目标赛季的联赛行。返回 pens 记录或 None。

    球员页 Standard Stats 表: 每行一个赛季，th[data-stat=season] 的文本
    形如 "2021-2022"（俱乐部）或 "2021 Argentina"（国家队）。取精确匹配行。
    """
    from bs4 import BeautifulSoup

    raw = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(strip_fbref_comments(raw), "lxml")

    # 球员主导航页的标准表 id=stats_standard_dom_lg（Dom League 表）；
    # 兼容旧版 id（stats_standard_dom_lg 不存在时退回全部 standard 表按行筛）
    table = soup.find("table", id="stats_standard_dom_lg")
    tables = [table] if table else soup.find_all("table", id=re.compile(r"stats_standard"))

    for tbl in tables:
        for row in tbl.find_all("tr"):
            season_th = row.find("th", {"data-stat": "season"})
            if season_th is None:
                continue
            if season_th.get_text(strip=True) != expect_season:
                continue
            att_cell = row.find("td", {"data-stat": "pens_att"})
            made_cell = row.find("td", {"data-stat": "pens_made"})
            if att_cell is None or made_cell is None:
                continue
            att_txt, made_txt = att_cell.get_text(strip=True), made_cell.get_text(strip=True)
            # 空单元格 = 该赛季联赛 0 次点球（合法值，非缺失）
            att = int(att_txt) if att_txt.isdigit() else 0
            made = int(made_txt) if made_txt.isdigit() else 0
            lg_cell = row.find("td", {"data-stat": "comp_level"}) or row.find("td", {"data-stat": "league_name"})
            league = lg_cell.get_text(strip=True) if lg_cell else ""
            squad_cell = row.find("td", {"data-stat": "squad"})
            squad = squad_cell.get_text(strip=True) if squad_cell else ""
            return {"player": html_path.stem.rsplit("_", 1)[0],
                    "season": expect_season,
                    "pens_att": att, "pens_made": made,
                    "league": league or squad,
                    "source_file": html_path.name,
                    "captured_date": date.today().isoformat()}
    return None


def cmd_ingest() -> int:
    if not PAGES_DIR.exists() or not list(PAGES_DIR.glob("*.html")):
        print(f"fbref_pages/ 无 HTML。浏览器打开球员页后存到 {PAGES_DIR}/，"
              f"文件名 <Player>_<season>.html（如 'Lionel Messi_2021-2022.html'）")
        return 1

    existing: dict[tuple[str, str], dict] = {}
    if PENS_CSV.exists():
        with open(PENS_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[(r["player"], r["season"])] = r

    added, updated, failed = 0, 0, 0
    for page in sorted(PAGES_DIR.glob("*.html")):
        stem = page.stem
        if "_" not in stem:
            print(f"  !! {page.name}: 文件名需为 <Player>_<season>.html，跳过")
            failed += 1
            continue
        player, _, season = stem.rpartition("_")
        rec = parse_player_page(page, season)
        if rec is None:
            print(f"  !! {page.name}: 未找到 {season} 联赛行（确认存的是球员主导航页？）")
            failed += 1
            continue
        key = (rec["player"], rec["season"])
        if key in existing:
            updated += 1
        else:
            added += 1
        existing[key] = rec
        print(f"  {rec['player']} {rec['season']}: att={rec['pens_att']} made={rec['pens_made']} ({rec['league']})")

    with open(PENS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PENS_COLUMNS)
        w.writeheader()
        w.writerows(existing.values())
    print(f"\n收录完成: 新增 {added}, 更新 {updated}, 失败 {failed} -> {PENS_CSV.relative_to(PROJECT_ROOT)}")
    return 0 if failed == 0 else 1


def cmd_list() -> int:
    if not PENS_CSV.exists():
        print("pens_data.csv 尚不存在，先 --ingest")
        return 1
    df = pd.read_csv(PENS_CSV, dtype=str).fillna("")
    print(df[["player", "season", "pens_att", "pens_made", "league"]].to_string(index=False))
    return 0


def needed_players() -> dict[str, set[str]]:
    """events.csv player 列 + matches.csv match_date -> 每个球员需要的赛季集合。"""
    ev, ms = load_events(), pd.read_csv(MATCHES_PATH, dtype=str).fillna("")
    if "player" not in ev.columns:
        print("events.csv 尚无 player 列。先运行 scripts/add_player_column.py 生成。",
              file=sys.stderr)
        raise SystemExit(1)
    ev = ev[ev["player"] != ""]
    ms_date = dict(zip(ms["match_id"], ms["match_date"]))
    out: dict[str, set[str]] = {}
    for _, r in ev.iterrows():
        mid = r["match_id"]
        if mid not in ms_date or not ms_date[mid]:
            continue
        d = pd.to_datetime(ms_date[mid], format="mixed").date()
        s = season_for_date(d)
        if s:
            out.setdefault(r["player"], set()).add(s)
    return out


def cmd_todo() -> int:
    need = needed_players()
    rows = [{"player": p, "seasons_needed": ",".join(sorted(ss))} for p, ss in sorted(need.items())]
    TODO_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TODO_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["player", "seasons_needed"])
        w.writeheader()
        w.writerows(rows)
    # 覆盖统计
    have: set[tuple[str, str]] = set()
    if PENS_CSV.exists():
        with open(PENS_CSV, newline="", encoding="utf-8") as f:
            have = {(r["player"], r["season"]) for r in csv.DictReader(f)}
    n_need = sum(len(ss) for ss in need.values())
    n_have = sum(1 for p, ss in need.items() for s in ss if (p, s) in have)
    print(f"待查球员 {len(need)} 人，(球员, 赛季) 组合 {n_need} 个，已收录 {n_have} -> {TODO_CSV.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_fill() -> int:
    if not PENS_CSV.exists():
        print("pens_data.csv 尚不存在，先 --ingest", file=sys.stderr)
        return 1
    ev = load_events()
    if "player" not in ev.columns:
        print("events.csv 尚无 player 列，无法按球员填充。", file=sys.stderr)
        return 1
    ms = pd.read_csv(MATCHES_PATH, dtype=str).fillna("")
    ms_date = dict(zip(ms["match_id"], ms["match_date"]))

    pens: dict[tuple[str, str], dict] = {}
    with open(PENS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pens[(r["player"], r["season"])] = r

    n_filled = n_missing = 0
    for _, r in ev.iterrows():
        if not r["player"]:
            n_missing += 1
            continue
        d = pd.to_datetime(ms_date.get(r["match_id"], ""), format="mixed", errors="coerce")
        if pd.isna(d):
            continue
        season = season_for_date(d.date())
        rec = pens.get((r["player"], season))
        if rec is None:
            # 查不到 = 数据源无覆盖，按新定义填 missing=1、att/goal=0
            r["history_attempts"], r["history_goals"], r["history_missing"] = "0", "0", "1"
            n_missing += 1
        else:
            r["history_attempts"] = rec["pens_att"]
            r["history_goals"] = rec["pens_made"]
            r["history_missing"] = "0"
            n_filled += 1

    tmp = EVENTS_PATH.with_suffix(".csv.tmp")
    ev.to_csv(tmp, index=False)
    tmp.replace(EVENTS_PATH)
    print(f"history_* 填充完成: {n_filled} 行有数据, {n_missing} 行 missing -> {EVENTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ingest", action="store_true", help="解析 fbref_pages/*.html 收录到 pens_data.csv")
    ap.add_argument("--list", action="store_true", help="查看已收录数据")
    ap.add_argument("--todo", action="store_true", help="生成待查清单")
    ap.add_argument("--fill", action="store_true", help="填充 events.csv 的 history_* 三列")
    args = ap.parse_args()

    if args.ingest:
        return cmd_ingest()
    if args.list:
        return cmd_list()
    if args.todo:
        return cmd_todo()
    if args.fill:
        return cmd_fill()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
