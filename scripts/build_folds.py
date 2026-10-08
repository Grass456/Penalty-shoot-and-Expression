#!/usr/bin/env python
"""生成 match-disjoint folds（pilot 阶段：leave-one-match-out）。

用法（项目根目录）:
    python scripts/build_folds.py             # 生成/覆盖 folds/match_disjoint.json
    python scripts/build_folds.py --validate  # 校验现有 folds 与 events.csv 是否一致

划分设计（2026-10-08 定，pilot 14 场）:
    - group = match_id（硬约束 4）：同一场点球大战的所有罚球只出现在一个集合；
    - test  = 1 场完整点球大战（leave-one-match-out，共 14 折）；
    - val   = 循环中的下一场 matches[(test_idx + 1) % n]，完全确定性、无随机种子，
              每场比赛在全部折中恰好担任一次 val；
    - train = 其余 12 场；
    - 有效样本 = exclusion_reason 为空且 face_path 非空（与特征缓存口径一致）；
    - test 折必须 goal/miss 两类齐全，否则该折 AUROC 不可计算——生成时强制中止。

folds 文件被所有模型共享（硬约束 4）；early stopping 与选模型只用 val（硬约束 6）。
划分完全由 events.csv 决定：events.csv 变更后必须重新生成本文件
（下游可用 --validate 发现 folds 过期）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
MATCHES_PATH = PROJECT_ROOT / "data" / "metadata" / "matches.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"

SCHEMA_VERSION = 1


# ---------------------------------------------------------------- 载入与口径

def load_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        print(f"缺 events 表: {EVENTS_PATH}", file=sys.stderr)
        raise SystemExit(1)
    ev = pd.read_csv(EVENTS_PATH)
    need = {"event_id", "match_id", "face_path", "result", "exclusion_reason"}
    miss = need - set(ev.columns)
    if miss:
        print(f"events.csv 缺列: {sorted(miss)}", file=sys.stderr)
        raise SystemExit(1)
    dup = ev.loc[ev["event_id"].duplicated(), "event_id"].tolist()
    if dup:
        print(f"event_id 重复: {dup}", file=sys.stderr)
        raise SystemExit(1)
    return ev


def valid_mask(ev: pd.DataFrame) -> pd.Series:
    """有效样本 = 未排除且有 face_path（与 cache_fer_features.py 口径一致）。"""
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    bad = ev.loc[excl != noface, "event_id"].tolist()
    if bad:
        print(f"  !! exclusion_reason 与 face_path 空缺不一致: {bad}", file=sys.stderr)
    return ~excl & ~noface


def stats_for(ids: list[str], valid: pd.DataFrame) -> dict:
    sub = valid[valid["event_id"].isin(ids)]
    return {
        "n_events": int(len(sub)),
        "n_goal": int((sub["result"] == "goal").sum()),
        "n_miss": int((sub["result"] == "miss").sum()),
    }


# ---------------------------------------------------------------- 划分计算

def compute_folds(valid: pd.DataFrame) -> dict:
    """确定性 LOMO：test=1 场，val=循环下一场，train=其余 12 场。"""
    matches = sorted(valid["match_id"].unique())
    n = len(matches)
    if n < 3:
        print(f"比赛数过少（{n}），无法划分 train/val/test", file=sys.stderr)
        raise SystemExit(1)
    ids_by_match = {
        m: sorted(valid.loc[valid["match_id"] == m, "event_id"].tolist()) for m in matches
    }

    folds = []
    for i, test_m in enumerate(matches):
        val_m = matches[(i + 1) % n]
        train_m = [m for m in matches if m != test_m and m != val_m]
        ids = {
            "test": ids_by_match[test_m],
            "val": ids_by_match[val_m],
            "train": sorted(e for m in train_m for e in ids_by_match[m]),
        }
        st = {k: stats_for(v, valid) for k, v in ids.items()}
        # test 缺类则该折 AUROC 不可算，直接中止；val 缺类只警告（early stopping 质量下降）
        for name in ("test", "val"):
            if st[name]["n_goal"] == 0 or st[name]["n_miss"] == 0:
                msg = (f"{name} 折 {test_m}: 缺类 "
                       f"(goal={st[name]['n_goal']}, miss={st[name]['n_miss']})")
                if name == "test":
                    print(f"  !! {msg}，AUROC 不可算", file=sys.stderr)
                    raise SystemExit(1)
                print(f"  -- {msg}（val 缺类，early stopping 仅供参考）", file=sys.stderr)
        folds.append({
            "fold_id": i,
            "test_matches": [test_m],
            "val_matches": [val_m],
            "train_matches": train_m,
            "test_event_ids": ids["test"],
            "val_event_ids": ids["val"],
            "train_event_ids": ids["train"],
            "stats": st,
        })

    n_excluded = int((~valid_mask_cache).sum()) if valid_mask_cache is not None else None
    doc = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "leave_one_match_out",
        "group_key": "match_id",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "val_assignment": "cyclic: val = matches[(test_index + 1) % n_matches]",
        "source": {
            "events": EVENTS_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "valid_filter": "exclusion_reason 为空且 face_path 非空",
        },
        "global_stats": {
            "n_matches": n,
            "n_valid_events": int(len(valid)),
            **stats_for(valid["event_id"].tolist(), valid),
            "miss_rate": round(float((valid["result"] == "miss").mean()), 4),
        },
        "folds": folds,
    }
    return doc


# ---------------------------------------------------------------- 一致性检查

def check_matches_table(ev: pd.DataFrame, valid: pd.DataFrame) -> list[str]:
    """对照 matches.csv 的 n_raw_penalties / n_valid_faces（缺值则跳过该场）。"""
    problems: list[str] = []
    if not MATCHES_PATH.exists():
        return problems
    m = pd.read_csv(MATCHES_PATH)
    n_raw_all = ev.groupby("match_id").size().to_dict()
    n_valid = valid.groupby("match_id").size().to_dict()
    for _, row in m.iterrows():
        mid = row["match_id"]
        raw = row.get("n_raw_penalties")
        if pd.notna(raw):
            if int(raw) != n_raw_all.get(mid, 0):
                problems.append(f"{mid}: events 行数 {n_raw_all.get(mid, 0)} != n_raw_penalties={int(raw)}")
        vf = row.get("n_valid_faces")
        if pd.notna(vf):
            if int(vf) != n_valid.get(mid, 0):
                problems.append(f"{mid}: 有效样本 {n_valid.get(mid, 0)} != n_valid_faces={int(vf)}")
    return problems


def check_doc(doc: dict, valid: pd.DataFrame) -> list[str]:
    """结构不变量：match 互斥、event 归属正确、覆盖完整、统计一致。"""
    problems: list[str] = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version={doc.get('schema_version')} != {SCHEMA_VERSION}")
    id2match = dict(zip(valid["event_id"], valid["match_id"]))
    seen_test: list[str] = []
    for fold in doc["folds"]:
        fid = fold["fold_id"]
        tm = set(fold["test_matches"])
        vm = set(fold["val_matches"])
        trm = set(fold["train_matches"])
        if tm & vm or tm & trm or vm & trm:
            problems.append(f"fold {fid}: train/val/test 按 match 相交（违反硬约束 4）")
        for key, mset in (("test_event_ids", tm), ("val_event_ids", vm), ("train_event_ids", trm)):
            ids = fold[key]
            if len(set(ids)) != len(ids):
                problems.append(f"fold {fid} {key}: event_id 重复")
            wrong = [e for e in ids if id2match.get(e) not in mset]
            if wrong:
                problems.append(f"fold {fid} {key}: event 的 match 不在所属集合: {wrong[:5]}")
            recomputed = stats_for(ids, valid)
            if fold["stats"][key.removesuffix("_event_ids")] != recomputed:
                problems.append(f"fold {fid} {key}: stats {fold['stats'][key.removesuffix('_event_ids')]} != 重算 {recomputed}")
        seen_test.extend(fold["test_event_ids"])
    # 覆盖完整性：每个有效样本恰好在一条折的 test 里
    if sorted(seen_test) != sorted(id2match):
        only_file = set(id2match) - set(seen_test)
        only_seen = set(seen_test) - set(id2match)
        problems.append(f"test 覆盖不完整: 缺 {sorted(only_file)[:5]} / 多 {sorted(only_seen)[:5]}")
    if len(seen_test) != len(set(seen_test)):
        problems.append("同一 event 出现在多条折的 test 中")
    return problems


def compare_docs(a: dict, b: dict) -> list[str]:
    """deep 比较（忽略 created_at），用于 --validate 检测 folds 过期。"""
    diffs: list[str] = []
    a, b = {k: v for k, v in a.items() if k != "created_at"}, {k: v for k, v in b.items() if k != "created_at"}
    if a == b:
        return diffs
    if a.get("folds") != b.get("folds"):
        for fa, fb in zip(a.get("folds", []), b.get("folds", [])):
            if fa != fb:
                diffs.append(f"fold {fa.get('fold_id')}: 与按当前 events.csv 重算结果不一致（folds 过期）")
        if len(a.get("folds", [])) != len(b.get("folds", [])):
            diffs.append(f"折数 {len(a.get('folds', []))} != 重算 {len(b.get('folds', []))}")
    for k in a:
        if k != "folds" and a.get(k) != b.get(k):
            diffs.append(f"{k}: {a.get(k)} != 重算 {b.get(k)}")
    return diffs


# ---------------------------------------------------------------- 主流程

def print_summary(doc: dict) -> None:
    g = doc["global_stats"]
    print(f"共 {g['n_matches']} 场 / {g['n_valid_events']} 有效样本 "
          f"(goal {g['n_goal']} / miss {g['n_miss']}, miss 率 {g['miss_rate']:.1%})，{doc['n_folds'] if 'n_folds' in doc else len(doc['folds'])} 折 LOMO\n")
    print(f"{'fold':>4}  {'test':>5} {'n(g/m)':>8}  {'val':>5}  {'train n(g/m)':>12}")
    for f in doc["folds"]:
        t, v, r = f["stats"]["test"], f["stats"]["val"], f["stats"]["train"]
        print(f"{f['fold_id']:>4}  {f['test_matches'][0]:>5} {t['n_goal']:>3}/{t['n_miss']:<4} "
              f"{f['val_matches'][0]:>5}  {r['n_goal']:>4}/{r['n_miss']:<4}  ({r['n_events']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="校验现有 folds 与 events.csv 一致，不写文件")
    args = ap.parse_args()

    global valid_mask_cache  # noqa: PLW0603 —— 仅供 compute_folds 统计排除数
    ev = load_events()
    mask = valid_mask(ev)
    valid_mask_cache = mask
    valid = ev[mask].copy()
    bad_result = set(valid["result"].unique()) - {"goal", "miss"}
    if bad_result:
        print(f"  !! result 含未知值: {bad_result}", file=sys.stderr)
        raise SystemExit(1)

    doc = compute_folds(valid)

    if args.validate:
        if not FOLDS_PATH.exists():
            print(f"缺 folds 文件: {FOLDS_PATH}", file=sys.stderr)
            return 1
        with open(FOLDS_PATH, encoding="utf-8") as f:
            on_disk = json.load(f)
        problems = check_doc(on_disk, valid) + compare_docs(on_disk, doc)
        if problems:
            print("folds 与当前 events.csv 不一致：")
            for p in problems:
                print(f"  !! {p}")
            print("→ 请重新运行 python scripts/build_folds.py 重新生成")
            return 1
        print(f"folds 与 events.csv 一致（{len(doc['folds'])} 折 LOMO）")
        return 0

    problems = check_doc(doc, valid) + check_matches_table(ev, valid)
    if problems:
        for p in problems:
            print(f"  !! {p}", file=sys.stderr)
        raise SystemExit(1)

    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print_summary(doc)
    print(f"\n已写入 {FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


valid_mask_cache = None

if __name__ == "__main__":
    raise SystemExit(main())