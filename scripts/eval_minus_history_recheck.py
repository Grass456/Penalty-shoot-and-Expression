#!/usr/bin/env python
"""A5-minus_history 独立复核（Frozen protocol 2037-2046）。

判读标准见 results/reports/minus_history_recheck_criteria_preregistered.txt（跑之前写定）。

用法（项目根目录）:
    python scripts/eval_minus_history_recheck.py               # 观测 + 200 次配对置换 + 判读
    python scripts/eval_minus_history_recheck.py --n-perm 0    # 只算观测值

协议（与 eval_context_ablation.py 逐行平行，只换 seed 集合与前置断言）:
    - 变体 full / minus_history；C=1.0，train-only 标准化，p̄ 由本折 train 估计（硬约束 5）；
    - split seed 2037-2046（与 A5 的 2027-2036 不相交），由 ers.make_folds 生成，不落盘到 folds/；
    - 前置断言（相对，不依赖落盘的主表 CSV）：full 在本 seed 集合上必须与
      eval_repeated_splits.ARMS 的 B1v2 同协议实现逐位一致（|Δ| ≤ 1e-12），
      确保本脚本的消融实现没有偏离主表；
    - 主判定（预注册 C1-C4）：
        C1 minus_history 相对零分布 p < 0.05；
        C2 Δ = minus_history − full > 0 且 ≥ 8/10 seed 为正；
        C3 Δ 的配对置换 p_Δ < 0.05；
        C4 minus_history 的 pooled_rank 均值 > 零分布 p95。
      四条全过才写"成立"，否则写"单一 seed 集合上的偏高，独立 seed 集合未复现"。

输出: results/reports/minus_history_recheck_<stamp>.json
      results/tables/minus_history_recheck_<stamp>.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_repeated_splits as ers  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPEATS_CSV = TABLES_DIR / "repeated_splits_20261104_181311.csv"
CRITERIA_PATH = REPORTS_DIR / "minus_history_recheck_criteria_preregistered.txt"
DEFAULT_SEEDS = "2037-2046"

FULL_COLS = ["sudden_death", "is_trailing", "kick_order_second", "score_diff",
             "has_history", "history_rate"]
DROP_HISTORY = ["has_history", "history_rate"]
GROUPS = {"history": DROP_HISTORY}
PRIMARY = ["full", "minus_history"]
ALPHA = 0.05
MIN_SEEDS_POSITIVE = 8
B1V2 = next(a for a in ers.ARMS if a["name"] == "B1v2")


def cols_for(variant: str) -> list[str]:
    if variant == "full":
        return list(FULL_COLS)
    if variant == "minus_history":
        return [c for c in FULL_COLS if c not in DROP_HISTORY]
    raise ValueError(f"未知变体 {variant}")


def p_right(null: np.ndarray, obs: float) -> float:
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


def run_variant(data: dict, cols: list[str], folds: list[dict], y: np.ndarray) -> list[tuple]:
    """与 eval_context_ablation.run_variant 相同实现（同一函数体，C=1.0，无随机性）。"""
    from sklearn.linear_model import LogisticRegression

    out = []
    for fold in folds:
        tr, te = fold["train"], fold["test"]
        fc = data["cfg"]["features"]
        ctx = ers.build_context_matrix_v2(data["valid"], p_bar=float(y[tr].mean()),
                                          alpha=fc["history_alpha"], clip=fc["score_diff_clip"])
        X = ctx[cols].to_numpy(dtype=np.float64)
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        lr = LogisticRegression(max_iter=5000, C=1.0, random_state=0)
        lr.fit(Z[tr], y[tr])
        out.append((te, y[te], lr.predict_proba(Z[te])[:, 1]))
    return out


def run_b1v2_ref(data: dict, folds: list[dict], y: np.ndarray, seed: int) -> list[tuple]:
    """前置断言用的主表 B1v2 实现（ers.fit_predict，走 design_matrix 的 ctx 块）。"""
    out = []
    for fold in folds:
        out.append((fold["test"], y[fold["test"]],
                    ers.fit_predict(data, B1V2, fold, y, seed)))
    return out


def replicate(data: dict, y: np.ndarray, seeds: list[int], full: bool = False) -> dict | None:
    res: dict[str, list] = {v: [] for v in PRIMARY}
    for s in seeds:
        built = ers.make_folds(data, y, s)
        if built is None:
            return None
        _, folds = built
        for v in PRIMARY:
            oof = run_variant(data, cols_for(v), folds, y)
            res[v].append(ers.full_stats(oof) if full else ers.pooled_rank(oof))
    return res


def perm_worker(data: dict, seeds: list[int], perm_seed: int) -> dict:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(perm_seed)
    for attempt in range(20):
        yp = rng.permutation(data["y"])
        r = replicate(data, yp, seeds)
        if r is not None:
            return {"pr": {k: float(np.mean(v)) for k, v in r.items()},
                    "per_seed": {k: [float(x) for x in v] for k, v in r.items()},
                    "redraws": attempt}
    raise RuntimeError(f"perm {perm_seed}: 20 次重抽都缺类")


def guard(data: dict, seeds: list[int]) -> dict:
    """full（本脚本实现）vs 主表 B1v2（ers 实现），用 2037-2046 中共同的 seed 对照。

    历史主表只覆盖 2026-2036；2037 起没有 B1v2 落盘值，故分两步：
    (1) 结构等价：full 与 B1v2 的列顺序 / C 相同（断言）；
    (2) 数值等价：在复算 seed 2027-2036 上逐位对照落盘 CSV（|Δ| ≤ 1e-12）。
    """
    assert cols_for("full") == list(B1V2_COLS), "full 列顺序与 main B1v2 不一致"
    assert B1V2["C"] == 1.0
    ref_seeds = list(range(2027, 2037))
    ref = pd.read_csv(REPEATS_CSV)
    ref = ref[(ref["arm"] == "B1v2") & (ref["split_seed"].isin(ref_seeds))]
    ref = ref.set_index("split_seed")["pooled_rank"]
    dev = []
    for s in ref_seeds:
        _, folds = ers.make_folds(data, data["y"], s)
        mine = ers.pooled_rank(run_variant(data, cols_for("full"), folds, data["y"]))
        theirs = ers.pooled_rank(run_b1v2_ref(data, folds, data["y"], s))
        dev += [mine - float(ref.loc[s]), theirs - float(ref.loc[s])]
    worst = float(np.max(np.abs(dev)))
    print(f"前置断言: full vs 落盘 B1v2（seed 2027-2036），最大偏差 {worst:.2e}（限 1e-12）")
    if worst > 1e-12:
        print("  !! full 未复现主表 B1v2，中止", file=sys.stderr)
        raise SystemExit(1)
    return {"method": "与落盘 repeated_splits_20261104_181311.csv 的 B1v2 逐位对照（seed 2027-2036）",
            "reference": REPEATS_CSV.relative_to(PROJECT_ROOT).as_posix(),
            "max_abs_dev": worst, "n_checked": len(ref_seeds)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--perm-seed", type=int, default=12345)
    ap.add_argument("--n-jobs", type=int, default=-2)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    seeds = ers.parse_seeds(args.seeds)
    if set(seeds) & set(range(2027, 2037)):
        print("  !! seed 集合与 A5（2027-2036）重叠，违反预注册第 2 节", file=sys.stderr)
        return 1
    data = ers.load_data()
    print(f"有效样本 {len(data['ids'])}，split seeds {seeds}，变体 {PRIMARY}")

    t0 = time.time()
    obs = replicate(data, data["y"], seeds, full=True)
    if obs is None:
        print("真实标签下划分缺类", file=sys.stderr)
        return 1
    print(f"观测值完成 {time.time() - t0:.0f}s")
    guard_info = guard(data, seeds)

    perms: list[dict] = []
    if args.n_perm > 0:
        t1 = time.time()
        perms = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(perm_worker)(data, seeds, args.perm_seed + i) for i in range(args.n_perm))
        print(f"置换 {args.n_perm} 次完成 {time.time() - t1:.0f}s，"
              f"重抽总数 {sum(p['redraws'] for p in perms)}")

    out = {}
    for v in PRIMARY:
        pr = np.array([s["pooled_rank"] for s in obs[v]])
        e = {"pooled_rank_mean": float(pr.mean()), "pooled_rank_std": float(pr.std()),
             "pooled_rank_min": float(pr.min()), "pooled_rank_max": float(pr.max()),
             "per_seed_pooled_rank": [round(float(x), 4) for x in pr],
             "cols": cols_for(v)}
        for k in ("fold_mean", "pooled_auprc", "pooled_brier", "pooled_logloss",
                  "pooled_balanced_acc"):
            e[f"{k}_mean"] = float(np.mean([s[k] for s in obs[v]]))
        if perms:
            null = np.array([p["pr"][v] for p in perms])
            e.update({"null_mean": float(null.mean()), "null_std": float(null.std()),
                      "null_p95": float(np.percentile(null, 95)), "p": p_right(null, e["pooled_rank_mean"])})
        out[v] = e

    comp = {"a": "minus_history", "b": "full"}
    da = np.array([s["pooled_rank"] for s in obs["minus_history"]])
    db = np.array([s["pooled_rank"] for s in obs["full"]])
    d = da - db
    comp.update({"delta_mean": float(d.mean()), "delta_std": float(d.std()),
                 "n_seeds_positive": int((d > 0).sum()), "n_seeds": int(len(d)),
                 "per_seed_delta": [round(float(x), 4) for x in d]})
    if perms:
        null = np.array([np.mean(np.array(p["per_seed"]["minus_history"]) - np.array(p["per_seed"]["full"]))
                         for p in perms])
        comp.update({"null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                     "p_delta": p_right(null, comp["delta_mean"])})
    else:
        comp["p_delta"] = None

    # ---- 预注册 C1-C4 ----
    c1 = bool(perms and out["minus_history"]["p"] < ALPHA)
    c2 = bool(comp["delta_mean"] > 0 and comp["n_seeds_positive"] >= MIN_SEEDS_POSITIVE)
    c3 = bool(perms and comp["p_delta"] is not None and comp["p_delta"] < ALPHA)
    c4 = bool(perms and out["minus_history"]["pooled_rank_mean"] > out["minus_history"]["null_p95"])
    holds = bool(c1 and c2 and c3 and c4)
    verdict = ("成立（C1-C4 全过）：minus_history 经独立 seed 集合复核后仍成立"
               if holds else "未复现：minus_history 是单一 seed 集合上的偏高，独立 seed 集合未复现，A5 线索结案为噪声")

    print(f"\n{'变体':>16} | {'pooled_rank':>12} | {'null_mean':>9} | {'p95':>6} | {'p':>6}")
    for v, e in out.items():
        print(f"{v:>16} | {e['pooled_rank_mean']:.4f}±{e['pooled_rank_std']:.3f} | "
              f"{e.get('null_mean', float('nan')):>9.4f} | {e.get('null_p95', float('nan')):>6.4f} | "
              f"{e.get('p', float('nan')):>6.3f}")
    print(f"Δ = minus_history − full: {comp['delta_mean']:+.4f}±{comp['delta_std']:.4f}  "
          f"{comp['n_seeds_positive']}/{comp['n_seeds']} seed>0  p_Δ={comp['p_delta'] if comp['p_delta'] is None else round(comp['p_delta'], 3)}")
    print(f"C1={c1}  C2={c2}  C3={c3}  C4={c4}  →  {verdict}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [{"variant": n, "split_seed": s, **st} for n, lst in obs.items() for s, st in zip(seeds, lst)]
    table = TABLES_DIR / f"minus_history_recheck_{stamp}.csv"
    pd.DataFrame(rows).to_csv(table, index=False, encoding="utf-8")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "A5 minus_history 独立复核：新 seed 集合 2037-2046，检验 0.563 / p=0.020 是否复现",
        "criteria": CRITERIA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_seeds": seeds, "n_perm": args.n_perm, "perm_seed": args.perm_seed,
        "perm_redraws_total": sum(p["redraws"] for p in perms),
        "n_valid_events": len(data["ids"]),
        "guard": guard_info,
        "variants": out, "comparison": comp,
        "criteria_check": {"C1_p_lt_0.05": c1, "C2_delta_pos_and_8of10": c2,
                           "C3_p_delta_lt_0.05": c3, "C4_above_null_p95": c4, "holds": holds},
        "verdict": verdict,
        "a5_reference": {"report": "results/reports/context_ablation_20261104_182831.json",
                         "seeds": "2027-2036", "minus_history_pooled_rank": 0.563, "p": 0.020},
        "per_seed_table": table.relative_to(PROJECT_ROOT).as_posix(),
    }
    path = REPORTS_DIR / f"minus_history_recheck_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n表:   {table.relative_to(PROJECT_ROOT).as_posix()}\n报告: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


B1V2_COLS = FULL_COLS   # ctx 块在 design_matrix 中的列顺序（build_context_matrix_v2 输出）

if __name__ == "__main__":
    raise SystemExit(main())
