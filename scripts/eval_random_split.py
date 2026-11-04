#!/usr/bin/env python
"""A6/H7 随机划分对照：random event split vs match-disjoint 的配对比较。

预注册判读标准见 results/reports/random_split_criteria_preregistered.txt（跑之前写定）。

用法（项目根目录）:
    python scripts/eval_random_split.py                        # 10 seed × 5 折 + 200 次配对置换
    python scripts/eval_random_split.py --n-perm 0             # 只算观测值
    python scripts/eval_random_split.py --seeds 2027 --n-perm 20

协议:
    - random 协议：event 级 StratifiedKFold(5, shuffle, seed) 一折作 test，其余样本再
      StratifiedKFold(5, 同 seed) 第一折作 val —— 无 group 键，同场罚球可跨集合（故意泄漏同场信息）；
    - matched 协议：build_folds.compute_folds（与 eval_repeated_splits.py 同一实现）；
    - 两协议共用同一批臂、同一 C、同一 train-only 标准化（复用 eval_repeated_splits 的
      fit_predict / design_matrix，避免实现漂移）；
    - 置换：全局打乱标签 → 同一份打乱标签同时重建两套 folds → 重训两协议全部臂，
      得到配对 Δ 零分布；两协议都须建出合法划分（test/val 两类齐全）否则整体重抽。

前置断言: matched 协议观测值必须逐臂逐 seed 复现 11-04 的重复划分结果
    （results/tables/repeated_splits_20261104_141443.csv，|Δ| < 1e-6），否则中止。

输出: folds/random/seed_<s>.json（审计用）、results/tables/random_split_<stamp>.csv、
      results/reports/random_split_<stamp>.json
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
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_repeated_splits as ers  # noqa: E402  复用同一套臂实现、统计量与置换口径

TABLES_DIR = ers.TABLES_DIR
REPORTS_DIR = ers.REPORTS_DIR
CRITERIA_PATH = REPORTS_DIR / "random_split_criteria_preregistered.txt"
REPEATED_CSV = TABLES_DIR / "repeated_splits_20261104_141443.csv"
# 修正 MLP init 之后重跑的主表（determinism.mlp_init_seeded = true）。B2v2 只有对这张表
# 才能逐位断言；对上面那张历史表只能做信息性报告。
REPEATED_CSV_DET = TABLES_DIR / "repeated_splits_20261104_181311.csv"
RANDOM_FOLDS_DIR = PROJECT_ROOT / "folds" / "random"

DEFAULT_SEEDS = "2027-2036"
ARM_KEYS = ["B1v2", "B2v2", "M1", "M3v2"]          # 表 3 的四个关键臂
ARMS = [a for a in ers.ARMS if a["name"] in ARM_KEYS]
N_SPLITS = 5
N_INNER = 5
ALPHA = 0.05
MIN_SEEDS_POSITIVE = 8


def p_right(null: np.ndarray, obs: float) -> float:
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


# ---------------------------------------------------------------- 划分

def random_folds(data: dict, y: np.ndarray, seed: int) -> list[dict] | None:
    """random 协议：event 级 StratifiedKFold(5) + 内层 val；任一折缺类则返回 None。"""
    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = []
    for _, te in skf.split(idx, y):
        rest = np.setdiff1d(idx, te)
        inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
        _, va_pos = next(iter(inner.split(rest, y[rest])))
        va = rest[va_pos]
        tr = np.setdiff1d(rest, va)
        for arr in (tr, va, te):
            if len(np.unique(y[arr])) < 2:
                return None
        folds.append({"train": tr, "val": va, "test": te})
    return folds


def save_random_folds(data: dict, folds: list[dict], seed: int) -> None:
    """random folds 落盘（审计用；判读与统计不依赖它）。"""
    ids = np.array(data["ids"])
    doc = {
        "schema_version": 1,
        "strategy": "random_event_stratified_kfold",
        "group_key": None,
        "stratify_key": "result",
        "n_splits": N_SPLITS,
        "seed": seed,
        "val_assignment": f"inner StratifiedKFold({N_INNER}, seed) on non-test events, first fold",
        "note": "A6/H7 对照协议：同一场点球大战的罚球可跨 train/val/test（plan §10.3）",
        "folds": [{"fold_id": i,
                   "test_event_ids": sorted(ids[f["test"]].tolist()),
                   "val_event_ids": sorted(ids[f["val"]].tolist()),
                   "train_event_ids": sorted(ids[f["train"]].tolist())}
                  for i, f in enumerate(folds)],
    }
    RANDOM_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    (RANDOM_FOLDS_DIR / f"seed_{seed}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 跑两协议

def replicate_both(data: dict, y: np.ndarray, seeds: list[int], full: bool = False,
                   save_folds: bool = False) -> dict | None:
    """在标签 y 下跑两协议 × 全部 seed × 全部臂。

    返回 {"random": {arm: [每 seed 统计]}, "matched": {arm: [每 seed 统计]}}；
    任一 seed 在任一协议下建不出合法划分（缺类）则返回 None，由调用方重抽。
    """
    res = {"random": {a["name"]: [] for a in ARMS}, "matched": {a["name"]: [] for a in ARMS}}
    for s in seeds:
        built = ers.make_folds(data, y, s)
        folds_r = random_folds(data, y, s)
        if built is None or folds_r is None:
            return None
        doc, folds_m = built
        if save_folds:
            ers.REPEATS_DIR.mkdir(parents=True, exist_ok=True)
            (ers.REPEATS_DIR / f"seed_{s}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            save_random_folds(data, folds_r, s)
        for arm in ARMS:
            for proto, folds in (("random", folds_r), ("matched", folds_m)):
                oof = ers.run_arm(data, arm, folds, y, s)
                res[proto][arm["name"]].append(
                    ers.full_stats(oof) if full else ers.pooled_rank(oof))
    return res


def perm_worker_factory():
    """返回带 folds 缓存的置换 worker（每进程一份缓存）。

    同一置换下 random / matched 两协议的 folds 只建一次；标签向量按字节做缓存键。
    建 folds 是纯 CPU（StratifiedGroupKFold）且占了置换阶段的大头，缓存后单次置换
    成本主要落在 LR 拟合上。
    """
    cache: dict[bytes, tuple] = {}

    def worker(data: dict, seeds: list[int], perm_seed: int) -> dict:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        rng = np.random.default_rng(perm_seed)
        for attempt in range(20):
            yp = rng.permutation(data["y"])
            key = yp.tobytes()
            built = cache.get(key)
            if built is None:
                built = []
                for s in seeds:
                    bm = ers.make_folds(data, yp, s)
                    fr = random_folds(data, yp, s)
                    if bm is None or fr is None:
                        built = None
                        break
                    built.append((bm[1], fr))
                cache[key] = built
                if len(cache) > 8:          # 有限缓存，避免进程内无界增长
                    cache.pop(next(iter(cache)))
            if built is None:
                continue
            out = {"random": {a["name"]: [] for a in ARMS},
                   "matched": {a["name"]: [] for a in ARMS}}
            for (folds_m, folds_r), s in zip(built, seeds):
                for arm in ARMS:
                    for proto, folds in (("random", folds_r), ("matched", folds_m)):
                        oof = ers.run_arm(data, arm, folds, yp, s)
                        out[proto][arm["name"]].append(ers.pooled_rank(oof))
            pr = {k: {n: float(np.mean(v)) for n, v in arms.items()} for k, arms in out.items()}
            delta = {n: pr["random"][n] - pr["matched"][n] for n in pr["random"]}
            return {"pr": pr, "delta": delta, "per_seed": out, "redraws": attempt}
        raise RuntimeError(f"perm {perm_seed}: 20 次重抽都不能让两协议建出合法划分")

    return worker


# ---------------------------------------------------------------- 前置断言

def guard_matched(data: dict, obs: dict, seeds: list[int]) -> dict:
    """matched 协议观测值必须复现 11-04 的重复划分结果，否则中止。

    对照表用 `repeated_splits_20261104_181311.csv`（修正 MLP init 之后重跑的那一版，
    determinism.mlp_init_seeded = true）—— 四臂全部**硬断言**：
      - LR 臂（B1v2 / M1 / M3v2）：|Δ| < 1e-6（这三臂本来无随机性）；
      - B2v2（MLP）：|Δ| < 1e-4。init 已固定，实测跨进程/跨 joblib 配置一致到 ±5e-5
        （该量级全部来自报告 JSON 的 4 位小数舍入），留 1e-4 只为吸收浮点归约顺序差异。

    历史表 `..._141443.csv` 只作信息性对照：那一版的 MLP 臂未固定 init，旧值是一次未受控
    抽样（逐 seed 偏差最大到 0.08），不能用来断言实现一致性。
    """
    if not REPEATED_CSV_DET.exists():
        print(f"缺少对照文件 {REPEATED_CSV_DET}（修正 MLP init 后的主表），无法做前置断言",
              file=sys.stderr)
        raise SystemExit(1)
    ref = pd.read_csv(REPEATED_CSV_DET)
    ref = ref[ref["arm"].isin(ARM_KEYS)].set_index(["arm", "split_seed"])["pooled_rank"]
    old = pd.read_csv(REPEATED_CSV) if REPEATED_CSV.exists() else None
    if old is not None:
        old = old[old["arm"].isin(ARM_KEYS)].set_index(["arm", "split_seed"])["pooled_rank"]

    tol = {"B2v2": 1e-4}
    note: dict = {"reference": REPEATED_CSV_DET.relative_to(PROJECT_ROOT).as_posix(),
                  "historical_reference": (REPEATED_CSV.relative_to(PROJECT_ROOT).as_posix()
                                           if old is not None else None)}
    for arm in ARM_KEYS:
        dev = []
        for i, s in enumerate(seeds):
            if (arm, s) not in ref.index:
                print(f"对照文件缺 {arm} / seed {s}", file=sys.stderr)
                raise SystemExit(1)
            dev.append(obs["matched"][arm][i]["pooled_rank"] - float(ref.loc[(arm, s)]))
        dev = np.array(dev)
        worst = float(np.max(np.abs(dev)))
        lim = tol.get(arm, 1e-6)
        entry = {"asserted": True, "max_abs_dev": float(worst), "tol": lim}
        if old is not None:
            dev_old = np.array([obs["matched"][arm][i]["pooled_rank"] - float(old.loc[(arm, s)])
                                for i, s in enumerate(seeds)])
            entry["max_abs_dev_vs_historical"] = round(float(np.max(np.abs(dev_old))), 4)
        note[arm] = entry
        print(f"前置断言: matched/{arm} vs 修正后主表 最大偏差 {worst:.2e}（限 {lim:.0e}）"
              + (f"；对历史表偏差 {entry['max_abs_dev_vs_historical']:.4f}"
                 if old is not None else ""))
        if worst > lim:
            print(f"  !! matched/{arm} 未复现主表结果，本次实现已偏离，中止", file=sys.stderr)
            raise SystemExit(1)
    return note


# ---------------------------------------------------------------- 判读

def verdicts(obs: dict, perms: list[dict]) -> dict:
    """按预注册标准判读：每臂两协议 pooled_rank + 配对 Δ + 泄漏签名。"""
    arms_out, h7 = {}, {}
    for name in ARM_KEYS:
        r = np.array([s["pooled_rank"] for s in obs["random"][name]])
        m = np.array([s["pooled_rank"] for s in obs["matched"][name]])
        d = r - m
        entry = {
            "random_pooled_rank_mean": float(r.mean()), "random_pooled_rank_std": float(r.std()),
            "matched_pooled_rank_mean": float(m.mean()), "matched_pooled_rank_std": float(m.std()),
            "delta_mean": float(d.mean()), "delta_std": float(d.std()),
            "n_seeds_positive": int((d > 0).sum()), "n_seeds": int(len(d)),
            "per_seed_random": [round(float(x), 4) for x in r],
            "per_seed_matched": [round(float(x), 4) for x in m],
            "per_seed_delta": [round(float(x), 4) for x in d],
        }
        for k in ("fold_mean", "pooled_auprc", "pooled_brier", "pooled_logloss",
                  "pooled_balanced_acc"):
            entry[f"random_{k}_mean"] = float(np.mean([s[k] for s in obs["random"][name]]))
            entry[f"matched_{k}_mean"] = float(np.mean([s[k] for s in obs["matched"][name]]))
        entry["n_test"] = obs["random"][name][0]["n_test"]
        if perms:
            nr = np.array([p["pr"]["random"][name] for p in perms])
            nm = np.array([p["pr"]["matched"][name] for p in perms])
            nd = np.array([p["delta"][name] for p in perms])
            entry.update({
                "random_null_mean": float(nr.mean()), "random_null_p95": float(np.percentile(nr, 95)),
                "random_p": p_right(nr, entry["random_pooled_rank_mean"]),
                "matched_null_mean": float(nm.mean()), "matched_null_p95": float(np.percentile(nm, 95)),
                "matched_p": p_right(nm, entry["matched_pooled_rank_mean"]),
                "delta_null_mean": float(nd.mean()), "delta_p": p_right(nd, entry["delta_mean"]),
                "delta_null_std": float(nd.std()),
            })
            entry["random_has_signal"] = entry["random_p"] < ALPHA
            entry["matched_has_signal"] = entry["matched_p"] < ALPHA
            h7[name] = bool(entry["n_seeds_positive"] >= MIN_SEEDS_POSITIVE
                            and entry["delta_p"] < ALPHA and entry["delta_mean"] > 0)
            if entry["random_has_signal"] and not entry["matched_has_signal"]:
                entry["leak_signature"] = "随机划分下的表观信号可由同场信息共享解释"
            elif entry["random_has_signal"] and entry["matched_has_signal"]:
                entry["leak_signature"] = "两协议下都显著（稳健效应，不能写成泄漏）"
            else:
                entry["leak_signature"] = "两种划分下均无信号"
        arms_out[name] = entry
    n_hold = int(sum(h7.values())) if perms else None
    return {"arms": arms_out, "h7_arm_holds": h7,
            "h7_holds_overall": (n_hold >= 3) if perms else None,
            "h7_n_arms_holding": n_hold}


# ---------------------------------------------------------------- 主流程

def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS, help="split seed，如 2027-2036 或 2026,2027")
    ap.add_argument("--n-perm", type=int, default=200, help="配对置换次数（0=跳过）")
    ap.add_argument("--perm-seed", type=int, default=12345)
    ap.add_argument("--n-jobs", type=int, default=-2)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    seeds = parse_seeds(args.seeds)
    data = ers.load_data()
    print(f"有效样本 {len(data['ids'])}，split seeds {seeds}，{len(ARMS)} 臂 × 2 协议")

    t0 = time.time()
    obs = replicate_both(data, data["y"], seeds, full=True, save_folds=True)
    if obs is None:
        print("真实标签下划分缺类，检查 build_folds / random_folds", file=sys.stderr)
        return 1
    print(f"观测值完成 {time.time() - t0:.0f}s")
    guard = guard_matched(data, obs, seeds)

    for name in ARM_KEYS:
        r = np.array([s["pooled_rank"] for s in obs["random"][name]])
        m = np.array([s["pooled_rank"] for s in obs["matched"][name]])
        print(f"  {name:>6}  random {r.mean():.4f}±{r.std():.4f}   "
              f"matched {m.mean():.4f}±{m.std():.4f}   Δ {(r - m).mean():+.4f}")

    perms: list[dict] = []
    if args.n_perm > 0:
        t1 = time.time()
        # 每进程一份 folds 缓存：LokyBackend 会在 worker 初始化时 pickle 该函数闭包
        worker = perm_worker_factory()
        perms = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(worker)(data, seeds, args.perm_seed + i) for i in range(args.n_perm))
        print(f"置换 {args.n_perm} 次完成 {time.time() - t1:.0f}s，"
              f"重抽总数 {sum(p['redraws'] for p in perms)}")

    v = verdicts(obs, perms)

    print(f"\n{'arm':>6} | {'random':>9} | {'matched':>9} | {'Δ':>8} | "
          f"{'seed>0':>7} | {'p_Δ':>6} | H7 | 泄漏签名")
    for name, e in v["arms"].items():
        print(f"{name:>6} | {e['random_pooled_rank_mean']:>9.4f} | "
              f"{e['matched_pooled_rank_mean']:>9.4f} | {e['delta_mean']:>+8.4f} | "
              f"{e['n_seeds_positive']:>3}/{e['n_seeds']:<3} | "
              f"{e.get('delta_p', float('nan')):>6.3f} | "
              f"{str(v['h7_arm_holds'].get(name, '-')):>3} | {e.get('leak_signature', '-')}")
    print(f"\nH7 整体: {v['h7_holds_overall']}（{v['h7_n_arms_holding']}/4 臂成立，判定线 ≥3）")
    for name, e in v["arms"].items():
        print(f"  {name}: random p={e.get('random_p', float('nan')):.3f} "
              f"(零分布 {e.get('random_null_mean', float('nan')):.4f}/p95 "
              f"{e.get('random_null_p95', float('nan')):.4f})；"
              f"matched p={e.get('matched_p', float('nan')):.3f} "
              f"(零分布 {e.get('matched_null_mean', float('nan')):.4f}/p95 "
              f"{e.get('matched_null_p95', float('nan')):.4f})")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for proto in ("random", "matched"):
        for name in ARM_KEYS:
            for s, st in zip(seeds, obs[proto][name]):
                rows.append({"protocol": proto, "arm": name, "split_seed": s, **st})
    table = TABLES_DIR / f"random_split_{stamp}.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "A6/H7：random event split vs match-disjoint 的配对对照",
        "criteria": CRITERIA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_seeds": seeds, "n_perm": args.n_perm, "perm_seed": args.perm_seed,
        "perm_redraws_total": sum(p["redraws"] for p in perms),
        "n_valid_events": len(data["ids"]),
        "guard": guard,
        "arms_spec": ARMS, "random_folds_dir": RANDOM_FOLDS_DIR.relative_to(PROJECT_ROOT).as_posix(),
        **v,
        "per_seed_table": table.relative_to(PROJECT_ROOT).as_posix(),
    }
    path = REPORTS_DIR / f"random_split_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n表:   {table.relative_to(PROJECT_ROOT).as_posix()}\n"
          f"报告: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

