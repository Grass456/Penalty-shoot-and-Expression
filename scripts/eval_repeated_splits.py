#!/usr/bin/env python
"""重复划分评估：10 组 split seed × 5 折，主臂 + 待复查臂 + 全流程 refit 置换零分布。

判读标准见 results/reports/repeated_splits_criteria_preregistered.txt（跑之前写定）。

用法（项目根目录）:
    python scripts/eval_repeated_splits.py                 # seed 2027-2036，200 次置换
    python scripts/eval_repeated_splits.py --n-perm 0      # 只算观测值
    python scripts/eval_repeated_splits.py --seeds 2026    # 单 seed（与 build_folds 默认划分一致）

协议:
    - 每个 split seed 用 build_folds.compute_folds 生成 5 折 + 内层 val，写入 folds/repeats/；
    - 所有臂共用同一组 folds（硬约束 4）；标准化 / p̄ / LR 只在 train 上 fit（硬约束 5）；
      B2v2 MLP 的 early stopping 只看 val（硬约束 6）；C 全部事先固定；
    - 356 子集臂（多帧）把每折 event 收缩到子集，与 eval_multiframe.py 口径一致；
    - 主统计量 pooled_rank（折内转秩后 pool）取 seed 均值；
    - 零分布：全局打乱标签 → 同 seed 重建 folds → 重训全部臂（所有臂共用同一次打乱，
      因此配对差也有零分布）。

输出: folds/repeats/seed_<s>.json、results/tables/repeated_splits_<stamp>.csv、
      results/reports/repeated_splits_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
# loky 子进程按模块名反序列化 run_baselines / build_folds，需要同样的搜索路径
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(PROJECT_ROOT), str(SCRIPTS_DIR)] + [x for x in [os.environ.get("PYTHONPATH")] if x])

import build_folds  # noqa: E402
from run_baselines import build_context_matrix_v2  # noqa: E402

FEATURES_DIR = PROJECT_ROOT / "features" / "fer"
REPEATS_DIR = PROJECT_ROOT / "folds" / "repeats"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "context_v2.yaml"
CRITERIA_PATH = REPORTS_DIR / "repeated_splits_criteria_preregistered.txt"

DEFAULT_SEEDS = "2027-2036"
MULTI_CACHE = "multiframe_mean_enet_b0_8_best_vgaf"
MULTI_VA_CACHE = "multiframe_mean_enet_b0_8_va_mtl"

# 预注册臂（与 criteria 文件逐项对应；C 事先固定）
ARMS: list[dict] = [
    {"name": "B1v2", "model": "lr", "feats": ["ctx"], "C": 1.0, "subset": False},
    {"name": "B2v2", "model": "mlp", "feats": ["ctx"], "C": None, "subset": False},
    {"name": "B3_vgaf", "model": "lr", "feats": ["emo:enet_b0_8_best_vgaf"], "C": 0.01, "subset": False},
    {"name": "B3_b2_7", "model": "lr", "feats": ["emo:enet_b2_7"], "C": 0.01, "subset": False},
    {"name": "M1", "model": "lr", "feats": ["emb:enet_b0_8_best_vgaf"], "C": 0.01, "subset": False},
    {"name": "B4", "model": "lr", "feats": ["emb:imagenet_b0"], "C": 0.01, "subset": False},
    {"name": "M3v2", "model": "lr", "feats": ["ctx", "emb:enet_b0_8_best_vgaf"], "C": 0.01, "subset": False},
    {"name": "VAemb", "model": "lr", "feats": ["emb:enet_b0_8_va_mtl"], "C": 0.01, "subset": False},
    {"name": "single356", "model": "lr", "feats": ["emb:enet_b0_8_best_vgaf"], "C": 1.0, "subset": True},
    {"name": "multimean", "model": "lr", "feats": [f"emb:{MULTI_CACHE}"], "C": 1.0, "subset": True},
    {"name": "multiva2", "model": "lr", "feats": [f"va:{MULTI_VA_CACHE}"], "C": 0.01, "subset": True},
]
COMPARISONS = [("M1", "B4"), ("M3v2", "B1v2"), ("VAemb", "M1"), ("B2v2", "B1v2"),
               ("multimean", "single356")]
FLAGGED = {"VAemb": ("VAemb", "M1"), "B2v2": ("B2v2", "B1v2"), "multimean": ("multimean", "single356")}
ALPHA = 0.05
MIN_SEEDS_POSITIVE = 8


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


# ---------------------------------------------------------------- 数据

def load_data() -> dict:
    """有效样本（按 event_id 排序）+ 所有臂用到的特征块，行顺序与 valid 对齐。

    356 子集缓存补成 385 行（缺失行为 NaN），子集臂只会用到子集行。
    """
    ev = build_folds.load_events()
    valid = ev[build_folds.valid_mask(ev)].sort_values("event_id").reset_index(drop=True)
    ids = valid["event_id"].tolist()
    pos = {e: i for i, e in enumerate(ids)}

    blocks: dict[str, np.ndarray] = {}
    subset_sets: list[set[str]] = []
    for key in sorted({f for a in ARMS for f in a["feats"] if f != "ctx"}):
        kind, cache = key.split(":")
        c = np.load(FEATURES_DIR / cache / "embeddings.npz", allow_pickle=True)
        cids = c["event_ids"].astype(str)
        arr = {"emb": "embedding", "emo": "emo_probs", "va": "va"}[kind]
        M = np.asarray(c[arr], dtype=np.float64)
        full = np.full((len(ids), M.shape[1]), np.nan)
        rows = [pos[e] for e in cids if e in pos]
        full[rows] = M[[i for i, e in enumerate(cids) if e in pos]]
        if cache.startswith("multiframe"):
            subset_sets.append({e for e in cids if e in pos})
        elif np.isnan(full).any():
            raise ValueError(f"{key}: {int(np.isnan(full).any(axis=1).sum())} 个有效样本不在缓存中")
        blocks[key] = full

    subset = set.intersection(*subset_sets)
    subset_mask = np.array([e in subset for e in ids])
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    y = (valid["result"] == "goal").astype(int).to_numpy()
    return {"valid": valid, "ids": ids, "pos": pos, "blocks": blocks,
            "subset_mask": subset_mask, "cfg": cfg, "y": y}

# ---------------------------------------------------------------- 划分

def make_folds(data: dict, y: np.ndarray, seed: int) -> tuple[dict, list[dict]] | None:
    """用标签 y（真实或置换）跑 build_folds.compute_folds，返回 (doc, 索引折)。

    置换标签下 test/val 偶尔缺类，compute_folds 会 SystemExit → 返回 None，由调用方重抽。
    """
    v = data["valid"].copy()
    v["result"] = np.where(y == 1, "goal", "miss")
    try:
        doc = build_folds.compute_folds(v, seed=seed)
    except SystemExit:
        return None
    pos = data["pos"]
    folds = [{s: np.array([pos[e] for e in f[f"{s}_event_ids"]]) for s in ("train", "val", "test")}
             for f in doc["folds"]]
    return doc, folds


# ---------------------------------------------------------------- 单臂

def design_matrix(data: dict, arm: dict, tr: np.ndarray, y: np.ndarray) -> np.ndarray:
    """拼接该臂的特征块（全 385 行）；ctx 的 p̄ 只由本折 train 估计（硬约束 5）。"""
    parts = []
    for key in arm["feats"]:
        if key == "ctx":
            fc = data["cfg"]["features"]
            ctx = build_context_matrix_v2(data["valid"], p_bar=float(y[tr].mean()),
                                          alpha=fc["history_alpha"], clip=fc["score_diff_clip"])
            assert list(ctx.columns) == fc["context_v2"]
            parts.append(ctx.to_numpy(dtype=np.float64))
        else:
            parts.append(data["blocks"][key])
    return np.hstack(parts)


def fit_predict(data: dict, arm: dict, fold: dict, y: np.ndarray, seed: int) -> np.ndarray:
    """单臂单折：train-only 标准化 → LR / MLP → test 概率。"""
    tr, va, te = fold["train"], fold["val"], fold["test"]
    X = design_matrix(data, arm, tr, y)
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0, ddof=1)  # 与 run_baselines（pandas std）一致
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    if arm["model"] == "lr":
        lr = LogisticRegression(max_iter=5000, C=arm["C"], random_state=seed)
        lr.fit(Z[tr], y[tr])
        return lr.predict_proba(Z[te])[:, 1]

    import torch
    from src.models.context_mlp import ContextMLP
    from src.training.train_loop import predict, train_one_fold
    torch.set_num_threads(1)
    mc, tc = data["cfg"]["model"], data["cfg"]["training"]
    model = ContextMLP(in_dim=Z.shape[1], hidden=mc["hidden"], dropout=mc["dropout"])
    train_one_fold(model, Z[tr], y[tr], Z[va], y[va], seed=seed,
                   max_epochs=tc["max_epochs"], patience=tc["patience"], lr=tc["lr"],
                   weight_decay=tc["weight_decay"], batch_size=tc["batch_size"])
    return predict(model, Z[te])


def run_arm(data: dict, arm: dict, folds: list[dict], y: np.ndarray, seed: int) -> list[tuple]:
    """返回 [(test_idx, y_test, p_test), ...]；子集臂先把每折收缩到 356 子集。"""
    out = []
    sm = data["subset_mask"]
    for fold in folds:
        f = {k: v[sm[v]] for k, v in fold.items()} if arm["subset"] else fold
        if len(f["test"]) == 0:
            continue
        out.append((f["test"], y[f["test"]], fit_predict(data, arm, f, y, seed)))
    return out

# ---------------------------------------------------------------- 统计量

def pooled_rank(oof: list[tuple]) -> float:
    ys = np.concatenate([o[1] for o in oof])
    rk = np.concatenate([(pd.Series(o[2]).rank().to_numpy() - 0.5) / len(o[2]) for o in oof])
    return float(roc_auc_score(ys, rk))


def full_stats(oof: list[tuple]) -> dict:
    """观测值的完整指标（硬约束 7）：秩口径 + pooled 概率口径。"""
    ys = np.concatenate([o[1] for o in oof])
    ps = np.concatenate([o[2] for o in oof])
    aucs = [roc_auc_score(o[1], o[2]) for o in oof if len(np.unique(o[1])) == 2]
    return {
        "pooled_rank": pooled_rank(oof),
        "fold_mean": float(np.mean(aucs)),
        "pooled_auprc": float(average_precision_score(ys, ps)),
        "pooled_brier": float(brier_score_loss(ys, ps)),
        "pooled_logloss": float(log_loss(ys, np.clip(ps, 1e-6, 1 - 1e-6))),
        "pooled_balanced_acc": float(balanced_accuracy_score(ys, (ps >= 0.5).astype(int))),
        "n_test": int(len(ys)),
    }


def replicate(data: dict, y: np.ndarray, seeds: list[int], full: bool = False,
              save_dir: Path | None = None) -> dict | None:
    """在标签 y 下跑全部 seed × 全部臂，返回 {arm: [每 seed 的统计]}。"""
    res: dict[str, list] = {a["name"]: [] for a in ARMS}
    for s in seeds:
        built = make_folds(data, y, s)
        if built is None:
            return None
        doc, folds = built
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / f"seed_{s}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        for arm in ARMS:
            oof = run_arm(data, arm, folds, y, s)
            res[arm["name"]].append(full_stats(oof) if full else pooled_rank(oof))
    return res


def perm_worker(data: dict, seeds: list[int], perm_seed: int) -> dict:
    """一次置换：全局打乱标签；缺类划分则重抽（记录重抽次数）。"""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(perm_seed)
    for attempt in range(20):
        yp = rng.permutation(data["y"])
        r = replicate(data, yp, seeds)
        if r is not None:
            return {"pr": {k: float(np.mean(v)) for k, v in r.items()},
                    "per_seed": {k: v for k, v in r.items()}, "redraws": attempt}
    raise RuntimeError(f"perm {perm_seed}: 20 次重抽都缺类")

# ---------------------------------------------------------------- 判读

def p_right(null: np.ndarray, obs: float) -> float:
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


def verdicts(obs: dict, perms: list[dict]) -> dict:
    """按预注册标准判读：臂 p 值 + 配对比较 (a)(b) + 三个待复查项。"""
    arms_out, comps_out = {}, {}
    for a in ARMS:
        n = a["name"]
        pr = np.array([s["pooled_rank"] for s in obs[n]])
        entry = {"pooled_rank_mean": float(pr.mean()), "pooled_rank_std": float(pr.std()),
                 "pooled_rank_min": float(pr.min()), "pooled_rank_max": float(pr.max()),
                 "per_seed_pooled_rank": [round(float(x), 4) for x in pr]}
        for k in ("fold_mean", "pooled_auprc", "pooled_brier", "pooled_logloss", "pooled_balanced_acc"):
            entry[f"{k}_mean"] = float(np.mean([s[k] for s in obs[n]]))
        entry["n_test"] = obs[n][0]["n_test"]
        if perms:
            null = np.array([p["pr"][n] for p in perms])
            entry.update({"null_mean": float(null.mean()), "null_std": float(null.std()),
                          "null_p95": float(np.percentile(null, 95)),
                          "p": p_right(null, entry["pooled_rank_mean"])})
            entry["has_signal"] = entry["p"] < ALPHA
        arms_out[n] = entry

    for a, b in COMPARISONS:
        da = np.array([s["pooled_rank"] for s in obs[a]])
        db = np.array([s["pooled_rank"] for s in obs[b]])
        d = da - db
        entry = {"delta_mean": float(d.mean()), "delta_std": float(d.std()),
                 "n_seeds_positive": int((d > 0).sum()), "n_seeds": int(len(d)),
                 "per_seed_delta": [round(float(x), 4) for x in d]}
        cond_a = entry["n_seeds_positive"] >= MIN_SEEDS_POSITIVE
        if perms:
            null = np.array([np.mean(np.array(p["per_seed"][a]) - np.array(p["per_seed"][b]))
                             for p in perms])
            entry.update({"null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                          "p": p_right(null, entry["delta_mean"])})
            better = a if entry["delta_mean"] > 0 else b
            cond_b = bool(arms_out[better].get("has_signal", False))
            entry.update({"cond_a_seeds_positive": cond_a, "cond_b_better_arm_has_signal": cond_b,
                          "holds": bool(cond_a and cond_b and entry["delta_mean"] > 0)})
        else:
            entry["cond_a_seeds_positive"] = cond_a
        comps_out[f"{a}-{b}"] = entry

    flagged = {}
    if perms:
        for arm, (a, b) in FLAGGED.items():
            ok = bool(arms_out[arm]["has_signal"] and comps_out[f"{a}-{b}"]["holds"])
            flagged[arm] = {"replicated": ok,
                            "wording": "复现" if ok else "单一划分上的偏高，重复划分未复现"}
    return {"arms": arms_out, "comparisons": comps_out, "flagged": flagged}

# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS, help="split seed，如 2027-2036 或 2026,2027")
    ap.add_argument("--n-perm", type=int, default=200, help="全流程 refit 置换次数（0=跳过）")
    ap.add_argument("--perm-seed", type=int, default=12345)
    ap.add_argument("--n-jobs", type=int, default=-2)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    seeds = parse_seeds(args.seeds)
    data = load_data()
    print(f"有效样本 {len(data['ids'])}，356 子集 {int(data['subset_mask'].sum())}，"
          f"split seeds {seeds}，{len(ARMS)} 臂")

    t0 = time.time()
    obs = replicate(data, data["y"], seeds, full=True, save_dir=REPEATS_DIR)
    if obs is None:
        print("真实标签下划分缺类，检查 build_folds", file=sys.stderr)
        return 1
    print(f"观测值完成 {time.time() - t0:.0f}s")
    for a in ARMS:
        pr = [s["pooled_rank"] for s in obs[a["name"]]]
        print(f"  {a['name']:>10} pooled_rank {np.mean(pr):.4f}±{np.std(pr):.4f} "
              f"[{min(pr):.3f}, {max(pr):.3f}]")

    perms: list[dict] = []
    if args.n_perm > 0:
        t1 = time.time()
        perms = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(perm_worker)(data, seeds, args.perm_seed + i) for i in range(args.n_perm))
        print(f"置换 {args.n_perm} 次完成 {time.time() - t1:.0f}s，"
              f"重抽总数 {sum(p['redraws'] for p in perms)}")

    v = verdicts(obs, perms)

    print(f"\n{'arm':>10} | {'pooled_rank':>12} | {'null_mean':>9} | {'p95':>6} | {'p':>6} | 信号")
    for n, e in v["arms"].items():
        print(f"{n:>10} | {e['pooled_rank_mean']:.4f}±{e['pooled_rank_std']:.3f} | "
              f"{e.get('null_mean', float('nan')):>9.4f} | {e.get('null_p95', float('nan')):>6.4f} | "
              f"{e.get('p', float('nan')):>6.3f} | {e.get('has_signal', '-')}")
    print()
    for k, e in v["comparisons"].items():
        print(f"{k:>20}: Δ {e['delta_mean']:+.4f}±{e['delta_std']:.4f}  "
              f"{e['n_seeds_positive']}/{e['n_seeds']} seed>0  p={e.get('p', float('nan')):.3f}  "
              f"成立={e.get('holds', '-')}")
    for k, e in v["flagged"].items():
        print(f"  待复查 {k}: {e['wording']}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [{"arm": n, "split_seed": s, **st} for n, lst in obs.items() for s, st in zip(seeds, lst)]
    table = TABLES_DIR / f"repeated_splits_{stamp}.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "重复划分（多 split seed × 5 折）+ 全流程 refit 置换零分布，复查单划分结论",
        "criteria": CRITERIA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_seeds": seeds, "n_perm": args.n_perm, "perm_seed": args.perm_seed,
        "perm_redraws_total": sum(p["redraws"] for p in perms),
        "n_valid_events": len(data["ids"]), "n_subset": int(data["subset_mask"].sum()),
        "folds_dir": REPEATS_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "arms_spec": ARMS, "comparisons_spec": COMPARISONS,
        **v,
        "per_seed_table": table.relative_to(PROJECT_ROOT).as_posix(),
    }
    path = REPORTS_DIR / f"repeated_splits_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n表:   {table.relative_to(PROJECT_ROOT).as_posix()}\n报告: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
