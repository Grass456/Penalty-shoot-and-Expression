#!/usr/bin/env python
"""FER 侧 C 扫描（诊断实验）：M1 fer_lr / B4 imagenet_lr 在同一 C 网格上的 40 折 LOMO 重跑。

背景（2026-10-31）：M1 的 C=0.01 是 14 场 pilot 时定的（n_train≈96），
40 场下 n_train≈366，正则强度是否仍合适未验证。本脚本对两个 embedding 臂
在同一 C 网格上重跑完整 LOMO 协议（其余协议与 run_baselines.py 逐字一致），
回答三个问题：
    1. test AUROC 对 C 的敏感度——U 型曲线是否存在、0.01 是否在平台区；
    2. val（单场）聚合后能否选出合理的 C*（硬约束 6 的合法选法）；
    3. M1−B4 配对差随 C 如何变化——H1 判据是否为正则强度的伪影。

判读纪律（同时写入报告 notes）：
    - 本扫描是敏感性诊断，不是选模型：test 曲线只用于判读平台区形状，
      不得按 test 选 C（硬约束 6）；若要改默认 C，以 val 聚合为准，
      且 M1/B4/M3 必须同步换 C 保持可比。
    - C=0.01 点应复现 run_baselines 40 场全臂结果（fer_lr 0.4754 /
      imagenet_lr 0.4711，report ...142449），作为脚本正确性 sanity check。

用法（项目根目录）:
    python scripts/sweep_fer_c.py                        # 默认网格 8 点
    python scripts/sweep_fer_c.py --grid 0.001,0.01,0.1  # 自定义网格
    python scripts/sweep_fer_c.py --arms fer_lr          # 只跑单臂

输入: folds/match_disjoint.json、data/metadata/events.csv、
      features/fer/{enet_b0_8_best_vgaf,imagenet_b0}/embeddings.npz
输出: results/tables/sweep_fer_c_<stamp>.csv   逐折逐 C 逐 split
      results/reports/sweep_fer_c_<stamp>.json 聚合 + val 选择 + 配对差
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

# 两臂共用同一协议与网格；缓存路径与 run_baselines.py 一致
ARMS: dict[str, Path] = {
    "fer_lr": PROJECT_ROOT / "features" / "fer" / "enet_b0_8_best_vgaf" / "embeddings.npz",
    "imagenet_lr": PROJECT_ROOT / "features" / "fer" / "imagenet_b0" / "embeddings.npz",
}
DEFAULT_GRID = "0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0"


def load_valid_events() -> pd.DataFrame:
    """有效样本口径与 run_baselines.py / build_folds.py / cache_fer_features.py 一致。"""
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def load_embeddings(cache_path: Path) -> pd.DataFrame:
    """读 embedding 缓存，返回 index=event_id 的 (N, 1280) 帧（对齐方式与 run_baselines 相同）。"""
    if not cache_path.exists():
        raise FileNotFoundError(f"缺特征缓存: {cache_path}（先跑 scripts/cache_fer_features.py）")
    cache = np.load(cache_path, allow_pickle=True)
    ids = cache["event_ids"].astype(str)
    return pd.DataFrame(cache["embedding"],
                        columns=[f"emb_{j:04d}" for j in range(cache["embedding"].shape[1])],
                        index=ids)


def metrics(y_true: np.ndarray, p_hat: np.ndarray) -> dict:
    """与 run_baselines.metrics 相同：AUROC / AUPRC / Balanced Acc / Brier。"""
    y = (np.asarray(y_true) == 1).astype(int)
    p = np.asarray(p_hat, dtype=float)
    return {
        "n": int(len(y)),
        "n_goal": int(y.sum()),
        "n_miss": int((1 - y).sum()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "balanced_acc": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))) if len(np.unique(p)) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def run_arm_at_c(folds: list[dict], emb: pd.DataFrame, y_all: pd.Series,
                 arm: str, c: float, seed: int, max_iter: int) -> list[dict]:
    """单臂单 C 的完整 LOMO：协议与 run_fer_lr/run_imagenet_lr 逐字一致（仅 C 可变）。"""
    rows = []
    for fold in folds:
        Xtr = emb.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)  # 标准化器只来自 train（硬约束 5）
        lr = LogisticRegression(max_iter=max_iter, C=c, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        for split in ("train", "val", "test"):
            ids = fold[f"{split}_event_ids"]
            Xs = (emb.loc[ids] - mu) / sd
            p = lr.predict_proba(Xs)[:, 1]
            m = metrics(y_all.loc[ids].values, p)
            rows.append({"arm": arm, "C": c, "fold_id": fold["fold_id"], "split": split, **m})
    return rows


def aggregate(rows: list[dict], arm: str, c: float, split: str) -> dict:
    sub = [r for r in rows if r["arm"] == arm and r["C"] == c and r["split"] == split]
    out = {"arm": arm, "C": c, "split": split, "n_folds": len(sub)}
    for k in ("auroc", "auprc", "balanced_acc", "brier"):
        vals = [r[k] for r in sub if r[k] is not None]
        out[k] = ({"mean": round(float(np.mean(vals)), 4),
                   "std": round(float(np.std(vals)), 4),
                   "n_valid_folds": len(vals)} if vals else None)
    return out


def paired_diff(rows: list[dict], c: float) -> dict:
    """逐折配对 test AUROC 差 fer_lr − imagenet_lr（H1 判据，两臂同折同 C）。"""
    fer = {r["fold_id"]: r["auroc"] for r in rows if r["arm"] == "fer_lr" and r["C"] == c and r["split"] == "test"}
    img = {r["fold_id"]: r["auroc"] for r in rows if r["arm"] == "imagenet_lr" and r["C"] == c and r["split"] == "test"}
    diffs = np.array([fer[f] - img[f] for f in fer if f in img and fer[f] is not None and img[f] is not None])
    return {"C": c, "n_pairs": int(len(diffs)),
            "mean": round(float(diffs.mean()), 4), "std": round(float(diffs.std()), 4),
            "n_fer_gt_img": int((diffs > 0).sum())}


def val_selection(rows: list[dict], grid: list[float]) -> tuple[dict, dict]:
    """两种合法选法（都不看 test）：

    global —— 聚合 val AUROC 最高的 C*（论文协议"定一次默认 C"的现实选法）；
    per_fold —— 每折用本折 val 选 C 再报本折 test（完全 within-protocol，
                但 val=单场噪声大，用于检验 val 是否携带任何信号）。
    平票取更小 C（更强正则，保守方向）。
    """
    out_global, out_per_fold = {}, {}
    for arm in ARMS:
        val_means = {}
        for c in grid:
            a = aggregate(rows, arm, c, "val")["auroc"]
            val_means[c] = a["mean"] if a else float("-inf")
        c_star = max(grid, key=lambda c: (val_means[c], -c))
        test_at = aggregate(rows, arm, c_star, "test")["auroc"]
        out_global[arm] = {"C_star": c_star, "val_auroc_mean": val_means[c_star],
                           "test_auroc_at_C_star": test_at["mean"] if test_at else None,
                           "val_auroc_by_C": {str(c): (v if v != float("-inf") else None) for c, v in val_means.items()}}
        # per-fold 选择
        folds_ids = sorted({r["fold_id"] for r in rows if r["arm"] == arm})
        chosen, test_vals = [], []
        for f in folds_ids:
            per_c = {c: next((r["auroc"] for r in rows
                              if r["arm"] == arm and r["C"] == c and r["split"] == "val" and r["fold_id"] == f), None)
                     for c in grid}
            valid = {c: v for c, v in per_c.items() if v is not None}
            if not valid:
                continue
            c_f = max(valid, key=lambda c: (valid[c], -c))
            chosen.append(c_f)
            t = next((r["auroc"] for r in rows
                      if r["arm"] == arm and r["C"] == c_f and r["split"] == "test" and r["fold_id"] == f), None)
            if t is not None:
                test_vals.append(t)
        counts = {str(c): chosen.count(c) for c in grid if chosen.count(c)}
        out_per_fold[arm] = {"test_auroc_mean": round(float(np.mean(test_vals)), 4) if test_vals else None,
                             "n_folds": len(test_vals), "chosen_C_counts": counts}
    return out_global, out_per_fold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=DEFAULT_GRID, help="逗号分隔的 C 网格（对数布点）")
    ap.add_argument("--arms", default="fer_lr,imagenet_lr", help="逗号分隔: fer_lr,imagenet_lr")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-iter", type=int, default=5000)
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        print(f"未知臂: {sorted(unknown)}（可用: {sorted(ARMS)}）", file=sys.stderr)
        return 1

    if not FOLDS_PATH.exists():
        print(f"缺 folds 文件: {FOLDS_PATH}", file=sys.stderr)
        return 1
    with open(FOLDS_PATH, encoding="utf-8") as f:  # Windows GBK 默认编码会炸非 ASCII 字符
        folds = json.load(f)["folds"]

    ev = load_valid_events()
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    embs = {}
    for arm in arms:
        emb = load_embeddings(ARMS[arm])
        missing = set(ev["event_id"]) - set(emb.index)
        if missing:
            raise ValueError(f"{arm}: {len(missing)} 个有效样本不在缓存中（需重跑 cache_fer_features.py）")
        embs[arm] = emb

    rows: list[dict] = []
    for c in grid:
        for arm in arms:
            rows += run_arm_at_c(folds, embs[arm], y_all, arm, c, args.seed, args.max_iter)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] C={c} 完成（{len(arms)} 臂 x {len(folds)} 折）")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = TABLES_DIR / f"sweep_fer_c_{stamp}.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)

    aggregates = [aggregate(rows, arm, c, split) for arm in arms for c in grid for split in ("train", "val", "test")]
    val_global, val_per_fold = val_selection(rows, grid)
    with open(FOLDS_PATH, "rb") as f:
        folds_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    ids_sha = hashlib.sha256(",".join(sorted(ev["event_id"])).encode()).hexdigest()[:16]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "FER 侧 C 敏感性扫描（诊断）：M1/B4 同网格重跑，检验 C=0.01 是否平台区、val 能否选 C、H1 差是否正则伪影",
        "seed": args.seed,
        "max_iter": args.max_iter,
        "grid": grid,
        "arms": arms,
        "folds": {"path": FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256_16": folds_sha},
        "events_fingerprint": ids_sha,
        "n_valid_events": int(len(ev)),
        "aggregates": aggregates,
        "val_selection_global": val_global,
        "val_selection_per_fold": val_per_fold,
        "paired_diff_fer_minus_imagenet": [paired_diff(rows, c) for c in grid],
        "notes": [
            "诊断扫描，非选模型：test 曲线只用于判读平台区，不得按 test 选 C（硬约束 6）",
            "若改默认 C：以 val 聚合为准，且 M1/B4/M3 同步换 C 保持可比",
            "C=0.01 点应复现 run_baselines 40 场结果（fer_lr 0.4754 / imagenet_lr 0.4711）",
        ],
        "per_fold_path": table_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    report_path = REPORTS_DIR / f"sweep_fer_c_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要：val/test AUROC + 配对差
    fmt = lambda d: f"{d['mean']:.3f}±{d['std']:.3f}" if d else "n/a"
    print(f"\n{'C':>8} | {'M1 val':>11} | {'M1 test':>11} | {'B4 val':>11} | {'B4 test':>11} | {'diff(test)':>11} | n_pos")
    for c in grid:
        mv = next(a for a in aggregates if a["arm"] == "fer_lr" and a["C"] == c and a["split"] == "val")["auroc"]
        mt = next(a for a in aggregates if a["arm"] == "fer_lr" and a["C"] == c and a["split"] == "test")["auroc"]
        bv = next(a for a in aggregates if a["arm"] == "imagenet_lr" and a["C"] == c and a["split"] == "val")["auroc"]
        bt = next(a for a in aggregates if a["arm"] == "imagenet_lr" and a["C"] == c and a["split"] == "test")["auroc"]
        pd_ = next(d for d in report["paired_diff_fer_minus_imagenet"] if d["C"] == c)
        print(f"{c:>8} | {fmt(mv):>11} | {fmt(mt):>11} | {fmt(bv):>11} | {fmt(bt):>11} | {pd_['mean']:>+11.3f} | {pd_['n_fer_gt_img']}/{pd_['n_pairs']}")
    for arm in arms:
        g = val_global[arm]
        print(f"\n[{arm}] val 全局选择 C*={g['C_star']} (val {g['val_auroc_mean']:.3f}) → test {g['test_auroc_at_C_star']}")
        print(f"[{arm}] val 逐折选择: {val_per_fold[arm]}")
    print(f"\n逐折表: {table_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"报告:   {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
