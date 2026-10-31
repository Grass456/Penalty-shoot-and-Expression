#!/usr/bin/env python
"""融合臂的 C 敏感性扫描：M3(v1) / M3v2(v2) 的"均匀 L2 压死 context"假设检验。

背景（2026-10-31）：M3v2（context v2 + FER embedding，C=0.01）test AUROC 0.468，
既低于 B1v2（0.577）也低于 M1（0.475）。系数分解显示 context 块被压到
||coef_ctx||=0.106 而 embedding 块 0.819，top10 里只有 1.07 个 context 维。
两个互斥解释：
    (a) 融合本身无价值（H3 阴性，与 M3 v1 结论一致）；
    (b) 单一 C 同时施加于 6 维 context 与 1280 维噪声 embedding 时，
        context 的可用信号被为压制高维噪声而设的强收缩牺牲掉——协议缺陷，
        不是融合的错。

判别方法：把 C 当作自由参数扫开（C 越小 embedding 噪声被压得越狠，
context 相对权重越高）。若存在某个 C 使 fusion ≥ B1v2，说明是 (b)；
若任何 C 下 fusion 都 ≤ 其 context 单独臂，则是 (a)。

注意：本脚本每个 C 下同时跑"融合臂"与"同 C 的纯 context 臂"，比较的是
**同一 C 下 fusion 相对 context-only 的增量**，而不是跨 C 比绝对值——
后者等于按 test 选 C，违反硬约束 6。脚本只作诊断，不改 M3/M3v2 协议。

用法（项目根目录）:
    python scripts/sweep_fused_c.py
    python scripts/sweep_fused_c.py --grid 0.0001,0.001,0.01,0.1

输入: folds/match_disjoint.json、data/metadata/events.csv、src/config/context_v2.yaml、
      features/fer/enet_b0_8_best_vgaf/embeddings.npz
输出: results/reports/sweep_fused_c_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "context_v2.yaml"
EMB_CACHE = PROJECT_ROOT / "features" / "fer" / "enet_b0_8_best_vgaf" / "embeddings.npz"

from run_baselines import build_context_matrix_v2  # noqa: E402


def load_valid_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def metrics(y_true, p_hat) -> dict:
    y = (np.asarray(y_true) == 1).astype(int)
    p = np.asarray(p_hat, dtype=float)
    return {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "balanced_acc": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))) if len(np.unique(p)) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def run_one(folds, ev, emb, cfg, y_all, kind: str, c: float, seed: int, max_iter: int) -> list[dict]:
    """kind ∈ {context_only, fused}；每个 C 下同协议跑一遍（train-only 标准化，硬约束 5）。"""
    feat = cfg["features"]["context_v2"]
    alpha, clip = cfg["features"]["history_alpha"], cfg["features"]["score_diff_clip"]
    rows = []
    for fold in folds:
        tr = fold["train_event_ids"]
        p_bar = float(y_all.loc[tr].mean())  # 收缩目标只来自 train
        X_all = build_context_matrix_v2(ev, p_bar=p_bar, alpha=alpha, clip=clip)
        if kind == "fused":
            X_all = pd.concat([X_all, emb], axis=1)
        Xtr = X_all.loc[tr]
        ytr = y_all.loc[tr].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=max_iter, C=c, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        coef = lr.coef_[0]
        extra = {}
        if kind == "fused":
            extra = {"ctx_coef_l2": round(float(np.linalg.norm(coef[:len(feat)])), 4),
                     "fer_coef_l2": round(float(np.linalg.norm(coef[len(feat):])), 4)}
        for split in ("train", "val", "test"):
            ids = fold[f"{split}_event_ids"]
            p = lr.predict_proba((X_all.loc[ids] - mu) / sd)[:, 1]
            m = metrics(y_all.loc[ids].values, p)
            rows.append({"kind": kind, "C": c, "fold_id": fold["fold_id"], "split": split, **m, **extra})
    return rows


def pull(rows, kind, c, split, key) -> float:
    vals = [r[key] for r in rows if r["kind"] == kind and r["C"] == c and r["split"] == split and r[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-iter", type=int, default=5000)
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",") if x.strip()]

    with open(FOLDS_PATH, encoding="utf-8") as f:  # Windows GBK 默认编码会炸非 ASCII
        folds = json.load(f)["folds"]
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if list(cfg["features"]["context_v2"]) is None:
        raise ValueError("context_v2 配置缺失")

    ev = load_valid_events()
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    cch = np.load(EMB_CACHE, allow_pickle=True)
    ids = cch["event_ids"].astype(str)
    missing = set(ev["event_id"]) - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} 个有效样本不在 FER 缓存中（需重跑 cache_fer_features.py）")
    emb = pd.DataFrame(cch["embedding"], columns=[f"e{j}" for j in range(cch["embedding"].shape[1])], index=ids)

    rows: list[dict] = []
    for c in grid:
        rows += run_one(folds, ev, emb, cfg, y_all, "context_only", c, args.seed, args.max_iter)
        rows += run_one(folds, ev, emb, cfg, y_all, "fused", c, args.seed, args.max_iter)
        print(f"C={c}: context_only test {pull(rows,'context_only',c,'test','auroc'):.4f} | "
              f"fused test {pull(rows,'fused',c,'test','auroc'):.4f} | "
              f"Δ {pull(rows,'fused',c,'test','auroc') - pull(rows,'context_only',c,'test','auroc'):+.4f} | "
              f"||ctx|| {pull(rows,'fused',c,'test','ctx_coef_l2'):.3f}")

    # 逐折配对差（同 C 同折）：fused − context_only，看方向是否稳定而非只比均值
    paired = {}
    for c in grid:
        co = {r["fold_id"]: r["auroc"] for r in rows
              if r["kind"] == "context_only" and r["C"] == c and r["split"] == "test"}
        fu = {r["fold_id"]: r["auroc"] for r in rows
              if r["kind"] == "fused" and r["C"] == c and r["split"] == "test"}
        d = np.array([fu[f] - co[f] for f in fu if f in co and fu[f] is not None and co[f] is not None])
        paired[str(c)] = {"mean_delta": round(float(d.mean()), 4), "std": round(float(d.std()), 4),
                          "n_folds_fused_ge_context": int((d >= 0).sum()), "n_pairs": int(len(d))}

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "检验 M3v2 低于 B1v2 的原因是融合无价值(H3 阴性)还是单一 C 压死 context(协议缺陷)",
        "grid": grid, "seed": args.seed,
        "note": "同 C 内比较 fused vs context_only（Δ 列）；跨 C 只看曲线的趋势与顶点，不按 test 选 C（硬约束 6）",
        "context_only_test_auroc": {str(c): round(pull(rows, "context_only", c, "test", "auroc"), 4) for c in grid},
        "fused_test_auroc": {str(c): round(pull(rows, "fused", c, "test", "auroc"), 4) for c in grid},
        "fused_ctx_coef_l2": {str(c): round(pull(rows, "fused", c, "test", "ctx_coef_l2"), 4) for c in grid},
        "fused_fer_coef_l2": {str(c): round(pull(rows, "fused", c, "test", "fer_coef_l2"), 4) for c in grid},
        "paired_fused_minus_context": paired,
        "per_fold_rows": rows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"sweep_fused_c_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
