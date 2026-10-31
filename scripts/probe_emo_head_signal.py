#!/usr/bin/env python
"""情绪头投影 vs 随机投影对照：backbone 对比里 B3 臂（8 维情绪概率）出现 0.57-0.61，
必须先排除"任何 8 维投影都能到 0.6"这一可能，再谈情绪语义。

背景（2026-10-31）：compare_fer_backbones.py 结果——embedding 臂全部 ≈0.5（无信号），
但 8 维情绪概率臂 enet_b2_8 0.609 / afew 0.576 / b2_7 0.568 / vgaf 0.503。
8 维情绪概率是 embedding 的一个**固定线性投影**（AffectNet 情绪头），
按同一协议施加于 e_i = W z_i + b_e（不用 softmax，保持与随机投影严格可比），
天然是线性模型能学到的子空间，因此单维数降低就可能带来提升。

三组对照（全部同协议、同 folds、同 C）:
    1. 情绪头投影      e = W z          （真实先验方向）
    2. 随机投影        e = G z, G~N(0,1) 行范数对齐 W 的均值行范数，K 次抽样
                       → "任意 8 维方向"的原假设分布
    3. 错配情绪头      e = W_other z     （同维 backbone 互换：vgaf↔afew, b2_8↔b2_7）
                       → 若错配也到 0.6，说明信号与"哪种情绪"无关

判读：只有 (1) 显著高于 (2) 与 (3) 时，才支持"情绪语义方向携带信号"。
     否则结论是"低维投影的正则化效应"，与情绪语义无关，B3 臂的 0.6 不可解释为发现。

用法（项目根目录）:
    python scripts/probe_emo_head_signal.py                    # 默认 K=20
    python scripts/probe_emo_head_signal.py --n-random 50 --c 0.01

输入: folds/match_disjoint.json、data/metadata/events.csv、features/fer/*/embeddings.npz
输出: results/tables/emo_head_probe_<stamp>.csv   逐折逐臂 AUROC
      results/reports/emo_head_probe_<stamp>.json 聚合 + 配对检验 + 投影统计
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FEATURES_DIR = PROJECT_ROOT / "features" / "fer"

BACKBONES = ["enet_b0_8_best_vgaf", "enet_b0_8_best_afew", "enet_b2_8", "enet_b2_7", "imagenet_b0"]
# 同维互换的错配对：(受试 arm, 提供情绪头的 arm)
MISMATCH = {
    "enet_b0_8_best_afew": "enet_b0_8_best_vgaf",
    "enet_b2_7": "enet_b2_8",
}


def load_valid_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def load_embedding(arm: str) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    """返回 (embedding 帧 index=event_id, emo_probs, n_cls, dim)。"""
    path = FEATURES_DIR / arm / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺特征缓存: {path}")
    c = np.load(path, allow_pickle=True)
    ids = c["event_ids"].astype(str)
    emb = pd.DataFrame(c["embedding"],
                       columns=[f"emb_{j:04d}" for j in range(c["embedding"].shape[1])], index=ids)
    probs = c["emo_probs"]
    n_cls, dim = probs.shape[1], c["embedding"].shape[1]
    return emb, probs, n_cls, dim


def emotion_head_weights(arm: str) -> np.ndarray | None:
    """重建情绪头权重 W (n_cls, D)：emotiefflib 的 classifier_weights。"""
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    fer = EmotiEffLibRecognizer(engine="torch", model_name=arm, device="cpu")
    return np.asarray(fer.classifier_weights, dtype=np.float64)


def metrics(y_true, p_hat) -> dict:
    y = (np.asarray(y_true) == 1).astype(int)
    p = np.asarray(p_hat, dtype=float)
    return {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "balanced_acc": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))) if len(np.unique(p)) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def run_projection_arm(folds, emb: pd.DataFrame, feats: pd.DataFrame, y_all, tag: str,
                       c: float, seed: int, max_iter: int) -> list[dict]:
    """把 feats（已投影好的低维帧，index=event_id）按标准协议跑 LOMO。

    逐折也返回 OOF 预测概率，用于 pooled AUROC（比逐折均值更稳定的估计量）。
    """
    rows = []
    for fold in folds:
        Xtr = feats.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=max_iter, C=c, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        ids = fold["test_event_ids"]
        p = lr.predict_proba((feats.loc[ids] - mu) / sd)[:, 1]
        y = y_all.loc[ids].values
        rows.append({"arm": tag, "C": c, "fold_id": fold["fold_id"], "n_test": len(ids),
                     "y": ",".join(str(int(v)) for v in y),
                     "p": ",".join(f"{v:.6f}" for v in p), **metrics(y, p)})
    return rows


def summarize(rows: list[dict], tag: str) -> dict:
    """逐折 AUROC 均值 + pooled（折内转秩后 pool，见 evaluate_oof_stats.py 的诊断）。

    pooled_rank 是 LOMO 下的推荐口径：折内转秩消除各折分数尺度差异，
    折间尺度不可比时 pooled_raw 会失真（v1 报告里同一组预测 fold_mean 0.599
    而 pooled_raw 0.492 即为此故）。
    """
    aurocs = np.array([r["auroc"] for r in rows if r["auroc"] is not None])
    y = np.concatenate([np.array([int(v) for v in r["y"].split(",")]) for r in rows])
    p = np.concatenate([np.array([float(v) for v in r["p"].split(",")]) for r in rows])
    rk = np.concatenate([(pd.Series(np.array([float(v) for v in r["p"].split(",")])).rank().to_numpy() - 0.5)
                         / r["n_test"] for r in rows])
    return {"arm": tag, "n_folds": len(aurocs),
            "auroc_mean": round(float(aurocs.mean()), 4), "auroc_std": round(float(aurocs.std()), 4),
            "auroc_se": round(float(aurocs.std() / np.sqrt(len(aurocs))), 4),
            "n_folds_gt_half": int((aurocs > 0.5).sum()),
            "pooled_raw": round(float(roc_auc_score(y, p)), 4),
            "pooled_rank": round(float(roc_auc_score(y, rk)), 4),
            "pooled_auprc": round(float(average_precision_score(y, p)), 4),
            "pooled_brier": round(float(brier_score_loss(y, p)), 4),
            "wilcoxon_vs_half_p": float(stats.wilcoxon(aurocs - 0.5).pvalue) if len(aurocs) > 5 else None,
            "ttest_vs_half_p": float(stats.ttest_1samp(aurocs, 0.5).pvalue)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-random", type=int, default=20, help="随机投影抽样次数")
    ap.add_argument("--c", type=float, default=0.01, help="固定 C（与 run_baselines 的 B3 臂一致）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--arms", default=",".join(BACKBONES))
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    with open(FOLDS_PATH, encoding="utf-8") as f:  # Windows GBK 默认编码会炸非 ASCII
        folds = json.load(f)["folds"]
    ev = load_valid_events()
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)

    embs, heads, dims = {}, {}, {}
    for arm in arms:
        emb, probs, n_cls, dim = load_embedding(arm)
        missing = set(ev["event_id"]) - set(emb.index)
        if missing:
            raise ValueError(f"{arm}: {len(missing)} 个有效样本不在缓存中")
        embs[arm], dims[arm] = emb, dim
        heads[arm] = emotion_head_weights(arm) if arm != "imagenet_b0" else None
        print(f"{arm}: dim={dim} n_cls={n_cls}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for arm in arms:
        emb = embs[arm]
        # 1. 情绪头投影（真实先验方向）
        if heads[arm] is not None:
            W = heads[arm]
            rows += run_projection_arm(folds, emb, pd.DataFrame(emb.to_numpy() @ W.T, index=emb.index),
                                       y_all, f"{arm}|emo_head", args.c, args.seed, args.max_iter)
            # 3. 错配情绪头
            src = MISMATCH.get(arm)
            if src and heads.get(src) is not None and dims[src] == dims[arm]:
                Wm = heads[src]
                rows += run_projection_arm(folds, emb, pd.DataFrame(emb.to_numpy() @ Wm.T, index=emb.index),
                                           y_all, f"{arm}|mismatch_head({src})", args.c, args.seed, args.max_iter)
        # 2. 随机投影（行范数对齐情绪头的均值行范数）
        ref_norm = float(np.linalg.norm(heads[arm], axis=1).mean()) if heads[arm] is not None else 5.0
        k_cls = heads[arm].shape[0] if heads[arm] is not None else 8
        for k in range(args.n_random):
            G = rng.standard_normal((k_cls, dims[arm]))
            G *= ref_norm / np.linalg.norm(G, axis=1, keepdims=True)
            rows += run_projection_arm(folds, emb, pd.DataFrame(emb.to_numpy() @ G.T, index=emb.index),
                                       y_all, f"{arm}|random{k:02d}", args.c, args.seed, args.max_iter)
        print(f"[{arm}] emo_head + {args.n_random} random + mismatch 完成")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = TABLES_DIR / f"emo_head_probe_{stamp}.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("y", "p")} for r in rows]).to_csv(table_path, index=False)

    summaries = {tag: summarize([r for r in rows if r["arm"] == tag], tag)
                 for tag in sorted({r["arm"] for r in rows})}
    # 逐 backbone 的判读：真实头 vs 随机头分布（逐折配对）
    verdicts = {}
    for arm in arms:
        real = {r["fold_id"]: r["auroc"] for r in rows
                if r["arm"] == f"{arm}|emo_head" and r["auroc"] is not None}
        if not real:
            continue
        rand_tags = [t for t in summaries if t.startswith(f"{arm}|random")]
        rand_per_fold = np.array([[next(r["auroc"] for r in rows
                                        if r["arm"] == t and r["fold_id"] == f) for f in real]
                                  for t in rand_tags])
        rand_mean = rand_per_fold.mean(axis=0)
        real_v = np.array([real[f] for f in real])
        # 每个折叠：真实头超过多少次随机抽样的分位
        pct_rank = (rand_per_fold < real_v).mean(axis=0)
        v = {
            "emo_head_auroc_mean": round(float(real_v.mean()), 4),
            "random_auroc_mean": round(float(rand_per_fold.mean()), 4),
            "random_auroc_std_across_draws": round(float(rand_per_fold.mean(axis=1).std()), 4),
            "delta_real_minus_random_perfold_mean": round(float((real_v - rand_mean).mean()), 4),
            "paired_ttest_p": float(stats.ttest_rel(real_v, rand_mean).pvalue),
            "n_folds_real_above_random_p90": int((pct_rank >= 0.90).sum()),
            "n_folds": int(len(real_v)),
            # pooled_rank 口径（推荐统计量）：真实头 vs 随机头分布
            "emo_head_pooled_rank": summaries[f"{arm}|emo_head"]["pooled_rank"],
            "random_pooled_rank_mean": round(float(np.mean([summaries[t]["pooled_rank"] for t in rand_tags])), 4),
            "random_pooled_rank_p95": round(float(np.percentile([summaries[t]["pooled_rank"] for t in rand_tags], 95)), 4),
            "n_random_draws": len(rand_tags),
        }
        src = MISMATCH.get(arm)
        if src and f"{arm}|mismatch_head({src})" in summaries:
            v["mismatch_head_auroc_mean"] = summaries[f"{arm}|mismatch_head({src})"]["auroc_mean"]
        verdicts[arm] = v

    with open(FOLDS_PATH, "rb") as f:
        folds_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "情绪头投影 vs 随机投影/错配头对照：B3 臂 0.57-0.61 是否只是低维投影的正则化效应",
        "C": args.c, "n_random": args.n_random, "seed": args.seed,
        "folds": {"path": FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256_16": folds_sha},
        "n_valid_events": int(len(ev)),
        "note_projection": "e = W z（未过 softmax），保证与随机投影严格同为线性投影；LR 本身线性，softmax 单调不改变排序信息",
        "summaries": summaries,
        "verdicts": verdicts,
        "interpretation_rule": "仅当真实头显著高于随机头分布（且错配头无明显优势）时，才支持情绪语义方向携带信号；否则判为低维投影的正则化效应",
        "per_fold_path": table_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    report_path = REPORTS_DIR / f"emo_head_probe_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'arm':>34} | {'fold-mean':>9} | {'pooled_rk':>9} | {'>0.5':>5} | {'p(t)':>8}")
    for tag, s in summaries.items():
        print(f"{tag:>34} | {s['auroc_mean']:>9.4f} | {s['pooled_rank']:>9.4f} | "
              f"{s['n_folds_gt_half']:>3}/{s['n_folds']} | {s['ttest_vs_half_p']:>8.4f}")
    print("\n=== 判读（真实情绪头 vs 随机投影分布）===")
    for arm, v in verdicts.items():
        line = (f"{arm:>22}: real {v['emo_head_auroc_mean']:.4f} vs random {v['random_auroc_mean']:.4f} "
                f"(±{v['random_auroc_std_across_draws']:.4f}) Δ={v['delta_real_minus_random_perfold_mean']:+.4f} "
                f"p={v['paired_ttest_p']:.4f} folds>rand_p90={v['n_folds_real_above_random_p90']}/{v['n_folds']}")
        if "mismatch_head_auroc_mean" in v:
            line += f" | mismatch {v['mismatch_head_auroc_mean']:.4f}"
        print(line)
    print(f"\n逐折表: {table_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"报告:   {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
