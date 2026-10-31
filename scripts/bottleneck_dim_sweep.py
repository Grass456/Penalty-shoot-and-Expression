#!/usr/bin/env python
"""瓶颈维度扫描：情绪头（8 维）的优势是真信号，还是"降维本身"的效应？

背景（2026-10-31，接 compare_fer_backbones.py + probe_emo_head_signal.py）:
    - 全维 LR（1280/1408 维）在各 backbone 上 test AUROC ≈0.44-0.52，无信号；
    - 但 8 维情绪头**线性投影** + 同协议 LR：b2_8 fold_mean 0.594 / pooled_rank 0.554，
      b2_7 0.567/0.535，afew 0.576/0.537，vgaf 0.503/0.488；
    - 100 次随机 8 维投影（同一 b2_8 embedding）fold_mean 均值 0.531±0.041、
      pooled_rank 0.513±0.032 —— **随机投影也高于 0.5**，
      说明"降维"本身就能把微弱信号从高维噪声里捞出来。

因此必须先做本对照，才能判断情绪头是否有独立贡献：
    对每个 backbone、每个瓶颈维度 k ∈ {2,4,8,16,32,64,128}，抽 M 个随机
    Gaussian 投影（行范数对齐，与情绪头同尺度），同协议跑 LOMO，得到该 k 下
    fold_mean / pooled_rank 的**分布**；再把情绪头（k=n_cls）与 PCA-k
    （每折只用 train 拟合，硬约束 5）标到同一张图上。

判读规则:
    - 若情绪头落在同 k 随机投影分布的常规范围内（<p95）→ 判为降维效应，
      情绪语义方向无独立贡献，B3 臂的 0.57-0.61 不可作为"情绪信号"证据；
    - 若情绪头显著超出同 k 随机分布，且 PCA-k（数据驱动方向）不优于随机
      → 支持"情绪先验方向携带信号"；
    - 若 PCA-k 明显优于随机 → 信号在表征的主成分方向上，与情绪语义无关。

用法（项目根目录）:
    python scripts/bottleneck_dim_sweep.py                       # 默认全臂
    python scripts/bottleneck_dim_sweep.py --arms enet_b2_8 --draws 50

输入: folds/match_disjoint.json、data/metadata/events.csv、features/fer/*/embeddings.npz
输出: results/tables/bottleneck_sweep_<stamp>.csv   每个投影的逐臂统计
      results/reports/bottleneck_sweep_<stamp>.json 分布汇总 + 判读
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FEATURES_DIR = PROJECT_ROOT / "features" / "fer"

DIMS = [2, 4, 8, 16, 32, 64, 128]
ARMS = ["enet_b2_8", "enet_b2_7", "enet_b0_8_best_afew", "enet_b0_8_best_vgaf", "imagenet_b0"]


def load_valid_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def lomo_oof(folds, feats: pd.DataFrame, y_all: pd.Series, c: float, seed: int, max_iter: int):
    """标准 LOMO，返回 (fold_auroc 列表, OOF y, OOF p)。标准化/LR 只 fit train（硬约束 5）。"""
    aucs, ys, ps = [], [], []
    for fold in folds:
        tr = fold["train_event_ids"]
        Xtr = feats.loc[tr]
        ytr = y_all.loc[tr].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=max_iter, C=c, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        te = fold["test_event_ids"]
        Xs = (feats.loc[te] - mu) / sd
        p = lr.predict_proba(Xs)[:, 1]
        y = y_all.loc[te].values
        if len(np.unique(y)) == 2:
            aucs.append(roc_auc_score(y, p))
        ys.append(y)
        ps.append(p)
    return np.array(aucs), ys, ps


def fold_stats(aucs: np.ndarray, ys, ps) -> dict:
    """两个口径：fold_mean（历史）与 pooled_rank（折内转秩后 pool，推荐）。"""
    ya, pa = np.concatenate(ys), np.concatenate(ps)
    rk = np.concatenate([(pd.Series(p).rank().to_numpy() - 0.5) / len(p) for p in ps])
    return {"fold_mean": float(aucs.mean()), "fold_mean_se": float(aucs.std() / np.sqrt(len(aucs))),
            "n_folds_gt_half": int((aucs > 0.5).sum()), "n_folds": int(len(aucs)),
            "pooled_rank": float(roc_auc_score(ya, rk))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--dims", default=",".join(str(d) for d in DIMS))
    ap.add_argument("--draws", type=int, default=50, help="每个维度抽样次数")
    ap.add_argument("--c", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-iter", type=int, default=5000)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    dims = [int(d) for d in args.dims.split(",") if d.strip()]

    with open(FOLDS_PATH, encoding="utf-8") as f:  # Windows GBK 默认编码会炸非 ASCII
        folds = json.load(f)["folds"]
    ev = load_valid_events()
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)

    from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # noqa: E402

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    for arm in arms:
        path = FEATURES_DIR / arm / "embeddings.npz"
        c = np.load(path, allow_pickle=True)
        ids = c["event_ids"].astype(str)
        missing = set(ev["event_id"]) - set(ids)
        if missing:
            raise ValueError(f"{arm}: {len(missing)} 个有效样本不在缓存中")
        X = pd.DataFrame(c["embedding"], index=ids).astype(np.float64)
        Xv = X.to_numpy()
        D = Xv.shape[1]

        # 情绪头点（k = n_cls，真实先验方向）；ImageNet 臂无情绪头，随机投影行范数用 1.0
        ref_norm = 1.0
        if arm != "imagenet_b0":
            fer = EmotiEffLibRecognizer(engine="torch", model_name=arm, device="cpu")
            W = np.asarray(fer.classifier_weights, dtype=np.float64)
            ref_norm = float(np.linalg.norm(W, axis=1).mean())
            k_cls = W.shape[0]
            aucs, ys, ps = lomo_oof(folds, pd.DataFrame(Xv @ W.T, index=ids), y_all, args.c, args.seed, args.max_iter)
            records.append({"arm": arm, "kind": "emo_head", "k": k_cls, "draw": -1,
                            "row_norm": round(ref_norm, 4), **fold_stats(aucs, ys, ps)})

        # 随机投影分布（行范数对齐情绪头，保证与真实头同尺度）
        for k in dims:
            for d_i in range(args.draws):
                G = rng.standard_normal((k, D))
                G *= ref_norm / np.linalg.norm(G, axis=1, keepdims=True)
                aucs, ys, ps = lomo_oof(folds, pd.DataFrame(Xv @ G.T, index=ids), y_all, args.c, args.seed, args.max_iter)
                records.append({"arm": arm, "kind": f"random_k{k}", "k": k, "draw": d_i,
                                **fold_stats(aucs, ys, ps)})
            sub = [r for r in records if r["arm"] == arm and r["kind"] == f"random_k{k}"]
            print(f"[{arm}] random k={k:>4}: fold_mean {np.mean([r['fold_mean'] for r in sub]):.4f}  "
                  f"pooled_rank {np.mean([r['pooled_rank'] for r in sub]):.4f}")
        # PCA-k（每折只用 train 拟合）
        yv = y_all.loc[X.index].values
        for k in dims:
            aucs, ys, ps = [], [], []
            for fold in folds:
                tr, te = fold["train_event_ids"], fold["test_event_ids"]
                pca = PCA(n_components=min(k, len(tr) - 1), random_state=args.seed)
                Ztr = pca.fit_transform(Xv[[X.index.get_loc(i) for i in tr]])  # 只用 train 拟合（硬约束 5）
                Zte = pca.transform(Xv[[X.index.get_loc(i) for i in te]])
                mu, sd = Ztr.mean(0), np.where(Ztr.std(0) == 0, 1.0, Ztr.std(0))
                lr = LogisticRegression(max_iter=args.max_iter, C=args.c, random_state=args.seed)
                lr.fit((Ztr - mu) / sd, y_all.loc[tr].values)
                p = lr.predict_proba((Zte - mu) / sd)[:, 1]
                y = y_all.loc[te].values
                if len(np.unique(y)) == 2:
                    aucs.append(roc_auc_score(y, p))
                ys.append(y)
                ps.append(p)
            records.append({"arm": arm, "kind": f"pca_k{k}", "k": k, "draw": -1,
                            **fold_stats(np.array(aucs), ys, ps)})

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = TABLES_DIR / f"bottleneck_sweep_{stamp}.csv"
    pd.DataFrame(records).to_csv(table_path, index=False)

    # 判读：情绪头 / PCA-k 相对同 k 随机分布的位置（两个统计量各算一份）
    verdicts: dict[str, dict] = {}
    for arm in arms:
        sub = [r for r in records if r["arm"] == arm]
        v: dict[str, dict] = {}
        emo = next((r for r in sub if r["kind"] == "emo_head"), None)
        if emo:
            ref = [r for r in sub if r["kind"] == f"random_k{emo['k']}"]
            if ref:
                for m in ("fold_mean", "pooled_rank"):
                    vals = np.array([r[m] for r in ref])
                    v.setdefault("emo_head", {})[m] = {
                        "value": round(emo[m], 4), "random_mean": round(float(vals.mean()), 4),
                        "random_p95": round(float(np.percentile(vals, 95)), 4),
                        "z_vs_random": round(float((emo[m] - vals.mean()) / (vals.std() + 1e-12)), 2),
                        "n_random_above": int((vals > emo[m]).sum()), "n_random": int(len(vals))}
        pca = next((r for r in sub if f"pca_k{emo['k']}" == r["kind"]), None) if emo else None
        if pca:
            v["pca_same_k"] = {m: {"value": round(pca[m], 4)} for m in ("fold_mean", "pooled_rank")}
        # 各 k 下随机分布的 pooled_rank 均值（看"降维效应"随 k 的形状）
        v["random_pooled_rank_by_k"] = {
            str(k): round(float(np.mean([r["pooled_rank"] for r in sub if r["kind"] == f"random_k{k}"])), 4)
            for k in dims if any(r["kind"] == f"random_k{k}" for r in sub)}
        verdicts[arm] = v

    with open(FOLDS_PATH, "rb") as f:
        folds_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "瓶颈维度扫描：情绪头（8 维）相对同维随机投影/PCA 的位置，判定 B3 臂 0.57-0.61 是降维效应还是情绪语义信号",
        "C": args.c, "draws_per_dim": args.draws, "dims": dims, "seed": args.seed,
        "folds": {"path": FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256_16": folds_sha},
        "n_valid_events": int(len(ev)),
        "statistics": {"fold_mean": "逐折 AUROC 均值（历史口径，零分布均值≈0.50，见 oof_stats 报告）",
                       "pooled_rank": "折内转秩后 pool 的 OOF AUROC（推荐口径）"},
        "verdicts": verdicts,
        "interpretation_rule": "情绪头须显著超出同 k 随机投影分布（>p95 或 z>2）才支持情绪语义方向有独立贡献；否则判为降维瓶颈效应",
        "per_draw_csv": table_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    report_path = REPORTS_DIR / f"bottleneck_sweep_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 判读（情绪头 vs 同 k 随机投影）===")
    for arm, v in verdicts.items():
        if "emo_head" in v:
            fm, pr = v["emo_head"]["fold_mean"], v["emo_head"]["pooled_rank"]
            print(f"{arm:>22}: emo_head fold_mean {fm['value']:.4f} (rand {fm['random_mean']:.4f}±, "
                  f"{fm['n_random_above']}/{fm['n_random']} above, z={fm['z_vs_random']:+.2f}) | "
                  f"pooled_rank {pr['value']:.4f} (rand {pr['random_mean']:.4f}, "
                  f"{pr['n_random_above']}/{pr['n_random']} above, z={pr['z_vs_random']:+.2f})")
        print(f"{'':>22}  random pooled_rank by k: {v['random_pooled_rank_by_k']}")
    print(f"\n逐抽样表: {table_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"报告:     {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
