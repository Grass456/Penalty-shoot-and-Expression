#!/usr/bin/env python
"""FER backbone 对比：4 个 EmotiEffLib backbone + ImageNet 对照，同一 40 折 LOMO 协议重跑。

目的（2026-10-31，接 sweep_fer_c.py 的结论）：C 扫描已排除"正则强度"这一解释，
frozen 单帧 embedding 在 40 场下无信号。本脚本回答下一层问题：
    换更强的 FER backbone（B2 / 视频微调 AFEW）能否救活 M1？
    以及 B3（情绪概率臂）是否只是 enet_b0_8_best_vgaf 的选型问题？

对比臂（embedding 维/D 见缓存 meta，均由 cache_fer_features.py 同流程产出）:
    enet_b0_8_best_vgaf   1280  AffectNet-8 → VGAF 视频微调（当前主臂）
    enet_b0_8_best_afew   1280  AffectNet-8 → AFEW 视频微调
    enet_b2_8             1408  AffectNet-8 静态（更大骨干）
    enet_b2_7             1408  AffectNet-7 静态（无 Contempt）
    imagenet_b0           1280  efficientnet_b0 对照（H3 判据基准，非 FER）

协议与 run_baselines.py 的 fer_lr / emotion_lr 逐字一致（train-only 标准化、
固定 C、无 val 依赖），只把 C 展开成 3 点网格 {0.003, 0.01, 0.1}——因为不同
backbone 维度不同（1280 vs 1408），同一 C 的隐式正则强度不严格可比，
多点报告避免结论依赖单个 C。

判读纪律：
    - test 曲线只用于判读，不得按 test 选 backbone（硬约束 6）；若最终要换
      backbone，以 val 聚合为准，且 B4 对照臂需同步重跑；
    - H1 判据 = FER arm − imagenet_b0 的逐折配对差，脚本直接输出；
    - 维度不同的臂之间，"都 ≈0.5"就足以否定"换 backbone 能救活"这一假设，
      不需要比较小数点后第三位。

用法（项目根目录）:
    python scripts/compare_fer_backbones.py                 # 全部 5 臂 + B3 情绪臂
    python scripts/compare_fer_backbones.py --grid 0.01     # 单点
    python scripts/compare_fer_backbones.py --no-emo        # 跳过情绪概率臂

输入: folds/match_disjoint.json、data/metadata/events.csv、features/fer/*/embeddings.npz
输出: results/tables/backbone_compare_<stamp>.csv   逐折逐臂逐 C 逐 split
      results/reports/backbone_compare_<stamp>.json 聚合 + 配对差 + 表征描述统计
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FEATURES_DIR = PROJECT_ROOT / "features" / "fer"

# 臂定义：key = 缓存子目录名，value = (显示名, 是否 FER 情绪域预训练)
BACKBONES: dict[str, dict] = {
    "enet_b0_8_best_vgaf": {"label": "enet_b0_8_best_vgaf (1280, VGAF video-ft)", "fer": True},
    "enet_b0_8_best_afew": {"label": "enet_b0_8_best_afew (1280, AFEW video-ft)", "fer": True},
    "enet_b2_8": {"label": "enet_b2_8 (1408, AffectNet-8 static)", "fer": True},
    "enet_b2_7": {"label": "enet_b2_7 (1408, AffectNet-7 static)", "fer": True},
    "imagenet_b0": {"label": "imagenet_b0 (1280, ImageNet control)", "fer": False},
}
CONTROL = "imagenet_b0"
DEFAULT_GRID = "0.003,0.01,0.1"
EMO_NAMES_8 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
EMO_NAMES_7 = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


def load_valid_events() -> pd.DataFrame:
    """有效样本口径与 run_baselines.py / build_folds.py / cache_fer_features.py 一致。"""
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def load_cache(arm_dir: str) -> dict:
    """读一个臂的缓存，返回 {ids, embedding_df, emo_df|None, meta}。"""
    path = FEATURES_DIR / arm_dir / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺特征缓存: {path}（先跑 cache_fer_features.py --model {arm_dir}）")
    c = np.load(path, allow_pickle=True)
    ids = c["event_ids"].astype(str)
    emb = pd.DataFrame(c["embedding"],
                       columns=[f"emb_{j:04d}" for j in range(c["embedding"].shape[1])], index=ids)
    probs = c["emo_probs"]
    emo = None
    if probs.shape[1] in (7, 8):
        names = EMO_NAMES_8 if probs.shape[1] == 8 else EMO_NAMES_7
        emo = pd.DataFrame(probs, columns=names, index=ids)
    meta = json.loads(str(c["meta"].item()) if c["meta"].ndim == 0 else str(c["meta"]))
    return {"ids": ids, "emb": emb, "emo": emo, "meta": meta}


def metrics(y_true: np.ndarray, p_hat: np.ndarray) -> dict:
    """与 run_baselines.metrics 相同：AUROC / AUPRC / Balanced Acc / Brier（硬约束 7）。"""
    y = (np.asarray(y_true) == 1).astype(int)
    p = np.asarray(p_hat, dtype=float)
    return {
        "n": int(len(y)), "n_goal": int(y.sum()), "n_miss": int((1 - y).sum()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "balanced_acc": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))) if len(np.unique(p)) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def run_lr_arm(folds: list[dict], X_all: pd.DataFrame, y_all: pd.Series,
               arm: str, kind: str, c: float, seed: int, max_iter: int) -> list[dict]:
    """单臂单 C 的完整 LOMO（协议与 run_baselines 的 fer_lr/emotion_lr 一致，仅 C 可变）。"""
    rows = []
    for fold in folds:
        Xtr = X_all.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)  # 标准化器只来自 train（硬约束 5）
        lr = LogisticRegression(max_iter=max_iter, C=c, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        for split in ("train", "val", "test"):
            ids = fold[f"{split}_event_ids"]
            p = lr.predict_proba((X_all.loc[ids] - mu) / sd)[:, 1]
            m = metrics(y_all.loc[ids].values, p)
            rows.append({"arm": arm, "kind": kind, "C": c, "fold_id": fold["fold_id"], "split": split, **m})
    return rows


def aggregate(rows: list[dict], arm: str, kind: str, c: float, split: str) -> dict:
    sub = [r for r in rows if r["arm"] == arm and r["kind"] == kind and r["C"] == c and r["split"] == split]
    out = {"arm": arm, "kind": kind, "C": c, "split": split, "n_folds": len(sub)}
    for k in ("auroc", "auprc", "balanced_acc", "brier"):
        vals = [r[k] for r in sub if r[k] is not None]
        out[k] = ({"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                   "n_valid_folds": len(vals)} if vals else None)
    return out


def paired_diff_vs_control(rows: list[dict], kind: str, c: float) -> dict:
    """逐折配对 test AUROC 差（各 FER backbone − imagenet_b0），H1 判据。"""
    ctrl = {r["fold_id"]: r["auroc"] for r in rows
            if r["arm"] == CONTROL and r["kind"] == kind and r["C"] == c and r["split"] == "test"}
    out = {}
    for arm, info in BACKBONES.items():
        if not info["fer"]:
            continue
        sub = {r["fold_id"]: r["auroc"] for r in rows
               if r["arm"] == arm and r["kind"] == kind and r["C"] == c and r["split"] == "test"}
        d = np.array([sub[f] - ctrl[f] for f in sub
                      if f in ctrl and sub[f] is not None and ctrl[f] is not None])
        out[arm] = {"C": c, "n_pairs": int(len(d)), "mean_diff": round(float(d.mean()), 4),
                    "std_diff": round(float(d.std()), 4), "n_backbone_gt_control": int((d > 0).sum())}
    return out


def representation_stats(cache: dict) -> dict:
    """表征描述统计（无监督，仅用于解释为什么各 backbone 都无信号；不参与选型）。

    participation_ratio: 中心化后奇异值平方和的平方 / 四次方和——"有效维数"，
    越小说明表征越坍缩到少数方向；mean_cos_sim: 原始 embedding 两两余弦相似度
    均值，接近 1 说明样本间几乎不可分（对 385 个样本的全样本计算，属描述性
    统计，不构成泄露——不含标签、不进模型）。
    """
    X = cache["emb"].to_numpy(dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    pr = float(s.sum() ** 2 / (s ** 2).sum()) if (s ** 2).sum() > 0 else None
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = Xn @ Xn.T
    n = len(X)
    return {"dim": int(X.shape[1]), "participation_ratio": round(pr, 2) if pr else None,
            "participation_ratio_frac": round(pr / X.shape[1], 4) if pr else None,
            "mean_cos_sim": round(float((cos.sum() - n) / (n * n - n)), 4),
            "mean_abs_emb": round(float(np.abs(X).mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=DEFAULT_GRID, help="逗号分隔的 C 网格")
    ap.add_argument("--arms", default=",".join(BACKBONES), help="逗号分隔的缓存目录名")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--no-emo", action="store_true", help="跳过 B3 情绪概率臂")
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(BACKBONES)
    if unknown:
        print(f"未知臂: {sorted(unknown)}（可用: {sorted(BACKBONES)}）", file=sys.stderr)
        return 1
    if CONTROL not in arms:
        print(f"!! 未包含对照臂 {CONTROL}，H1 配对差将无法计算", file=sys.stderr)

    if not FOLDS_PATH.exists():
        print(f"缺 folds 文件: {FOLDS_PATH}", file=sys.stderr)
        return 1
    with open(FOLDS_PATH, encoding="utf-8") as f:  # Windows GBK 默认编码会炸非 ASCII 字符
        folds = json.load(f)["folds"]

    ev = load_valid_events()
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    caches = {}
    for arm in arms:
        cache = load_cache(arm)
        missing = set(ev["event_id"]) - set(cache["ids"])
        if missing:
            raise ValueError(f"{arm}: {len(missing)} 个有效样本不在缓存中（需重跑 cache_fer_features.py）")
        caches[arm] = cache
        print(f"{arm}: dim={cache['emb'].shape[1]} img_size={cache['meta']['img_size']} "
              f"n_cls={cache['meta']['n_emotion_classes']}")

    rows: list[dict] = []
    for c in grid:
        for arm in arms:
            rows += run_lr_arm(folds, caches[arm]["emb"], y_all, arm, "embedding", c, args.seed, args.max_iter)
            if not args.no_emo and caches[arm]["emo"] is not None:
                rows += run_lr_arm(folds, caches[arm]["emo"], y_all, arm, "emo_probs", c, args.seed, args.max_iter)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] C={c} 完成（{len(arms)} 臂 x {len(folds)} 折）")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = TABLES_DIR / f"backbone_compare_{stamp}.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)

    kinds = ["embedding"] + ([] if args.no_emo else ["emo_probs"])
    aggregates = [aggregate(rows, arm, kind, c, split)
                  for arm in arms for kind in kinds for c in grid for split in ("train", "val", "test")]
    with open(FOLDS_PATH, "rb") as f:
        folds_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    ids_sha = hashlib.sha256(",".join(sorted(ev["event_id"])).encode()).hexdigest()[:16]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "FER backbone 对比：换 backbone（B2 / AFEW）能否救活 frozen 单帧 embedding（H1）",
        "seed": args.seed, "max_iter": args.max_iter, "grid": grid, "arms": arms,
        "folds": {"path": FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256_16": folds_sha},
        "events_fingerprint": ids_sha, "n_valid_events": int(len(ev)),
        "backbone_meta": {a: {**caches[a]["meta"], **BACKBONES[a]} for a in arms},
        "representation_stats": {a: representation_stats(caches[a]) for a in arms},
        "aggregates": aggregates,
        "paired_diff_vs_imagenet": {kind: {str(c): paired_diff_vs_control(rows, kind, c) for c in grid}
                                    for kind in kinds},
        "notes": [
            "test 曲线只用于判读，不得按 test 选 backbone（硬约束 6）；换 backbone 须以 val 聚合为准并同步重跑 B4",
            "不同 backbone 维度不同（1280/1408），同一 C 的隐式正则强度不严格可比，故报 3 点网格",
            "C=0.01 的 enet_b0_8_best_vgaf 点应复现 run_baselines 40 场 fer_lr≈0.475",
            "representation_stats 为无监督描述统计（无标签、不进模型），仅用于解释表征是否坍缩",
        ],
        "per_fold_path": table_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    report_path = REPORTS_DIR / f"backbone_compare_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fmt = lambda d: f"{d['mean']:.3f}±{d['std']:.3f}" if d else "n/a"
    for kind in kinds:
        print(f"\n=== {kind} / test AUROC ===")
        print(f"{'arm':>22} | {'C':>7} | {'test AUROC':>13} | {'AUPRC':>13} | {'Brier':>13}")
        for arm in arms:
            for c in grid:
                t = next(a for a in aggregates if a["arm"] == arm and a["kind"] == kind and a["C"] == c and a["split"] == "test")
                print(f"{arm:>22} | {c:>7} | {fmt(t['auroc']):>13} | {fmt(t['auprc']):>13} | {fmt(t['brier']):>13}")
        for c in grid:
            d = paired_diff_vs_control(rows, kind, c)
            s = "  ".join(f"{a}:{v['mean_diff']:+.3f}" for a, v in d.items())
            print(f"  [C={c}] H1 配对差 (backbone − imagenet): {s}")
    print("\n=== 表征描述统计（无监督）===")
    for a in arms:
        st = report["representation_stats"][a]
        print(f"{a:>22}: dim={st['dim']:>4} PR={st['participation_ratio']:>7} ({st['participation_ratio_frac']:.3f} of dim) "
              f"mean_cos_sim={st['mean_cos_sim']:.4f}")
    print(f"\n逐折表: {table_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"报告:   {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
