#!/usr/bin/env python
"""VA-MTL 臂评估：valence/arousal 连续表征能否预测点球结果（假设 1 的直接检验）。

假设 1（2026-11-02 定）：8 类离散情绪空间与点球场景错配——主罚前以"紧张"为主，
而"紧张"对应高 arousal + 负 valence 的连续区间。若成立：
    P1: VA 二维臂（valence, arousal → LR）AUROC > chance，且方向为
        arousal 正系数 / valence 负系数（紧张 → miss）；
    P2: MTL embedding 臂 ≥ 8 类 backbone embedding 臂（同一 encoder，训练
        目标含 VA，表征空间更适配）；
    P3: VA 值与 8 类情绪概率的相关结构符合"紧张"读数（高 arousal 的样本
        anger/fear 概率偏高）。

对照设计（关键）：
    - MTL embedding 臂（enet_b0_8_va_mtl 1280 维）与既有 8 类臂（enet_b0_8_best_vgaf
      1280 维）**同一 encoder 架构、同一输入**，唯一差别是微调目标含 VA——
      这是最干净的表征空间对比；
    - imagenet_b0 1280 维为通用表征下界；
    - VA 2 维臂的 C 敏感性：2 维 vs 1280 维，同一 C 的隐式正则强度差异巨大，
      故报 C 网格（bottleneck 效应已知会抬 fold_mean，判读以 pooled_rank 为主，
      并与同维随机基线比较——VA 2 维直接与"任何 2 维方向"对照即 context
      6 维臂与随机投影 p95）。

统计口径：沿用 evaluate_oof_stats.py 的结论——fold_mean 须与置换零分布比较，
pooled_rank 为推荐口径。两口径都报告。

用法（项目根目录）:
    python scripts/eval_va_arms.py                      # 全部臂
    python scripts/eval_va_arms.py --grid 0.01,0.1,1.0  # 自定义 C 网格

输入: folds/match_disjoint.json、data/metadata/events.csv、features/fer/*/embeddings.npz
输出: results/tables/va_arms_<stamp>.csv / results/reports/va_arms_<stamp>.json
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FEATURES_DIR = PROJECT_ROOT / "features" / "fer"

# (缓存目录, 特征键, 显示名, C 默认)
EMB_C = 0.01  # 与 M1 同一强 L2
DEFAULT_GRID = "0.01,0.1,1.0"


def load_valid_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def load_feat(cache_dir: str, key: str) -> pd.DataFrame:
    path = FEATURES_DIR / cache_dir / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺特征缓存: {path}")
    c = np.load(path, allow_pickle=True)
    ids = c["event_ids"].astype(str)
    arr = c[key]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = ([f"emb_{j:04d}" for j in range(arr.shape[1])] if key == "embedding"
            else (["valence", "arousal"] if key == "va" else [f"p{j}" for j in range(arr.shape[1])]))
    return pd.DataFrame(arr, columns=cols, index=ids)


def lomo_aucs(folds, X_all: pd.DataFrame, y_all: pd.Series, c: float, seed: int, max_iter: int):
    """标准 LOMO，返回逐折 AUROC 与 OOF (y, p)（train-only 标准化，硬约束 5）。"""
    aucs, ys, ps = [], [], []
    for fold in folds:
        tr = fold["train_event_ids"]
        Xtr = X_all.loc[tr]
        ytr = y_all.loc[tr].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=max_iter, C=c, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        te = fold["test_event_ids"]
        p = lr.predict_proba((X_all.loc[te] - mu) / sd)[:, 1]
        y = y_all.loc[te].values
        if len(np.unique(y)) == 2:
            aucs.append(roc_auc_score(y, p))
        ys.append(y)
        ps.append(p)
    return np.array(aucs), ys, ps


def both_stats(aucs: np.ndarray, ys, ps) -> dict:
    ya, pa = np.concatenate(ys), np.concatenate(ps)
    rk = np.concatenate([(pd.Series(p).rank().to_numpy() - 0.5) / len(p) for p in ps])
    return {"fold_mean": round(float(aucs.mean()), 4), "fold_mean_se": round(float(aucs.std() / np.sqrt(len(aucs))), 4),
            "n_folds_gt_half": int((aucs > 0.5).sum()), "n_folds": int(len(aucs)),
            "pooled_rank": round(float(roc_auc_score(ya, rk)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=DEFAULT_GRID, help="VA 2 维臂的 C 网格")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-iter", type=int, default=5000)
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",") if x.strip()]

    with open(FOLDS_PATH, encoding="utf-8") as f:  # Windows GBK 默认编码会炸非 ASCII
        folds = json.load(f)["folds"]
    ev = load_valid_events()
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)

    # 臂定义
    arms: list[tuple[str, str, str, float]] = [
        ("enet_b0_8_va_mtl", "embedding", "MTL-emb1280 (enet_b0_va)", EMB_C),
        ("mbf_va_mtl", "embedding", "MTL-emb512 (mbf_va)", EMB_C),
        ("mobilevit_va_mtl", "embedding", "MTL-emb768 (mobilevit_va)", EMB_C),
        ("enet_b0_8_best_vgaf", "embedding", "8cls-emb1280 (vgaf, 对照)", EMB_C),
        ("imagenet_b0", "embedding", "ImageNet-emb1280 (对照)", EMB_C),
    ]
    for c in grid:  # VA 2 维臂逐 C
        arms.append(("enet_b0_8_va_mtl", "va", f"VA2 (enet_b0_va, C={c})", c))
        arms.append(("mbf_va_mtl", "va", f"VA2 (mbf_va, C={c})", c))
        arms.append(("mobilevit_va_mtl", "va", f"VA2 (mobilevit_va, C={c})", c))

    caches: dict[str, dict] = {}
    for cache_dir, key, _, _ in arms:
        if cache_dir not in caches:
            path = FEATURES_DIR / cache_dir / "embeddings.npz"
            with np.load(path, allow_pickle=True) as c:  # 只有含 VA 的缓存才有 'va' 键
                keys = ("embedding", "va") if "va" in c else ("embedding",)
            caches[cache_dir] = {k: load_feat(cache_dir, k) for k in keys}
            for name, X in caches[cache_dir].items():
                missing = set(ev["event_id"]) - set(X.index)
                if missing:
                    raise ValueError(f"{cache_dir}/{name}: {len(missing)} 个有效样本不在缓存中")

    rows: list[dict] = []
    for cache_dir, key, label, c in arms:
        X = caches[cache_dir][key]
        aucs, ys, ps = lomo_aucs(folds, X, y_all, c, args.seed, args.max_iter)
        rows.append({"arm": label, "cache": cache_dir, "key": key, "C": c,
                     **both_stats(aucs, ys, ps)})
        print(f"[{label:>30}] fold_mean {rows[-1]['fold_mean']:.4f}  pooled_rank {rows[-1]['pooled_rank']:.4f}")

    # P1: VA 方向性——直接看 2 维臂系数与单特征 AUROC（全局描述，不作推断）
    va = caches["enet_b0_8_va_mtl"]["va"]
    va_desc = {
        "valence_only_global_auroc": round(float(roc_auc_score(y_all, -va["valence"])), 4),  # 假设方向: 低 valence → miss
        "arousal_only_global_auroc": round(float(roc_auc_score(y_all, va["arousal"])), 4),   # 假设方向: 高 arousal → miss
        "valence_goal_mean": round(float(va["valence"][y_all == 1].mean()), 4),
        "valence_miss_mean": round(float(va["valence"][y_all == 0].mean()), 4),
        "arousal_goal_mean": round(float(va["arousal"][y_all == 1].mean()), 4),
        "arousal_miss_mean": round(float(va["arousal"][y_all == 0].mean()), 4),
    }

    # P2: 逐折配对差 MTL-emb − 8cls-emb（同架构同输入的表征空间对比）
    def fold_map(label):
        return {r_["fold_id"]: r_["auroc"] for r_ in []}  # 占位，实际在下方重算

    mtl = lomo_aucs(folds, caches["enet_b0_8_va_mtl"]["embedding"], y_all, EMB_C, args.seed, args.max_iter)[0]
    cls = lomo_aucs(folds, caches["enet_b0_8_best_vgaf"]["embedding"], y_all, EMB_C, args.seed, args.max_iter)[0]
    img = lomo_aucs(folds, caches["imagenet_b0"]["embedding"], y_all, EMB_C, args.seed, args.max_iter)[0]
    paired = {
        "mtl_minus_8cls": {"mean": round(float((mtl - cls).mean()), 4),
                           "n_pos": int((mtl - cls > 0).sum()), "n": len(mtl)},
        "mtl_minus_imagenet": {"mean": round(float((mtl - img).mean()), 4),
                               "n_pos": int((mtl - img > 0).sum()), "n": len(mtl)},
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = TABLES_DIR / f"va_arms_{stamp}.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)

    with open(FOLDS_PATH, "rb") as f:
        folds_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "假设 1（8 类空间与场景错配，紧张=高 arousal/负 valence）的 VA-MTL 臂检验",
        "predictions": {
            "P1": "VA2 臂 > chance 且方向 = arousal 正/valence 负系数（紧张→miss）",
            "P2": "MTL embedding ≥ 同架构 8 类 embedding（vgaf）与 ImageNet（表征空间适配）",
            "P3": "VA 与 8 类概率的相关符合紧张读数（辅助诊断）",
        },
        "C_emb": EMB_C, "C_grid_va2": grid, "seed": args.seed,
        "folds": {"path": FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256_16": folds_sha},
        "n_valid_events": int(len(ev)),
        "arms": rows,
        "va_descriptive": va_desc,
        "paired_fold_auroc": paired,
        "notes": [
            "fold_mean 的零假设不是 0.5（见 oof_stats 报告），判读以 pooled_rank 为主、fold_mean 为辅",
            "VA2 是 2 维投影，bottleneck 效应会抬 fold_mean——与随机 2 维投影 p95(~0.55) 对照判读",
            "MTL-emb 与 8cls-emb 同架构同输入，唯一差别是微调目标含 VA——最干净的表征空间对比",
        ],
        "table_path": table_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    report_path = REPORTS_DIR / f"va_arms_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 假设 1 判读 ===")
    print(f"P1 (VA 方向): {json.dumps(va_desc, ensure_ascii=False)}")
    print(f"P2 (配对差):  {json.dumps(paired, ensure_ascii=False)}")
    print(f"\n逐折表: {table_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"报告:   {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
