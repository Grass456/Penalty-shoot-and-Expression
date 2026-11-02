#!/usr/bin/env python
"""M5 多帧聚合评估：多帧 embedding vs 单帧 embedding 在同一样本子集上的 LOMO 对比。

背景（2026-11-02）：cache_multiframe_features.py 产出 356/385 踢次的多帧聚合
embedding（特写窗口内球员帧逐维均值）。方向 2 的假设 = 单帧测量噪声（选帧
中位距触球 9.3s，一半样本取自走位阶段）掩埋了 FER 信号；若成立，多帧聚合
（测量噪声平均）应抬升 AUROC。

公平对比的关键口径：多帧臂只有 356 个样本（28 踢次无球员帧），与单帧臂的
385 不同。**两组必须在同一 356 子集上跑**（同 folds 子集、同协议），否则
差异混入样本集变化。368 = 385 − (28 空踢次 + 1 窗口过短)；folds 重建为
"356 子集上的 LOMO"（按 match_id 分组，跳过全空场次）。

臂设计：
    single_356   单帧 embedding（features/fer/enet_b0_8_best_vgaf），356 子集
    multi_mean   多帧逐维均值（主模式）
    multi_median 逐维中位数（离群帧更稳）
    multi_last8  最接近触球的 8 帧均值（"越近越紧张"假设的独立检验）
    多帧臂若优于单帧臂 → 测量噪声是主因；两者皆 ≈0.5 → 单帧/多帧都无信号

统计口径：fold_mean + pooled_rank 双报（pooled_rank 为推荐口径，
见 evaluate_oof_stats.py 的置换诊断）。配对差 multi − single 逐折给出。

用法（项目根目录）:
    python scripts/eval_multiframe.py
    python scripts/eval_multiframe.py --grid 0.003,0.01,0.1   # C 网格
输入: folds/match_disjoint.json、events.csv、features/fer/{enet_b0_8_best_vgaf,
    multiframe_mean_*,multiframe_median_*,multiframe_last_k_k8_*}/embeddings.npz
输出: results/tables/multiframe_eval_<stamp>.csv / results/reports/multiframe_eval_<stamp>.json
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

MULTI_DIR = "multiframe_mean_enet_b0_8_best_vgaf"      # 主模式缓存（含全部踢次的 meta）
SINGLE_CACHE = "enet_b0_8_best_vgaf"
EMB_C_DEFAULT = 0.01
DEFAULT_GRID = "0.01"  # 与 M1 同 C；扫点可加


def load_valid_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def load_emb(cache_name: str) -> pd.DataFrame:
    path = FEATURES_DIR / cache_name / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺缓存: {path}（先跑对应 cache_*_features.py）")
    c = np.load(path, allow_pickle=True)
    ids = c["event_ids"].astype(str)
    return pd.DataFrame(c["embedding"],
                        columns=[f"e{j:04d}" for j in range(c["embedding"].shape[1])], index=ids)


def rebuild_folds_356(folds: list[dict], subset_ids: set[str]) -> list[dict]:
    """把 40 折 LOMO 收缩到 356 子集：每折去掉不在子集的 event_id；test 全空的折丢弃。

    注意：train 集也随之变小（该场的空踢次从 train 中消失）——这是正确口径：
    多帧臂的模型就不该见到没有特征的样本。
    """
    out = []
    for fold in folds:
        f2 = {"fold_id": fold["fold_id"]}
        for split in ("train", "val", "test"):
            ids = [i for i in fold[f"{split}_event_ids"] if i in subset_ids]
            f2[f"{split}_event_ids"] = ids
        if f2["test_event_ids"]:
            out.append(f2)
    return out


def lomo_stats(folds, X_all: pd.DataFrame, y_all: pd.Series, c: float, seed: int, max_iter: int) -> dict:
    """LOMO + 双口径统计（fold_mean / pooled_rank）。"""
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
    ya, pa = np.concatenate(ys), np.concatenate(ps)
    rk = np.concatenate([(pd.Series(p).rank().to_numpy() - 0.5) / len(p) for p in ps])
    aucs = np.array(aucs)
    return {"fold_mean": round(float(aucs.mean()), 4),
            "fold_mean_se": round(float(aucs.std() / np.sqrt(len(aucs))), 4),
            "n_folds_gt_half": int((aucs > 0.5).sum()), "n_folds": int(len(aucs)),
            "pooled_rank": round(float(roc_auc_score(ya, rk)), 4),
            "_aucs": aucs, "_ys": ys, "_ps": ps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=DEFAULT_GRID)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-iter", type=int, default=5000)
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",") if x.strip()]

    with open(FOLDS_PATH, encoding="utf-8") as f:  # Windows GBK 默认编码会炸非 ASCII
        folds40 = json.load(f)["folds"]
    ev = load_valid_events()
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)

    # 多帧缓存（主模式）定义子集
    multi = load_emb(MULTI_DIR)
    subset = set(multi.index)
    folds = rebuild_folds_356(folds40, subset)
    n_test = sum(len(f["test_event_ids"]) for f in folds)
    print(f"子集 {len(subset)} 样本，重建后 {len(folds)} 折 / test {n_test} 样本")

    # 单帧在同一子集上（index 对齐取子集）
    single = load_emb(SINGLE_CACHE).loc[sorted(subset)]
    missing = set(subset) - set(single.index)
    if missing:
        raise ValueError(f"{len(missing)} 个多帧样本不在单帧缓存中")

    # 多帧变体：median / last_k（若无缓存则跳过并提示）
    variants = {"multi_mean": multi}
    for name in ("multiframe_median_enet_b0_8_best_vgaf", "multiframe_last_k_k8_enet_b0_8_best_vgaf"):
        p = FEATURES_DIR / name / "embeddings.npz"
        if p.exists():
            variants["multi_median" if "median" in name else "multi_last8"] = load_emb(name)
        else:
            print(f"跳过 {name}（无缓存；用 cache_multiframe_features.py --agg {'median' if 'median' in name else 'last_k'} 生成）")

    # 合并臂（假设 1 × 方向 2 的交集）：多帧 VA backbone 的 embedding 与 VA 均值
    va_cache = FEATURES_DIR / "multiframe_mean_enet_b0_8_va_mtl" / "embeddings.npz"
    if va_cache.exists():
        c = np.load(va_cache, allow_pickle=True)
        ids_va = c["event_ids"].astype(str)
        va_emb = pd.DataFrame(c["embedding"],
                              columns=[f"v{j:04d}" for j in range(c["embedding"].shape[1])], index=ids_va)
        common = sorted(subset & set(ids_va))
        variants["multi_va_emb"] = va_emb.loc[common]
        if "va" in c:
            va2 = pd.DataFrame(c["va"], columns=["valence", "arousal"], index=ids_va)
            variants["multi_va2"] = va2.loc[common]
            # 拼接臂：vgaf 多帧 embedding + VA2（合并的"最强形式"）
            variants["multi_vgaf+va2"] = pd.concat([multi.loc[common], va2.loc[common]], axis=1)
        if len(common) < len(subset):
            print(f"合并臂子集 {len(common)}/{len(subset)}（VA 多帧缓存缺 {len(subset) - len(common)} 个样本，"
                  f"其对比臂自动对齐到同一子集）")
            for k in list(variants):
                variants[k] = variants[k].loc[common]
            subset = set(common)
            folds = rebuild_folds_356(folds40, subset)
    else:
        print("跳过合并臂（无 multiframe_mean_enet_b0_8_va_mtl 缓存；"
              "用 cache_multiframe_features.py --model enet_b0_8_va_mtl 生成）")

    arms = [("single_356", single)] + [(k, v.loc[sorted(subset)]) for k, v in variants.items()]

    results: dict[str, dict] = {}
    per_fold: dict[str, np.ndarray] = {}
    for label, X in arms:
        for c in grid:
            s = lomo_stats(folds, X, y_all, c, args.seed, args.max_iter)
            key = f"{label}|C={c}"
            results[key] = {k: v for k, v in s.items() if not k.startswith("_")}
            per_fold[key] = s["_aucs"]
            print(f"[{key:>22}] fold_mean {s['fold_mean']:.4f}±{s['fold_mean_se']:.4f} "
                  f"({s['n_folds_gt_half']}/{s['n_folds']}>0.5)  pooled_rank {s['pooled_rank']:.4f}")

    # 配对差（同折）：各多帧变体 − single_356（同 C）
    paired = {}
    ref_key = f"single_356|C={grid[0]}"
    for key, aucs in per_fold.items():
        if key.startswith("multi") and ref_key in per_fold and key != ref_key:
            ref = per_fold[ref_key]
            d = aucs - ref
            paired[key] = {"mean_delta": round(float(d.mean()), 4), "std": round(float(d.std()), 4),
                           "n_multi_gt_single": int((d > 0).sum()), "n": int(len(d))}

    # 空踢次画像：28 个无帧踢次是否系统性偏斜（选择偏差检查）
    empty_ids = sorted(set(ev["event_id"]) - subset)
    empty_profile = {}
    if empty_ids:
        ee = ev[ev.event_id.isin(empty_ids)]
        vv = ev[ev.event_id.isin(subset)]
        empty_profile = {
            "n_empty": len(empty_ids),
            "goal_rate_empty": round(float((ee["result"] == "goal").mean()), 4),
            "goal_rate_kept": round(float((vv["result"] == "goal").mean()), 4),
            "note": "若两者 goal 率差异大，356 子集的结论不能外推回 385",
        }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = TABLES_DIR / f"multiframe_eval_{stamp}.csv"
    rows = [{"arm_key": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(table_path, index=False)

    with open(FOLDS_PATH, "rb") as f:
        folds_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "M5：多帧聚合 vs 单帧（同 356 子集同协议），检验方向 2（单帧测量噪声掩埋信号）",
        "hypothesis": "若 multi > single：测量噪声是主因；若两者皆≈0.5：单帧/多帧都无信号（更强负面结论）",
        "grid": grid, "seed": args.seed,
        "folds": {"path": FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256_16": folds_sha,
                  "rebuilt": "40 折 LOMO 收缩到 356 子集（剔除空踢次，丢全空折）"},
        "n_subset": len(subset), "n_folds": len(folds),
        "arms": results,
        "paired_multi_minus_single": paired,
        "empty_kick_profile": empty_profile,
        "notes": [
            "单帧/多帧在同一 356 子集上对比，样本集变化不进入差异",
            "fold_mean 须与置换零分布比较（约 0.50±0.04），pooled_rank 为推荐口径",
            "multi_last8 与 multi_mean 的差 = 时间加权效应（越近触球的帧权重更高）",
        ],
        "table_path": table_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    report_path = REPORTS_DIR / f"multiframe_eval_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 配对差（multi − single，同折）===")
    for k, v in paired.items():
        print(f"{k:>14}: Δ={v['mean_delta']:+.4f}±{v['std']:.4f}  multi>single in {v['n_multi_gt_single']}/{v['n']} folds")
    if empty_profile:
        print(f"\n空踢次画像: {json.dumps(empty_profile, ensure_ascii=False)}")
    print(f"\n表:   {table_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"报告: {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
