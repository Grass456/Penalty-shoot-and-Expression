#!/usr/bin/env python
"""OOF 统计量诊断：逐折 AUROC 均值 vs pooled OOF AUROC，哪个统计量可信？

背景（2026-10-31）：probe_emo_head_signal.py 出现自相矛盾的结果——
同一组预测（enet_b2_8 情绪头投影，40 折 LOMO）：
    逐折 AUROC 均值 = 0.5994（25/40 折 >0.5）
    pooled OOF AUROC = 0.4921（低于 0.5）
两者本应一致，差异说明"逐折 AUROC 再取均值"这个聚合方式在本数据规模下不可靠。

待量化验证的假设：LOMO 40 折，每折 test 仅 6-10 个样本，单折 AUROC 是极小
样本上的秩统计（观测 std≈0.24）。逐折均值的标准误 ≈0.038，且该统计量对
"折内排序尚可、折间尺度不可比"会系统偏高。pooled 把 385 个 OOF 预测放一起
算一次秩，方差小，但要求折间分数可比——各折标准化参数与 LR 系数都不同，
故 pooled 也可能失真。因此比较 4 个统计量，并各自构造置换零分布：

    A. fold_mean     逐折 AUROC 均值（历史口径，对照旧结果用）
    B. pooled_raw    原始 OOF 概率直接 pool
    C. pooled_rank   每折内先转秩到 (0,1) 再 pool —— 折间尺度无关，只保留
                     折内排序信息，是 LOMO 下的正确做法（推荐口径）
    D. pooled_z      每折内 z-score 后 pool（对折内分布形状敏感，作对照）

零分布 = 每折内独立置换标签（保留折内类比例），破坏"折内特征-标签"关联但
保留折结构与类比例。1000 次。关键点：fold_mean 的零假设不是 0.5，而是其零
分布均值（可能明显偏离 0.5），因此 p 值必须与零分布比较；同时报告每个统计量
在零分布下的离散度，用于判断真实值偏离是否可辨识。

用法（项目根目录）:
    python scripts/evaluate_oof_stats.py                    # 默认臂集，1000 次置换
    python scripts/evaluate_oof_stats.py --n-perm 2000

输入: folds/match_disjoint.json、data/metadata/events.csv、features/fer/*/embeddings.npz
输出: results/reports/oof_stats_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FEATURES_DIR = PROJECT_ROOT / "features" / "fer"
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "context_v2.yaml"

BACKBONES = ["enet_b0_8_best_vgaf", "enet_b0_8_best_afew", "enet_b2_8", "enet_b2_7", "imagenet_b0"]


def load_valid_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def oof_predict(folds, X_all, y_all, c: float, seed: int, max_iter: int) -> dict:
    """标准 LOMO，返回 {fold_id: (y_test, p_test)}。标准化/LR 只在 train 上 fit（硬约束 5）。

    X_all 可以是 DataFrame（全局特征）或 callable(tr_ids, y_tr) -> DataFrame
    （用于 context 臂的逐折收缩参数重算，见 make_context_provider）。
    """
    out = {}
    for fold in folds:
        tr_ids = fold["train_event_ids"]
        X = X_all(tr_ids, y_all.loc[tr_ids].values) if callable(X_all) else X_all
        Xtr = X.loc[tr_ids]
        ytr = y_all.loc[tr_ids].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=max_iter, C=c, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        ids = fold["test_event_ids"]
        out[fold["fold_id"]] = (y_all.loc[ids].values, lr.predict_proba((X.loc[ids] - mu) / sd)[:, 1])
    return out


def make_context_provider(ev: pd.DataFrame):
    """context_v2 特征提供者：逐折用本折 train 的 p̄ 重算 history_rate（硬约束 5）。

    直接复用 run_baselines.build_context_matrix_v2，不重复实现特征逻辑。
    """
    import yaml
    from run_baselines import build_context_matrix_v2  # scripts/ 已在 sys.path（见 main）

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["features"]

    def provider(tr_ids, y_tr):
        return build_context_matrix_v2(ev, p_bar=float(y_tr.mean()),
                                       alpha=cfg["history_alpha"], clip=cfg["score_diff_clip"])
    return provider


def stats_from_oof(oof: dict) -> dict:
    """4 个统计量（A fold_mean / B pooled_raw / C pooled_rank / D pooled_z）。"""
    ys = [v[0] for v in oof.values()]
    ps = [v[1] for v in oof.values()]
    aucs = np.array([roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
                     for y, p in zip(ys, ps)])
    aucs = aucs[~np.isnan(aucs)]
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    rk = np.concatenate([(pd.Series(p).rank().to_numpy() - 0.5) / len(p) for p in ps])
    z = np.concatenate([(p - p.mean()) / (p.std() if p.std() > 0 else 1.0) for p in ps])
    return {
        "fold_mean": float(aucs.mean()),
        "fold_mean_std": float(aucs.std()),
        "fold_mean_se": float(aucs.std() / np.sqrt(len(aucs))),
        "n_folds_gt_half": int((aucs > 0.5).sum()),
        "n_folds": int(len(aucs)),
        "pooled_raw": float(roc_auc_score(y_all, p_all)),
        "pooled_rank": float(roc_auc_score(y_all, rk)),
        "pooled_z": float(roc_auc_score(y_all, z)),
    }


def perm_test(oof: dict, n_perm: int, rng: np.random.Generator, observed: dict) -> dict:
    """置换零分布 + 右侧 p 值。每折内独立打乱标签（保留折内类比例与折结构）。"""
    keys = list(oof.keys())
    metrics = ("fold_mean", "pooled_raw", "pooled_rank", "pooled_z")
    null = {m: np.empty(n_perm) for m in metrics}
    for i in range(n_perm):
        shuffled = {k: (rng.permutation(oof[k][0]), oof[k][1]) for k in keys}
        s = stats_from_oof(shuffled)
        for m in metrics:
            null[m][i] = s[m]
    return {m: {
        "null_mean": round(float(null[m].mean()), 4),
        "null_std": round(float(null[m].std()), 4),
        "null_p95": round(float(np.percentile(null[m], 95)), 4),
        "observed": round(float(observed[m]), 4),
        "p_right_tail": round(float((null[m] >= observed[m]).mean()), 4),
        "z_vs_null": round(float((observed[m] - null[m].mean()) / (null[m].std() + 1e-12)), 2),
    } for m in metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--no-ctx", action="store_true", help="跳过 context 臂")
    ap.add_argument("--no-fused", action="store_true", help="跳过融合臂（M3v2：context v2 + FER embedding）")
    args = ap.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # 复用 run_baselines 的特征构造
    with open(FOLDS_PATH, encoding="utf-8") as f:  # Windows GBK 默认编码会炸非 ASCII
        folds = json.load(f)["folds"]
    ev = load_valid_events()
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)

    arms: dict[str, tuple] = {}
    if not args.no_ctx:
        provider = make_context_provider(ev)
        arms["context_v2_lr"] = (provider, 1.0)
    for bb in BACKBONES:
        path = FEATURES_DIR / bb / "embeddings.npz"
        if not path.exists():
            print(f"跳过 {bb}（无缓存）", file=sys.stderr)
            continue
        c = np.load(path, allow_pickle=True)
        ids = c["event_ids"].astype(str)
        missing = set(ev["event_id"]) - set(ids)
        if missing:
            raise ValueError(f"{bb}: {len(missing)} 个有效样本不在缓存中")
        emb = pd.DataFrame(c["embedding"], columns=[f"e{j}" for j in range(c["embedding"].shape[1])], index=ids)
        arms[f"{bb}|emb"] = (emb, 0.01)
        if c["emo_probs"].shape[1] in (7, 8):
            emo = pd.DataFrame(c["emo_probs"], columns=[f"p{j}" for j in range(c["emo_probs"].shape[1])], index=ids)
            arms[f"{bb}|emo"] = (emo, 0.01)

    # 融合臂：context v2 6 维 + FER embedding 1280 维（M3v2，回答 H3 的融合增益）
    if not args.no_fused:
        provider = make_context_provider(ev)
        base = FEATURES_DIR / "enet_b0_8_best_vgaf" / "embeddings.npz"
        c = np.load(base, allow_pickle=True)
        ids = c["event_ids"].astype(str)
        emb = pd.DataFrame(c["embedding"], columns=[f"e{j}" for j in range(c["embedding"].shape[1])], index=ids)

        def fused_provider(tr_ids, y_tr, _ctx=provider, _emb=emb):
            return pd.concat([_ctx(tr_ids, y_tr), _emb], axis=1)

        arms["fused_v2|ctx+emb"] = (fused_provider, 0.01)

    rng = np.random.default_rng(args.seed)
    results = {}
    for name, (X, c) in arms.items():
        oof = oof_predict(folds, X, y_all, c, args.seed, args.max_iter)
        s = stats_from_oof(oof)
        nulls = perm_test(oof, args.n_perm, rng, s)
        results[name] = {"C": c, "stats": s, "null_and_p": nulls}
        print(f"[{name:>26}] fold_mean {s['fold_mean']:.4f}±{s['fold_mean_se']:.4f} "
              f"(null {nulls['fold_mean']['null_mean']:.4f}, p={nulls['fold_mean']['p_right_tail']:.3f}) | "
              f"pooled_rank {s['pooled_rank']:.4f} (null {nulls['pooled_rank']['null_mean']:.4f}, "
              f"p={nulls['pooled_rank']['p_right_tail']:.3f})")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "OOF 聚合统计量诊断：fold_mean 与 pooled 的差异、各统计量的置换零分布与 p 值",
        "n_perm": args.n_perm, "seed": args.seed, "n_valid_events": int(len(ev)),
        "recommended_statistic": "pooled_rank",
        "why": [
            "fold_mean 的零假设不是 0.5：单折 6-10 样本的 AUROC 在置换下期望偏离 0.5，须与其零分布比较",
            "pooled_raw 受折间分数尺度不一致影响（各折标准化与系数不同）",
            "pooled_rank 折内转秩，折间尺度无关且保留全部折内排序信息",
        ],
        "arms": results,
    }
    path = REPORTS_DIR / f"oof_stats_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
