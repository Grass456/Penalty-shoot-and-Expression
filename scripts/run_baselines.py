#!/usr/bin/env python
"""Pilot smoke test 基线：Majority (B0) 与 Context Logistic Regression (B1)。

用法（项目根目录）:
    python scripts/run_baselines.py --models majority,context_lr
    python scripts/run_baselines.py --validate   # 只检查 folds 与特征缓存，不训练

协议（plan.md §10.1、§22 Action 3）:
    - 逐折读取 folds/match_disjoint.json（LOMO，14 折），只统计有效样本
      （exclusion_reason 空且 face_path 非空，与 build_folds.py / 特征缓存口径一致）；
    - B0 Majority：预测常数 p = 训练集 goal 率（拟合参数只来自训练集，硬约束 5）；
    - B1 Context LR：四类 context（主客场暂缓，DATA_DICTIONARY §2.3），
      数值列标准化 + Beta 平滑历史率（α/β 只由训练集定，硬约束 5）；
    - 指标：AUROC / AUPRC / Balanced Acc / Brier（硬约束 7），逐折 + 聚合；
    - 结果落盘 results/tables/ 与 results/reports/，报告里记录配置与数据指纹。

pilot 结果只用于发现实现问题和估算信号强度，不作最终论文结论（plan.md §10.1）。
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

# Beta 平滑历史率（plan.md §6.6.1；pilot 固定值，正式阶段由训练集确定——
# 当前两个基线对 α/β 不敏感到需要调参的程度，先显式固定并记录在报告里）
BETA_ALPHA, BETA_BETA = 1.0, 1.0

CONTEXT_NUMERIC = ["round_number", "team_score_before", "opponent_score_before",
                   "kicks_taken_before", "history_rate", "history_missing"]
# history_missing=1 时 history_rate 无意义 → 由缺失指示列让 LR 自行学偏置；
# attempts=0（跨 3 季非主罚）是合法值，rate 按 plan 的平滑公式落在 0.5，不视为缺失


def load_valid_events() -> pd.DataFrame:
    """有效样本口径与 build_folds.py / cache_fer_features.py 一致。"""
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def build_context_matrix(ev: pd.DataFrame, alpha: float = BETA_ALPHA, beta: float = BETA_BETA) -> pd.DataFrame:
    """从 events 行构造 context 特征帧（列顺序 = CONTEXT_NUMERIC）。

    注意 .to_numpy()：直接传 Series 会被构造器按索引对齐，
    而 index=ev["event_id"] 与 RangeIndex 对不上，全部变 NaN。
    """
    rate = (ev["history_goals"] + alpha) / (ev["history_attempts"] + alpha + beta)
    return pd.DataFrame({
        "round_number": ev["round_number"].astype(float).to_numpy(),
        "team_score_before": ev["team_score_before"].astype(float).to_numpy(),
        "opponent_score_before": ev["opponent_score_before"].astype(float).to_numpy(),
        "kicks_taken_before": ev["kicks_taken_before"].astype(float).to_numpy(),
        "history_rate": rate.to_numpy(),
        "history_missing": ev["history_missing"].astype(float).to_numpy(),
    }, index=ev["event_id"])


def metrics(y_true: np.ndarray, p_hat: np.ndarray) -> dict:
    """AUROC / AUPRC / Balanced Acc / Brier。p_hat 为常数时 sklearn 也能算 AUROC。"""
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


def run_majority(folds: list[dict], ev: pd.DataFrame) -> list[dict]:
    """B0：p = 训练集 goal 率（每折重新拟合，参数只来自 train，硬约束 5）。"""
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    rows = []
    for fold in folds:
        ytr = y_all.loc[fold["train_event_ids"]].values
        p_train = float(ytr.mean())
        for split in ("train", "val", "test"):
            ids = fold[f"{split}_event_ids"]
            m = metrics(y_all.loc[ids].values, np.full(len(ids), p_train))
            rows.append({"fold_id": fold["fold_id"], "model": "majority", "split": split,
                         "param": round(p_train, 6), **m})
    return rows


def run_context_lr(folds: list[dict], ev: pd.DataFrame, seed: int) -> list[dict]:
    """B1：标准化 + LR，全部只在 train 上 fit（硬约束 5、6）。"""
    X_all = build_context_matrix(ev)
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    rows = []
    for fold in folds:
        Xtr = X_all.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)  # 标准化器只来自 train
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        for split in ("train", "val", "test"):
            Xs = (X_all.loc[fold[f"{split}_event_ids"]] - mu) / sd
            p = lr.predict_proba(Xs)[:, 1]
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "context_lr", "split": split,
                         **m, "coef": dict(zip(CONTEXT_NUMERIC, np.round(lr.coef_[0], 4).tolist())),
                         "intercept": round(float(lr.intercept_[0]), 4)})
    return rows


def aggregate(rows: list[dict], model: str, split: str = "test") -> dict:
    """test 折聚合：均值 ± 标准差（AUROC 等逐折值）。"""
    sub = [r for r in rows if r["model"] == model and r["split"] == split]
    out = {"model": model, "split": split, "n_folds": len(sub)}
    for k in ("auroc", "auprc", "balanced_acc", "brier"):
        vals = [r[k] for r in sub if r[k] is not None]
        out[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                  "n_valid_folds": len(vals)} if vals else None
    return out


def save_outputs(all_rows: list[dict], models: list[str], seed: int) -> tuple[Path, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(models)

    per_fold = pd.DataFrame([{k: v for k, v in r.items() if k not in ("coef", "intercept")}
                             for r in all_rows])
    table_path = TABLES_DIR / f"baselines_{tag}_{stamp}.csv"
    per_fold.to_csv(table_path, index=False)

    agg = [aggregate(all_rows, m, s) for m in models for s in ("train", "val", "test")]
    with open(FOLDS_PATH, "rb") as f:
        folds_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    ev = load_valid_events()
    ids_sha = hashlib.sha256(",".join(sorted(ev["event_id"])).encode()).hexdigest()[:16]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "seed": seed,
        "folds": {"path": FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256_16": folds_sha},
        "events_fingerprint": ids_sha,
        "n_valid_events": int(len(ev)),
        "context_features": CONTEXT_NUMERIC,
        "beta_smoothing": {"alpha": BETA_ALPHA, "beta": BETA_BETA},
        "aggregates": agg,
        "per_fold_path": table_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    report_path = REPORTS_DIR / f"baselines_{tag}_{stamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return table_path, report_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="majority,context_lr",
                    help="逗号分隔: majority,context_lr（默认全部）")
    ap.add_argument("--seed", type=int, default=2026, help="LR 随机种子（lbfgs 几乎无随机性，仅为可复现记录）")
    ap.add_argument("--validate", action="store_true", help="只检查 folds 与 events 一致性，不训练")
    args = ap.parse_args()

    if not FOLDS_PATH.exists():
        print(f"缺 folds 文件: {FOLDS_PATH}", file=sys.stderr)
        return 1
    with open(FOLDS_PATH, encoding="utf-8") as f:
        folds_doc = json.load(f)
    folds = folds_doc["folds"]

    ev = load_valid_events()
    if args.validate:
        print(f"folds: {folds_doc['strategy']}, {len(folds)} 折；有效样本 {len(ev)}")
        covered = sorted(e for fold in folds for e in fold["test_event_ids"])
        if covered != sorted(ev["event_id"]):
            print("  !! folds 的 test 覆盖与当前有效样本不一致，请重新生成 folds", file=sys.stderr)
            return 1
        print("folds 与 events.csv 一致")
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - {"majority", "context_lr"}
    if unknown:
        print(f"未知模型: {sorted(unknown)}（可用: majority, context_lr）", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    if "majority" in models:
        all_rows += run_majority(folds, ev)
    if "context_lr" in models:
        all_rows += run_context_lr(folds, ev, args.seed)
    if not all_rows:
        print("没有可运行的模型", file=sys.stderr)
        return 1

    table_path, report_path = save_outputs(all_rows, models, args.seed)
    for m in models:
        for split in ("val", "test"):
            a = aggregate(all_rows, m, split)
            if a["auroc"] is None:
                continue
            fmt = lambda d: f"{d['mean']:.3f}±{d['std']:.3f}" if d else "n/a"
            print(f"[{m} / {split}] AUROC {fmt(a['auroc'])}  "
                  f"AUPRC {fmt(a['auprc'])}  "
                  f"BalAcc {fmt(a['balanced_acc'])}  "
                  f"Brier {fmt(a['brier'])}")
    print(f"\n逐折表: {table_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"报告:   {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())