#!/usr/bin/env python
"""Pilot smoke test 基线：Majority (B0)、Context LR (B1)、Context MLP (B2)、Emotion Prob LR (B3)。

用法（项目根目录）:
    python scripts/run_baselines.py --models majority,context_lr
    python scripts/run_baselines.py --models context_mlp --config src/config/context_mlp.yaml
    python scripts/run_baselines.py --models emotion_lr
    python scripts/run_baselines.py --validate   # 只检查 folds 与特征缓存，不训练

协议（plan.md §10.1、§22 Action 3）:
    - 逐折读取 folds/match_disjoint.json（LOMO，14 折），只统计有效样本
      （exclusion_reason 空且 face_path 非空，与 build_folds.py / 特征缓存口径一致）；
    - B0 Majority：预测常数 p = 训练集 goal 率（拟合参数只来自训练集，硬约束 5）；
    - B1 Context LR：四类 context（主客场暂缓，DATA_DICTIONARY §2.3），
      数值列标准化 + Beta 平滑历史率（α/β 只由训练集定，硬约束 5）；
    - B2 Context MLP：同一 6 维输入，低容量 MLP（src/models/context_mlp.py），
      训练走 src/training/train_loop.py（early stopping 只看 val AUROC），
      超参从 src/config/context_mlp.yaml 读；
    - B3 Emotion Prob LR：8 维情绪概率（FER 缓存 emo_probs）+ 标准化 + LR，
      回答"情绪概率本身有无预测信号"；
    - M1 FER embedding LR：1280 维 frozen embedding + 强 L2 LR（C=0.01 固定，
      理由见 run_fer_lr docstring），回答 H1（FER 表征有无额外信号）；
    - M3 Fused LR：context 6 维 + FER embedding 拼接（1286 维），同一 C=0.01，
      回答 H3（context 之外 FER 有无融合增益）；
    - M3v2 Fused LR v2：context 基座换成已定稿的 v2 六维（与 B1v2 同基座），
      其余协议与 M3 完全一致，使 M3v2 − M3 只归因于 context 基座；
    - B4 ImageNet LR：efficientnet_b0 frozen embedding 对照臂，与 M1 同协议，
      M1-B4 之差 = 情绪域预训练增益（H1 判据）；
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
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.context_mlp import ContextMLP, GroupedContextMLP  # noqa: E402
from src.training.train_loop import predict, train_one_fold  # noqa: E402

EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FOLDS_PATH = PROJECT_ROOT / "folds" / "match_disjoint.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

# Beta 平滑历史率（plan.md §6.6.1；pilot 固定值，正式阶段由训练集确定——
# 当前两个基线对 α/β 不敏感到需要调参的程度，先显式固定并记录在报告里）
BETA_ALPHA, BETA_BETA = 1.0, 1.0

FER_CACHE = PROJECT_ROOT / "features" / "fer" / "enet_b0_8_best_vgaf" / "embeddings.npz"
IMAGENET_CACHE = PROJECT_ROOT / "features" / "fer" / "imagenet_b0" / "embeddings.npz"
# 情绪类别顺序 = cache_fer_features.py 写入顺序（8 类，argmax/审计用同一顺序）
EMO_CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

CONTEXT_NUMERIC = ["round_number", "team_score_before", "opponent_score_before",
                   "kicks_taken_before", "history_rate", "history_missing"]
# history_missing=1 时 history_rate 无意义 → 由缺失指示列让 LR 自行学偏置；
# attempts=0（跨 3 季非主罚）是合法值，rate 按 plan 的平滑公式落在 0.5，不视为缺失

CONTEXT_V2 = ["sudden_death", "is_trailing", "kick_order_second", "score_diff",
              "has_history", "history_rate"]


def build_context_matrix_v2(ev: pd.DataFrame, p_bar: float, alpha: float = 5.0,
                            clip: int = 2) -> pd.DataFrame:
    """v2 context 特征（6 维，定义见 src/config/context_v2.yaml 头注）。

    样本口径与 v1 完全一致（385 行，含 history_missing=1 的 3 行——m009 的
    全部 miss 在其中，排除会击穿 fold 8 的负类）。无历史记录（attempts=0 或
    history_missing=1）时 history_rate 统一填收缩先验 p̄，has_history=0，
    与 attempts=0 的多数行同构处理，不设样本级排除。

    p̄（收缩目标）由调用方传入训练集估计值——本函数不估计任何统计量，
    保证 α/p̄/截断界全部来自 train（硬约束 5）。
    """
    sd = (ev["team_score_before"].astype(float) - ev["opponent_score_before"].astype(float)).clip(-clip, clip)
    rate = (ev["history_goals"] + alpha * p_bar) / (ev["history_attempts"] + alpha)
    no_hist = (ev["history_attempts"].fillna(0) <= 0) | (ev["history_missing"].fillna(0).astype(int) == 1)
    rate = rate.where(~no_hist, p_bar)
    return pd.DataFrame({
        "sudden_death": ev["sudden_death"].astype(float).to_numpy(),
        "is_trailing": (ev["team_score_before"] < ev["opponent_score_before"]).astype(float).to_numpy(),
        "kick_order_second": (ev["kick_order_in_round"] == "second").astype(float).to_numpy(),
        "score_diff": sd.to_numpy(),
        "has_history": (ev["history_attempts"] > 0).astype(float).to_numpy(),
        "history_rate": rate.to_numpy(),
    }, index=ev["event_id"])


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


def run_context_mlp(folds: list[dict], ev: pd.DataFrame, cfg: dict, seed: int) -> list[dict]:
    """B2：与 B1 相同的 6 维 context，低容量 MLP + val-AUROC early stopping。"""
    feat = cfg["features"]["context_numeric"]
    X_all = build_context_matrix(ev, cfg["features"]["beta_alpha"], cfg["features"]["beta_beta"])
    # 列顺序对齐配置（当前 build_context_matrix 已按 CONTEXT_NUMERIC 排列，此处显式校验）
    assert list(X_all.columns) == feat, f"特征列 {list(X_all.columns)} != 配置 {feat}"
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    mc, tc = cfg["model"], cfg["training"]
    rows = []
    for fold in folds:
        Xtr = X_all.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)  # 标准化器只来自 train（硬约束 5）
        Xtr_s, Xva_s, Xte_s = ((X_all.loc[fold[f"{s}_event_ids"]] - mu) / sd for s in ("train", "val", "test"))
        model = ContextMLP(in_dim=len(feat), hidden=mc["hidden"], dropout=mc["dropout"])
        state, hist = train_one_fold(
            model, Xtr_s.to_numpy(), ytr, Xva_s.to_numpy(),
            y_all.loc[fold["val_event_ids"]].values,
            seed=seed, max_epochs=tc["max_epochs"], patience=tc["patience"],
            lr=tc["lr"], weight_decay=tc["weight_decay"], batch_size=tc["batch_size"],
        )
        model.load_state_dict(state)
        for split, Xs in (("train", Xtr_s), ("val", Xva_s), ("test", Xte_s)):
            p = predict(model, Xs.to_numpy())
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "context_mlp", "split": split,
                         **m, "best_epoch": hist["best_epoch"],
                         "best_val_auroc": round(hist["best_val_auroc"], 4)})
    return rows


def run_context_lr_v2(folds: list[dict], ev_all: pd.DataFrame, seed: int,
                      cfg: dict) -> list[dict]:
    """B1v2：v2 context 特征 + LR，协议与 B1 相同（标准化只来自 train，硬约束 5）。

    α/p̄（history_rate 收缩参数）与截断界只由各折训练集估计。
    """
    feat = cfg["features"]["context_v2"]
    alpha = cfg["features"]["history_alpha"]
    clip = cfg["features"]["score_diff_clip"]
    y_all = (ev_all.set_index("event_id")["result"] == "goal").astype(int)
    rows = []
    for fold in folds:
        tr_ids = fold["train_event_ids"]
        p_bar = float(y_all.loc[tr_ids].mean())  # 收缩目标只来自 train
        X_all = build_context_matrix_v2(ev_all, p_bar=p_bar, alpha=alpha, clip=clip)
        assert list(X_all.columns) == feat, f"特征列 {list(X_all.columns)} != 配置 {feat}"
        Xtr = X_all.loc[tr_ids]
        ytr = y_all.loc[tr_ids].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        for split in ("train", "val", "test"):
            ids = fold[f"{split}_event_ids"]
            Xs = (X_all.loc[ids] - mu) / sd
            p = lr.predict_proba(Xs)[:, 1]
            m = metrics(y_all.loc[ids].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "context_lr_v2", "split": split,
                         **m, "p_bar": round(p_bar, 4),
                         "coef": dict(zip(feat, np.round(lr.coef_[0], 4).tolist())),
                         "intercept": round(float(lr.intercept_[0]), 4)})
    return rows


def run_context_mlp_v2(folds: list[dict], ev_all: pd.DataFrame, cfg: dict, seed: int) -> list[dict]:
    """B2v2：v2 context 特征 + ContextMLP，协议与 B2 相同。"""
    feat = cfg["features"]["context_v2"]
    alpha = cfg["features"]["history_alpha"]
    clip = cfg["features"]["score_diff_clip"]
    y_all = (ev_all.set_index("event_id")["result"] == "goal").astype(int)
    mc, tc = cfg["model"], cfg["training"]
    rows = []
    for fold in folds:
        tr_ids = fold["train_event_ids"]
        p_bar = float(y_all.loc[tr_ids].mean())
        X_all = build_context_matrix_v2(ev_all, p_bar=p_bar, alpha=alpha, clip=clip)
        assert list(X_all.columns) == feat, f"特征列 {list(X_all.columns)} != 配置 {feat}"
        Xtr = X_all.loc[tr_ids]
        ytr = y_all.loc[tr_ids].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        Xtr_s, Xva_s, Xte_s = ((X_all.loc[fold[f"{s}_event_ids"]] - mu) / sd for s in ("train", "val", "test"))
        model = ContextMLP(in_dim=len(feat), hidden=mc["hidden"], dropout=mc["dropout"])
        state, hist = train_one_fold(
            model, Xtr_s.to_numpy(), ytr, Xva_s.to_numpy(),
            y_all.loc[fold["val_event_ids"]].values,
            seed=seed, max_epochs=tc["max_epochs"], patience=tc["patience"],
            lr=tc["lr"], weight_decay=tc["weight_decay"], batch_size=tc["batch_size"],
        )
        model.load_state_dict(state)
        for split, Xs in (("train", Xtr_s), ("val", Xva_s), ("test", Xte_s)):
            p = predict(model, Xs.to_numpy())
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "context_mlp_v2", "split": split,
                         **m, "best_epoch": hist["best_epoch"],
                         "best_val_auroc": round(hist["best_val_auroc"], 4)})
    return rows


def run_context_lr_v2_inter(folds: list[dict], ev_all: pd.DataFrame, seed: int,
                            cfg: dict) -> list[dict]:
    """B1v2-int：v2 特征 + 2 个组内交互项（is_trailing×sudden_death, score_diff×sudden_death）。

    交互项在标准化**之后**相乘（对 0/1 × 连续量，标准化后交互的语义是
    "sudden_death 调制该特征的标准化偏移"），协议与 B1v2 相同。
    """
    feat = cfg["features"]["context_v2"]
    alpha = cfg["features"]["history_alpha"]
    clip = cfg["features"]["score_diff_clip"]
    y_all = (ev_all.set_index("event_id")["result"] == "goal").astype(int)
    inter_cols = ["is_trailing_x_sudden_death", "score_diff_x_sudden_death"]
    rows = []
    for fold in folds:
        tr_ids = fold["train_event_ids"]
        p_bar = float(y_all.loc[tr_ids].mean())
        X_all = build_context_matrix_v2(ev_all, p_bar=p_bar, alpha=alpha, clip=clip)
        mu, sd = X_all.loc[tr_ids].mean(axis=0), X_all.loc[tr_ids].std(axis=0).replace(0, 1.0)
        Xz = (X_all - mu) / sd  # 交互在标准化后构造
        Xz = pd.concat([Xz,
                        (Xz["is_trailing"] * Xz["sudden_death"]).rename(inter_cols[0]),
                        (Xz["score_diff"] * Xz["sudden_death"]).rename(inter_cols[1])], axis=1)
        feat_i = feat + inter_cols
        assert list(Xz.columns) == feat_i
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        lr.fit(Xz.loc[tr_ids], y_all.loc[tr_ids].values)
        for split in ("train", "val", "test"):
            ids = fold[f"{split}_event_ids"]
            p = lr.predict_proba(Xz.loc[ids])[:, 1]
            m = metrics(y_all.loc[ids].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "context_lr_v2_inter", "split": split,
                         **m, "p_bar": round(p_bar, 4),
                         "coef": dict(zip(feat_i, np.round(lr.coef_[0], 4).tolist())),
                         "intercept": round(float(lr.intercept_[0]), 4)})
    return rows


def run_context_mlp_grouped(folds: list[dict], ev_all: pd.DataFrame, cfg: dict, seed: int) -> list[dict]:
    """B2v2-g：分组双隐层 MLP（GroupedContextMLP），其余协议与 B2v2 相同。"""
    feat = cfg["features"]["context_v2"]
    alpha = cfg["features"]["history_alpha"]
    clip = cfg["features"]["score_diff_clip"]
    y_all = (ev_all.set_index("event_id")["result"] == "goal").astype(int)
    mc, tc = cfg["model"], cfg["training"]
    gp = mc.get("grouped", {})
    rows = []
    for fold in folds:
        tr_ids = fold["train_event_ids"]
        p_bar = float(y_all.loc[tr_ids].mean())
        X_all = build_context_matrix_v2(ev_all, p_bar=p_bar, alpha=alpha, clip=clip)
        assert list(X_all.columns) == feat
        Xtr = X_all.loc[tr_ids]
        ytr = y_all.loc[tr_ids].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        Xtr_s, Xva_s, Xte_s = ((X_all.loc[fold[f"{s}_event_ids"]] - mu) / sd for s in ("train", "val", "test"))
        model = GroupedContextMLP(hidden_situ=gp.get("hidden_situ", 2),
                                  hidden_hist=gp.get("hidden_hist", 1),
                                  dropout=mc["dropout"])
        state, hist = train_one_fold(
            model, Xtr_s.to_numpy(), ytr, Xva_s.to_numpy(),
            y_all.loc[fold["val_event_ids"]].values,
            seed=seed, max_epochs=tc["max_epochs"], patience=tc["patience"],
            lr=tc["lr"], weight_decay=tc["weight_decay"], batch_size=tc["batch_size"],
        )
        model.load_state_dict(state)
        for split, Xs in (("train", Xtr_s), ("val", Xva_s), ("test", Xte_s)):
            p = predict(model, Xs.to_numpy())
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "context_mlp_v2_grouped", "split": split,
                         **m, "best_epoch": hist["best_epoch"],
                         "best_val_auroc": round(hist["best_val_auroc"], 4)})
    return rows


def load_fer_cache(cache_path: Path = FER_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读 FER/ImageNet 特征缓存，返回 (emo_probs_df, embedding_df)，index=event_id。

    缓存的 event_ids 生成口径与 load_valid_events 一致，但仍按 event_id 对齐
    （而非位置拼接），events.csv 变动后能立刻暴露不一致而不是静默错位。
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"缺特征缓存: {cache_path}（先跑 scripts/cache_fer_features.py）")
    cache = np.load(cache_path, allow_pickle=True)
    ids = cache["event_ids"].astype(str)
    probs = cache["emo_probs"]
    emo = (pd.DataFrame(probs, columns=EMO_CLASSES, index=ids)
           if probs.shape[1] == len(EMO_CLASSES) else pd.DataFrame(index=ids))  # ImageNet 臂无情绪头
    emb = pd.DataFrame(cache["embedding"], columns=[f"emb_{j:04d}" for j in range(cache["embedding"].shape[1])], index=ids)
    return emo, emb


def run_emotion_lr(folds: list[dict], ev: pd.DataFrame, seed: int) -> list[dict]:
    """B3：8 维情绪概率 + LR，协议与 B1 完全一致（标准化只来自 train，硬约束 5）。

    emo_probs 已在 [0,1] 且行和为 1，标准化主要为与 B1 协议一致（多组 LR 之间
    的 C 语义可比），对 LR 本身影响很小。
    """
    emo, _ = load_fer_cache()
    missing = set(ev["event_id"]) - set(emo.index)
    if missing:
        raise ValueError(f"{len(missing)} 个有效样本不在特征缓存中（events.csv 是否已变更？需重跑 cache_fer_features.py）: {sorted(missing)[:5]}...")
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    rows = []
    for fold in folds:
        Xtr = emo.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        for split in ("train", "val", "test"):
            Xs = (emo.loc[fold[f"{split}_event_ids"]] - mu) / sd
            p = lr.predict_proba(Xs)[:, 1]
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "emotion_lr", "split": split,
                         **m, "coef": dict(zip(EMO_CLASSES, np.round(lr.coef_[0], 4).tolist())),
                         "intercept": round(float(lr.intercept_[0]), 4)})
    return rows


def run_fer_lr(folds: list[dict], ev: pd.DataFrame, seed: int) -> list[dict]:
    """M1：1280 维 frozen FER embedding + 强 L2 LR（核心臂，回答 H1）。

    正则策略（选定理由）：n_train≈96 vs d=1280，过拟合是唯一风险，而 val=单场
    （6-10 样本）已被 B2 证明不可靠（7/14 折 best_epoch=1）——所以不做逐折
    C 扫描（等于在噪声上调参），而是固定强 L2：C=0.01（比 B1/B3 的 C=1.0 强
    两个数量级），等价于先验收缩到零附近，让 LR 近似 ridge 回归。确定性、
    无 val 依赖、跨折可比。PCA 降维备选未采用：主成分方向无监督，可能丢掉
    与结果对齐的低方差方向，且引入"保留多少维"这个同样只能拍脑袋的超参。

    逐折表不记录 1280 维系数（CSV 过大），只记 ||coef||_2 与 top-10 |coef| 维度名。
    """
    _, emb = load_fer_cache()
    missing = set(ev["event_id"]) - set(emb.index)
    if missing:
        raise ValueError(f"{len(missing)} 个有效样本不在特征缓存中（events.csv 是否已变更？需重跑 cache_fer_features.py）: {sorted(missing)[:5]}...")
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    rows = []
    for fold in folds:
        Xtr = emb.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)  # 标准化器只来自 train
        lr = LogisticRegression(max_iter=5000, C=0.01, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        coef = lr.coef_[0]
        top10 = sorted(emb.columns, key=lambda c: abs(coef[emb.columns.get_loc(c)]), reverse=True)[:10]
        for split in ("train", "val", "test"):
            Xs = (emb.loc[fold[f"{split}_event_ids"]] - mu) / sd
            p = lr.predict_proba(Xs)[:, 1]
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "fer_lr", "split": split,
                         **m, "coef_l2": round(float(np.linalg.norm(coef)), 4),
                         "top10_abs_coef": top10})
    return rows


def run_fused_lr_v2(folds: list[dict], ev_all: pd.DataFrame, seed: int, cfg: dict) -> list[dict]:
    """M3v2：context **v2** 6 维 + FER embedding 1280 维拼接 + 强 L2 LR（回答 H3 融合增益）。

    与 M3（v1 context）唯一差别是 context 基座换成已定稿的 v2 六维（B1v2 主臂 0.577）。
    正则与协议刻意与 M3 完全一致（同一 C=0.01 施加于全部 1286 维、标准化只来自 train），
    这样 M3v2 − M3 的差就只归因于 context 基座，而不是融合超参。

    α（history_rate 收缩强度）与 p̄（收缩目标）仍只由各折训练集估计（硬约束 5）；
    v2 的 build_context_matrix_v2 在每折内用本折 train 的 p̄ 重建。
    """
    feat = cfg["features"]["context_v2"]
    alpha = cfg["features"]["history_alpha"]
    clip = cfg["features"]["score_diff_clip"]
    _, emb = load_fer_cache()
    missing = set(ev_all["event_id"]) - set(emb.index)
    if missing:
        raise ValueError(f"{len(missing)} 个有效样本不在特征缓存中（events.csv 是否已变更？需重跑 cache_fer_features.py）: {sorted(missing)[:5]}...")
    y_all = (ev_all.set_index("event_id")["result"] == "goal").astype(int)
    rows = []
    for fold in folds:
        tr_ids = fold["train_event_ids"]
        p_bar = float(y_all.loc[tr_ids].mean())  # 收缩目标只来自 train
        ctx = build_context_matrix_v2(ev_all, p_bar=p_bar, alpha=alpha, clip=clip)
        assert list(ctx.columns) == feat, f"特征列 {list(ctx.columns)} != 配置 {feat}"
        X_all = pd.concat([ctx, emb], axis=1)
        assert list(X_all.columns) == feat + list(emb.columns)  # 列对齐防拼接错位
        Xtr = X_all.loc[tr_ids]
        ytr = y_all.loc[tr_ids].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=5000, C=0.01, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        coef = lr.coef_[0]
        top10 = sorted(X_all.columns, key=lambda c: abs(coef[X_all.columns.get_loc(c)]), reverse=True)[:10]
        ctx_l2 = float(np.linalg.norm(coef[:len(feat)]))   # context 部分的系数范数
        fer_l2 = float(np.linalg.norm(coef[len(feat):]))   # embedding 部分
        for split in ("train", "val", "test"):
            Xs = (X_all.loc[fold[f"{split}_event_ids"]] - mu) / sd
            p = lr.predict_proba(Xs)[:, 1]
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "fused_lr_v2", "split": split,
                         **m, "p_bar": round(p_bar, 4),
                         "coef_l2": round(float(np.linalg.norm(coef)), 4),
                         "ctx_coef_l2": round(ctx_l2, 4), "fer_coef_l2": round(fer_l2, 4),
                         "n_ctx_in_top10": int(sum(1 for c in top10 if c in feat)),
                         "top10_abs_coef": top10})
    return rows


def run_fused_lr(folds: list[dict], ev: pd.DataFrame, seed: int) -> list[dict]:
    """M3：context 6 维 + FER embedding 1280 维拼接 + 强 L2 LR（回答 H3 融合增益）。

    正则策略：与 M1 相同的固定 C=0.01（单一 C 施加于全部 1286 维）。拼接维度下
    context 特征会被同样的强收缩压制——这是有意的保守选择：M3 的判读是
    "在同等正则强度下融合是否仍不低于 M1/B1"，而不是"调出最优融合超参"。
    若 M3 test 低于 M1，pilot 阶段不改协议（硬约束：不追求提升而调协议），
    扩数据后再议。标准化只来自 train（硬约束 5）。

    逐折表记录 coef_l2 与 top-10 |coef| 维度名（可看出 context 是否进入 top10）。
    """
    _, emb = load_fer_cache()
    missing = set(ev["event_id"]) - set(emb.index)
    if missing:
        raise ValueError(f"{len(missing)} 个有效样本不在特征缓存中（events.csv 是否已变更？需重跑 cache_fer_features.py）: {sorted(missing)[:5]}...")
    ctx = build_context_matrix(ev)
    assert list(ctx.columns) == CONTEXT_NUMERIC
    X_all = pd.concat([ctx, emb], axis=1)
    assert list(X_all.columns) == CONTEXT_NUMERIC + list(emb.columns)  # 列对齐防拼接错位
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    rows = []
    for fold in folds:
        Xtr = X_all.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=5000, C=0.01, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        coef = lr.coef_[0]
        top10 = sorted(X_all.columns, key=lambda c: abs(coef[X_all.columns.get_loc(c)]), reverse=True)[:10]
        for split in ("train", "val", "test"):
            Xs = (X_all.loc[fold[f"{split}_event_ids"]] - mu) / sd
            p = lr.predict_proba(Xs)[:, 1]
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "fused_lr", "split": split,
                         **m, "coef_l2": round(float(np.linalg.norm(coef)), 4),
                         "top10_abs_coef": top10})
    return rows


def run_imagenet_lr(folds: list[dict], ev: pd.DataFrame, seed: int) -> list[dict]:
    """B4：ImageNet 对照臂（efficientnet_b0 frozen embedding，1280 维）+ 与 M1
    完全相同的协议（C=0.01 强 L2）。M1 vs B4 之差 = FER 情绪域预训练的增益（H1 判据）。
    """
    _, emb = load_fer_cache(IMAGENET_CACHE)
    missing = set(ev["event_id"]) - set(emb.index)
    if missing:
        raise ValueError(f"{len(missing)} 个有效样本不在 ImageNet 缓存中（需重跑 cache_fer_features.py --imagenet）: {sorted(missing)[:5]}...")
    y_all = (ev.set_index("event_id")["result"] == "goal").astype(int)
    rows = []
    for fold in folds:
        Xtr = emb.loc[fold["train_event_ids"]]
        ytr = y_all.loc[fold["train_event_ids"]].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0).replace(0, 1.0)
        lr = LogisticRegression(max_iter=5000, C=0.01, random_state=seed)
        lr.fit((Xtr - mu) / sd, ytr)
        coef = lr.coef_[0]
        top10 = sorted(emb.columns, key=lambda c: abs(coef[emb.columns.get_loc(c)]), reverse=True)[:10]
        for split in ("train", "val", "test"):
            Xs = (emb.loc[fold[f"{split}_event_ids"]] - mu) / sd
            p = lr.predict_proba(Xs)[:, 1]
            m = metrics(y_all.loc[fold[f"{split}_event_ids"]].values, p)
            rows.append({"fold_id": fold["fold_id"], "model": "imagenet_lr", "split": split,
                         **m, "coef_l2": round(float(np.linalg.norm(coef)), 4),
                         "top10_abs_coef": top10})
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


def save_outputs(all_rows: list[dict], models: list[str], seed: int,
                 cfg: dict | None = None) -> tuple[Path, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(models)

    per_fold = pd.DataFrame([{k: v for k, v in r.items() if k not in ("coef", "intercept", "top10_abs_coef")}
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
        "config": cfg,
        "folds": {"path": FOLDS_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256_16": folds_sha},
        "events_fingerprint": ids_sha,
        "n_valid_events": int(len(ev)),
        "context_features": CONTEXT_NUMERIC,
        "beta_smoothing": {"alpha": BETA_ALPHA, "beta": BETA_BETA},
        "fer_lr": {"C": 0.01, "max_iter": 5000,
                   "rationale": "n_train~96 vs d=1280，固定强 L2；val 单场不可靠故不做逐折 C 扫描"},
        "fused_lr": {"C": 0.01, "max_iter": 5000,
                     "rationale": "与 M1 同一 C 施加于全部 1286 维；M3 判读为同正则强度下融合是否不低于单臂，不做融合超参调优"},
        "fused_lr_v2": {"C": 0.01, "max_iter": 5000,
                        "rationale": "context 基座换 v2 六维（与 B1v2 主臂一致），正则/协议与 M3 完全相同，使 M3v2-M3 只归因于 context 基座"},
        "context_v2": {"history_alpha": "由配置", "note": "B1v2/M3v2 的 p̄（收缩目标）逐折只由 train 估计（硬约束 5）"},
        "imagenet_lr": {"C": 0.01, "max_iter": 5000,
                        "rationale": "与 M1 完全同协议，仅特征来源不同（ImageNet vs FER 预训练）"},
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
                    help="逗号分隔: majority,context_lr,context_mlp,emotion_lr,fer_lr,fused_lr,fused_lr_v2,imagenet_lr,context_lr_v2,context_mlp_v2（默认前两个）")
    ap.add_argument("--config", default="src/config/context_mlp.yaml",
                    help="context_mlp 的超参配置（plan §8：配置驱动）")
    ap.add_argument("--seed", type=int, default=2026, help="随机种子（LR/lbfgs 几乎无随机性，仅为可复现记录）")
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
    unknown = set(models) - {"majority", "context_lr", "context_mlp", "emotion_lr",
                             "fer_lr", "fused_lr", "fused_lr_v2", "imagenet_lr",
                             "context_lr_v2", "context_mlp_v2",
                             "context_lr_v2_inter", "context_mlp_v2_grouped"}
    if unknown:
        print(f"未知模型: {sorted(unknown)}（可用: majority, context_lr, context_mlp, emotion_lr, fer_lr, fused_lr, fused_lr_v2, imagenet_lr, context_lr_v2, context_mlp_v2, context_lr_v2_inter, context_mlp_v2_grouped）", file=sys.stderr)
        return 1

    cfg = None
    if {"context_mlp", "context_lr_v2", "context_mlp_v2", "fused_lr_v2",
        "context_lr_v2_inter", "context_mlp_v2_grouped"} & set(models):
        cfg_path = PROJECT_ROOT / args.config
        if not cfg_path.exists():
            print(f"缺配置文件: {cfg_path}", file=sys.stderr)
            return 1
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    all_rows: list[dict] = []
    if "majority" in models:
        all_rows += run_majority(folds, ev)
    if "context_lr" in models:
        all_rows += run_context_lr(folds, ev, args.seed)
    if "context_mlp" in models:
        all_rows += run_context_mlp(folds, ev, cfg, args.seed)
    if "context_lr_v2" in models:
        all_rows += run_context_lr_v2(folds, ev, args.seed, cfg)
    if "context_mlp_v2" in models:
        all_rows += run_context_mlp_v2(folds, ev, cfg, args.seed)
    if "context_lr_v2_inter" in models:
        all_rows += run_context_lr_v2_inter(folds, ev, args.seed, cfg)
    if "context_mlp_v2_grouped" in models:
        all_rows += run_context_mlp_grouped(folds, ev, cfg, args.seed)
    if "emotion_lr" in models:
        all_rows += run_emotion_lr(folds, ev, args.seed)
    if "fer_lr" in models:
        all_rows += run_fer_lr(folds, ev, args.seed)
    if "fused_lr" in models:
        all_rows += run_fused_lr(folds, ev, args.seed)
    if "fused_lr_v2" in models:
        all_rows += run_fused_lr_v2(folds, ev, args.seed, cfg)
    if "imagenet_lr" in models:
        all_rows += run_imagenet_lr(folds, ev, args.seed)
    if not all_rows:
        print("没有可运行的模型", file=sys.stderr)
        return 1

    table_path, report_path = save_outputs(all_rows, models, args.seed, cfg)
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