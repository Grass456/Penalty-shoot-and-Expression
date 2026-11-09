#!/usr/bin/env python
"""M4-stacked 融合评估：微调 FER 支 + context 支（LR / MLP 两变体）→ 3 参数 logit 融合。

预注册判读标准（跑前写定）：results/reports/m4_fusion_criteria_preregistered.txt
结构说明与对 plan §7.4 的偏离见该文件第 1 节。

臂：
    - M4_lr   ：context 支 = B1v2 口径（六维 + LR C=1），融合层在**内层 val** 上拟合 3 参数；
    - M4_mlp  ：context 支 = B2v2 口径（ContextMLP 16/0.3），同上；
    - M4_lr_tr / M4_mlp_tr：描述性敏感性，融合层改在 train 上拟合；
    - M2      ：FER 支单独（= 本脚本训出的 FER 支 logits），免费获得，用于 M4 − M2 比较；
    - B1v2 / B2v2 / M1：只作前置一致性的哨兵（必须复现主表 181311）。

关键设计：**每个 (split seed, 折) 的 FER 支只训一次**，两个 context 变体与两种融合拟合位置
共用同一份 ℓ_f；因此 `M4_lr − M4_mlp` 的差只归因于 context 支，`M4_*_tr − M4_*` 只归因于
融合层的拟合位置。

用法（项目根目录，需 PYTHONIOENCODING=utf-8）:
    python scripts/eval_m4_fusion.py --smoke        # 1 seed × 1 折，量级检查
    python scripts/eval_m4_fusion.py --n-perm 0     # 只算观测值（10 seed × 5 折，约 10 min）
    python scripts/eval_m4_fusion.py                # 全量：观测 + 100 次置换（约 2 h）

输出: results/tables/m4_fusion_<stamp>.csv、results/reports/m4_fusion_<stamp>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# cuBLAS 确定性要求（与 eval_m2_finetune 同）：必须在 torch/CUDA 初始化前设置
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import eval_repeated_splits as ers  # noqa: E402
from run_baselines import build_context_matrix_v2  # noqa: E402
from src.data.face_images import load_face_tensor  # noqa: E402
from src.training.finetune_loop import logits_images, train_one_fold_images  # noqa: E402
from src.training.train_loop import logits as mlp_logits  # noqa: E402
from src.training.train_loop import train_one_fold  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "m4_fusion.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
CRITERIA_PATH = REPORTS_DIR / "m4_fusion_criteria_preregistered.txt"
M2_CSV = PROJECT_ROOT / "results" / "tables" / "m2_finetune_20261106_145706.csv"

# 哨兵臂定义必须与主表 ARMS 逐字段一致，前置断言才有意义
M1_ARM = {"name": "M1", "model": "lr", "feats": ["emb:enet_b0_8_best_vgaf"], "C": 0.01, "subset": False}
B1V2_ARM = {"name": "B1v2", "model": "lr", "feats": ["ctx"], "C": 1.0, "subset": False}
B2V2_ARM = {"name": "B2v2", "model": "mlp", "feats": ["ctx"], "C": None, "subset": False}

FUSION_ARMS = ("M4_lr", "M4_mlp", "M4_lr_tr", "M4_mlp_tr")
ALPHA = 0.05


def setup_determinism() -> None:
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def load_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- 数据

def load_all(cfg: dict):
    """事件数据 + 人脸张量，带行序断言（与 eval_m2_finetune.load_all 同）。"""
    import torch
    data = ers.load_data()
    X, ids = load_face_tensor(data["valid"], PROJECT_ROOT, img_size=int(cfg["fer"]["img_size"]))
    if ids != data["ids"]:
        raise AssertionError("人脸张量行顺序与 data['ids'] 不一致 —— folds 索引会错位")
    if not torch.isfinite(X).all():
        raise AssertionError("人脸张量含 NaN/Inf")
    return data, X, ids


def build_pristine(cfg: dict, device: str):
    from cache_fer_features import build_fer
    name = cfg["fer"]["backbone"]
    backbone, _, emb_dim, _, prep = build_fer(name, device)
    if int(prep[0]) != int(cfg["fer"]["img_size"]):
        raise AssertionError(f"配置 img_size 与 backbone 的 {prep[0]} 不符")
    return backbone, emb_dim


def context_matrix(data: dict, fold: dict, y: np.ndarray) -> np.ndarray:
    """context v2 六维（p̄ 只由本折 train 估计，硬约束 5）。"""
    fc = data["cfg"]["features"]
    ctx = build_context_matrix_v2(data["valid"], p_bar=float(y[fold["train"]].mean()),
                                  alpha=fc["history_alpha"], clip=fc["score_diff_clip"])
    assert list(ctx.columns) == fc["context_v2"]
    return ctx.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------- 支 logits

def fer_branch_logits(X, fold: dict, y: np.ndarray, seed: int, cfg: dict,
                      pristine, emb_dim: int, device: str,
                      record_history: bool = False) -> tuple[dict, dict]:
    """训练 FER 支（M2 结构），返回 (train/val/test logits, 训练信息)。"""
    import torch
    from src.models.fer_finetune import FinetuneFER

    tr, va, te = fold["train"], fold["val"], fold["test"]
    mc, tc = cfg["fer_branch"], cfg["fer_branch"]
    torch.manual_seed(int(seed))          # 必须在构造模型之前（见 mlp-init-determinism-fix）
    model = FinetuneFER(copy.deepcopy(pristine), emb_dim=emb_dim,
                        unfreeze_depth=int(mc["unfreeze_depth"]),
                        bn_train=bool(mc["bn_train"]))
    _, info = train_one_fold_images(
        model, X[tr], y[tr], X[va], y[va], seed=seed,
        max_epochs=int(tc["max_epochs"]), patience=int(tc["patience"]),
        lr_head=float(tc["lr_head"]), lr_backbone=float(tc["lr_backbone"]),
        weight_decay=float(tc["weight_decay"]), batch_size=int(tc["batch_size"]),
        device=device, amp=bool(tc["amp"]))
    out = {s: logits_images(model, X[idx], device=device)
           for s, idx in (("train", tr), ("val", va), ("test", te))}
    keep = {"best_epoch": info["best_epoch"], "best_val_auroc": info["best_val_auroc"],
            "n_epochs_run": info["n_epochs_run"]}
    if record_history:
        keep["history"] = info["history"]
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out, keep


def ctx_branch_logits(data: dict, fold: dict, y: np.ndarray, seed: int,
                      variant: str, model_cfg: dict) -> dict:
    """训练 context 支（LR 或 MLP），返回 train/val/test logits。

    两个变体都与主表逐字段同构：LR 走 ers.fit_predict 的 LR 分支，MLP 走同一 ContextMLP
    构造顺序（先 manual_seed 再建模，否则不可复现）。
    """
    tr, va, te = fold["train"], fold["val"], fold["test"]
    X = context_matrix(data, fold, y)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    if variant == "lr":
        lr = LogisticRegression(max_iter=int(model_cfg["max_iter"]), C=float(model_cfg["C"]),
                                random_state=seed)
        lr.fit(Z[tr], y[tr])
        return {s: lr.decision_function(Z[idx]) for s, idx in
                (("train", tr), ("val", va), ("test", te))}

    import torch
    from src.models.context_mlp import ContextMLP
    torch.set_num_threads(1)
    torch.manual_seed(seed)               # 先 seed 再构造（2026-11-04 勘误）
    model = ContextMLP(in_dim=Z.shape[1], hidden=int(model_cfg["hidden"]),
                       dropout=float(model_cfg["dropout"]))
    train_one_fold(model, Z[tr], y[tr], Z[va], y[va], seed=seed,
                   max_epochs=int(model_cfg["max_epochs"]), patience=int(model_cfg["patience"]),
                   lr=float(model_cfg["lr"]), weight_decay=float(model_cfg["weight_decay"]),
                   batch_size=model_cfg["batch_size"])
    return {s: mlp_logits(model, Z[idx]) for s, idx in
            (("train", tr), ("val", va), ("test", te))}


# ---------------------------------------------------------------- 融合

def standardize_branch(lg: dict, tr_idx: np.ndarray) -> tuple[dict, float, float]:
    """用 train logits 的均值/标准差标准化三分支 logits（硬约束 5）。"""
    m, s = float(lg["train"].mean()), float(lg["train"].std())
    if s == 0:
        s = 1.0
    return {k: (v - m) / s for k, v in lg.items()}, m, s


def fuse(zf: dict, zc: dict, y: np.ndarray, fold: dict, fit_on: str,
         C: float, max_iter: int, seed: int) -> tuple[np.ndarray, dict]:
    """3 参数 logit 融合：在 fit_on（'val' 或 'train'）上拟合，返回 test 概率与权重。"""
    idx = fold[fit_on]
    Z = np.column_stack([zf[fit_on], zc[fit_on]])
    if len(np.unique(y[idx])) < 2:       # 置换下可能出现单类 val
        return np.full(len(fold["test"]), float(y[idx].mean())), \
            {"w_fer": float("nan"), "w_ctx": float("nan"), "b": float("nan"), "degenerate": True}
    fus = LogisticRegression(max_iter=int(max_iter), C=float(C), random_state=seed)
    fus.fit(Z, y[idx])
    Zte = np.column_stack([zf["test"], zc["test"]])
    p = fus.predict_proba(Zte)[:, 1]
    return p, {"w_fer": float(fus.coef_[0][0]), "w_ctx": float(fus.coef_[0][1]),
               "b": float(fus.intercept_[0]), "degenerate": False}


def all_arms_one_fold(X, data: dict, fold: dict, y: np.ndarray, seed: int, cfg: dict,
                      pristine, emb_dim: int, device: str,
                      record_history: bool = False) -> tuple[dict, dict]:
    """单折：训 FER 支一次 → 两 context 变体 × 两种融合拟合位置 → 全部臂的 test oof 条目。

    返回 (臂名 → (test_idx, y_test, p_test), 附加记录)。
    """
    te = fold["test"]
    yte = y[te]
    fg, info = fer_branch_logits(X, fold, y, seed, cfg, pristine, emb_dim, device,
                                 record_history=record_history)
    zf, muf, sdf = standardize_branch(fg, fold["train"])
    fus_cfg = cfg["fusion"]

    out: dict[str, tuple] = {
        # FER 支单独 = M2（logits 排序与概率一致）
        "M2": (te, yte, _sigmoid(fg["test"])),
    }
    wrec: dict[str, dict] = {}
    for variant in ("lr", "mlp"):
        cg = ctx_branch_logits(data, fold, y, seed, variant,
                               cfg["context_branch"]["lr"] if variant == "lr"
                               else cfg["context_branch"]["mlp"])
        zc, muc, sdc = standardize_branch(cg, fold["train"])
        for fit_on in ("val", "train"):
            name = f"M4_{variant}" + ("" if fit_on == "val" else "_tr")
            p, w = fuse(zf, zc, y, fold, fit_on, fus_cfg["C"], fus_cfg["max_iter"], seed)
            out[name] = (te, yte, p)
            wrec[name] = {**w, "fit_on": fit_on, "ctx_branch": variant,
                          "logit_scale_fer": sdf, "logit_scale_ctx": sdc,
                          "logit_mean_fer": muf, "logit_mean_ctx": muc}
    return out, {"fer_train_info": info, "fusion_weights": wrec}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ---------------------------------------------------------------- 单 seed 观测

def run_seed(X, data: dict, folds: list[dict], y: np.ndarray, seed: int, cfg: dict,
             pristine, emb_dim: int, device: str, record_history: bool = False) -> dict:
    """单 split seed × 5 折：返回 臂 → oof 列表、逐折权重、FER 训练信息。

    同时算出三个哨兵臂（B1v2 / B2v2 / M1），走与主表完全相同的 ers.run_arm 路径 ——
    这样观测值与置换零分布用的是同一段代码，配对 Δ 的零分布才成立。
    """
    oof: dict[str, list] = {**{a: [] for a in FUSION_ARMS}, "M2": []}
    weights: dict[str, list] = {a: [] for a in FUSION_ARMS}
    infos: list[dict] = []
    for fi, fold in enumerate(folds):
        if len(fold["test"]) == 0:
            continue
        per_fold, extra = all_arms_one_fold(X, data, fold, y, seed, cfg, pristine,
                                            emb_dim, device,
                                            record_history=(record_history and fi == 0))
        for a, tup in per_fold.items():
            oof[a].append(tup)
        for a, w in extra["fusion_weights"].items():
            weights[a].append({"fold": fi, **w})
        infos.append({"fold": fi, **extra["fer_train_info"]})
    for arm in (B1V2_ARM, B2V2_ARM, M1_ARM):
        oof[arm["name"]] = ers.run_arm(data, arm, folds, y, seed)
    return {"oof": oof, "weights": weights, "train_info": infos}


# ---------------------------------------------------------------- 前置断言

def guard_arms(data: dict, folds: list[dict], y: np.ndarray, seed: int, cfg: dict,
               fresh: bool = True) -> list[dict]:
    """三个哨兵臂（B1v2 / B2v2 / M1）逐 seed 复现主表 181311；任一超容差即中止。

    `fresh=False` 时由调用方直接传观测阶段算好的 res['oof']（避免重复训练 MLP），
    只做比较。此时以 arm_of 的形式接受 (name, oof)。
    """
    tol = float(cfg["eval"]["guard_tol"])
    ref = pd.read_csv(PROJECT_ROOT / cfg["eval"]["guard_csv"])
    out = []
    for arm in (B1V2_ARM, B2V2_ARM, M1_ARM):
        ref_map = ref[ref["arm"] == arm["name"]].set_index("split_seed")["pooled_rank"].to_dict()
        got = ers.pooled_rank(ers.run_arm(data, arm, folds, y, seed))
        want = ref_map.get(int(seed))
        if want is None:
            raise AssertionError(f"前置断言: 主表缺 {arm['name']} seed={seed}")
        dev = abs(got - want)
        if dev > tol:
            raise AssertionError(
                f"前置断言失败: seed {seed} 的 {arm['name']} pooled_rank = {got:.10f}，"
                f"主表 = {want:.10f}，|Δ| = {dev:.2e} > {tol:.1e}。中止。")
        out.append({"arm": arm["name"], "seed": int(seed), "recomputed": float(got),
                    "reference": float(want), "abs_dev": float(dev)})
    return out


def guard_from_oof(oof: dict, seed: int, cfg: dict) -> list[dict]:
    """用观测阶段已算好的哨兵 oof 做同一断言（不重复训练）。"""
    tol = float(cfg["eval"]["guard_tol"])
    ref = pd.read_csv(PROJECT_ROOT / cfg["eval"]["guard_csv"])
    out = []
    for name in ("B1v2", "B2v2", "M1"):
        ref_map = ref[ref["arm"] == name].set_index("split_seed")["pooled_rank"].to_dict()
        got = ers.pooled_rank(oof[name])
        want = ref_map.get(int(seed))
        if want is None:
            raise AssertionError(f"前置断言: 主表缺 {name} seed={seed}")
        dev = abs(got - want)
        if dev > tol:
            raise AssertionError(
                f"前置断言失败: seed {seed} 的 {name} pooled_rank = {got:.10f}，"
                f"主表 = {want:.10f}，|Δ| = {dev:.2e} > {tol:.1e}。中止。")
        out.append({"arm": name, "seed": int(seed), "recomputed": float(got),
                    "reference": float(want), "abs_dev": float(dev)})
    return out


def guard_m2(obs_seed: dict, seed: int) -> dict:
    """观测值里本脚本的 M2 必须复现 M2 报告（同一 config、同一 seed、同一 folds）。"""
    if not M2_CSV.exists():
        return {"arm": "M2", "skipped": f"{M2_CSV.name} 不存在"}
    ref = pd.read_csv(M2_CSV)
    row = ref[(ref["arm"] == "M2") & (ref["split_seed"] == seed)]
    if row.empty:
        return {"arm": "M2", "skipped": f"无 seed={seed} 行"}
    got = obs_seed["stats"]["M2"]["pooled_rank"]
    want = float(row["pooled_rank"].iloc[0])
    return {"arm": "M2", "seed": int(seed), "recomputed": float(got),
            "reference": want, "abs_dev": float(abs(got - want)),
            "reference_csv": M2_CSV.relative_to(PROJECT_ROOT).as_posix()}


# ---------------------------------------------------------------- 置换

def perm_worker_one(X, data: dict, seeds: list[int], perm_seed: int, cfg: dict,
                    pristine, emb_dim: int, device: str) -> dict:
    """一次置换（单 split seed）：打乱标签 → 重建 folds → 全流程 refit（含 FER 支）。

    run_seed 内部同时算出 B1v2 / B2v2 / M1，因此配对 Δ 的零分布与观测值完全同源。
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(perm_seed)
    y = data["y"]
    for attempt in range(20):
        yp = rng.permutation(y)
        rec: dict = {"redraws": attempt}
        ok = False
        for s in seeds:
            built = ers.make_folds(data, yp, s)
            if built is None:
                continue
            _, folds = built
            res = run_seed(X, data, folds, yp, s, cfg, pristine, emb_dim, device)
            for a, lst in res["oof"].items():
                rec.setdefault(a, {})[s] = ers.pooled_rank(lst)
            ok = True
        if ok:
            return rec
    raise RuntimeError(f"perm {perm_seed}: 20 次重抽都缺类")


def p_right(null: np.ndarray, obs: float) -> float:
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


# ---------------------------------------------------------------- 主流程

def summary_of(oof: dict) -> dict:
    return {a: ers.full_stats(lst) for a, lst in oof.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="覆盖配置里的观测 seed（如 2027 或 2027-2036）")
    ap.add_argument("--n-perm", type=int, default=None, help="覆盖配置里的置换次数（0 = 只算观测）")
    ap.add_argument("--perm-seeds", default=None, help="覆盖配置里的置换 split seed")
    ap.add_argument("--smoke", action="store_true", help="1 seed × 1 折 × 1 置换，量级检查（不落盘）")
    args = ap.parse_args()

    setup_determinism()
    cfg = load_cfg()
    seeds = ers.parse_seeds(args.seeds or cfg["eval"]["default_seeds"])
    perm_seeds = ers.parse_seeds(args.perm_seeds or cfg["eval"]["perm_seeds"])
    n_perm = int(cfg["eval"]["n_perm"] if args.n_perm is None else args.n_perm)
    smoke = bool(args.smoke)
    if smoke:
        seeds, perm_seeds, n_perm = seeds[:1], perm_seeds[:1], 1

    device = "cuda" if _cuda() else "cpu"
    print(f"device={device}  seeds={seeds}  perm_seeds={perm_seeds}  n_perm={n_perm}"
          + ("  [smoke]" if smoke else ""))

    data, X, _ = load_all(cfg)
    y = data["y"]
    pristine, emb_dim = build_pristine(cfg, device)
    print(f"有效样本 {len(y)}（goal {int(y.sum())} / miss {int((1 - y).sum())}）  emb_dim={emb_dim}")

    # ---------------- 观测值 ----------------
    obs: dict[int, dict] = {}
    guards: list[dict] = []
    for s in seeds:
        built = ers.make_folds(data, y, s)
        if built is None:
            raise SystemExit(f"seed {s}: 真实标签下划分缺类")
        _, folds = built
        t0 = time.time()
        res = run_seed(X, data, folds, y, s, cfg, pristine, emb_dim, device,
                       record_history=False)
        st = summary_of(res["oof"])
        st = {a: st[a] for a in (*FUSION_ARMS, "M2", "B1v2", "B2v2", "M1")}
        obs[s] = {"stats": st, "weights": res["weights"], "train_info": res["train_info"]}
        guards.extend(guard_from_oof(res["oof"], s, cfg))   # 哨兵已在 run_seed 内算好
        line = "  ".join(f"{a} {st[a]['pooled_rank']:.4f}"
                         for a in ("M4_lr", "M4_mlp", "M2"))
        print(f"  seed {s}: {line}  | {time.time() - t0:.0f}s")
        if smoke:
            w = res["weights"]["M4_lr"][0]
            print(f"    [smoke] M4_lr 融合权重 w_fer={w['w_fer']:+.3f} "
                  f"w_ctx={w['w_ctx']:+.3f}  尺度 {w['logit_scale_fer']:.3f}/{w['logit_scale_ctx']:.3f}")

    if smoke:
        print("\n[smoke] 不落盘，结束。")
        return 0

    # ---------------- 置换（降级口径） ----------------
    perms: list[dict] = []
    if n_perm > 0:
        print(f"\n置换：{n_perm} 次 × split seed {perm_seeds}（全流程 refit）")
        t0 = time.time()
        for i in range(n_perm):
            rec = perm_worker_one(X, data, perm_seeds, int(cfg["eval"]["perm_seed"]) + i,
                                 cfg, pristine, emb_dim, device)
            perms.append(rec)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  perm {i + 1}/{n_perm}  ({time.time() - t0:.0f}s)")

    # ---------------- 判读 ----------------
    obs_mean = {a: float(np.mean([obs[s]["stats"][a]["pooled_rank"] for s in seeds]))
                for a in (*FUSION_ARMS, "M2", "B1v2", "B2v2", "M1")}
    obs_sd = {a: float(np.std([obs[s]["stats"][a]["pooled_rank"] for s in seeds]))
              for a in (*FUSION_ARMS, "M2", "B1v2", "B2v2", "M1")}

    def null_delta(a: str, b: str, s: int) -> np.ndarray:
        return np.array([p[a][s] - p[b][s] for p in perms
                         if a in p and b in p and s in p[a] and s in p[b]])

    # 配对比较：b 为 B1v2 / B2v2 时用本脚本在同一次运行里算出的哨兵 oof（与零分布同源，
    # 因此置换下的 B1v2/B2v2 是重训的，不是引用主表的固定数字）
    comparisons: dict[str, dict] = {}
    pairings = [("M4_lr", "B1v2"), ("M4_mlp", "B2v2"), ("M4_lr", "M4_mlp"),
                ("M4_lr", "M2"), ("M4_mlp", "M2"), ("M4_lr_tr", "M4_lr"), ("M4_mlp_tr", "M4_mlp")]
    for a, b in pairings:
        d10 = np.array([obs[s]["stats"][a]["pooled_rank"] - obs[s]["stats"][b]["pooled_rank"]
                        for s in seeds])
        entry: dict = {"a_mean_10seed": obs_mean[a], "b_mean_10seed": obs_mean[b],
                       "delta_mean_10seed": float(d10.mean()),
                       "delta_std_10seed": float(d10.std()),
                       "n_seed_positive": int((d10 > 0).sum()), "n_seeds": len(seeds),
                       "per_seed_delta": {int(s): round(float(obs[s]["stats"][a]["pooled_rank"]
                                                             - obs[s]["stats"][b]["pooled_rank"]), 4)
                                          for s in seeds}}
        if perms:
            s0 = perm_seeds[0]
            obs_d = float(obs[s0]["stats"][a]["pooled_rank"] - obs[s0]["stats"][b]["pooled_rank"])
            null = null_delta(a, b, s0)
            entry.update({
                "test_seed": s0, "obs_delta_at_test_seed": obs_d,
                "null_mean": float(null.mean()) if len(null) else None,
                "null_std": float(null.std()) if len(null) else None,
                "null_p95": float(np.percentile(null, 95)) if len(null) else None,
                "p_delta": p_right(null, obs_d) if len(null) else None,
            })
        # 判定：H8a = M4_lr − B1v2，H8b = M4_mlp − B2v2
        key = f"{a} - {b}"
        if key in ("M4_lr - B1v2", "M4_mlp - B2v2") and entry.get("p_delta") is not None:
            entry["is_primary"] = True
            entry["hypothesis"] = "H8a" if b == "B1v2" else "H8b"
            entry["passes"] = bool(entry["p_delta"] < ALPHA)
        comparisons[key] = entry

    # 可解释性读数
    expl: dict[str, dict] = {}
    for a in FUSION_ARMS:
        wf = np.array([w["w_fer"] for s in seeds for w in obs[s]["weights"][a] if np.isfinite(w["w_fer"])])
        wc = np.array([w["w_ctx"] for s in seeds for w in obs[s]["weights"][a] if np.isfinite(w["w_ctx"])])
        per_seed_wf = {s: float(np.mean([w["w_fer"] for w in obs[s]["weights"][a]])) for s in seeds}
        expl[a] = {
            "w_fer_mean": float(wf.mean()), "w_fer_std": float(wf.std()),
            "w_fer_min": float(wf.min()), "w_fer_max": float(wf.max()),
            "w_ctx_mean": float(wc.mean()), "w_ctx_std": float(wc.std()),
            "n_folds": int(len(wf)),
            "w_fer_positive_folds": int((wf > 0).sum()),
            "w_fer_positive_seeds": int(sum(1 for v in per_seed_wf.values() if v > 0)),
            "per_seed_w_fer": per_seed_wf,
            "w_ctx_positive_seeds": int(sum(1 for s in seeds
                                            if np.mean([w["w_ctx"] for w in obs[s]["weights"][a]]) > 0)),
            "caveat": ("w 的绝对大小在支 logit 重标定下不可识别，只有 (w_f:w_c:b) 等价类可读；"
                       "两支都无信号时 w 随 val 噪声漂移，符号不稳定不得读成'FER 支为负贡献'；"
                       "'w_f≈0' 只能写成'未检出 FER 支的可测增量'。"),
        }

    # ---------------- 落盘 ----------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in seeds:
        for a in (*FUSION_ARMS, "M2", "B1v2", "B2v2", "M1"):
            st = obs[s]["stats"][a]
            rows.append({"arm": a, "split_seed": s, **{k: v for k, v in st.items()
                                                       if not isinstance(v, dict)}})
    pd.DataFrame(rows).to_csv(TABLES_DIR / f"m4_fusion_{stamp}.csv", index=False, encoding="utf-8")

    wrows = []
    for s in seeds:
        for a in FUSION_ARMS:
            for w in obs[s]["weights"][a]:
                wrows.append({"arm": a, "split_seed": s, **w})
    pd.DataFrame(wrows).to_csv(TABLES_DIR / f"m4_fusion_weights_{stamp}.csv",
                               index=False, encoding="utf-8")

    m2_guard = guard_m2(obs[seeds[0]], seeds[0])
    report = {
        "created_at": stamp,
        "purpose": "M4-stacked 融合（微调 FER 支 + context 支 → 3 参数 logit 融合）；判读 H8a/H8b",
        "criteria": CRITERIA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config": cfg,
        "device": device,
        "eval_seeds": seeds, "perm_seeds": perm_seeds, "n_perm": n_perm,
        "degraded_permutation": bool(n_perm > 0 and len(perm_seeds) == 1),
        "n_valid_events": int(len(y)),
        "determinism": {"mlp_init_seeded": True, "cudnn_deterministic": True,
                        "tf32": False, "amp": False,
                        "note": "两支均先 manual_seed 再构造；融合层 3 参数只在 val 或 train 上拟合。"},
        "obs_summary": {a: {"pooled_rank_mean": obs_mean[a], "pooled_rank_std": obs_sd[a],
                            "per_seed_pooled_rank": {int(s): round(obs[s]["stats"][a]["pooled_rank"], 4)
                                                     for s in seeds},
                            "full_stats_by_seed": {int(s): obs[s]["stats"][a] for s in seeds}}
                        for a in (*FUSION_ARMS, "M2")},
        "comparisons": comparisons,
        "interpretability": expl,
        "guards": {"arms_181311": guards, "M2_report": m2_guard},
        "permutation": {"null_pooled_rank": {a: {"mean": float(np.mean([p[a][perm_seeds[0]]
                                                                       for p in perms if a in p])),
                                                 "p95": float(np.percentile([p[a][perm_seeds[0]]
                                                                             for p in perms if a in p], 95))}
                                             for a in (*FUSION_ARMS, "M2", "B1v2", "B2v2", "M1")}
                        if perms else None,
                        "n": len(perms)},
        "limitations": [
            "检测下限：零分布 p95 ≈ 0.55–0.57，385 样本下低于约 0.55 的效应测不出来。",
            "置换零分布只在 seed 2027 那套 folds 上成立（降级口径），不可与主表其他臂比强度。",
            "融合层在内层 val（每折 40–74 例）上拟合；M4_*_tr 敏感性对照用于检查拟合位置的影响。",
            "本臂只检验两支 logit 的线性加权，不检验特征层交互项（对 plan §7.4 的偏离，见预注册）。",
        ],
    }
    with open(REPORTS_DIR / f"m4_fusion_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # ---------------- 控制台输出 ----------------
    print("\n" + "=" * 78)
    print("观测值（10-seed 均值 ± sd，pooled_rank）")
    print("=" * 78)
    for a in ("M4_lr", "M4_mlp", "M4_lr_tr", "M4_mlp_tr", "M2"):
        print(f"  {a:10s} {obs_mean[a]:.4f} ± {obs_sd[a]:.4f}")
    print("\n配对比较（检验 seed 上的 Δ 与置换 p_Δ）")
    print("=" * 78)
    for k, v in comparisons.items():
        pd_ = v.get("p_delta")
        print(f"  {k:20s} Δ_10seed={v['delta_mean_10seed']:+.4f}  "
              f"{v['n_seed_positive']}/{v['n_seeds']} 为正"
              + (f"  obs_Δ={v['obs_delta_at_test_seed']:+.4f}  p_Δ={pd_:.4f}" if pd_ is not None else ""))
    print("\n可解释性读数（w_fer / w_ctx）")
    print("=" * 78)
    for a in FUSION_ARMS:
        e = expl[a]
        print(f"  {a:10s} w_fer={e['w_fer_mean']:+.3f}±{e['w_fer_std']:.3f} "
              f"(为正 {e['w_fer_positive_seeds']}/{len(seeds)} seed)  "
              f"w_ctx={e['w_ctx_mean']:+.3f}±{e['w_ctx_std']:.3f} "
              f"(为正 {e['w_ctx_positive_seeds']}/{len(seeds)} seed)")
    print(f"\nM2 复现检查: {m2_guard}")
    print(f"落盘: results/tables/m4_fusion_{stamp}.csv")
    print(f"      results/reports/m4_fusion_{stamp}.json")
    return 0


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
