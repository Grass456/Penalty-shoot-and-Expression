#!/usr/bin/env python
"""M2 部分微调评估：冻结臂 M1 vs 部分微调臂 M2（plan.md §9 M2 / §12 A3，判读线 H4）。

预注册判读标准（跑前写定）：results/reports/m2_finetune_criteria_preregistered.txt

臂：
    - M1  = 冻结 FER embedding + LR（C=0.01，与主表同参数），逐折现场重训 ——
            它同时是前置一致性的哨兵（必须复现主表 181311 至 |Δ|<1e-6）。
    - M2  = enet_b0_8_best_vgaf，解冻 blocks[-1] + 线性头，判别式 lr，图像直接前向。
    - m2_head = 只训头（backbone 全冻结），退化检查，信息性不判定。

协议：
    - folds 复用 eval_repeated_splits.make_folds（5 折 SGKF + 内层 val，硬约束 4）；
    - 观测值：10 个 split seed（2027-2036）× 5 折，Δ = M2 − M1 逐 seed 配对；
    - 置换（降级口径，用户 2026-11-05 选定）：100 次 × 单 split seed，打乱全部标签后
      重建 folds 并重训 M2 与 M1（全流程 refit，backbone 不冻结）；
      p_Δ = (1 + #{null_Δ ≥ obs_Δ}) / (1 + n_perm)。因单 seed 无符号检验，
      H4 判定 = p_Δ < 0.05 单独成立。
    - 图像管线一致性：载入后立刻用 backbone batch=1 前向抽查 8 个事件，与
      features/fer/<backbone>/embeddings.npz 比对（M2 读图，其余臂读缓存，必须同源）。

用法（项目根目录，需 PYTHONIOENCODING=utf-8）:
    python scripts/eval_m2_finetune.py --smoke            # 1 seed × 1 折 × 1 置换，量级检查
    python scripts/eval_m2_finetune.py --n-perm 0         # 只算观测值（10 seed × 5 折）
    python scripts/eval_m2_finetune.py                    # 全量：观测 + 100 次置换
    python scripts/eval_m2_finetune.py --seeds 2027 --n-perm 100 --perm-seeds 2027

输出: results/tables/m2_finetune_<stamp>.csv、results/reports/m2_finetune_<stamp>.json
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# cuBLAS 确定性要求（torch.use_deterministic_algorithms 对 CUDA>=10.2 的 GEMM 强制要求）。
# 必须在任何 torch / CUDA 初始化之前设置，否则该设置不生效并刷 warning。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import eval_repeated_splits as ers  # noqa: E402
from src.data.face_images import load_face_tensor  # noqa: E402
from src.training.finetune_loop import predict_images, train_one_fold_images  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "m2_finetune.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
CRITERIA_PATH = REPORTS_DIR / "m2_finetune_criteria_preregistered.txt"

# M1 的臂定义必须与主表 ARMS 里的 M1 逐字段一致（前置断言才有意义）
M1_ARM = {"name": "M1", "model": "lr", "feats": ["emb:enet_b0_8_best_vgaf"], "C": 0.01, "subset": False}
ALPHA = 0.05


def setup_determinism() -> None:
    """cuDNN 确定性 + 线程数固定（微调逐折在同一 GPU 上串行，复现优先于吞吐）。"""
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def load_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_seeds(s: str) -> list[int]:
    return ers.parse_seeds(s)


# ---------------------------------------------------------------- 数据 / 模型

def load_all(cfg: dict) -> tuple[dict, "np.ndarray", list[str]]:
    """载入事件数据 + 人脸张量，并做两次对齐断言。"""
    import torch

    data = ers.load_data()
    X, ids = load_face_tensor(data["valid"], PROJECT_ROOT, img_size=int(cfg["fer"]["img_size"]))
    if ids != data["ids"]:
        raise AssertionError("人脸张量行顺序与 data['ids'] 不一致 —— folds 索引会错位")
    if not torch.isfinite(X).all():
        raise AssertionError("人脸张量含 NaN/Inf")
    return data, X, ids


def build_pristine(cfg: dict, device: str):
    """加载 FER backbone（原始权重），并做图像管线 ↔ 特征缓存的一致性抽查。"""
    from cache_fer_features import build_fer

    name = cfg["fer"]["backbone"]
    backbone, _, emb_dim, _, prep = build_fer(name, device)
    img_size = int(prep[0])
    if img_size != int(cfg["fer"]["img_size"]):
        raise AssertionError(f"配置 img_size={cfg['fer']['img_size']} 与 backbone 的 {img_size} 不符")
    return backbone, emb_dim, img_size


def train_predict_m2(X, fold: dict, y: np.ndarray, seed: int, cfg: dict, pristine,
                     emb_dim: int, device: str, unfreeze_depth: int | None = None,
                     record_history: bool = False) -> tuple[np.ndarray, dict]:
    """单折部分微调：从 pristine 拷贝 backbone → seed → 建模 → 训练 → test 概率。

    顺序不可调换：先 manual_seed 再构造模型（head 的 init 取自全局 RNG），
    见 memory/mlp-init-determinism-fix。
    """
    import torch
    from src.models.fer_finetune import FinetuneFER

    tr, va, te = fold["train"], fold["val"], fold["test"]
    mc, tc = cfg["model"], cfg["training"]
    depth = int(mc["unfreeze_depth"]) if unfreeze_depth is None else int(unfreeze_depth)

    torch.manual_seed(int(seed))
    model = FinetuneFER(copy.deepcopy(pristine), emb_dim=emb_dim, unfreeze_depth=depth,
                        bn_train=bool(mc["bn_train"]))
    _, info = train_one_fold_images(
        model, X[tr], y[tr], X[va], y[va], seed=seed,
        max_epochs=int(tc["max_epochs"]), patience=int(tc["patience"]),
        lr_head=float(tc["lr_head"]), lr_backbone=float(tc["lr_backbone"]),
        weight_decay=float(tc["weight_decay"]), batch_size=int(tc["batch_size"]),
        device=device, amp=bool(tc["amp"]))
    p = predict_images(model, X[te], device=device)
    keep = {"best_epoch": info["best_epoch"], "best_val_auroc": info["best_val_auroc"],
            "n_epochs_run": info["n_epochs_run"]}
    if record_history:
        keep["history"] = info["history"]
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return p, keep


def run_m2(X, folds: list[dict], y: np.ndarray, seed: int, cfg: dict,
           pristine, emb_dim: int, device: str, unfreeze_depth: int | None = None,
           record_history: bool = False) -> tuple[list[tuple], list[dict]]:
    """单 seed × 5 折：返回 (oof, 每折训练信息)。"""
    oof, infos = [], []
    for fi, fold in enumerate(folds):
        te = fold["test"]
        if len(te) == 0:
            continue
        p, info = train_predict_m2(X, fold, y, seed, cfg, pristine, emb_dim, device,
                                   unfreeze_depth=unfreeze_depth,
                                   record_history=record_history)
        info["fold"] = fi
        infos.append(info)
        oof.append((te, y[te], p))
    return oof, infos


# ---------------------------------------------------------------- 前置断言

def guard_m1(data: dict, folds: list[dict], y: np.ndarray, seed: int, cfg: dict) -> dict:
    """M1 冻结臂必须复现主表 181311（folds / 标签 / 口径对齐的守门）。

    逐 seed 断言：|pooled_rank(本脚本) − pooled_rank(主表)| < guard_tol。
    LR 臂逐位可复现，1e-6 是留了浮点余量的严格阈值。失败即中止 —— 否则
    M2 与 M1 的 Δ 建立在两套不同的 folds 上，对比无意义。
    """
    tol = float(cfg["eval"]["guard_tol"])
    ref_csv = PROJECT_ROOT / cfg["eval"]["guard_csv"]
    ref = pd.read_csv(ref_csv)
    ref_m1 = ref[ref["arm"] == "M1"].set_index("split_seed")["pooled_rank"].to_dict()

    oof = ers.run_arm(data, M1_ARM, folds, y, seed)
    got = ers.pooled_rank(oof)
    want = ref_m1.get(int(seed))
    if want is None:
        raise AssertionError(f"前置断言: 主表 {ref_csv.name} 缺 M1 seed={seed} 的行")
    dev = abs(got - want)
    if dev > tol:
        raise AssertionError(
            f"前置断言失败: seed {seed} 的 M1 pooled_rank = {got:.10f}，主表 = {want:.10f}，"
            f"|Δ| = {dev:.2e} > {tol:.1e}。folds/标签/口径与主表不一致，中止。")
    return {"seed": int(seed), "pooled_rank_recomputed": float(got),
            "pooled_rank_reference": float(want), "abs_dev": float(dev),
            "reference_csv": ref_csv.relative_to(PROJECT_ROOT).as_posix(), "tol": tol}


# ---------------------------------------------------------------- 主评估

def run_arm_m2(X, folds: list[dict], y: np.ndarray, seed: int, cfg: dict,
               pristine, emb_dim: int, device: str, record_history: bool = False) -> dict:
    """单 seed 的 M2 统计（含逐折训练信息）。"""
    oof, infos = run_m2(X, folds, y, seed, cfg, pristine, emb_dim, device,
                        record_history=record_history)
    st = ers.full_stats(oof)
    st["folds"] = infos
    return st


def run_head_only(X, folds: list[dict], y: np.ndarray, seed: int, cfg: dict,
                  pristine, emb_dim: int, device: str) -> dict:
    """退化检查：backbone 全冻结只训头（unfreeze_depth=0）。信息性，不判定。"""
    oof, infos = run_m2(X, folds, y, seed, cfg, pristine, emb_dim, device,
                        unfreeze_depth=0)
    st = ers.full_stats(oof)
    st["folds"] = infos
    return st


def replicate_obs(data: dict, X, seeds: list[int], cfg: dict, pristine, emb_dim: int,
                  device: str, with_head_only: bool, record_history_seed: int | None) -> dict:
    """观测值：每个 split seed 跑 M1（哨兵）+ M2（+ 可选的只训头对照）。"""
    out: dict = {"M1": {}, "M2": {}, "M2_head": {}, "guards": []}
    for s in seeds:
        built = ers.make_folds(data, data["y"], s)
        if built is None:
            raise SystemExit(f"seed {s}: 真实标签下划分缺类")
        _, folds = built
        out["guards"].append(guard_m1(data, folds, data["y"], s, cfg))
        t = time.time()
        st = run_arm_m2(X, folds, data["y"], s, cfg, pristine, emb_dim, device,
                        record_history=(record_history_seed == s))
        out["M2"][s] = st
        out["M1"][s] = ers.full_stats(ers.run_arm(data, M1_ARM, folds, data["y"], s))
        if with_head_only:
            out["M2_head"][s] = run_head_only(X, folds, data["y"], s, cfg,
                                              pristine, emb_dim, device)
        print(f"  seed {s}: M2 {st['pooled_rank']:.4f} | M1 {out['M1'][s]['pooled_rank']:.4f} "
              f"| Δ {st['pooled_rank'] - out['M1'][s]['pooled_rank']:+.4f} "
              f"| val_auroc {np.mean([f['best_val_auroc'] for f in st['folds']]):.3f} "
              f"| {time.time() - t:.0f}s")
    return out


# ---------------------------------------------------------------- 置换

def perm_worker_one(data: dict, X, seeds: list[int], perm_seed: int, cfg: dict,
                    pristine, emb_dim: int, device: str) -> dict:
    """一次置换（单 split seed）：打乱标签 → 重建 folds → 重训 M2 与 M1。

    缺类划分重抽（最多 20 次），与 eval_repeated_splits.perm_worker 同约定。
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(perm_seed)
    y = data["y"]
    for attempt in range(20):
        yp = rng.permutation(y)
        built = ers.make_folds(data, yp, seeds[0])
        if built is None:
            continue
        _, folds = built
        rec = {"redraws": attempt}
        for arm in ("M1", "M2"):
            if arm == "M1":
                st = ers.full_stats(ers.run_arm(data, M1_ARM, folds, yp, seeds[0]))
            else:
                st = run_arm_m2(X, folds, yp, seeds[0], cfg, pristine, emb_dim, device)
            rec[arm] = float(st["pooled_rank"])
        rec["delta"] = rec["M2"] - rec["M1"]
        return rec
    raise RuntimeError(f"perm {perm_seed}: 20 次重抽都缺类")


def p_right(null: np.ndarray, obs: float) -> float:
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


# ---------------------------------------------------------------- 判读

def verdicts(obs: dict, perms: list[dict], seeds: list[int], perm_seeds: list[int],
             n_perm: int, degraded: bool) -> dict:
    """按预注册标准判读 H4：per-seed 描述 + 单 seed 配对检验。

    检验统计量必须与零分布同 seed：置换只在 perm_seed 那一套 folds 上跑，因此
    **obs_Δ 取该 seed 的配对 Δ**（不是 10 seed 均值 —— 均值方差只有单 seed 的 ~1/√10，
    拿它去比单 seed 零分布会凭空放大显著性）。10 seed 均值只作描述性稳定性指标。
    """
    out: dict = {"arm_summary": {}, "comparison": {}}

    for arm in ("M1", "M2", "M2_head"):
        d = obs[arm]
        if not d:
            continue
        pr = np.array([d[s]["pooled_rank"] for s in seeds])
        entry = {"pooled_rank_mean": float(pr.mean()), "pooled_rank_std": float(pr.std()),
                 "pooled_rank_min": float(pr.min()), "pooled_rank_max": float(pr.max()),
                 "per_seed_pooled_rank": {int(s): round(float(d[s]["pooled_rank"]), 4)
                                          for s in seeds}}
        for k in ("fold_mean", "pooled_auprc", "pooled_brier", "pooled_logloss",
                  "pooled_balanced_acc"):
            entry[f"{k}_mean"] = float(np.mean([d[s][k] for s in seeds]))
        if all("folds" in d[s] for s in seeds):  # M1 是 LR，没有逐折训练信息
            entry["best_val_auroc_mean"] = float(np.mean(
                [f["best_val_auroc"] for s in seeds for f in d[s]["folds"]]))
            entry["best_epoch_mean"] = float(np.mean(
                [f["best_epoch"] for s in seeds for f in d[s]["folds"]]))
            entry["n_epochs_run_mean"] = float(np.mean(
                [f["n_epochs_run"] for s in seeds for f in d[s]["folds"]]))
        out["arm_summary"][arm] = entry

    da = np.array([obs["M2"][s]["pooled_rank"] for s in seeds])
    db = np.array([obs["M1"][s]["pooled_rank"] for s in seeds])
    d = da - db
    cmp_entry = {
        "delta_mean": float(d.mean()), "delta_std": float(d.std()),
        "delta_min": float(d.min()), "delta_max": float(d.max()),
        "per_seed_delta": {int(s): round(float(x), 4) for s, x in zip(seeds, d)},
        "n_seeds": int(len(seeds)), "n_seeds_positive": int((d > 0).sum()),
        "direction_consistent": bool((d > 0).all() or (d < 0).all()),
    }

    if perms:
        null = np.array([p["delta"] for p in perms])
        null_m2 = np.array([p["M2"] for p in perms])
        null_m1 = np.array([p["M1"] for p in perms])
        if len(perm_seeds) != 1:
            raise NotImplementedError(
                "置换覆盖多个 split seed 时零分布口径未定义（当前预注册为单 seed）；"
                "请用 --perm-seeds 指定单个 seed")
        test_seed = perm_seeds[0]
        obs_delta = float(obs["M2"][test_seed]["pooled_rank"]
                          - obs["M1"][test_seed]["pooled_rank"])
        cmp_entry.update({
            "null_delta_mean": float(null.mean()), "null_delta_std": float(null.std()),
            "null_delta_p95": float(np.percentile(null, 95)),
            "p_delta": p_right(null, obs_delta),
            "obs_delta_for_test": obs_delta,
            "test_seed": int(test_seed),
            "delta_mean_all_seeds_descriptive": float(d.mean()),
            "null_M2_mean": float(null_m2.mean()), "null_M2_p95": float(np.percentile(null_m2, 95)),
            "p_M2": p_right(null_m2, obs["M2"][test_seed]["pooled_rank"]),
            "null_M1_mean": float(null_m1.mean()), "null_M1_p95": float(np.percentile(null_m1, 95)),
            "p_M1": p_right(null_m1, obs["M1"][test_seed]["pooled_rank"]),
            "n_perm": int(len(perms)),
            "perm_redraws_total": int(sum(p["redraws"] for p in perms)),
            "degraded_note": ("降级口径：置换仅单 split seed，p 分辨率 0.01，无符号检验；"
                              "H4 判定 = p_delta < 0.05 单独成立，检验统计量取该 seed 的配对 Δ。"
                              "与主表其他臂（200 perm × 10 seed）不可直接比强度。"
                              if degraded else
                              "置换覆盖全部观测 seed；判读按预注册标准（p_delta 与符号检验并用）"),
        })
        cmp_entry["holds"] = bool(cmp_entry["p_delta"] < ALPHA)
        cmp_entry["h4_supported"] = cmp_entry["holds"]
    else:
        cmp_entry["holds"] = None
        cmp_entry["h4_supported"] = None
        cmp_entry["note"] = "未跑置换（--n-perm 0）：只给描述统计，不能判 H4"

    out["comparison"] = cmp_entry
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=None, help="观测 split seed，如 2027-2036（默认读配置）")
    ap.add_argument("--perm-seeds", default=None, help="置换用的 split seed（默认读配置，单 seed）")
    ap.add_argument("--n-perm", type=int, default=None, help="置换次数（默认读配置）")
    ap.add_argument("--perm-seed", type=int, default=None, help="置换随机种子（默认读配置）")
    ap.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：1 split seed × 第 1 折 × 1 次置换，只检查管线不落盘")
    ap.add_argument("--no-head-only", action="store_true", help="跳过只训头的退化检查")
    ap.add_argument("--save-history-seed", type=int, default=None,
                    help="把该 seed 的逐 epoch 训练历史写进报告（默认只存汇总）")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    cfg = load_cfg()
    ec = cfg["eval"]
    seeds = parse_seeds(args.seeds or ec["default_seeds"])
    perm_seeds = parse_seeds(args.perm_seeds or ec["perm_seeds"])
    n_perm = int(ec["n_perm"] if args.n_perm is None else args.n_perm)
    perm_seed = int(ec["perm_seed"] if args.perm_seed is None else args.perm_seed)

    import torch
    setup_determinism()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # 降级判定：置换覆盖的 split seed 少于观测用的（如 10 seed 观测 + 单 seed 置换）
    degraded = sorted(perm_seeds) != sorted(seeds)

    print(f"设备 {device} | 观测 seeds {seeds} | 置换 {n_perm} 次 @ seed {perm_seeds} | "
          f"backbone {cfg['fer']['backbone']} | 解冻 blocks[-{cfg['model']['unfreeze_depth']}]")
    data, X, ids = load_all(cfg)
    print(f"有效样本 {len(ids)}，人脸张量 {tuple(X.shape)}")

    pristine, emb_dim, img_size = build_pristine(cfg, device)
    from src.data.face_images import assert_embedding_parity
    dev = assert_embedding_parity(X, ids, PROJECT_ROOT / cfg["fer"]["cache_npz"], pristine,
                                  device=device, n_check=8)
    print(f"图像管线一致性: 与缓存 embedding 的 max|Δ| = {dev:.2e}（batch=1 抽查 8 例）")

    # ---------------- 冒烟：1 seed × 1 折，只报量级，不落盘
    if args.smoke:
        built = ers.make_folds(data, data["y"], seeds[0])
        if built is None:
            raise SystemExit("冒烟: 划分缺类")
        _, folds = built
        g = guard_m1(data, folds, data["y"], seeds[0], cfg)
        print(f"前置断言 seed {seeds[0]}: M1 {g['pooled_rank_recomputed']:.10f} vs "
              f"主表 {g['pooled_rank_reference']:.10f} (|Δ| = {g['abs_dev']:.2e})")
        one = [folds[0]]
        t = time.time()
        st_obs = run_arm_m2(X, one, data["y"], seeds[0], cfg, pristine, emb_dim, device,
                            record_history=True)
        print(f"M2 1 折: pooled_rank {st_obs['pooled_rank']:.4f} | "
              f"val_auroc {st_obs['folds'][0]['best_val_auroc']:.3f} | "
              f"epochs {st_obs['folds'][0]['n_epochs_run']} | {time.time() - t:.0f}s")
        print("  历史:", [f"{h['epoch']}:{h['val_auroc']:.3f}"
                          for h in st_obs["folds"][0]["history"]][:12], "...")
        if n_perm > 0:
            t = time.time()
            rec = perm_worker_one(data, X, perm_seeds, perm_seed, cfg, pristine, emb_dim, device)
            print(f"置换 1 次: M2 {rec['M2']:.4f} M1 {rec['M1']:.4f} Δ {rec['delta']:+.4f} "
                  f"({time.time() - t:.0f}s)")
        print("冒烟完成（未落盘）")
        return 0

    # ---------------- 观测值
    t0 = time.time()
    obs = replicate_obs(data, X, seeds, cfg, pristine, emb_dim, device,
                        with_head_only=not args.no_head_only,
                        record_history_seed=args.save_history_seed)
    print(f"观测值完成 {time.time() - t0:.0f}s")

    # ---------------- 置换
    perms: list[dict] = []
    if n_perm > 0:
        t1 = time.time()
        for i in range(n_perm):
            perms.append(perm_worker_one(data, X, perm_seeds, perm_seed + i, cfg,
                                         pristine, emb_dim, device))
            if (i + 1) % 10 == 0 or i + 1 == n_perm:
                el = time.time() - t1
                print(f"  置换 {i + 1}/{n_perm}  {el:.0f}s  预计剩余 "
                      f"{el / (i + 1) * (n_perm - i - 1) / 60:.1f}min")
        print(f"置换 {n_perm} 次完成 {time.time() - t1:.0f}s，"
              f"重抽总数 {sum(p['redraws'] for p in perms)}")

    v = verdicts(obs, perms, seeds, perm_seeds, n_perm, degraded)

    print(f"\n{'arm':>8} | {'pooled_rank':>14} | {'val_auroc':>9} | {'best_epoch':>10}")
    for n, e in v["arm_summary"].items():
        va = f"{e['best_val_auroc_mean']:.3f}" if "best_val_auroc_mean" in e else "—"
        be = f"{e['best_epoch_mean']:.1f}" if "best_epoch_mean" in e else "—"
        print(f"{n:>8} | {e['pooled_rank_mean']:.4f}±{e['pooled_rank_std']:.3f} | "
              f"{va:>9} | {be:>10}")
    c = v["comparison"]
    print(f"\nM2 − M1: Δ(10 seed 描述) {c['delta_mean']:+.4f}±{c['delta_std']:.4f}  "
          f"{c['n_seeds_positive']}/{c['n_seeds']} seed>0")
    if perms:
        print(f"  检验用 seed {c['test_seed']} 的配对 Δ = {c['obs_delta_for_test']:+.4f}  "
              f"零分布 {c['null_delta_mean']:+.4f}±{c['null_delta_std']:.4f}  "
              f"p_Δ = {c['p_delta']:.3f}  → H4 {'成立' if c['h4_supported'] else '不支持'}")
        print(f"  M2 单独: p = {c['p_M2']:.3f} (null p95 {c['null_M2_p95']:.4f}) | "
              f"M1 单独: p = {c['p_M1']:.3f} (null p95 {c['null_M1_p95']:.4f})")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    for arm in ("M1", "M2", "M2_head"):
        for s in seeds:
            if s not in obs.get(arm, {}):
                continue
            st = obs[arm][s]
            row = {"arm": arm, "split_seed": s,
                   **{k: v_ for k, v_ in st.items() if k != "folds"}}
            if "folds" in st:  # M1 是 LR，没有逐折训练信息
                row["best_val_auroc_mean"] = float(np.mean([f["best_val_auroc"]
                                                            for f in st["folds"]]))
                row["best_epoch_mean"] = float(np.mean([f["best_epoch"]
                                                        for f in st["folds"]]))
            rows.append(row)
    table = TABLES_DIR / f"m2_finetune_{stamp}.csv"
    pd.DataFrame(rows).to_csv(table, index=False)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "M2 部分微调 vs M1 冻结（plan §9 / §12 A3，假设 H4）",
        "criteria": CRITERIA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "device": device, "eval_seeds": seeds, "perm_seeds": perm_seeds,
        "n_perm": n_perm, "perm_seed": perm_seed, "degraded_permutation": degraded,
        "n_valid_events": len(ids), "img_size": img_size, "emb_dim": emb_dim,
        "image_pipeline_parity_max_abs_dev": dev,
        "determinism": {"mlp_init_seeded": True, "cudnn_deterministic": True,
                        "tf32": False, "amp": bool(cfg["training"]["amp"]),
                        "note": "构造 FinetuneFER 前显式 manual_seed（见 mlp-init-determinism-fix）；"
                                "cuDNN deterministic + allow_tf32=False + AMP 关闭。"},
        "arm_specs": {"M1": M1_ARM,
                      "M2": {"backbone": cfg["fer"]["backbone"],
                             "unfreeze_depth": cfg["model"]["unfreeze_depth"],
                             "bn_train": cfg["model"]["bn_train"], **cfg["training"]},
                      "M2_head": {"note": "退化检查：unfreeze_depth=0，信息性不判定"}},
        "obs_table": table.relative_to(PROJECT_ROOT).as_posix(),
        "observation": {arm: {str(s): {k: v_ for k, v_ in obs[arm][s].items() if k != "folds"}
                              for s in obs.get(arm, {})} for arm in ("M1", "M2", "M2_head")},
        "train_info_per_seed": {arm: {str(s): obs[arm][s]["folds"] for s in obs.get(arm, {})}
                                for arm in ("M2", "M2_head")},
        "guards": obs.get("guards", []),
        "permutation": perms,
        **v,
    }
    path = REPORTS_DIR / f"m2_finetune_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n表:   {table.relative_to(PROJECT_ROOT).as_posix()}\n"
          f"报告: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
