#!/usr/bin/env python
"""诊断（描述性，不判定）：两支分支 logit 的尺度与重叠，用于判断 logit 融合是否可读。

背景：M4 拟从 plan §7.4 的"投影+拼接"改为"逐支输出 1 维 logit 再加权汇总"。
要读 `w_fer ≈ 0` 这句话，前提是两支 logit 在同一尺度上可比（否则 w 的绝对大小
在重标定下不可识别）。本脚本回答三个问题：

  1. 两支 logit 在 train / 内层 val / test 上的均值与标准差各是多少？
  2. 两支的尺度差多少倍（决定是否必须先标准化）？
  3. 在 train 统计量上标准化后，于内层 val 上拟合 3 参数融合 (w_f, w_c, b)，
     w_f / w_c 的逐折取值与符号是否稳定？融合后 test pooled_rank 是多少？

这不是预注册实验，不做判定、不进论文数字；只用于决定 M4 的结构。

用法（项目根目录，PYTHONIOENCODING=utf-8）：
    python scripts/diag_branch_logits.py                # 默认 seed 2027-2036
    python scripts/diag_branch_logits.py --seeds 2027   # 单 seed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_repeated_splits as ers  # noqa: E402
from run_baselines import build_context_matrix_v2  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FER_KEY = "emb:enet_b0_8_best_vgaf"
C_FER = 0.01   # 与主表 M1 一致
C_CTX = 1.0    # 与主表 B1v2 一致


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def branch_logits(data: dict, fold: dict, y: np.ndarray, seed: int) -> dict:
    """返回两支在同一折上的 (train/val/test) logits，标准化只用 train 统计量。"""
    tr, va, te = fold["train"], fold["val"], fold["test"]
    fc = data["cfg"]["features"]
    ctx = build_context_matrix_v2(data["valid"], p_bar=float(y[tr].mean()),
                                  alpha=fc["history_alpha"], clip=fc["score_diff_clip"])
    Xc = ctx.to_numpy(dtype=np.float64)
    Xf = data["blocks"][FER_KEY]

    out: dict[str, dict] = {}
    for tag, X, C in (("fer", Xf, C_FER), ("ctx", Xc, C_CTX)):
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        lr = LogisticRegression(max_iter=5000, C=C, random_state=seed)
        lr.fit(Z[tr], y[tr])
        out[tag] = {
            "train": lr.decision_function(Z[tr]),
            "val": lr.decision_function(Z[va]),
            "test": lr.decision_function(Z[te]),
            "prob_test": lr.predict_proba(Z[te])[:, 1],
        }
        # 标准化所需的 train 统计量（融合层会用到）
        out[tag]["_mu"], out[tag]["_sd"] = float(out[tag]["train"].mean()), float(out[tag]["train"].std())
    out["_te_idx"], out["_y_te"] = te, y[te]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2027-2036")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    data = ers.load_data()
    y = data["y"]

    rows, w_rows, fused_oof = [], [], []
    for seed in seeds:
        made = ers.make_folds(data, y, seed)
        if made is None:
            print(f"[skip] seed {seed}: 折构建失败")
            continue
        _, folds = made
        for fi, fold in enumerate(folds):
            if len(fold["test"]) == 0:
                continue
            b = branch_logits(data, fold, y, seed)
            for tag in ("fer", "ctx"):
                r = {"seed": seed, "fold": fi, "branch": tag}
                for split in ("train", "val", "test"):
                    v = b[tag][split]
                    r[f"{split}_mean"] = float(v.mean())
                    r[f"{split}_std"] = float(v.std())
                    r[f"{split}_min"] = float(v.min())
                    r[f"{split}_max"] = float(v.max())
                rows.append(r)

            # 尺度比（test 上的 std 之比与极差之比）
            sf, sc = b["fer"]["test"].std(), b["ctx"]["test"].std()
            rf = float(np.ptp(b["fer"]["test"]))
            rc = float(np.ptp(b["ctx"]["test"]))

            # 3 参数融合：train 统计量标准化 -> 在 val 上拟合 -> test 评估
            zf = {s: (b["fer"][s] - b["fer"]["_mu"]) / (b["fer"]["_sd"] or 1.0)
                  for s in ("train", "val", "test")}
            zc = {s: (b["ctx"][s] - b["ctx"]["_mu"]) / (b["ctx"]["_sd"] or 1.0)
                  for s in ("train", "val", "test")}
            Zva = np.column_stack([zf["val"], zc["val"]])
            if len(np.unique(y[fold["val"]])) == 2:
                fus = LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
                fus.fit(Zva, y[fold["val"]])
                wf, wc = float(fus.coef_[0][0]), float(fus.coef_[0][1])
                Zte = np.column_stack([zf["test"], zc["test"]])
                pte = fus.predict_proba(Zte)[:, 1]
                fused_oof.append((fold["test"], y[fold["test"]], pte))
            else:
                wf, wc = float("nan"), float("nan")
            w_rows.append({"seed": seed, "fold": fi, "w_fer": wf, "w_ctx": wc,
                           "scale_ratio_std": float(sf / sc) if sc else float("nan"),
                           "scale_ratio_range": float(rf / rc) if rc else float("nan"),
                           "fer_test_std": float(sf), "ctx_test_std": float(sc),
                           "n_val": int(len(fold["val"])), "n_test": int(len(fold["test"]))})
        # 单 seed 的 M1 / B1v2 参照（pooled_rank）
        print(f"seed {seed} 完成")

    df = pd.DataFrame(rows)
    wdf = pd.DataFrame(w_rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"diag_branch_logits_{stamp}.csv", index=False, encoding="utf-8")
    wdf.to_csv(TABLES_DIR / f"diag_branch_fusionw_{stamp}.csv", index=False, encoding="utf-8")

    # ---------------- 汇总输出 ----------------
    print("\n" + "=" * 76)
    print("1. 两支 logit 的分布（全 seed × 折 汇总）")
    print("=" * 76)
    for tag in ("fer", "ctx"):
        sub = df[df["branch"] == tag]
        for split in ("train", "val", "test"):
            m, s = sub[f"{split}_mean"], sub[f"{split}_std"]
            print(f"  {tag:3s} {split:5s}: mean={m.mean():+.3f}±{m.std():.3f}  "
                  f"std={s.mean():.3f}±{s.std():.3f}  (范围 {s.min():.3f}–{s.max():.3f})")

    print("\n" + "=" * 76)
    print("2. 尺度比（FER 支 / Context 支）")
    print("=" * 76)
    for col, lab in (("scale_ratio_std", "test std 之比"), ("scale_ratio_range", "test 极差之比")):
        v = wdf[col].dropna()
        print(f"  {lab}: 中位 {v.median():.2f}  范围 {v.min():.2f}–{v.max():.2f}")

    print("\n" + "=" * 76)
    print("3. 标准化后在 val 上拟合的 3 参数融合 (w_fer, w_ctx)")
    print("=" * 76)
    wf, wc = wdf["w_fer"].dropna(), wdf["w_ctx"].dropna()
    print(f"  w_fer: mean={wf.mean():+.3f}  sd={wf.std():.3f}  "
          f"中位={wf.median():+.3f}  范围 {wf.min():+.3f} ~ {wf.max():+.3f}")
    print(f"  w_ctx: mean={wc.mean():+.3f}  sd={wc.std():.3f}  "
          f"中位={wc.median():+.3f}  范围 {wc.min():+.3f} ~ {wc.max():+.3f}")
    print(f"  w_fer 为正的折: {int((wf > 0).sum())}/{len(wf)}   "
          f"|w_fer| > |w_ctx| 的折: {int((wf.abs() > wc.abs()).sum())}/{len(wf)}")
    if fused_oof:
        print(f"  融合（仅 val 拟合 3 参数）test pooled_rank = {ers.pooled_rank(fused_oof):.4f}")

    print(f"\n落盘: results/tables/diag_branch_logits_{stamp}.csv")
    print(f"      results/tables/diag_branch_fusionw_{stamp}.csv")
    with open(REPORTS_DIR / f"diag_branch_logits_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "c_fer": C_FER, "c_ctx": C_CTX,
                   "w_fer_mean": float(wf.mean()), "w_ctx_mean": float(wc.mean()),
                   "w_fer_positive": int((wf > 0).sum()), "n_folds": int(len(wf)),
                   "scale_ratio_std_median": float(wdf["scale_ratio_std"].median()),
                   "fused_pooled_rank": float(ers.pooled_rank(fused_oof)) if fused_oof else None,
                   "note": "描述性诊断，不判定、不进论文数字"},
                  f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
