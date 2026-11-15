#!/usr/bin/env python
"""生成论文图（Figure 1-3）：全部由既有报告 JSON 的 point estimate / std 绘制，不重训、不重跑。

用法（项目根目录）:
    python scripts/make_figures.py

数据来源（均为 results/reports/ 下已定稿的报告）:
    - repeated_splits_20261104_181311.json  主表 11 臂 + 配对比较零分布 p95
    - minus_history_recheck_20261115_100010.json  minus_history 复核（新 seed 集合）
    - a7_quality_20261110_112148.json       A7 子集数字（只在子集内比较，单独成面板）
    - m2_finetune_20261106_145706.json / m4_fusion_20261109_132841.json  作为表中数字使用，不出图

图:
    fig1_main_results.pdf/.png
        (a) 11 臂 pooled_rank 点估计 ± sd，叠加主表零分布均值与 p95 带；
        (b) 四个配对比较 Δ = a − b 的点估计与置换零分布 p95（正/负方向）。
    注意：零分布只以 mean / p95 两个分位点从报告取得；带用均匀色块表示区间，
    不是从置换样本重建的分布。

fig2_seed_stability.pdf/.png
        B1v2 / M1 / M4_lr / minus_history 的逐 seed pooled_rank 折线（数据取自各自报告，
        M4_lr 取自 m4_fusion 报告）。显示：任何"点估计差"都在 seed 间波动之内。

fig3_signal_map.pdf/.png
        (a) 单变量 context 效应（miss 率差），由 events.csv 直接计算（描述性）；
        (b) 两级评估（全量 / A7 子集）下 M1 与 B1v2 的位置，说明 A7 的子集构成漂移。

所有数字与 results/reports/summary_5fold_20261104.md 一致；脚本不做任何统计推断。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "results" / "reports"
FIGURES = PROJECT_ROOT / "results" / "figures"
EVENTS = PROJECT_ROOT / "data" / "metadata" / "events.csv"

GRAY = "#4d4d4d"
BLUE = "#1f6fb2"
ORANGE = "#d95f02"
LIGHT = "#e8e8e8"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def fig_main_results(main: dict) -> None:
    arms = main["arms"]
    order = ["B1v2", "B2v2", "B3_b2_7", "B3_vgaf", "M1", "B4", "M3v2", "VAemb",
             "single356", "multimean", "multiva2"]
    null_mean = float(np.mean([arms[a]["null_mean"] for a in order]))
    null_p95 = float(np.mean([arms[a]["null_p95"] for a in order]))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    y = np.arange(len(order))
    vals = np.array([arms[a]["pooled_rank_mean"] for a in order])
    errs = np.array([arms[a]["pooled_rank_std"] for a in order])
    ax.axvspan(null_mean - 0.0, null_p95, color=LIGHT, zorder=0,
               label=f"null mean–p95 ({null_mean:.3f}–{null_p95:.3f})")
    ax.axvline(0.5, color=GRAY, lw=0.8, ls=":")
    for i, a in enumerate(order):
        color = BLUE if a == "B1v2" else GRAY
        ax.errorbar(vals[i], y[i], xerr=errs[i], fmt="o", color=color, ms=4.5,
                    capsize=2.5, lw=1.0, zorder=3)
        ax.text(vals[i] + errs[i] + 0.004, y[i], f"{vals[i]:.3f}  (p={arms[a]['p']:.2f})",
                va="center", fontsize=6.6, color=GRAY)
    ax.set_yticks(y)
    ax.set_yticklabels([a.replace("_", " ") for a in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.64)
    ax.set_xlabel("pooled rank AUC (mean $\\pm$ sd over 10 split seeds)", fontsize=8)
    ax.set_title("(a) All arms vs. permutation null", fontsize=8.5, loc="left")
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    ax = axes[1]
    comps = main["comparisons"]
    names = ["M1-B4", "M3v2-B1v2", "B2v2-B1v2", "VAemb-M1", "multimean-single356"]
    labels = ["M1 − B4\n(FER vs ImageNet)", "M3v2 − B1v2\n(fused vs context)",
              "B2v2 − B1v2\n(MLP vs LR)", "VAemb − M1\n(VA vs 8-class)",
              "multimean − single356\n(multi- vs single-frame)"]
    yy = np.arange(len(names))
    d = np.array([comps[n]["delta_mean"] for n in names])
    sd = np.array([comps[n]["delta_std"] for n in names])
    p95 = np.array([comps[n]["null_p95"] for n in names])
    for i in range(len(names)):
        ax.barh(yy[i], 2 * p95[i], left=-p95[i], height=0.55, color=LIGHT, zorder=0)
        ax.errorbar(d[i], yy[i], xerr=sd[i], fmt="o", color=GRAY, ms=4.5, capsize=2.5, lw=1.0, zorder=3)
        ax.text(d[i] + sd[i] + 0.006, yy[i], f"{d[i]:+.3f}", va="center", fontsize=6.8, color=GRAY)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(-0.13, 0.09)
    ax.set_xlabel("paired $\\Delta$ of pooled rank", fontsize=8)
    ax.set_title("(b) Pre-registered paired comparisons\n(grey band = permutation null $\\pm$p95)", fontsize=8.5, loc="left")
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig1_main_results.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_seed_stability(main: dict, m2: dict, recheck: dict) -> None:
    """逐 seed pooled_rank：主表 2027–2036（B1v2 / M1）与复核 2037–2046（full / minus_history）。

    两组 seed 集合不相交，横轴按序排列并在中间留空隙，避免误读成同一段 seed 序列。
    """
    seeds_main = main["split_seeds"]
    seeds_re = recheck["split_seeds"]
    gap = 1.2
    x_main = np.arange(len(seeds_main), dtype=float)
    x_re = np.arange(len(seeds_re), dtype=float) + len(seeds_main) + gap
    series = {
        "B1v2 (context LR), seeds 2027–36": (x_main, main["arms"]["B1v2"]["per_seed_pooled_rank"], BLUE, -0.08),
        "M1 (frozen FER), seeds 2027–36": (x_main, main["arms"]["M1"]["per_seed_pooled_rank"], ORANGE, 0.08),
        "full (= B1v2 recipe), seeds 2037–46": (x_re, recheck["variants"]["full"]["per_seed_pooled_rank"], GRAY, -0.08),
        "minus_history, seeds 2037–46": (x_re, recheck["variants"]["minus_history"]["per_seed_pooled_rank"], "#8c6bb1", 0.08),
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axvspan(-0.5, len(seeds_main) - 0.5 + 0.4, color="#f2f2f2", zorder=0)
    for label, (x, vals, c, off) in series.items():
        ax.plot(x + off, vals, marker="o", ms=3.4, lw=1.0, color=c, label=label)
    ax.axhline(float(np.mean(main["arms"]["B1v2"]["per_seed_pooled_rank"])), color=BLUE, lw=0.6, ls="--", alpha=0.7)
    ax.axhline(float(np.mean(recheck["variants"]["minus_history"]["per_seed_pooled_rank"])), color="#8c6bb1", lw=0.6, ls="--", alpha=0.7)
    ax.text(len(seeds_main) - 0.6, 0.418, "main seeds", fontsize=6.5, ha="right", color=GRAY)
    ax.text(len(seeds_main) + gap + 0.6, 0.418, "recheck seeds", fontsize=6.5, ha="left", color=GRAY)
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_xticks(list(x_main) + list(x_re))
    ax.set_xticklabels([str(s) for s in seeds_main] + [str(s) for s in seeds_re], fontsize=6.5)
    ax.set_xlabel("split seed", fontsize=8)
    ax.set_ylabel("pooled rank AUC", fontsize=8)
    ax.set_ylim(0.41, 0.60)
    ax.set_xlim(-0.8, x_re[-1] + 0.8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.3, frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.set_title("Point estimates move with the seed set; dashed = mean within each set", fontsize=8.5, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig2_seed_stability.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_signal_map(main: dict, a7: dict) -> None:
    ev = pd.read_csv(EVENTS, encoding="utf-8")
    v = ev[ev["exclusion_reason"].fillna("") == ""].copy()
    v["miss"] = (v["result"] == "miss").astype(float)
    v["is_trailing"] = (v["team_score_before"] < v["opponent_score_before"]).astype(float)
    v["second"] = (v["kick_order_in_round"] == "second").astype(float)
    base = v["miss"].mean()
    feats = {
        "trailing": "is_trailing",
        "sudden death": "sudden_death",
        "second kicker": "second",
        "round ≥ 4": None,
    }
    v["round_ge4"] = (v["round_number"] >= 4).astype(float)
    feats["round ≥ 4"] = "round_ge4"
    rows = []
    for name, col in feats.items():
        g = v.groupby(col)["miss"]
        rows.append((name, g.mean().loc[1.0] - g.mean().loc[0.0], len(v)))
    hist = (v["history_goals"] / v["history_attempts"].replace(0, np.nan)).fillna(0.5)
    hi = hist >= hist.median()
    rows.append(("history ≥ median", v.loc[hi, "miss"].mean() - v.loc[~hi, "miss"].mean(), len(v)))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    names = [r[0] for r in rows]
    diffs = np.array([r[1] for r in rows])
    yy = np.arange(len(names))
    ax.barh(yy, diffs, color=[ORANGE if d < 0 else BLUE for d in diffs], height=0.55)
    ax.axvline(0, color=GRAY, lw=0.8)
    for i, d in enumerate(diffs):
        ax.text(d + (0.004 if d >= 0 else -0.004), yy[i], f"{d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=7, color=GRAY)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"miss-rate difference (base rate = {base:.3f})", fontsize=8)
    ax.set_title("(a) Univariate context effects (descriptive)", fontsize=8.5, loc="left")
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    ax = axes[1]
    pairs = [("B1v2 context LR (image-independent)", main["arms"]["B1v2"]["pooled_rank_mean"], 0.5418, BLUE),
             ("M1 frozen FER", main["arms"]["M1"]["pooled_rank_mean"], 0.4494, ORANGE)]
    for i, (label, fullv, a7v, c) in enumerate(pairs):
        ax.plot([0, 1], [fullv, a7v], marker="o", ms=5, lw=1.2, color=c, label=label)
        ax.annotate(f"{fullv:.3f}", (0, fullv), textcoords="offset points", xytext=(-34, 0),
                    fontsize=7, color=c, va="center")
        ax.annotate(f"{a7v:.3f}", (1, a7v), textcoords="offset points", xytext=(7, 0),
                    fontsize=7, color=c, va="center")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.axhline(0.553, color=GRAY, lw=0.8, ls="--", alpha=0.8)
    ax.axhline(0.570, color=GRAY, lw=0.8, ls="--", alpha=0.8)
    ax.text(1.44, 0.553, "null p95\n(385)", fontsize=6.3, color=GRAY, va="bottom")
    ax.text(1.44, 0.570, "null p95\n(333)", fontsize=6.3, color=GRAY, va="top")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["all 385 kicks\n(40 matches)", "quality subset\n333 kicks (35 matches)"], fontsize=7.5)
    ax.set_xlim(-0.35, 1.75)
    ax.set_ylim(0.43, 0.585)
    ax.set_ylabel("pooled rank AUC", fontsize=8)
    ax.set_title("(b) A7 subset: both arms drop together", fontsize=8.5, loc="left")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.6, frameon=False, loc="lower left")
    ax.grid(axis="y", lw=0.4, alpha=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig3_signal_map.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    main_r = load("repeated_splits_20261104_181311.json")
    m2_r = load("m2_finetune_20261106_145706.json")
    recheck = load("minus_history_recheck_20261115_100010.json")
    a7 = load("a7_quality_20261110_112148.json")
    fig_main_results(main_r)
    fig_seed_stability(main_r, m2_r, recheck)
    fig_signal_map(main_r, a7)
    print("图已写入 results/figures/：fig1_main_results, fig2_seed_stability, fig3_signal_map (.pdf/.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
