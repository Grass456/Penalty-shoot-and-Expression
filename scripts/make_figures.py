#!/usr/bin/env python
"""Generate the paper figures from finalized report files.

Usage from the project root::

    python scripts/make_figures.py

The script only reads finalized JSON/CSV reports. It does not retrain models or
recompute any statistics. Generated files are written to results/figures:

* fig1_all_arms - the requested 14-arm comparison;
* fig1_comparisons - the four requested pairwise comparisons;
* fig2_seed_stability - B1 and M1 over the main split seeds;
* fig3_context_ablation - the primary context feature-group ablation;
* fig3_a7_quality - the A7-high subset results for B1, M1, and M3.

Display labels intentionally use the paper-facing names requested for this
version. B1 and M3 are used instead of report keys B1v2 and M3v2. There is no
v1/v2 distinction in the figures.
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
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"

GRAY = "#4d4d4d"
BLUE = "#1f6fb2"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
LIGHT = "#e8e8e8"


def load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_arm_records(main: dict, m4: dict) -> dict[str, dict]:
    """Return the 14 requested arms under the requested display names."""
    records = {
        "B1": main["arms"]["B1v2"],
        "B2": main["arms"]["B2v2"],
        "B3_vgaf": main["arms"]["B3_vgaf"],
        "B3_b2_7": main["arms"]["B3_b2_7"],
        "M1": main["arms"]["M1"],
        "B4": main["arms"]["B4"],
        "M3": main["arms"]["M3v2"],
        "VAemb": main["arms"]["VAemb"],
        "single356": main["arms"]["single356"],
        "multimean": main["arms"]["multimean"],
        "multiva2": main["arms"]["multiva2"],
    }
    for name in ("M2", "M4_lr", "M4_mlp"):
        records[name] = m4["obs_summary"][name]
    return records


def fig1_all_arms(main: dict, m4: dict) -> None:
    order = [
        "B1", "B2", "B3_vgaf", "B3_b2_7", "M1", "B4", "M3", "VAemb",
        "single356", "multimean", "multiva2", "M2", "M4_lr", "M4_mlp",
    ]
    records = main_arm_records(main, m4)
    main_order = list(main["arms"])
    null_mean = float(np.mean([main["arms"][a]["null_mean"] for a in main_order]))
    null_p95 = float(np.mean([main["arms"][a]["null_p95"] for a in main_order]))

    colors = {name: GRAY for name in order}
    colors.update({
        "B1": BLUE, "M1": ORANGE, "M2": PURPLE,
        "M4_lr": "#238b45", "M4_mlp": "#66a61e",
    })

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    y = np.arange(len(order))
    vals = np.array([records[name]["pooled_rank_mean"] for name in order])
    errs = np.array([records[name]["pooled_rank_std"] for name in order])
    ax.axvspan(
        null_mean,
        null_p95,
        color=LIGHT,
        zorder=0,
        label=f"permutation null mean-p95 ({null_mean:.3f}-{null_p95:.3f})",
    )
    ax.axvline(0.5, color=GRAY, lw=0.8, ls=":")
    for i, name in enumerate(order):
        ax.errorbar(
            vals[i],
            y[i],
            xerr=errs[i],
            fmt="o",
            color=colors[name],
            ms=4.8,
            capsize=2.5,
            lw=1.0,
            zorder=3,
        )
        ax.text(
            vals[i] + errs[i] + 0.004,
            y[i],
            f"{vals[i]:.3f}",
            va="center",
            fontsize=6.8,
            color=GRAY,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.64)
    ax.set_xlabel("pooled rank AUC (mean +/- sd over 10 split seeds)", fontsize=8)
    ax.set_title("All arms", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    fig.tight_layout()
    save_figure(fig, "fig1_all_arms")


def fig1_comparisons(main: dict, m4: dict) -> None:
    comparisons = [
        ("M1 - B4", main["comparisons"]["M1-B4"]),
        ("M4_lr - B1", m4["comparisons"]["M4_lr - B1v2"]),
        ("VAemb - M1", main["comparisons"]["VAemb-M1"]),
        (
            "single356 - multimean",
            {
                "delta_mean": -main["comparisons"]["multimean-single356"]["delta_mean"],
                "delta_std": main["comparisons"]["multimean-single356"]["delta_std"],
                "null_p95": main["comparisons"]["multimean-single356"]["null_p95"],
            },
        ),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(comparisons))
    for i, (_, result) in enumerate(comparisons):
        p95 = float(result["null_p95"])
        delta = float(result["delta_mean"] if "delta_mean" in result else result["delta_mean_10seed"])
        sd = float(result["delta_std"] if "delta_std" in result else result["delta_std_10seed"])
        ax.barh(y[i], 2 * p95, left=-p95, height=0.55, color=LIGHT, zorder=0)
        ax.errorbar(
            delta,
            y[i],
            xerr=sd,
            fmt="o",
            color=GRAY,
            ms=4.8,
            capsize=2.5,
            lw=1.0,
            zorder=3,
        )
        ax.text(
            delta + sd + 0.006,
            y[i],
            f"{delta:+.3f}",
            va="center",
            fontsize=6.8,
            color=GRAY,
        )
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in comparisons], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.13, 0.13)
    ax.set_xlabel("paired delta of pooled rank (first - second)", fontsize=8)
    ax.set_title("Requested pairwise comparisons", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    fig.tight_layout()
    save_figure(fig, "fig1_comparisons")


def fig2_seed_stability(main: dict) -> None:
    seeds = np.array(main["split_seeds"], dtype=int)
    b1 = np.array(main["arms"]["B1v2"]["per_seed_pooled_rank"], dtype=float)
    m1 = np.array(main["arms"]["M1"]["per_seed_pooled_rank"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(seeds, b1, marker="o", ms=3.8, lw=1.1, color=BLUE, label="B1")
    ax.plot(seeds, m1, marker="o", ms=3.8, lw=1.1, color=ORANGE, label="M1")
    ax.axhline(float(np.mean(b1)), color=BLUE, lw=0.7, ls="--", alpha=0.8)
    ax.axhline(float(np.mean(m1)), color=ORANGE, lw=0.7, ls="--", alpha=0.8)
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_xticks(seeds)
    ax.set_xticklabels([str(seed) for seed in seeds], fontsize=6.8)
    ax.set_xlabel("split seed", fontsize=8)
    ax.set_ylabel("pooled rank AUC", fontsize=8)
    ax.set_ylim(0.41, 0.59)
    ax.set_xlim(seeds[0] - 0.5, seeds[-1] + 0.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.set_title("Seed stability on the main split set", fontsize=9, loc="left")
    fig.tight_layout()
    save_figure(fig, "fig2_seed_stability")


def fig3_context_ablation(ablation: dict) -> None:
    display = {
        "full": "Full context",
        "minus_round_order": "Without round/order",
        "minus_score_state": "Without score state",
        "minus_history": "Without history",
    }
    variants = ablation["primary_variants"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(variants))
    for i, variant in enumerate(variants):
        result = ablation["variants"][variant]
        color = BLUE if variant == "full" else GRAY
        ax.errorbar(
            result["pooled_rank_mean"],
            y[i],
            xerr=result["pooled_rank_std"],
            fmt="o",
            color=color,
            ms=4.8,
            capsize=2.5,
            lw=1.0,
            zorder=3,
        )
        ax.text(
            result["pooled_rank_mean"] + result["pooled_rank_std"] + 0.004,
            y[i],
            f"{result['pooled_rank_mean']:.3f}",
            va="center",
            fontsize=6.8,
            color=GRAY,
        )
    ax.axvline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([display[variant] for variant in variants], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 0.60)
    ax.set_xlabel("pooled rank AUC (mean +/- sd over 10 split seeds)", fontsize=8)
    ax.set_title("Context feature-group ablation", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    fig.tight_layout()
    save_figure(fig, "fig3_context_ablation")


def fig3_a7_quality(a7: dict) -> None:
    table = pd.read_csv(TABLES / "a7_quality_20261110_112148.csv")
    subset = table[(table["proxy"] == "crop_short") & (table["regime"] == "A7-high")].copy()
    display = {"B1v2": "B1", "M1": "M1", "M3v2": "M3"}
    order = ["M1", "M3v2", "B1v2"]
    colors = {"M1": ORANGE, "M3v2": GRAY, "B1v2": BLUE}
    grouped = subset.groupby("arm")["pooled_rank"]
    means = grouped.mean().reindex(order).to_numpy()
    sds = grouped.std(ddof=1).reindex(order).to_numpy()

    null = a7["verdict"]["crop_short"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(order))
    for i, arm in enumerate(order):
        ax.errorbar(
            means[i],
            y[i],
            xerr=sds[i],
            fmt="o",
            color=colors[arm],
            ms=4.8,
            capsize=2.5,
            lw=1.0,
            zorder=3,
        )
        ax.text(
            means[i] + sds[i] + 0.004,
            y[i],
            f"{means[i]:.3f}",
            va="center",
            fontsize=6.8,
            color=GRAY,
        )
    ax.axvspan(
        null["H9a_null_mean"],
        null["H9a_null_p95"],
        color=LIGHT,
        zorder=0,
        label=f"M1 permutation null mean-p95 ({null['H9a_null_mean']:.3f}-{null['H9a_null_p95']:.3f})",
    )
    ax.axvline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([display[arm] for arm in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.60)
    ax.set_xlabel("pooled rank AUC on A7-high (mean +/- sd over 10 split seeds)", fontsize=8)
    ax.set_title("A7 high-quality subset (crop short side >= 80 px)", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    fig.tight_layout()
    save_figure(fig, "fig3_a7_quality")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    main_report = load_json("repeated_splits_20261104_181311.json")
    m4_report = load_json("m4_fusion_20261109_132841.json")
    ablation_report = load_json("context_ablation_20261104_182831.json")
    a7_report = load_json("a7_quality_20261110_112148.json")

    fig1_all_arms(main_report, m4_report)
    fig1_comparisons(main_report, m4_report)
    fig2_seed_stability(main_report)
    fig3_context_ablation(ablation_report)
    fig3_a7_quality(a7_report)
    print("Wrote five split figures to results/figures/ (.pdf and .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
