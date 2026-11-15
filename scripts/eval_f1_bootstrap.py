#!/usr/bin/env python
"""补算主重复划分实验的 goal-class F1 与比赛级 bootstrap CI。

本脚本复用 ``eval_repeated_splits.py`` 的数据、折划分、特征构造和训练协议，
不会修改既有 folds。它重新得到每个 split seed 的 OOF 概率，然后计算：

* ``pooled_f1_goal``：以 ``p >= 0.5`` 判为 goal 的正类 F1；
* pooled_rank、AUROC、AUPRC、balanced accuracy、Brier、LogLoss；
* 以比赛为 cluster 的 percentile bootstrap 95% CI。

Bootstrap 的对象是 40 场比赛：每次有放回抽取比赛，保留该比赛在每个
split seed 中的全部 OOF 预测，再对 10 个 split seed 的指标取均值。这样不会
把同一场比赛中的罚球当成独立样本；split seed 造成的训练/划分波动则由逐 seed
结果和 mean ± sd 单独报告。

用法（项目根目录，需完整 penalty-fer 环境）::

    python scripts/eval_f1_bootstrap.py --n-bootstrap 2000
    python scripts/eval_f1_bootstrap.py --seeds 2027-2036 --n-bootstrap 2000

输出::

    results/tables/f1_bootstrap_<stamp>_per_seed.csv
    results/tables/f1_bootstrap_<stamp>_summary.csv
    results/reports/f1_bootstrap_<stamp>.json

默认跑主表的 11 个 repeated-split 臂。加入 ``--include-finetune`` 后，脚本还会
复用 M4 的 FER 微调管线，额外输出 M2、M4_lr、M4_mlp 及两个敏感性臂的 F1 与 CI；
该选项需要 PyTorch 和人脸图像，运行时间明显更长。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, log_loss,
                             roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_arms(s: str | None, names: list[str]) -> list[str]:
    if not s:
        return names
    requested = [x.strip() for x in s.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}; available={names}")
    return requested


def records_from_oof(oof: list[tuple], valid: pd.DataFrame) -> list[dict]:
    """把 run_arm 的 OOF tuples 转成带 match_id/fold_id 的可 bootstrap 记录。"""
    out = []
    for fold_id, (idx, y, p) in enumerate(oof):
        idx = np.asarray(idx, dtype=int)
        out.append({
            "fold_id": int(fold_id),
            "match_id": valid.iloc[idx]["match_id"].astype(str).to_numpy(),
            "y": np.asarray(y, dtype=int),
            "p": np.asarray(p, dtype=float),
        })
    return out


def metrics_from_records(records: list[dict], sampled_matches: np.ndarray | None = None) -> dict:
    """计算一组 OOF 记录的 pooled 指标。

    sampled_matches 为 cluster bootstrap 的比赛 ID 序列；重复出现的比赛会
    重复追加其全部事件，从而保留 cluster 相关性。
    """
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    for rec in records:
        if sampled_matches is None:
            y, p = rec["y"], rec["p"]
        else:
            parts_y, parts_p = [], []
            for match_id in sampled_matches:
                mask = rec["match_id"] == match_id
                if mask.any():
                    parts_y.append(rec["y"][mask])
                    parts_p.append(rec["p"][mask])
            if not parts_y:
                continue
            y = np.concatenate(parts_y)
            p = np.concatenate(parts_p)
        if len(y) == 0:
            continue
        ys.append(y)
        ps.append(p)
        ranks.append((pd.Series(p).rank().to_numpy() - 0.5) / len(p))

    if not ys:
        return {k: float("nan") for k in (
            "pooled_rank", "pooled_auroc", "pooled_auprc", "pooled_f1_goal",
            "pooled_balanced_acc", "pooled_brier", "pooled_logloss")}

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    rank_all = np.concatenate(ranks)
    pred_all = (p_all >= 0.5).astype(int)
    if len(np.unique(y_all)) < 2:
        auc = float("nan")
        auprc = float("nan")
        rank_auc = float("nan")
    else:
        auc = float(roc_auc_score(y_all, p_all))
        auprc = float(average_precision_score(y_all, p_all))
        rank_auc = float(roc_auc_score(y_all, rank_all))
    return {
        "pooled_rank": rank_auc,
        "pooled_auroc": auc,
        "pooled_auprc": auprc,
        "pooled_f1_goal": float(f1_score(y_all, pred_all, zero_division=0)),
        "pooled_balanced_acc": float(balanced_accuracy_score(y_all, pred_all)),
        "pooled_brier": float(brier_score_loss(y_all, p_all)),
        "pooled_logloss": float(log_loss(y_all, np.clip(p_all, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }


def bootstrap_mean_ci(by_seed: dict[int, list[dict]], match_ids: np.ndarray,
                      n_bootstrap: int, seed: int) -> dict:
    """Cluster bootstrap CI for the mean metric over split seeds."""
    rng = np.random.default_rng(seed)
    metrics = ("pooled_rank", "pooled_auroc", "pooled_auprc", "pooled_f1_goal",
               "pooled_balanced_acc", "pooled_brier", "pooled_logloss")
    boot = {m: np.empty(n_bootstrap, dtype=float) for m in metrics}
    for b in range(n_bootstrap):
        sampled = rng.choice(match_ids, size=len(match_ids), replace=True)
        per_metric = {m: [] for m in metrics}
        for records in by_seed.values():
            one_seed = metrics_from_records(records, sampled)
            for metric in metrics:
                per_metric[metric].append(one_seed[metric])
        for metric in metrics:
            vals = np.asarray(per_metric[metric], dtype=float)
            boot[metric][b] = float(np.nanmean(vals))
    return {
        metric: {
            "lower": float(np.nanpercentile(values, 2.5)),
            "upper": float(np.nanpercentile(values, 97.5)),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(seed),
        }
        for metric, values in boot.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="2027-2036")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=2048)
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm names; default: all available arms")
    ap.add_argument("--include-finetune", action="store_true",
                    help="also run M2/M4 fine-tuned arms (requires PyTorch and face images)")
    ap.add_argument("--device", default=None, help="cuda / cpu for --include-finetune")
    args = ap.parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap should be at least 100")

    # Import after argparse so ``--help`` works even when the training environment
    # (notably PyTorch) has not been installed yet.
    import eval_repeated_splits as ers  # noqa: PLC0415

    seeds = parse_seeds(args.seeds)
    arm_by_name = {a["name"]: a for a in ers.ARMS}
    finetune_names = ["M2", "M4_lr", "M4_mlp", "M4_lr_tr", "M4_mlp_tr"]
    available_names = list(arm_by_name) + (finetune_names if args.include_finetune else [])
    arm_names = parse_arms(args.arms, available_names)
    repeated_names = [name for name in arm_names if name in arm_by_name]
    finetune_requested = [name for name in arm_names if name in finetune_names]
    if finetune_requested and not args.include_finetune:
        raise ValueError("M2/M4 require --include-finetune")
    data = ers.load_data()
    valid = data["valid"]
    match_ids = np.sort(valid["match_id"].astype(str).unique())

    m4 = None
    ft_cfg = None
    pristine = None
    emb_dim = None
    device = None
    X = None
    if finetune_requested:
        import eval_m4_fusion as m4  # noqa: PLC0415
        import torch  # noqa: PLC0415
        m4.setup_determinism()
        ft_cfg = m4.load_cfg()
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        data_ft, X, ids = m4.load_all(ft_cfg)
        if data_ft["ids"] != data["ids"] or ids != data["ids"]:
            raise AssertionError("fine-tune data row order does not match repeated-split data")
        pristine, emb_dim = m4.build_pristine(ft_cfg, device)
        print(f"fine-tune device={device} arms={finetune_requested}", flush=True)

    per_seed_rows: list[dict] = []
    records: dict[str, dict[int, list[dict]]] = {name: {} for name in arm_names}
    for split_seed in seeds:
        built = ers.make_folds(data, data["y"], split_seed)
        if built is None:
            raise RuntimeError(f"could not build valid folds for seed {split_seed}")
        _, folds = built
        for name in repeated_names:
            oof = ers.run_arm(data, arm_by_name[name], folds, data["y"], split_seed)
            records[name][split_seed] = records_from_oof(oof, valid)
            row = {"arm": name, "split_seed": int(split_seed),
                   **metrics_from_records(records[name][split_seed])}
            per_seed_rows.append(row)
        if finetune_requested:
            ft = m4.run_seed(X, data, folds, data["y"], split_seed, ft_cfg,
                             pristine, emb_dim, device)
            for name in finetune_requested:
                oof = ft["oof"][name]
                records[name][split_seed] = records_from_oof(oof, valid)
                row = {"arm": name, "split_seed": int(split_seed),
                       **metrics_from_records(records[name][split_seed])}
                per_seed_rows.append(row)
        print(f"finished split seed {split_seed}", flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    metric_names = [c for c in per_seed.columns if c not in ("arm", "split_seed")]
    summary_rows = []
    ci_by_arm = {}
    for name in arm_names:
        sub = per_seed[per_seed["arm"] == name]
        row = {"arm": name, "n_split_seeds": int(len(sub))}
        for metric in metric_names:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_sd"] = float(sub[metric].std(ddof=1))
        ci = bootstrap_mean_ci(records[name], match_ids, args.n_bootstrap,
                               args.bootstrap_seed)
        ci_by_arm[name] = ci
        for metric, bounds in ci.items():
            row[f"{metric}_bootstrap_lower"] = bounds["lower"]
            row[f"{metric}_bootstrap_upper"] = bounds["upper"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_path = TABLES_DIR / f"f1_bootstrap_{stamp}_per_seed.csv"
    summary_path = TABLES_DIR / f"f1_bootstrap_{stamp}_summary.csv"
    report_path = REPORTS_DIR / f"f1_bootstrap_{stamp}.json"
    per_seed.to_csv(per_seed_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "补算 goal-class F1 与比赛级 cluster bootstrap CI",
        "split_seeds": seeds,
        "arms": arm_names,
        "n_valid_events": int(len(valid)),
        "n_matches": int(len(match_ids)),
        "threshold": 0.5,
        "positive_class": "goal",
        "bootstrap": {
            "unit": "match_id",
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.bootstrap_seed),
            "aggregation": "mean over split seeds after each match-cluster resample",
            "ci": "percentile 95% (2.5%, 97.5%)",
            "does_not_include_training_variation": "split-seed mean ± sd reports that separately",
        },
        "per_seed": per_seed.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "bootstrap_ci": ci_by_arm,
        "outputs": {
            "per_seed_csv": per_seed_path.relative_to(PROJECT_ROOT).as_posix(),
            "summary_csv": summary_path.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"per-seed table: {per_seed_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"summary table:  {summary_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"report:         {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
