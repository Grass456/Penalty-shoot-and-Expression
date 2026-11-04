#!/usr/bin/env python
"""A5 context 特征组消融：逐组移除后重跑，看性能由哪一类变量支撑。

预注册判读标准见 results/reports/context_ablation_criteria_preregistered.txt（跑之前写定）。

用法（项目根目录）:
    python scripts/eval_context_ablation.py                  # 4 主变体 + 6 单维 + MLP，200 次置换
    python scripts/eval_context_ablation.py --n-perm 0        # 只算观测值
    python scripts/eval_context_ablation.py --variants full,minus_history --seeds 2027

协议:
    - 划分 / 统计量 / 置换与 eval_repeated_splits.py 完全一致（同一 make_folds、同一 pooled_rank、
      同一 perm seed 12345、5000 次 max_iter 的 LR、train-only 标准化）；
    - 特征矩阵 = build_context_matrix_v2 的**列子集** —— 同一函数、同一 train-only p̄ 与 α，
      消融只改变输入维度，不引入预处理语义差异；
    - 主臂 LR C=1.0（与 B1v2 相同）；MLP 变体（B2v2 设定）只报观测值，不做置换（成本）。

前置断言: full 变体须逐 seed 复现主表的 B1v2（table repeated_splits_20261104_181311.csv，
    |Δ| < 1e-6），否则中止 —— 消融实现若引入额外变化，全部 Δ 不可信。
    （B1v2 是 LR 臂，在修正 MLP init 前后的两张主表里逐位相同。）

输出: results/tables/context_ablation_<stamp>.csv、results/reports/context_ablation_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_repeated_splits as ers  # noqa: E402  复用 folds / 统计量 / 置换口径

TABLES_DIR = ers.TABLES_DIR
REPORTS_DIR = ers.REPORTS_DIR
CRITERIA_PATH = REPORTS_DIR / "context_ablation_criteria_preregistered.txt"
REPEATED_CSV = TABLES_DIR / "repeated_splits_20261104_181311.csv"   # 修正 MLP init 后的主表

DEFAULT_SEEDS = "2027-2036"
FULL_COLS = ["sudden_death", "is_trailing", "kick_order_second", "score_diff",
             "has_history", "history_rate"]
GROUPS = {                                    # plan §6.6 的五类 → 实际编码的三类
    "round_order": ["sudden_death", "kick_order_second"],
    "score_state": ["is_trailing", "score_diff"],
    "history": ["has_history", "history_rate"],
}
PRIMARY = ["full"] + [f"minus_{g}" for g in GROUPS]
ALPHA = 0.05
MIN_SEEDS_POSITIVE = 8


def cols_for(variant: str) -> list[str]:
    """变体名 → 保留的列（顺序恒定，等于 FULL_COLS 的子序列）。"""
    if variant == "full":
        return list(FULL_COLS)
    if variant.startswith("minus_"):
        drop = GROUPS.get(variant[6:], [variant[6:]])     # 组名或单维名
        keep = [c for c in FULL_COLS if c not in drop]
        if not keep:
            raise ValueError(f"{variant}: 移除后无剩余列")
        return keep
    raise ValueError(f"未知变体 {variant}")


def all_variants() -> list[str]:
    return PRIMARY + [f"minus_{c}" for c in FULL_COLS]


def p_right(null: np.ndarray, obs: float) -> float:
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


# ---------------------------------------------------------------- 单变体

def run_variant(data: dict, cols: list[str], folds: list[dict], y: np.ndarray,
                 seed: int) -> list[tuple]:
    """LR C=1.0 单变体：列子集 → train-only 标准化 → 逐折 test 概率。"""
    from sklearn.linear_model import LogisticRegression

    out = []
    for fold in folds:
        tr, te = fold["train"], fold["test"]
        fc = data["cfg"]["features"]
        ctx = ers.build_context_matrix_v2(data["valid"], p_bar=float(y[tr].mean()),
                                          alpha=fc["history_alpha"], clip=fc["score_diff_clip"])
        X = ctx[cols].to_numpy(dtype=np.float64)
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        lr = LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
        lr.fit(Z[tr], y[tr])
        out.append((te, y[te], lr.predict_proba(Z[te])[:, 1]))
    return out


def run_variant_mlp(data: dict, cols: list[str], folds: list[dict], y: np.ndarray,
                     seed: int) -> list[tuple]:
    """B2v2 设定的 MLP 变体（次要、描述性；只报观测值）。"""
    import torch
    from src.models.context_mlp import ContextMLP
    from src.training.train_loop import predict, train_one_fold

    torch.set_num_threads(1)
    mc, tc = data["cfg"]["model"], data["cfg"]["training"]
    out = []
    for fold in folds:
        tr, va, te = fold["train"], fold["val"], fold["test"]
        fc = data["cfg"]["features"]
        ctx = ers.build_context_matrix_v2(data["valid"], p_bar=float(y[tr].mean()),
                                          alpha=fc["history_alpha"], clip=fc["score_diff_clip"])
        X = ctx[cols].to_numpy(dtype=np.float64)
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        torch.manual_seed(seed)
        model = ContextMLP(in_dim=Z.shape[1], hidden=mc["hidden"], dropout=mc["dropout"])
        train_one_fold(model, Z[tr], y[tr], Z[va], y[va], seed=seed,
                       max_epochs=tc["max_epochs"], patience=tc["patience"], lr=tc["lr"],
                       weight_decay=tc["weight_decay"], batch_size=tc["batch_size"])
        out.append((te, y[te], predict(model, Z[te])))
    return out


def replicate(data: dict, y: np.ndarray, variants: list[str], seeds: list[int],
              full: bool = False) -> dict | None:
    """在标签 y 下跑全部 seed × 全部变体（LR）。"""
    res: dict[str, list] = {v: [] for v in variants}
    for s in seeds:
        built = ers.make_folds(data, y, s)
        if built is None:
            return None
        _, folds = built
        for v in variants:
            oof = run_variant(data, cols_for(v), folds, y, s)
            res[v].append(ers.full_stats(oof) if full else ers.pooled_rank(oof))
    return res


def perm_worker_factory(variants: list[str]):
    """返回带 folds 缓存的置换 worker（每进程一份缓存，按标签向量的字节做键）。

    建 folds（StratifiedGroupKFold）在置换阶段占大头，缓存后单次置换的成本主要落在
    LR 拟合上；缓存设上界以免进程内无界增长。
    """
    cache: dict[bytes, object] = {}

    def worker(data: dict, seeds: list[int], perm_seed: int) -> dict:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        rng = np.random.default_rng(perm_seed)
        for attempt in range(20):
            yp = rng.permutation(data["y"])
            key = yp.tobytes()
            if key in cache:
                folds_all = cache[key]
            else:
                folds_all = []
                for s in seeds:
                    built = ers.make_folds(data, yp, s)
                    if built is None:
                        folds_all = None
                        break
                    folds_all.append(built[1])
                cache[key] = folds_all
                if len(cache) > 8:
                    cache.pop(next(iter(cache)))
            if folds_all is None:
                continue
            res: dict[str, list] = {v: [] for v in variants}
            for folds, s in zip(folds_all, seeds):
                for v in variants:
                    res[v].append(ers.pooled_rank(
                        run_variant(data, cols_for(v), folds, yp, s)))
            return {"pr": {k: float(np.mean(v)) for k, v in res.items()},
                    "per_seed": {k: [float(x) for x in v] for k, v in res.items()},
                    "redraws": attempt}
        raise RuntimeError(f"perm {perm_seed}: 20 次重抽都缺类")

    return worker


# ---------------------------------------------------------------- 前置断言

def guard_full(obs: dict, seeds: list[int]) -> dict:
    """full 变体必须逐 seed 复现 11-04 的 B1v2，否则中止（LR 臂无随机性）。"""
    if not REPEATED_CSV.exists():
        print(f"缺少对照文件 {REPEATED_CSV}，无法做前置断言", file=sys.stderr)
        raise SystemExit(1)
    ref = pd.read_csv(REPEATED_CSV)
    ref = ref[ref["arm"] == "B1v2"].set_index("split_seed")["pooled_rank"]
    dev = []
    for i, s in enumerate(seeds):
        if s not in ref.index:
            print(f"对照文件缺 B1v2 / seed {s}", file=sys.stderr)
            raise SystemExit(1)
        dev.append(obs["full"][i]["pooled_rank"] - float(ref.loc[s]))
    worst = float(np.max(np.abs(dev)))
    print(f"前置断言: full vs 11-04 的 B1v2，最大偏差 {worst:.2e}（逐位限 1e-06）")
    if worst > 1e-6:
        print("  !! full 变体未复现 B1v2，消融实现已偏离，中止", file=sys.stderr)
        raise SystemExit(1)
    return {"reference": REPEATED_CSV.relative_to(PROJECT_ROOT).as_posix(),
            "max_abs_dev": worst}


# ---------------------------------------------------------------- 判读

def verdicts(obs: dict, perms: list[dict], variants: list[str]) -> dict:
    """按预注册标准判读：每组 Δ = full − minus_group 的 seed 一致性与配对置换 p。"""
    out = {}
    for v in variants:
        pr = np.array([s["pooled_rank"] for s in obs[v]])
        e = {"pooled_rank_mean": float(pr.mean()), "pooled_rank_std": float(pr.std()),
             "per_seed_pooled_rank": [round(float(x), 4) for x in pr],
             "cols": cols_for(v), "n_test": obs[v][0]["n_test"]}
        for k in ("fold_mean", "pooled_auprc", "pooled_brier", "pooled_logloss",
                  "pooled_balanced_acc"):
            e[f"{k}_mean"] = float(np.mean([s[k] for s in obs[v]]))
        if perms and v in PRIMARY:
            null = np.array([p["pr"][v] for p in perms])
            e.update({"null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                      "p": p_right(null, e["pooled_rank_mean"])})
            e["has_signal"] = e["p"] < ALPHA
        out[v] = e

    comps = {}
    if perms:
        for g in GROUPS:
            a, b = "full", f"minus_{g}"
            da = np.array([s["pooled_rank"] for s in obs[a]])
            db = np.array([s["pooled_rank"] for s in obs[b]])
            d = da - db
            null = np.array([np.mean(np.array(p["per_seed"][a]) - np.array(p["per_seed"][b]))
                             for p in perms])
            e = {"delta_mean": float(d.mean()), "delta_std": float(d.std()),
                 "n_seeds_positive": int((d > 0).sum()), "n_seeds": int(len(d)),
                 "per_seed_delta": [round(float(x), 4) for x in d],
                 "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                 "p": p_right(null, float(d.mean()))}
            e["group_has_positive_contribution"] = bool(
                e["n_seeds_positive"] >= MIN_SEEDS_POSITIVE and e["p"] < ALPHA and e["delta_mean"] > 0)
            comps[g] = e
    return {"variants": out, "groups": comps}


# ---------------------------------------------------------------- 主流程

def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--n-perm", type=int, default=200, help="配对置换次数（0=跳过）")
    ap.add_argument("--perm-seed", type=int, default=12345)
    ap.add_argument("--n-jobs", type=int, default=-2)
    ap.add_argument("--variants", default=",".join(all_variants()),
                    help="逗号分隔的变体名（full / minus_<组名或单维名>）")
    ap.add_argument("--with-mlp", action="store_true",
                    help="额外跑 full 的 MLP 变体（次要、描述性，不做置换）")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    seeds = parse_seeds(args.seeds)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        cols_for(v)   # 早失败：未知变体名立即报错
    data = ers.load_data()
    print(f"有效样本 {len(data['ids'])}，split seeds {seeds}，LR 变体 {len(variants)} 个"
          f"（主判定 {len(PRIMARY)}）")

    t0 = time.time()
    obs = replicate(data, data["y"], variants, seeds, full=True)
    if obs is None:
        print("真实标签下划分缺类，检查 build_folds", file=sys.stderr)
        return 1
    print(f"观测值完成 {time.time() - t0:.0f}s")
    guard = guard_full(obs, seeds)

    mlp_obs = None
    if args.with_mlp:
        t = time.time()
        mlp_obs = {}
        for s in seeds:
            _, folds = ers.make_folds(data, data["y"], s)
            oof = run_variant_mlp(data, cols_for("full"), folds, data["y"], s)
            mlp_obs[s] = ers.full_stats(oof)
        print(f"MLP 变体完成 {time.time() - t:.0f}s（描述性，不做置换）")

    for v in variants:
        e = obs[v]
        pr = np.array([s["pooled_rank"] for s in e])
        print(f"  {v:>22} pooled_rank {pr.mean():.4f}±{pr.std():.4f}  cols={len(cols_for(v))}")

    perms: list[dict] = []
    if args.n_perm > 0:
        t1 = time.time()
        # 主判定变体跑置换；次要变体（单维）只在观测阶段出现，不参与显著性
        worker = perm_worker_factory([v for v in variants if v in PRIMARY])
        perms = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(worker)(data, seeds, args.perm_seed + i) for i in range(args.n_perm))
        print(f"置换 {args.n_perm} 次完成 {time.time() - t1:.0f}s，"
              f"重抽总数 {sum(p['redraws'] for p in perms)}")

    v = verdicts(obs, perms, variants)

    print(f"\n{'变体':>22} | {'pooled_rank':>12} | {'p':>6} | 信号")
    for name, e in v["variants"].items():
        print(f"{name:>22} | {e['pooled_rank_mean']:.4f}±{e['pooled_rank_std']:.3f} | "
              f"{e.get('p', float('nan')):>6.3f} | {e.get('has_signal', '-')}")
    print()
    for g, e in v["groups"].items():
        print(f"移除 {g:>12}: Δ(full−minus) {e['delta_mean']:+.4f}±{e['delta_std']:.4f}  "
              f"{e['n_seeds_positive']}/{e['n_seeds']} seed>0  p={e.get('p', float('nan')):.3f}  "
              f"有正贡献={e.get('group_has_positive_contribution', '-')}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [{"variant": name, "split_seed": s, **st}
            for name, lst in obs.items() for s, st in zip(seeds, lst)]
    if mlp_obs:
        rows += [{"variant": "full_mlp", "split_seed": s, **st} for s, st in mlp_obs.items()]
    table = TABLES_DIR / f"context_ablation_{stamp}.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "A5：context 特征组消融（逐组移除），判断性能由哪类变量支撑",
        "criteria": CRITERIA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_seeds": seeds, "n_perm": args.n_perm, "perm_seed": args.perm_seed,
        "perm_redraws_total": sum(p["redraws"] for p in perms),
        "n_valid_events": len(data["ids"]),
        "guard": guard,
        "primary_variants": PRIMARY, "feature_groups": GROUPS, "full_cols": FULL_COLS,
        "not_modeled": {                      # 用户 2026-11-04 决定不纳入
            "venue_status": "主客场未编码（plan §6.6.2），记为偏离",
            "competition_category": "赛事类别未编码（plan §6.6.5），虽可由 matches.competition "
                                    "直接映射为 3 类，经用户决定不纳入",
        },
        "mlp_variant_note": ("full_mlp 为描述性，不参与判定；其 init 已固定（torch.manual_seed）"
                             if mlp_obs else None),
        "full_mlp": ({f"{s}": st for s, st in mlp_obs.items()} if mlp_obs else None),
        **v,
        "per_seed_table": table.relative_to(PROJECT_ROOT).as_posix(),
    }
    path = REPORTS_DIR / f"context_ablation_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n表:   {table.relative_to(PROJECT_ROOT).as_posix()}\n"
          f"报告: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
