#!/usr/bin/env python
"""A7 高质量人脸子集评估：图像质量是否限制了 FER 的预测信号（plan.md §12 A7）。

预注册判读标准（跑前写定）：results/reports/a7_quality_criteria_preregistered.txt
假设 H9 = H9a ∧ H9b（见该文件第 4 节）。

核心问题与设计要点
------------------
plan §12 A7 的原话是「在高质量人脸样本上重复主要比较」，但**直接拿子集的 pooled_rank 与
全量的 0.467 相减是无效的**：两个数字建立在不同的样本、以及不同的零分布上（样本从 385 降到
333 后检测下限更差）。因此本脚本设两个 regime：

    A7-high : 场级完整的高质量子集（某场全部有效踢次的 crop 短边 >= 阈值，整场进入/退出）
    A7-ctrl : 随机抽等量样本，重复 n_ctrl 次 —— 隔离「样本量/构成」与「图像质量」

判定：
    H9a = M1 在 A7-high 上对其自身 200 次全流程置换零分布 p < 0.05
    H9b = M1 在 A7-high 上落在 A7-ctrl 随机子集分布的上尾 p95 之外
    H9 成立当且仅当两者同时通过。

口径选择（用户 2026-11-10 选定，对 plan §0.5「样本级 ≥80 → 375/385」的偏离）：
    场级完整 —— 质量按比赛聚簇，样本级剔除会改变保留场次的内部构成，使子集与全量在被测
    场的构成上不可比。实测阈值 80px 保留 35 场 / 333 样本。

臂：M1（主）/ M3v2 / B1v2（方法学阴性对照），全部 LR 冻结特征，C 沿用主表定值不重调。

用法（项目根目录，需 PYTHONIOENCODING=utf-8）:
    python scripts/eval_a7_quality.py --profile        # 只打质量画像，不训练
    python scripts/eval_a7_quality.py --smoke          # 1 seed × 1 折 × 少量置换，量级检查
    python scripts/eval_a7_quality.py --n-perm 0       # 只算观测值（10 seed × 5 折，秒级）
    python scripts/eval_a7_quality.py                  # 全量：观测 + 200 置换 + 200 控制子集

输出: results/tables/a7_quality_<stamp>.csv、results/reports/a7_quality_<stamp>.json
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_repeated_splits as ers  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "a7_quality.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
CRITERIA_PATH = REPORTS_DIR / "a7_quality_criteria_preregistered.txt"


def load_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_seeds(s: str) -> list[int]:
    return ers.parse_seeds(s)


# ---------------------------------------------------------------- 质量画像

def crop_short_sides(valid: pd.DataFrame) -> np.ndarray:
    """存盘 crop 的短边像素（PIL 现算，不依赖外部表）。缺失文件直接报错。"""
    from PIL import Image

    out = []
    missing = []
    for p in valid["face_path"]:
        fp = PROJECT_ROOT / str(p)
        if not fp.exists():
            missing.append(str(p))
            continue
        w, h = Image.open(fp).size
        out.append(min(w, h))
    if missing:
        raise FileNotFoundError(f"{len(missing)} 张人脸 crop 缺失，如 {missing[0]}")
    return np.asarray(out, dtype=np.float64)


def face_w_px_by_event(valid: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """从 face_frames.csv 取 YuNet 原始检测宽度（覆盖不全，缺的样本不参与场级聚合）。"""
    q = cfg["quality_proxies"]["face_w_px"]
    ff = pd.read_csv(PROJECT_ROOT / q["csv"], encoding="utf-8")
    ff["event_id"] = ff["match_id"].astype(str) + "-" + ff["kick_seq"].astype(str)
    m = ff.dropna(subset=[q["column"]]).groupby("event_id")[q["column"]].median()
    return {e: float(v) for e, v in m.items() if e in set(valid["event_id"])}


# ---------------------------------------------------------------- 子集

def choose_matches_crop(valid: pd.DataFrame, short: np.ndarray, cfg: dict) -> tuple[set, dict]:
    """主口径：场级 min(短边) >= 阈值 → 该场整体进入。

    返回 (保留场集合, 删除信息)。保留集是 **match_id**；与 event_id 的换算见
    keep_events_of（子集过滤必须按 event_id 做，因为 folds 暴露的是位置索引）。
    """
    thr = float(cfg["subset"]["min_short_px"])
    df = valid.assign(_short=short)
    mq = df.groupby("match_id")["_short"].min()
    keep = set(mq[mq >= thr].index)
    drop = sorted(set(mq.index) - keep)
    stats = {m: float(mq[m]) for m in drop}
    return keep, {"threshold_px": thr, "dropped_matches": drop,
                  "dropped_match_min_short": stats,
                  "n_keep_matches": len(keep),
                  "n_drop_matches": len(drop)}


def keep_events_of(valid: pd.DataFrame, keep_matches: set) -> set:
    """把「保留场集合」换算成「保留 event_id 集合」（子集过滤统一用 event_id）。"""
    return set(valid.loc[valid["match_id"].isin(keep_matches), "event_id"])


def choose_matches_facew(valid: pd.DataFrame, cfg: dict) -> tuple[set, dict]:
    """次口径：场级中位 face_w_px 最低的 n 场退出（等量删除，保证两代理样本量一致）。"""
    q = cfg["quality_proxies"]["face_w_px"]
    n_drop = int(q["n_drop_matches"])
    fw = face_w_px_by_event(valid, cfg)
    df = valid.assign(_fw=valid["event_id"].map(fw))
    mq = df.groupby("match_id")["_fw"].median().dropna().sort_values()
    drop = sorted(mq.index[:n_drop])
    keep = set(mq.index) - set(drop)
    return keep, {"n_drop_matches": n_drop, "dropped_matches": drop,
                  "dropped_match_median_face_w": {m: float(mq[m]) for m in drop},
                  "coverage_events": len(fw)}


def subset_stats(valid: pd.DataFrame, keep_events: set) -> dict:
    s = valid[valid["event_id"].isin(keep_events)]
    y = (s["result"] == "goal").astype(int).to_numpy()
    return {"n_matches": int(s["match_id"].nunique()), "n_events": int(len(s)),
            "n_goal": int(y.sum()), "n_miss": int((1 - y).sum()),
            "goal_rate": float(y.mean())}


# ---------------------------------------------------------------- 单臂

def run_arm_subset(data: dict, arm: dict, folds: list[dict], y: np.ndarray,
                   keep: set, seed: int) -> list[tuple]:
    """在 folds 原样保留的前提下，把每折的 train/val/test 收缩到 keep 子集再训 LR。

    folds **不重建**（硬约束 4 的划分与主表一致）；标准化与 p̄ 只由该折 train 的子集部分
    估计（硬约束 5，由 ers.design_matrix / fit_predict 内部保证）。
    """
    v = data["valid"]
    mask = v["event_id"].isin(keep).to_numpy()
    out = []
    for fold in folds:
        f = {k: v2[mask[v2]] for k, v2 in fold.items()}
        if len(f["test"]) == 0 or len(f["train"]) == 0:
            continue
        if len(np.unique(y[f["train"]])) < 2:      # 置换下可能单类
            continue
        out.append((f["test"], y[f["test"]], ers.fit_predict(data, arm, f, y, seed)))
    return out


def random_subset(data: dict, size: int, rng: np.random.Generator) -> set:
    """随机抽 size 个 event（不按场抽，与 A7-high 的场级剔法形成对照）。"""
    ids = np.asarray(data["ids"])
    return set(rng.choice(ids, size=size, replace=False).tolist())


# ---------------------------------------------------------------- 主流程

def profile_only(data: dict, cfg: dict) -> int:
    valid = data["valid"]
    short = crop_short_sides(valid)
    print("=" * 74)
    print("A7 质量画像（描述性，不判定）")
    print("=" * 74)
    print(f"crop 短边: min {short.min():.0f}  p25 {np.percentile(short, 25):.0f}  "
          f"中位 {np.median(short):.0f}  p75 {np.percentile(short, 75):.0f}  max {short.max():.0f}")
    for t in (60, 80, 96, 120, 150, 200):
        k = int((short >= t).sum())
        print(f"  短边 >= {t:3d}: {k:3d} 样本 ({100 * k / len(short):.1f}%)")
    keep_m, info = choose_matches_crop(valid, short, cfg)
    keep = keep_events_of(valid, keep_m)
    st = subset_stats(valid, keep)
    print(f"\n[主口径 crop_short] 删除场 {info['dropped_matches']}")
    print(f"  保留 {st['n_matches']} 场 / {st['n_events']} 样本  "
          f"(goal {st['n_goal']} / miss {st['n_miss']}, goal 率 {st['goal_rate']:.3f})")
    keep_m2, info2 = choose_matches_facew(valid, cfg)
    keep2 = keep_events_of(valid, keep_m2)
    st2 = subset_stats(valid, keep2)
    print(f"[次口径 face_w_px] 删除场 {info2['dropped_matches']}  (覆盖 {info2['coverage_events']} 事件)")
    print(f"  保留 {st2['n_matches']} 场 / {st2['n_events']} 样本  "
          f"(goal {st2['n_goal']} / miss {st2['n_miss']}, goal 率 {st2['goal_rate']:.3f})")
    print(f"  两口径删除场交集: {sorted(set(info['dropped_matches']) & set(info2['dropped_matches']))}")
    y = data["y"]
    from scipy.stats import pointbiserialr
    r, p = pointbiserialr(y, short)
    print(f"\ncrop 短边 vs result: r={r:+.4f} p={p:.3f}（与标签无关才使质量不成为中介变量）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--n-ctrl", type=int, default=None)
    ap.add_argument("--proxy", default=None, choices=["crop_short", "face_w_px", "both"])
    ap.add_argument("--profile", action="store_true", help="只打质量画像，不训练")
    ap.add_argument("--smoke", action="store_true", help="1 seed × 少量置换，量级检查（不落盘）")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cfg = load_cfg()
    data = ers.load_data()
    y = data["y"]

    if args.profile:
        return profile_only(data, cfg)

    seeds = parse_seeds(args.seeds or cfg["eval"]["seeds"])
    n_perm = int(cfg["eval"]["n_perm"] if args.n_perm is None else args.n_perm)
    n_ctrl = int(cfg["eval"]["n_ctrl"] if args.n_ctrl is None else args.n_ctrl)
    test_seed = int(cfg["eval"]["test_seed"])
    proxies = ["crop_short", "face_w_px"] if (args.proxy in (None, "both")) else [args.proxy]
    arms = cfg["arms"]
    smoke = bool(args.smoke)
    if smoke:
        seeds, n_perm, n_ctrl = seeds[:1], min(n_perm, 2), min(n_ctrl, 2)

    valid = data["valid"]
    short = crop_short_sides(valid)
    keep_by_proxy: dict[str, tuple[set, dict]] = {}
    for px in proxies:
        km, info = (choose_matches_crop(valid, short, cfg) if px == "crop_short"
                    else choose_matches_facew(valid, cfg))
        keep_by_proxy[px] = (keep_events_of(valid, km), info)

    print(f"seeds={seeds}  n_perm={n_perm}  n_ctrl={n_ctrl}  proxies={proxies}"
          + ("  [smoke]" if smoke else ""))

    report: dict = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "purpose": "A7 高质量人脸子集：图像质量是否限制 FER 信号（H9a ∧ H9b）",
        "criteria": CRITERIA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config": cfg,
        "n_valid_events": int(len(y)),
        "proxies": {}, "guards": {},
    }

    # ---- 折可用性断言：子集下每折 val/test 必须两类俱全 ----
    # 注意：ers.make_folds 返回的折是**位置索引**（data['ids'] 的下标），不是 event_id，
    # 因此必须用 data['ids'][i] 反查，不能直接拿 valid['event_id'].iloc[i]（那是全表下标）。
    guard_rows = []
    ids_arr = np.asarray(data["ids"])
    id_to_keep = {px: set(k) for px, (k, _) in keep_by_proxy.items()}
    for px, (keep, info) in keep_by_proxy.items():
        report["proxies"][px] = {"drop_info": info, "subset": subset_stats(valid, keep)}
        for s in seeds:
            built = ers.make_folds(data, y, s)
            if built is None:
                raise SystemExit(f"seed {s}: 真实标签下划分缺类")
            _, folds = built
            for fi, fold in enumerate(folds):
                for sp in ("val", "test"):
                    idx = [i for i in fold[sp] if ids_arr[i] in keep]
                    vals = [int(y[i]) for i in idx]
                    if not vals or len(set(vals)) < 2:
                        raise AssertionError(
                            f"{px} seed {s} fold {fi} 的 {sp} 在子集下单类"
                            f"（n={len(vals)}）—— 子集不可用，中止。")
        guard_rows.append({"proxy": px, "ok": True, "seeds_checked": len(seeds)})
    report["guards"]["fold_class_balance"] = guard_rows

    # ---- 观测值 ----
    obs: dict = {}
    for px, (keep, _) in keep_by_proxy.items():
        obs[px] = {}
        for s in seeds:
            _, folds = ers.make_folds(data, y, s)
            rec = {}
            for arm in arms:
                oof = run_arm_subset(data, arm, folds, y, keep, s)
                rec[arm["name"]] = ers.full_stats(oof)
            obs[px][s] = rec
            if s == seeds[0] or s == test_seed:
                print(f"  [{px}] seed {s}: " + "  ".join(
                    f"{a['name']} {rec[a['name']]['pooled_rank']:.4f}" for a in arms))

    # ---- A7-high 专用置换零分布（主检验 seed） ----
    _, folds_test = ers.make_folds(data, y, test_seed)
    perm_records: dict = {px: [] for px in proxies}
    if n_perm > 0:
        t0 = time.time()
        rng_base = np.random.default_rng(int(cfg["eval"]["perm_seed"]))
        perm_seeds = rng_base.integers(0, 2**31 - 1, size=n_perm)
        for i, ps in enumerate(perm_seeds):
            rng = np.random.default_rng(int(ps))
            yp = rng.permutation(y)
            built = ers.make_folds(data, yp, test_seed)
            if built is None:
                continue
            _, folds_p = built
            for px, (keep, _) in keep_by_proxy.items():
                rec = {}
                for arm in arms:
                    oof = run_arm_subset(data, arm, folds_p, yp, keep, test_seed)
                    rec[arm["name"]] = ers.pooled_rank(oof) if oof else float("nan")
                perm_records[px].append(rec)
            if (i + 1) % 50 == 0:
                print(f"  perm {i + 1}/{n_perm}  ({time.time() - t0:.0f}s)")
        print(f"  置换完成 ({time.time() - t0:.0f}s)")

    # ---- A7-ctrl：随机等量样本子集（不置换标签） ----
    ctrl_records: dict = {px: [] for px in proxies}
    if n_ctrl > 0:
        t0 = time.time()
        rng = np.random.default_rng(int(cfg["eval"]["ctrl_seed"]))
        for i in range(n_ctrl):
            for px, (keep, _) in keep_by_proxy.items():
                size = len(keep)
                rs = random_subset(data, size, rng)
                rec = {}
                for arm in arms:
                    oof = run_arm_subset(data, arm, folds_test, y, rs, test_seed)
                    rec[arm["name"]] = ers.pooled_rank(oof) if oof else float("nan")
                ctrl_records[px].append(rec)
            if (i + 1) % 50 == 0:
                print(f"  ctrl {i + 1}/{n_ctrl}  ({time.time() - t0:.0f}s)")
        print(f"  控制子集完成 ({time.time() - t0:.0f}s)")

    # ---- 判读 ----
    alpha = float(cfg["判定"]["alpha"])
    pct = float(cfg["判定"]["h9b_pct"])
    verdict: dict = {}
    for px in proxies:
        primary = next(a["name"] for a in arms if a.get("primary"))
        obs_pr = obs[px][test_seed][primary]["pooled_rank"]
        entry: dict = {"primary_arm": primary, "obs_pooled_rank_test_seed": obs_pr,
                       "obs_pooled_rank_mean_10seed": float(np.mean(
                           [obs[px][s][primary]["pooled_rank"] for s in seeds])),
                       "obs_pooled_rank_std_10seed": float(np.std(
                           [obs[px][s][primary]["pooled_rank"] for s in seeds]))}
        if perm_records[px]:
            null = np.array([r[primary] for r in perm_records[px] if np.isfinite(r[primary])])
            p_a = ers.p_right(null, obs_pr)
            entry.update({"H9a_null_mean": float(null.mean()),
                          "H9a_null_p95": float(np.percentile(null, 95)),
                          "H9a_p": float(p_a), "H9a_pass": bool(p_a < alpha),
                          "n_perm_effective": int(len(null))})
        else:
            entry.update({"H9a_p": None, "H9a_pass": None})
        if ctrl_records[px]:
            ctrl = np.array([r[primary] for r in ctrl_records[px] if np.isfinite(r[primary])])
            thr = float(np.percentile(ctrl, pct))
            entry.update({"H9b_ctrl_mean": float(ctrl.mean()), "H9b_ctrl_std": float(ctrl.std()),
                          "H9b_ctrl_p": float(np.mean(ctrl <= obs_pr)),
                          "H9b_threshold_p95": thr,
                          "H9b_pass": bool(obs_pr > thr), "n_ctrl_effective": int(len(ctrl))})
        else:
            entry.update({"H9b_pass": None})
        entry["secondary_arms"] = {a["name"]: {
            "obs_test_seed": obs[px][test_seed][a["name"]]["pooled_rank"],
            "obs_mean_10seed": float(np.mean([obs[px][s][a["name"]]["pooled_rank"] for s in seeds])),
            "ctrl_mean": float(np.mean([r[a["name"]] for r in ctrl_records[px]])) if ctrl_records[px] else None,
        } for a in arms}
        if entry.get("H9a_pass") is not None and entry.get("H9b_pass") is not None:
            entry["H9_holds"] = bool(entry["H9a_pass"] and entry["H9b_pass"])
        verdict[px] = entry
    report["verdict"] = verdict

    # ---- 落盘 ----
    if smoke:
        print("\n[smoke] 不落盘，结束。")
        for px, v in verdict.items():
            print(f"  {px}: obs={v['obs_pooled_rank_test_seed']:.4f} "
                  f"p={v.get('H9a_p')} ctrl_p95={v.get('H9b_threshold_p95')}")
        return 0

    stamp = report["created_at"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for px in proxies:
        for s in seeds:
            for arm in arms:
                st = obs[px][s][arm["name"]]
                rows.append({"proxy": px, "arm": arm["name"], "split_seed": s, "regime": "A7-high",
                             **{k: v for k, v in st.items() if not isinstance(v, dict)}})
    for px in proxies:
        for i, r in enumerate(perm_records[px]):
            for k, v in r.items():
                rows.append({"proxy": px, "arm": k, "split_seed": test_seed, "regime": "A7-high-perm",
                             "iter": i, "pooled_rank": v})
        for i, r in enumerate(ctrl_records[px]):
            for k, v in r.items():
                rows.append({"proxy": px, "arm": k, "split_seed": test_seed, "regime": "A7-ctrl",
                             "iter": i, "pooled_rank": v})
    pd.DataFrame(rows).to_csv(TABLES_DIR / f"a7_quality_{stamp}.csv", index=False, encoding="utf-8")
    report["limitations"] = [
        "场级完整口径删掉 5 整场（52 样本），子集与全量不属于同一评估目标；所有 A7 数字只在子集内部比较，不得与全量的 0.467 / 0.553 相减。",
        "333 样本 / miss 90 下检测下限比全量更差（零分布 p95 见 H9a_null_p95），阴性结论只能写成『在该子集的检测下限内未显示质量效应』。",
        "face_w_px 只覆盖 340/385，次口径的场级中位基于部分样本。",
        "crop 短边由人工截帧决定，不完全等于检测质量；两代理一致性是主要缓释手段。",
        "无超参 sweep：C 沿用主表定值，不为子集重调（否则引入子集上的选择偏差）。",
        "B1v2 为方法学阴性对照（context 与图像质量无关）；若它也在 A7-high 上抬升，说明效应来自子集构成。",
    ]
    with open(REPORTS_DIR / f"a7_quality_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # ---- 控制台汇总 ----
    print("\n" + "=" * 76)
    print("A7 判读（主臂 = M1；H9 成立当且仅当 H9a ∧ H9b）")
    print("=" * 76)
    for px, v in verdict.items():
        print(f"\n[{px}]  子集 {report['proxies'][px]['subset']['n_matches']} 场 / "
              f"{report['proxies'][px]['subset']['n_events']} 样本")
        print(f"  观测: 10-seed {v['obs_pooled_rank_mean_10seed']:.4f} ± "
              f"{v['obs_pooled_rank_std_10seed']:.4f}   检验 seed {test_seed}: "
              f"{v['obs_pooled_rank_test_seed']:.4f}")
        if v.get("H9a_p") is not None:
            print(f"  H9a: 置换零分布 mean {v['H9a_null_mean']:.4f} / p95 {v['H9a_null_p95']:.4f} "
                  f"→ p={v['H9a_p']:.4f}  {'通过' if v['H9a_pass'] else '未通过'}")
        if v.get("H9b_pass") is not None:
            print(f"  H9b: 随机子集分布 mean {v['H9b_ctrl_mean']:.4f} ± {v['H9b_ctrl_std']:.4f} "
                  f"/ p95 {v['H9b_threshold_p95']:.4f} → 百分位 {v['H9b_ctrl_p']:.3f}  "
                  f"{'通过' if v['H9b_pass'] else '未通过'}")
        if v.get("H9_holds") is not None:
            print(f"  → H9 {'成立' if v['H9_holds'] else '不支持'}")
        for nm, s2 in v["secondary_arms"].items():
            if nm == v["primary_arm"]:
                continue
            print(f"    次要臂 {nm}: 子集 {s2['obs_mean_10seed']:.4f}  "
                  f"vs 随机子集 {s2['ctrl_mean']:.4f}")
    print(f"\n落盘: results/tables/a7_quality_{stamp}.csv")
    print(f"      results/reports/a7_quality_{stamp}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
