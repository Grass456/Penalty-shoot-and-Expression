#!/usr/bin/env python
"""VA-MTL 臂特征缓存：提取 valence-arousal 多任务 backbone 的 embedding + VA 值。

动机（2026-11-02，接 fer-backbone-compare 的结论）：
    8 类离散情绪空间与点球场景错配——主罚前表情以"紧张"为主，而 8 类空间无
    对应类别（55% 样本被判 Anger 是模型对紧绷表情的先验偏置）。valence-arousal
    是连续维度，"紧张" = 高 arousal + 负 valence，无需微调即可检验该假设。

    预言（假设 1 的可检验形式）：miss 前的帧应呈高 arousal / 低 valence，
    使 VA 二维臂的 AUROC 高于 chance，且方向与"紧张导致 miss"一致。

臂定义（3 个 MTL backbone，均为 emotiefflib 现成权重）:
    enet_b0_8_va_mtl    1280 维 embedding, 224px, 8 类 + VA
    mbf_va_mtl           512 维, 112px, 0.5/0.5 归一化
    mobilevit_va_mtl     768 维, 224px, 8 类 + VA

输出（与 cache_fer_features.py 同构，分目录落盘）:
    features/fer/<model>/embeddings.npz
        event_ids / embedding / emo_probs(8) / va (N,2) [valence, arousal] / meta
    features/fer/<model>/cache_report.json

用法（项目根目录）:
    python scripts/cache_va_features.py --model enet_b0_8_va_mtl [--force]

复用 cache_fer_features.py 的 load_events / make_preprocess（同一预处理路径）。
VA 顺序: classifier_weights 的最后两维 = (valence, arousal)，tanh 不改变排序
统计量，直接用线性头输出；报告里同时落 tanh 后的值便于人读。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_fer_features import EMO_NAMES_8, FEATURES_DIR, load_events, make_preprocess  # noqa: E402

VA_MODELS = {
    "enet_b0_8_va_mtl": {"emb_dim": 1280, "note": "enet_b0 + VA 多任务头"},
    "mbf_va_mtl": {"emb_dim": 512, "note": "MobileFaceNet + VA, 112px, mean/std=0.5"},
    "mobilevit_va_mtl": {"emb_dim": 768, "note": "MobileViT + VA"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="enet_b0_8_va_mtl", choices=sorted(VA_MODELS))
    ap.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = FEATURES_DIR / args.model
    emb_path = out_dir / "embeddings.npz"
    if emb_path.exists() and not args.force:
        print(f"缓存已存在: {emb_path}（--force 重算）")
        return 0

    import torch
    from PIL import Image
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    rows = load_events()
    face_paths = [PROJECT_ROOT / r["face_path"] for r in rows]
    missing = [str(p) for p in face_paths if not p.exists()]
    if missing:
        print(f"缺 {len(missing)} 张脸图，如 {missing[0]}", file=sys.stderr)
        return 1

    torch.use_deterministic_algorithms(True, warn_only=True)

    fer = EmotiEffLibRecognizer(engine="torch", model_name=args.model, device=args.device)
    emo_weights = np.asarray(fer.classifier_weights, dtype=np.float32)  # (10, D): 8 情绪 + VA
    emo_bias = np.asarray(fer.classifier_bias, dtype=np.float32)
    n_total, emb_dim = emo_weights.shape
    assert n_total == 10, f"MTL 头应为 10 维输出（8 情绪 + 2 VA），实际 {n_total}"
    va_w, va_b = emo_weights[-2:], emo_bias[-2:]      # 最后一维排列: (valence, arousal)
    e_w, e_b = emo_weights[:8], emo_bias[:8]
    # 三个 MTL backbone 去头方式不同（库已处理）：mbf 用 fc、mobilevit 用 head.fc、
    # 其余用 classifier；Identity 替换后 fer.model(x) 即 embedding（同 enet_b0 系）
    emb_model = fer.model
    emb_model.eval()

    prep = (int(fer.img_size), tuple(float(v) for v in fer.mean), tuple(float(v) for v in fer.std))
    preprocess = make_preprocess(*prep)

    print(f"arm={args.model} | device={args.device} | emb_dim={emb_dim} | img_size={prep[0]} | n={len(rows)}")
    embeddings = np.zeros((len(rows), emb_dim), dtype=np.float32)
    emo_probs = np.zeros((len(rows), 8), dtype=np.float32)
    va = np.zeros((len(rows), 2), dtype=np.float32)  # [valence, arousal]（线性头原始输出）

    with torch.no_grad():
        for i, (r, p) in enumerate(zip(rows, face_paths)):
            img = np.asarray(Image.open(p).convert("RGB"))
            x = preprocess(img).unsqueeze(0).to(args.device)
            z = emb_model(x)
            if z.dim() > 2:
                z = z.flatten(1)
            embeddings[i] = z.squeeze(0).float().cpu().numpy()
            tw = torch.as_tensor(e_w, dtype=z.dtype, device=z.device)
            tb = torch.as_tensor(e_b, dtype=z.dtype, device=z.device)
            emo_probs[i] = torch.softmax(z @ tw.T + tb, dim=1).squeeze(0).float().cpu().numpy()
            va[i] = (z @ torch.as_tensor(va_w, dtype=z.dtype, device=z.device).T
                     + torch.as_tensor(va_b, dtype=z.dtype, device=z.device)).squeeze(0).float().cpu().numpy()
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                print(f"  {i + 1}/{len(rows)}")

    assert np.isfinite(embeddings).all() and np.isfinite(va).all()
    assert np.allclose(emo_probs.sum(axis=1), 1.0, atol=1e-4)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "arm": args.model,
        "model_name": args.model,
        "engine": "torch",
        "img_size": int(prep[0]),
        "embedding_dim": emb_dim,
        "n_emotion_classes": 8,
        "has_va": True,
        "va_order": ["valence", "arousal"],
        "va_raw": "linear head output（未过 tanh；单调不改变排序统计量）",
        "normalization_mean": [float(v) for v in prep[1]],
        "normalization_std": [float(v) for v in prep[2]],
        "built_date": date.today().isoformat(),
        "n_samples": len(rows),
        "event_ids_source": "events.csv (face_path 非空且未排除)",
        "note": VA_MODELS[args.model]["note"],
    }
    np.savez_compressed(emb_path, event_ids=np.array([r["event_id"] for r in rows]),
                        embedding=embeddings, emo_probs=emo_probs, va=va,
                        meta=np.array(json.dumps(meta)))
    report = {**meta,
              "mean_max_prob": float(emo_probs.max(axis=1).mean()),
              "neutral_share_argmax": float((emo_probs.argmax(axis=1) == EMO_NAMES_8.index("Neutral")).mean()),
              "va_mean": [round(float(va[:, 0].mean()), 4), round(float(va[:, 1].mean()), 4)],
              "va_std": [round(float(va[:, 0].std()), 4), round(float(va[:, 1].std()), 4)],
              "valence_range": [round(float(va[:, 0].min()), 3), round(float(va[:, 0].max()), 3)],
              "arousal_range": [round(float(va[:, 1].min()), 3), round(float(va[:, 1].max()), 3)],
              "va_corr": round(float(np.corrcoef(va[:, 0], va[:, 1])[0, 1]), 4)}
    (out_dir / "cache_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"缓存完成: {emb_path.relative_to(PROJECT_ROOT)}")
    print(f"  {len(rows)} 样本 x {emb_dim} 维 + VA(2) | 报告: {(out_dir / 'cache_report.json').relative_to(PROJECT_ROOT)}")
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
