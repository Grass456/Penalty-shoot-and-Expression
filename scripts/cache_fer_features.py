#!/usr/bin/env python
"""FER 特征缓存脚本：对 events.csv 的每条可建模样本提取 embedding + 情绪概率，落盘缓存。

设计（CLAUDE.md「数据 pipeline 与模型解耦」）:
    下游分类器（Context-only / FER-only / 融合）一律读本脚本产出，不再过 FER 前向。
    每个样本缓存三类输出（plan §7.2）:
        embedding    z_i = f_FER(x_i)                (1280 维, enet_b0)
        emo_probs    e_i = softmax(W_e z_i + b_e)    (8 维)
        emo_label    argmax(e_i)                     (字符串, 便于审计)

用法（项目根目录）:
    python scripts/cache_fer_features.py                    # 全量
    python scripts/cache_fer_features.py --model enet_b2_8  # 换 backbone（分目录存）
    python scripts/cache_fer_features.py --force            # 忽略已有缓存重算

输入:
    data/metadata/events.csv        只取 face_path 非空且未排除的行（exclusion_reason 为空）
    data/faces/<match>/kNN.jpg      人脸 crop（capture_face.py 产出）

输出（gitignored）:
    features/fer/<backbone>/embeddings.npz
        event_ids : (N,) str 数组 —— 行序与 events.csv 过滤后一致，是下游 join 键
        embedding : (N, D) float32
        emo_probs : (N, 8) float32
        meta      : JSON 字符串（model_name / img_size / normalization / 构建信息）
    features/fer/<backbone>/cache_report.json     提取统计（复现/审计用）

确定性: 固定 torch.use_deterministic_algorithms + 单进程顺序前向（114 张图秒级完成，
无需并行）。ImageNet 对照臂用 --imagenet efficientnet_b0 走同一流程，输出分目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
FACES_DIR = PROJECT_ROOT / "data" / "faces"
FEATURES_DIR = PROJECT_ROOT / "features" / "fer"
YUNET_PATH = PROJECT_ROOT / "models" / "detection" / "face_detection_yunet_2023mar.onnx"

VALID_MODEL_NAMES = {"enet_b0_8_best_vgaf", "enet_b0_8_best_afew", "enet_b2_8", "enet_b2_7"}
IMAGENET_CONTROL = "efficientnet_b0"  # 与 enet_b0_* 同构（H3 对照臂，plan §7.5）


def load_events() -> list[dict]:
    """只取可建模样本: face_path 非空 + 未排除。"""
    with open(EVENTS_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["face_path"].strip() and not r["exclusion_reason"].strip()]
    if not rows:
        print("events.csv 无可建模样本（face_path 空/已排除）", file=sys.stderr)
        raise SystemExit(1)
    return rows


def build_fer(model_name: str, device: str):
    """加载 EmotiEffLib 并取 backbone 引用。

    emotiefflib 1.1.1 的 torch 引擎加载时已把 model.classifier 换成 nn.Identity
    （notebook 验证过），fer.model(x) 直接输出 (N, D) embedding；EmotiEffLibRecognizer
    本身不是 Module（不可调用），前向必须走 fer.model。classifier_weights=(8, D)
    是随包情绪头权重，用于手算情绪概率，避免第二次前向。
    """
    import torch
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    fer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)
    emo_weights = fer.classifier_weights  # (8, D) —— 情绪头权重，保留用于算概率
    emo_bias = np.asarray(getattr(fer, "classifier_bias", np.zeros(emo_weights.shape[0])),
                          dtype=np.float32)  # 库的 _get_probab 含 bias，必须带上方能对齐
    n_classes, emb_dim = emo_weights.shape
    assert isinstance(fer.model.classifier, torch.nn.Identity), \
        f"意外: classifier = {fer.model.classifier}（预期 1.1.1 已 Identity 化）"
    fer.model.eval()
    return fer.model, (emo_weights, emo_bias), emb_dim, n_classes


def build_imagenet(device: str):
    """ImageNet 对照臂: timm efficientnet_b0 去分类头，输出与 enet_b0 同为 1280 维。"""
    import timm
    import torch

    model = timm.create_model(IMAGENET_CONTROL, pretrained=True, num_classes=0)
    model.eval().to(device)
    return model, None, 1280, 0


def make_preprocess(fer_like):
    """显式预处理（RGB + resize + mean/std），不依赖库私有 API。

    notebook 验证过: emotiefflib torch 引擎按 RGB 解释数组（PIL 路径），
    这里保持同一数值路径。fer_like 只需提供 img_size / mean / std 三个属性。
    """
    from torchvision import transforms

    size = (fer_like.img_size, fer_like.img_size) if isinstance(fer_like.img_size, int) \
        else tuple(fer_like.img_size)
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([float(v) for v in fer_like.mean], [float(v) for v in fer_like.std]),
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="enet_b0_8_best_vgaf", choices=sorted(VALID_MODEL_NAMES),
                    help="EmotiEffLib backbone（默认与 notebook 验证一致）")
    ap.add_argument("--imagenet", action="store_true",
                    help="提取 ImageNet 对照臂特征（--model 被忽略，输出到 imagenet_b0/）")
    ap.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="忽略已有缓存重算")
    args = ap.parse_args()

    arm = "imagenet_b0" if args.imagenet else args.model
    out_dir = FEATURES_DIR / arm
    emb_path = out_dir / "embeddings.npz"
    if emb_path.exists() and not args.force:
        print(f"缓存已存在: {emb_path}（--force 重算）")
        return 0

    rows = load_events()
    face_paths = [PROJECT_ROOT / r["face_path"] for r in rows]
    missing = [str(p) for p in face_paths if not p.exists()]
    if missing:
        print(f"缺 {len(missing)} 张脸图，如 {missing[0]}", file=sys.stderr)
        return 1

    import torch
    from PIL import Image

    torch.use_deterministic_algorithms(True, warn_only=True)

    # 归一化参数：两臂统一用 ImageNet 统计量（emotiefflib 权重即以该统计量训练/对齐，
    # notebook 已验证同路径数值），FER 臂与 ImageNet 臂预处理完全一致才满足 H3 的公平对照
    class _Norm:
        img_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    preprocess = make_preprocess(_Norm)

    if args.imagenet:
        model, emo_head, emb_dim, n_classes = build_imagenet(args.device)
    else:
        model, emo_head, emb_dim, n_classes = build_fer(args.model, args.device)

    print(f"arm={arm} | device={args.device} | emb_dim={emb_dim} | n={len(rows)}")
    embeddings = np.zeros((len(rows), emb_dim), dtype=np.float32)
    emo_probs = np.zeros((len(rows), n_classes), dtype=np.float32) if n_classes else None

    # 顺序前向（114 张图 < 10s GPU / ~1min CPU，无需 batch 并行的复杂度）
    with torch.no_grad():
        for i, (r, p) in enumerate(zip(rows, face_paths)):
            img = np.asarray(Image.open(p).convert("RGB"))
            x = preprocess(img).unsqueeze(0).to(args.device)
            z = model(x)
            if z.dim() > 2:  # 防御: 某些 timm 模型返回 (N, C, 1, 1)
                z = z.flatten(1)
            embeddings[i] = z.squeeze(0).float().cpu().numpy()
            if emo_head is not None:
                w, b = emo_head
                w = torch.as_tensor(w, dtype=z.dtype, device=z.device)
                b = torch.as_tensor(b, dtype=z.dtype, device=z.device)
                logits = z @ w.T + b
                emo_probs[i] = torch.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"  {i + 1}/{len(rows)}")

    # 一致性自检（notebook 已验证的量，这里作为落盘前的最后防线）
    assert embeddings.shape == (len(rows), emb_dim)
    assert np.isfinite(embeddings).all(), "embedding 含 NaN/Inf"
    if emo_probs is not None:
        assert np.allclose(emo_probs.sum(axis=1), 1.0, atol=1e-4), "情绪概率行和偏离 1"

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "arm": arm,
        "model_name": IMAGENET_CONTROL if args.imagenet else args.model,
        "engine": "torch",
        "img_size": 224,
        "embedding_dim": emb_dim,
        "n_emotion_classes": n_classes,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "built_date": date.today().isoformat(),
        "n_samples": len(rows),
        "event_ids_source": "events.csv (face_path 非空且未排除)",
    }
    np.savez_compressed(emb_path, event_ids=np.array([r["event_id"] for r in rows]),
                        embedding=embeddings,
                        emo_probs=emo_probs if emo_probs is not None else np.zeros((len(rows), 0), np.float32),
                        meta=np.array(json.dumps(meta)))
    report = {**meta, "mean_max_prob": float(emo_probs.max(axis=1).mean()) if emo_probs is not None and n_classes else None,
              "neutral_share_argmax": float((emo_probs.argmax(axis=1) == list(
                  ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]).index("Neutral")
              ).mean()) if emo_probs is not None and n_classes else None}
    (out_dir / "cache_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"缓存完成: {emb_path.relative_to(PROJECT_ROOT)}")
    print(f"  {len(rows)} 样本 x {emb_dim} 维 | 报告: {(out_dir / 'cache_report.json').relative_to(PROJECT_ROOT)}")
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
