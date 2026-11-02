#!/usr/bin/env python
"""多帧聚合特征缓存：特写窗口内逐帧提取球员人脸 embedding，聚合后落盘。

动机（2026-11-02，用户方向 2）：单帧协议的测量噪声——人工截帧在导播切特写的
瞬间完成，实测选帧距触球提前量中位 9.3s（>15s 的占 84/385），一半样本取自
走位阶段而非主罚前的紧绷时刻。多帧聚合 = 对窗口内全部合格球员帧的 embedding
取均值，压测量噪声。**聚合 embedding，不做 8 类 argmax 投票**（标签层最噪）。

窗口与身份验证（与用户确认的设计）:
    窗口   = [face_closeup_s, contact_time_s - 0.5s]（marker 人工标注的特写
             起点 → 触球前护栏；硬约束 1：全部帧严格早于触球前 0.5s）
    采样   = 2fps 均匀（窗口中位 9.5s → 中位 ~18 帧）
    身份   = SFace 余弦相似度 vs 该踢次人工验证锚（face_path 的 crop），
             阈值 0.363（OpenCV 官方默认）；同帧多脸取过阈值的最高相似者
    锚缺失 = 12 个 FULLFRAME 行退化为"窗口内最大脸"并标记 anchor_missing
             （方案 A，用户确认）；锚检出失败同样退化并标记

目录约定（用户要求：不碰原数据，单独目录）:
    data/faces_multiframe/<match>/<kick>/frame_<t>.jpg   保留的球员帧 crop
                                                         （gitignored，个人数据）
    features/fer/multiframe_enet_b0_8_best_vgaf/embeddings.npz
        event_ids / embedding (N,1280 均值) / embedding_per_frame_stats /
        emo_probs (N,8 均值) / meta
    data/metadata/multiframe_audit.csv                    逐帧审计（硬约束 10）
    data/metadata/multiframe_summary.csv                  逐踢次统计

聚合模式（--agg）:
    mean    embedding 逐维均值（主模式，测量噪声平均）
    median  逐维中位数（对离群帧更稳）
    last_k  最接近触球的 k 帧均值（k=--last-k，默认 8；测试"越近越紧张"）

用法（项目根目录）:
    python scripts/cache_multiframe_features.py --model enet_b0_8_best_vgaf
    python scripts/cache_multiframe_features.py --agg last_k --last-k 8
    python scripts/cache_multiframe_features.py --save-frames   # 首次跑建议开，便于抽查
    python scripts/cache_multiframe_features.py --max-kicks 20  # 只跑前 20 个踢次（调试）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "metadata" / "events.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FACES_DIR = PROJECT_ROOT / "data" / "faces"                 # 锚（人工验证），只读
MF_FACES_DIR = PROJECT_ROOT / "data" / "faces_multiframe"   # 多帧 crop（新目录，不碰原数据）
FEATURES_DIR = PROJECT_ROOT / "features" / "fer"
YUNET_PATH = PROJECT_ROOT / "models" / "detection" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = PROJECT_ROOT / "models" / "detection" / "face_recognition_sface_2021dec.onnx"
AUDIT_PATH = PROJECT_ROOT / "data" / "metadata" / "multiframe_audit.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "metadata" / "multiframe_summary.csv"

SAMPLE_FPS = 2.0          # 窗口内采样帧率
GUARD_S = 0.5             # 触球前护栏（硬约束 1，与单帧协议一致）
MIN_FACE_W = 48           # 人脸最小宽度 px（比单帧的 20px 严：聚合要质量不要数量）
MIN_CONF = 0.6            # YuNet 置信度阈值
SIM_THRESHOLD = 0.363     # SFace 官方余弦阈值（同一人）
MAX_FACES_PER_FRAME = 8   # 防御：单帧检出脸数上限（观众席帧直接跳过）


def load_valid_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_PATH)
    excl = ev["exclusion_reason"].fillna("").astype(str).str.strip() != ""
    noface = ev["face_path"].fillna("").astype(str).str.strip() == ""
    return ev[~excl & ~noface].copy()


def build_fer(model_name: str, device: str):
    """与 cache_fer_features.build_fer 同构（embedding + 情绪头），返回 (model, head, dim, prep)。

    MTL 模型（名字含 _mtl）额外返回 VA 头 (2, D)：classifier_weights 最后两维
    = (valence, arousal)，情绪头取前 8 维。
    """
    import torch
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    fer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)
    emo_weights = np.asarray(fer.classifier_weights, dtype=np.float32)
    emo_bias = np.asarray(getattr(fer, "classifier_bias", np.zeros(emo_weights.shape[0])), dtype=np.float32)
    va_head = None
    if "_mtl" in model_name:
        assert emo_weights.shape[0] == 10, f"MTL 头应为 10 维（8 情绪 + 2 VA），实际 {emo_weights.shape}"
        va_head = (emo_weights[-2:], emo_bias[-2:])   # (valence, arousal)
        emo_weights, emo_bias = emo_weights[:8], emo_bias[:8]
    n_classes, emb_dim = emo_weights.shape
    fer.model.eval()
    prep = (int(fer.img_size), tuple(float(v) for v in fer.mean), tuple(float(v) for v in fer.std))
    return fer.model, (emo_weights, emo_bias), emb_dim, prep, va_head


def make_preprocess(img_size: int, mean, std):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((int(img_size), int(img_size))),
        transforms.ToTensor(),
        transforms.Normalize([float(v) for v in mean], [float(v) for v in std]),
    ])


def detect_faces(det, frame) -> np.ndarray | None:
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    return faces


def anchor_feature(rec, det, anchor_path: Path) -> tuple[np.ndarray | None, str]:
    """锚 embedding；锚图无脸时返回 (None, 'anchor_no_face')。"""
    im = cv2.imread(str(anchor_path))
    if im is None:
        return None, "anchor_unreadable"
    faces = detect_faces(det, im)
    if faces is None or not len(faces):
        return None, "anchor_no_face"
    al = rec.alignCrop(im, faces[0])
    return rec.feature(al), "ok"


def process_kick(r, det, rec, cap, fps, fer_model, fer_head, preprocess, device,
                 agg: str, last_k: int, save_frames: bool, torch, va_head=None) -> dict:
    """单踢次：窗口采样 → 逐帧检测/身份验证/前向 → 聚合。返回 summary 行。"""
    import torch as _torch
    match_id, kick = r["match_id"], r["face_path"].split("/")[-1].replace(".jpg", "")
    t0 = float(r["face_closeup_s"])
    t1 = float(r["contact_time_s"]) - GUARD_S
    row = {"event_id": r["event_id"], "match_id": match_id, "kick": kick,
           "window_s": round(t1 - t0, 2), "anchor_status": "", "n_sampled": 0,
           "n_detected": 0, "n_player": 0, "n_identity_mismatch": 0,
           "n_too_small": 0, "n_no_face": 0, "n_forward_fail": 0,
           "mean_sim": None, "median_face_w": None, "emb_dim": None}

    # 锚
    anchor_path = PROJECT_ROOT / r["face_path"]
    f_anchor, a_status = anchor_feature(rec, det, anchor_path)
    row["anchor_status"] = a_status
    use_identity = f_anchor is not None

    kick_dir = MF_FACES_DIR / match_id / kick
    if save_frames:
        kick_dir.mkdir(parents=True, exist_ok=True)

    ts = np.arange(t0, max(t0, t1), 1.0 / SAMPLE_FPS)
    feats, probs, sims, widths, va_vals = [], [], [], [], []
    audit_rows = []
    for t in ts:
        row["n_sampled"] += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            audit_rows.append((t, "read_fail", None, None))
            continue
        faces = detect_faces(det, frame)
        if faces is None or not len(faces):
            row["n_no_face"] += 1
            audit_rows.append((t, "no_face", None, None))
            continue
        row["n_detected"] += 1
        if len(faces) > MAX_FACES_PER_FRAME:  # 观众席大远景
            row["n_no_face"] += 1
            audit_rows.append((t, "too_many_faces", None, None))
            continue
        # 逐脸：SFace 相似度选最佳（过阈值）者
        best = None  # (sim, w, aligned, feat)
        for f in faces:
            w = float(f[2])
            if w < MIN_FACE_W:
                continue
            al = rec.alignCrop(frame, f)
            ft = rec.feature(al)
            sim = float(rec.match(f_anchor, ft, cv2.FaceRecognizerSF_FR_COSINE)) if use_identity else 0.0
            if not use_identity:
                # 锚缺失退化：取窗口内最大脸（方案 A），sim 记 NaN
                if best is None or w > best[1]:
                    best = (float("nan"), w, al, ft)
            elif sim >= SIM_THRESHOLD and (best is None or sim > best[0]):
                best = (sim, w, al, ft)
        if best is None:
            # 有检出但全被尺寸/身份过滤
            if use_identity:
                row["n_identity_mismatch"] += 1
                audit_rows.append((t, "identity_mismatch", None, None))
            else:
                row["n_too_small"] += 1
                audit_rows.append((t, "too_small", None, None))
            continue
        sim, w, al, ft = best
        # FER 前向（SFace aligned crop 112px → FER 预处理 resize 到 img_size）
        try:
            rgb = cv2.cvtColor(al, cv2.COLOR_BGR2RGB)
            x = preprocess(rgb).unsqueeze(0).to(device)
            with _torch.no_grad():
                z = fer_model(x)
                if z.dim() > 2:
                    z = z.flatten(1)
                w_e, b_e = fer_head
                logits = z @ torch.as_tensor(w_e, dtype=z.dtype, device=z.device).T \
                    + torch.as_tensor(b_e, dtype=z.dtype, device=z.device)
                p = torch.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()
            feats.append(z.squeeze(0).float().cpu().numpy())
            probs.append(p)
            if va_head is not None:
                w_v, b_v = va_head
                va_vals.append((z @ torch.as_tensor(w_v, dtype=z.dtype, device=z.device).T
                                + torch.as_tensor(b_v, dtype=z.dtype, device=z.device)
                                ).squeeze(0).float().cpu().numpy())
            sims.append(sim)
            widths.append(w)
            row["n_player"] += 1
            audit_rows.append((t, "kept", round(sim, 4) if not np.isnan(sim) else None, int(w)))
            if save_frames:
                cv2.imwrite(str(kick_dir / f"frame_{t:08.2f}.jpg"), al)
        except Exception:
            row["n_forward_fail"] += 1
            audit_rows.append((t, "forward_fail", None, None))

    row["mean_sim"] = round(float(np.nanmean(sims)), 4) if sims else None
    row["median_face_w"] = int(np.median(widths)) if widths else None
    row["emb_dim"] = int(feats[0].shape[0]) if feats else None
    if feats:
        F = np.stack(feats)
        P = np.stack(probs)
        if agg == "median":
            row["embedding"] = np.median(F, axis=0)
            row["emo_probs"] = np.median(P, axis=0)
            if va_vals:
                row["va"] = np.median(np.stack(va_vals), axis=0)
        elif agg == "last_k":
            Fk, Pk = F[-last_k:], P[-last_k:]  # ts 升序 → 末尾即最接近触球
            row["embedding"] = Fk.mean(axis=0)
            row["emo_probs"] = Pk.mean(axis=0)
            if va_vals:
                row["va"] = np.stack(va_vals)[-last_k:].mean(axis=0)
        else:
            row["embedding"] = F.mean(axis=0)
            row["emo_probs"] = P.mean(axis=0)
            if va_vals:
                row["va"] = np.stack(va_vals).mean(axis=0)
    for t, reason, sim, w in audit_rows:
        yield ({"event_id": r["event_id"], "t": round(float(t), 3), "frame_reason": reason,
                "sim": sim, "face_w": w}, row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="enet_b0_8_best_vgaf")
    ap.add_argument("--agg", default="mean", choices=["mean", "median", "last_k"])
    ap.add_argument("--last-k", type=int, default=8)
    ap.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    ap.add_argument("--save-frames", action="store_true", help="保留球员帧 crop 到 data/faces_multiframe/")
    ap.add_argument("--max-kicks", type=int, default=0, help="只处理前 N 个踢次（0=全部，调试用）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    arm = f"multiframe_{args.agg}{'_k' + str(args.last_k) if args.agg == 'last_k' else ''}_{args.model}"
    out_dir = FEATURES_DIR / arm
    emb_path = out_dir / "embeddings.npz"
    if emb_path.exists() and not args.force:
        print(f"缓存已存在: {emb_path}（--force 重算）")
        return 0

    import torch
    from PIL import Image  # noqa: F401（make_preprocess 的 ToPILImage 依赖）

    ev = load_valid_events()
    if args.max_kicks:
        ev = ev.head(args.max_kicks)

    torch.use_deterministic_algorithms(True, warn_only=True)
    det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), MIN_CONF, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    fer_model, fer_head, emb_dim, prep, va_head = build_fer(args.model, args.device)
    preprocess = make_preprocess(*prep)
    print(f"arm={arm} | device={args.device} | agg={args.agg} | n_kicks={len(ev)} | va_head={'yes' if va_head is not None else 'no'}")

    audit_rows, summaries, embeddings, probs_all, va_all, ids_order = [], [], [], [], [], []
    for _, r in ev.iterrows():
        cap = cv2.VideoCapture(str(RAW_DIR / f"{r['match_id']}.mp4"))
        if not cap.isOpened():
            print(f"!! 打不开视频 {r['match_id']}", file=sys.stderr)
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        gen = process_kick(r, det, rec, cap, fps, fer_model, fer_head, preprocess,
                           args.device, args.agg, args.last_k, args.save_frames, torch, va_head)
        got_summary = None
        for audit, summary in gen:
            audit_rows.append(audit)
            got_summary = summary
        cap.release()
        if got_summary is None:
            continue
        if "embedding" in got_summary:
            embeddings.append(got_summary.pop("embedding"))
            probs_all.append(got_summary.pop("emo_probs"))
            va_all.append(got_summary.pop("va") if "va" in got_summary else None)
            ids_order.append(got_summary["event_id"])
        else:
            embeddings.append(None)
            probs_all.append(None)
            va_all.append(None)
            ids_order.append(got_summary["event_id"])
        summaries.append(got_summary)
        s = got_summary
        print(f"  {s['event_id']:>12}: win {s['window_s']:>5.1f}s | sampled {s['n_sampled']:>3} "
              f"| player {s['n_player']:>2} | mismatch {s['n_identity_mismatch']:>2} "
              f"| no_face {s['n_no_face']:>2} | sim {s['mean_sim']}")

    # 落盘
    ok = [i for i, e in zip(ids_order, embeddings) if e is not None]
    n_empty = len(ids_order) - len(ok)
    E = np.stack([e for e in embeddings if e is not None]).astype(np.float32)
    P = np.stack([p for p in probs_all if p is not None]).astype(np.float32)
    has_va = va_head is not None and any(v is not None for v in va_all)
    # VA 与 embedding 同序同长：只取有帧踢次的行（空踢次的占位零行剔除），
    # 否则 va (384,2) 与 event_ids (356) 错位，下游对齐会静默读错
    VA = (np.stack([v for v, e in zip(va_all, embeddings) if e is not None]).astype(np.float32)
          if has_va else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "arm": arm, "model_name": args.model, "engine": "torch",
        "img_size": int(prep[0]), "embedding_dim": emb_dim,
        "n_emotion_classes": 8, "aggregation": args.agg,
        "has_va": has_va, "va_order": ["valence", "arousal"] if has_va else None,
        "last_k": args.last_k if args.agg == "last_k" else None,
        "window": "[face_closeup_s, contact_time_s - 0.5]",
        "sample_fps": SAMPLE_FPS, "guard_s": GUARD_S,
        "min_face_w": MIN_FACE_W, "min_conf": MIN_CONF,
        "sim_threshold": SIM_THRESHOLD,
        "identity": "SFace vs 人工验证锚（face_path crop）；锚缺失退化为窗口最大脸",
        "built_date": date.today().isoformat(),
        "n_kicks_total": len(ev), "n_kicks_with_frames": len(ok), "n_kicks_empty": n_empty,
        "event_ids_source": "events.csv (face_path 非空且未排除)",
    }
    save_kw = {"va": VA} if has_va else {}
    np.savez_compressed(emb_path, event_ids=np.array(ok),
                        embedding=E, emo_probs=P, meta=np.array(json.dumps(meta)), **save_kw)
    pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(audit_rows).to_csv(AUDIT_PATH, index=False)
    report = {**meta,
              "frames_per_kick_mean": round(float(np.mean([s["n_player"] for s in summaries])), 2),
              "frames_per_kick_median": float(np.median([s["n_player"] for s in summaries])),
              "kicks_with_lt3_player_frames": int(sum(1 for s in summaries if s["n_player"] < 3)),
              "identity_mismatch_rate": round(float(np.sum([s["n_identity_mismatch"] for s in summaries])
                                                    / max(1, np.sum([s["n_sampled"] for s in summaries]))), 4)}
    (out_dir / "cache_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n缓存完成: {emb_path.relative_to(PROJECT_ROOT)}")
    print(f"  {len(ok)}/{len(ev)} 踢次有球员帧 | 每踢帧数 mean {report['frames_per_kick_mean']} "
          f"median {report['frames_per_kick_median']} | <3 帧踢次 {report['kicks_with_lt3_player_frames']}")
    print(f"  审计: {AUDIT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  汇总: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
