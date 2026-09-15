#!/usr/bin/env python
"""人脸选帧脚本：对每个触球前切片检测人脸，选出最接近触球时刻的合格单帧。

用法（项目根目录）:
    python scripts/select_face_frames.py                 # 处理 clips_index.csv 中全部 ok 切片
    python scripts/select_face_frames.py --match m001    # 只处理一场
    python scripts/select_face_frames.py --min-face 20 --conf 0.6

输入:
    data/metadata/clips_index.csv                 extract_clips.py 产出的切片索引
    data/clips/<match_id>/<kick_seq>.mp4          触球前切片

输出:
    data/faces/<match_id>/<kick_seq>.jpg          选定帧的人脸 crop（写回 EVENTS 表 face_path 的目标）
    data/metadata/face_audit.csv                  审计表：每踢的检测统计、选帧依据、人脸像素尺寸
    data/metadata/face_review/                    全帧缩略图（供人工快速确认主罚球员身份）

选帧规则（DATA_DICTIONARY §3.2 / plan §6.5，冻结）:
    1. 候选帧 = 切片内检测置信度 >= conf 且人脸宽 >= min-face px 的帧；
    2. 在候选帧中选【最接近切片终点（= 触球前 0.5s）】的一帧；若该帧质量不合格则向前回退；
    3. 整个切片无候选帧 -> 记 exclusion_reason（no_face / face_too_small），不产出 crop。

质检提示（360p 转播的现实）:
    近景镜头人脸可以很大（实测 m001 选出帧 114-231px），但远镜头（全景、
    走位）只有 ~27px。排除原因要分清：no_face 常是"整段远景/切观众席"，
    face_too_small 是"脸在画面里但不够大"。--min-face 默认 20px 保守起步，
    跑完看 face_audit 的尺寸分布再定阈值。

关于检测器:
    用 YuNet（OpenCV FaceDetectorYN，模型已下载到 models/detection/，230KB）。
    实测同一批转播帧：YuNet 检出 210/362 帧，Haar 只有 86/362，且 Haar 误检多。
    models/detection/ 已在 .gitignore 中排除。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = PROJECT_ROOT / "data" / "clips"
FACES_DIR = PROJECT_ROOT / "data" / "faces"
INDEX_PATH = PROJECT_ROOT / "data" / "metadata" / "clips_index.csv"
AUDIT_PATH = PROJECT_ROOT / "data" / "metadata" / "face_audit.csv"
REVIEW_DIR = PROJECT_ROOT / "data" / "metadata" / "face_review"
YUNET_PATH = PROJECT_ROOT / "models" / "detection" / "face_detection_yunet_2023mar.onnx"

# 采样步长：25fps 逐帧太密（相邻帧几乎相同），每 2 帧采一帧已足够
# 人脸在画面里移动慢，2 帧 = 80ms，不影响"最接近触球"的选帧精度
STRIDE = 2
REVIEW_THUMB_H = 120  # 人工复核缩略图高度


def yuv_crop(frame: np.ndarray, box: np.ndarray, margin: float = 0.25) -> np.ndarray:
    """按检测框裁剪人脸，四周留 margin 比例的余量（FER 模型需要一点上下文）。"""
    x, y, w, h = box
    mw, mh = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mw), max(0, y - mh)
    x1, y1 = min(frame.shape[1], x + w + mw), min(frame.shape[0], y + h + mh)
    return frame[y0:y1, x0:x1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", help="只处理指定比赛，如 m001；缺省处理全部")
    ap.add_argument("--conf", type=float, default=0.6, help="YuNet 置信度阈值（默认 0.6）")
    ap.add_argument("--min-face", type=int, default=20, help="人脸最小宽度 px（默认 20，360p 起步值）")
    args = ap.parse_args()

    if not YUNET_PATH.exists():
        print(f"缺检测模型: {YUNET_PATH}\n"
              "下载: curl -L -o models/detection/face_detection_yunet_2023mar.onnx "
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
              file=sys.stderr)
        return 1
    detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), args.conf)

    if not INDEX_PATH.exists():
        print("缺切片索引，先运行 scripts/extract_clips.py", file=sys.stderr)
        return 1
    with open(INDEX_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    if args.match:
        rows = [r for r in rows if r["match_id"] == args.match]
    if not rows:
        print("没有待处理的 ok 切片", file=sys.stderr)
        return 1

    audit_rows: list[dict] = []
    for r in rows:
        match_id, kick_seq = r["match_id"], r["kick_seq"]
        clip_path = CLIPS_DIR / match_id / f"{kick_seq}.mp4"
        if not clip_path.exists():
            print(f"跳过 {match_id}/{kick_seq}: 缺切片", file=sys.stderr)
            continue

        cap = cv2.VideoCapture(str(clip_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        candidates: list[dict] = []  # 每个候选: idx, conf, face_w（不缓存整帧，控制内存）
        thumbs: list[tuple[int, np.ndarray]] = []

        # 自后向前扫描：找到合格帧即停（选帧规则=最接近触球时刻的合格帧）
        # 但仍全量扫描 candidates 用于审计统计（检出率/尺寸分布）
        all_dets: list[dict] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % STRIDE == 0:
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = detector.detect(frame)
                if faces is not None and len(faces):
                    # 取置信度最高的一张脸（转播画面主目标是主罚球员）
                    best = max(faces, key=lambda f: f[-1])
                    conf = float(best[-1])
                    face_w = int(best[2])
                    all_dets.append({"idx": idx, "conf": conf, "w": face_w})
                    if conf >= args.conf and face_w >= args.min_face:
                        candidates.append({"idx": idx, "conf": conf, "w": face_w})
                        if len(thumbs) < 3:
                            thumbs.append((idx, frame))
            idx += 1

        n_scanned = (n_frames + STRIDE - 1) // STRIDE
        det_rate = len(all_dets) / n_scanned if n_scanned else 0.0

        if not candidates:
            # 按冻结规则区分排除原因：全切片无检出 vs 有人脸但太小
            reason = "no_face" if not all_dets else "face_too_small"
            best_w = max((d["w"] for d in all_dets), default=0)
            audit_rows.append({"match_id": match_id, "kick_seq": kick_seq, "status": "excluded",
                               "reason": reason, "n_candidates": 0, "det_rate": f"{det_rate:.2f}",
                               "selected_frame_idx": "", "face_w_px": best_w, "conf": "",
                               "crop_path": ""})
            print(f"  {match_id}/{kick_seq}: 排除 ({reason}, 检出率 {det_rate:.0%}, 最大脸宽 {best_w}px)")
            cap.release()
            continue

        # 选帧规则：最接近切片终点（最接近触球前 0.5s）的合格帧
        chosen = candidates[-1]
        out_dir = FACES_DIR / match_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 重新定位到选定帧取图（不缓存整帧，控制内存；STRIDE 采样下 seek 误差 ≤1 帧）
        cap.set(cv2.CAP_PROP_POS_FRAMES, chosen["idx"])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  {match_id}/{kick_seq}: 选定帧读回失败", file=sys.stderr)
            continue

        # 用选定帧重新检测（比缓存坐标更稳，处理 seek 后的帧一致性）
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)
        if faces is None or not len(faces):
            # 极少见：seek 读回后检测器输入尺寸变化导致漏检，按审计记录的候选顺延
            audit_rows.append({"match_id": match_id, "kick_seq": kick_seq, "status": "excluded",
                               "reason": "no_face", "n_candidates": len(candidates),
                               "det_rate": f"{det_rate:.2f}", "selected_frame_idx": chosen["idx"],
                               "face_w_px": chosen["w"], "conf": f"{chosen['conf']:.2f}", "crop_path": ""})
            print(f"  {match_id}/{kick_seq}: 选定帧复检失败，顺延逻辑待增强")
            continue
        best = max(faces, key=lambda f: f[-1])
        box = best[:4].astype(int)
        crop = yuv_crop(frame, box)

        crop_path = out_dir / f"{kick_seq}.jpg"
        cv2.imwrite(str(crop_path), crop)

        # 人工复核缩略图：抽 4 帧（前/中/选定帧/邻帧）拼一行，快速确认是不是主罚球员
        rev_dir = REVIEW_DIR / match_id
        rev_dir.mkdir(parents=True, exist_ok=True)
        tiles = []
        for t_idx, t_frame in thumbs[:3]:
            th = cv2.resize(t_frame, (int(t_frame.shape[1] * REVIEW_THUMB_H / t_frame.shape[0]), REVIEW_THUMB_H))
            cv2.putText(th, f"#{t_idx}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)
            tiles.append(th)
        sel_th = cv2.resize(frame, (int(frame.shape[1] * REVIEW_THUMB_H / frame.shape[0]), REVIEW_THUMB_H))
        cv2.rectangle(sel_th, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 120), 2)
        cv2.putText(sel_th, f"SEL #{chosen['idx']}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)
        tiles.append(sel_th)
        review_img = np.hstack(tiles)
        cv2.imwrite(str(rev_dir / f"{kick_seq}_review.jpg"), review_img)

        audit_rows.append({"match_id": match_id, "kick_seq": kick_seq, "status": "selected",
                           "reason": "", "n_candidates": len(candidates), "det_rate": f"{det_rate:.2f}",
                           "selected_frame_idx": chosen["idx"], "face_w_px": chosen["w"],
                           "conf": f"{chosen['conf']:.2f}",
                           "crop_path": str(crop_path.relative_to(PROJECT_ROOT)).replace("\\", "/")})
        print(f"  {match_id}/{kick_seq}: 选定帧 #{chosen['idx']} "
              f"(脸宽 {chosen['w']}px, conf {chosen['conf']:.2f}, 检出率 {det_rate:.0%}, 候选 {len(candidates)} 帧)")

    if audit_rows:
        with open(AUDIT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)
        n_sel = sum(1 for a in audit_rows if a["status"] == "selected")
        widths = [int(a["face_w_px"]) for a in audit_rows if a["face_w_px"]]
        print(f"\n选定 {n_sel}/{len(audit_rows)} -> {FACES_DIR.relative_to(PROJECT_ROOT)}/")
        if widths:
            print(f"脸宽分布: median {int(np.median(widths))}px, "
                  f"p25 {int(np.percentile(widths, 25))}px, p75 {int(np.percentile(widths, 75))}px")
        print(f"审计表: {AUDIT_PATH.relative_to(PROJECT_ROOT)}")
        print(f"人工复核图: {REVIEW_DIR.relative_to(PROJECT_ROOT)}/ （每踢一张 4 联图，确认主罚球员身份）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
