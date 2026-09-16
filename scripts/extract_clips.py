#!/usr/bin/env python
"""切片脚本：根据打点时间戳从原始视频切出触球前片段。

用法（项目根目录）:
    python scripts/extract_clips.py                # 处理所有有 marker 的比赛
    python scripts/extract_clips.py --match m004   # 只处理一场
    python scripts/extract_clips.py --pad-before 8 --pad-after 6 --end-margin 0.5

输入:
    data/raw/<match_id>.mp4                       原始视频
    data/metadata/kick_markers/<match_id>_markers.csv   marker.py 的打点输出

输出:
    data/clips/<match_id>/<kick_seq>.mp4          触球前切片
    data/metadata/clips_index.csv                 切片索引：起止秒、模式、状态

切片规则（2026-09-16 更新，双打点方案）:
    有特写点时: 窗口 = [face_closeup_s - pad_before, min(face_closeup_s + pad_after, contact_time_s - end_margin)]
                特写点由人工锁定转播给脸的时段，切片小而准（默认 8+6=14s）；
    漏打特写点: 回退旧逻辑 窗口 = [max(上一踢触球, contact_time_s - 45), contact_time_s - end_margin]
                （45s 标定依据见 DATA_DICTIONARY CHANGELOG 2026-09-15 条目）
    终点恒受 contact_time_s - 0.5s 硬截断——切片永不越过触球前 0.5s（防泄漏底线）。

说明:
    - 用 OpenCV 逐帧读写，不依赖 ffmpeg（本机未装）；
    - 双打点下切片 ~14s/踢，比 45s 窗口小 3 倍，选帧扫描同步提速；
    - OpenCV 只写视频帧，切片无声——对本研究无影响；
    - 若最后打点之后视频仍有 >40s 尾巴，提示可能有漏标（领奖/回放除外，人工核对）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MARK_DIR = PROJECT_ROOT / "data" / "metadata" / "kick_markers"
CLIPS_DIR = PROJECT_ROOT / "data" / "clips"
INDEX_PATH = PROJECT_ROOT / "data" / "metadata" / "clips_index.csv"

# mp4v 编码器：OpenCV 内置可用，兼容性足够本 pipeline 内部使用（后续选帧读回）
FOURCC = "mp4v"

# 漏打特写点时的回退窗口（DATA_DICTIONARY CHANGELOG 2026-09-15 标定）
FALLBACK_PAD_BEFORE = 45.0


def load_kicks(mark_path: Path) -> list[dict]:
    """读打点文件，兼容旧格式（无 face_closeup_s 列）。"""
    with open(mark_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("contact_time_s")]
    if not rows:
        raise ValueError(f"{mark_path} 无有效打点")
    kicks = [{"face_closeup_s": float(r["face_closeup_s"]) if r.get("face_closeup_s") else None,
              "contact_time_s": float(r["contact_time_s"])} for r in rows]
    kicks.sort(key=lambda k: k["contact_time_s"])
    return kicks


def check_tail_coverage(match_id: str, last_contact: float, duration_s: float, min_tail: float = 40.0) -> None:
    """最后打点后仍剩很长视频，通常意味着领奖/回放或漏标——提示人工核对。"""
    tail = duration_s - last_contact
    if tail > min_tail:
        print(f"  !! {match_id}: 最后打点后仍有 {tail:.0f}s 视频——检查是否漏标了后续罚球"
              f"（完整点球大战通常 8-10+ 踢）")


def extract_clip(cap: cv2.VideoCapture, out_path: Path, start_f: int, end_f: int,
                 fps: float, w: int, h: int) -> float:
    """写出 [start_f, end_f) 帧到 out_path，返回实际写出时长（秒）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*FOURCC), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 打开失败: {out_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    written = 0
    for _ in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    return written / fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", help="只处理指定比赛，如 m004；缺省处理全部")
    ap.add_argument("--pad-before", type=float, default=8.0,
                    help="特写点向前回看秒数（默认 8s，特写已由人工打点锁定）")
    ap.add_argument("--pad-after", type=float, default=6.0,
                    help="特写点向后延伸秒数（默认 6s，仍受触球点前 0.5s 硬截断）")
    ap.add_argument("--end-margin", type=float, default=0.5, help="切片终点距触球时刻的余量（默认 0.5s）")
    args = ap.parse_args()

    mark_files = sorted(MARK_DIR.glob("*_markers.csv"))
    if args.match:
        mark_files = [MARK_DIR / f"{args.match}_markers.csv"]
    mark_files = [p for p in mark_files if p.exists()]
    if not mark_files:
        print("未找到任何 marker 文件。先用 scripts/marker.py 打点。", file=sys.stderr)
        return 1

    index_rows: list[dict] = []
    for mark_path in mark_files:
        match_id = mark_path.name.replace("_markers.csv", "")
        video_path = RAW_DIR / f"{match_id}.mp4"
        if not video_path.exists():
            print(f"跳过 {match_id}: 缺视频 {video_path}", file=sys.stderr)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"跳过 {match_id}: 视频无法打开", file=sys.stderr)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = n_frames / fps

        kicks = load_kicks(mark_path)
        n_fallback = sum(1 for k in kicks if k["face_closeup_s"] is None)
        print(f"{match_id}: {len(kicks)} 踢（特写点 {len(kicks) - n_fallback}，回退 {n_fallback}）"
              f" | {duration:.0f}s {w}x{h}@{fps:.0f}fps")
        check_tail_coverage(match_id, kicks[-1]["contact_time_s"], duration)

        for i, kick in enumerate(kicks):
            kick_seq = f"k{i + 1:02d}"
            contact_t = kick["contact_time_s"]
            face_t = kick["face_closeup_s"]
            if face_t is not None:
                # 双打点方案：特写点锁定窗口，终点受触球点硬截断（防泄漏底线）
                start_t = max(0.0, face_t - args.pad_before)
                end_t = min(face_t + args.pad_after, contact_t - args.end_margin)
                mode = "closeup"
            else:
                # 漏打特写点：回退 45s 旧窗口
                prev_contact = kicks[i - 1]["contact_time_s"] if i > 0 else 0.0
                start_t = max(prev_contact, contact_t - FALLBACK_PAD_BEFORE)
                end_t = contact_t - args.end_margin
                mode = "fallback"
            if end_t - start_t < 1.0:
                # 窗口过短或打点异常：不产出切片，写索引供人工排查
                print(f"  !! {kick_seq}: 切片过短 ({end_t - start_t:.1f}s)，打点可能异常，跳过")
                index_rows.append({"match_id": match_id, "kick_seq": kick_seq,
                                   "contact_time_s": f"{contact_t:.2f}",
                                   "face_closeup_s": "" if face_t is None else f"{face_t:.2f}",
                                   "clip_start_s": f"{start_t:.2f}", "clip_end_s": f"{end_t:.2f}",
                                   "clip_duration_s": f"{end_t - start_t:.2f}", "mode": mode,
                                   "status": "too_short"})
                continue

            out_path = CLIPS_DIR / match_id / f"{kick_seq}.mp4"
            actual = extract_clip(cap, out_path,
                                  int(start_t * fps), int(end_t * fps), fps, w, h)
            index_rows.append({"match_id": match_id, "kick_seq": kick_seq,
                               "contact_time_s": f"{contact_t:.2f}",
                               "face_closeup_s": "" if face_t is None else f"{face_t:.2f}",
                               "clip_start_s": f"{start_t:.2f}", "clip_end_s": f"{end_t:.2f}",
                               "clip_duration_s": f"{actual:.2f}", "mode": mode,
                               "status": "ok"})
            print(f"  {kick_seq} [{mode}]: {start_t:7.2f} -> {end_t:7.2f}s  ({actual:.1f}s)  "
                  f"{out_path.relative_to(PROJECT_ROOT)}")
        cap.release()

    if index_rows:
        with open(INDEX_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
        n_ok = sum(1 for r in index_rows if r["status"] == "ok")
        print(f"\n共 {n_ok}/{len(index_rows)} 个切片 -> {CLIPS_DIR.relative_to(PROJECT_ROOT)}/")
        print(f"索引: {INDEX_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
