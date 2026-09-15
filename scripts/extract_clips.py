#!/usr/bin/env python
"""切片脚本：根据打点时间戳从原始视频切出触球前片段。

用法（项目根目录）:
    python scripts/extract_clips.py                # 处理所有有 marker 的比赛
    python scripts/extract_clips.py --match m004   # 只处理一场
    python scripts/extract_clips.py --pad-before 15 --end-margin 0.5

输入:
    data/raw/<match_id>.mp4                       原始视频
    data/metadata/kick_markers/<match_id>_markers.csv   marker.py 的打点输出

输出（每个切片一个 mp4，路径约定与 DATA_DICTIONARY 一致）:
    data/clips/<match_id>/<kick_seq>.mp4          触球前切片（可视为静音视频）
    data/metadata/clips_index.csv                 切片索引：时长、起止秒、覆盖状态

切片规则（DATA_DICTIONARY §3.1，冻结）:
    起点 = max(上一次触球时刻, 本次触球时刻 - pad_before)
    终点 = 本次触球时刻 - end_margin（默认 0.5s，严格触球前）

    pad-before 默认 45s（2026-09-15 实测标定）：转播给主罚球员的面部特写
    通常出现在踢前 20-40s（球员从底线走回罚球点、放球、深呼吸阶段），
    而 15s 窗口只覆盖助跑阶段——画面是远景/球/门将，没有大脸。
    m001/k02 实测：15s 窗口 0 帧合格脸，扩到 45s 后 97 帧（103-227px）。

说明:
    - 用 OpenCV 逐帧读写，不依赖 ffmpeg（本机未装）；
    - 360p 25fps 下单切片 ~15s，14 场总输出 < 500MB；
    - 若视频码流含音轨，OpenCV 只写视频帧，切片自然无声——对本研究无影响；
    - marker 覆盖检查：若最后打点之后视频仍有 <40s 尾巴，提示可能有漏标
      （一场 8-10 踢的点球大战很少在最后一踢后还留 1 分钟以上）。
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


def load_stamps(mark_path: Path) -> list[float]:
    with open(mark_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stamps = sorted(float(r["contact_time_s"]) for r in rows if r.get("contact_time_s"))
    if not stamps:
        raise ValueError(f"{mark_path} 无有效打点")
    return stamps


def check_tail_coverage(match_id: str, stamps: list[float], duration_s: float, min_tail: float = 40.0) -> None:
    """最后打点后仍剩很长视频，通常意味着漏标（点球大战以最后一踢结束）。"""
    tail = duration_s - stamps[-1]
    if tail > min_tail:
        print(f"  !! {match_id}: 最后打点后仍有 {tail:.0f}s 视频——检查是否漏标了后续罚球"
              f"（kicks={len(stamps)}，完整点球大战通常 8-10+ 踢）")


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
    ap.add_argument("--pad-before", type=float, default=45.0,
                    help="切片最长回看秒数（默认 45s：转播特写多在踢前 20-40s，15s 只覆盖助跑）")
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

        stamps = load_stamps(mark_path)
        print(f"{match_id}: {len(stamps)} 踢 | {duration:.0f}s {w}x{h}@{fps:.0f}fps")
        check_tail_coverage(match_id, stamps, duration)

        for i, contact_t in enumerate(stamps):
            kick_seq = f"k{i + 1:02d}"
            start_t = max(stamps[i - 1] if i > 0 else 0.0, contact_t - args.pad_before)
            end_t = contact_t - args.end_margin
            if end_t - start_t < 1.0:
                # 踢间隔过密或打点异常：不产出切片，写索引供人工排查
                print(f"  !! {kick_seq}: 切片过短 ({end_t - start_t:.1f}s)，打点可能异常，跳过")
                index_rows.append({"match_id": match_id, "kick_seq": kick_seq,
                                   "contact_time_s": f"{contact_t:.2f}", "clip_start_s": f"{start_t:.2f}",
                                   "clip_end_s": f"{end_t:.2f}", "clip_duration_s": f"{end_t - start_t:.2f}",
                                   "status": "too_short"})
                continue

            out_path = CLIPS_DIR / match_id / f"{kick_seq}.mp4"
            actual = extract_clip(cap, out_path,
                                  int(start_t * fps), int(end_t * fps), fps, w, h)
            index_rows.append({"match_id": match_id, "kick_seq": kick_seq,
                               "contact_time_s": f"{contact_t:.2f}", "clip_start_s": f"{start_t:.2f}",
                               "clip_end_s": f"{end_t:.2f}", "clip_duration_s": f"{actual:.2f}",
                               "status": "ok"})
            print(f"  {kick_seq}: {start_t:7.2f} -> {end_t:7.2f}s  ({actual:.1f}s)  {out_path.relative_to(PROJECT_ROOT)}")
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
