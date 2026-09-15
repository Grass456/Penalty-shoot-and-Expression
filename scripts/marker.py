#!/usr/bin/env python
"""标记辅助工具：播放点球大战视频，按快捷键记录时间戳。

用法（在项目根目录）:
    python scripts/marker.py --match m004

操作:
    空格       = 打点：记录当前播放时间为一次触球时刻（contact_time_s）
    d          = 撤销上一个打点（注意：快进/快退改用 a / e 键）
    a / e      = 快退/快进 5 秒
    p          = 暂停/继续
    q / ESC    = 退出（自动保存）

说明: 方向键在很多平台的 OpenCV HighGUI 里返回非标准键值（Windows 上左右键分别
      返回 0 和 1，与 Linux 的 81/83 不同），因此快捷跳转用 a/e 键更可靠；
      程序对两套键值都做了兼容。

输出:
    data/metadata/kick_markers/m004_markers.csv  ->  kick_seq, contact_time_s
    标注员随后参考此文件把 contact_time_s 等填入 data/metadata/events.csv，
    同时人工判断 result 与轮次/比分状态。

视频窗口内同步显示当前时间与已打点数；播放速度自动降为 0.5x，
便于抓触球瞬间（默认标准速度，抓触球帧可加 --speed 0.5 慢放）。

说明: 不依赖 ffmpeg，仅用 OpenCV 读帧。打点精度约 1 帧（0.04s@25fps），
      足够满足 DATA_DICTIONARY 对触球时刻的精度要求（切片终点留 0.5s 余量）。
      默认标准速度播放；抓触球瞬间可加 --speed 0.5 慢放。
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

# 简易按键打点 UI：窗口大小按 720p 上限缩放，全屏视频不影响时间戳精度
MAX_WIN_W = 1280


def load_existing(mark_path: Path) -> list[float]:
    """断点续标：已有 marker 文件则读入已打点的时间戳。"""
    if not mark_path.exists():
        return []
    with open(mark_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stamps = [float(r["contact_time_s"]) for r in rows if r.get("contact_time_s")]
    print(f"载入已有打点 {len(stamps)} 个: {stamps}")
    return stamps


def save(mark_path: Path, stamps: list[float]) -> None:
    mark_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mark_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kick_seq", "contact_time_s"])
        for i, t in enumerate(stamps, 1):
            w.writerow([f"k{i:02d}", f"{t:.2f}"])
    print(f"已保存 {len(stamps)} 个打点 -> {mark_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True, help="比赛编号，如 m004")
    ap.add_argument("--speed", type=float, default=1.0, help="播放速度倍率（默认 1.0 标准速度；0.5 慢放抓触球帧）")
    args = ap.parse_args()

    video_path = RAW_DIR / f"{args.match}.mp4"
    if not video_path.exists():
        print(f"找不到视频: {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_min = n_frames / fps / 60
    print(f"{video_path.name}: {fps:.1f}fps, {dur_min:.1f} min | speed={args.speed}x")
    print("空格=打点  d=撤销  a/d键=±5s  p=暂停  q=退出保存\n")

    MARK_DIR.mkdir(parents=True, exist_ok=True)
    mark_path = MARK_DIR / f"{args.match}_markers.csv"
    stamps = load_existing(mark_path)

    # 从上次最后打点稍前的位置继续，避免重复看已标过的部分
    frame_idx = int((stamps[-1] - 5) * fps) if stamps else 0
    frame_idx = max(frame_idx, 0)

    # 每帧停留时长由播放速度决定：0.5x @ 25fps = 80ms/帧。
    # 逐帧 read 顺序推进，不做定时 seek——按时间差 seek 既慢又不准，
    # 还会让实际速度取决于循环耗时（正是最初版本默认速度偏快的原因）。
    frame_delay_ms = max(1, int(1000 / (fps * args.speed)))
    paused = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("视频播放完毕")
            break
        cur_t = frame_idx / fps

        # 缩放显示窗口，保持标注体验一致
        h, w = frame.shape[:2]
        scale = min(1.0, MAX_WIN_W / w)
        disp = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

        hud = f"{args.match}  t={cur_t:8.2f}s  kicks={len(stamps)}"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        cv2.imshow("marker", disp)

        key = cv2.waitKey(frame_delay_ms if not paused else 30) & 0xFF
        frame_idx += 1
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            stamps.append(round(cur_t, 2))
            print(f"  打点 k{len(stamps):02d} @ {cur_t:.2f}s")
        elif key == ord("d"):
            if stamps:
                t = stamps.pop()
                print(f"  撤销 k{len(stamps) + 1:02d} @ {t:.2f}s")
        elif key in (81, 2, ord("a")):  # 左方向键(win=0/2) 或 a：快退 5s
            frame_idx = max(0, frame_idx - int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            print(f"  快退至 {frame_idx / fps:.2f}s")
        elif key in (83, 3, ord("e")):  # 右方向键(win=1/3) 或 e：快进 5s
            frame_idx = min(n_frames - 1, frame_idx + int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            print(f"  快进至 {frame_idx / fps:.2f}s")
        elif key == ord("p"):
            paused = not paused
            print(f"  {'暂停' if paused else '继续'}")

    cap.release()
    cv2.destroyAllWindows()
    save(mark_path, stamps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
