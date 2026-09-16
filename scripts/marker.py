#!/usr/bin/env python
"""标记辅助工具：播放点球大战视频，按快捷键记录每次罚球的两个时间戳。

用法（在项目根目录）:
    python scripts/marker.py --match m004

操作（每踢两步，按观看顺序）:
    f          = 打【特写】点：转播给主罚球员面部特写的时刻（face_closeup_s）
    空格       = 打【触球】点：脚触球的瞬间（contact_time_s）
    d          = 撤销上一个打点（f 或空格均可）
    a / e      = 快退/快进 5 秒（方向键多数平台不可靠，用 a/e）
    p          = 暂停/继续
    q / ESC    = 退出（自动保存）

流程: 看到转播给特写 -> 按 f -> 球员助跑 -> 脚触球瞬间 -> 按空格 -> 下一踢。
      每一踢必须以空格（触球）结束才完整；f 可省略（切片脚本会回退旧窗口逻辑）。

输出:
    data/metadata/kick_markers/m004_markers.csv
        列: kick_seq, face_closeup_s, contact_time_s
    标注员随后参考此文件填写 data/metadata/events.csv 的 result/轮次/比分状态。

切片如何用这两个点（extract_clips.py）:
    窗口 = [face_closeup_s - 8s, min(face_closeup_s + 6s, contact_time_s - 0.5s)]
    特写点锁定转播给脸的时段，触球点做防泄漏硬上界（切片永不越过触球前 0.5s）。
    漏打 f 的踢自动回退: [contact_time_s - 45s, contact_time_s - 0.5s]。

按键兼容性: 方向键在 Windows 的 OpenCV HighGUI 返回 0/1/2/3，Linux 返回 81/83 等，
      两套键值都做了兼容，但推荐用 a/e。

打点精度约 1 帧（0.04s@25fps）。默认标准速度；抓触球瞬间建议 --speed 0.5 慢放。
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

MAX_WIN_W = 1280  # 显示窗口宽度上限，不影响打点精度


def load_existing(mark_path: Path) -> list[dict]:
    """断点续标：读入已打点。旧格式（无 face_closeup_s 列）也能读。"""
    if not mark_path.exists():
        return []
    with open(mark_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("contact_time_s")]
    kicks = [{"face_closeup_s": float(r["face_closeup_s"]) if r.get("face_closeup_s") else None,
              "contact_time_s": float(r["contact_time_s"])} for r in rows]
    print(f"载入已有打点 {len(kicks)} 踢")
    return kicks


def save(mark_path: Path, kicks: list[dict]) -> None:
    mark_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mark_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kick_seq", "face_closeup_s", "contact_time_s"])
        for i, k in enumerate(kicks, 1):
            fc = f"{k['face_closeup_s']:.2f}" if k["face_closeup_s"] is not None else ""
            w.writerow([f"k{i:02d}", fc, f"{k['contact_time_s']:.2f}"])
    n_face = sum(1 for k in kicks if k["face_closeup_s"] is not None)
    print(f"已保存 {len(kicks)} 踢（含特写点 {n_face} 个）-> {mark_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True, help="比赛编号，如 m004")
    ap.add_argument("--speed", type=float, default=1.0, help="播放速度倍率（默认 1.0；0.5 慢放抓触球帧）")
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
    print(f"{video_path.name}: {fps:.1f}fps, {n_frames / fps / 60:.1f} min | speed={args.speed}x")
    print("f=特写点  空格=触球点  d=撤销  a/e=±5s  p=暂停  q=退出保存\n")

    MARK_DIR.mkdir(parents=True, exist_ok=True)
    mark_path = MARK_DIR / f"{args.match}_markers.csv"
    kicks = load_existing(mark_path)

    # 续标起点：丢弃只有特写点、未按触球点的未完成踢
    while kicks and kicks[-1]["contact_time_s"] is None:
        print("  丢弃最后一踢（只有特写点）")
        kicks.pop()
    start_t = (kicks[-1]["contact_time_s"] - 5) if kicks else 0.0
    frame_idx = max(0, int(start_t * fps))

    frame_delay_ms = max(1, int(1000 / (fps * args.speed)))
    paused = False
    pending_face: float | None = None  # 当前踢的特写点（f 已按、空格未按）

    while True:
        ok, frame = cap.read()
        if not ok:
            print("视频播放完毕")
            break
        cur_t = frame_idx / fps

        h, w = frame.shape[:2]
        scale = min(1.0, MAX_WIN_W / w)
        disp = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

        n_done = len(kicks)
        status = f"待触球 f@{pending_face:.2f}s" if pending_face is not None else "等待特写 f"
        hud = f"{args.match}  t={cur_t:8.2f}s  kicks={n_done}  [{status}]"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
        cv2.imshow("marker", disp)

        key = cv2.waitKey(frame_delay_ms if not paused else 30) & 0xFF
        frame_idx += 1

        if key in (ord("q"), 27):
            break
        elif key == ord("f"):
            pending_face = round(cur_t, 2)
            print(f"  特写点 @ {cur_t:.2f}s （踢 k{n_done + 1:02d}，按空格完成）")
        elif key == ord(" "):
            kicks.append({"face_closeup_s": pending_face, "contact_time_s": round(cur_t, 2)})
            tag = f"f@{pending_face:.2f} " if pending_face is not None else "(无特写点) "
            print(f"  k{n_done + 1:02d}: {tag}contact @ {cur_t:.2f}s")
            pending_face = None
        elif key == ord("d"):
            if pending_face is not None:
                print(f"  撤销特写点 @ {pending_face:.2f}s")
                pending_face = None
            elif kicks:
                last = kicks.pop()
                tag = f"f@{last['face_closeup_s']:.2f} " if last["face_closeup_s"] is not None else ""
                print(f"  撤销整踢 k{len(kicks) + 1:02d}: {tag}contact @ {last['contact_time_s']:.2f}s")
        elif key in (81, 2, ord("a")):
            frame_idx = max(0, frame_idx - int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            print(f"  快退至 {frame_idx / fps:.2f}s")
        elif key in (83, 3, ord("e")):
            frame_idx = min(n_frames - 1, frame_idx + int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            print(f"  快进至 {frame_idx / fps:.2f}s")
        elif key == ord("p"):
            paused = not paused
            print(f"  {'暂停' if paused else '继续'}")

    if pending_face is not None:
        print(f"提示: 末尾有一个未完成的特写点（@{pending_face:.2f}s），已丢弃；如需保留请重打")
    cap.release()
    cv2.destroyAllWindows()
    save(mark_path, kicks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
