#!/usr/bin/env python
"""人工截帧脚本：播放视频，看到主罚球员的合格面部特写时按空格直接截取该帧。

背景（2026-09-16，替代自动选帧）:
    自动 pipeline（切片->检测->选帧）在 360p 转播上会截到门将/队友/观众的人脸，
    人工 review 成本反而高。既然标注员本来就要通看全场，不如在看到目标脸的瞬间
    直接打点截帧——身份判断由人完成，脚本只负责裁剪和记录。

用法（项目根目录）:
    python scripts/capture_face.py --match m004

操作:
    空格       = 截取当前帧的人脸 -> data/faces/<match>/kNN.jpg（同踢再按即覆盖）
    d          = 撤销上一次截取（删除文件并移除记录）
    a / e      = 快退/快进 5 秒
    p          = 暂停/继续
    q / ESC    = 退出

踢次自动对齐:
    读取 marker.py 的打点文件（contact_time_s），当前踢 = 下一个未到的触球时刻。
    HUD 显示当前踢号与距触球的倒计时。
    防泄漏护栏: 距触球 <0.5s 或已过触球时按空格会被拒绝并警告——
    截帧时刻严格早于触球前 0.5s 由工具强制保证。

无 marker 文件时:
    退化为顺序编号 k01, k02...（无防泄漏护栏，建议先跑 marker.py）。

输出:
    data/faces/<match>/kNN.jpg                 人脸 crop（YuNet 检测框 + 25% 余量）
    data/metadata/face_frames.csv              截帧记录：时刻、脸宽、置信度（人工流程的审计表）

检测器: YuNet（models/detection/，与 select_face_frames.py 共用）。
        当前帧未检出人脸时保存全帧并在 HUD/终端警告，由人工决定删（d）或保留。
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
FACES_DIR = PROJECT_ROOT / "data" / "faces"
FRAMES_LOG = PROJECT_ROOT / "data" / "metadata" / "face_frames.csv"
YUNET_PATH = PROJECT_ROOT / "models" / "detection" / "face_detection_yunet_2023mar.onnx"

MAX_WIN_W = 1280
CROP_MARGIN = 0.25  # 人脸框四周余量比例


def load_contacts(mark_path: Path) -> list[float] | None:
    """读 marker 文件的触球时刻；无文件/无有效行返回 None（顺序编号模式）。"""
    if not mark_path.exists():
        return None
    with open(mark_path, newline="", encoding="utf-8") as f:
        rows = [float(r["contact_time_s"]) for r in csv.DictReader(f) if r.get("contact_time_s")]
    return sorted(rows) if rows else None


def current_kick(contacts: list[float] | None, t: float) -> tuple[int, float | None]:
    """当前踢号（1-based）与本次触球时刻。t 之后最近的触球属于当前踢。"""
    if contacts is None:
        return 1, None
    for i, c in enumerate(contacts, 1):
        if t < c:
            return i, c
    return len(contacts) + 1, None  # 最后打点之后（领奖/回放段）


def detect_face(detector, frame):
    """返回 (box, conf) 或 None。box = (x, y, w, h)。"""
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    if faces is None or not len(faces):
        return None
    best = max(faces, key=lambda f: f[-1])
    return best[:4].astype(int), float(best[-1])


def crop_face(frame, box) -> "cv2.Mat":
    x, y, w, h = box
    mw, mh = int(w * CROP_MARGIN), int(h * CROP_MARGIN)
    x0, y0 = max(0, x - mw), max(0, y - mh)
    x1, y1 = min(frame.shape[1], x + w + mw), min(frame.shape[0], y + h + mh)
    return frame[y0:y1, x0:x1]


def append_log(row: dict) -> None:
    """即时追加审计行（崩溃安全）。文件已存在且无表头则补写。"""
    new_file = not FRAMES_LOG.exists()
    with open(FRAMES_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["match_id", "kick_seq", "frame_time_s",
                                          "face_w_px", "conf", "crop_path"])
        if new_file:
            w.writeheader()
        w.writerow(row)


def load_log(match_id: str) -> list[dict]:
    if not FRAMES_LOG.exists():
        return []
    with open(FRAMES_LOG, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["match_id"] == match_id]


def rewrite_log(match_id: str, remaining: list[dict]) -> None:
    """把本场的行重写为 remaining（撤销时用），其他场的行原样保留。"""
    others = []
    if FRAMES_LOG.exists():
        with open(FRAMES_LOG, newline="", encoding="utf-8") as f:
            others = [r for r in csv.DictReader(f) if r["match_id"] != match_id]
    with open(FRAMES_LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["match_id", "kick_seq", "frame_time_s",
                                          "face_w_px", "conf", "crop_path"])
        w.writeheader()
        w.writerows(others + remaining)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True, help="比赛编号，如 m004")
    ap.add_argument("--speed", type=float, default=1.0, help="播放速度倍率（默认 1.0）")
    args = ap.parse_args()

    if not YUNET_PATH.exists():
        print(f"缺检测模型: {YUNET_PATH}", file=sys.stderr)
        return 1
    detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.6)

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
    contacts = load_contacts(MARK_DIR / f"{args.match}_markers.csv")
    if contacts is None:
        print("!! 无 marker 文件：踢次按顺序编号，无防泄漏护栏（建议先跑 marker.py）")
    else:
        print(f"载入 {len(contacts)} 个触球点，踢次自动对齐")

    captures = load_log(args.match)  # 断点续作：已截的踢显示在 HUD
    captured_kicks = {r["kick_seq"] for r in captures}
    print(f"已有截帧 {len(captures)} 张: {sorted(captured_kicks)}")

    out_dir = FACES_DIR / args.match
    out_dir.mkdir(parents=True, exist_ok=True)

    print("空格=截帧  d=撤销  a/e=±5s  p=暂停  q=退出\n")
    frame_idx = 0
    frame_delay_ms = max(1, int(1000 / (fps * args.speed)))
    paused = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("视频播放完毕")
            break
        cur_t = frame_idx / fps
        kick_no, contact_t = current_kick(contacts, cur_t)
        kick_seq = f"k{kick_no:02d}"
        remain = (contact_t - cur_t) if contact_t is not None else None

        h, w = frame.shape[:2]
        scale = min(1.0, MAX_WIN_W / w)
        disp = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

        if remain is None:
            guard = "END/回放段"
        elif remain < 0.5:
            guard = "!! 泄漏风险区"
        else:
            guard = f"距触球 {remain:5.1f}s"
        done_mark = "*" if kick_seq in captured_kicks else " "
        hud = f"{args.match} t={cur_t:7.2f}s {kick_seq}{done_mark} [{guard}] caps={len(captures)}"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
        cv2.imshow("capture_face", disp)

        key = cv2.waitKey(frame_delay_ms if not paused else 30) & 0xFF
        frame_idx += 1

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if remain is not None and remain < 0.5:
                print(f"  !! 拒绝截帧: 距触球仅 {remain:.2f}s（<0.5s 防泄漏边界）"
                      f"——这一踢请回退用更早的特写帧")
                continue
            hit = detect_face(detector, frame)
            if hit is None:
                out_path = out_dir / f"{kick_seq}_FULLFRAME.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"  {kick_seq}: !! 当前帧未检出人脸，已存全帧 {out_path.name}，人工决定删(d)或自行裁剪")
                append_log({"match_id": args.match, "kick_seq": kick_seq,
                            "frame_time_s": f"{cur_t:.2f}", "face_w_px": 0, "conf": "",
                            "crop_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")})
            else:
                box, conf = hit
                crop = crop_face(frame, box)
                out_path = out_dir / f"{kick_seq}.jpg"
                cv2.imwrite(str(out_path), crop)
                overwrite = kick_seq in captured_kicks
                print(f"  {kick_seq}: 截帧 @ {cur_t:.2f}s (脸宽 {box[2]}px, conf {conf:.2f})"
                      f"{' [覆盖]' if overwrite else ''}")
                captures = [r for r in captures if r["kick_seq"] != kick_seq]
                append_log({"match_id": args.match, "kick_seq": kick_seq,
                            "frame_time_s": f"{cur_t:.2f}", "face_w_px": int(box[2]),
                            "conf": f"{conf:.2f}",
                            "crop_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")})
            captured_kicks.add(kick_seq)
        elif key == ord("d"):
            if captures:
                last = captures.pop()
                p = PROJECT_ROOT / last["crop_path"]
                if p.exists():
                    p.unlink()
                rewrite_log(args.match, captures)
                captured_kicks.discard(last["kick_seq"])
                print(f"  撤销 {last['kick_seq']} @ {last['frame_time_s']}s（已删文件）")
            else:
                print("  无可撤销的截帧")
        elif key in (81, 2, ord("a")):
            frame_idx = max(0, frame_idx - int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key in (83, 3, ord("e")):
            frame_idx = min(n_frames - 1, frame_idx + int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n本场共 {len(captures)} 张截帧 -> {out_dir.relative_to(PROJECT_ROOT)}/")
    n_kicks = len(contacts) if contacts else "?"
    missing = [f"k{i:02d}" for i in range(1, n_kicks + 1) if f"k{i:02d}" not in captured_kicks] \
        if contacts else []
    if missing:
        print(f"未截的踢: {', '.join(missing)}（这些踢将无 face_path，填 exclusion_reason）")
    print(f"审计: {FRAMES_LOG.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
