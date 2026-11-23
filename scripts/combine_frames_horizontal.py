#!/usr/bin/env python
"""将一个目录中的多张 frame 横向拼接为一张图片。

项目根目录下运行示例::

    python scripts/combine_frames_horizontal.py
    python scripts/combine_frames_horizontal.py \
        --input-dir data/faces_multiframe/m001/k02 \
        --output data/faces_multiframe/m001/k02/frames_horizontal.jpg

默认只读取 ``frame_*.jpg``，并按文件名排序。不同尺寸的图片会保持宽高比，
按最高图片的高度等比例缩放，图片之间的间距使用 ``--background`` 指定的颜色填充。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "faces_multiframe" / "m001" / "k02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="包含 frame 图片的目录（默认：data/faces_multiframe/m001/k02）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出路径（默认：输入目录下的 frames_horizontal.jpg）",
    )
    parser.add_argument(
        "--pattern",
        default="frame_*.jpg",
        help="输入文件匹配模式（默认：frame_*.jpg）",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="相邻图片之间的间距，单位为像素（默认：0）",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="填充颜色，支持 white、black 或 #RRGGBB（默认：white）",
    )
    return parser.parse_args()


def combine_frames(
    input_dir: Path,
    output_path: Path,
    pattern: str,
    gap: int,
    background: str,
) -> tuple[int, tuple[int, int]]:
    if gap < 0:
        raise ValueError("--gap 不能为负数")

    frame_paths = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not frame_paths:
        raise FileNotFoundError(f"目录中没有匹配 {pattern!r} 的图片：{input_dir}")

    try:
        background_rgb = ImageColor.getrgb(background)
    except ValueError as exc:
        raise ValueError(f"无法识别背景色：{background!r}") from exc

    images = []
    try:
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                images.append(image.convert("RGB"))

        target_height = max(image.height for image in images)
        normalized = []
        for image in images:
            if image.height == target_height:
                normalized.append(image)
                continue

            width = round(image.width * target_height / image.height)
            normalized.append(image.resize((width, target_height), Image.Resampling.LANCZOS))

        total_width = sum(image.width for image in normalized) + gap * (len(normalized) - 1)
        canvas = Image.new("RGB", (total_width, target_height), background_rgb)

        x = 0
        for image in normalized:
            y = (target_height - image.height) // 2
            canvas.paste(image, (x, y))
            x += image.width + gap

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return len(frame_paths), canvas.size
    finally:
        for image in images:
            image.close()


def main() -> int:
    args = parse_args()
    output_path = args.output or args.input_dir / "frames_horizontal.jpg"

    try:
        count, size = combine_frames(
            input_dir=args.input_dir,
            output_path=output_path,
            pattern=args.pattern,
            gap=args.gap,
            background=args.background,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"错误：{exc}")
        return 1

    print(f"已拼接 {count} 张图片：{output_path} ({size[0]}x{size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
