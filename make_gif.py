#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
from PIL import Image


def natural_key(path: Path):
    """
    例:
    img_2.png, img_10.png を自然順で並べる
    """
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r"(\d+)", path.name)
    ]


def collect_png_files(input_dir: Path, pattern: str):
    files = sorted(input_dir.glob(pattern), key=natural_key)
    return [f for f in files if f.suffix.lower() == ".png"]


def make_gif(
    input_dir: Path,
    pattern: str,
    output_dir: Path,
    output_name: str,
    max_frames: int | None,
    duration: int,
    loop: int,
):
    png_files = collect_png_files(input_dir, pattern)

    if not png_files:
        raise FileNotFoundError(
            f"No PNG files found in '{input_dir}' matching pattern '{pattern}'"
        )

    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--frames must be a positive integer")
        png_files = png_files[:max_frames]

    if len(png_files) == 0:
        raise ValueError("No frames selected after applying --frames")

    frames = []
    for file in png_files:
        img = Image.open(file).convert("RGB")
        frames.append(img)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False,
    )

    print("GIF saved successfully")
    print(f"Input dir    : {input_dir}")
    print(f"Pattern      : {pattern}")
    print(f"Frame count  : {len(frames)}")
    print(f"Duration(ms) : {duration}")
    print(f"Output dir   : {output_dir}")
    print(f"Output name  : {output_name}")
    print(f"Output path  : {output_path}")
    print("Used files:")
    for f in png_files:
        print(f"  {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Create GIF from PNG files"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing PNG files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Filename pattern to match, e.g. 'frame_*.png'"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Maximum number of frames to use from the beginning"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=200,
        help="Frame duration in milliseconds"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save the GIF"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="output.gif",
        help="Output GIF file name"
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Loop count for GIF (0 means infinite)"
    )

    args = parser.parse_args()

    make_gif(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_dir=args.output_dir,
        output_name=args.output_name,
        max_frames=args.frames,
        duration=args.duration,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()