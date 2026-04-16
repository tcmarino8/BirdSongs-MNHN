"""Utilities for converting image folders and TIFF frame stacks.

This script provides two command-line workflows:

1. Convert folders of still images between JPG/JPEG and TIF/TIFF.
2. Convert a folder of TIFF frames into an AVI video.

The image converter creates a sibling output folder using a suffix that reflects
the target format:

- ``*_tif`` for JPG/JPEG -> TIFF conversions.
- ``*_jpg`` for TIFF -> JPG conversions.

The TIFF-stack converter is intended for frame folders exported from imaging or
tracking pipelines where each TIFF file is one video frame.

Examples
--------
Convert JPG folders to TIFF:
    python data-converters.py images Cam1_Img00UND_new Cam2_Img00UND_new

Convert TIFF folders back to JPG:
    python data-converters.py images --to jpg Cam1_Img00UND_new_tif

Let the script infer the direction from folder contents:
    python data-converters.py images some_image_folder

Convert a TIFF stack to AVI with numeric frame ordering:
    python data-converters.py tiff-stack "path/to/frames" "path/to/output.avi" --fps 750

Disable numeric sorting and use plain lexical filename sorting:
    python data-converters.py tiff-stack frames_dir output.avi --no-sort-numeric
"""

from __future__ import annotations

import argparse
import pathlib
import re

import cv2
from PIL import Image


def convert_folder(input_dir: pathlib.Path, target_format: str = "auto") -> pathlib.Path:
    """Convert a folder of still images and return the output directory.

    Parameters
    ----------
    input_dir
        Directory containing only source images for one conversion direction.
    target_format
        Conversion target. Use ``"tif"`` for JPG/JPEG to TIFF, ``"jpg"`` for
        TIFF to JPG, or ``"auto"`` to infer the direction from the folder's
        contents.

    Returns
    -------
    pathlib.Path
        The output directory created next to ``input_dir``.

    Raises
    ------
    ValueError
        If the folder does not exist, contains mixed source formats, contains no
        supported files, or contains no files matching the requested conversion.
    """
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a directory")

    format_pairs = {
        "tif": (["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"], ".tif", "_tif", "TIFF"),
        "jpg": (["*.tif", "*.tiff", "*.TIF", "*.TIFF"], ".jpg", "_jpg", "JPEG"),
    }

    if target_format == "auto":
        jpg_count = sum(1 for path in input_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg"})
        tif_count = sum(1 for path in input_dir.glob("*") if path.suffix.lower() in {".tif", ".tiff"})
        if jpg_count and not tif_count:
            target_format = "tif"
        elif tif_count and not jpg_count:
            target_format = "jpg"
        elif jpg_count and tif_count:
            raise ValueError(
                f"{input_dir} contains both JPG and TIF files. Use --to tif or --to jpg explicitly."
            )
        else:
            raise ValueError(f"No JPG/JPEG/TIF/TIFF files found in {input_dir}")

    patterns, out_suffix, folder_suffix, pil_format = format_pairs[target_format]
    output_dir = input_dir.with_name(f"{input_dir.name}{folder_suffix}")
    output_dir.mkdir(exist_ok=True)

    converted_count = 0
    for pattern in patterns:
        for src_path in input_dir.glob(pattern):
            with Image.open(src_path) as image:
                out_path = output_dir / f"{src_path.stem}{out_suffix}"
                if pil_format == "JPEG" and image.mode in ("RGBA", "LA", "P"):
                    image = image.convert("RGB")
                image.save(out_path, format=pil_format)
                converted_count += 1

    if converted_count == 0:
        raise ValueError(f"No matching files found in {input_dir} for target format '{target_format}'")

    return output_dir


def _to_uint8_bgr(frame):
    """Normalize image data to the format expected by OpenCV video writing.

    OpenCV ``VideoWriter`` expects 8-bit BGR frames. This helper converts
    grayscale, BGRA, and non-8-bit TIFF image arrays into that format.
    """
    if frame is None:
        raise ValueError("Encountered unreadable frame.")

    if frame.dtype != "uint8":
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame

    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def tiff_stack_to_avi(
    input_folder: pathlib.Path,
    output_path: pathlib.Path,
    fps: int = 500,
    fourcc: str = "MJPG",
    sort_numeric: bool = True,
) -> dict[str, object]:
    """Convert a folder of TIFF frames into an AVI file.

    Parameters
    ----------
    input_folder
        Folder containing ``.tif`` or ``.tiff`` frames.
    output_path
        Full output path for the AVI file.
    fps
        Frames per second to encode into the AVI.
    fourcc
        FourCC codec string passed to OpenCV, such as ``"MJPG"`` or ``"XVID"``.
    sort_numeric
        When ``True``, sort frames by the last numeric token in each filename.
        This is useful for names like ``frame_1.tif``, ``frame_2.tif``,
        ``frame_10.tif``. When ``False``, use default lexical sorting.

    Returns
    -------
    dict[str, object]
        A summary containing the output path, frame count, codec, fps, and frame
        dimensions.

    Raises
    ------
    FileNotFoundError
        If the input folder does not exist or contains no TIFF files.
    RuntimeError
        If OpenCV cannot open the requested output video for writing.
    ValueError
        If a frame cannot be read or if frame sizes are inconsistent.
    """
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    tif_files = [
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    if not tif_files:
        raise FileNotFoundError(f"No .tif or .tiff files found in: {input_folder}")

    if sort_numeric:
        def sort_key(path: pathlib.Path) -> tuple[int, int | str, str]:
            match = re.search(r"(\d+)(?!.*\d)", path.name)
            if match:
                return (0, int(match.group(1)), path.name.lower())
            return (1, path.name.lower(), path.name.lower())

        tif_files = sorted(tif_files, key=sort_key)
    else:
        tif_files = sorted(tif_files)

    first_frame = _to_uint8_bgr(cv2.imread(str(tif_files[0]), cv2.IMREAD_UNCHANGED))
    height, width = first_frame.shape[:2]

    if output_path.parent != pathlib.Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc),
        float(fps),
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for output: {output_path}")

    frames_written = 0
    try:
        for tif_path in tif_files:
            frame = _to_uint8_bgr(cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED))
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    "Frame size mismatch in %s. Expected %sx%s, got %sx%s."
                    % (tif_path, width, height, frame.shape[1], frame.shape[0])
                )
            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()

    return {
        "output_path": str(output_path),
        "frames_written": frames_written,
        "fps": fps,
        "codec": fourcc,
        "width": width,
        "height": height,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface parser.

    The CLI is split into subcommands so image-folder conversion and TIFF-stack
    video creation remain explicit and discoverable in ``--help`` output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert folders of still images between JPG/JPEG and TIF/TIFF, "
            "or convert a TIFF frame stack into an AVI video."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data-converters.py images Cam1_Img00UND_new\n"
            "  python data-converters.py images --to jpg Cam1_Img00UND_new_tif\n"
            "  python data-converters.py images some_folder\n"
            "  python data-converters.py tiff-stack frames_dir output.avi --fps 750\n"
            "  python data-converters.py tiff-stack frames_dir output.avi --fourcc XVID"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser(
        "images",
        help="Convert folders between JPG/JPEG and TIF/TIFF.",
        description=(
            "Convert one or more folders of still images. By default, the script "
            "inspects each folder and infers the direction when only one supported "
            "source format is present."
        ),
    )
    image_parser.add_argument(
        "folders",
        nargs="+",
        type=pathlib.Path,
        help="Paths to folders containing JPG/JPEG or TIF/TIFF files.",
    )
    image_parser.add_argument(
        "--to",
        choices=["auto", "tif", "jpg"],
        default="auto",
        help=(
            "Target format. Use 'tif' for JPG/JPEG to TIFF, 'jpg' for TIFF to JPG, "
            "or 'auto' to infer the direction from the folder contents."
        ),
    )

    stack_parser = subparsers.add_parser(
        "tiff-stack",
        help="Convert a TIFF frame folder into an AVI file.",
        description=(
            "Build an AVI from a directory of TIFF frames. Frames are sorted by the "
            "last number in each filename by default, which is usually correct for "
            "camera-exported frame sequences."
        ),
    )
    stack_parser.add_argument(
        "input_folder",
        type=pathlib.Path,
        help="Folder containing TIFF frames to encode in order.",
    )
    stack_parser.add_argument(
        "output_path",
        type=pathlib.Path,
        help="Destination AVI path. Parent folders are created automatically.",
    )
    stack_parser.add_argument(
        "--fps",
        type=int,
        default=500,
        help="Frames per second for the AVI output. Default: 500.",
    )
    stack_parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="FourCC codec string for OpenCV VideoWriter. Default: MJPG.",
    )
    stack_parser.add_argument(
        "--no-sort-numeric",
        action="store_true",
        help="Disable numeric filename sorting and use standard lexical ordering instead.",
    )

    return parser


def main() -> None:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "images":
        for folder in args.folders:
            output_dir = convert_folder(folder, target_format=args.to)
            print(f"Converted {folder} -> {output_dir}")
        return

    summary = tiff_stack_to_avi(
        input_folder=args.input_folder,
        output_path=args.output_path,
        fps=args.fps,
        fourcc=args.fourcc,
        sort_numeric=not args.no_sort_numeric,
    )
    print("TIFF stack conversion complete")
    print(summary)


if __name__ == "__main__":
    main()