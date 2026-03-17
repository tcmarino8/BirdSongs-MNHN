"""Batch-convert image folders between JPG and TIF formats.

Example
-------
>>> python jpg_to_tiff.py Cam1_Img00UND_new Cam2_Img00UND_new
Converted Cam1_Img00UND_new -> Cam1_Img00UND_new_tif
Converted Cam2_Img00UND_new -> Cam2_Img00UND_new_tif

>>> python jpg_to_tiff.py --to jpg Cam1_Img00UND_new_tif
Converted Cam1_Img00UND_new_tif -> Cam1_Img00UND_new_tif_jpg
"""

import argparse
import pathlib
from PIL import Image

def convert_folder(input_dir: pathlib.Path, target_format: str = "auto") -> pathlib.Path:
    """Convert image files in ``input_dir`` and return the new output folder."""
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a directory")

    format_pairs = {
        "tif": (["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"], ".tif", "_tif", "TIFF"),
        "jpg": (["*.tif", "*.tiff", "*.TIF", "*.TIFF"], ".jpg", "_jpg", "JPEG"),
    }

    if target_format == "auto":
        jpg_count = sum(1 for p in input_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg"})
        tif_count = sum(1 for p in input_dir.glob("*") if p.suffix.lower() in {".tif", ".tiff"})
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
            with Image.open(src_path) as img:
                out_path = output_dir / (src_path.stem + out_suffix)
                if pil_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                img.save(out_path, format=pil_format)
                converted_count += 1

    if converted_count == 0:
        raise ValueError(f"No matching files found in {input_dir} for target format '{target_format}'")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert image folders between JPG and TIF formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:\n  python jpg_to_tiff.py Cam1_Img00UND_new\n  python jpg_to_tiff.py --to jpg Cam1_Img00UND_new_tif""",
    )
    parser.add_argument(
        "folders",
        nargs="+",
        type=pathlib.Path,
        help="Paths to folders containing JPG/JPEG or TIF/TIFF files.",
    )
    parser.add_argument(
        "--to",
        choices=["auto", "tif", "jpg"],
        default="auto",
        help="Target format. 'auto' infers direction from folder contents.",
    )
    args = parser.parse_args()
    for folder in args.folders:
        out = convert_folder(folder, target_format=args.to)
        print(f"Converted {folder} -> {out}")


if __name__ == "__main__":
    main()