"""Batch-convert folders of JPG images to TIF format.

Example
-------
>>> python jpg_to_tiff.py Cam1_Img00UND_new Cam2_Img00UND_new
Converted Cam1_Img00UND_new -> Cam1_Img00UND_new_tif
Converted Cam2_Img00UND_new -> Cam2_Img00UND_new_tif
"""

import argparse
import pathlib
from PIL import Image

def convert_folder(input_dir: pathlib.Path) -> pathlib.Path:
    """Convert all JPG images in ``input_dir`` and return the new TIF folder."""
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a directory")
    output_dir = input_dir.with_name(f"{input_dir.name}_tif")
    output_dir.mkdir(exist_ok=True)
    for jpg_path in input_dir.glob("*.jpg"):
        with Image.open(jpg_path) as img:
            tif_path = output_dir / (jpg_path.stem + ".tif")
            img.save(tif_path)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JPG folders to corresponding TIF folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:\n  python jpg_to_tiff.py Cam1_Img00UND_new""",
    )
    parser.add_argument(
        "folders",
        nargs="+",
        type=pathlib.Path,
        help="Paths to folders containing JPG files.",
    )
    args = parser.parse_args()
    for folder in args.folders:
        out = convert_folder(folder)
        print(f"Converted {folder} -> {out}")


if __name__ == "__main__":
    main()