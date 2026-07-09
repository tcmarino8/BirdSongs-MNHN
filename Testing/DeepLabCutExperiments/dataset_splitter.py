from __future__ import annotations

"""Dataset splitting utilities for XROMM DeepLabCut experiments.

FAIR-oriented design notes:
- Findable: writes per-trial metadata with deterministic split indices.
- Accessible: uses plain CSV/JSON and folder structures.
- Interoperable: keeps coordinate table format unchanged.
- Reusable: deterministic random seed, explicit version fields, and summary manifest.
"""

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_METADATA_NAME = "split_metadata.json"


@dataclass
class TrialSplitResult:
    trial_name: str
    trial_path: Path
    available_frames: int
    requested_test_frames: int
    selected_test_frames: int
    selected_train_frames: int
    skipped: bool
    reason: str = ""



def _list_images_sorted(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        idx = int(digits) if digits else -1
        return idx, path.name.lower()

    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=sort_key,
    )



def _write_subset_csv(df: pd.DataFrame, indices: list[int], out_csv: Path) -> None:
    subset = df.iloc[indices].copy()
    subset.to_csv(out_csv, index=False)



def _copy_subset_images(images: list[Path], indices: list[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        shutil.copy2(images[idx], out_dir / images[idx].name)



def _trial_split_metadata(
    *,
    trial_name: str,
    trial_path: Path,
    seed: int,
    test_frame_count_requested: int,
    available_frames: int,
    test_indices: list[int],
    train_indices: list[int],
    n_cam1_images: int,
    n_cam2_images: int,
    n_csv_rows: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "created_unix": int(time.time()),
        "trial_name": trial_name,
        "trial_path": str(trial_path),
        "seed": int(seed),
        "test_frame_count_requested": int(test_frame_count_requested),
        "available_frames_used": int(available_frames),
        "cam1_image_count": int(n_cam1_images),
        "cam2_image_count": int(n_cam2_images),
        "csv_row_count": int(n_csv_rows),
        "test_indices": [int(x) for x in test_indices],
        "train_indices": [int(x) for x in train_indices],
    }



def split_trial_complete_set(
    trial_dir: Path,
    *,
    test_frame_count: int = 300,
    seed: int = 42,
    overwrite: bool = True,
) -> TrialSplitResult:
    """Split one trial folder containing CompleteSet into train/test subsets.

    Expected trial structure:
        trial_dir/
            CompleteSet/
                Cam1/
                Cam2/
                LabeledBodyPartsCoordinates.csv

    Output structure:
        trial_dir/
            train/
                Cam1/
                Cam2/
                LabeledBodyPartsCoordinates.csv
            test/
                Cam1/
                Cam2/
                LabeledBodyPartsCoordinates.csv
            split_metadata.json
    """
    trial_dir = Path(trial_dir)
    complete_set = trial_dir / "CompleteSet"
    cam1_dir = complete_set / "Cam1"
    cam2_dir = complete_set / "Cam2"
    labels_csv = complete_set / "LabeledBodyPartsCoordinates.csv"

    if not complete_set.exists():
        return TrialSplitResult(
            trial_name=trial_dir.name,
            trial_path=trial_dir,
            available_frames=0,
            requested_test_frames=int(test_frame_count),
            selected_test_frames=0,
            selected_train_frames=0,
            skipped=True,
            reason="Missing CompleteSet folder",
        )

    if not labels_csv.exists():
        return TrialSplitResult(
            trial_name=trial_dir.name,
            trial_path=trial_dir,
            available_frames=0,
            requested_test_frames=int(test_frame_count),
            selected_test_frames=0,
            selected_train_frames=0,
            skipped=True,
            reason="Missing LabeledBodyPartsCoordinates.csv",
        )

    cam1_images = _list_images_sorted(cam1_dir)
    cam2_images = _list_images_sorted(cam2_dir)
    labels_df = pd.read_csv(labels_csv)

    n_cam1 = len(cam1_images)
    n_cam2 = len(cam2_images)
    n_rows = len(labels_df)
    available = min(n_cam1, n_cam2, n_rows)

    if available <= 1:
        return TrialSplitResult(
            trial_name=trial_dir.name,
            trial_path=trial_dir,
            available_frames=available,
            requested_test_frames=int(test_frame_count),
            selected_test_frames=0,
            selected_train_frames=0,
            skipped=True,
            reason="Not enough aligned frames across Cam1/Cam2/CSV",
        )

    requested = int(test_frame_count)
    selected_test_n = min(requested, available)

    rng = random.Random(int(seed))
    all_indices = list(range(available))
    test_indices = sorted(rng.sample(all_indices, selected_test_n))
    test_index_set = set(test_indices)
    train_indices = [idx for idx in all_indices if idx not in test_index_set]

    train_root = trial_dir / "train"
    test_root = trial_dir / "test"

    if overwrite:
        for path in (train_root, test_root):
            if path.exists():
                shutil.rmtree(path)

    _copy_subset_images(cam1_images, train_indices, train_root / "Cam1")
    _copy_subset_images(cam2_images, train_indices, train_root / "Cam2")
    _write_subset_csv(labels_df, train_indices, train_root / "LabeledBodyPartsCoordinates.csv")

    _copy_subset_images(cam1_images, test_indices, test_root / "Cam1")
    _copy_subset_images(cam2_images, test_indices, test_root / "Cam2")
    _write_subset_csv(labels_df, test_indices, test_root / "LabeledBodyPartsCoordinates.csv")

    metadata = _trial_split_metadata(
        trial_name=trial_dir.name,
        trial_path=trial_dir,
        seed=int(seed),
        test_frame_count_requested=requested,
        available_frames=available,
        test_indices=test_indices,
        train_indices=train_indices,
        n_cam1_images=n_cam1,
        n_cam2_images=n_cam2,
        n_csv_rows=n_rows,
    )
    (trial_dir / SPLIT_METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrialSplitResult(
        trial_name=trial_dir.name,
        trial_path=trial_dir,
        available_frames=available,
        requested_test_frames=requested,
        selected_test_frames=len(test_indices),
        selected_train_frames=len(train_indices),
        skipped=False,
    )



def split_all_trials(
    data_root: str | Path,
    *,
    test_frame_count: int = 300,
    seed: int = 42,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Split all immediate trial folders under data_root and write a manifest."""
    root = Path(data_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    results: list[TrialSplitResult] = []
    for trial_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        result = split_trial_complete_set(
            trial_dir,
            test_frame_count=int(test_frame_count),
            seed=int(seed),
            overwrite=bool(overwrite),
        )
        results.append(result)

    rows = [
        {
            "trial_name": r.trial_name,
            "trial_path": str(r.trial_path),
            "available_frames": int(r.available_frames),
            "requested_test_frames": int(r.requested_test_frames),
            "selected_test_frames": int(r.selected_test_frames),
            "selected_train_frames": int(r.selected_train_frames),
            "skipped": bool(r.skipped),
            "reason": r.reason,
        }
        for r in results
    ]
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(root / "split_manifest.csv", index=False)
    return manifest_df
