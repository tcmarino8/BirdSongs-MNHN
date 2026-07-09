from __future__ import annotations

import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BIRD_CODE_MAP = {
    "DB": "DavidBowie",
    "Tulio": "Tulio",
}


def _ensure_codetesting_on_syspath() -> Path:
    """Make sibling Code-Testing importable for DLCsupport/xrommtools modules."""
    repo_root = Path(__file__).resolve().parents[2]
    code_testing = repo_root / "Code-Testing"
    if str(code_testing) not in sys.path:
        sys.path.insert(0, str(code_testing))
    return code_testing


def _load_dlcsupport() -> Any:
    _ensure_codetesting_on_syspath()
    import DLCsupport as dlcs

    return dlcs


def _candidate_dummy_videos(bird: str, trial_num: int) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    tnum = int(trial_num)
    return [
        repo_root / "ProcessingData" / bird / f"Trial{tnum}" / "Cam1.avi",
        repo_root / "DeepLabCut2" / bird / f"TrainingData_T{tnum}" / "cam1_stack.avi",
        repo_root / "DeepLabCut" / bird / f"TrainingData_T{tnum}" / "Cam1.avi",
        repo_root / "DeepLabCut" / bird / f"TrainingData_T{tnum}" / "db_cam1.avi",
    ]


def _resolve_fallback_dummy_video(bird: str, trial_num: int) -> Path | None:
    for candidate in _candidate_dummy_videos(bird=bird, trial_num=int(trial_num)):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _list_images_sorted(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    def _key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)(?!.*\d)", path.stem)
        idx = int(match.group(1)) if match else -1
        return idx, path.name.lower()

    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=_key,
    )


def _parse_bodypart_columns(df: pd.DataFrame) -> dict[str, dict[str, tuple[str, str]]]:
    bodyparts: dict[str, dict[str, tuple[str, str]]] = {}
    cols = list(df.columns)
    for col in cols:
        m = re.match(r"(?P<bodypart>.+)_cam(?P<cam>[12])_X$", str(col))
        if m is None:
            continue
        bodypart = str(m.group("bodypart"))
        cam = f"cam{m.group('cam')}"
        y_col = f"{bodypart}_{cam}_Y"
        if y_col not in cols:
            continue
        bodyparts.setdefault(bodypart, {})[cam] = (str(col), str(y_col))
    return bodyparts


def _frame_displacement(df: pd.DataFrame, bodyparts: dict[str, dict[str, tuple[str, str]]], camera: str, start_frame: int, end_frame: int) -> float | None:
    if start_frame < 0 or end_frame < 0 or start_frame >= len(df) or end_frame >= len(df):
        return None

    dists: list[float] = []
    for _, cams in bodyparts.items():
        cols = cams.get(camera)
        if cols is None:
            continue
        x_col, y_col = cols

        x0 = pd.to_numeric(df.at[int(start_frame), x_col], errors="coerce")
        y0 = pd.to_numeric(df.at[int(start_frame), y_col], errors="coerce")
        x1 = pd.to_numeric(df.at[int(end_frame), x_col], errors="coerce")
        y1 = pd.to_numeric(df.at[int(end_frame), y_col], errors="coerce")
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
            continue
        dists.append(float(np.hypot(float(x1) - float(x0), float(y1) - float(y0))))

    if not dists:
        return None
    return float(np.mean(dists))


def _allocate_counts(weights: list[float], total_count: int) -> list[int]:
    if total_count <= 0 or not weights:
        return [0 for _ in weights]

    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(np.sum(w)) <= 0:
        w = np.ones(len(weights), dtype=float)
    w = w / float(np.sum(w))

    raw = w * int(total_count)
    counts = np.floor(raw).astype(int)
    remaining = int(total_count - np.sum(counts))
    if remaining > 0:
        remainders = raw - counts
        for idx in np.argsort(-remainders)[:remaining]:
            counts[int(idx)] += 1
    return [int(x) for x in counts.tolist()]


def _build_zone_specs(frame_limit: int, zone_size: int = 150) -> list[dict[str, Any]]:
    if frame_limit <= 0:
        return []

    starts = list(range(0, int(frame_limit), int(zone_size)))
    zones: list[dict[str, Any]] = []
    for idx, start in enumerate(starts):
        end = min(int(frame_limit), int(start + zone_size))
        zones.append(
            {
                "zone_index": int(idx),
                "start": int(start),
                "end": int(end),
                "label": f"Z{idx + 1} {start + 1}-{end}",
            }
        )
    return zones


def _select_random_indices(frame_count: int, count: int, rng: np.random.Generator) -> list[int]:
    count = min(int(count), int(frame_count))
    if count <= 0:
        return []
    picks = np.sort(rng.choice(np.arange(frame_count, dtype=int), size=count, replace=False))
    return [int(x) for x in picks.tolist()]


def _select_displacement_indices(
    labels_df: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    zone_size: int = 150,
) -> tuple[list[int], dict[str, Any]]:
    frame_count = int(len(labels_df))
    count = min(int(count), frame_count)
    if count <= 0 or frame_count <= 0:
        return [], {"zone_size": int(zone_size), "zone_scores": []}

    zones = _build_zone_specs(frame_limit=frame_count, zone_size=int(zone_size))
    bodyparts = _parse_bodypart_columns(labels_df)

    zone_scores: list[float] = []
    for zone in zones:
        start = int(zone["start"])
        end = int(max(start, zone["end"] - 1))
        camera_scores: list[float] = []
        for camera in ("cam1", "cam2"):
            disp = _frame_displacement(labels_df, bodyparts, camera, start, end)
            if disp is not None and np.isfinite(disp):
                camera_scores.append(float(disp))
        zone_scores.append(float(np.mean(camera_scores)) if camera_scores else 0.0)

    zone_counts = _allocate_counts(zone_scores, count)
    selected: list[int] = []
    for zone, zone_count in zip(zones, zone_counts):
        if zone_count <= 0:
            continue
        zone_frames = np.arange(int(zone["start"]), int(zone["end"]), dtype=int)
        if zone_frames.size == 0:
            continue
        pick_count = min(int(zone_count), int(zone_frames.size))
        picks = np.sort(rng.choice(zone_frames, size=pick_count, replace=False))
        selected.extend(int(x) for x in picks.tolist())

    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(frame_count, dtype=int), np.asarray(sorted(set(selected)), dtype=int), assume_unique=False)
        fill_count = min(int(count - len(selected)), int(remaining.size))
        if fill_count > 0:
            fill = np.sort(rng.choice(remaining, size=fill_count, replace=False))
            selected.extend(int(x) for x in fill.tolist())

    selected = sorted(set(selected))[:count]
    meta = {
        "zone_size": int(zone_size),
        "zone_scores": [float(x) for x in zone_scores],
        "zone_counts": [int(x) for x in zone_counts],
    }
    return selected, meta


def _ensure_dummy_video_from_images(
    cam1_images: list[Path],
    out_video: Path,
    fps: int = 100,
    fallback_video: Path | None = None,
) -> Path:
    if out_video.exists():
        return out_video
    if fallback_video is not None and fallback_video.exists() and fallback_video.is_file():
        return fallback_video
    if not cam1_images:
        raise FileNotFoundError("No Cam1 images available to create a dummy video")

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV (cv2) is required to create a dummy video for DLC project setup when no fallback .avi exists"
        ) from exc

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow (PIL) is required to create a dummy video from image stacks") from exc

    first = np.asarray(Image.open(cam1_images[0]).convert("RGB"))
    h, w = first.shape[:2]
    out_video.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"XVID"), float(fps), (int(w), int(h)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_video}")

    # Only a short clip is needed to satisfy DLC project creation.
    for image_path in cam1_images[: min(60, len(cam1_images))]:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        bgr = rgb[:, :, ::-1]
        writer.write(np.asarray(bgr, dtype=np.uint8))
    writer.release()
    return out_video


def _k_center_greedy(embeddings: Any, k: int, torch_mod: Any) -> list[int]:
    n_samples = int(embeddings.shape[0])
    if n_samples <= 0 or k <= 0:
        return []
    if k >= n_samples:
        return list(range(n_samples))

    selected = [int(torch_mod.randint(0, n_samples, (1,)).item())]
    dist = torch_mod.cdist(embeddings, embeddings[selected]).min(dim=1).values
    for _ in range(int(k) - 1):
        idx = int(torch_mod.argmax(dist).item())
        selected.append(idx)
        new_dist = torch_mod.cdist(embeddings, embeddings[[idx]]).squeeze(1)
        dist = torch_mod.minimum(dist, new_dist)
    return [int(x) for x in selected]


def _compute_dino_embeddings(image_paths: list[Path], batch_size: int = 16) -> tuple[Any, Any]:
    try:
        import torch
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("DINO selection requires torch and torchvision") from exc

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("DINO selection requires Pillow (PIL)") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    embeddings = []
    for start in range(0, len(image_paths), int(batch_size)):
        batch_paths = image_paths[start : start + int(batch_size)]
        tensors = []
        for image_path in batch_paths:
            with Image.open(image_path) as image_file:
                tensors.append(transform(image_file.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model(batch)
        embeddings.append(feats.detach().cpu())

    if not embeddings:
        raise RuntimeError("No embeddings were computed for DINO selection")

    emb = torch.cat(embeddings, dim=0)
    emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return emb, torch


def _select_dino_indices(cam1_images: list[Path], count: int, batch_size: int = 16) -> list[int]:
    frame_count = int(len(cam1_images))
    count = min(int(count), frame_count)
    if count <= 0:
        return []

    emb, torch_mod = _compute_dino_embeddings(cam1_images, batch_size=int(batch_size))
    selected = _k_center_greedy(emb, int(count), torch_mod)
    return sorted(int(x) for x in selected)


def _labels_csv_in_train(train_dir: Path) -> Path:
    preferred = train_dir / "LabeledBodyPartsCoordinates.csv"
    if preferred.exists():
        return preferred
    matches = sorted([p for p in train_dir.glob("*.csv") if p.is_file()])
    if not matches:
        raise FileNotFoundError(f"No labels CSV found in {train_dir}")
    return matches[0]


def parse_trial_folder_name(trial_folder_name: str) -> tuple[str, int]:
    """Parse folder names like DB_T15 -> (DavidBowie, 15)."""
    m = re.fullmatch(r"(?P<bird>[A-Za-z]+)_T(?P<trial>\d+)", str(trial_folder_name))
    if m is None:
        raise ValueError(f"Could not parse trial folder name: {trial_folder_name}")

    bird_token = str(m.group("bird"))
    trial_num = int(m.group("trial"))
    bird_name = BIRD_CODE_MAP.get(bird_token, bird_token)
    return bird_name, trial_num


@dataclass
class BuildRow:
    trial_folder: str
    bird: str
    trial_num: int
    method: str
    nframes: int
    epochs: int
    output_root: str
    config_path: str
    selection_count: int
    status: str
    message: str



def _prepare_dlc_dataset(
    *,
    trial_dir: Path,
    bird: str,
    trial_num: int,
    method: str,
    selected_indices: list[int],
    nframes: int,
    epochs: int,
    frame_selection_seed: int,
    task: str,
    experimenter: str,
    finetune_experimenter: str,
) -> dict[str, Any]:

    train_dir = trial_dir / "train"
    cam1_images = _list_images_sorted(train_dir / "Cam1")
    cam2_images = _list_images_sorted(train_dir / "Cam2")
    labels_csv = _labels_csv_in_train(train_dir)
    labels_df = pd.read_csv(labels_csv)

    max_available = min(len(cam1_images), len(cam2_images), len(labels_df))
    selected = sorted(set(int(i) for i in selected_indices if 0 <= int(i) < max_available))
    if not selected:
        raise ValueError(f"No valid selected frames for {trial_dir.name} {method} nframes={nframes}")

    method_root = trial_dir / f"{method}_train" / f"nframes_{int(nframes)}"
    models_dir = method_root / "ModelsToTune"
    models_dir.mkdir(parents=True, exist_ok=True)

    clean_trial = method_root / "ModelTraining"
    if clean_trial.exists():
        shutil.rmtree(clean_trial)
    (clean_trial / "Cam1").mkdir(parents=True, exist_ok=True)
    (clean_trial / "Cam2").mkdir(parents=True, exist_ok=True)

    for idx in selected:
        shutil.copy2(cam1_images[int(idx)], clean_trial / "Cam1" / cam1_images[int(idx)].name)
        shutil.copy2(cam2_images[int(idx)], clean_trial / "Cam2" / cam2_images[int(idx)].name)

    coord_cols = [c for c in labels_df.columns if re.search(r"_cam[12]_[XY]$", str(c))]
    if not coord_cols:
        raise ValueError("No coordinate columns matching *_cam[12]_[XY] in labels CSV")

    selected_df = labels_df.iloc[selected][coord_cols].copy()
    selected_df.to_csv(clean_trial / "UpdatedLabels-2Dpoints.csv", index=False)

    combined_config: Path | None = None
    dlc_setup_status = "dataset_only"
    dlc_setup_message = "DLC project/build step was not attempted"

    try:
        dlcs = _load_dlcsupport()
        dummy_video = method_root / "dummy_cam1.avi"
        fallback_video = _resolve_fallback_dummy_video(bird=bird, trial_num=int(trial_num))
        resolved_dummy_video = _ensure_dummy_video_from_images(
            cam1_images,
            dummy_video,
            fallback_video=fallback_video,
        )

        combined_config = dlcs.create_combined_project_if_missing(
            task=task,
            experimenter=finetune_experimenter,
            combined_project_root=models_dir,
            dummy_video=resolved_dummy_video,
        )
        dlcs.apply_bird_bodyparts_to_configs({bird: [combined_config]}, strict=True)

        dataset_name = f"{method.capitalize()}ModelForTrial{trial_num}_N{int(nframes)}"
        dlcs.build_combined_dataset(
            combined_config=combined_config,
            data_path=clean_trial,
            dataset_name=dataset_name,
            experimenter=experimenter,
            nframes=int(len(selected)),
            frame_selection_seed=int(frame_selection_seed),
        )
        dlc_setup_status = "ok"
        dlc_setup_message = "Combined DLC dataset created"
    except Exception as exc:
        dlc_setup_status = "dataset_only"
        dlc_setup_message = str(exc)

    metadata = {
        "version": 1,
        "created_unix": int(time.time()),
        "trial_folder": trial_dir.name,
        "bird": bird,
        "trial_num": int(trial_num),
        "method": method,
        "update_set": "Random" if method == "random" else method.capitalize(),
        "nframes": int(nframes),
        "epochs_for_training": int(epochs),
        "frame_selection_seed": int(frame_selection_seed),
        "selected_indices": [int(i) for i in selected],
        "source_train_dir": str(train_dir),
        "combined_config": str(combined_config) if combined_config is not None else "",
        "dlc_setup_status": dlc_setup_status,
        "dlc_setup_message": dlc_setup_message,
    }
    metadata_path = method_root / "dataset_build_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "combined_config": str(combined_config) if combined_config is not None else "",
        "metadata_path": str(metadata_path),
        "dlc_setup_status": dlc_setup_status,
        "dlc_setup_message": dlc_setup_message,
    }



def build_three_training_sets_for_trial(
    trial_dir: Path,
    *,
    nframes_values: tuple[int, int] = (100, 50),
    epochs: int = 125,
    frame_selection_seed: int = 42,
    zone_size: int = 150,
    task: str = "Canari",
    experimenter: str = "Tyler",
    finetune_experimenter: str = "FineTuner",
    dino_batch_size: int = 16,
) -> pd.DataFrame:
    trial_dir = Path(trial_dir)
    bird, trial_num = parse_trial_folder_name(trial_dir.name)

    train_dir = trial_dir / "train"
    cam1_images = _list_images_sorted(train_dir / "Cam1")
    cam2_images = _list_images_sorted(train_dir / "Cam2")
    labels_df = pd.read_csv(_labels_csv_in_train(train_dir))

    frame_count = min(len(cam1_images), len(cam2_images), len(labels_df))
    if frame_count <= 0:
        raise ValueError(f"No aligned train frames available in: {trial_dir}")

    rows: list[BuildRow] = []
    for nframes in [int(nframes_values[0]), int(nframes_values[1])]:
        method_plan: list[tuple[str, list[int], dict[str, Any]]] = []
        rng = np.random.default_rng(int(frame_selection_seed + nframes))

        random_indices = _select_random_indices(frame_count, int(nframes), rng)
        method_plan.append(("random", random_indices, {"zone_size": None}))

        displacement_indices, disp_meta = _select_displacement_indices(
            labels_df.iloc[:frame_count].copy(),
            count=int(nframes),
            rng=rng,
            zone_size=int(zone_size),
        )
        method_plan.append(("displacement", displacement_indices, disp_meta))

        dino_exception: Exception | None = None
        dino_indices: list[int] = []
        try:
            dino_indices = _select_dino_indices(
                cam1_images[:frame_count],
                count=int(nframes),
                batch_size=int(dino_batch_size),
            )
        except Exception as exc:
            dino_exception = exc

        if dino_exception is None:
            method_plan.append(("dino", dino_indices, {"zone_size": None}))
        else:
            dino_root = trial_dir / "dino_train" / f"nframes_{int(nframes)}"
            dino_root.mkdir(parents=True, exist_ok=True)
            dino_error_meta = {
                "version": 1,
                "created_unix": int(time.time()),
                "trial_folder": trial_dir.name,
                "bird": bird,
                "trial_num": int(trial_num),
                "method": "dino",
                "nframes": int(nframes),
                "epochs_for_training": int(epochs),
                "frame_selection_seed": int(frame_selection_seed),
                "status": "error",
                "error": str(dino_exception),
            }
            (dino_root / "dataset_build_metadata.json").write_text(
                json.dumps(dino_error_meta, indent=2),
                encoding="utf-8",
            )
            rows.append(
                BuildRow(
                    trial_folder=trial_dir.name,
                    bird=bird,
                    trial_num=int(trial_num),
                    method="dino",
                    nframes=int(nframes),
                    epochs=int(epochs),
                    output_root=str(dino_root.resolve()),
                    config_path="",
                    selection_count=0,
                    status="error",
                    message=str(dino_exception),
                )
            )

        for method, selected_indices, _meta in method_plan:
            try:
                result = _prepare_dlc_dataset(
                    trial_dir=trial_dir,
                    bird=bird,
                    trial_num=trial_num,
                    method=method,
                    selected_indices=selected_indices,
                    nframes=int(nframes),
                    epochs=int(epochs),
                    frame_selection_seed=int(frame_selection_seed),
                    task=task,
                    experimenter=experimenter,
                    finetune_experimenter=finetune_experimenter,
                )
                cfg = str(result.get("combined_config", ""))
                metadata_path = str(result.get("metadata_path", ""))
                setup_status = str(result.get("dlc_setup_status", "dataset_only"))
                setup_message = str(result.get("dlc_setup_message", ""))
                rows.append(
                    BuildRow(
                        trial_folder=trial_dir.name,
                        bird=bird,
                        trial_num=int(trial_num),
                        method=method,
                        nframes=int(nframes),
                        epochs=int(epochs),
                        output_root=str((trial_dir / f"{method}_train" / f"nframes_{int(nframes)}").resolve()),
                        config_path=cfg,
                        selection_count=int(len(selected_indices)),
                        status="ok" if setup_status == "ok" else "partial",
                        message=setup_message if setup_message else metadata_path,
                    )
                )
            except Exception as exc:
                rows.append(
                    BuildRow(
                        trial_folder=trial_dir.name,
                        bird=bird,
                        trial_num=int(trial_num),
                        method=method,
                        nframes=int(nframes),
                        epochs=int(epochs),
                        output_root=str((trial_dir / f"{method}_train" / f"nframes_{int(nframes)}").resolve()),
                        config_path="",
                        selection_count=int(len(selected_indices)),
                        status="error",
                        message=str(exc),
                    )
                )

    return pd.DataFrame([r.__dict__ for r in rows])



def build_three_training_sets_for_all_trials(
    data_root: str | Path,
    *,
    nframes_values: tuple[int, int] = (100, 50),
    epochs: int = 125,
    frame_selection_seed: int = 42,
    zone_size: int = 150,
    task: str = "Canari",
    experimenter: str = "Tyler",
    finetune_experimenter: str = "FineTuner",
    dino_batch_size: int = 16,
) -> pd.DataFrame:
    data_root = Path(data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    outputs: list[pd.DataFrame] = []
    for trial_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        try:
            trial_df = build_three_training_sets_for_trial(
                trial_dir=trial_dir,
                nframes_values=nframes_values,
                epochs=int(epochs),
                frame_selection_seed=int(frame_selection_seed),
                zone_size=int(zone_size),
                task=task,
                experimenter=experimenter,
                finetune_experimenter=finetune_experimenter,
                dino_batch_size=int(dino_batch_size),
            )
            outputs.append(trial_df)
        except Exception as exc:
            outputs.append(
                pd.DataFrame(
                    [
                        {
                            "trial_folder": trial_dir.name,
                            "bird": "",
                            "trial_num": -1,
                            "method": "all",
                            "nframes": -1,
                            "epochs": int(epochs),
                            "output_root": str(trial_dir),
                            "config_path": "",
                            "selection_count": 0,
                            "status": "error",
                            "message": str(exc),
                        }
                    ]
                )
            )

    out_df = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    manifest_path = data_root / "training_set_build_manifest.csv"
    out_df.to_csv(manifest_path, index=False)
    return out_df
