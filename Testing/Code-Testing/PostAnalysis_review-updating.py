from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_dog
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import deeplabcut as dlc
import xrommtools_copy as xt
import DLCsupport as dlcs
import cv2 
import torch
from torchvision import transforms

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WORKFLOW_STATE_FILE = "postanalysis_workflow.json"


# Ideal folder structure:

# BirdProject/
# │
# ├── Code/
# │   ├── dlcsupport.py
# │   ├── PostAnalysis_review.py
# │   └── ...
# │
# ├── Model_Zoo/
# │   ├── DB_15_17_Trained/
# │   ├── Tulio_10_05_Trained/
# │   ├── Miguel_06_Trained/
# │   └── ...
# │
# └── Data/


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ZOO = PROJECT_ROOT / "Model_Zoo"

MODEL_DIRS = {
    "DavidBowie": MODEL_ZOO / "DB_15_17_Trained",
    "Tulio": MODEL_ZOO / "Tulio_10_05_Trained",
    "Miguel": MODEL_ZOO / "Miguel_06_Trained",
    "Endive": MODEL_ZOO / "Endive_42_Trained",
}

# print(MODEL_DIRS)
# Default inference configs associated with each bird.
# Keep this map editable as birds/models change.
BIRD_CONFIG_PATHS = {
    "DavidBowie": MODEL_DIRS["DavidBowie"] / "Canari-FineTuner-2026-07-19" / "config.yaml",
    "Tulio": MODEL_DIRS["Tulio"] / "Canari-FineTuner-2026-07-19" / "config.yaml",
    "Miguel": MODEL_DIRS["Miguel"] / "Canari-FineTuner-2026-07-17" / "config.yaml",
    "Endive": MODEL_DIRS["Endive"] / "Canari-FineTuner-2026-07-17" / "config.yaml",
}


 # Default snapshots associated with each bird for fine-tuning.
BIRD_SNAPSHOT_PATHS = {
    "DavidBowie": MODEL_DIRS["DavidBowie"] / "Canari-FineTuner-2026-07-19" /
                   "dlc-models-pytorch" / "iteration-0" /
                   "CanariJul19-trainset95shuffle1" / "train" / "snapshot-125.pt",

    "Tulio": MODEL_DIRS["Tulio"] / "Canari-FineTuner-2026-07-19" /
              "dlc-models-pytorch" / "iteration-0" /
              "CanariJul19-trainset95shuffle1" / "train" / "snapshot-125.pt",

    "Miguel": MODEL_DIRS["Miguel"] / "Canari-FineTuner-2026-07-17" /
               "dlc-models-pytorch" / "iteration-0" /
               "CanariJul17-trainset95shuffle1" / "train" / "snapshot-125.pt",

    "Endive": MODEL_DIRS["Endive"] / "Canari-FineTuner-2026-07-17" /
               "dlc-models-pytorch" / "iteration-0" /
               "CanariJul17-trainset95shuffle1" / "train" / "snapshot-125.pt",
}
# print(BIRD_SNAPSHOT_PATHS)


def _load_data_converter_module() -> Any:
	"""Load data-converters.py as module dc (used for jpg stack to avi conversion)."""
	module_path = Path(__file__).with_name("data-converters.py")
	if not module_path.exists():
		raise FileNotFoundError(f"Could not find converter module: {module_path}")
	spec = importlib.util.spec_from_file_location("dc", str(module_path))
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Could not load module spec for: {module_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _workflow_state_path(trial_dir: str | Path) -> Path:
	return Path(trial_dir) / WORKFLOW_STATE_FILE


def _load_saved_frame_range(trial_dir: str | Path) -> tuple[int, int] | None:
	state_path = _workflow_state_path(trial_dir)
	if not state_path.exists():
		return None
	try:
		data = json.loads(state_path.read_text(encoding="utf-8"))
	except Exception:
		return None

	frame_range = data.get("frame_range", {}) if isinstance(data, dict) else {}
	start = frame_range.get("start")
	end = frame_range.get("end")
	try:
		start_i = int(start)
		end_i = int(end)
	except Exception:
		return None

	if end_i < start_i:
		start_i, end_i = end_i, start_i
	return start_i, end_i


def _save_frame_range(trial_dir: str | Path, start_frame: int, end_frame: int, source: str = "") -> None:
	state_path = _workflow_state_path(trial_dir)
	state_path.parent.mkdir(parents=True, exist_ok=True)

	start_i = int(start_frame)
	end_i = int(end_frame)
	if end_i < start_i:
		start_i, end_i = end_i, start_i

	data: dict[str, Any] = {}
	if state_path.exists():
		try:
			loaded = json.loads(state_path.read_text(encoding="utf-8"))
			if isinstance(loaded, dict):
				data = loaded
		except Exception:
			data = {}

	data["frame_range"] = {"start": start_i, "end": end_i}
	data["last_source"] = str(source)
	data["updated_unix"] = int(time.time())
	state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_darkest_pixel(
	img: np.ndarray,
	row: float,
	col: float,
	occupied: set[tuple[int, int]] | None = None,
	window_size: int = 5,
) -> tuple[int, int]:
	"""Find the darkest available pixel near (row, col) within a local window."""
	half = int(window_size) // 2

	r0 = max(0, int(row) - half)
	r1 = min(img.shape[0], int(row) + half + 1)
	c0 = max(0, int(col) - half)
	c1 = min(img.shape[1], int(col) + half + 1)

	window = img[r0:r1, c0:c1]
	if window.size == 0:
		return int(np.clip(int(row), 0, max(img.shape[0] - 1, 0))), int(np.clip(int(col), 0, max(img.shape[1] - 1, 0)))

	flat = np.argsort(window, axis=None)
	used = occupied if occupied is not None else set()

	for idx in flat:
		lr, lc = np.unravel_index(idx, window.shape)
		rr, cc = int(r0 + lr), int(c0 + lc)
		if (rr, cc) not in used:
			used.add((rr, cc))
			return rr, cc

	return int(r0), int(c0)

# FIXME: Need to add animal so that if pig is there we change hyperparams...
def find_dark_blob_centroid(
	img: np.ndarray,
	row: float,
	col: float,
	occupied: set[tuple[int, int]] | None = None,
	window_size: int = 9,
	camera: str = "cam1",
	background_sigma: float | None = None,
	min_sigma: float | None = None,
	max_sigma: float | None = None,
	sigma_ratio: float | None = None,
	threshold: float | None = None,
	overlap: float | None = None,
	max_match_dist: float = 15.0,
) -> tuple[int, int]:
	"""Find nearest DoG blob center to (row, col), with occupied and darkest-pixel fallback."""
	gray = _to_gray_float(img)
	blobs = _detect_dark_blobs_blobdog(
		gray,
		camera=camera,
		background_sigma=background_sigma,
		min_sigma=min_sigma,
		max_sigma=max_sigma,
		sigma_ratio=sigma_ratio,
		threshold=threshold,
		overlap=overlap,
	)
	if blobs.size == 0:
		return find_darkest_pixel(img, row=row, col=col, occupied=occupied, window_size=window_size)

	used = occupied if occupied is not None else set()
	target = np.asarray([float(row), float(col)], dtype=float)
	centers = blobs[:, :2]
	dists = np.linalg.norm(centers - target[None, :], axis=1)
	order = np.argsort(dists)

	for idx in order:
		dist = float(dists[int(idx)])
		if dist > float(max_match_dist):
			continue
		rr = int(round(float(blobs[int(idx), 0])))
		cc = int(round(float(blobs[int(idx), 1])))
		if (rr, cc) in used:
			continue
		used.add((rr, cc))
		# return rr, cc
		# Trying to get darkest pixel working
		return(find_darkest_pixel(img, row=rr, col=cc, occupied=occupied, window_size=window_size))

	return find_darkest_pixel(img, row=row, col=col, occupied=occupied, window_size=window_size)


def _blobdog_camera_defaults(camera: str, animal: str = 'bird') -> dict[str, float]:
	cam = _cam_norm(camera)
	base = {
		"background_sigma": 20.0,
		"min_sigma": 1.0,
		"max_sigma": 2.0,
		"sigma_ratio": 1.3,
		"threshold": 0.03,
		"overlap": 0.5,
	}
	if animal == 'Pig':
		base['sigma_ratio'] = 3
		return base
	if cam == "cam2":
		base["min_sigma"] = 1.5
		base["overlap"] = 0.8
	
	return base


def _to_gray_float(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return np.asarray(image, dtype=np.float64)
	if image.ndim >= 3:
		return np.asarray(image[..., :3], dtype=np.float64).mean(axis=2)
	raise ValueError("image must be 2D grayscale or 3D RGB-like array")


def _detect_dark_blobs_blobdog(
	
	gray: np.ndarray,
	camera: str,
	animal: str = 'bird',
	background_sigma: float | None = None,
	min_sigma: float | None = None,
	max_sigma: float | None = None,
	sigma_ratio: float | None = None,
	threshold: float | None = None,
	overlap: float | None = None,

) -> np.ndarray:
	"""Detect dark blobs using the same DoG flow tested in Bead_Segmentation."""
	try:
		from scipy.ndimage import gaussian_filter
		from skimage.feature import blob_dog
	except Exception:
		return np.empty((0, 3), dtype=float)

	defaults = _blobdog_camera_defaults(camera, animal)
	params = {
		"background_sigma": float(defaults["background_sigma"] if background_sigma is None else background_sigma),
		"min_sigma": float(defaults["min_sigma"] if min_sigma is None else min_sigma),
		"max_sigma": float(defaults["max_sigma"] if max_sigma is None else max_sigma),
		"sigma_ratio": float(defaults["sigma_ratio"] if sigma_ratio is None else sigma_ratio),
		"threshold": float(defaults["threshold"] if threshold is None else threshold),
		"overlap": float(defaults["overlap"] if overlap is None else overlap),
	}

	bg = gaussian_filter(gray, sigma=float(params["background_sigma"]))
	hp = bg - gray
	rng = float(hp.max() - hp.min())
	if not np.isfinite(rng) or rng <= 1e-12:
		return np.empty((0, 3), dtype=float)
	hp = (hp - float(hp.min())) / rng

	blobs = blob_dog(
		hp,
		min_sigma=float(params["min_sigma"]),
		max_sigma=float(params["max_sigma"]),
		sigma_ratio=float(params["sigma_ratio"]),
		threshold=float(params["threshold"]),
		overlap=float(params["overlap"]),
	)
	if blobs is None or len(blobs) == 0:
		return np.empty((0, 3), dtype=float)
	return np.asarray(blobs, dtype=float)


def _extract_camera_predictions(row: pd.Series, camera: str) -> tuple[np.ndarray, list[str]]:
	"""Extract camera-specific predicted coordinates as [y, x] and bodypart prefixes."""
	pred_points: list[list[float]] = []
	bodyparts: list[str] = []
	x_suffix = f"_{_cam_norm(camera)}_X"

	for col in row.index:
		col_str = str(col)
		if not col_str.endswith(x_suffix):
			continue
		prefix = col_str[:-2]
		x_val = float(pd.to_numeric(row.get(f"{prefix}_X"), errors="coerce"))
		y_val = float(pd.to_numeric(row.get(f"{prefix}_Y"), errors="coerce"))
		if not (np.isfinite(x_val) and np.isfinite(y_val)):
			continue
		pred_points.append([y_val, x_val])
		bodyparts.append(prefix)

	if not pred_points:
		return np.empty((0, 2), dtype=float), []
	return np.asarray(pred_points, dtype=float), bodyparts

# FIXME: Need to add animal so that if pig is there we change hyperparams...
def _correct_camera_frame(
	image: np.ndarray,
	row: pd.Series,
	camera: str = "cam1",
	background_sigma: float | None = None,
	min_sigma: float | None = None,
	max_sigma: float | None = None,
	sigma_ratio: float | None = None,
	threshold: float | None = None,
	overlap: float | None = None,
	max_match_dist: float = 15.0,
) -> dict[str, float]:
	"""Correct one camera row using DoG blobs + Hungarian assignment (Cell 47-54 flow)."""
	try:
		from scipy.optimize import linear_sum_assignment
		from scipy.spatial.distance import cdist
	except Exception:
		return {}

	gray = _to_gray_float(image)
	blobs = _detect_dark_blobs_blobdog(
		gray,
		camera=camera,
		background_sigma=background_sigma,
		min_sigma=min_sigma,
		max_sigma=max_sigma,
		sigma_ratio=sigma_ratio,
		threshold=threshold,
		overlap=overlap,
	)
	if blobs.size == 0:
		return {}

	pred_points, bodyparts = _extract_camera_predictions(row, camera)
	if pred_points.size == 0:
		return {}

	blob_centers = blobs[:, :2]
	D = cdist(pred_points, blob_centers)
	D[D > float(max_match_dist)] = 1e6

	pred_idx, blob_idx = linear_sum_assignment(D)
	valid = D[pred_idx, blob_idx] < float(max_match_dist)

	updates: dict[str, float] = {}
	for p, b in zip(pred_idx[valid], blob_idx[valid]):
		py, px = pred_points[int(p)]
		by, bx, sigma = blobs[int(b)]
		radius = float(sigma) * float(np.sqrt(2.0))
		dist = float(np.hypot(py - by, px - bx))
		if dist > radius:
			py = float(by)
			px = float(bx)

		bp = str(bodyparts[int(p)])
		updates[f"{bp}_X"] = float(px)
		updates[f"{bp}_Y"] = float(py)

	return updates


def parse_frame_number_from_stem(stem: str) -> int | None:
	"""Extract the last integer token from a file stem."""
	match = re.search(r"(\d+)(?!.*\d)", stem)
	return int(match.group(1)) if match else None


def collect_images(image_dir: Path) -> list[Path]:
	"""Collect and sort image files from a directory."""
	imgs = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
	return sorted(
		imgs,
		key=lambda p: (
			parse_frame_number_from_stem(p.stem) is None,
			parse_frame_number_from_stem(p.stem) or 0,
			p.name.lower(),
		),
	)


def _cam_norm(value: Any) -> str:
	"""Normalize camera tokens to cam1/cam2."""
	s = str(value).lower().strip()
	if s in {"1", "cam1", "camera1"}:
		return "cam1"
	if s in {"2", "cam2", "camera2"}:
		return "cam2"
	if "cam2" in s or "camera2" in s:
		return "cam2"
	return "cam1"


# def _find_first_dir(root: Path, pattern: str) -> Path:
# 	"""Return first matching directory under root for a glob pattern."""
# 	root_lower =  Path(str(root).lower())
# 	print(root_lower)
# 	matches = sorted([p for p in root_lower.rglob(pattern.lower()) if p.is_dir()])
# 	if not matches:
# 		raise FileNotFoundError(f"Could not find folder matching {pattern!r} under {root_lower}")
# 	return matches[0]

from pathlib import Path
import fnmatch


def _find_first_dir(root: Path, pattern: str) -> Path:
    """Return first matching directory under root, case-insensitively."""
    root = Path(root)

    pattern_lower = pattern.lower()

    matches = sorted(
        p
        for p in root.rglob("*")
        if p.is_dir()
        and fnmatch.fnmatch(p.name.lower(), pattern_lower)
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find folder matching {pattern!r} under {root}"
        )

    return matches[0]




def _resolve_cam_dirs(root: Path) -> dict[str, Path]:
	"""Resolve Cam1_Img00UND* and Cam2_Img00UND* directories."""
	return {
		"cam1": _find_first_dir(root, "*cam1*"),
		"cam2": _find_first_dir(root, "*cam2*"),
	}


def _resolve_prediction_csvs(root: Path) -> dict[str, Path]:
	"""Resolve prediction CSVs from post_processed_data_* or a single combined CSV."""
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Prediction folder does not exist: {root}")

	post_dirs = sorted([p for p in root.rglob("post_processed_data_*") if p.is_dir()])
	cam_csv: dict[str, Path] = {}
	for d in post_dirs:
		csvs = sorted([p for p in d.glob("*.csv") if p.is_file()])
		for p in csvs:
			low = p.name.lower()
			if "cam1dlc" in low and "cam1" not in cam_csv:
				cam_csv["cam1"] = p
			if "cam2dlc" in low and "cam2" not in cam_csv:
				cam_csv["cam2"] = p

	if "cam1" in cam_csv and "cam2" in cam_csv:
		return cam_csv

	def _is_combined_cam_csv(csv_path: Path) -> bool:
		try:
			headers = pd.read_csv(csv_path, nrows=0).columns
		except Exception:
			return False
		cams: set[str] = set()
		coords: dict[str, set[str]] = {"1": set(), "2": set()}
		for col in headers:
			match = re.match(r"(?P<bodypart>.+)_cam(?P<cam>[12])_(?P<coord>[XY])$", str(col))
			if match is None:
				continue
			cam_id = str(match.group("cam"))
			coord = str(match.group("coord")).upper()
			cams.add(cam_id)
			coords.setdefault(cam_id, set()).add(coord)
		return cams == {"1", "2"} and {"X", "Y"}.issubset(coords.get("1", set())) and {"X", "Y"}.issubset(coords.get("2", set()))

	root_csvs = sorted([p for p in root.glob("*.csv") if p.is_file()])
	combined_candidates = [p for p in root_csvs if _is_combined_cam_csv(p)]
	if not combined_candidates:
		all_csvs = sorted([p for p in root.rglob("*.csv") if p.is_file() and "post_processed_data_" not in str(p).lower()])
		combined_candidates = [p for p in all_csvs if _is_combined_cam_csv(p)]

	if len(combined_candidates) == 1:
		combined_path = combined_candidates[0]
		return {"cam1": combined_path, "cam2": combined_path}
	if len(combined_candidates) > 1:
		preferred = [p for p in combined_candidates if "pred" in p.name.lower()]
		chosen = preferred[0] if preferred else combined_candidates[0]
		return {"cam1": chosen, "cam2": chosen}

	missing = [c for c in ("cam1", "cam2") if c not in cam_csv]
	if missing:
		raise FileNotFoundError(
			"Missing expected prediction CSV(s). Expected either cam1/cam2 files under post_processed_data_* "
			"or one combined CSV in truth-style format with both cam1 and cam2 columns under: "
			f"{root}"
		)

	return cam_csv


def _cleanup_intermediate_dlc_outputs(root: Path) -> None:
	"""Remove DLC per-camera intermediate outputs once combined XMALab output exists."""
	patterns = (
		"Cam1DLC*.csv",
		"Cam2DLC*.csv",
		"Cam1DLC*.h5",
		"Cam2DLC*.h5",
		"Cam1DLC*.pickle",
		"Cam2DLC*.pickle",
	)
	for pattern in patterns:
		for path in root.glob(pattern):
			if not path.is_file():
				continue
			try:
				path.unlink()
			except Exception:
				pass


def _truth_csv_to_long(truth_csv_path: Path) -> pd.DataFrame:
	"""Convert truth CSV with columns like bodypart_cam1_X into long form."""
	truth_df = pd.read_csv(truth_csv_path)
	rows = []
	for col in truth_df.columns:
		match = re.match(r"(?P<bodypart>.+)_cam(?P<cam>[12])_(?P<coord>[XY])$", col)
		if match is None:
			continue
		rows.append((match.group("bodypart"), f"cam{match.group('cam')}", match.group("coord").lower(), col))

	if not rows:
		return pd.DataFrame(columns=["frame_id", "bodypart", "camera", "x_true", "y_true"])

	meta = pd.DataFrame(rows, columns=["bodypart", "camera", "coord", "column"])
	xmeta = meta[meta["coord"] == "x"].rename(columns={"column": "xcol"})
	ymeta = meta[meta["coord"] == "y"].rename(columns={"column": "ycol"})
	pairs = xmeta.merge(ymeta[["bodypart", "camera", "ycol"]], on=["bodypart", "camera"], how="inner")

	frame_ids = np.arange(1, len(truth_df) + 1)
	out = []
	for _, row in pairs.iterrows():
		out.append(
			pd.DataFrame(
				{
					"frame_id": frame_ids,
					"bodypart": row["bodypart"],
					"camera": row["camera"],
					"x_true": pd.to_numeric(truth_df[row["xcol"]], errors="coerce"),
					"y_true": pd.to_numeric(truth_df[row["ycol"]], errors="coerce"),
				}
			)
		)

	return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _load_prediction_long(pred_file: Path) -> pd.DataFrame:
	"""Load prediction file into long form with frame/bodypart/camera/x/y/likelihood."""
	if pred_file.suffix.lower() == ".parquet":
		raw = pd.read_parquet(pred_file)
	else:
		raw = pd.read_csv(pred_file)

	low_cols = {str(c).lower(): c for c in raw.columns}

	def pick(*names: str) -> str | None:
		for name in names:
			if name in low_cols:
				return low_cols[name]
		return None

	c_camera = pick("camera", "cam")
	c_frame = pick("frame_id", "frame", "image_index", "frame_idx")
	c_bp = pick("bodypart", "part", "keypoint")
	c_x = pick("x_pred", "x", "pred_x")
	c_y = pick("y_pred", "y", "pred_y")
	c_l = pick("likelihood", "conf", "confidence", "p")

	if c_bp and c_x and c_y:
		out = pd.DataFrame(
			{
				"camera": raw[c_camera].map(_cam_norm) if c_camera else "cam1",
				"frame_id": pd.to_numeric(raw[c_frame], errors="coerce") if c_frame else np.nan,
				"bodypart": raw[c_bp].astype(str),
				"x_pred": pd.to_numeric(raw[c_x], errors="coerce"),
				"y_pred": pd.to_numeric(raw[c_y], errors="coerce"),
				"likelihood": pd.to_numeric(raw[c_l], errors="coerce") if c_l else np.nan,
			}
		)
		if out["frame_id"].isna().all():
			out["frame_id"] = out.groupby("camera").cumcount() + 1
		out["frame_id"] = out["frame_id"].astype(int)
		return out.dropna(subset=["x_pred", "y_pred"])

	# Combined truth-style columns: bodypart_cam1_X, bodypart_cam2_Y, etc.
	rows = []
	for col in raw.columns:
		match = re.match(r"(?P<bodypart>.+)_cam(?P<cam>[12])_(?P<coord>[XY])$", str(col))
		if match is None:
			continue
		rows.append((match.group("bodypart"), f"cam{match.group('cam')}", match.group("coord").lower(), str(col)))

	if rows:
		meta = pd.DataFrame(rows, columns=["bodypart", "camera", "coord", "column"])
		xmeta = meta[meta["coord"] == "x"].rename(columns={"column": "xcol"})
		ymeta = meta[meta["coord"] == "y"].rename(columns={"column": "ycol"})
		pairs = xmeta.merge(ymeta[["bodypart", "camera", "ycol"]], on=["bodypart", "camera"], how="inner")
		frame_ids = np.arange(0, len(raw), dtype=int)
		out = []
		for _, row in pairs.iterrows():
			out.append(
				pd.DataFrame(
					{
						"camera": row["camera"],
						"frame_id": frame_ids,
						"bodypart": str(row["bodypart"]),
						"x_pred": pd.to_numeric(raw[row["xcol"]], errors="coerce"),
						"y_pred": pd.to_numeric(raw[row["ycol"]], errors="coerce"),
						"likelihood": np.nan,
					}
				)
			)
		if out:
			return pd.concat(out, ignore_index=True).dropna(subset=["x_pred", "y_pred"])

	# Fallback for DLC-style multi-index csv
	wide = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
	camera = "cam2" if "cam2" in pred_file.name.lower() else "cam1"
	subset_idx = pd.to_numeric(pd.Index(wide.index), errors="coerce").fillna(0).astype(int)

	bodyparts = pd.Index(wide.columns.get_level_values(1)).unique()
	out = []
	for bp in bodyparts:
		bp_cols = wide.xs(bp, axis=1, level=1, drop_level=False)
		coords = bp_cols.columns.get_level_values(2).astype(str).str.lower().tolist()
		if "x" not in coords or "y" not in coords:
			continue

		xcol = bp_cols.columns[coords.index("x")]
		ycol = bp_cols.columns[coords.index("y")]

		if "likelihood" in coords:
			lcol = bp_cols.columns[coords.index("likelihood")]
			like = pd.to_numeric(bp_cols[lcol], errors="coerce").to_numpy(dtype=float)
		else:
			like = np.full(bp_cols.shape[0], np.nan)

		out.append(
			pd.DataFrame(
				{
					"camera": camera,
					"frame_id": (subset_idx + 1).to_numpy(),
					"bodypart": str(bp),
					"x_pred": pd.to_numeric(bp_cols[xcol], errors="coerce").to_numpy(dtype=float),
					"y_pred": pd.to_numeric(bp_cols[ycol], errors="coerce").to_numpy(dtype=float),
					"likelihood": like,
				}
			)
		)

	if not out:
		raise ValueError(f"Could not parse predictions from: {pred_file}")

	return pd.concat(out, ignore_index=True).dropna(subset=["x_pred", "y_pred"])


def _auto_find_truth_csv(root: Path) -> Path | None:
	"""Optionally auto-find a likely truth CSV under root."""
	candidates = sorted(
		[
			p
			for p in root.rglob("*.csv")
			if p.is_file() and not p.name.lower().startswith("post_processed_data")
		]
	)
	for p in candidates:
		low = p.name.lower()
		if "truth" in low or "label" in low:
			return p
	return None


def _choose_directory_gui(title: str, prompt: str) -> Path:
	"""Show an informational prompt, then open a directory picker."""
	try:
		import tkinter as tk
		from tkinter import filedialog, messagebox
	except Exception as exc:
		raise RuntimeError(
			"GUI file picker is unavailable. Provide folder paths on the command line instead."
		) from exc

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	messagebox.showinfo(title=title, message=prompt)

	folder_path = filedialog.askdirectory(title=title)
	root.destroy()

	if not folder_path:
		raise RuntimeError(f"No folder selected for: {title}")
	return Path(folder_path)


def _choose_inputs_gui() -> tuple[Path, Path | None, Path | None]:
	"""Prompt user for same-folder mode, then collect required folder paths."""
	try:
		import tkinter as tk
		from tkinter import filedialog, messagebox, simpledialog
	except Exception as exc:
		raise RuntimeError(
			"GUI prompts are unavailable. Provide folder paths on the command line instead."
		) from exc

	def _maybe_predict_if_images_only(shared_root: Path) -> tuple[Path, Path | None, Path | None]:
		if not _looks_like_image_only_selection(shared_root):
			return shared_root, None, None

		prompt_root = tk.Tk()
		prompt_root.withdraw()
		prompt_root.attributes("-topmost", True)

		should_predict = bool(
			messagebox.askyesno(
				title="Images Detected",
				message=(
					"The selected folder appears to contain images but no predictions.\n\n"
					"Do you want to run prediction now?"
				),
				parent=prompt_root,
			)
		)
		if not should_predict:
			prompt_root.destroy()
			raise RuntimeError("No prediction files found in selected folder. Prediction was cancelled.")

		trial_dir = _infer_trial_dir_from_selected_folder(shared_root)
		base_dir = trial_dir.parent.parent if trial_dir.parent.parent.exists() else trial_dir.parent
		try:
			context = _extract_trial_context_from_path(trial_dir)
			bird = str(context["bird"])
		except Exception:
			bird_input = simpledialog.askstring(
				"Bird Name",
				"Could not infer bird from path. Enter bird name:",
				initialvalue=str(trial_dir.parent.name),
				parent=prompt_root,
			)
			if bird_input is None or str(bird_input).strip() == "":
				prompt_root.destroy()
				raise RuntimeError("Prediction cancelled: no bird name provided.")
			bird = str(bird_input).strip()

			trial_input = simpledialog.askinteger(
				"Trial Number",
				"Could not infer trial from path. Enter trial number:",
				initialvalue=1,
				minvalue=1,
				parent=prompt_root,
			)
			if trial_input is None:
				prompt_root.destroy()
				raise RuntimeError("Prediction cancelled: no trial number provided.")

			trial_num = int(trial_input)
			trial_dir = Path(base_dir) / bird / f"Trial{trial_num}"
			if not trial_dir.exists():
				prompt_root.destroy()
				raise FileNotFoundError(
					f"The provided bird/trial path does not exist: {trial_dir}"
				)

		saved_range = _load_saved_frame_range(trial_dir)
		default_start = int(saved_range[0]) if saved_range is not None else 0
		default_end = int(saved_range[1]) if saved_range is not None else int(default_start + 1000)

		use_range = bool(
			messagebox.askyesno(
				title="Frame Range",
				message=(
					"Do you want to set a frame range for prediction/correction?"
					+ (
						f"\n\nSaved range found: {default_start} to {default_end}."
						if saved_range is not None
						else ""
					)
				),
				parent=prompt_root,
			)
		)
		frame_range = None
		if use_range:
			start_frame = simpledialog.askinteger(
				"Start Frame",
				"Start frame:",
				initialvalue=int(default_start),
				minvalue=0,
				parent=prompt_root,
			)
			if start_frame is None:
				prompt_root.destroy()
				raise RuntimeError("Prediction cancelled: no start frame provided.")

			end_frame = simpledialog.askinteger(
				"End Frame",
				"End frame:",
				initialvalue=int(default_end if saved_range is not None else int(start_frame) + 1000),
				minvalue=0,
				parent=prompt_root,
			)
			if end_frame is None:
				prompt_root.destroy()
				raise RuntimeError("Prediction cancelled: no end frame provided.")

			frame_range = [int(start_frame), int(end_frame)]
			_save_frame_range(trial_dir, int(start_frame), int(end_frame), source="predict_trial_from_jpg_stacks")
		elif saved_range is not None:
			frame_range = [int(default_start), int(default_end)]

		default_config = BIRD_CONFIG_PATHS.get(bird)
		config_path = Path(default_config) if default_config else None
		use_default = bool(config_path is not None and config_path.exists())

		if use_default:
			use_default = bool(
				messagebox.askyesno(
					title="Model Config",
					message=(
						f"Use default config for {bird}?\n\n"
						f"{config_path}"
					),
					parent=prompt_root,
				)
			)

		if not use_default:
			picked = filedialog.askopenfilename(
				title="Select DeepLabCut config.yaml",
				filetypes=[("YAML", "*.yaml *.yml"), ("All Files", "*.*")],
				parent=prompt_root,
			)
			if not picked:
				prompt_root.destroy()
				raise RuntimeError("Prediction cancelled: no config file selected.")
			config_path = Path(picked)

		prompt_root.destroy()

		result = predict_trial_from_jpg_stacks(
			trial_dir=trial_dir,
			config_path=config_path,
			frame_range=frame_range,
			xma_base_name="NoUpdateModel",
		)
		return Path(result["trial_dir"]), None, None

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	same_root = messagebox.askyesno(
		title="Prediction Viewer Setup",
		message="Are your images and predictions in the same folder?",
	)
	root.destroy()

	if same_root:
		shared_root = _choose_directory_gui(
			title="Select Shared Root Folder",
			prompt=(
				"Select the folder containing prediction CSVs and both camera image folders "
				"(for example cam1 and cam2 folders)."
			),
		)
		return _maybe_predict_if_images_only(shared_root)

	pred_root = _choose_directory_gui(
		title="Select Predictions Folder",
		prompt="Select the folder containing prediction CSV files.",
	)
	cam1_dir = _choose_directory_gui(
		title="Select Cam1 Image Folder",
		prompt="Select the folder containing Cam1 images.",
	)
	cam2_dir = _choose_directory_gui(
		title="Select Cam2 Image Folder",
		prompt="Select the folder containing Cam2 images.",
	)
	return pred_root, cam1_dir, cam2_dir


def _active_update_roots(base_dir: str | Path, bird: str, trial_num: int) -> list[Path]:
	base_dir = Path(base_dir)
	return [
		base_dir / bird / "active_updates" / f"Trial{trial_num}",
		base_dir / bird / f"Trial{trial_num}" / "active_updates",
	]


def _extract_trial_context_from_path(path_like: str | Path) -> dict[str, Any]:
	"""Infer bird/trial/base/trial_dir from any path containing .../<bird>/TrialN/..."""
	path_obj = Path(path_like).resolve()
	parts = list(path_obj.parts)
	trial_idx = None
	trial_name = ""
	for idx, token in enumerate(parts):
		if re.fullmatch(r"Trial\d+", str(token), flags=re.IGNORECASE):
			trial_idx = idx
			trial_name = str(token)
			break
	if trial_idx is None or trial_idx <= 0:
		raise ValueError(f"Could not infer bird/trial from path: {path_obj}")

	bird = str(parts[trial_idx - 1])
	trial_num = int(re.search(r"(\d+)", trial_name).group(1))
	base_dir = Path(*parts[:trial_idx - 1])
	trial_dir = Path(*parts[:trial_idx + 1])
	return {
		"bird": bird,
		"trial_num": trial_num,
		"base_dir": base_dir,
		"trial_dir": trial_dir,
	}


def _looks_like_image_only_selection(selected_dir: Path) -> bool:
	"""Return True when a folder appears to have JPG stacks but no prediction CSVs yet."""
	if not selected_dir.exists() or not selected_dir.is_dir():
		return False

	files = [p for p in selected_dir.iterdir() if p.is_file()]
	has_images_here = any(p.suffix.lower() in IMAGE_EXTS for p in files)
	has_csv_here = any(p.suffix.lower() == ".csv" for p in files)

	if has_images_here and not has_csv_here:
		return True

	cam1_dir = selected_dir / "Cam1"
	cam2_dir = selected_dir / "Cam2"
	if cam1_dir.is_dir() and cam2_dir.is_dir():
		cam1_images = any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in cam1_dir.iterdir())
		cam2_images = any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in cam2_dir.iterdir())
		if cam1_images and cam2_images:
			try:
				_resolve_prediction_csvs(selected_dir)
			except Exception:
				return True

	return False


def _infer_trial_dir_from_selected_folder(selected_dir: Path) -> Path:
	"""Map a selected folder to trial_dir containing Cam1/Cam2 image folders."""
	selected_dir = Path(selected_dir)
	if (selected_dir / "Cam1").is_dir() and (selected_dir / "Cam2").is_dir():
		return selected_dir

	name_low = selected_dir.name.lower()
	if name_low in {"cam1", "cam2"}:
		trial_dir = selected_dir.parent
		if (trial_dir / "Cam1").is_dir() and (trial_dir / "Cam2").is_dir():
			return trial_dir

	raise FileNotFoundError(
		"Could not infer trial directory with Cam1/Cam2 from selected folder. "
		f"Selected: {selected_dir}"
	)


def predict_trial_from_jpg_stacks(
	trial_dir: str | Path,
	config_path: str | Path,
	fps: int = 500,
	batchsize: int = 16,
	save_as_csv: bool = True,
	xma_base_name: str = "NoUpdateModel",
	frame_range: list[int] | tuple[int, int] | None = None,
) -> dict[str, Any]:
	"""Predict from Cam1/Cam2 JPG stacks for a selected trial directory and save combined output."""
	import deeplabcut as dlc
	import xrommtools_copy as xt

	dc = _load_data_converter_module()
	trial_dir = Path(trial_dir)
	config_path = Path(config_path)

	cam1_dir = trial_dir / "Cam1"
	cam2_dir = trial_dir / "Cam2"
	cam1_avi = trial_dir / "Cam1.avi"
	cam2_avi = trial_dir / "Cam2.avi"
	total_frames = len(list(cam1_dir.glob("*.jpg")))

	if not cam1_dir.exists() or not cam2_dir.exists():
		raise FileNotFoundError(
			f"Expected Cam1/Cam2 folders in {trial_dir}, but one or both are missing."
		)

	if frame_range is None:
		start_frame = 0
		end_frame = int(total_frames)
	else:
		start_frame, end_frame = map(int, frame_range)
		if end_frame < start_frame:
			start_frame, end_frame = end_frame, start_frame

	if not cam1_avi.exists():
		dc.jpg_stack_to_avi(
			input_folder=cam1_dir,
			output_path=cam1_avi,
			fps=int(fps),
			start_frame=int(start_frame),
			end_frame=int(end_frame),
		)
	if not cam2_avi.exists():
		dc.jpg_stack_to_avi(
			input_folder=cam2_dir,
			output_path=cam2_avi,
			fps=int(fps),
			start_frame=int(start_frame),
			end_frame=int(end_frame),
		)

	if not sorted(trial_dir.glob("Cam1DLC*.csv"), key=lambda path: path.stat().st_mtime):
		dlc.analyze_videos(
			str(config_path),
			[str(cam1_avi), str(cam2_avi)],
			save_as_csv=bool(save_as_csv),
			destfolder=str(trial_dir),
			batchsize=int(batchsize),
		)

	cam1_candidates = sorted(trial_dir.glob("Cam1DLC*.csv"), key=lambda path: path.stat().st_mtime)
	cam2_candidates = sorted(trial_dir.glob("Cam2DLC*.csv"), key=lambda path: path.stat().st_mtime)
	if not cam1_candidates or not cam2_candidates:
		raise FileNotFoundError(
			f"Could not find Cam1DLC*.csv / Cam2DLC*.csv in {trial_dir} after prediction."
		)

	cam1_dlc_csv = cam1_candidates[-1]
	cam2_dlc_csv = cam2_candidates[-1]
	pred_df = xt.dlc_to_xma(str(cam1_dlc_csv), str(cam2_dlc_csv), str(xma_base_name), str(trial_dir))
	_cleanup_intermediate_dlc_outputs(trial_dir)

	nan_padded_df = pad_df_to_trial_length(pred_df, total_frames=total_frames, start_frame=int(start_frame))
	corrected_df = correct_frames_from_df_fast(
		trial_dir=trial_dir,
		pred_df=nan_padded_df,
		frames_to_correct=list(range(int(start_frame), int(end_frame))),
		max_match_dist=15.0,
	)

	pred_csv = trial_dir / f"{xma_base_name}-Predicted2DPoints.csv"
	corrected_df.to_csv(pred_csv, na_rep="NaN", index=False)

	return {
		"trial_dir": trial_dir,
		"cam1_avi": cam1_avi,
		"cam2_avi": cam2_avi,
		"cam1_dlc_csv": cam1_dlc_csv,
		"cam2_dlc_csv": cam2_dlc_csv,
		"pred_csv": pred_csv,
	}


def _list_active_update_dirs(base_dir: str | Path, bird: str, trial_num: int) -> list[Path]:
	active_dirs: list[Path] = []
	seen: set[str] = set()

	for root in _active_update_roots(base_dir, bird, trial_num):
		if not root.exists():
			continue
		for path in sorted(p for p in root.iterdir() if p.is_dir()):
			resolved = str(path.resolve()).lower()
			if resolved in seen:
				continue
			seen.add(resolved)
			active_dirs.append(path)

	return active_dirs


def _resolve_active_update_dir(
	base_dir: str | Path,
	bird: str,
	trial_num: int,
	update_set: str | None = None,
) -> Path:
	active_dirs = _list_active_update_dirs(base_dir, bird, trial_num)

	if len(active_dirs) == 0:
		searched = "\n".join(str(path) for path in _active_update_roots(base_dir, bird, trial_num))
		raise FileNotFoundError(
			"No active update folders found. Checked:\n"
			f"{searched}"
		)

	if update_set is not None:
		matches = [path for path in active_dirs if path.name.lower() == update_set.lower()]
		if len(matches) == 0:
			options = ", ".join(path.name for path in active_dirs)
			raise FileNotFoundError(
				f"Update set '{update_set}' not found for {bird} Trial{trial_num}. "
				f"Available: {options}"
			)
		return matches[0]

	if len(active_dirs) == 1:
		return active_dirs[0]

	unified_matches = [path for path in active_dirs if path.name.lower() == "unified"]
	if len(unified_matches) == 1:
		return unified_matches[0]

	options = ", ".join(path.name for path in active_dirs)
	raise ValueError(
		f"Multiple active update folders found for {bird} Trial{trial_num}: {options}. "
		"Pass update_set=... to choose one explicitly."
	)


def pad_df_to_trial_length(df: pd.DataFrame, total_frames: int, start_frame: int) -> pd.DataFrame:
	"""Pad a subset prediction dataframe to full trial length using NaN rows."""
	full_df = pd.DataFrame(np.nan, index=range(int(total_frames)), columns=df.columns)
	start = int(start_frame)
	stop = int(start + len(df))
	for col in df.columns:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	full_df.iloc[start:stop] = df.to_numpy()
	return full_df


def correct_frames_from_df_fast(
	trial_dir: Path,
	frames_to_correct: list[int] | tuple[int, int],
	pred_df: pd.DataFrame,
	max_match_dist: float = 15.0,
	cam_names: tuple[str, str] = ("Cam1", "Cam2"),
	verbose: bool = True,
) -> pd.DataFrame:
	"""Apply blob-based camera correction over selected frame indices."""
	trial_dir = Path(trial_dir)
	df_corrected = pred_df.copy(deep=True)

	cam_dirs = {cam: trial_dir / cam for cam in cam_names}
	image_paths = {cam: collect_images(folder) for cam, folder in cam_dirs.items()}
	cams = list(image_paths.keys())

	if len(cams) >= 2:
		n0 = len(image_paths[cams[0]])
		for c in cams[1:]:
			if len(image_paths[c]) != n0:
				print(f"Cam mismatch: {cams[0]}={n0}, {c}={len(image_paths[c])}")

	frames = sorted(set(int(x) for x in frames_to_correct))
	valid_frames = [
		i
		for i in frames
		if 0 <= i < len(df_corrected)
		and all(i < len(image_paths[c]) for c in cams)
	]

	if not valid_frames:
		raise ValueError("No valid frames after filtering.")

	t0 = time.perf_counter()

	def process_frame(frame_idx: int) -> tuple[int, dict[str, float]]:
		row = df_corrected.iloc[frame_idx]
		frame_updates: dict[str, float] = {}
		for camera in cams:
			img_path = image_paths[camera][frame_idx]
			with Image.open(img_path) as image_file:
				gray = np.asarray(image_file.convert("L"), dtype=np.float64)
			updates = _correct_camera_frame(
				image=gray,
				row=row,
				camera=camera.lower(),
				max_match_dist=max_match_dist,
			)
			if updates:
				frame_updates.update(updates)
		return frame_idx, frame_updates

	with ThreadPoolExecutor() as executor:
		futures = {executor.submit(process_frame, frame_idx): frame_idx for frame_idx in valid_frames}
		for n, future in enumerate(as_completed(futures), start=1):
			frame_idx, updates = future.result()
			if updates:
				df_corrected.loc[df_corrected.index[frame_idx], list(updates.keys())] = list(updates.values())
			if verbose and (n % 250 == 0 or n == len(valid_frames)):
				print(f"Corrected {n}/{len(valid_frames)} frames")

	elapsed = time.perf_counter() - t0
	if verbose:
		print(f"Done {len(valid_frames)} frames in {elapsed:.1f}s")

	return df_corrected


def train_update_model(
	bird: str,
	trial_num: int,
	snapshot_path: str | Path | None = None,
	epochs: int = 100,
	nframes: int = 30,
	frame_selection_seed: int = 42,
	task: str = "Canari",
	experimenter: str = "Tyler",
	finetune_experimenter: str = "FineTuner",
	base_dir: str | Path = (
		r"C:\Users\Salle-Cineradio\Documents\MachineLearning"
		r"\BirdSongs-MNHN\Testing\ProcessingData"
	),
	update_set: str | None = None,
) -> dict[str, Any]:
	"""Create/update a DLC training dataset for an active-update set and train the model."""
	import DLCsupport as dlcs

	src_trial = _resolve_active_update_dir(
		base_dir=base_dir,
		bird=bird,
		trial_num=trial_num,
		update_set=update_set,
	)

	if snapshot_path is None:
		snapshot_path = BIRD_SNAPSHOT_PATHS.get(str(bird))

	if snapshot_path is not None and not Path(snapshot_path).exists():
		raise FileNotFoundError(f"Snapshot path does not exist:\n{snapshot_path}")

	if not src_trial.exists():
		raise FileNotFoundError(f"Trial folder not found:\n{src_trial}")

	models_dir = src_trial / "ModelsToTune"

	# Check if the ModelsToTune directory exists
	# FIXME: we need this below logic to apply before it prompts you model hyper params (num images, num epochs etc...)
	if models_dir.exists():
		
		# Check if the combined config file exists
		configs = list(models_dir.rglob("config.yaml"))
		if len(configs) > 0:
			return {
				"bird": bird,
				"trial": trial_num,
				"update_set": src_trial.name,
				"config": configs[0],
				"elapsed_seconds": 0,
			}

	models_dir.mkdir(exist_ok=True)

	dummy_video = Path(base_dir) / bird / f"Trial{trial_num}" / "Cam1.avi"
	combined_config = dlcs.create_combined_project_if_missing(
		task=task,
		experimenter=finetune_experimenter,
		combined_project_root=models_dir,
		dummy_video=dummy_video,
	)

	dlcs.apply_bird_bodyparts_to_configs({bird: [combined_config]}, strict=True)

	clean_trial = src_trial / "ModelTraining"
	clean_trial.mkdir(parents=True, exist_ok=True)

	for cam_folder in ["cam1", "cam2", "Cam1", "Cam2"]:
		src = src_trial / cam_folder
		if not src.exists() or not src.is_dir():
			continue
		dst_name = "Cam1" if cam_folder.lower() == "cam1" else "Cam2"
		dst = clean_trial / dst_name
		if dst.exists():
			shutil.rmtree(dst)
		shutil.copytree(src, dst)

	truth_csv = src_trial / "data" / "corrections_autosave.csv"
	if not truth_csv.exists():
		raise FileNotFoundError(f"Missing label file:\n{truth_csv}")

	df = pd.read_csv(truth_csv)
	coord_cols = [c for c in df.columns if re.search(r"_cam[12]_[XY]$", str(c))]
	if len(coord_cols) == 0:
		raise ValueError("No coordinate columns matching *_cam[12]_[XY] found.")

	clean_points_csv = clean_trial / "UpdatedLabels-2Dpoints.csv"
	df[coord_cols].to_csv(clean_points_csv, index=False)

	dataset_name = f"UpdateModelForTrial{trial_num}"
	dlcs.build_combined_dataset(
		combined_config=combined_config,
		data_path=clean_trial,
		dataset_name=dataset_name,
		experimenter=experimenter,
		nframes=nframes,
		frame_selection_seed=frame_selection_seed,
	)

	t0 = time.perf_counter()
	if snapshot_path is not None:
		dlcs.create_and_train(
			config_path=combined_config,
			epochs=epochs,
			snapshot_path=str(snapshot_path),
		)
	else:
		dlcs.create_and_train(
			config_path=combined_config,
			epochs=epochs,
		)
	elapsed = time.perf_counter() - t0

	models_base = combined_config.parent
	for train_dir in models_base.glob("dlc-models-pytorch/iteration-*/*/train"):
		for file in train_dir.glob("*best*.pt"):
			file.unlink()

	print(
		f"\nTraining complete"
		f"\nBird: {bird}"
		f"\nTrial: {trial_num}"
		f"\nUpdate set: {src_trial.name}"
		f"\nElapsed: {elapsed:.1f} sec ({elapsed / 60:.1f} min)"
	)

	return {
		"bird": bird,
		"trial": trial_num,
		"update_set": src_trial.name,
		"config": combined_config,
		"elapsed_seconds": elapsed,
	}


def _predict_with_updated_model(
	bird: str,
	trial_num: int,
	frame_boundary: list[int],
	model_name: str = "ModelUpdateMonday",
	batchsize: int = 16,
	base_dir: str | Path = (
		r"C:\Users\Salle-Cineradio\Documents\MachineLearning"
		r"\BirdSongs-MNHN\Testing\ProcessingData"
	),
	update_set: str | None = None,
	# FIXME: fps needs to be added to the update inputs
	fps = 500,
) -> dict[str, Any]:
	"""Run DLC inference with newest trained model and write corrected full-length predictions."""
	import deeplabcut as dlc
	import xrommtools_copy as xt
	dc = _load_data_converter_module()
	import cv2 

	base = Path(base_dir)
	trial_dir = base / bird / f"Trial{trial_num}"
	src_trial = _resolve_active_update_dir(base_dir=base, bird=bird, trial_num=trial_num, update_set=update_set)

	models_dir = src_trial / "ModelsToTune"
	configs = list(models_dir.rglob("config.yaml"))
	if len(configs) == 0:
		raise FileNotFoundError(f"No config.yaml found in:\n{models_dir}")

	config_path = max(configs, key=lambda p: p.stat().st_mtime)

	start_frame, end_frame = frame_boundary
	cam1 = trial_dir / "Cam1.avi"
	cam2 = trial_dir / "Cam2.avi"
	cam1_dir = trial_dir / "Cam1"
	total_frames = len(list(cam1_dir.glob("*.jpg")))
	cam2_dir = trial_dir / "Cam2"

	if not cam1.exists():
		raise FileNotFoundError(cam1)

	if not cam2.exists():
		raise FileNotFoundError(cam2)
	
	cap = cv2.VideoCapture(cam1)
	num_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if num_vid_frames > end_frame - start_frame:
		print("remaking videos to fit the frame boundries defined")
		dc.jpg_stack_to_avi(input_folder=cam1_dir, output_path=cam1, fps=int(fps), start_frame = start_frame, end_frame = end_frame)
		dc.jpg_stack_to_avi(input_folder=cam2_dir, output_path=cam2, fps=int(fps), start_frame = start_frame, end_frame = end_frame)
	else:
		print("Cine Image Conversions Found")
	print("Now To Analyze the videos with existing model!")

	# cam1 = trial_dir / "Cam1.avi"
	# cam2 = trial_dir / "Cam2.avi"
	# cam1_dir = trial_dir / "Cam1"
	# total_frames = len(list(cam1_dir.glob("*.jpg")))

	# if not cam1.exists() or not cam2.exists():
	# 	raise FileNotFoundError("Cam1.avi and Cam2.avi are required for updated model prediction.")

	updated_dir = trial_dir / "Updated"
	updated_dir.mkdir(exist_ok=True)

	if not sorted(updated_dir.glob("Cam1DLC*.csv"), key=lambda path: path.stat().st_mtime):
		dlc.analyze_videos(
			str(config_path),
			[str(cam1), str(cam2)],
			save_as_csv=True,
			destfolder=str(updated_dir),
			batchsize=int(batchsize),
		)

	cam1_candidates = sorted(updated_dir.glob("Cam1DLC*.csv"), key=lambda path: path.stat().st_mtime)
	cam2_candidates = sorted(updated_dir.glob("Cam2DLC*.csv"), key=lambda path: path.stat().st_mtime)
	if not cam1_candidates or not cam2_candidates:
		raise FileNotFoundError(f"Could not find Cam1DLC*.csv / Cam2DLC*.csv in {updated_dir} after prediction.")

	cam1_dlc_csv = cam1_candidates[-1]
	cam2_dlc_csv = cam2_candidates[-1]
	pred_df = xt.dlc_to_xma(str(cam1_dlc_csv), str(cam2_dlc_csv), model_name, str(updated_dir))
	_cleanup_intermediate_dlc_outputs(updated_dir)

	start_frame = int(frame_boundary[0])
	end_frame = int(frame_boundary[1])
	nan_padded_df = pad_df_to_trial_length(pred_df, total_frames=total_frames, start_frame=start_frame)

	corrected_df = correct_frames_from_df_fast(
		trial_dir=trial_dir,
		pred_df=nan_padded_df,
		frames_to_correct=list(range(start_frame, end_frame)),
		max_match_dist=15.0,
	)

	pred_csv = trial_dir / "Updated_Predictions.csv"
	corrected_df.to_csv(pred_csv, na_rep="NaN", index=False)

	return {
		"config": config_path,
		"update_set": src_trial.name,
		"Updated Csv": pred_csv,
		"output_dir": updated_dir,
	}


def train_and_predict_from_corrections(
	bird: str,
	trial_num: int,
	frame_boundary: list[int],
	orig_snapshot_path: str | Path | None = None,
	epochs: int = 100,
	nframes: int = 30,
	frame_selection_seed: int = 42,
	task: str = "Canari",
	experimenter: str = "Tyler",
	finetune_experimenter: str = "FineTuner",
	base_path: str | Path = (
		r"C:\Users\Salle-Cineradio\Documents\MachineLearning"
		r"\BirdSongs-MNHN\Testing\ProcessingData"
	),
	update_set: str | None = None,
	model_name: str = "ModelUpdateMonday",
	batchsize: int = 16,
) -> dict[str, Any]:
	"""Train on corrected subset and run updated prediction pipeline."""
	train_result = train_update_model(
		bird=bird,
		trial_num=trial_num,
		snapshot_path=orig_snapshot_path,
		epochs=epochs,
		nframes=nframes,
		frame_selection_seed=frame_selection_seed,
		task=task,
		experimenter=experimenter,
		finetune_experimenter=finetune_experimenter,
		base_dir=base_path,
		update_set=update_set,
	)
	predict_result = _predict_with_updated_model(
		bird=bird,
		trial_num=trial_num,
		frame_boundary=frame_boundary,
		model_name=model_name,
		batchsize=batchsize,
		base_dir=base_path,
		update_set=update_set,
	)
	return {
		"train": train_result,
		"predict": predict_result,
	}

# MAYBE EXCESSIVE! LOOK AT PREVIOUS ONE
def _Train_and_Predict(
	bird: str,
	trial_num: int,
	frame_boundary: list[int],
	orig_snapshot_path: str | Path | None = None,
	epochs: int = 100,
	nframes: int = 30,
	frame_selection_seed: int = 42,
	task: str = "Canari",
	experimenter: str = "Tyler",
	finetune_experimenter: str = "FineTuner",
	base_path: str | Path = (
		r"C:\Users\Salle-Cineradio\Documents\MachineLearning"
		r"\BirdSongs-MNHN\Testing\ProcessingData"
	),
	update_set: str | None = None,
	model_name: str = "ModelUpdateMonday",
	batchsize: int = 16,
) -> dict[str, Any]:
	"""Notebook-compatible alias for training then predicting updated points."""
	return train_and_predict_from_corrections(
		bird=bird,
		trial_num=trial_num,
		frame_boundary=frame_boundary,
		orig_snapshot_path=orig_snapshot_path,
		epochs=epochs,
		nframes=nframes,
		frame_selection_seed=frame_selection_seed,
		task=task,
		experimenter=experimenter,
		finetune_experimenter=finetune_experimenter,
		base_path=base_path,
		update_set=update_set,
		model_name=model_name,
		batchsize=batchsize,
	)


@dataclass
class OverlayState:
	"""State container for interactive overlay viewer."""
	pred_csv_by_cam: dict[str, Path]
	pred_root: Path
	images_by_cam: dict[str, list[Path]]
	pred_long: pd.DataFrame
	truth_long: pd.DataFrame
	truth_found: bool
	bodypart_color_map: dict[str, str]


def make_postanalysis_overlay_popout(
	prediction_path: str | Path,
	search_truth: bool = False,
	truth_csv_path: str | Path | None = None,
	default_camera: str = "cam1",
	default_frame_pos: int = 1,
	cam1_image_dir: str | Path | None = None,
	cam2_image_dir: str | Path | None = None,
) -> None:
	"""Launch an interactive matplotlib popout to inspect predictions on image stacks."""
	root = Path(prediction_path)
	if root.is_file():
		root = root.parent

	pred_csv_by_cam = _resolve_prediction_csvs(root)

	if cam1_image_dir is not None or cam2_image_dir is not None:
		if cam1_image_dir is None or cam2_image_dir is None:
			raise ValueError("Provide both cam1_image_dir and cam2_image_dir, or neither.")
		cam_dirs = {
			"cam1": Path(cam1_image_dir),
			"cam2": Path(cam2_image_dir),
		}
		for camera_name, camera_dir in cam_dirs.items():
			if not camera_dir.exists() or not camera_dir.is_dir():
				raise FileNotFoundError(f"Image directory for {camera_name} does not exist: {camera_dir}")
	else:
		cam_dirs = _resolve_cam_dirs(root)

	images_by_cam = {
		"cam1": collect_images(cam_dirs["cam1"]),
		"cam2": collect_images(cam_dirs["cam2"]),
	}

	if pred_csv_by_cam["cam1"] == pred_csv_by_cam["cam2"]:
		pred_long = _load_prediction_long(pred_csv_by_cam["cam1"])
	else:
		pred_long = pd.concat(
			[_load_prediction_long(pred_csv_by_cam["cam1"]), _load_prediction_long(pred_csv_by_cam["cam2"])],
			ignore_index=True,
		)

	truth_long = pd.DataFrame(columns=["frame_id", "bodypart", "camera", "x_true", "y_true"])
	truth_found = False

	if truth_csv_path is not None:
		tpath = Path(truth_csv_path)
		if not tpath.exists():
			raise FileNotFoundError(f"Provided truth CSV does not exist: {tpath}")
		truth_long = _truth_csv_to_long(tpath)
		truth_found = not truth_long.empty
	elif search_truth:
		tpath = _auto_find_truth_csv(root)
		if tpath is not None:
			truth_long = _truth_csv_to_long(tpath)
			truth_found = not truth_long.empty

	state = OverlayState(
		pred_csv_by_cam=pred_csv_by_cam,
		pred_root=root,
		images_by_cam=images_by_cam,
		pred_long=pred_long,
		truth_long=truth_long,
		truth_found=truth_found,
		bodypart_color_map={},
	)

	all_bodyparts = sorted(pd.Index(state.pred_long["bodypart"].dropna().astype(str)).unique().tolist())
	if not state.truth_long.empty:
		truth_parts = pd.Index(state.truth_long["bodypart"].dropna().astype(str)).unique().tolist()
		all_bodyparts = sorted(set(all_bodyparts).union(set(truth_parts)))
	palette = plt.cm.get_cmap("tab20", max(len(all_bodyparts), 1))
	state.bodypart_color_map = {bp: mcolors.to_hex(palette(i)) for i, bp in enumerate(all_bodyparts)}

	cam = _cam_norm(default_camera)
	max_len = max(1, len(state.images_by_cam.get(cam, []))) - 1
	start_pos = min(max(0, int(default_frame_pos)), max_len)
	current_points_by_cam: dict[str, list[dict[str, Any]]] = {"cam1": [], "cam2": []}
	selected_text_by_cam: dict[str, Any | None] = {"cam1": None, "cam2": None}
	view_limits_by_cam: dict[str, dict[str, tuple[float, float] | None]] = {
		"cam1": {"xlim": None, "ylim": None},
		"cam2": {"xlim": None, "ylim": None},
	}
	rng = np.random.default_rng()
	selected_frames: list[int] = []
	selected_frame_index: int | None = None
	selected_frame_meta: dict[int, dict[str, Any]] = {}
	selection_marker_artists: list[Any] = []
	selection_controls_visible = False
	selection_mode = ""
	selection_summary_lines: list[str] = []
	selection_source_camera = cam
	processing_root: Path | None = None
	current_bird: str | None = None
	current_trial_num: int | None = None
	dino_components: dict[str, Any] = {}
	dino_embeddings_by_cam: dict[str, Any] = {}

	fig, main_axes = plt.subplots(1, 2, figsize=(13.8, 8.4))
	main_axes_map = {"cam1": main_axes[0], "cam2": main_axes[1]}
	fig.subplots_adjust(left=0.08, right=0.8, bottom=0.35)

	ax_correction_tab = fig.add_axes([0.82, 0.92, 0.16, 0.05])
	ax_switch_bird = fig.add_axes([0.01, 0.94, 0.09, 0.045])
	ax_switch_trial = fig.add_axes([0.11, 0.94, 0.09, 0.045])
	ax_checks = fig.add_axes([0.82, 0.49, 0.16, 0.19])
	ax_select_random = fig.add_axes([0.82, 0.20, 0.16, 0.04])
	ax_select_displacement = fig.add_axes([0.82, 0.15, 0.16, 0.04])
	ax_select_dino = fig.add_axes([0.82, 0.10, 0.16, 0.04])
	ax_correction = fig.add_axes([0.82, 0.05, 0.16, 0.04])
	ax_prev = fig.add_axes([0.015, 0.252, 0.045, 0.05])
	ax_frame = fig.add_axes([0.08, 0.26, 0.62, 0.03])
	ax_next = fig.add_axes([0.705, 0.252, 0.045, 0.05])
	ax_resample = fig.add_axes([0.755, 0.252, 0.055, 0.05])
	# ax_like = fig.add_axes([0.08, 0.21, 0.62, 0.03])
	ax_pred_size = fig.add_axes([0.08, 0.16, 0.29, 0.03])
	ax_pred_alpha = fig.add_axes([0.46, 0.16, 0.29, 0.03])
	ax_pred_color = fig.add_axes([0.82, 0.40, 0.16, 0.035])
	ax_nframes = fig.add_axes([0.82, 0.345, 0.16, 0.035])
	ax_window_start = fig.add_axes([0.82, 0.685, 0.060, 0.03])
	ax_window_end = fig.add_axes([0.905, 0.685, 0.075, 0.03])
	ax_color_mode = fig.add_axes([0.82, 0.25, 0.16, 0.08])
	ax_selection_info = fig.add_axes([0.82, 0.005, 0.16, 0.035])
	ax_info_frame = fig.add_axes([0, 0.255, 0.022, 0.035])
	# ax_info_like = fig.add_axes([0.045, 0.205, 0.022, 0.035])
	ax_info_nframes = fig.add_axes([0.79, 0.345, 0.022, 0.035])
	ax_info_range = fig.add_axes([0.975, 0.708, 0.018, 0.026])

	check = CheckButtons(ax_checks, ["Show pred", "Annotate"], [True, False])
	btn_correction_tab = Button(ax_correction_tab, "Correction Tab")
	btn_switch_bird = Button(ax_switch_bird, "Bird")
	btn_switch_trial = Button(ax_switch_trial, "Trial")
	btn_select_random = Button(ax_select_random, "Select Random Frames")
	btn_select_displacement = Button(ax_select_displacement, "Select Displacement")
	btn_select_dino = Button(ax_select_dino, "Select DINO Frames")
	btn_correction = Button(ax_correction, "Save Correction Frames")
	# button_use_previous = Button(
	# 	description="Use Prior Frame",
	# 	button_style="info",
	# )

	btn_prev = Button(ax_prev, "<")
	slider_frame = Slider(ax_frame, "frame", 0, float(max_len), valinit=float(start_pos), valstep=1)
	btn_next = Button(ax_next, ">")
	btn_resample = Button(ax_resample, "R20")
	# slider_like = Slider(ax_like, "min_likelihood", 0.0, 1.0, valinit=0.0, valstep=0.01)
	slider_pred_size = Slider(ax_pred_size, "Size", 10, 300, valinit=60, valstep=1)
	slider_pred_alpha = Slider(ax_pred_alpha, "Alpha", 0.05, 1.0, valinit=0.9, valstep=0.01)
	textbox_pred_color = TextBox(ax_pred_color, "Color", initial="deepskyblue")
	textbox_nframes = TextBox(ax_nframes, "ToCorrect", initial="30")
	textbox_window_start = TextBox(ax_window_start, "Start", initial="0")
	textbox_window_end = TextBox(ax_window_end, "End", initial="1")
	radio_color_mode = RadioButtons(ax_color_mode, ["fixed", "by_name"], active=1)
	ax_selection_info.set_axis_off()
	range_title_text = fig.text(0.82, 0.72, "Select Frame Range", fontsize=9, fontweight="bold")

	info_buttons: list[Button] = []

	def _add_info_bubble(info_ax: Any, message: str, figure:str) -> Button:
		btn = Button(info_ax, "(i)")
		btn.color = "#6f54f4"
		btn.hovercolor = "#3d8cd6"
		btn.label.set_fontsize(10)

		def _on_info(_event: Any) -> None:
			try:
				import tkinter as tk
				from tkinter import messagebox
				popup_root = figure.canvas.manager.window

				# popup_root = tk.Tk()
				# popup_root.withdraw()
				# popup_root.attributes("-topmost", True)
				messagebox.showinfo(parent=popup_root,
					title="Information",
					message=message)
				# popup_root.destroy()
			except Exception:
				print(f"Info: {message}")

		btn.on_clicked(_on_info)
		info_buttons.append(btn)
		return btn

	_add_info_bubble(ax_info_frame, 
				  "Frame selector: move through synchronized image frames.",
				  figure = fig)
	# _add_info_bubble(ax_info_like, "Min likelihood: hide points with lower confidence.")
	_add_info_bubble(ax_info_nframes, "Correction set size: choose 30, 40, or 50 frames.",
				  figure = fig)
	_add_info_bubble(
		ax_info_range,
		"Many of the frames will be black, so select the range where they are colorred. This is usually around 2500 frames.",
		figure = fig)

	def _layout_main_axes() -> None:
		fig_w, _ = fig.get_size_inches()
		if fig_w >= 12.0:
			right_x = 0.82
			right_w = 0.16
			left_info_x = 0.045
		else:
			right_x = 0.78
			right_w = 0.20
			left_info_x = 0.03

		fig.subplots_adjust(left=0.08, right=max(0.76, right_x - 0.02), bottom=0.35)

		ax_correction_tab.set_position([right_x, 0.92, right_w, 0.05])
		ax_switch_bird.set_position([0.01, 0.94, 0.09, 0.045])
		ax_switch_trial.set_position([0.11, 0.94, 0.09, 0.045])
		ax_checks.set_position([right_x, 0.49, right_w, 0.19])
		ax_select_random.set_position([right_x, 0.20, right_w, 0.04])
		ax_select_displacement.set_position([right_x, 0.15, right_w, 0.04])
		ax_select_dino.set_position([right_x, 0.10, right_w, 0.04])
		ax_correction.set_position([right_x, 0.05, right_w, 0.04])
		ax_pred_color.set_position([right_x, 0.40, right_w, 0.035])
		ax_nframes.set_position([right_x, 0.345, right_w, 0.035])
		ax_window_start.set_position([right_x, 0.685, right_w * 0.45, 0.03])
		ax_window_end.set_position([right_x + right_w * 0.60, 0.685, right_w * 0.50, 0.03])
		ax_color_mode.set_position([right_x, 0.25, right_w, 0.08])
		ax_selection_info.set_position([right_x, 0.005, right_w, 0.035])

		ax_info_frame.set_position([0.0, 0.255, 0.022, 0.035])
		# ax_info_like.set_position([left_info_x, 0.205, 0.022, 0.035])
		ax_info_nframes.set_position([min(0.98, right_x + right_w + 0.004), 0.345, 0.022, 0.035])
		ax_info_range.set_position([min(0.98, right_x + right_w + 0.004), 0.708, 0.018, 0.026])

		range_title_text.set_position((right_x, 0.72))

	def _scaled_font(figure_obj: Any, base_size: float, ref_w: float, ref_h: float, min_size: float, max_size: float) -> float:
		fig_w, fig_h = figure_obj.get_size_inches()
		scale = float(np.sqrt((fig_w * fig_h) / max(ref_w * ref_h, 1e-6)))
		return float(np.clip(base_size * scale, min_size, max_size))

	def _apply_main_widget_scaling(_event: Any = None) -> None:
		_layout_main_axes()
		button_font = _scaled_font(fig, base_size=9.5, ref_w=12.2, ref_h=8.4, min_size=8.0, max_size=16.0)
		control_font = _scaled_font(fig, base_size=9.0, ref_w=12.2, ref_h=8.4, min_size=7.5, max_size=14.0)
		value_font = _scaled_font(fig, base_size=8.5, ref_w=12.2, ref_h=8.4, min_size=7.0, max_size=13.0)

		for btn in (btn_correction_tab, btn_switch_bird, btn_switch_trial, btn_select_random, btn_select_displacement, btn_select_dino, btn_correction, btn_prev, btn_next, btn_resample):
			btn.label.set_fontsize(button_font)

		for txt in check.labels:
			txt.set_fontsize(control_font)
		for txt in radio_color_mode.labels:
			txt.set_fontsize(control_font)

		for sld in (slider_frame, slider_pred_size, slider_pred_alpha):
			# slider_like
			sld.label.set_fontsize(control_font)
			sld.valtext.set_fontsize(value_font)

		for txtbox in (textbox_pred_color, textbox_nframes):
			txtbox.label.set_fontsize(control_font)
			txtbox.text_disp.set_fontsize(value_font)

		for txtbox in (textbox_window_start, textbox_window_end):
			txtbox.label.set_fontsize(control_font)
			txtbox.text_disp.set_fontsize(value_font)

		range_title_text.set_fontsize(control_font)

		fig.canvas.draw_idle()

	_apply_main_widget_scaling()
	fig.canvas.mpl_connect("resize_event", _apply_main_widget_scaling)

	def _safe_color(value: str, fallback: str) -> str:
		try:
			mcolors.to_rgba(value)
			return value
		except Exception:
			return fallback

	def _load_dino_components() -> dict[str, Any]:
		if dino_components:
			return dino_components

		try:
			import torch
			from torchvision import transforms
		except Exception as exc:
			raise RuntimeError(
				"DINO selection requires torch and torchvision in the active environment."
			) from exc

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
		dino_components.update(
			{
				"torch": torch,
				"transform": transform,
				"model": model,
				"device": device,
			}
		)
		return dino_components

	def _build_frame_alignment(ids_series: pd.Series, image_paths: list[Path]) -> dict[str, Any]:
		ids = {
			int(v)
			for v in pd.to_numeric(ids_series, errors="coerce").dropna().astype(int).tolist()
		}
		tokens = [parse_frame_number_from_stem(p.stem) for p in image_paths]
		if not ids:
			return {"name": "pos", "score": 0.0}

		candidates: list[tuple[str, Any]] = [
			("token", lambda pos, token: int(token) if token is not None else None),
			("token+1", lambda pos, token: int(token) + 1 if token is not None else None),
			("token-1", lambda pos, token: int(token) - 1 if token is not None else None),
			("pos", lambda pos, token: int(pos)),
			("pos+1", lambda pos, token: int(pos) + 1),
		]

		best_name = "pos"
		best_score = -1.0
		for name, fn in candidates:
			valid = 0
			hits = 0
			for pos, token in enumerate(tokens):
				mapped = fn(int(pos), token)
				if mapped is None:
					continue
				valid += 1
				if int(mapped) in ids:
					hits += 1
			score = float(hits) / float(valid) if valid > 0 else 0.0
			if score > best_score:
				best_score = score
				best_name = name

		return {"name": str(best_name), "score": float(best_score)}

	def _apply_frame_alignment(frame_pos: int, image_path: Path, alignment: dict[str, Any]) -> int:
		token = parse_frame_number_from_stem(image_path.stem)
		mode = str(alignment.get("name", "pos"))
		if mode == "token" and token is not None:
			return int(token)
		if mode == "token+1" and token is not None:
			return int(token) + 1
		if mode == "token-1" and token is not None:
			return int(token) - 1
		if mode == "pos+1":
			return int(frame_pos) + 1
		return int(frame_pos)

	pred_alignment_by_cam: dict[str, dict[str, Any]] = {}
	truth_alignment_by_cam: dict[str, dict[str, Any]] = {}

	def _recompute_alignments(verbose: bool = True) -> None:
		pred_alignment_by_cam.clear()
		truth_alignment_by_cam.clear()

		for _camera_name in ("cam1", "cam2"):
			cam_images = state.images_by_cam.get(_camera_name, [])
			pred_cam = state.pred_long[state.pred_long["camera"] == _camera_name]
			if pred_cam.empty:
				pred_alignment_by_cam[_camera_name] = {"name": "pos", "score": 0.0}
			else:
				pred_alignment_by_cam[_camera_name] = _build_frame_alignment(pred_cam["frame_id"], cam_images)

			if state.truth_long.empty:
				truth_alignment_by_cam[_camera_name] = pred_alignment_by_cam[_camera_name]
			else:
				truth_cam = state.truth_long[state.truth_long["camera"] == _camera_name]
				if truth_cam.empty:
					truth_alignment_by_cam[_camera_name] = pred_alignment_by_cam[_camera_name]
				else:
					truth_alignment_by_cam[_camera_name] = _build_frame_alignment(truth_cam["frame_id"], cam_images)

		if verbose:
			for _camera_name in ("cam1", "cam2"):
				_pred_mode = pred_alignment_by_cam.get(_camera_name, {}).get("name", "pos")
				_pred_score = pred_alignment_by_cam.get(_camera_name, {}).get("score", 0.0)
				_truth_mode = truth_alignment_by_cam.get(_camera_name, {}).get("name", "pos")
				_truth_score = truth_alignment_by_cam.get(_camera_name, {}).get("score", 0.0)
				print(
					f"Frame alignment {_camera_name}: pred={_pred_mode} ({_pred_score:.3f}), "
					f"truth={_truth_mode} ({_truth_score:.3f})"
				)

	_recompute_alignments(verbose=True)

	def _init_context_from_state() -> None:
		nonlocal processing_root
		nonlocal current_bird
		nonlocal current_trial_num
		try:
			ctx = _extract_trial_context_from_path(state.pred_root)
			processing_root = Path(ctx["base_dir"])
			current_bird = str(ctx["bird"])
			current_trial_num = int(ctx["trial_num"])
		except Exception:
			processing_root = None
			current_bird = None
			current_trial_num = None

	def _ensure_processing_root() -> Path | None:
		nonlocal processing_root
		if processing_root is not None and processing_root.exists():
			return processing_root
		try:
			import tkinter as tk
			from tkinter import filedialog, messagebox
		except Exception:
			return None
		root_picker = tk.Tk()
		root_picker.withdraw()
		root_picker.attributes("-topmost", True)
		messagebox.showinfo(
			title="Select ProcessingData",
			message="Select the parent ProcessingData folder containing bird folders.",
			parent=root_picker,
		)
		picked = filedialog.askdirectory(title="Select ProcessingData folder", parent=root_picker)
		root_picker.destroy()
		if not picked:
			return None
		processing_root = Path(picked)
		return processing_root

	def _list_birds(base_dir: Path) -> list[str]:
		birds: list[str] = []
		for p in sorted([d for d in base_dir.iterdir() if d.is_dir()]):
			if any(re.fullmatch(r"Trial\d+", t.name, flags=re.IGNORECASE) for t in p.iterdir() if t.is_dir()):
				birds.append(p.name)
		return birds

	def _list_trials(base_dir: Path, bird_name: str) -> list[int]:
		bird_dir = base_dir / bird_name
		trials: list[int] = []
		if not bird_dir.exists() or not bird_dir.is_dir():
			return trials
		for p in sorted([d for d in bird_dir.iterdir() if d.is_dir()]):
			m = re.fullmatch(r"Trial(\d+)", p.name, flags=re.IGNORECASE)
			if m is None:
				continue
			trials.append(int(m.group(1)))
		return sorted(trials)

	def _rebuild_bodypart_colors() -> None:
		all_bodyparts_local = sorted(pd.Index(state.pred_long["bodypart"].dropna().astype(str)).unique().tolist())
		if not state.truth_long.empty:
			truth_parts = pd.Index(state.truth_long["bodypart"].dropna().astype(str)).unique().tolist()
			all_bodyparts_local = sorted(set(all_bodyparts_local).union(set(truth_parts)))
		palette_local = plt.cm.get_cmap("tab20", max(len(all_bodyparts_local), 1))
		state.bodypart_color_map = {bp: mcolors.to_hex(palette_local(i)) for i, bp in enumerate(all_bodyparts_local)}

	def _reload_trial_into_view(trial_dir: Path) -> None:
		nonlocal selected_frames
		nonlocal selected_frame_index
		nonlocal selected_frame_meta
		nonlocal selection_mode
		nonlocal selection_summary_lines
		nonlocal selection_source_camera
		trial_dir = Path(trial_dir)
		pred_csv_map = _resolve_prediction_csvs(trial_dir)
		cam_dirs_local = _resolve_cam_dirs(trial_dir)
		images_map = {
			"cam1": collect_images(cam_dirs_local["cam1"]),
			"cam2": collect_images(cam_dirs_local["cam2"]),
		}
		if pred_csv_map["cam1"] == pred_csv_map["cam2"]:
			pred_long_local = _load_prediction_long(pred_csv_map["cam1"])
		else:
			pred_long_local = pd.concat(
				[_load_prediction_long(pred_csv_map["cam1"]), _load_prediction_long(pred_csv_map["cam2"])],
				ignore_index=True,
			)

		truth_local = pd.DataFrame(columns=["frame_id", "bodypart", "camera", "x_true", "y_true"])
		truth_local_found = False
		if truth_csv_path is not None and Path(truth_csv_path).exists():
			truth_local = _truth_csv_to_long(Path(truth_csv_path))
			truth_local_found = not truth_local.empty
		elif search_truth:
			tpath = _auto_find_truth_csv(trial_dir)
			if tpath is not None:
				truth_local = _truth_csv_to_long(tpath)
				truth_local_found = not truth_local.empty

		state.pred_root = trial_dir
		state.pred_csv_by_cam = pred_csv_map
		state.images_by_cam = images_map
		state.pred_long = pred_long_local
		state.truth_long = truth_local
		state.truth_found = truth_local_found
		_rebuild_bodypart_colors()
		_recompute_alignments(verbose=True)

		selected_frames = []
		selected_frame_index = None
		selected_frame_meta = {}
		selection_mode = ""
		selection_summary_lines = []
		selection_source_camera = "cam1"
		_set_selection_controls_visible(False)

		shared_n = max(1, _shared_frame_limit()) - 1
		slider_frame.valmax = float(shared_n)
		if slider_frame.val > shared_n:
			slider_frame.set_val(float(shared_n))
		redraw()

	def _choose_bird_dialog() -> str | None:
		base_dir = _ensure_processing_root()
		if base_dir is None:
			return None
		birds = _list_birds(base_dir)
		if not birds:
			print(f"No bird folders with Trial* found under: {base_dir}")
			return None
		try:
			import tkinter as tk
			from tkinter import simpledialog
		except Exception:
			return None
		root_prompt = tk.Tk()
		root_prompt.withdraw()
		root_prompt.attributes("-topmost", True)
		choice = simpledialog.askstring(
			"Select Bird",
			"Available birds:\n" + "\n".join(birds) + "\n\nEnter bird name:",
			initialvalue=current_bird or birds[0],
			parent=root_prompt,
		)
		root_prompt.destroy()
		if choice is None:
			return None
		choice = str(choice).strip()
		if choice not in birds:
			print(f"Bird '{choice}' not found under {base_dir}")
			return None
		return choice

	def _choose_trial_dialog(bird_name: str) -> int | None:
		base_dir = _ensure_processing_root()
		if base_dir is None:
			return None
		trials = _list_trials(base_dir, bird_name)
		if not trials:
			print(f"No Trial folders found for bird {bird_name} under {base_dir}")
			return None
		try:
			import tkinter as tk
			from tkinter import simpledialog
		except Exception:
			return None
		root_prompt = tk.Tk()
		root_prompt.withdraw()
		root_prompt.attributes("-topmost", True)
		choice = simpledialog.askinteger(
			"Select Trial",
			f"Available trials for {bird_name}:\n" + ", ".join(str(t) for t in trials) + "\n\nEnter trial number:",
			initialvalue=current_trial_num or trials[0],
			minvalue=min(trials),
			maxvalue=max(trials),
			parent=root_prompt,
		)
		root_prompt.destroy()
		if choice is None:
			return None
		if int(choice) not in trials:
			print(f"Trial {choice} is not available for bird {bird_name}")
			return None
		return int(choice)

	_init_context_from_state()

	def _frame_pos_to_pred_id(camera: str, frame_pos: int) -> int:
		images = state.images_by_cam.get(camera, [])
		if not (0 <= int(frame_pos) < len(images)):
			return int(frame_pos)
		return _apply_frame_alignment(int(frame_pos), images[int(frame_pos)], pred_alignment_by_cam.get(camera, {"name": "pos"}))

	def _frame_pos_to_truth_id(camera: str, frame_pos: int) -> int:
		images = state.images_by_cam.get(camera, [])
		if not (0 <= int(frame_pos) < len(images)):
			return int(frame_pos)
		return _apply_frame_alignment(int(frame_pos), images[int(frame_pos)], truth_alignment_by_cam.get(camera, {"name": "pos"}))

	def _pred_points(camera: str, frame_pos: int) -> pd.DataFrame:
		mapped_frame_id = _frame_pos_to_pred_id(camera, int(frame_pos))
		d = state.pred_long[
			(state.pred_long["camera"] == camera) & (state.pred_long["frame_id"] == int(mapped_frame_id))
		].copy()
		return d[["bodypart", "x_pred", "y_pred", "likelihood"]].rename(columns={"x_pred": "x", "y_pred": "y"})

	def _true_points(camera: str, frame_pos: int) -> pd.DataFrame:
		if state.truth_long.empty:
			return pd.DataFrame(columns=["bodypart", "x", "y"])
		mapped_frame_id = _frame_pos_to_truth_id(camera, int(frame_pos))
		d = state.truth_long[
			(state.truth_long["camera"] == camera) & (state.truth_long["frame_id"] == int(mapped_frame_id))
		].copy()
		return d[["bodypart", "x_true", "y_true"]].rename(columns={"x_true": "x", "y_true": "y"})

	def _shared_frame_limit() -> int:
		counts = [len(v) for v in state.images_by_cam.values() if len(v) > 0]
		return min(counts) if counts else 0

	def _build_zone_specs(frame_limit: int) -> list[dict[str, Any]]:
		if frame_limit <= 0:
			return []

		starts = [0]
		if frame_limit > 399:
			starts.extend(range(399, frame_limit, 400))

		zones: list[dict[str, Any]] = []
		for idx, start in enumerate(starts):
			end = starts[idx + 1] if idx + 1 < len(starts) else frame_limit
			end = max(start + 1, min(end, frame_limit))
			zones.append(
				{
					"zone_index": idx,
					"start": int(start),
					"end": int(end),
					"label": f"Z{idx + 1} {start + 1}-{end}",
				}
			)
		return zones

	def _frame_displacement(camera: str, start_frame: int, end_frame: int) -> float | None:
		start_pts = _pred_points(camera, start_frame)
		end_pts = _pred_points(camera, end_frame)
		if start_pts.empty or end_pts.empty:
			return None

		merged = start_pts.merge(end_pts, on="bodypart", suffixes=("_start", "_end"))
		if merged.empty:
			return None

		dx = pd.to_numeric(merged["x_end"], errors="coerce") - pd.to_numeric(merged["x_start"], errors="coerce")
		dy = pd.to_numeric(merged["y_end"], errors="coerce") - pd.to_numeric(merged["y_start"], errors="coerce")
		dist = np.sqrt(np.square(dx) + np.square(dy))
		valid = dist[np.isfinite(dist)]
		if len(valid) == 0:
			return None
		return float(np.mean(valid))

	def _current_selection_window() -> tuple[int, int]:
		frame_limit = _shared_frame_limit()
		if frame_limit <= 0:
			return 0, -1

		default_start = 0
		default_end = int(frame_limit - 1)

		try:
			window_start = int(float(str(textbox_window_start.text).strip()))
		except Exception:
			window_start = default_start
		try:
			window_end = int(float(str(textbox_window_end.text).strip()))
		except Exception:
			window_end = default_end

		window_start = int(np.clip(window_start, 0, default_end))
		window_end = int(np.clip(window_end, 0, default_end))
		if window_start > window_end:
			window_start, window_end = window_end, window_start

		textbox_window_start.set_val(str(window_start))
		textbox_window_end.set_val(str(window_end))
		return window_start, window_end

	def _window_frame_candidates() -> np.ndarray:
		window_start, window_end = _current_selection_window()
		if window_end < window_start:
			return np.asarray([], dtype=int)
		return np.arange(window_start, window_end + 1, dtype=int)

	def _target_correction_count() -> int:
		allowed = np.asarray([30, 40, 50, 100,  300], dtype=int)
		try:
			requested = int(float(str(textbox_nframes.text).strip()))
		except Exception:
			requested = 30
		chosen = int(allowed[np.argmin(np.abs(allowed - requested))])
		if str(textbox_nframes.text).strip() != str(chosen):
			textbox_nframes.set_val(str(chosen))
		return chosen

	def _allocate_counts(weights: list[float], total_count: int) -> list[int]:
		if total_count <= 0 or not weights:
			return [0 for _ in weights]

		weights_arr = np.asarray(weights, dtype=float)
		weights_arr = np.where(np.isfinite(weights_arr) & (weights_arr > 0), weights_arr, 0.0)
		if float(np.sum(weights_arr)) <= 0:
			weights_arr = np.ones(len(weights), dtype=float)
		weights_arr = weights_arr / float(np.sum(weights_arr))

		raw = weights_arr * int(total_count)
		counts = np.floor(raw).astype(int)
		remaining = int(total_count - np.sum(counts))
		if remaining > 0:
			remainders = raw - counts
			for idx in np.argsort(-remainders)[:remaining]:
				counts[int(idx)] += 1
		return counts.tolist()

	def _zone_for_frame(frame_idx: int, zones: list[dict[str, Any]]) -> dict[str, Any] | None:
		for zone in zones:
			if zone["start"] <= frame_idx < zone["end"]:
				return zone
		return zones[-1] if zones else None

	def _build_random_selection() -> tuple[list[int], dict[int, dict[str, Any]], list[str]]:
		candidates = _window_frame_candidates()
		count = min(_target_correction_count(), int(candidates.size))
		if count <= 0:
			return [], {}, ["Random", "No frames in window"]

		window_start, window_end = _current_selection_window()
		frames = np.sort(rng.choice(candidates, size=count, replace=False)).tolist()
		meta = {int(frame): {"label": "Random sample", "color": "crimson"} for frame in frames}
		return [int(frame) for frame in frames], meta, [f"Random {count} frames", f"Window {window_start}-{window_end}"]

	def _build_displacement_selection() -> tuple[list[int], dict[int, dict[str, Any]], list[str]]:
		window_start, window_end = _current_selection_window()
		frame_limit = int(max(0, window_end - window_start + 1))
		count = min(_target_correction_count(), frame_limit)
		zones = _build_zone_specs(frame_limit)
		if count <= 0 or not zones:
			return [], {}, ["Disp", "No frames in window"]

		zone_scores: list[float] = []
		for zone in zones:
			start_frame = int(window_start + zone["start"])
			end_frame = int(min(window_end, window_start + zone["start"] + 400))
			camera_scores: list[float] = []
			for camera_name in ("cam1", "cam2"):
				if end_frame >= len(state.images_by_cam.get(camera_name, [])) or start_frame >= len(state.images_by_cam.get(camera_name, [])):
					continue
				disp = _frame_displacement(camera_name, start_frame, end_frame)
				if disp is not None and np.isfinite(disp):
					camera_scores.append(float(disp))
			zone_scores.append(float(np.mean(camera_scores)) if camera_scores else 0.0)

		zone_counts = _allocate_counts(zone_scores, count)
		total_score = float(np.sum(zone_scores))
		zone_weights = [float(score) / total_score for score in zone_scores] if total_score > 0 else [1.0 / len(zones) for _ in zones]
		palette = plt.cm.get_cmap("tab10", max(len(zones), 1))

		selected: list[int] = []
		meta: dict[int, dict[str, Any]] = {}
		for zone, alloc_count, zone_weight, zone_score, color_idx in zip(zones, zone_counts, zone_weights, zone_scores, range(len(zones))):
			zone_frames = np.arange(window_start + zone["start"], window_start + zone["end"], dtype=int)
			if alloc_count <= 0 or zone_frames.size == 0:
				continue
			pick_count = min(int(alloc_count), int(zone_frames.size))
			picks = np.sort(rng.choice(zone_frames, size=pick_count, replace=False))
			color = mcolors.to_hex(palette(color_idx))
			for frame in picks.tolist():
				selected.append(int(frame))
				meta[int(frame)] = {
					"label": str(zone["label"]),
					"zone_weight": float(zone_weight),
					"avg_displacement": float(zone_score),
					"color": color,
				}

		if len(selected) < count:
			remaining = np.setdiff1d(np.arange(window_start, window_end + 1, dtype=int), np.asarray(selected, dtype=int), assume_unique=False)
			fill_count = min(int(count - len(selected)), int(remaining.size))
			if fill_count > 0:
				fill = np.sort(rng.choice(remaining, size=fill_count, replace=False))
				for frame in fill.tolist():
					zone = _zone_for_frame(int(frame) - window_start, zones)
					color = "gray"
					label = "Fallback"
					zone_weight = 0.0
					zone_score = 0.0
					if zone is not None:
						color = mcolors.to_hex(palette(int(zone["zone_index"])))
						label = str(zone["label"])
						zone_weight = float(zone_weights[int(zone["zone_index"])])
						zone_score = float(zone_scores[int(zone["zone_index"])])
					selected.append(int(frame))
					meta[int(frame)] = {
						"label": label,
						"zone_weight": zone_weight,
						"avg_displacement": zone_score,
						"color": color,
					}

		selected = sorted(set(int(frame) for frame in selected))[:count]
		summary_lines = [f"Disp {count} frames", f"Window {window_start}-{window_end}"]
		for zone, zone_weight, zone_count in zip(zones, zone_weights, zone_counts):
			if zone_count > 0:
				summary_lines.append(f"Z{zone['zone_index'] + 1} {zone_weight:.2f} ({zone_count})")
		if len(summary_lines) > 5:
			summary_lines = summary_lines[:4] + ["..."]
		return selected, meta, summary_lines

	def _compute_dino_embeddings(camera: str) -> Any:
		frame_limit = _shared_frame_limit()
		if frame_limit <= 0:
			return None
		cache_key = f"{camera}:{frame_limit}"
		if cache_key in dino_embeddings_by_cam:
			return dino_embeddings_by_cam[cache_key]

		components = _load_dino_components()
		torch = components["torch"]
		transform = components["transform"]
		model = components["model"]
		device = components["device"]

		image_paths = state.images_by_cam.get(camera, [])[:frame_limit]
		if not image_paths:
			dino_embeddings_by_cam[cache_key] = None
			return None

		embeddings = []
		batch_size = 16
		for start in range(0, len(image_paths), batch_size):
			batch_paths = image_paths[start:start + batch_size]
			batch_tensors = []
			for image_path in batch_paths:
				with Image.open(image_path) as image_file:
					batch_tensors.append(transform(image_file.convert("RGB")))
			batch = torch.stack(batch_tensors).to(device)
			with torch.no_grad():
				feats = model(batch)
			embeddings.append(feats.detach().cpu())

		if not embeddings:
			dino_embeddings_by_cam[cache_key] = None
			return None

		embedding_tensor = torch.cat(embeddings, dim=0)
		embedding_tensor = embedding_tensor / embedding_tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)
		dino_embeddings_by_cam[cache_key] = embedding_tensor
		return embedding_tensor

	def _k_center_greedy(embedding_tensor: Any, k: int) -> list[int]:
		components = _load_dino_components()
		torch = components["torch"]
		n_samples = int(embedding_tensor.shape[0])
		if n_samples <= 0 or k <= 0:
			return []
		if k >= n_samples:
			return list(range(n_samples))

		selected = [int(torch.randint(0, n_samples, (1,)).item())]
		dist = torch.cdist(embedding_tensor, embedding_tensor[selected]).min(dim=1).values
		for _ in range(k - 1):
			idx = int(torch.argmax(dist).item())
			selected.append(idx)
			new_dist = torch.cdist(embedding_tensor, embedding_tensor[[idx]]).squeeze(1)
			dist = torch.minimum(dist, new_dist)
		return selected

	def _build_dino_selection(camera: str) -> tuple[list[int], dict[int, dict[str, Any]], list[str]]:
		candidates = _window_frame_candidates()
		count = min(_target_correction_count(), int(candidates.size))
		if count <= 0:
			return [], {}, ["DINO", "No frames in window"]

		embedding_tensor = _compute_dino_embeddings(camera)
		if embedding_tensor is None:
			return [], {}, ["DINO", f"No embeddings for {camera}"]

		components = _load_dino_components()
		torch = components["torch"]
		candidate_idx = [int(i) for i in candidates.tolist() if 0 <= int(i) < int(embedding_tensor.shape[0])]
		if not candidate_idx:
			return [], {}, ["DINO", "No embeddings in window"]
		embeddings_subset = embedding_tensor[torch.as_tensor(candidate_idx, dtype=torch.long)]
		count = min(count, int(embeddings_subset.shape[0]))
		selected_local = sorted(int(idx) for idx in _k_center_greedy(embeddings_subset, count))
		selected_idx = [int(candidate_idx[idx]) for idx in selected_local]
		window_start, window_end = _current_selection_window()
		meta: dict[int, dict[str, Any]] = {}
		for frame in selected_idx:
			meta[int(frame)] = {
				"label": f"DINO {camera}",
				"source_camera": camera,
				"color": "seagreen",
			}
		return selected_idx, meta, [f"DINO {count} frames", f"Source {camera}", f"Window {window_start}-{window_end}"]

	def _method_folder_name(method_name: str) -> str:
		return {"random": "random", "displacement": "displacement", "dino": "dino"}.get(method_name, method_name)

	def _list_active_update_sets() -> list[str]:
		active_root = state.pred_root / "active_updates"
		if not active_root.exists() or not active_root.is_dir():
			return []
		sets: list[str] = []
		for candidate in sorted([p for p in active_root.iterdir() if p.is_dir()]):
			if (candidate / "cam1").is_dir() and (candidate / "cam2").is_dir():
				sets.append(candidate.name)
		return sets

	def _frames_for_active_update_set(set_name: str) -> list[int]:
		method_dir = state.pred_root / "active_updates" / str(set_name)
		cam_frames: dict[str, set[int]] = {}
		for camera_name in ("cam1", "cam2"):
			camera_dir = method_dir / camera_name
			if not camera_dir.exists() or not camera_dir.is_dir():
				cam_frames[camera_name] = set()
				continue

			subset_images = [p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
			name_to_index = {img_path.name: idx for idx, img_path in enumerate(state.images_by_cam.get(camera_name, []))}
			cam_frames[camera_name] = {
				int(name_to_index[p.name])
				for p in subset_images
				if p.name in name_to_index
			}

		if cam_frames.get("cam1") and cam_frames.get("cam2"):
			frames = sorted(cam_frames["cam1"].intersection(cam_frames["cam2"]))
			if frames:
				return [int(frame) for frame in frames]

		merged = sorted(cam_frames.get("cam1", set()).union(cam_frames.get("cam2", set())))
		return [int(frame) for frame in merged]

	def _export_active_updates_subset(method_name: str, frames: list[int]) -> Path:
		method_dir = state.pred_root / "active_updates" / _method_folder_name(method_name)
		cam1_dir = method_dir / "cam1"
		cam2_dir = method_dir / "cam2"
		data_dir = method_dir / "data"

		if method_dir.exists():
			shutil.rmtree(method_dir)
		cam1_dir.mkdir(parents=True, exist_ok=True)
		cam2_dir.mkdir(parents=True, exist_ok=True)
		data_dir.mkdir(parents=True, exist_ok=True)

		for camera_name, export_dir in (("cam1", cam1_dir), ("cam2", cam2_dir)):
			images = state.images_by_cam.get(camera_name, [])
			for frame in sorted(set(int(frame) for frame in frames)):
				if 0 <= frame < len(images):
					shutil.copy2(images[frame], export_dir / images[frame].name)

		return method_dir

	def _extract_trial_context_from_export_dir(export_dir: Path) -> dict[str, Any]:
		return _extract_trial_context_from_path(export_dir)

	def _prompt_train_predict_inputs(
		export_dir: Path,
		frames_local: list[int],
		default_update_set: str,
	) -> dict[str, Any] | None:
		try:
			import tkinter as tk
			from tkinter import filedialog, messagebox, simpledialog
		except Exception as exc:
			raise RuntimeError("Tkinter prompts are required for train-and-predict confirmation.") from exc

		context = _extract_trial_context_from_export_dir(export_dir)
		bird = str(context["bird"])
		trial_num = int(context["trial_num"])
		saved_range = _load_saved_frame_range(Path(context["trial_dir"]))
		if saved_range is not None:
			default_start, default_end = int(saved_range[0]), int(saved_range[1])
		else:
			default_start = int(min(frames_local)) if frames_local else 0
			default_end = int(max(frames_local)) if frames_local else 0
		snapshot_default = BIRD_SNAPSHOT_PATHS.get(bird)

		root = tk.Tk()
		root.withdraw()
		root.attributes("-topmost", True)

		proceed = bool(
			messagebox.askyesno(
				title="Train And Update Predictions",
				message=(
					f"Bird: {bird}\n"
					f"Trial: {trial_num}\n"
					f"Update set: {default_update_set}\n\n"
					"Are you ready to train and update points?"
				),
			)
		)
		if not proceed:
			root.destroy()
			return None

		start_frame = simpledialog.askinteger(
			"Start Frame",
			"Start frame for correction-aware update:",
			initialvalue=default_start,
			minvalue=0,
			parent=root,
		)
		if start_frame is None:
			root.destroy()
			return None

		end_frame = simpledialog.askinteger(
			"End Frame",
			"End frame for correction-aware update:",
			initialvalue=default_end,
			minvalue=0,
			parent=root,
		)
		if end_frame is None:
			root.destroy()
			return None

		epochs = simpledialog.askinteger(
			"Epochs",
			"Training epochs:",
			initialvalue=30,
			minvalue=1,
			parent=root,
		)
		if epochs is None:
			root.destroy()
			return None

		nframes = simpledialog.askinteger(
			"Frames",
			"Number of frames to sample for model update:",
			initialvalue=30,
			minvalue=1,
			parent=root,
		)
		if nframes is None:
			root.destroy()
			return None

		snapshot_path = snapshot_default
		if snapshot_path is None or not Path(snapshot_path).exists():
			messagebox.showinfo(
				title="Snapshot Needed",
				message=(
					f"No valid default snapshot found for {bird}.\n"
					"Please select a snapshot .pt file."
				),
			)
			picked = filedialog.askopenfilename(
				title="Select Snapshot File",
				filetypes=[("PyTorch Snapshot", "*.pt"), ("All Files", "*.*")],
			)
			if not picked:
				root.destroy()
				return None
			snapshot_path = picked

		root.destroy()

		if int(end_frame) < int(start_frame):
			start_frame, end_frame = int(end_frame), int(start_frame)

		_save_frame_range(
			Path(context["trial_dir"]),
			int(start_frame),
			int(end_frame),
			source="train_and_predict",
		)

		return {
			"bird": bird,
			"trial_num": trial_num,
			"base_path": Path(context["base_dir"]),
			"update_set": str(default_update_set),
			"orig_snapshot_path": str(snapshot_path),
			"frame_boundary": [int(start_frame), int(end_frame)],
			"epochs": int(epochs),
			"nframes": int(nframes),
		}

	def _run_train_and_predict_from_review(export_dir: Path, frames_local: list[int], method_name: str) -> bool:
		params = _prompt_train_predict_inputs(export_dir, frames_local, default_update_set=method_name)
		if params is None:
			return False

		result = _Train_and_Predict(
			bird=str(params["bird"]),
			trial_num=int(params["trial_num"]),
			frame_boundary=list(params["frame_boundary"]),
			orig_snapshot_path=str(params["orig_snapshot_path"]),
			epochs=int(params["epochs"]),
			nframes=int(params["nframes"]),
			base_path=Path(params["base_path"]),
			update_set=str(params["update_set"]),
		)

		pred_csv = Path(result["predict"]["Updated Csv"])
		if not pred_csv.exists():
			raise FileNotFoundError(f"Updated prediction CSV was not created: {pred_csv}")

		state.pred_csv_by_cam = {"cam1": pred_csv, "cam2": pred_csv}
		state.pred_long = _load_prediction_long(pred_csv)
		_recompute_alignments(verbose=True)
		redraw()
		return True

	def _open_correction_subset_view(method_name: str, frames: list[int], export_dir: Path) -> None:
		frames_local = [int(frame) for frame in sorted(set(frames))]
		if not frames_local:
			raise RuntimeError("No selected frames are available for correction review.")
		shared_frame_limit = _shared_frame_limit()
		if shared_frame_limit <= 0:
			raise RuntimeError("No shared frames available for correction review.")
		max_review_frame = int(shared_frame_limit - 1)
		start_review_frame = int(np.clip(frames_local[0], 0, max_review_frame))

		review_fig, review_axes = plt.subplots(1, 2, figsize=(16.0, 9.0), dpi=100)
		review_fig.subplots_adjust(left=0.04, right=0.98, bottom=0.24, top=0.90, wspace=0.05)
		for review_ax in review_axes:
			review_ax.set_axis_off()

		ax_review_prev = review_fig.add_axes([0.1, 0.09, 0.05, 0.05])
		ax_review_slider = review_fig.add_axes([0.21, 0.095, 0.34, 0.035])
		ax_review_next = review_fig.add_axes([0.57, 0.09, 0.05, 0.05])
		# ax_review_current = review_fig.add_axes([0.59, 0.09, 0.11, 0.05])
		ax_review_pred_size = review_fig.add_axes([0.72, 0.095, 0.11, 0.03])
		ax_bulk_snap = review_fig.add_axes([0.62, 0.03, 0.1, 0.055])
		ax_snap_mode = review_fig.add_axes([0.72, 0.03, 0.11, 0.055])
		# ax_zoom_reset = review_fig.add_axes([0.84, 0.045, 0.12, 0.04])
		ax_apply_frame = review_fig.add_axes([0.72, 0.005, 0.24, 0.025])
		ax_prediction_source = review_fig.add_axes([0.84, 0.03, 0.11, 0.055])
		ax_camera_mode = review_fig.add_axes([0.55, 0.03, 0.075, 0.055])
		ax_finish_corrections = review_fig.add_axes([0.10, 0.03, 0.35, 0.055])
		ax_info_updating = review_fig.add_axes([0.67, 0.005, 0.03, 0.025])
		
		radio_camera_mode = RadioButtons(ax_camera_mode, ["Both", "cam1", "cam2"], active=0,)
		btn_review_prev = Button(ax_review_prev, "<")
		btn_review_next = Button(ax_review_next, ">")
		# btn_review_current = Button(ax_review_current, "Current Frame")
		# btn_zoom_reset = Button(ax_zoom_reset, "Reset 30x30")
		btn_apply_frame = Button(ax_apply_frame, "Apply Snap To Frame")
		btn_finish_corrections = Button(ax_finish_corrections, "Are you finished correcting?")
		check_bulk_snap = CheckButtons(ax_bulk_snap, ["Auto snap frame"], [False])
		radio_snap_mode = RadioButtons(ax_snap_mode, ["blob", "darkest"], active=0)
		radio_prediction_source = RadioButtons(ax_prediction_source, ["Current", "Prior frame"], active=0)
		_add_info_bubble(ax_info_updating, 
				   "Correct and Update the Training Data: \n--The Buttons above allow for rapid labeling.\n--Selecting snap to darkest pixel or closest blob will help existing predictions arrive at the center of the marker.\n--Pulling information from the previous frame will allow you to carry over locational information from the previous frame to help you label the current frame.",
				   figure=review_fig)
		ax_info_updating.set_position([0.67, 0.005, 0.03, 0.025])
		slider_review = Slider( 
			ax_review_slider,
			"frame_pos",
			0,
			float(max_review_frame),
			valinit=float(start_review_frame),
			valstep=1,
		)
		slider_review_pred_size = Slider(
			ax_review_pred_size,
			"pred_size",
			10,
			300,
			valinit=float(slider_pred_size.val),
			valstep=1,
		)
		zoom_axes = {
			"cam1": review_axes[0].inset_axes([0.02, 0.02, 0.34, 0.34]),
			"cam2": review_axes[1].inset_axes([0.64, 0.02, 0.34, 0.34]),
		}
		for zax in zoom_axes.values():
			zax.set_facecolor("black")
			zax.set_xticks([])
			zax.set_yticks([])

		correction_cache: dict[tuple[int, str, str], tuple[float, float]] = {}
		correction_pixel_index: dict[tuple[int, str, str], tuple[int, int]] = {}
		# likelihood_lookup: dict[tuple[int, str, str], float] = {}
		active_marker: dict[str, Any] | None = None
		active_selected_index = 0
		drag_state: dict[str, Any] = {"active": False, "camera": None, "bodypart": None, "moved": False}
		zoom_half_window: dict[str, float] = {"cam1": 15.0, "cam2": 15.0}
		last_bulk_apply_signature: tuple[int, str] | None = None
		hit_threshold_px = 12.0
		export_frames = sorted(set(int(frame) for frame in frames_local))
		image_pos_by_cam_name: dict[str, dict[str, int]] = {
			"cam1": {p.name: int(i) for i, p in enumerate(state.images_by_cam.get("cam1", []))},
			"cam2": {p.name: int(i) for i, p in enumerate(state.images_by_cam.get("cam2", []))},
		}

		def _resolve_frame_pos_for_saved_row(camera_name: str, row: pd.Series, row_idx: int) -> int | None:
			camera_name = _cam_norm(str(camera_name))
			if camera_name not in ("cam1", "cam2"):
				return None

			# 1) Explicit frame_pos from autosave.
			if "frame_pos" in row.index:
				frame_pos_val = pd.to_numeric(row.get("frame_pos"), errors="coerce")
				if np.isfinite(frame_pos_val):
					frame_pos = int(frame_pos_val)
					if frame_pos in export_frames:
						return frame_pos

			# 2) Direct image-name match.
			image_name_candidates = [
				"image_name",
				f"image_{camera_name}",
				"image_cam1" if camera_name == "cam1" else "image_cam2",
			]
			for image_col in image_name_candidates:
				if image_col not in row.index:
					continue
				name = str(row.get(image_col, "")).strip()
				if not name:
					continue
				frame_pos = image_pos_by_cam_name.get(camera_name, {}).get(name)
				if frame_pos is not None and int(frame_pos) in export_frames:
					return int(frame_pos)

			# 3) Frame-id columns, handling either raw frame_pos ids or aligned prediction ids.
			frame_id_candidates = [
				"frame_id",
				f"frame_id_{camera_name}",
				"frame_id_cam1" if camera_name == "cam1" else "frame_id_cam2",
			]
			for frame_col in frame_id_candidates:
				if frame_col not in row.index:
					continue
				frame_id_val = pd.to_numeric(row.get(frame_col), errors="coerce")
				if not np.isfinite(frame_id_val):
					continue
				frame_id = int(frame_id_val)

				if frame_id in export_frames:
					return int(frame_id)

				matched = [
					int(fp)
					for fp in export_frames
					if int(_frame_pos_to_pred_id(camera_name, int(fp))) == int(frame_id)
				]
				if len(matched) == 1:
					return int(matched[0])
				if len(matched) > 1:
					return int(min(matched, key=lambda v: abs(int(v) - int(frame_id))))

			# 4) Legacy wide-file fallback: row order corresponds to export frame order.
			if 0 <= int(row_idx) < len(export_frames):
				return int(export_frames[int(row_idx)])

			return None

		def _load_existing_corrections() -> None:
			auto_path = export_dir / "data" / "corrections_autosave.csv"
			if not auto_path.exists():
				return
			try:
				saved_df = pd.read_csv(auto_path)
			except Exception as exc:
				print(f"Warning: could not read existing corrections at {auto_path}: {exc}")
				return
			if saved_df.empty:
				return

			loaded_count = 0
			required_long = {"camera", "bodypart", "x_corrected", "y_corrected"}
			if required_long.issubset(set(saved_df.columns)):
				for row_idx, (_, row) in enumerate(saved_df.iterrows()):
					camera_name = _cam_norm(str(row.get("camera", "")))
					if camera_name not in ("cam1", "cam2"):
						continue
					bodypart_name = str(row.get("bodypart", "")).strip()
					if not bodypart_name:
						continue
					x_val = float(pd.to_numeric(row.get("x_corrected"), errors="coerce"))
					y_val = float(pd.to_numeric(row.get("y_corrected"), errors="coerce"))
					if not (np.isfinite(x_val) and np.isfinite(y_val)):
						continue
					frame_pos = _resolve_frame_pos_for_saved_row(camera_name, row, row_idx)
					if frame_pos is None:
						continue
					key = (int(frame_pos), str(camera_name), str(bodypart_name))
					correction_cache[key] = (float(x_val), float(y_val))
					correction_pixel_index[key] = (int(round(y_val)), int(round(x_val)))
					loaded_count += 1
			else:
				col_map: dict[tuple[str, str, str], str] = {}
				for col in saved_df.columns:
					m = re.match(r"(?P<bodypart>.+)_cam(?P<cam>[12])_(?P<coord>[XY])$", str(col))
					if m is None:
						continue
					bodypart_name = str(m.group("bodypart"))
					camera_name = f"cam{m.group('cam')}"
					coord = str(m.group("coord")).upper()
					col_map[(bodypart_name, camera_name, coord)] = str(col)

				bodyparts = sorted({bp for (bp, _, _coord) in col_map.keys()})
				for row_idx, (_, row) in enumerate(saved_df.iterrows()):
					for camera_name in ("cam1", "cam2"):
						frame_pos = _resolve_frame_pos_for_saved_row(camera_name, row, row_idx)
						if frame_pos is None:
							continue
						for bodypart_name in bodyparts:
							x_col = col_map.get((bodypart_name, camera_name, "X"))
							y_col = col_map.get((bodypart_name, camera_name, "Y"))
							if x_col is None or y_col is None:
								continue
							x_val = float(pd.to_numeric(row.get(x_col), errors="coerce"))
							y_val = float(pd.to_numeric(row.get(y_col), errors="coerce"))
							if not (np.isfinite(x_val) and np.isfinite(y_val)):
								continue
							key = (int(frame_pos), str(camera_name), str(bodypart_name))
							correction_cache[key] = (float(x_val), float(y_val))
							correction_pixel_index[key] = (int(round(y_val)), int(round(x_val)))
							loaded_count += 1

			if loaded_count > 0:
				print(f"Loaded {loaded_count} saved corrections from {auto_path}")

		def _initialize_corrections_from_predictions() -> None:
			for frame_pos in export_frames:
				for camera_name in ("cam1", "cam2"):
					pts = _pred_points(camera_name, int(frame_pos))
					if pts.empty:
						continue
					for _, row in pts.iterrows():
						bodypart = str(row["bodypart"])
						x_val = float(pd.to_numeric(row["x"], errors="coerce"))
						y_val = float(pd.to_numeric(row["y"], errors="coerce"))
						if not (np.isfinite(x_val) and np.isfinite(y_val)):
							continue
						key = (int(frame_pos), str(camera_name), bodypart)
						correction_cache[key] = (x_val, y_val)
						row_int = int(round(y_val))
						col_int = int(round(x_val))
						correction_pixel_index[key] = (row_int, col_int)
						# likelihood_lookup[key] = float(pd.to_numeric(row["likelihood"], errors="coerce"))

		def _image_name_for(camera_name: str, frame_pos: int) -> str:
			images = state.images_by_cam.get(camera_name, [])
			if 0 <= int(frame_pos) < len(images):
				return str(images[int(frame_pos)].name)
			return ""

		def _autosave_corrections() -> None:
			if not correction_cache or not export_frames:
				return
			auto_path = export_dir / "data" / "corrections_autosave.csv"
			auto_path.parent.mkdir(parents=True, exist_ok=True)
			rows: list[dict[str, Any]] = []
			for (frame_pos, camera_name, bodypart_name), (x_val, y_val) in sorted(correction_cache.items()):
				if int(frame_pos) not in export_frames:
					continue
				key = (int(frame_pos), str(camera_name), str(bodypart_name))
				rows.append(
					{
						"frame_pos": int(frame_pos),
						"image_name": _image_name_for(str(camera_name), int(frame_pos)),
						"frame_id": int(frame_pos),
						"camera": str(camera_name),
						"bodypart": str(bodypart_name),
						"x_corrected": float(x_val),
						"y_corrected": float(y_val),
						# "likelihood": float(likelihood_lookup.get(key, np.nan)),
					}
				)
			if not rows:
				return
			long_df = pd.DataFrame(rows).sort_values(["frame_pos", "camera", "bodypart"], kind="mergesort")

			# Save only XMALab-style wide output with row order matching selected image frames.
			xmalab_rows: list[dict[str, Any]] = []
			bodyparts = sorted(pd.Index(long_df["bodypart"].astype(str)).unique().tolist())
			for frame_pos in export_frames:
				frame_row: dict[str, Any] = {}
				frame_row["frame_pos"] = int(frame_pos)
				frame_row["frame_id_cam1"] = int(_frame_pos_to_pred_id("cam1", int(frame_pos)))
				frame_row["frame_id_cam2"] = int(_frame_pos_to_pred_id("cam2", int(frame_pos)))
				frame_row["image_cam1"] = _image_name_for("cam1", int(frame_pos))
				frame_row["image_cam2"] = _image_name_for("cam2", int(frame_pos))
				for bodypart_name in bodyparts:
					for camera_name, cam_id in (("cam1", "1"), ("cam2", "2")):
						match = long_df[
							(long_df["frame_pos"] == int(frame_pos))
							& (long_df["camera"] == str(camera_name))
							& (long_df["bodypart"] == str(bodypart_name))
						]
						x_col = f"{bodypart_name}_cam{cam_id}_X"
						y_col = f"{bodypart_name}_cam{cam_id}_Y"
						if match.empty:
							frame_row[x_col] = np.nan
							frame_row[y_col] = np.nan
						else:
							frame_row[x_col] = float(match.iloc[0]["x_corrected"])
							frame_row[y_col] = float(match.iloc[0]["y_corrected"])
				xmalab_rows.append(frame_row)
			To_Save_XmaLabStyle = pd.DataFrame(xmalab_rows)
			XmaLab_Accessible_df = To_Save_XmaLabStyle.iloc[:, 5:]
			XmaLab_Accessible_df.to_csv(auto_path, index=False)

		def _resolve_point_xy(camera_name: str, frame_pos: int, bodypart_name: str) -> tuple[float, float] | None:
			pts = _pred_points(camera_name, frame_pos)
			if pts.empty:
				return None
			pts = pts[pts["bodypart"].astype(str) == str(bodypart_name)]
			if pts.empty:
				return None
			row = pts.iloc[0]
			key = (int(frame_pos), str(camera_name), str(bodypart_name))
			if key in correction_cache:
				return correction_cache[key]
			x_val = float(pd.to_numeric(row["x"], errors="coerce"))
			y_val = float(pd.to_numeric(row["y"], errors="coerce"))
			if np.isfinite(x_val) and np.isfinite(y_val):
				return x_val, y_val
			return None

		def _pred_points_with_corrections(camera_name: str, frame_pos: int) -> pd.DataFrame:
			# min_like: float
			pts = _pred_points(camera_name, frame_pos)
			if pts.empty:
				return pts
			# pts = pts[pts["likelihood"].fillna(0.0) >= min_like].copy()
			if pts.empty:
				return pts
			for idx, row in pts.iterrows():
				key = (int(frame_pos), str(camera_name), str(row["bodypart"]))
				if key in correction_cache:
					new_x, new_y = correction_cache[key]
					pts.at[idx, "x"] = float(new_x)
					pts.at[idx, "y"] = float(new_y)
			return pts

		def _nearest_marker_in_axes(event: Any, camera_name: str, frame_pos: int) -> tuple[str, float, float, float] | None:
			# min_like: float
			if event.x is None or event.y is None:
				return None
			# min_like
			pts = _pred_points_with_corrections(camera_name, frame_pos)
			if pts.empty:
				return None
			click_px = np.array([event.x, event.y], dtype=float)
			best_bodypart = None
			best_x = 0.0
			best_y = 0.0
			best_dist = float("inf")
			for _, row in pts.iterrows():
				pt_px = np.array(event.inaxes.transData.transform((float(row["x"]), float(row["y"]))), dtype=float)
				dist = float(np.linalg.norm(click_px - pt_px))
				if dist < best_dist:
					best_dist = dist
					best_bodypart = str(row["bodypart"])
					best_x = float(row["x"])
					best_y = float(row["y"])
			if best_bodypart is None or best_dist > hit_threshold_px:
				return None
			return best_bodypart, best_x, best_y, best_dist

		def _set_active_marker(frame_pos: int, source_camera: str, bodypart_name: str) -> None:
			nonlocal active_marker
			active_marker = {
				"frame_pos": int(frame_pos),
				"source_camera": str(source_camera),
				"bodypart": str(bodypart_name),
			}

		def _apply_edit(frame_pos: int, camera_name: str, bodypart_name: str, x_val: float, y_val: float, autosave: bool = True) -> None:
			correction_cache[(int(frame_pos), str(camera_name), str(bodypart_name))] = (float(x_val), float(y_val))
			if bool(autosave):
				_autosave_corrections()

			# New feature for similar frame corrections!
		def _get_prediction(
			frame_pos: int,
			camera_name: str,
			bodypart_name: str,
			pred_row: pd.Series,
		) -> tuple[float, float]:

			use_previous = (
				radio_prediction_source.value_selected == "Prior frame"
			)

			#
			# PRIOR FRAME
			#
			if use_previous and frame_pos > 0:

				prev_key = (
					frame_pos - 1,
					camera_name,
					bodypart_name,
				)
				# print(prev_key)
				if prev_key in correction_cache:
					return correction_cache[prev_key]

			#
			# CURRENT FRAME (existing behaviour)
			#
			key = (
				frame_pos,
				camera_name,
				bodypart_name,
			)
			# print(key)
			if key in correction_cache:
				return correction_cache[key]

			return (
				float(pd.to_numeric(pred_row["x"], errors="coerce")),
				float(pd.to_numeric(pred_row["y"], errors="coerce")),
			)
		def _selected_cameras():
			mode = str(radio_camera_mode.value_selected).lower()
			if mode == "both":
				return ("cam1", "cam2")
			return (mode,)

		def _snap_to_dark_pixel(frame_pos: int, camera_name: str, bodypart_name: str, x_val: float, y_val: float) -> tuple[float, float]:
			use_previous = (
				radio_prediction_source.value_selected == "Prior frame"
			)
			# PRIOR FRAME
			if use_previous and frame_pos > 0:
				frame_pos = frame_pos - 1
			else:
				frame_pos = frame_pos
			images = state.images_by_cam.get(camera_name, [])
			if not (0 <= int(frame_pos) < len(images)):
				return float(x_val), float(y_val)

			with Image.open(images[int(frame_pos)]) as image_file:
				gray_img = np.asarray(image_file.convert("L"))

			key = (int(frame_pos), str(camera_name), str(bodypart_name))
			occupied_pixels: set[tuple[int, int]] = {
				(rr, cc)
				for (fpos, cam_name, bp_name), (rr, cc) in correction_pixel_index.items()
				if int(fpos) == int(frame_pos) and str(cam_name) == str(camera_name) and str(bp_name) != str(bodypart_name)
			}

			if key in correction_pixel_index:
				old_rr, old_cc = correction_pixel_index[key]
				occupied_pixels.discard((int(old_rr), int(old_cc)))

			snap_mode = str(radio_snap_mode.value_selected).lower().strip()
			if snap_mode == "darkest":
				rr, cc = find_darkest_pixel(
					gray_img,
					row=float(y_val),
					col=float(x_val),
					occupied=occupied_pixels,
					window_size=5,
				)
			else:
				rr, cc = find_dark_blob_centroid(
					gray_img,
					row=float(y_val),
					col=float(x_val),
					occupied=occupied_pixels,
					window_size=9,
					camera=str(camera_name),
					background_sigma=20.0,
					max_sigma=2.0,
					sigma_ratio=1.3,
					threshold=0.03,
					max_match_dist=15.0,
				)
			correction_pixel_index[key] = (int(rr), int(cc))
			return float(cc), float(rr)

		def _apply_snapped_edit(frame_pos: int, camera_name: str, bodypart_name: str, x_val: float, y_val: float) -> None:
			snapped_x, snapped_y = _snap_to_dark_pixel(frame_pos, camera_name, bodypart_name, x_val, y_val)
			_apply_edit(frame_pos, camera_name, bodypart_name, snapped_x, snapped_y)

		def _apply_snap_to_entire_frame(frame_pos: int) -> int:

			# FIXME: Frame Position should be -1 for historical xy coords if previous is selected, and 0 for image...
			snap_mode = str(radio_snap_mode.value_selected).lower().strip()
			update_count = 0
			for camera_name in _selected_cameras():
				images = state.images_by_cam.get(camera_name, [])

				if snap_mode == "blob":
					pts = _pred_points(camera_name, int(frame_pos))
					if pts.empty:
						continue

					row_data: dict[str, float] = {}
					for _, pred_row in pts.iterrows():
						bodypart_name = str(pred_row["bodypart"])
						# key = (int(frame_pos), str(camera_name), bodypart_name)
						
						x_src, y_src = _get_prediction(
							frame_pos,
							camera_name,
							bodypart_name,
							pred_row,
						)
					
						if not (np.isfinite(x_src) and np.isfinite(y_src)):
							continue
						prefix = f"{bodypart_name}_{camera_name}"
						row_data[f"{prefix}_X"] = float(x_src)
						row_data[f"{prefix}_Y"] = float(y_src)


					if not row_data:
						continue
					# Check for previous frame
					use_previous = (radio_prediction_source.value_selected == "Prior frame")
					
					with Image.open(images[int(frame_pos)]) as image_file:
						gray_img = np.asarray(image_file.convert("L"), dtype=np.float64)

					updates = _correct_camera_frame(
						image=gray_img,
						row=pd.Series(row_data),
						camera=str(camera_name),
						background_sigma=20.0,
						max_sigma=2.0,
						sigma_ratio=1.3,
						threshold=0.03,
						overlap=0.5 if str(camera_name) == "cam1" else 0.8,
						min_sigma=1.0 if str(camera_name) == "cam1" else 1.5,
						max_match_dist=15.0,
					)

					for col, value in updates.items():
						if not str(col).endswith(f"_{camera_name}_X"):
							continue
						bodypart_name = str(col)[: -len(f"_{camera_name}_X")]
						y_col = f"{bodypart_name}_{camera_name}_Y"
						if y_col not in updates:
							continue
						x_new = float(pd.to_numeric(value, errors="coerce"))
						y_new = float(pd.to_numeric(updates[y_col], errors="coerce"))
						if not (np.isfinite(x_new) and np.isfinite(y_new)):
							continue
						snapped_x, snapped_y = _snap_to_dark_pixel(int(frame_pos), str(camera_name), bodypart_name, float(x_new), float(y_new))  # Snapping built into blob
						_apply_edit(int(frame_pos), str(camera_name), bodypart_name, snapped_x, snapped_y, autosave=False)
						correction_pixel_index[(int(frame_pos), str(camera_name), bodypart_name)] = (int(round(y_new)), int(round(x_new)))
						update_count += 1
				else:
					pts = _pred_points(camera_name, int(frame_pos))
					if pts.empty:
						continue
					for _, pred_row in pts.iterrows():
						bodypart_name = str(pred_row["bodypart"])
						# key = (int(frame_pos), str(camera_name), str(bodypart_name))
					
						x_src, y_src = _get_prediction(
							frame_pos,
							camera_name,
							bodypart_name,
							pred_row,
						)
						
						if not (np.isfinite(x_src) and np.isfinite(y_src)):
							continue
						snapped_x, snapped_y = _snap_to_dark_pixel(int(frame_pos), str(camera_name), bodypart_name, float(x_src), float(y_src))
						_apply_edit(int(frame_pos), str(camera_name), bodypart_name, float(snapped_x), float(snapped_y), autosave=False)
						update_count += 1
			if update_count > 0:
				_autosave_corrections()
			return int(update_count)

		def _active_marker_xy(camera_name: str, frame_pos: int) -> tuple[float, float] | None:
			if active_marker is None:
				return None
			if int(active_marker["frame_pos"]) != int(frame_pos):
				return None
			return _resolve_point_xy(str(camera_name), int(frame_pos), str(active_marker["bodypart"]))

		_initialize_corrections_from_predictions()
		_load_existing_corrections()
		_autosave_corrections()

		def _apply_review_widget_scaling(_event: Any = None) -> None:
			button_font = _scaled_font(review_fig, base_size=10.0, ref_w=16.0, ref_h=9.0, min_size=8.0, max_size=16.0)
			control_font = _scaled_font(review_fig, base_size=9.5, ref_w=16.0, ref_h=9.0, min_size=8.0, max_size=14.0)
			value_font = _scaled_font(review_fig, base_size=9.0, ref_w=16.0, ref_h=9.0, min_size=7.0, max_size=13.0)
			btn_review_prev.label.set_fontsize(button_font)
			btn_review_next.label.set_fontsize(button_font)
			btn_finish_corrections.label.set_fontsize(control_font)
			# btn_review_current.label.set_fontsize(control_font)
			# btn_zoom_reset.label.set_fontsize(control_font)
			btn_apply_frame.label.set_fontsize(control_font)
			for sld in (slider_review, slider_review_pred_size):
				sld.label.set_fontsize(control_font)
				sld.valtext.set_fontsize(value_font)
			for txt in radio_snap_mode.labels:
				txt.set_fontsize(control_font)
			for txt in check_bulk_snap.labels:
				txt.set_fontsize(control_font)
			review_fig.canvas.draw_idle()

		manager = getattr(review_fig.canvas, "manager", None)
		if manager is not None:
			try:
				manager.set_window_title("Correction Review")
			except Exception:
				pass

		review_limits: dict[str, tuple[tuple[float, float], tuple[float, float]] | None] = {
			"cam1": None,
			"cam2": None,
		}

		def _review_colors(bodyparts: pd.Series, fallback: str) -> list[str] | str:
			if str(radio_color_mode.value_selected) != "by_name":
				return fallback
			return [state.bodypart_color_map.get(str(bp), fallback) for bp in bodyparts]

		def _redraw_review(*_args: Any) -> None:
			nonlocal active_selected_index
			nonlocal last_bulk_apply_signature
			frame_pos = int(slider_review.val)
			auto_bulk_enabled = bool(check_bulk_snap.get_status()[0])
			snap_mode = str(radio_snap_mode.value_selected)
			bulk_signature = (int(frame_pos), str(snap_mode))
			if auto_bulk_enabled:
				if last_bulk_apply_signature != bulk_signature:
					_apply_snap_to_entire_frame(int(frame_pos))
					last_bulk_apply_signature = bulk_signature
			else:
				last_bulk_apply_signature = None
			if frame_pos in frames_local:
				active_selected_index = int(frames_local.index(frame_pos))
			if active_marker is not None and int(active_marker.get("frame_pos", -1)) != int(frame_pos):
				drag_state["active"] = False
			annotate = bool(check.get_status()[1])
			# min_like = float(slider_like.val)
			pred_size = float(slider_review_pred_size.val)
			pred_alpha = float(slider_pred_alpha.val)
			pred_color = _safe_color(textbox_pred_color.text.strip(), "deepskyblue")

			for review_ax, camera_name in zip(review_axes, ("cam1", "cam2")):
				review_ax.clear()
				review_ax.set_axis_off()
				images = state.images_by_cam.get(camera_name, [])
				zoom_ax = zoom_axes[camera_name]
				zoom_ax.clear()
				zoom_ax.set_facecolor("black")
				zoom_ax.set_xticks([])
				zoom_ax.set_yticks([])
				if not (0 <= frame_pos < len(images)):
					review_ax.text(0.5, 0.5, f"No image for {camera_name} frame {frame_pos}", ha="center", va="center", transform=review_ax.transAxes)
					zoom_ax.text(0.5, 0.5, "No image", ha="center", va="center", color="white", transform=zoom_ax.transAxes, fontsize=8)
					continue

				with Image.open(images[frame_pos]) as image_file:
					img = np.asarray(image_file.convert("RGB"))
					review_ax.imshow(img, interpolation="nearest")
					zoom_ax.imshow(img, interpolation="nearest")
				img_h, img_w = int(img.shape[0]), int(img.shape[1])

				pred_pts = _pred_points_with_corrections(camera_name, frame_pos)
				# print(pred_pts)
				# min_like
				if not pred_pts.empty:
					review_ax.scatter(
						pred_pts["x"],
						pred_pts["y"],
						c=_review_colors(pred_pts["bodypart"], pred_color),
						s=pred_size,
						alpha=pred_alpha,
						edgecolors="white",
						linewidths=1.0,
					)
					if annotate:
						for _, row in pred_pts.iterrows():
							label_color = state.bodypart_color_map.get(str(row["bodypart"]), pred_color) if str(radio_color_mode.value_selected) == "by_name" else pred_color
							review_ax.text(row["x"] + 3, row["y"] + 3, str(row["bodypart"]), fontsize=8, color=label_color)

				marker_xy = _active_marker_xy(camera_name, frame_pos)
				if active_marker is not None and marker_xy is not None:
					mx, my = marker_xy
					review_ax.scatter([mx], [my], s=max(220.0, pred_size * 2.5), facecolors="none", edgecolors="lime", linewidths=2.0, zorder=5)
					review_ax.text(mx + 4, my - 4, str(active_marker["bodypart"]), color="lime", fontsize=8)


				if marker_xy is not None:
					mx, my = marker_xy
					half = float(zoom_half_window[camera_name])
					zoom_ax.set_xlim(max(0.0, mx - half), min(float(img_w), mx + half))
					zoom_ax.set_ylim(min(float(img_h), my + half), max(0.0, my - half))
					zoom_ax.scatter([mx], [my], s=36, c="lime", marker="+", linewidths=1.4)
					zoom_ax.set_title(
						f"{camera_name} zoom {int(2 * half)}x{int(2 * half)}",
						fontsize=_scaled_font(review_fig, base_size=8.0, ref_w=16.0, ref_h=9.0, min_size=7.0, max_size=11.0),
					)
				else:
					zoom_ax.set_xlim(0, float(img_w))
					zoom_ax.set_ylim(float(img_h), 0)
					zoom_ax.text(0.5, 0.5, "Select marker", ha="center", va="center", color="white", transform=zoom_ax.transAxes, fontsize=8)

				review_ax.set_title(
					f"{camera_name} | frame {frame_pos}",
					fontsize=_scaled_font(review_fig, base_size=10.0, ref_w=16.0, ref_h=9.0, min_size=8.0, max_size=14.0),
				)
				if review_limits[camera_name] is None:
					review_ax.set_xlim(0, float(img_w))
					review_ax.set_ylim(float(img_h), 0)
				else:
					xlim, ylim = review_limits[camera_name]
					review_ax.set_xlim(xlim)
					review_ax.set_ylim(ylim)
				review_ax.set_aspect("equal")

			review_fig.suptitle(
				f"Correction Review | {method_name.title()} | selected {active_selected_index + 1}/{len(frames_local)} (frame {frames_local[active_selected_index]}) | edits {len(correction_cache)} \n Click marker to select. Drag to move or click elsewhere to set a new coordinate.",
				fontsize=_scaled_font(review_fig, base_size=11.0, ref_w=16.0, ref_h=9.0, min_size=9.0, max_size=16.0),
			)
			review_fig.canvas.draw_idle()

		def _on_review_scroll(event: Any) -> None:
			for camera_name, zoom_ax in zoom_axes.items():
				if event.inaxes == zoom_ax:
					step = getattr(event, "step", 0)
					if step > 0:
						zoom_half_window[camera_name] = max(5.0, zoom_half_window[camera_name] / 1.2)
					elif step < 0:
						zoom_half_window[camera_name] = min(220.0, zoom_half_window[camera_name] * 1.2)
					else:
						button = str(getattr(event, "button", "")).lower()
						if button == "up":
							zoom_half_window[camera_name] = max(5.0, zoom_half_window[camera_name] / 1.2)
						elif button == "down":
							zoom_half_window[camera_name] = min(220.0, zoom_half_window[camera_name] * 1.2)
					_redraw_review()
					return

			for review_ax, camera_name in zip(review_axes, ("cam1", "cam2")):
				if event.inaxes != review_ax:
					continue
				if event.xdata is None or event.ydata is None:
					return

				xlim = review_ax.get_xlim()
				ylim = review_ax.get_ylim()
				xdata = float(event.xdata)
				ydata = float(event.ydata)
				step = getattr(event, "step", 0)
				if step > 0:
					scale = 1.0 / 1.2
				elif step < 0:
					scale = 1.2
				else:
					button = str(getattr(event, "button", "")).lower()
					scale = (1.0 / 1.2) if button == "up" else 1.2

				new_w = (xlim[1] - xlim[0]) * scale
				new_h = (ylim[1] - ylim[0]) * scale
				rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0]) if (xlim[1] - xlim[0]) != 0 else 0.5
				rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0]) if (ylim[1] - ylim[0]) != 0 else 0.5

				review_ax.set_xlim([xdata - new_w * rel_x, xdata + new_w * (1.0 - rel_x)])
				review_ax.set_ylim([ydata - new_h * rel_y, ydata + new_h * (1.0 - rel_y)])
				review_limits[camera_name] = (
					tuple(float(v) for v in review_ax.get_xlim()),
					tuple(float(v) for v in review_ax.get_ylim()),
				)
				review_fig.canvas.draw_idle()
				return

		def _on_review_press(event: Any) -> None:
			if event.inaxes not in review_axes:
				return
			if event.xdata is None or event.ydata is None:
				return
			frame_pos = int(slider_review.val)
			camera_name = "cam1" if event.inaxes == review_axes[0] else "cam2"
			# min_like = float(slider_like.val)
			nearest = _nearest_marker_in_axes(event, camera_name, frame_pos)
			# min_like

			if nearest is not None:
				bodypart_name, x_near, y_near, _ = nearest
				_apply_snapped_edit(frame_pos, camera_name, bodypart_name, float(x_near), float(y_near))
				_set_active_marker(frame_pos, camera_name, bodypart_name)
				drag_state["active"] = True
				drag_state["camera"] = camera_name
				drag_state["bodypart"] = bodypart_name
				drag_state["moved"] = False
				_redraw_review()
				return

			if active_marker is None:
				return
			if int(active_marker.get("frame_pos", -1)) != int(frame_pos):
				return
			bodypart_name = str(active_marker["bodypart"])
			_apply_snapped_edit(frame_pos, camera_name, bodypart_name, float(event.xdata), float(event.ydata))
			_set_active_marker(frame_pos, camera_name, bodypart_name)
			_redraw_review()

		def _on_review_motion(event: Any) -> None:
			if not drag_state.get("active", False):
				return
			if event.inaxes not in review_axes:
				return
			if event.xdata is None or event.ydata is None:
				return
			camera_name = str(drag_state.get("camera", ""))
			bodypart_name = str(drag_state.get("bodypart", ""))
			expected_ax = review_axes[0] if camera_name == "cam1" else review_axes[1]
			if event.inaxes != expected_ax:
				return
			frame_pos = int(slider_review.val)
			_apply_snapped_edit(frame_pos, camera_name, bodypart_name, float(event.xdata), float(event.ydata))
			drag_state["moved"] = True
			_set_active_marker(frame_pos, camera_name, bodypart_name)
			_redraw_review()

		def _on_review_release(_event: Any) -> None:
			if drag_state.get("active", False):
				drag_state["active"] = False
				drag_state["camera"] = None
				drag_state["bodypart"] = None

		def _on_zoom_reset(_event: Any) -> None:
			zoom_half_window["cam1"] = 15.0
			zoom_half_window["cam2"] = 15.0
			_redraw_review()

		def _on_review_prev(_event: Any) -> None:
			nonlocal active_selected_index
			active_selected_index = int((active_selected_index - 1) % len(frames_local))
			slider_review.set_val(float(frames_local[active_selected_index]))

		def _on_review_next(_event: Any) -> None:
			nonlocal active_selected_index
			active_selected_index = int((active_selected_index + 1) % len(frames_local))
			slider_review.set_val(float(frames_local[active_selected_index]))

		def _on_review_current(_event: Any) -> None:
			slider_review.set_val(float(frames_local[active_selected_index]))

		def _on_apply_frame(_event: Any) -> None:
			nonlocal last_bulk_apply_signature
			frame_pos = int(slider_review.val)
			updated = _apply_snap_to_entire_frame(frame_pos)
			last_bulk_apply_signature = (int(frame_pos), str(radio_snap_mode.value_selected))
			print(f"Bulk frame snap: updated {updated} points on frame {frame_pos}.")
			_redraw_review()

		def _on_bulk_toggle(_label: str) -> None:
			nonlocal last_bulk_apply_signature
			if not bool(check_bulk_snap.get_status()[0]):
				last_bulk_apply_signature = None
			_redraw_review()

		def _on_finish_corrections(_event: Any) -> None:
			try:
				_autosave_corrections()
				success = _run_train_and_predict_from_review(export_dir=export_dir, frames_local=frames_local, method_name=method_name)
				if success:
					print("Train-and-predict complete. Main viewer refreshed with updated predictions.")
					plt.close(review_fig)
			except Exception as exc:
				print(f"Train and update failed: {exc}")

		btn_review_prev.on_clicked(_on_review_prev)
		btn_review_next.on_clicked(_on_review_next)
		# btn_review_current.on_clicked(_on_review_current)
		# btn_zoom_reset.on_clicked(_on_zoom_reset)
		btn_apply_frame.on_clicked(_on_apply_frame)
		btn_finish_corrections.on_clicked(_on_finish_corrections)
		slider_review.on_changed(_redraw_review)
		slider_review_pred_size.on_changed(_redraw_review)
		check_bulk_snap.on_clicked(_on_bulk_toggle)
		radio_snap_mode.on_clicked(_redraw_review)
		review_fig.canvas.mpl_connect("scroll_event", _on_review_scroll)
		review_fig.canvas.mpl_connect("button_press_event", _on_review_press)
		review_fig.canvas.mpl_connect("motion_notify_event", _on_review_motion)
		review_fig.canvas.mpl_connect("button_release_event", _on_review_release)
		review_fig.canvas.mpl_connect("resize_event", _apply_review_widget_scaling)
		# Keep widget references attached to the figure so callbacks remain active.
		setattr(
			review_fig,
			"_correction_widgets",
			{
				"btn_prev": btn_review_prev,
				"btn_next": btn_review_next,
				"btn_finish": btn_finish_corrections,
				# "btn_current": btn_review_current,
				# "btn_zoom_reset": btn_zoom_reset,
				"btn_apply_frame": btn_apply_frame,
				"slider": slider_review,
				"pred_size": slider_review_pred_size,
				"bulk_snap": check_bulk_snap,
				"snap_mode": radio_snap_mode,
				"zoom_axes": zoom_axes,
			},
		)
		_apply_review_widget_scaling()
		_redraw_review()
		review_fig.show()

	def _open_active_updates_browser() -> None:
		available_sets = _list_active_update_sets()
		if not available_sets:
			raise RuntimeError("No active_updates subsets found. Create one first with a frame selection and Correction.")

		browser_fig, browser_ax = plt.subplots(figsize=(4.8, 3.8))
		browser_fig.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.90)
		browser_ax.set_axis_off()

		radio_ax = browser_fig.add_axes([0.12, 0.30, 0.76, 0.52])
		open_ax = browser_fig.add_axes([0.12, 0.08, 0.36, 0.12])
		cancel_ax = browser_fig.add_axes([0.54, 0.08, 0.34, 0.12])

		radio_sets = RadioButtons(radio_ax, available_sets, active=0)
		btn_open = Button(open_ax, "Open")
		btn_cancel = Button(cancel_ax, "Close")
		browser_fig.suptitle("Correction Tab: active_updates", fontsize=11)

		def _apply_browser_widget_scaling(_event: Any = None) -> None:
			button_font = _scaled_font(browser_fig, base_size=9.5, ref_w=4.8, ref_h=3.8, min_size=8.0, max_size=14.0)
			radio_font = _scaled_font(browser_fig, base_size=9.0, ref_w=4.8, ref_h=3.8, min_size=7.5, max_size=13.0)
			title_font = _scaled_font(browser_fig, base_size=11.0, ref_w=4.8, ref_h=3.8, min_size=9.0, max_size=16.0)
			btn_open.label.set_fontsize(button_font)
			btn_cancel.label.set_fontsize(button_font)
			for txt in radio_sets.labels:
				txt.set_fontsize(radio_font)
			if browser_fig._suptitle is not None:
				browser_fig._suptitle.set_fontsize(title_font)
			browser_fig.canvas.draw_idle()

		manager = getattr(browser_fig.canvas, "manager", None)
		if manager is not None:
			try:
				manager.set_window_title("Correction Tab")
			except Exception:
				pass

		def _open_selected(_event: Any) -> None:
			set_name = str(radio_sets.value_selected)
			frames = _frames_for_active_update_set(set_name)
			if not frames:
				print(f"Correction tab: no index-matched frames found in active_updates/{set_name}.")
				return
			_open_correction_subset_view(set_name, frames, state.pred_root / "active_updates" / set_name)
			plt.close(browser_fig)

		def _close_browser(_event: Any) -> None:
			plt.close(browser_fig)

		btn_open.on_clicked(_open_selected)
		btn_cancel.on_clicked(_close_browser)
		browser_fig.canvas.mpl_connect("resize_event", _apply_browser_widget_scaling)
		setattr(
			browser_fig,
			"_correction_tab_widgets",
			{
				"radio": radio_sets,
				"open": btn_open,
				"cancel": btn_cancel,
			},
		)
		_apply_browser_widget_scaling()
		browser_fig.show()

	def _set_selection(frames: list[int], meta: dict[int, dict[str, Any]], mode: str, summary_lines: list[str]) -> None:
		nonlocal selected_frames
		nonlocal selected_frame_index
		nonlocal selected_frame_meta
		nonlocal selection_mode
		nonlocal selection_summary_lines
		nonlocal selection_source_camera

		selected_frames = [int(frame) for frame in frames]
		selected_frame_index = 0 if selected_frames else None
		selected_frame_meta = meta
		selection_mode = mode
		selection_summary_lines = summary_lines
		if selected_frames:
			selection_source_camera = str(meta.get(selected_frames[0], {}).get("source_camera", selection_source_camera))

	def _draw_selection_markers() -> None:
		nonlocal selection_marker_artists
		for artist in selection_marker_artists:
			try:
				artist.remove()
			except Exception:
				pass
		selection_marker_artists = []

		if not selection_controls_visible or not selected_frames:
			return

		current_frame = None
		if selected_frame_index is not None and 0 <= selected_frame_index < len(selected_frames):
			current_frame = int(selected_frames[selected_frame_index])

		for frame in selected_frames:
			meta = selected_frame_meta.get(int(frame), {})
			line = ax_frame.axvline(
				float(frame),
				color=str(meta.get("color", "crimson")),
				linewidth=2.2 if current_frame == int(frame) else 1.1,
				alpha=1.0 if current_frame == int(frame) else 0.5,
				zorder=0,
			)
			selection_marker_artists.append(line)

	def _update_selection_info(frame_pos: int) -> None:
		nonlocal selected_frame_index
		ax_selection_info.clear()
		ax_selection_info.set_axis_off()

		if not selection_controls_visible:
			ax_frame.set_title("", fontsize=9, pad=6)
			return

		if not selected_frames:
			ax_frame.set_title("No selected frames", fontsize=9, pad=6)
			ax_selection_info.text(0.0, 0.95, "No frame set", va="top", fontsize=8)
			return

		if frame_pos in selected_frames:
			selected_frame_index = int(selected_frames.index(frame_pos))
			meta = selected_frame_meta.get(frame_pos, {})
			if selection_mode == "displacement":
				ax_frame.set_title(
					f"Displacement set: {selected_frame_index + 1}/{len(selected_frames)} | {meta.get('label', 'Zone')} | weight {float(meta.get('zone_weight', 0.0)):.2f} | avg {float(meta.get('avg_displacement', 0.0)):.2f}",
					fontsize=9,
					pad=6,
				)
			elif selection_mode == "dino":
				ax_frame.set_title(
					f"DINO set: {selected_frame_index + 1}/{len(selected_frames)} | source {meta.get('source_camera', selection_source_camera)} | frame {frame_pos}",
					fontsize=9,
					pad=6,
				)
			else:
				ax_frame.set_title(
					f"Random set: {selected_frame_index + 1}/{len(selected_frames)} | frame {frame_pos}",
					fontsize=9,
					pad=6,
				)
		else:
			ax_frame.set_title(
				f"{selection_mode.title() if selection_mode else 'Selected'} set: {len(selected_frames)} frames | current frame {frame_pos} (not selected)",
				fontsize=9,
				pad=6,
			)

		y = 0.95
		for line in selection_summary_lines:
			ax_selection_info.text(0.0, y, str(line), va="top", fontsize=8)
			y -= 0.22
		if frame_pos in selected_frame_meta:
			meta = selected_frame_meta[frame_pos]
			ax_selection_info.text(0.0, max(y, 0.05), str(meta.get("label", "")), va="bottom", fontsize=8, color=str(meta.get("color", "black")))

	def _goto_selected_index(index: int) -> None:
		nonlocal selected_frame_index
		if not selected_frames:
			return
		selected_frame_index = int(index) % len(selected_frames)
		slider_frame.set_val(float(selected_frames[selected_frame_index]))

	def _set_selection_controls_visible(visible: bool) -> None:
		nonlocal selection_controls_visible
		selection_controls_visible = bool(visible)
		ax_prev.set_visible(selection_controls_visible)
		ax_next.set_visible(selection_controls_visible)
		ax_resample.set_visible(selection_controls_visible)
		redraw()

	def _activate_selection(mode: str, force_resample: bool = False, source_camera: str | None = None) -> None:
		if not force_resample and selection_controls_visible and selection_mode == mode:
			_set_selection_controls_visible(False)
			return

		if force_resample or selection_mode != mode or not selected_frames:
			if mode == "displacement":
				frames, meta, summary_lines = _build_displacement_selection()
			elif mode == "dino":
				frames, meta, summary_lines = _build_dino_selection(source_camera or selection_source_camera)
			else:
				frames, meta, summary_lines = _build_random_selection()
			_set_selection(frames, meta, mode, summary_lines)

		_set_selection_controls_visible(True)
		if selected_frames:
			_goto_selected_index(0)

	def redraw(*_args: Any) -> None:
		shared_count = _shared_frame_limit()
		n = max(1, shared_count) - 1
		slider_frame.valmax = float(n)
		if slider_frame.val > n:
			slider_frame.set_val(n)

		frame_pos = int(slider_frame.val)
		show_pred = bool(check.get_status()[0])
		annotate = bool(check.get_status()[1])
		# min_like = float(slider_like.val)
		pred_size = float(slider_pred_size.val)
		pred_alpha = float(slider_pred_alpha.val)
		pred_color = _safe_color(textbox_pred_color.text.strip(), "deepskyblue")
		color_mode = str(radio_color_mode.value_selected)

		for camera_name, camera_ax in main_axes_map.items():
			imgs = state.images_by_cam.get(camera_name, [])

			if camera_ax.has_data():
				view_limits_by_cam[camera_name]["xlim"] = tuple(float(v) for v in camera_ax.get_xlim())
				view_limits_by_cam[camera_name]["ylim"] = tuple(float(v) for v in camera_ax.get_ylim())

			camera_ax.clear()
			current_points_by_cam[camera_name].clear()
			if selected_text_by_cam.get(camera_name) is not None:
				selected_text_by_cam[camera_name] = None

			if not imgs or not (0 <= frame_pos < len(imgs)):
				camera_ax.text(0.5, 0.5, f"No image for {camera_name}", ha="center", va="center", transform=camera_ax.transAxes)
				camera_ax.set_axis_off()
				continue

			image_path = imgs[frame_pos]
			frame_id_pred = int(_frame_pos_to_pred_id(camera_name, frame_pos))
			name_frame_token = parse_frame_number_from_stem(image_path.stem)

			with Image.open(image_path) as im:
				img = np.asarray(im.convert("RGB"))
			camera_ax.imshow(img)

			if show_pred:
				pred_pts = _pred_points(camera_name, frame_pos)
				if not pred_pts.empty:
					if color_mode == "by_name":
						pred_colors = [state.bodypart_color_map.get(str(bp), pred_color) for bp in pred_pts["bodypart"]]
					else:
						pred_colors = pred_color
					camera_ax.scatter(
						pred_pts["x"],
						pred_pts["y"],
						c=pred_colors,
						s=pred_size,
						alpha=pred_alpha,
						edgecolors="white",
						linewidths=1.0,
						label="Predicted",
					)
					if annotate:
						for _, row in pred_pts.iterrows():
							label_color = state.bodypart_color_map.get(str(row["bodypart"]), pred_color) if color_mode == "by_name" else pred_color
							camera_ax.text(row["x"] + 3, row["y"] + 3, str(row["bodypart"]), fontsize=8, color=label_color)
					for _, row in pred_pts.iterrows():
						current_points_by_cam[camera_name].append(
							{
								"x": float(row["x"]),
								"y": float(row["y"]),
								"bodypart": str(row["bodypart"]),
								"kind": "Predicted",
							}
						)

			camera_ax.set_title(
				f"{camera_name} | frame_pos={frame_pos} | pred_frame_id={frame_id_pred} | file_token={name_frame_token}\n"
				f"pred_csv={state.pred_csv_by_cam[camera_name].name}"
			)
			camera_ax.set_axis_off()

			if view_limits_by_cam[camera_name]["xlim"] is not None and view_limits_by_cam[camera_name]["ylim"] is not None:
				camera_ax.set_xlim(view_limits_by_cam[camera_name]["xlim"])
				camera_ax.set_ylim(view_limits_by_cam[camera_name]["ylim"])

			handles, labels = camera_ax.get_legend_handles_labels()
			if handles:
				camera_ax.legend(loc="upper right")

		_update_selection_info(frame_pos)
		_draw_selection_markers()
		title_left = f"{current_bird} Trial{current_trial_num}" if current_bird is not None and current_trial_num is not None else str(state.pred_root)
		fig.suptitle(f"PostAnalysis Viewer | {title_left} | frame {frame_pos}", fontsize=11)

		fig.canvas.draw_idle()

	def _on_click(event: Any) -> None:
		camera_name = None
		for cam_name, camera_ax in main_axes_map.items():
			if event.inaxes == camera_ax:
				camera_name = cam_name
				break
		if camera_name is None:
			return
		if event.xdata is None or event.ydata is None:
			return
		camera_ax = main_axes_map[camera_name]
		points = current_points_by_cam.get(camera_name, [])
		if not points:
			return

		click_px = np.array([event.x, event.y], dtype=float)
		best = None
		best_dist = float("inf")
		for p in points:
			pt_px = np.array(camera_ax.transData.transform((p["x"], p["y"])), dtype=float)
			d = float(np.linalg.norm(click_px - pt_px))
			if d < best_dist:
				best_dist = d
				best = p

		# Only select when user clicks near a point.
		if best is None or best_dist > 12.0:
			return

		if selected_text_by_cam.get(camera_name) is not None:
			try:
				selected_text_by_cam[camera_name].remove()
			except Exception:
				pass

		selected_text_by_cam[camera_name] = camera_ax.text(
			best["x"] + 6,
			best["y"] + 6,
			f"{best['kind']} | {best['bodypart']}\n(x={best['x']:.1f}, y={best['y']:.1f})",
			fontsize=9,
			color="white",
			bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"},
		)
		fig.canvas.draw_idle()

	def _on_scroll(event: Any) -> None:
		"""Zoom in/out around cursor using mouse wheel."""
		camera_name = None
		for cam_name, camera_ax in main_axes_map.items():
			if event.inaxes == camera_ax:
				camera_name = cam_name
				break
		if camera_name is None:
			return
		if event.xdata is None or event.ydata is None:
			return
		camera_ax = main_axes_map[camera_name]

		xlim = camera_ax.get_xlim()
		ylim = camera_ax.get_ylim()
		xdata = float(event.xdata)
		ydata = float(event.ydata)

		# Scroll up -> zoom in, scroll down -> zoom out.
		step = getattr(event, "step", 0)
		if step > 0:
			scale = 1.0 / 1.2
		elif step < 0:
			scale = 1.2
		else:
			button = str(getattr(event, "button", "")).lower()
			scale = (1.0 / 1.2) if button == "up" else 1.2

		new_w = (xlim[1] - xlim[0]) * scale
		new_h = (ylim[1] - ylim[0]) * scale

		rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0]) if (xlim[1] - xlim[0]) != 0 else 0.5
		rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0]) if (ylim[1] - ylim[0]) != 0 else 0.5

		camera_ax.set_xlim([xdata - new_w * rel_x, xdata + new_w * (1 - rel_x)])
		camera_ax.set_ylim([ydata - new_h * rel_y, ydata + new_h * (1 - rel_y)])
		view_limits_by_cam[camera_name]["xlim"] = tuple(float(v) for v in camera_ax.get_xlim())
		view_limits_by_cam[camera_name]["ylim"] = tuple(float(v) for v in camera_ax.get_ylim())
		fig.canvas.draw_idle()

	def _on_prev(_event: Any) -> None:
		if not selection_controls_visible or not selected_frames:
			return
		curr = selected_frame_index
		if curr is None:
			curr_frame = int(slider_frame.val)
			curr = selected_frames.index(curr_frame) if curr_frame in selected_frames else 0
		_goto_selected_index(curr - 1)

	def _on_next(_event: Any) -> None:
		if not selection_controls_visible or not selected_frames:
			return
		curr = selected_frame_index
		if curr is None:
			curr_frame = int(slider_frame.val)
			curr = selected_frames.index(curr_frame) if curr_frame in selected_frames else -1
		_goto_selected_index(curr + 1)

	def _on_resample(_event: Any) -> None:
		if not selection_controls_visible:
			return
		_activate_selection(selection_mode or "random", force_resample=True, source_camera=selection_source_camera)

	def _on_select_random_frames(_event: Any) -> None:
		_activate_selection("random")

	def _on_select_displacement_frames(_event: Any) -> None:
		_activate_selection("displacement")

	def _on_select_dino_frames(_event: Any) -> None:
		_activate_selection("dino", source_camera=selection_source_camera)

	def _on_switch_bird(_event: Any) -> None:
		nonlocal current_bird
		nonlocal current_trial_num
		nonlocal processing_root
		bird_choice = _choose_bird_dialog()
		if bird_choice is None:
			return
		base_dir = _ensure_processing_root()
		if base_dir is None:
			return
		trial_choice = _choose_trial_dialog(bird_choice)
		if trial_choice is None:
			return
		trial_dir = Path(base_dir) / bird_choice / f"Trial{int(trial_choice)}"
		if not trial_dir.exists():
			print(f"Selected trial path does not exist: {trial_dir}")
			return
		_reload_trial_into_view(trial_dir)
		processing_root = Path(base_dir)
		current_bird = str(bird_choice)
		current_trial_num = int(trial_choice)
		print(f"Loaded: {current_bird} Trial{current_trial_num}")

	def _on_switch_trial(_event: Any) -> None:
		nonlocal current_trial_num
		nonlocal current_bird
		base_dir = _ensure_processing_root()
		if base_dir is None:
			return
		bird_name = current_bird
		if bird_name is None:
			bird_name = _choose_bird_dialog()
			if bird_name is None:
				return
		trial_choice = _choose_trial_dialog(str(bird_name))
		if trial_choice is None:
			return
		trial_dir = Path(base_dir) / str(bird_name) / f"Trial{int(trial_choice)}"
		if not trial_dir.exists():
			print(f"Selected trial path does not exist: {trial_dir}")
			return
		_reload_trial_into_view(trial_dir)
		current_bird = str(bird_name)
		current_trial_num = int(trial_choice)
		print(f"Loaded: {current_bird} Trial{current_trial_num}")

	def _on_open_correction_tab(_event: Any) -> None:
		try:
			_open_active_updates_browser()
		except Exception as exc:
			print(f"Correction tab failed: {exc}")

	def _on_correction_export(_event: Any) -> None:
		try:
			if not selected_frames or not selection_mode:
				raise RuntimeError("Create a frame selection first with Random, Displacement, or DINO before opening Correction.")

			frames = [int(frame) for frame in sorted(set(selected_frames))]
			method_name = _method_folder_name(selection_mode)
			export_dir = _export_active_updates_subset(method_name, frames)
			_open_correction_subset_view(method_name, frames, export_dir)
		except Exception as exc:
			print(f"Correction export failed: {exc}")

	def _prompt_open_existing_active_updates() -> None:
		available_sets = _list_active_update_sets()
		if not available_sets:
			return

		message = (
			f"Found {len(available_sets)} correction set(s) in active_updates under:\n"
			f"{state.pred_root}\n\n"
			"Open Correction Tab now?"
		)

		open_now = False
		try:
			import tkinter as tk
			from tkinter import messagebox

			prompt_root = tk.Tk()
			prompt_root.withdraw()
			prompt_root.attributes("-topmost", True)
			open_now = bool(
				messagebox.askyesno(
					title="Existing Corrections Found",
					message=message,
				)
			)
			prompt_root.destroy()
		except Exception:
			print(message)
			print("Tip: click 'Correction Tab' to open these sets.")
			return

		if open_now:
			try:
				_open_active_updates_browser()
			except Exception as exc:
				print(f"Could not open Correction Tab automatically: {exc}")

	check.on_clicked(redraw)
	btn_correction_tab.on_clicked(_on_open_correction_tab)
	btn_switch_bird.on_clicked(_on_switch_bird)
	btn_switch_trial.on_clicked(_on_switch_trial)
	btn_select_random.on_clicked(_on_select_random_frames)
	btn_select_displacement.on_clicked(_on_select_displacement_frames)
	btn_select_dino.on_clicked(_on_select_dino_frames)
	btn_correction.on_clicked(_on_correction_export)
	btn_prev.on_clicked(_on_prev)
	btn_next.on_clicked(_on_next)
	btn_resample.on_clicked(_on_resample)
	slider_frame.on_changed(redraw)
	# slider_like.on_changed(redraw)
	slider_pred_size.on_changed(redraw)
	slider_pred_alpha.on_changed(redraw)
	textbox_pred_color.on_submit(redraw)
	textbox_nframes.on_submit(redraw)
	radio_color_mode.on_clicked(redraw)
	fig.canvas.mpl_connect("button_press_event", _on_click)
	fig.canvas.mpl_connect("scroll_event", _on_scroll)

	_prompt_open_existing_active_updates()
	_set_selection_controls_visible(False)
	redraw()
	print("Close the plot window to exit.")
	plt.show()


def _build_parser() -> argparse.ArgumentParser:
	"""Build command-line argument parser."""
	parser = argparse.ArgumentParser(
		description=(
			"Launch a popout viewer overlaying predicted points on Cam1/Cam2 image stacks. "
			"Expected folder layout: cam1_img00und, cam2_img00und, and post_processed_data_*/cam1DLC*.csv + cam2DLC*.csv."
		)
	)
	parser.add_argument(
		"prediction_path",
		nargs="?",
		type=str,
		default=None,
		help="Folder path containing prediction CSVs (and optionally image stacks). If omitted, pickers open.",
	)
	parser.add_argument(
		"--cam1-dir",
		type=str,
		default=None,
		help="Optional explicit Cam1 image folder.",
	)
	parser.add_argument(
		"--cam2-dir",
		type=str,
		default=None,
		help="Optional explicit Cam2 image folder.",
	)
	parser.add_argument(
		"--search-truth",
		action="store_true",
		help="If set, auto-search for a likely truth CSV under the prediction folder.",
	)
	parser.add_argument(
		"--truth-csv",
		type=str,
		default=None,
		help="Optional explicit truth CSV path.",
	)
	parser.add_argument(
		"--camera",
		type=str,
		default="cam1",
		choices=["cam1", "cam2"],
		help="Default camera when window opens.",
	)
	parser.add_argument(
		"--frame-pos",
		type=int,
		default=1,
		help="Default frame index position when window opens.",
	)
	parser.add_argument(
		"--browse",
		action="store_true",
		help="Open GUI prompts and ask whether predictions/images share one folder.",
	)
	return parser


def main() -> None:
	"""CLI entry point for terminal use."""
	parser = _build_parser()
	args = parser.parse_args()

	if args.browse or not args.prediction_path:
		prediction_path, cam1_dir, cam2_dir = _choose_inputs_gui()
	else:
		prediction_path = Path(args.prediction_path)
		cam1_dir = Path(args.cam1_dir) if args.cam1_dir else None
		cam2_dir = Path(args.cam2_dir) if args.cam2_dir else None

	if (cam1_dir is None) ^ (cam2_dir is None):
		parser.error("Use both --cam1-dir and --cam2-dir together, or omit both.")

	make_postanalysis_overlay_popout(
		prediction_path=prediction_path,
		search_truth=bool(args.search_truth),
		truth_csv_path=args.truth_csv,
		default_camera=args.camera,
		default_frame_pos=int(args.frame_pos),
		cam1_image_dir=cam1_dir,
		cam2_image_dir=cam2_dir,
	)


if __name__ == "__main__":
	main()
