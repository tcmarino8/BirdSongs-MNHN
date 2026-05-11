from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.widgets import CheckButtons, RadioButtons, Slider, TextBox
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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


def _find_first_dir(root: Path, pattern: str) -> Path:
	"""Return first matching directory under root for a glob pattern."""
	matches = sorted([p for p in root.rglob(pattern) if p.is_dir()])
	if not matches:
		raise FileNotFoundError(f"Could not find folder matching {pattern!r} under {root}")
	return matches[0]


def _resolve_cam_dirs(root: Path) -> dict[str, Path]:
	"""Resolve Cam1_Img00UND* and Cam2_Img00UND* directories."""
	return {
		"cam1": _find_first_dir(root, "*cam1_img00und*"),
		"cam2": _find_first_dir(root, "*cam2_img00und*"),
	}


def _resolve_prediction_csvs(root: Path) -> dict[str, Path]:
	"""Resolve prediction CSVs from post_processed_data_*/cam1DLC*.csv and cam2DLC*.csv."""
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Prediction folder does not exist: {root}")

	post_dirs = sorted([p for p in root.rglob("post_processed_data_*") if p.is_dir()])
	if not post_dirs:
		raise FileNotFoundError(
			f"Could not find a post_processed_data_* folder under: {root}"
		)

	cam_csv: dict[str, Path] = {}
	for d in post_dirs:
		csvs = sorted([p for p in d.glob("*.csv") if p.is_file()])
		for p in csvs:
			low = p.name.lower()
			if "cam1dlc" in low and "cam1" not in cam_csv:
				cam_csv["cam1"] = p
			if "cam2dlc" in low and "cam2" not in cam_csv:
				cam_csv["cam2"] = p

	missing = [c for c in ("cam1", "cam2") if c not in cam_csv]
	if missing:
		raise FileNotFoundError(
			"Missing expected prediction CSV(s) under post_processed_data_*: "
			+ ", ".join(missing)
		)

	return cam_csv


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


def _choose_prediction_path_gui() -> Path:
	"""Show onboarding message, then open folder picker for prediction input."""
	try:
		import tkinter as tk
		from tkinter import filedialog, messagebox
	except Exception as exc:
		raise RuntimeError(
			"GUI file picker is unavailable. Provide prediction_path on the command line instead."
		) from exc

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	messagebox.showinfo(
		title="Prediction Viewer Setup",
		message=(
			"you now have predicted points on a video, select the folder where those predictions lie. "
			"after you do so we will generate a way to view the success!"
		),
	)

	folder_path = filedialog.askdirectory(
		title="Select folder containing cam1_img00und, cam2_img00und, and post_processed_data_*",
	)
	root.destroy()

	if not folder_path:
		raise RuntimeError("No folder selected.")

	return Path(folder_path)


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
) -> None:
	"""Launch an interactive matplotlib popout to inspect predictions on image stacks."""
	root = Path(prediction_path)
	if root.is_file():
		root = root.parent

	pred_csv_by_cam = _resolve_prediction_csvs(root)

	cam_dirs = _resolve_cam_dirs(root)
	images_by_cam = {
		"cam1": collect_images(cam_dirs["cam1"]),
		"cam2": collect_images(cam_dirs["cam2"]),
	}

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
	current_points: list[dict[str, Any]] = []
	selected_text = None
	view_limits: dict[str, tuple[float, float] | None] = {"xlim": None, "ylim": None}

	fig, ax = plt.subplots(figsize=(12.2, 8.4))
	fig.subplots_adjust(left=0.08, right=0.8, bottom=0.35)

	ax_cam = fig.add_axes([0.82, 0.72, 0.16, 0.18])
	ax_checks = fig.add_axes([0.82, 0.49, 0.16, 0.19])
	ax_frame = fig.add_axes([0.08, 0.26, 0.62, 0.03])
	ax_like = fig.add_axes([0.08, 0.21, 0.62, 0.03])
	ax_offset = fig.add_axes([0.08, 0.16, 0.62, 0.03])
	ax_pred_size = fig.add_axes([0.08, 0.11, 0.29, 0.03])
	ax_pred_alpha = fig.add_axes([0.41, 0.11, 0.29, 0.03])
	ax_true_size = fig.add_axes([0.08, 0.06, 0.29, 0.03])
	ax_true_alpha = fig.add_axes([0.41, 0.06, 0.29, 0.03])
	ax_pred_color = fig.add_axes([0.82, 0.40, 0.16, 0.035])
	ax_true_color = fig.add_axes([0.82, 0.345, 0.16, 0.035])
	ax_color_mode = fig.add_axes([0.82, 0.25, 0.16, 0.08])

	radio_cam = RadioButtons(ax_cam, ["cam1", "cam2"], active=0 if cam == "cam1" else 1)
	check = CheckButtons(ax_checks, ["Show pred", "Show true", "Annotate"], [True, False, False])
	if not state.truth_found:
		labels = check.labels
		if len(labels) >= 2:
			labels[1].set_alpha(0.35)

	slider_frame = Slider(ax_frame, "frame_pos", 0, float(max_len), valinit=float(start_pos), valstep=1)
	slider_like = Slider(ax_like, "min_likelihood", 0.0, 1.0, valinit=0.0, valstep=0.01)
	slider_offset = Slider(ax_offset, "truth_offset", -5000, 5000, valinit=0, valstep=1)
	slider_pred_size = Slider(ax_pred_size, "pred_size", 10, 300, valinit=60, valstep=1)
	slider_pred_alpha = Slider(ax_pred_alpha, "pred_alpha", 0.05, 1.0, valinit=0.9, valstep=0.01)
	slider_true_size = Slider(ax_true_size, "true_size", 10, 300, valinit=45, valstep=1)
	slider_true_alpha = Slider(ax_true_alpha, "true_alpha", 0.05, 1.0, valinit=0.9, valstep=0.01)
	textbox_pred_color = TextBox(ax_pred_color, "pred_color", initial="deepskyblue")
	textbox_true_color = TextBox(ax_true_color, "true_color", initial="orange")
	radio_color_mode = RadioButtons(ax_color_mode, ["fixed", "by_name"], active=1)

	for text in radio_color_mode.labels:
		text.set_fontsize(9)

	def _safe_color(value: str, fallback: str) -> str:
		try:
			mcolors.to_rgba(value)
			return value
		except Exception:
			return fallback

	def _pred_points(camera: str, frame_id: int) -> pd.DataFrame:
		d = state.pred_long[
			(state.pred_long["camera"] == camera) & (state.pred_long["frame_id"] == int(frame_id))
		].copy()
		return d[["bodypart", "x_pred", "y_pred", "likelihood"]].rename(columns={"x_pred": "x", "y_pred": "y"})

	def _true_points(camera: str, frame_id: int) -> pd.DataFrame:
		if state.truth_long.empty:
			return pd.DataFrame(columns=["bodypart", "x", "y"])
		d = state.truth_long[
			(state.truth_long["camera"] == camera) & (state.truth_long["frame_id"] == int(frame_id))
		].copy()
		return d[["bodypart", "x_true", "y_true"]].rename(columns={"x_true": "x", "y_true": "y"})

	def redraw(*_args: Any) -> None:
		nonlocal selected_text
		nonlocal view_limits
		camera = _cam_norm(radio_cam.value_selected)
		imgs = state.images_by_cam.get(camera, [])

		n = max(1, len(imgs)) - 1
		slider_frame.valmax = float(n)
		if slider_frame.val > n:
			slider_frame.set_val(n)

		frame_pos = int(slider_frame.val)
		show_pred = bool(check.get_status()[0])
		show_true = bool(check.get_status()[1]) and state.truth_found
		annotate = bool(check.get_status()[2])
		min_like = float(slider_like.val)
		truth_offset = int(slider_offset.val)
		pred_size = float(slider_pred_size.val)
		pred_alpha = float(slider_pred_alpha.val)
		true_size = float(slider_true_size.val)
		true_alpha = float(slider_true_alpha.val)
		pred_color = _safe_color(textbox_pred_color.text.strip(), "deepskyblue")
		true_color = _safe_color(textbox_true_color.text.strip(), "orange")
		color_mode = str(radio_color_mode.value_selected)

		# Preserve current zoom/pan before redraw so interaction persists.
		if ax.has_data():
			view_limits["xlim"] = tuple(float(v) for v in ax.get_xlim())
			view_limits["ylim"] = tuple(float(v) for v in ax.get_ylim())

		ax.clear()
		current_points.clear()
		selected_text = None

		if not imgs:
			ax.text(0.5, 0.5, f"No images for {camera}", ha="center", va="center", transform=ax.transAxes)
			ax.set_axis_off()
			fig.canvas.draw_idle()
			return

		image_path = imgs[frame_pos]
		# Use stack position as the canonical frame id for overlay matching.
		# This ensures the first image is always frame 0 regardless of filename token.
		frame_id_img = int(frame_pos)
		name_frame_token = parse_frame_number_from_stem(image_path.stem)
		truth_frame_id = int(frame_id_img + truth_offset)

		with Image.open(image_path) as im:
			img = np.asarray(im.convert("RGB"))
		ax.imshow(img)

		if show_pred:
			pred_pts = _pred_points(camera, frame_id_img)
			if not pred_pts.empty:
				pred_pts = pred_pts[pred_pts["likelihood"].fillna(0.0) >= min_like].copy()
			if not pred_pts.empty:
				if color_mode == "by_name":
					pred_colors = [state.bodypart_color_map.get(str(bp), pred_color) for bp in pred_pts["bodypart"]]
				else:
					pred_colors = pred_color
				ax.scatter(
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
						ax.text(row["x"] + 3, row["y"] + 3, str(row["bodypart"]), fontsize=8, color=label_color)
				for _, row in pred_pts.iterrows():
					current_points.append(
						{
							"x": float(row["x"]),
							"y": float(row["y"]),
							"bodypart": str(row["bodypart"]),
							"kind": "Predicted",
						}
					)

		if show_true:
			true_pts = _true_points(camera, truth_frame_id)
			if not true_pts.empty:
				if color_mode == "by_name":
					true_colors = [state.bodypart_color_map.get(str(bp), true_color) for bp in true_pts["bodypart"]]
				else:
					true_colors = true_color
				ax.scatter(
					true_pts["x"],
					true_pts["y"],
					c=true_colors,
					s=true_size,
					alpha=true_alpha,
					marker="x",
					linewidths=1.8,
					label="True",
				)
				if annotate:
					for _, row in true_pts.iterrows():
						label_color = state.bodypart_color_map.get(str(row["bodypart"]), true_color) if color_mode == "by_name" else true_color
						ax.text(row["x"] + 3, row["y"] + 3, str(row["bodypart"]), fontsize=8, color=label_color)
				for _, row in true_pts.iterrows():
					current_points.append(
						{
							"x": float(row["x"]),
							"y": float(row["y"]),
							"bodypart": str(row["bodypart"]),
							"kind": "True",
						}
					)

		ax.set_title(
			f"{camera} | frame_pos={frame_pos} | image_frame_id={frame_id_img} | truth_frame_id={truth_frame_id} | file_token={name_frame_token}\n"
			f"pred_csv={state.pred_csv_by_cam[camera].name} | point_customizer={color_mode}"
		)
		ax.set_axis_off()

		# Reapply previous view limits when available.
		if view_limits["xlim"] is not None and view_limits["ylim"] is not None:
			ax.set_xlim(view_limits["xlim"])
			ax.set_ylim(view_limits["ylim"])

		handles, labels = ax.get_legend_handles_labels()
		if handles:
			ax.legend(loc="upper right")

		fig.canvas.draw_idle()

	def _on_click(event: Any) -> None:
		nonlocal selected_text
		if event.inaxes != ax:
			return
		if event.xdata is None or event.ydata is None:
			return
		if not current_points:
			return

		click_px = np.array([event.x, event.y], dtype=float)
		best = None
		best_dist = float("inf")
		for p in current_points:
			pt_px = np.array(ax.transData.transform((p["x"], p["y"])), dtype=float)
			d = float(np.linalg.norm(click_px - pt_px))
			if d < best_dist:
				best_dist = d
				best = p

		# Only select when user clicks near a point.
		if best is None or best_dist > 12.0:
			return

		if selected_text is not None:
			try:
				selected_text.remove()
			except Exception:
				pass

		selected_text = ax.text(
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
		nonlocal view_limits
		if event.inaxes != ax:
			return
		if event.xdata is None or event.ydata is None:
			return

		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
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

		ax.set_xlim([xdata - new_w * rel_x, xdata + new_w * (1 - rel_x)])
		ax.set_ylim([ydata - new_h * rel_y, ydata + new_h * (1 - rel_y)])
		view_limits["xlim"] = tuple(float(v) for v in ax.get_xlim())
		view_limits["ylim"] = tuple(float(v) for v in ax.get_ylim())
		fig.canvas.draw_idle()

	radio_cam.on_clicked(redraw)
	check.on_clicked(redraw)
	slider_frame.on_changed(redraw)
	slider_like.on_changed(redraw)
	slider_offset.on_changed(redraw)
	slider_pred_size.on_changed(redraw)
	slider_pred_alpha.on_changed(redraw)
	slider_true_size.on_changed(redraw)
	slider_true_alpha.on_changed(redraw)
	textbox_pred_color.on_submit(redraw)
	textbox_true_color.on_submit(redraw)
	radio_color_mode.on_clicked(redraw)
	fig.canvas.mpl_connect("button_press_event", _on_click)
	fig.canvas.mpl_connect("scroll_event", _on_scroll)

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
		help="Folder path containing image stacks and post_processed_data_* predictions. If omitted, a picker opens.",
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
		help="Force opening a file/folder picker even if prediction_path is provided.",
	)
	return parser


def main() -> None:
	"""CLI entry point for terminal use."""
	parser = _build_parser()
	args = parser.parse_args()

	if args.browse or not args.prediction_path:
		prediction_path = _choose_prediction_path_gui()
	else:
		prediction_path = Path(args.prediction_path)

	make_postanalysis_overlay_popout(
		prediction_path=prediction_path,
		search_truth=bool(args.search_truth),
		truth_csv_path=args.truth_csv,
		default_camera=args.camera,
		default_frame_pos=int(args.frame_pos),
	)


if __name__ == "__main__":
	main()
