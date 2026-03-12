"""Rigid body correction script (CSV in, CSV out).

Inputs (4):
1) Key_Rigid_body: path to reference rigid-body CSV (obj1)
2) ToCorrect_Rigid_Body: path to rigid-body CSV to correct (obj2)
3) stable_frame: integer row index used as stable reference in both files
4) out_file_name: filename/path for correction error results CSV

Also writes:
	Corrected_<ToCorrect_Rigid_Body_stem>.csv
in the same directory as ToCorrect_Rigid_Body.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _get_matrix_headers(fieldnames: List[str], csv_path: Path) -> List[str]:
	if len(fieldnames) < 16:
		raise ValueError(
			f"CSV '{csv_path}' has {len(fieldnames)} columns; at least 16 are required."
		)
	# Use the first 16 columns as the 4x4 matrix in the original CSV order.
	return fieldnames[:16]


def _rows_to_matrices(rows: List[dict], matrix_headers: List[str], csv_path: Path) -> np.ndarray:
	matrices = []
	for i, row in enumerate(rows):
		try:
			values = [float(row[h]) for h in matrix_headers]
		except (TypeError, ValueError, KeyError) as exc:
			raise ValueError(
				f"Could not parse matrix values in '{csv_path}', row index {i}."
			) from exc
		matrices.append(np.array(values, dtype=float).reshape(4, 4))
	return np.array(matrices)


def _load_csv(csv_path: Path) -> tuple[List[dict], List[str], List[str], np.ndarray]:
	with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError(f"CSV '{csv_path}' has no header row.")
		fieldnames = list(reader.fieldnames)
		matrix_headers = _get_matrix_headers(fieldnames, csv_path)
		rows = list(reader)

	if not rows:
		raise ValueError(f"CSV '{csv_path}' has no data rows.")

	matrices = _rows_to_matrices(rows, matrix_headers, csv_path)
	return rows, fieldnames, matrix_headers, matrices


def _matrix_to_row_values(m: np.ndarray) -> List[str]:
	flat = m.reshape(-1)
	formatted: List[str] = []
	for v in flat:
		if np.isnan(v):
			formatted.append("NaN")
		else:
			formatted.append(f"{v:.10f}")
	return formatted


def _same_path(a: Path, b: Path) -> bool:
	return a.resolve(strict=False) == b.resolve(strict=False)


def _append_mode_suffix(path: Path, stable_frame_mode: str) -> Path:
	return path.with_name(f"{path.stem}_{stable_frame_mode}{path.suffix}")


def _is_finite_matrix(m: np.ndarray) -> bool:
	return bool(np.all(np.isfinite(m)))


def _try_inverse(m: np.ndarray) -> np.ndarray | None:
	try:
		inv = np.linalg.inv(m)
	except np.linalg.LinAlgError:
		return None
	if not _is_finite_matrix(inv):
		return None
	return inv


def _build_relative_maps(
	obj1_mats: np.ndarray,
	obj2_mats: np.ndarray,
) -> Tuple[np.ndarray, List[int]]:
	n_frames = len(obj1_mats)
	rel_maps = np.full((n_frames, 4, 4), np.nan, dtype=float)
	valid_indices: List[int] = []

	for i in range(n_frames):
		m1 = obj1_mats[i]
		m2 = obj2_mats[i]
		if not _is_finite_matrix(m1) or not _is_finite_matrix(m2):
			continue
		inv_m2 = _try_inverse(m2)
		if inv_m2 is None:
			continue
		rel = m1 @ inv_m2
		if not _is_finite_matrix(rel):
			continue
		rel_maps[i] = rel
		valid_indices.append(i)

	return rel_maps, valid_indices


def _select_stable_frame(
	stable_frame: int,
	stable_frame_mode: str,
	obj1_mats: np.ndarray,
	obj2_mats: np.ndarray,
	tie_tol: float,
) -> Tuple[int, List[int]]:
	rel_maps, valid_indices = _build_relative_maps(obj1_mats, obj2_mats)
	if not valid_indices:
		raise ValueError("No valid frames found (NaN/Inf or non-invertible transforms in all frames).")

	if stable_frame_mode == "explicit":
		if stable_frame < 0 or stable_frame >= len(obj1_mats):
			raise IndexError(
				f"stable_frame={stable_frame} is out of range 0..{len(obj1_mats)-1}."
			)
		if stable_frame not in set(valid_indices):
			raise ValueError(
				f"stable_frame={stable_frame} is invalid (NaN/Inf or non-invertible transform)."
			)
		return stable_frame, valid_indices

	if stable_frame_mode == "random":
		rng = np.random.default_rng()
		return int(rng.choice(valid_indices)), valid_indices

	valid_rel = rel_maps[valid_indices].reshape(len(valid_indices), 16)
	scores = np.empty(len(valid_indices), dtype=float)
	for i in range(len(valid_indices)):
		d = np.linalg.norm(valid_rel - valid_rel[i], axis=1)
		d[i] = np.nan
		if stable_frame_mode == "mean":
			scores[i] = float(np.nanmean(d))
		elif stable_frame_mode == "median":
			scores[i] = float(np.nanmedian(d))
		else:
			raise ValueError(f"Unknown stable_frame_mode: {stable_frame_mode}")

	best = float(np.nanmin(scores))
	tie_positions = np.where(np.abs(scores - best) <= tie_tol)[0]
	rng = np.random.default_rng()  # true random tie-break
	chosen_pos = int(rng.choice(tie_positions))
	return valid_indices[chosen_pos], valid_indices


def _write_corrected_csv(
	input_rows_obj2: List[dict],
	fieldnames_obj2: List[str],
	obj2_matrix_headers: List[str],
	corrected_matrices: np.ndarray,
	output_path: Path,
) -> None:
	out_rows = []
	for i, original_row in enumerate(input_rows_obj2):
		row_copy = dict(original_row)
		corrected_vals = _matrix_to_row_values(corrected_matrices[i])
		for h, v in zip(obj2_matrix_headers, corrected_vals):
			row_copy[h] = v
		out_rows.append(row_copy)

	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames_obj2)
		writer.writeheader()
		writer.writerows(out_rows)


def _write_error_results_csv(
	error_matrices: np.ndarray,
	obj2_matrix_headers: List[str],
	output_path: Path,
) -> None:
	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(obj2_matrix_headers)
		for i, err in enumerate(error_matrices):
			writer.writerow(_matrix_to_row_values(err))


def correct_rigid_body(
	key_rigid_body: Path,
	to_correct_rigid_body: Path,
	stable_frame: int,
	stable_frame_mode: str,
	tie_tol: float,
	out_file_name: Path,
) -> tuple[Path, Path, int, int]:
	# Normalize input paths early so downstream joins/comparisons are stable.
	key_rigid_body = key_rigid_body.resolve(strict=False)
	to_correct_rigid_body = to_correct_rigid_body.resolve(strict=False)

	if not key_rigid_body.exists() or not key_rigid_body.is_file():
		raise FileNotFoundError(f"Key_Rigid_body not found: {key_rigid_body}")
	if not to_correct_rigid_body.exists() or not to_correct_rigid_body.is_file():
		raise FileNotFoundError(f"ToCorrect_Rigid_Body not found: {to_correct_rigid_body}")

	obj1_rows, _, obj1_matrix_headers, obj1_mats = _load_csv(key_rigid_body)
	obj2_rows, obj2_fieldnames, obj2_matrix_headers, obj2_mats = _load_csv(to_correct_rigid_body)

	if len(obj1_matrix_headers) != 16 or len(obj2_matrix_headers) != 16:
		raise ValueError("Both input CSVs must provide 16 matrix columns for a 4x4 transform.")

	if len(obj1_rows) != len(obj2_rows):
		raise ValueError(
			"Input CSV files must have the same number of rows. "
			f"Got {len(obj1_rows)} and {len(obj2_rows)}."
		)

	selected_stable_frame, valid_indices = _select_stable_frame(
		stable_frame=stable_frame,
		stable_frame_mode=stable_frame_mode,
		obj1_mats=obj1_mats,
		obj2_mats=obj2_mats,
		tie_tol=tie_tol,
	)
	valid_set = set(valid_indices)
	skipped_count = len(obj1_rows) - len(valid_indices)

	# Difference matrix from stable frame:
	# difference = obj1(stable) * inverse(obj2(stable))
	difference = obj1_mats[selected_stable_frame] @ np.linalg.inv(obj2_mats[selected_stable_frame])

	# Apply correction to every frame: corrected_obj2 = difference * obj2
	corrected_obj2 = np.full((len(obj2_mats), 4, 4), np.nan, dtype=float)

	# Error matrix like the Maya script:
	# error = obj1 * inverse(corrected_obj2)
	error_mats = np.full((len(obj1_mats), 4, 4), np.nan, dtype=float)
	for i in range(len(obj1_mats)):
		if i not in valid_set:
			continue
		new_matrix = difference @ obj2_mats[i]
		if not _is_finite_matrix(new_matrix):
			continue
		inv_new = _try_inverse(new_matrix)
		if inv_new is None:
			continue
		err = obj1_mats[i] @ inv_new
		if not _is_finite_matrix(err):
			continue
		corrected_obj2[i] = new_matrix
		error_mats[i] = err

	corrected_base = to_correct_rigid_body.with_name(
		f"Corrected_{to_correct_rigid_body.stem}.csv"
	)
	corrected_path = _append_mode_suffix(corrected_base, stable_frame_mode)

	error_path = out_file_name
	if error_path.is_absolute():
		pass
	elif error_path.parent == Path("."):
		# Bare filename: place next to ToCorrect_Rigid_Body.
		error_path = to_correct_rigid_body.parent / error_path
	else:
		# Relative path with directories: respect caller-provided location.
		error_path = error_path.resolve(strict=False)
	error_path = _append_mode_suffix(error_path, stable_frame_mode)

	if _same_path(corrected_path, key_rigid_body) or _same_path(corrected_path, to_correct_rigid_body):
		raise ValueError("Safety check failed: corrected output path matches an input file path.")
	if _same_path(error_path, key_rigid_body) or _same_path(error_path, to_correct_rigid_body):
		raise ValueError("Safety check failed: out_file_name matches an input file path.")

	if corrected_path.exists():
		raise FileExistsError(
			f"Refusing to overwrite existing file: {corrected_path}. "
			"Rename or move the old file first."
		)
	if error_path.exists():
		raise FileExistsError(
			f"Refusing to overwrite existing file: {error_path}. "
			"Choose a new out_file_name or remove the old file first."
		)

	_write_corrected_csv(
		obj2_rows,
		obj2_fieldnames,
		obj2_matrix_headers,
		corrected_obj2,
		corrected_path,
	)
	_write_error_results_csv(error_mats, obj2_matrix_headers, error_path)

	return corrected_path, error_path, selected_stable_frame, skipped_count


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Correct a rigid-body transform CSV from a reference CSV.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=(
			"Example:\n"
			"  python Ramphica_adjustment.py "
			"RigidBody001_Cranium_transformationFiltered_20Hz.csv "
			"RigidBody002_Maxilla_transformationFiltered_20Hz.csv "
			"100 "
			"--stable-frame-mode mean "
			"CorrectedTransform_Results_Full.csv"
		),
	)
	parser.add_argument("Key_Rigid_body", type=Path, help="Path to obj1 CSV (reference).")
	parser.add_argument(
		"ToCorrect_Rigid_Body",
		type=Path,
		help="Path to obj2 CSV (will be corrected).",
	)
	parser.add_argument(
		"stable_frame",
		type=int,
		help="Stable reference row index (0-based); used when --stable-frame-mode explicit.",
	)
	parser.add_argument(
		"--stable-frame-mode",
		choices=["explicit", "mean", "median", "random"],
		default="mean",
		help="Stable frame selection mode. Default: mean (relative-map).",
	)
	parser.add_argument(
		"--tie-tol",
		type=float,
		default=1e-12,
		help="Tolerance for considering mean/median scores tied (ties are broken randomly).",
	)
	parser.add_argument(
		"out_file_name",
		type=Path,
		help="Output filename/path for correction error results CSV.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	corrected_path, error_path, selected_stable_frame, skipped_count = correct_rigid_body(
		key_rigid_body=args.Key_Rigid_body,
		to_correct_rigid_body=args.ToCorrect_Rigid_Body,
		stable_frame=args.stable_frame,
		stable_frame_mode=args.stable_frame_mode,
		tie_tol=args.tie_tol,
		out_file_name=args.out_file_name,
	)
	print(f"Stable frame mode: {args.stable_frame_mode}")
	print(f"Selected stable frame: {selected_stable_frame}")
	print(f"Skipped invalid frames: {skipped_count}")
	print(f"Corrected rigid-body CSV written to: {corrected_path}")
	print(f"Correction error results written to: {error_path}")


if __name__ == "__main__":
	main()
