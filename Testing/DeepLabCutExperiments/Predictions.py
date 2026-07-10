from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_COORD_RE = re.compile(r"^(?P<bodypart>.+)_cam(?P<cam>[12])_(?P<coord>[XY])$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_testing_dir() -> Path:
    return _repo_root() / "Code-Testing"


def _ensure_code_testing_on_path() -> Path:
    code_dir = _code_testing_dir()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    return code_dir


def _load_data_converter_module() -> Any:
    module_path = _code_testing_dir() / "data-converters.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find converter module: {module_path}")

    spec = importlib.util.spec_from_file_location("dc", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter module spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    idx = int(match.group(1)) if match else -1
    return idx, path.name.lower()


def _list_images_sorted(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS],
        key=_image_sort_key,
    )


def _resolve_frame_window(
    n_cam1: int,
    n_cam2: int,
    frame_range: tuple[int, int] | list[int] | None,
) -> tuple[int, int, int]:
    n_effective = int(min(n_cam1, n_cam2))
    if n_effective <= 0:
        raise ValueError("No images available in Cam1/Cam2 folders")

    if frame_range is None:
        start = 0
        end = n_effective - 1
    else:
        start, end = map(int, frame_range)
        if end < start:
            start, end = end, start

        start = max(0, start)
        end = min(n_effective - 1, end)

    if end < start:
        raise ValueError(f"Invalid frame range after clamping: start={start}, end={end}")

    return start, end, n_effective


def _ensure_avi_from_jpg_stack(
    jpg_dir: Path,
    avi_path: Path,
    *,
    fps: int,
    start_frame: int,
    end_frame: int,
    force_rebuild: bool,
) -> dict[str, Any]:
    dc = _load_data_converter_module()

    if force_rebuild and avi_path.exists():
        avi_path.unlink()

    if not avi_path.exists():
        return dc.jpg_stack_to_avi(
            input_folder=jpg_dir,
            output_path=avi_path,
            fps=int(fps),
            start_frame=int(start_frame),
            end_frame=int(end_frame),
        )

    return {
        "output_path": str(avi_path),
        "frames_written": int(end_frame - start_frame + 1),
        "fps": int(fps),
    }


def _latest_csv(trial_dir: Path, pattern: str) -> Path:
    candidates = sorted(trial_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No files found for pattern {pattern} in {trial_dir}")
    return candidates[-1]


def _collect_xy_pairs(columns: list[str]) -> list[tuple[str, str, str, str]]:
    info: dict[tuple[str, str], dict[str, str]] = {}
    for c in columns:
        match = _COORD_RE.match(str(c))
        if match is None:
            continue
        bodypart = str(match.group("bodypart"))
        camera = f"cam{match.group('cam')}"
        coord = str(match.group("coord")).upper()
        info.setdefault((bodypart, camera), {})[coord] = str(c)

    pairs: list[tuple[str, str, str, str]] = []
    for (bodypart, camera), cols in sorted(info.items()):
        if "X" in cols and "Y" in cols:
            pairs.append((bodypart, camera, cols["X"], cols["Y"]))
    return pairs


def score_prediction_vs_truth(
    pred_csv: Path,
    truth_csv: Path,
    threshold_px: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pred_df = pd.read_csv(pred_csv)
    truth_df = pd.read_csv(truth_csv)

    n_frames = int(min(len(pred_df), len(truth_df)))
    pred_df = pred_df.iloc[:n_frames].copy()
    truth_df = truth_df.iloc[:n_frames].copy()

    common_cols = [c for c in pred_df.columns if c in truth_df.columns and _COORD_RE.match(str(c))]
    pairs = _collect_xy_pairs(common_cols)
    if len(pairs) == 0:
        raise ValueError(f"No shared *_cam[12]_[XY] coordinate columns between {pred_csv} and {truth_csv}")

    rows = []
    for frame_idx in range(n_frames):
        for bodypart, camera, xcol, ycol in pairs:
            px = pd.to_numeric(pred_df.at[frame_idx, xcol], errors="coerce")
            py = pd.to_numeric(pred_df.at[frame_idx, ycol], errors="coerce")
            tx = pd.to_numeric(truth_df.at[frame_idx, xcol], errors="coerce")
            ty = pd.to_numeric(truth_df.at[frame_idx, ycol], errors="coerce")

            if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(tx) and np.isfinite(ty)):
                continue

            dist = float(np.hypot(float(px) - float(tx), float(py) - float(ty)))
            rows.append(
                {
                    "frame_id": int(frame_idx),
                    "camera": camera,
                    "bodypart": bodypart,
                    "distance_px": dist,
                    "within_threshold": bool(dist <= float(threshold_px)),
                }
            )

    error_df = pd.DataFrame(rows)
    if error_df.empty:
        raise ValueError(f"No valid comparable points found for {pred_csv.name}")

    rmse_px = float(np.sqrt(np.mean(np.square(error_df["distance_px"].to_numpy(dtype=float)))))
    pct_points_within = float(100.0 * error_df["within_threshold"].mean())

    frame_all_within = error_df.groupby("frame_id", as_index=True)["within_threshold"].all()
    pct_frames_all_points_within = float(100.0 * frame_all_within.mean())

    counts_within_by_frame = (
        error_df.groupby("frame_id", as_index=True)["within_threshold"].sum().reindex(range(n_frames), fill_value=0)
    )
    mean_n_within_per_frame = float(counts_within_by_frame.mean())
    std_n_within_per_frame = float(counts_within_by_frame.std(ddof=0))

    summary = {
        "rmse_px": rmse_px,
        "threshold_px": float(threshold_px),
        "percent_points_within_threshold": pct_points_within,
        "percent_frames_all_points_within_threshold": pct_frames_all_points_within,
        "mean_n_predictions_within_threshold_per_frame": mean_n_within_per_frame,
        "std_n_predictions_within_threshold_per_frame": std_n_within_per_frame,
        "n_points_compared": int(len(error_df)),
        "n_frames_compared": int(n_frames),
    }

    return error_df, summary


def predict_trial_from_jpg_stacks_safe(
    trial_dir: str | Path,
    config_path: str | Path,
    *,
    fps: int = 500,
    batchsize: int = 16,
    save_as_csv: bool = True,
    xma_base_name: str = "NoUpdateModel",
    frame_range: tuple[int, int] | list[int] | None = None,
    force_rebuild_avi: bool = True,
    cleanup_dlc_csv: bool = True,
) -> dict[str, Any]:
    _ensure_code_testing_on_path()

    import deeplabcut as dlc  # noqa: WPS433
    import xrommtools_copy as xt  # noqa: WPS433

    trial_dir = Path(trial_dir)
    config_path = Path(config_path)

    cam1_dir = trial_dir / "Cam1"
    cam2_dir = trial_dir / "Cam2"
    cam1_avi = trial_dir / "Cam1.avi"
    cam2_avi = trial_dir / "Cam2.avi"

    if not cam1_dir.exists() or not cam2_dir.exists():
        raise FileNotFoundError(f"Expected Cam1/Cam2 folders in {trial_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cam1_imgs = _list_images_sorted(cam1_dir)
    cam2_imgs = _list_images_sorted(cam2_dir)
    start_frame, end_frame, n_effective = _resolve_frame_window(len(cam1_imgs), len(cam2_imgs), frame_range)

    _ensure_avi_from_jpg_stack(
        cam1_dir,
        cam1_avi,
        fps=int(fps),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        force_rebuild=bool(force_rebuild_avi),
    )
    _ensure_avi_from_jpg_stack(
        cam2_dir,
        cam2_avi,
        fps=int(fps),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        force_rebuild=bool(force_rebuild_avi),
    )
    
    if cleanup_dlc_csv:
        # Clear prior DLC analysis artifacts so analyze_videos does not short-circuit
        # with "already analyzed" when rerunning the same trial repeatedly.
        for pat in ("Cam1DLC*", "Cam2DLC*"):
            for old_artifact in trial_dir.glob(pat):
                if old_artifact.is_file():
                    old_artifact.unlink(missing_ok=True)
   
    
    dlc.analyze_videos(
        str(config_path),
        [str(cam1_avi), str(cam2_avi)],
        save_as_csv=bool(save_as_csv),
        destfolder=str(trial_dir),
        batchsize=int(batchsize),
    )

    cam1_dlc_csv = _latest_csv(trial_dir, "Cam1DLC*.csv")
    cam2_dlc_csv = _latest_csv(trial_dir, "Cam2DLC*.csv")

    pred_df = xt.dlc_to_xma(str(cam1_dlc_csv), str(cam2_dlc_csv), str(xma_base_name), str(trial_dir))
    pred_df = pred_df.iloc[: int(n_effective)].copy()

    if len(pred_df) < int(n_effective):
        pad_rows = int(n_effective) - len(pred_df)
        pad_df = pd.DataFrame(np.nan, index=range(pad_rows), columns=pred_df.columns)
        pred_df = pd.concat([pred_df, pad_df], ignore_index=True)

    pred_csv = trial_dir / f"{xma_base_name}-Predicted2DPoints.csv"
    pred_df.to_csv(pred_csv, na_rep="NaN", index=False)

    return {
        "trial_dir": trial_dir,
        "cam1_avi": cam1_avi,
        "cam2_avi": cam2_avi,
        "cam1_dlc_csv": cam1_dlc_csv,
        "cam2_dlc_csv": cam2_dlc_csv,
        "pred_csv": pred_csv,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "n_effective_frames": int(n_effective),
    }


def _trial_key_from_dir_name(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<birdtok>[A-Za-z]+)_T(?P<trial>\d+)", str(name).strip())
    if match is None:
        raise ValueError(f"Could not parse trial dir name: {name}")

    birdtok = str(match.group("birdtok"))
    trial_num = int(match.group("trial"))
    bird_map = {"DB": "DavidBowie", "Tulio": "Tulio"}
    return bird_map.get(birdtok, birdtok), trial_num


def _load_eval_trials(data_root: Path, bird: str) -> list[tuple[int, Path]]:
    eval_trials: list[tuple[int, Path]] = []
    for trial_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        try:
            bird_name, trial_num = _trial_key_from_dir_name(trial_dir.name)
        except Exception:
            continue

        if bird_name != bird:
            continue

        test_dir = trial_dir / "test"
        if (test_dir / "Cam1").exists() and (test_dir / "Cam2").exists() and (test_dir / "LabeledBodyPartsCoordinates.csv").exists():
            eval_trials.append((int(trial_num), test_dir))

    return eval_trials


def _load_models_from_build_manifest(build_manifest_path: Path, bird: str) -> pd.DataFrame:
    build_manifest_df = pd.read_csv(build_manifest_path)

    required = {"bird", "trial_num", "method", "nframes", "config_path"}
    missing = required - set(build_manifest_df.columns)
    if missing:
        raise ValueError(f"training_set_build_manifest missing required columns: {sorted(missing)}")

    model_df = build_manifest_df.copy()
    model_df["bird"] = model_df["bird"].astype(str)
    model_df["method"] = model_df["method"].astype(str).str.lower().str.strip()
    model_df["trial_num"] = pd.to_numeric(model_df["trial_num"], errors="coerce").astype("Int64")
    model_df["nframes"] = pd.to_numeric(model_df["nframes"], errors="coerce").astype("Int64")

    model_df = model_df[
        (model_df["bird"] == bird)
        & model_df["config_path"].astype(str).str.len().gt(0)
        & model_df["trial_num"].notna()
        & model_df["nframes"].notna()
    ].copy()

    if model_df.empty:
        return model_df

    model_df["trial_num"] = model_df["trial_num"].astype(int)
    model_df["nframes"] = model_df["nframes"].astype(int)
    model_df["config_path"] = model_df["config_path"].astype(str)

    return model_df.drop_duplicates(subset=["config_path"]).reset_index(drop=True)


def evaluate_within_across_for_bird(
    *,
    bird: str,
    data_root: str | Path,
    build_manifest_path: str | Path,
    threshold_px: float = 5.0,
    fps: int = 500,
    batchsize: int = 16,
    delete_intermediate_predictions: bool = False,
    force_rebuild_avi: bool = True,
) -> dict[str, Any]:
    data_root = Path(data_root)
    build_manifest_path = Path(build_manifest_path)

    model_df = _load_models_from_build_manifest(build_manifest_path, bird)
    if model_df.empty:
        raise RuntimeError(f"No models found for bird={bird} in {build_manifest_path}")

    eval_trials = _load_eval_trials(data_root, bird)
    if len(eval_trials) == 0:
        raise RuntimeError(f"No eval test folders found for bird={bird} under {data_root}")

    score_rows: list[dict[str, Any]] = []
    error_rows: list[pd.DataFrame] = []

    run_idx = 0
    for _, model_row in model_df.sort_values(["trial_num", "method", "nframes"]).iterrows():
        train_trial = int(model_row["trial_num"])
        method = str(model_row["method"])
        nframes = int(model_row["nframes"])
        config_path = Path(str(model_row["config_path"]))

        if not config_path.exists():
            score_rows.append(
                {
                    "bird": bird,
                    "train_trial": int(train_trial),
                    "eval_trial": np.nan,
                    "pair": "",
                    "pair_type": "",
                    "method": method,
                    "nframes": int(nframes),
                    "model": f"{method}_n{nframes}_TrainT{train_trial}",
                    "rmse_px": np.nan,
                    "threshold_px": float(threshold_px),
                    "percent_points_within_threshold": np.nan,
                    "percent_frames_all_points_within_threshold": np.nan,
                    "mean_n_predictions_within_threshold_per_frame": np.nan,
                    "std_n_predictions_within_threshold_per_frame": np.nan,
                    "n_points_compared": 0,
                    "n_frames_compared": 0,
                    "config_path": str(config_path),
                    "pred_csv": "",
                    "status": "error",
                    "error": f"Config not found: {config_path}",
                }
            )
            continue

        for eval_trial, test_dir in eval_trials:
            run_idx += 1
            pair_type = "within" if train_trial == eval_trial else "across"
            pair_label = f"TrainT{train_trial}_EvalT{eval_trial}"
            model_label = f"{method}_n{nframes}_TrainT{train_trial}"
            xma_name = f"{bird}_{model_label}_EvalT{eval_trial}"
            print(test_dir)
            try:
                
                pred_result = predict_trial_from_jpg_stacks_safe(
                    trial_dir=test_dir,
                    config_path=config_path,
                    fps=int(fps),
                    batchsize=int(batchsize),
                    save_as_csv=True,
                    xma_base_name=xma_name,
                    frame_range=None,
                    force_rebuild_avi=bool(force_rebuild_avi),
                    cleanup_dlc_csv=True,
                )

                pred_csv = Path(pred_result["pred_csv"])
                truth_csv = test_dir / "LabeledBodyPartsCoordinates.csv"
                err_df, summary = score_prediction_vs_truth(pred_csv=pred_csv, truth_csv=truth_csv, threshold_px=float(threshold_px))

                err_df = err_df.copy()
                err_df["bird"] = bird
                err_df["train_trial"] = int(train_trial)
                err_df["eval_trial"] = int(eval_trial)
                err_df["pair"] = pair_label
                err_df["pair_type"] = pair_type
                err_df["method"] = method
                err_df["nframes"] = int(nframes)
                err_df["model"] = model_label
                error_rows.append(err_df)

                score_rows.append(
                    {
                        "bird": bird,
                        "train_trial": int(train_trial),
                        "eval_trial": int(eval_trial),
                        "pair": pair_label,
                        "pair_type": pair_type,
                        "method": method,
                        "nframes": int(nframes),
                        "model": model_label,
                        **summary,
                        "config_path": str(config_path),
                        "pred_csv": str(pred_csv),
                        "status": "ok",
                        "error": "",
                    }
                )

                if delete_intermediate_predictions:
                    pred_csv.unlink(missing_ok=True)

            except Exception as exc:
                score_rows.append(
                    {
                        "bird": bird,
                        "train_trial": int(train_trial),
                        "eval_trial": int(eval_trial),
                        "pair": pair_label,
                        "pair_type": pair_type,
                        "method": method,
                        "nframes": int(nframes),
                        "model": model_label,
                        "rmse_px": np.nan,
                        "threshold_px": float(threshold_px),
                        "percent_points_within_threshold": np.nan,
                        "percent_frames_all_points_within_threshold": np.nan,
                        "mean_n_predictions_within_threshold_per_frame": np.nan,
                        "std_n_predictions_within_threshold_per_frame": np.nan,
                        "n_points_compared": 0,
                        "n_frames_compared": 0,
                        "config_path": str(config_path),
                        "pred_csv": "",
                        "status": "error",
                        "error": str(exc),
                    }
                )

    scores_df = pd.DataFrame(score_rows)
    errors_df = pd.concat(error_rows, ignore_index=True) if error_rows else pd.DataFrame()

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_root = data_root / "evaluation_results"
    out_root.mkdir(parents=True, exist_ok=True)

    full_scores_csv = out_root / f"{bird}_within_across_scores_{stamp}.csv"
    scores_df.to_csv(full_scores_csv, index=False)

    full_errors_csv = None
    if not errors_df.empty:
        full_errors_csv = out_root / f"{bird}_within_across_point_errors_{stamp}.csv"
        errors_df.to_csv(full_errors_csv, index=False)

    per_eval_paths: list[Path] = []
    if not scores_df.empty and "eval_trial" in scores_df.columns:
        for eval_trial, sub_df in scores_df.groupby("eval_trial", dropna=False):
            if pd.isna(eval_trial):
                continue
            out_csv = out_root / f"{bird}_EvalT{int(eval_trial)}_within_across_scores_{stamp}.csv"
            sub_df.sort_values(["pair_type", "train_trial", "method", "nframes"]).to_csv(out_csv, index=False)
            per_eval_paths.append(out_csv)

    return {
        "bird": bird,
        "scores_df": scores_df,
        "errors_df": errors_df,
        "full_scores_csv": full_scores_csv,
        "full_errors_csv": full_errors_csv,
        "per_eval_paths": per_eval_paths,
    }


def evaluate_within_across_all_birds(
    *,
    data_root: str | Path,
    build_manifest_path: str | Path,
    birds: list[str] | None = None,
    threshold_px: float = 5.0,
    fps: int = 500,
    batchsize: int = 16,
    delete_intermediate_predictions: bool = False,
    force_rebuild_avi: bool = True,
) -> dict[str, Any]:
    build_manifest_path = Path(build_manifest_path)
    build_manifest_df = pd.read_csv(build_manifest_path)

    if birds is None:
        birds = sorted(build_manifest_df["bird"].dropna().astype(str).unique().tolist())

    all_results: dict[str, Any] = {}
    all_scores: list[pd.DataFrame] = []

    for bird in birds:
        bird_result = evaluate_within_across_for_bird(
            bird=str(bird),
            data_root=data_root,
            build_manifest_path=build_manifest_path,
            threshold_px=float(threshold_px),
            fps=int(fps),
            batchsize=int(batchsize),
            delete_intermediate_predictions=bool(delete_intermediate_predictions),
            force_rebuild_avi=bool(force_rebuild_avi),
        )
        all_results[str(bird)] = bird_result
        all_scores.append(bird_result["scores_df"])

    combined_scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()

    return {
        "results_by_bird": all_results,
        "combined_scores_df": combined_scores,
    }
