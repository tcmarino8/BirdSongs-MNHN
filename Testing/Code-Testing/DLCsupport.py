"""DeepLabCut support utilities for XROMM / bird-song analysis workflows.

All functions are stateless: they receive every required path and setting as
an explicit argument instead of reading module-level globals.  This makes
them safe to import from a notebook, call from another script, or invoke
directly from the command line.

``deeplabcut`` and ``xrommtools`` are imported lazily (inside the functions
that need them) so the module can be loaded for path utilities alone in
environments where those heavy packages are not yet installed.

-------------------------------------------------------------------------------
Notebook usage
-------------------------------------------------------------------------------
Place ``DLCsupport.py`` in the same folder as your notebook and import it:

    import DLCsupport as dlcs
    from DLCsupport import (
        as_posix_str, validate_path_exists, find_latest_snapshot,
        load_bodyparts, assert_bodyparts_match,
        create_combined_project_if_missing, require_bodyparts_review_before_training,
        run_combined_experiment,
    )

Common tasks:

    # Print bodyparts from a project config
    parts = dlcs.load_bodyparts(CAM1_CONFIG)

    # Verify both cameras share the same bodyparts before transfer training
    dlcs.assert_bodyparts_match(CAM1_CONFIG, CAM2_CONFIG)

    # Find the latest saved model snapshot
    snap = dlcs.find_latest_snapshot(CAM1_CONFIG)
    print(snap)

    # Create (or reuse) a combined-camera project
    combined_config = dlcs.create_combined_project_if_missing(
        task=TASK,
        experimenter=EXPERIMENTER,
        combined_project_root=COMBINED_PROJECT_ROOT,
        dummy_video=DUMMY_VIDEO_CAM1,
    )

    # Gate: prompt for bodyparts review then run training + analysis
    dlcs.require_bodyparts_review_before_training(
        run_training=RUN_TRAINING,
        config_paths=[CAM1_CONFIG, CAM2_CONFIG, combined_config],
    )

    result = dlcs.run_combined_experiment(
        combined_config=combined_config,
        epochs=200,
        run_training=RUN_TRAINING,
        run_analysis=RUN_VIDEO_ANALYSIS,
        test_videos=[TEST_VIDEO_CAM1, TEST_VIDEO_CAM2],
        data_path=DATA_PATH,
        dataset_name=DATASET_NAME,
        experimenter=EXPERIMENTER,
        nframes=800,
        frame_selection_seed=42,
    )
    print(result)

    # Convert a TIFF frame folder to an AVI video
    summary = dlcs.tiff_stack_to_avi(
        input_folder=ROOT / "newdata/Miguel20260401Trial6_Cam3/deeper/20786 - (Slave)_F6_4987 - CopyUND",
        output_path=ROOT / "newdata/Miguel20260401Trial6_Cam3/OOD-tester-filtered.avi",
        fps=750,
    )
    print(summary)

-------------------------------------------------------------------------------
Command-line usage
-------------------------------------------------------------------------------
Run ``python DLCsupport.py --help`` to see all available subcommands.

    # Print the bodyparts list from a project config
    python DLCsupport.py bodyparts path/to/config.yaml

    # Find the most recent model snapshot
    python DLCsupport.py snapshot path/to/config.yaml

    # List scorer names found in labeled-data H5 files
    python DLCsupport.py scorers path/to/config.yaml

    # Verify two configs share the same bodyparts (required for transfer training)
    python DLCsupport.py check-bodyparts path/to/config_a.yaml path/to/config_b.yaml

    
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ===========================================================================
# Bird-specific bodypart configuration
# ===========================================================================


# Add or update per-bird bodypart lists here. Keys are normalized internally,
# so "DavidBowie" and "davidbowie" resolve to the same entry.
BIRD_BODYPARTS: dict[str, list[str]] = {
    "Endive": [
        "Tongue_Marker",
         "Glottis_Marker",
        "Cranium_Marker_Anterior", 
        "Cranium_Marker_Middle_Left", 
        "Cranium_Marker_Posterior", 
        "Cranium_Marker_Middle_Right", 
        "Maxilla_Marker_Left_Posterior", 
        "Maxilla_Marker_Left_Anterior", 
        "Maxilla_Marker_Right_Posterior", 
        "Mandible_Marker_Right", 
        "Mandible_Marker_Left_Anterior", 
        "Mandible_Marker_Left_Posterior", 
        "Neck_Marker_Right_Superior", 
        "Neck_Marker_Left_Superior", 
        "Neck_Marker_Left_Inferior", 
        "Neck_Marker_Right_Inferior", 
        "Keel_Marker_Left_Superior", 
        "Keel_Marker_Left_Inferior", 
        "Keel_Marker_Right_Superior", 
        "Keel_Marker_Right_Inferior", 
        "Pelvis_Marker_Left", 
        "Pelvis_Marker_Right_Superior", 
        "Pelvis_Marker_Right_Inferior", 
        "Maxilla_Marker_Right_Anterior"
    ],
    "DavidBowie": [
        "Cranium_Left_Anterior", 
        "Cranium_Left_Middle", 
        "Cranium_Right_Middle", 
        "Cranium_Right_Posterior", 
        "Beak_Mandible_Left_Posterior", 
        "Beak_Mandibile_Left_Anterior", 
        "Beak_Mandible_Right_Anterior", 
        "Beak_Maxilla_Right_Anterior", 
        "Beak_Maxilla_Right_Posterior", 
        "Beak_Maxilla_Left_Anterior", 
        "Beak_Maxilla_Left_Posterior", 
        "Tongue_Lower_Palette", 
        "Glottis", 
        "Superior_Trachea", 
        "Keel_Dorsal_Anterior", 
        "Keel_Dorsal_Posterior", 
        "Keel_Inferior", 
        "Pelvis_Right_Inferior", 
        "Pelvis_Right_Superior", 
        "Pelvis_Left_Superior", 
        "Pelvis_Left_Inferior"
    ]

}


def normalize_bird_key(bird_name: str) -> str:
    """Normalize a bird name to a compact case-insensitive lookup key."""
    return re.sub(r"[^a-z0-9]", "", str(bird_name).lower())


def get_bird_bodyparts(
    bird_name: str,
    *,
    strict: bool = True,
) -> list[str]:
    """Return configured bodyparts for a bird from :data:`BIRD_BODYPARTS`."""
    lookup = {normalize_bird_key(k): list(v) for k, v in BIRD_BODYPARTS.items()}
    key = normalize_bird_key(bird_name)
    if key not in lookup:
        if strict:
            raise KeyError(
                f"No bodyparts configured for bird '{bird_name}'. "
                "Add this bird to BIRD_BODYPARTS in DLCsupport.py."
            )
        return []
    return lookup[key]


def edit_bodyparts_in_config(
    config_path: Path | str,
    bodyparts: list[str],
) -> None:
    """Update ``bodyparts`` in a DLC config using ``edit_config`` when available."""
    config_path = Path(config_path)
    if not bodyparts:
        raise ValueError(f"Refusing to write empty bodyparts list to {config_path}")

    try:
        from deeplabcut.utils.auxiliaryfunctions import edit_config  # noqa: PLC0415

        edit_config(as_posix_str(config_path), {"bodyparts": bodyparts})
    except Exception:
        # Fallback for environments where auxiliary edit_config import path differs.
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        cfg["bodyparts"] = list(bodyparts)
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)


def apply_bird_bodyparts_to_configs(
    configs_by_bird: dict[str, list[Path | str]],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Apply configured bird bodyparts to each config and return a summary table."""
    rows: list[dict[str, str | int]] = []

    for bird_name, config_paths in configs_by_bird.items():
        parts = get_bird_bodyparts(bird_name, strict=strict)
        if not parts:
            rows.append(
                {
                    "bird": bird_name,
                    "config": "(none)",
                    "status": "skipped-no-bodyparts",
                    "n_bodyparts": 0,
                }
            )
            continue

        for cfg in config_paths:
            cfg = Path(cfg)
            validate_path_exists(cfg, f"config for bird {bird_name}")
            edit_bodyparts_in_config(cfg, parts)
            applied = load_bodyparts(cfg)
            status = "ok" if applied == parts else "mismatch-after-write"
            rows.append(
                {
                    "bird": bird_name,
                    "config": str(cfg),
                    "status": status,
                    "n_bodyparts": len(applied),
                }
            )

    return pd.DataFrame(rows)


# ===========================================================================
# Data containers
# ===========================================================================


@dataclass
class ExperimentResult:
    """Record the outcome of one training/analysis experiment.

    Attributes
    ----------
    name : str
        Short identifier for this run, e.g. ``"combined_cam1_cam2"``.
    config_used : str
        Absolute POSIX path to the ``config.yaml`` used for training or
        evaluation.
    init_snapshot : str
        Path to the ``.pt`` snapshot used to initialise weights, or the
        string ``"fresh"`` when training started from backbone pretrained
        weights.
    trained : bool
        ``True`` when the training step was executed in this run.
    analyzed : bool
        ``True`` when video analysis was executed in this run.
    notes : str
        Free-text field; usually contains the path to the latest snapshot
        produced by training, or states why the run was a dry run.
    """

    name: str
    config_used: str
    init_snapshot: str
    trained: bool
    analyzed: bool
    notes: str

    def __str__(self) -> str:
        return (
            f"Experiment : {self.name}\n"
            f"Config     : {self.config_used}\n"
            f"Init snap  : {self.init_snapshot}\n"
            f"Trained    : {self.trained}\n"
            f"Analyzed   : {self.analyzed}\n"
            f"Notes      : {self.notes}"
        )


# ===========================================================================
# Path utilities
# ===========================================================================


def as_posix_str(path_obj: Path | str) -> str:
    """Return an absolute POSIX-style (forward-slash) path string.

    DeepLabCut and related tools expect string paths with forward slashes
    on all platforms.  This helper resolves relative paths and normalises
    the separator.

    Parameters
    ----------
    path_obj : Path | str
        Input path.  Relative paths are resolved against the current working
        directory.

    Returns
    -------
    str
        Absolute path using ``/`` as the separator.

    Examples
    --------
    >>> as_posix_str(Path("..") / "DeepLabCut" / "config.yaml")
    'C:/Users/.../DeepLabCut/config.yaml'
    """
    return str(Path(path_obj).resolve()).replace("\\", "/")


def validate_path_exists(path_obj: Path | str, label: str) -> None:
    """Raise ``FileNotFoundError`` if *path_obj* does not exist on disk.

    Parameters
    ----------
    path_obj : Path | str
        Path to check.
    label : str
        Human-readable description used in the error message so callers can
        identify which path was missing, e.g. ``"cam1 test video"``.

    Raises
    ------
    FileNotFoundError
        If ``path_obj`` does not point to an existing file or directory.
    """
    if not Path(path_obj).exists():
        raise FileNotFoundError(f"Missing {label}: {path_obj}")


# ===========================================================================
# Config and snapshot helpers
# ===========================================================================


def get_project_dir_from_config(config_path: Path | str) -> Path:
    """Return the DeepLabCut project root directory for a given config path.

    Parameters
    ----------
    config_path : Path | str
        Path to a ``config.yaml`` file inside a DeepLabCut project.

    Returns
    -------
    Path
        The directory that directly contains the ``config.yaml`` file.
    """
    return Path(config_path).parent


def find_latest_snapshot(config_path: Path | str) -> str:
    """Return the path to the most recently saved PyTorch training snapshot.

    Searches the standard DLC PyTorch model directory tree::

        <project>/dlc-models-pytorch/iteration-0/*/train/snapshots/snapshot-*.pt

    Parameters
    ----------
    config_path : Path | str
        Path to the project ``config.yaml``.

    Returns
    -------
    str
        Absolute POSIX path to the latest ``snapshot-*.pt`` file.  Results
        are sorted lexicographically, which matches chronological order
        because DLC names snapshots by epoch number.

    Raises
    ------
    FileNotFoundError
        If no snapshots are found under the expected pattern.

    Examples
    --------
    >>> snap = find_latest_snapshot(Path("project/config.yaml"))
    >>> print(snap)
    '.../snapshots/snapshot-200.pt'
    """
    project_dir = get_project_dir_from_config(config_path)
    pattern = (
        project_dir
        / "dlc-models-pytorch"
        / "iteration-0"
        / "*"
        / "train"
        / "snapshots"
        / "snapshot-*.pt"
    )
    snapshots = sorted(_glob.glob(as_posix_str(pattern)))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under pattern: {pattern}")
    return snapshots[-1]


def load_bodyparts(config_path: Path | str) -> list:
    """Read the ``bodyparts`` list from a DeepLabCut ``config.yaml``.

    Parameters
    ----------
    config_path : Path | str
        Path to the ``config.yaml`` file.

    Returns
    -------
    list
        The ``bodyparts`` list, or an empty list when the key is absent.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("bodyparts", [])


def get_available_scorers(config_path: Path | str) -> list[str]:
    """Discover scorer names from labeled-data H5 files in a DLC project.

    Reads every ``CollectedData_*.h5`` file under the project's
    ``labeled-data/`` folder and collects the unique scorer names stored in
    MultiIndex column level 0.

    Parameters
    ----------
    config_path : Path | str
        Path to the project ``config.yaml``.

    Returns
    -------
    list[str]
        Sorted list of unique scorer names.  Empty if no H5 files exist or
        none contain a valid scorer name.
    """
    project_dir = get_project_dir_from_config(config_path)
    scorer_set: set[str] = set()
    for h5_path in project_dir.glob("labeled-data/*/CollectedData_*.h5"):
        try:
            df = pd.read_hdf(as_posix_str(h5_path))
        except Exception:
            continue
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 0:
            values = [str(v) for v in df.columns.get_level_values(0).unique()]
            scorer_set.update(v for v in values if v and v.lower() != "nan")
    return sorted(scorer_set)


def ensure_config_scorer_matches_data(config_path: Path | str) -> str | None:
    """Synchronise the ``scorer`` field in ``config.yaml`` with labeled-data.

    If the scorer name written in ``config.yaml`` does not match any scorer
    name found in the project H5 files, the config is updated in-place to
    use the first available scorer.  This prevents DLC from silently
    creating duplicate labeled-data entries when the scorer field drifts
    from the actual data.

    Parameters
    ----------
    config_path : Path | str
        DeepLabCut config path to inspect and, if necessary, update.

    Returns
    -------
    str | None
        The scorer name now active in the config, or ``None`` when no scorer
        could be found in the project data.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    available = get_available_scorers(config_path)
    if not available:
        logger.warning("No scorer found in labeled-data for %s", config_path)
        return None

    current = str(cfg.get("scorer", "")).strip()
    if current in available:
        return current

    chosen = available[0]
    cfg["scorer"] = chosen
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    logger.warning(
        "Updated scorer in %s from '%s' to '%s'", config_path, current, chosen
    )
    return chosen


def assert_bodyparts_match(
    config_a: Path | str,
    config_b: Path | str,
) -> None:
    """Raise if two DLC projects define different bodyparts lists.

    Snapshot transfer training is only valid when both projects share
    *identical* bodyparts in the same order.  Call this before initiating
    a transfer to catch mismatches early.

    Parameters
    ----------
    config_a : Path | str
        First project ``config.yaml`` path.
    config_b : Path | str
        Second project ``config.yaml`` path.

    Raises
    ------
    ValueError
        If the bodyparts lists differ between the two configs, with a
        message showing both lists.
    """
    bpa = load_bodyparts(config_a)
    bpb = load_bodyparts(config_b)
    if bpa != bpb:
        raise ValueError(
            "Bodyparts mismatch between configs; snapshot transfer would be invalid.\n"
            f"  {config_a}: {bpa}\n"
            f"  {config_b}: {bpb}"
        )


# ===========================================================================
# Project and dataset management
# ===========================================================================
def set_net_type(config_path: Path, net_type: str) -> None:
    """Patch the net_type field in a DLC config.yaml in-place."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["default_net_type"] = net_type
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

def create_combined_project_if_missing(
    task: str,
    experimenter: str,
    combined_project_root: Path | str,
    dummy_video: Path | str,
) -> Path:
    """Create a new combined-camera DLC project, or reuse an existing one.

    DeepLabCut requires a video at project creation time.  When training data
    is loaded from XMA point files the video is not actually used, but DLC
    still needs it to initialise the project structure.  The ``dummy_video``
    argument satisfies this requirement with any existing ``.avi`` file.

    If a project folder matching ``<task>-<experimenter>-*`` already exists
    inside ``combined_project_root``, its ``config.yaml`` is returned
    immediately without creating anything new.

    Parameters
    ----------
    task : str
        Task name — first component of the DLC project folder name
        (e.g. ``"Canari"``).
    experimenter : str
        Experimenter name — second component of the folder name
        (e.g. ``"Tyler"``).
    combined_project_root : Path | str
        Parent directory in which the combined project folder is created.
    dummy_video : Path | str
        Path to any existing video file used only to satisfy DLC's project
        creation API.

    Returns
    -------
    Path
        Path to the ``config.yaml`` of the (new or existing) project.

    Raises
    ------
    FileNotFoundError
        If *dummy_video* does not exist on disk.
    """
    import deeplabcut  # noqa: PLC0415 – lazy import keeps startup fast

    dummy_video = Path(dummy_video)
    combined_project_root = Path(combined_project_root)

    validate_path_exists(dummy_video, "dummy video")
    combined_project_root.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        combined_project_root.glob(f"{task}-{experimenter}-*/config.yaml")
    )
    if existing:
        logger.info("Reusing existing combined project: %s", existing[-1])
        return existing[-1]

    try:
        config_path = deeplabcut.create_new_project(
            task,
            experimenter,
            [as_posix_str(dummy_video)],
            as_posix_str(combined_project_root),
            copy_videos=False,
        )  
        # FIXME: Edit Config...
        # FIXME: augmentation adjustment in pose_config
    except TypeError:
        # Fallback for DLC builds that require keyword-only arguments.
        config_path = deeplabcut.create_new_project(
            project=task,
            experimenter=experimenter,
            videos=[as_posix_str(dummy_video)],
            working_directory=as_posix_str(combined_project_root),
            copy_videos=False,
        )

    logger.info("Created combined project: %s", config_path)
    return Path(config_path)


def build_combined_dataset(
    combined_config: Path | str,
    data_path: Path | str,
    dataset_name: str,
    experimenter: str,
    nframes: int,
    frame_selection_seed: int | None = None,
) -> None:
    """Populate a combined project's labeled-data from XMA training folders.

    Calls ``xrommtools.xma_to_dlc`` with ``nnetworks=1`` so frames from
    both cameras are included in a single DLC dataset.  The random state is
    saved and restored around the call so the seed only affects frame
    sampling and does not bleed into subsequent operations.

    Parameters
    ----------
    combined_config : Path | str
        Path to the combined project ``config.yaml``.
    data_path : Path | str
        Root directory containing XMA-style trial sub-folders, each with a
        2-D points CSV and one or more camera video files or frame folders.
    dataset_name : str
        Dataset name forwarded to ``xma_to_dlc``
        (used as the labeled-data sub-folder prefix).
    experimenter : str
        Scorer name forwarded to ``xma_to_dlc``
        (written into the H5 column MultiIndex).
    nframes : int
        Total number of frames to sample across all trials.
    frame_selection_seed : int | None, optional
        Random seed for reproducible frame selection.  When ``None``, the
        current ``random`` module state is used unchanged.

    Raises
    ------
    FileNotFoundError
        If *data_path* does not exist.
    """
    import xrommtools  # noqa: PLC0415

    combined_config = Path(combined_config)
    data_path = Path(data_path)

    validate_path_exists(data_path, "training data path")

    prev_state = random.getstate()
    try:
        if frame_selection_seed is not None:
            random.seed(frame_selection_seed)
            logger.info("Frame selection seed set to %s", frame_selection_seed)

        xrommtools.xma_to_dlc(
            as_posix_str(combined_config),
            as_posix_str(data_path),
            dataset_name,
            experimenter,
            nframes,
            nnetworks=1,
        )
    finally:
        random.setstate(prev_state)


# ===========================================================================
# Training
# ===========================================================================


def require_bodyparts_review_before_training(
    run_training: bool,
    config_paths: list[Path | str] | None = None,
    configs_by_bird: dict[str, list[Path | str]] | None = None,
) -> None:
    """Gate training on explicit confirmation that bodyparts have been reviewed.

    Prints the bodyparts count and first five entries for every config in
    *config_paths*, then blocks until the user types the exact phrase
    ``CONFIRMED BODYPARTS``.  This is a last-chance check before committing
    to a potentially long training run with the wrong keypoint definitions.

    Parameters
    ----------
    run_training : bool
        When ``False`` the gate is skipped entirely (useful for dry-run mode).
    config_paths : list[Path | str] | None, optional
        Legacy input: flat list of configs to review in one group.
    configs_by_bird : dict[str, list[Path | str]] | None, optional
        Preferred input: mapping of bird name to config list. Each bird is
        reviewed and confirmed exactly once.

    Raises
    ------
    RuntimeError
        If the user does not enter the confirmation phrase exactly.
    """
    if not run_training:
        print("Bodyparts checkpoint skipped (RUN_TRAINING is False).")
        return

    if configs_by_bird is None:
        if not config_paths:
            raise ValueError(
                "Provide either config_paths or configs_by_bird for bodyparts review."
            )
        configs_by_bird = {"all": list(config_paths)}

    print("\nReview bodyparts per bird before training:")
    for bird_name, cfg_list in configs_by_bird.items():
        print(f"\nBird: {bird_name}")
        expected_parts = get_bird_bodyparts(bird_name, strict=False)
        if expected_parts:
            print(f"  configured bodyparts ({len(expected_parts)}): {expected_parts}")
        else:
            print("  configured bodyparts: (none configured in BIRD_BODYPARTS)")

        for cfg in cfg_list:
            cfg = Path(cfg)
            validate_path_exists(cfg, f"config {cfg.name}")
            parts = load_bodyparts(cfg)
            print(f"  {cfg}")
            print(f"    bodyparts ({len(parts)}): {parts}")

        phrase = f"CONFIRMED {bird_name}"
        ack = input(f"Type exactly '{phrase}' to proceed with {bird_name}: ").strip()
        if ack != phrase:
            raise RuntimeError(
                "Training stopped. Update bodyparts in the config files and "
                f"confirm with the exact phrase '{phrase}'."
            )

    print("All bird confirmations accepted. Training can proceed.")


def apply_and_review_bird_bodyparts(
    run_training: bool,
    configs_by_bird: dict[str, list[Path | str]],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Apply bodyparts from :data:`BIRD_BODYPARTS`, then require per-bird review."""
    summary_df = apply_bird_bodyparts_to_configs(configs_by_bird, strict=strict)
    require_bodyparts_review_before_training(
        run_training=run_training,
        configs_by_bird=configs_by_bird,
    )
    return summary_df


def create_and_train(
    config_path: Path | str,
    epochs: int,
    snapshot_path: str | None = None,
    modelprefix: str | None = None,
    post_dataset_callback=None,
    train_network_kwargs: dict | None = None,
) -> str:
    """Create a DLC training dataset and run model training.

    Before calling ``create_training_dataset`` the config scorer is
    synchronised with whatever scorer name appears in the labeled-data H5
    files (see :func:`ensure_config_scorer_matches_data`), preventing
    silent scorer mismatches.

    Parameters
    ----------
    config_path : Path | str
        Project ``config.yaml`` path.
    epochs : int
        Number of training epochs.
    snapshot_path : str | None, optional
        Absolute path to a ``.pt`` snapshot to use as the starting weights.
        When ``None`` training begins from the backbone's pretrained
        ImageNet weights (no transfer).
    modelprefix : str | None, optional
        Optional DLC model prefix label, useful for tagging augmentation
        variants in one experiment matrix.
    post_dataset_callback : callable | None, optional
        Optional callback invoked as ``post_dataset_callback(config_path)``
        immediately after ``create_training_dataset`` and before
        ``train_network``. Use this to patch generated training configs
        (for example, augmentation options in ``pose_cfg.yaml``).
    train_network_kwargs : dict | None, optional
        Extra keyword arguments forwarded to ``deeplabcut.train_network``.

    Returns
    -------
    str
        POSIX path to the latest snapshot produced by this training run.
    """
    import deeplabcut  # noqa: PLC0415

    config_path = Path(config_path)
    ensure_config_scorer_matches_data(config_path)
    deeplabcut.create_training_dataset(as_posix_str(config_path))  
    if post_dataset_callback is not None:
        post_dataset_callback(config_path)

    train_kwargs = {"epochs": epochs}
    if snapshot_path is not None:
        train_kwargs["snapshot_path"] = snapshot_path
    if modelprefix:
        train_kwargs["modelprefix"] = modelprefix
    if train_network_kwargs:
        train_kwargs.update(train_network_kwargs)

    deeplabcut.train_network(as_posix_str(config_path), **train_kwargs)

    return find_latest_snapshot(config_path)


# ===========================================================================
# Evaluation and analysis
# ===========================================================================


def run_eval_bundle(
    config_path: Path | str,
    videos: list[Path | str],
    run_analysis: bool,
    dest_folder: Path | str | None = None,
) -> bool:
    """Run video analysis and produce labeled outputs for a list of videos.

    For each video, calls ``deeplabcut.analyze_videos``,
    ``create_labeled_video``, and ``plot_trajectories`` in sequence.

    Parameters
    ----------
    config_path : Path | str
        Project ``config.yaml`` path.
    videos : list[Path | str]
        Videos to process.
    run_analysis : bool
        When ``False`` the function returns immediately without running
        anything.  This mirrors the ``RUN_VIDEO_ANALYSIS`` safety toggle
        used in the notebook.
    dest_folder : Path | str | None, optional
        Output directory for all generated files.  When ``None`` (default),
        each video's own parent directory is used as the destination.

    Returns
    -------
    bool
        ``True`` when analysis was executed, ``False`` when skipped.
    """
    import deeplabcut  # noqa: PLC0415

    if not run_analysis:
        return False

    config_str = as_posix_str(Path(config_path))
    for video_path in videos:
        video_path = Path(video_path)
        validate_path_exists(video_path, f"test video {video_path.name}")
        dest = (
            as_posix_str(Path(dest_folder))
            if dest_folder
            else as_posix_str(video_path.parent)
        )
        deeplabcut.analyze_videos(
            config_str, [as_posix_str(video_path)], destfolder=dest
        )
        deeplabcut.create_labeled_video(
            config_str, [as_posix_str(video_path)], videotype=".avi", destfolder=dest
        )
        deeplabcut.plot_trajectories(
            config_str, [as_posix_str(video_path)], videotype=".avi", destfolder=dest
        )
    return True


def run_combined_experiment(
    combined_config: Path | str,
    epochs: int,
    run_training: bool,
    run_analysis: bool,
    test_videos: list[Path | str],
    data_path: Path | str,
    dataset_name: str,
    experimenter: str,
    nframes: int,
    frame_selection_seed: int | None = None,
) -> ExperimentResult:
    """Top-level orchestration for the combined Cam1+Cam2 training experiment.

    Builds the combined dataset, trains the model, and optionally runs video
    analysis — all in one call.  When ``run_training=False`` the function
    returns immediately with a dry-run :class:`ExperimentResult` so the rest
    of the notebook can continue without side effects.

    Parameters
    ----------
    combined_config : Path | str
        Path to the combined project ``config.yaml``.  Create it first with
        :func:`create_combined_project_if_missing`.
    epochs : int
        Training epochs.
    run_training : bool
        When ``False``, short-circuits without building a dataset or training.
    run_analysis : bool
        When ``True``, runs video analysis on *test_videos* after training.
    test_videos : list[Path | str]
        Videos used for post-training analysis.  Ignored when
        ``run_analysis=False``.
    data_path : Path | str
        Root directory of XMA training data trial folders.
    dataset_name : str
        Dataset name passed to ``xma_to_dlc``.
    experimenter : str
        Scorer name passed to ``xma_to_dlc``.
    nframes : int
        Total labeled frames to sample for the combined dataset.
    frame_selection_seed : int | None, optional
        Random seed for reproducible frame sampling.

    Returns
    -------
    ExperimentResult
        Structured summary of what was (or was not) executed.
    """
    combined_config = Path(combined_config)

    if not run_training:
        return ExperimentResult(
            name="combined_cam1_cam2",
            config_used=as_posix_str(combined_config),
            init_snapshot="fresh",
            trained=False,
            analyzed=False,
            notes="Dry run. Set run_training=True to execute.",
        )

    build_combined_dataset(
        combined_config=combined_config,
        data_path=data_path,
        dataset_name=dataset_name,
        experimenter=experimenter,
        nframes=nframes,
        frame_selection_seed=frame_selection_seed,
    )
    latest = create_and_train(combined_config, epochs=epochs)
    analyzed = run_eval_bundle(combined_config, test_videos, run_analysis)

    return ExperimentResult(
        name="combined_cam1_cam2",
        config_used=as_posix_str(combined_config),
        init_snapshot="fresh",
        trained=True,
        analyzed=analyzed,
        notes=f"Latest snapshot: {latest}",
    )


# ===========================================================================
# Command-line interface
# ===========================================================================


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI for DLCsupport.py.

    Subcommands provide quick inspection tasks that are commonly needed
    without launching a full notebook session.
    """
    parser = argparse.ArgumentParser(
        prog="DLCsupport",
        description=(
            "DeepLabCut support utilities for XROMM / bird-song workflows.\n"
            "Each subcommand performs a quick inspection or conversion task."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python DLCsupport.py bodyparts       project/config.yaml\n"
            "  python DLCsupport.py snapshot        project/config.yaml\n"
            "  python DLCsupport.py scorers         project/config.yaml\n"
            "  python DLCsupport.py check-bodyparts config_a.yaml config_b.yaml\n"
            "  python DLCsupport.py tiff-to-avi     frames/ output.avi --fps 750"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- bodyparts ----------------------------------------------------------
    p = sub.add_parser(
        "bodyparts",
        help="Print the numbered bodyparts list from a config.yaml.",
        description=(
            "Read and display the bodyparts defined in a DeepLabCut config.yaml."
        ),
    )
    p.add_argument("config", type=Path, help="Path to config.yaml.")

    # ---- snapshot -----------------------------------------------------------
    p = sub.add_parser(
        "snapshot",
        help="Print the path to the latest training snapshot.",
        description=(
            "Search the standard DLC PyTorch model tree and print the path "
            "of the most recent snapshot-*.pt file."
        ),
    )
    p.add_argument("config", type=Path, help="Path to config.yaml.")

    # ---- scorers ------------------------------------------------------------
    p = sub.add_parser(
        "scorers",
        help="List scorer names found in labeled-data H5 files.",
        description=(
            "Scan every CollectedData_*.h5 file in labeled-data/ and print "
            "the unique scorer names stored at MultiIndex level 0."
        ),
    )
    p.add_argument("config", type=Path, help="Path to config.yaml.")

    # ---- check-bodyparts ----------------------------------------------------
    p = sub.add_parser(
        "check-bodyparts",
        help="Verify two configs share the same bodyparts (required for transfer training).",
        description=(
            "Compare the bodyparts lists in two config.yaml files.  "
            "Exits with code 0 on a match, code 1 on a mismatch."
        ),
    )
    p.add_argument("config_a", type=Path, help="First project config.yaml.")
    p.add_argument("config_b", type=Path, help="Second project config.yaml.")

    # ---- tiff-to-avi --------------------------------------------------------
    p = sub.add_parser(
        "tiff-to-avi",
        help="Convert a folder of TIFF frames into an AVI video.",
        description=(
            "Encode a numbered TIFF frame sequence into an AVI file.  "
            "Frames are sorted by the last numeric token in each filename by default."
        ),
    )
    p.add_argument("input_folder", type=Path, help="Folder containing TIFF frames.")
    p.add_argument("output_path", type=Path, help="Destination AVI file path.")
    p.add_argument(
        "--fps", type=int, default=500,
        help="Frames per second to encode (default: 500).",
    )
    p.add_argument(
        "--fourcc", default="MJPG",
        help="FourCC codec string for OpenCV VideoWriter (default: MJPG).",
    )
    p.add_argument(
        "--no-sort-numeric",
        action="store_true",
        help="Use plain lexical filename sorting instead of numeric ordering.",
    )

    return parser


def _cli_main() -> None:
    """Entry point for command-line execution."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command == "bodyparts":
        parts = load_bodyparts(args.config)
        print(f"Bodyparts ({len(parts)}) in {args.config}:")
        for i, bp in enumerate(parts, 1):
            print(f"  {i:>3}. {bp}")

    elif args.command == "snapshot":
        try:
            print(find_latest_snapshot(args.config))
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "scorers":
        scorers = get_available_scorers(args.config)
        if not scorers:
            print("No scorer names found in labeled-data H5 files.")
        else:
            print(f"Scorers found ({len(scorers)}):")
            for s in scorers:
                print(f"  {s}")

    elif args.command == "check-bodyparts":
        try:
            assert_bodyparts_match(args.config_a, args.config_b)
            print("OK: bodyparts match.")
        except ValueError as exc:
            print(f"MISMATCH: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "tiff-to-avi":
        summary = tiff_stack_to_avi(
            input_folder=args.input_folder,
            output_path=args.output_path,
            fps=args.fps,
            fourcc=args.fourcc,
            sort_numeric=not args.no_sort_numeric,
        )
        print("Conversion complete:")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli_main()
