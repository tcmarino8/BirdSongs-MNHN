from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_codetesting_on_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    code_testing = repo_root / "Code-Testing"
    if str(code_testing) not in sys.path:
        sys.path.insert(0, str(code_testing))
    return code_testing


_ensure_codetesting_on_syspath()
import DLCsupport as dlcs  # noqa: E402


@dataclass
class TrainJob:
    bird: str
    trial_num: int
    method: str
    nframes: int
    config_path: str


@dataclass
class TrainResult:
    bird: str
    trial_num: int
    method: str
    nframes: int
    config_path: str
    status: str
    elapsed_seconds: float
    latest_snapshot: str
    removed_best_checkpoints: int
    error: str


# DB token is already expanded in the generated manifest, but keep this for safety.
BIRD_TOKEN_MAP: dict[str, str] = {
    "DB": "DavidBowie",
    "Tulio": "Tulio",
}


def _norm_bird(raw: str) -> str:
    return BIRD_TOKEN_MAP.get(str(raw), str(raw))


def _load_jobs(
    manifest_path: Path,
    *,
    methods: set[str],
    nframes: set[int] | None = None,
) -> list[TrainJob]:
    df = pd.read_csv(manifest_path)
    if "config_path" not in df.columns:
        raise ValueError(f"Manifest missing config_path column: {manifest_path}")

    jobs: list[TrainJob] = []
    seen: set[str] = set()

    for _, row in df.iterrows():
        method = str(row.get("method", "")).strip().lower()
        if method not in methods:
            continue

        nframe_val = int(row.get("nframes", -1))
        if nframes is not None and nframe_val not in nframes:
            continue

        config_path = str(row.get("config_path", "")).strip()
        if config_path == "":
            continue

        key = str(Path(config_path).resolve()).lower()
        if key in seen:
            continue
        seen.add(key)

        bird = _norm_bird(str(row.get("bird", "")))
        jobs.append(
            TrainJob(
                bird=bird,
                trial_num=int(row.get("trial_num", -1)),
                method=method,
                nframes=nframe_val,
                config_path=config_path,
            )
        )

    return jobs


def _apply_bodypart_sync_per_bird(jobs: list[TrainJob]) -> pd.DataFrame:
    configs_by_bird: dict[str, list[Path]] = {}
    for job in jobs:
        configs_by_bird.setdefault(job.bird, []).append(Path(job.config_path))

    # Deduplicate while preserving order.
    for bird, cfgs in list(configs_by_bird.items()):
        uniq: list[Path] = []
        seen: set[str] = set()
        for cfg in cfgs:
            key = str(cfg.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(cfg)
        configs_by_bird[bird] = uniq

    summary_df = dlcs.apply_bird_bodyparts_to_configs(configs_by_bird, strict=True)
    return summary_df


def _job_label(job: TrainJob) -> str:
    return f"{job.bird}_T{job.trial_num} | nframes={job.nframes} | method={job.method}"


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(round(width * (current / total)))
    return f"[{('#' * filled) + ('-' * (width - filled))}]"


def _unlink_best_snapshots(config_path: str | Path) -> int:
    combined_config = Path(config_path)
    models_base = combined_config.parent
    removed = 0
    for train_dir in models_base.glob("dlc-models-pytorch/iteration-*/*/train"):
        for file in train_dir.glob("*best*.pt"):
            file.unlink()
            removed += 1
    return removed


class _EpochProgressStream(io.TextIOBase):
    _epoch_re = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")

    def __init__(self, wrapped: io.TextIOBase, job_tag: str) -> None:
        self._wrapped = wrapped
        self._job_tag = job_tag
        self._buffer = ""
        self._last_epoch = -1

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._wrapped.write(s)
        self._wrapped.flush()
        self._buffer += s

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))
        return len(s)

    def flush(self) -> None:
        self._wrapped.flush()

    def _handle_line(self, line: str) -> None:
        match = self._epoch_re.search(line)
        if match is None:
            return

        epoch_idx = int(match.group(1))
        total_epochs = int(match.group(2))
        if epoch_idx == self._last_epoch:
            return
        self._last_epoch = epoch_idx

        bar = _progress_bar(epoch_idx, total_epochs)
        self._wrapped.write(
            f"[epoch-progress] {self._job_tag} {bar} {epoch_idx}/{total_epochs}\n"
        )
        self._wrapped.flush()


def _train_one_job(
    job: TrainJob,
    *,
    epochs: int,
    train_network_kwargs: dict[str, Any] | None,
    modelprefix: str | None,
    assigned_gpu: str | None,
    job_index: int,
    total_jobs: int,
) -> TrainResult:
    if assigned_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    job_tag = _job_label(job)
    print(f"\n[train-job {job_index}/{total_jobs}] {job_tag}")
    print(f"[train-job config] {job.config_path}")
    if assigned_gpu is not None:
        print(f"[train-job gpu] CUDA_VISIBLE_DEVICES={assigned_gpu}")

    t0 = time.perf_counter()
    try:
        progress_stream = _EpochProgressStream(sys.stdout, job_tag)
        with contextlib.redirect_stdout(progress_stream):
            latest_snapshot = dlcs.create_and_train(
                config_path=job.config_path,
                epochs=int(epochs),
                modelprefix=modelprefix,
                train_network_kwargs=train_network_kwargs,
            )

        removed_best = _unlink_best_snapshots(job.config_path)
        elapsed = time.perf_counter() - t0
        print(
            f"[train-job done] {job_tag} | elapsed={elapsed/60.0:.2f} min | "
            f"removed_best_checkpoints={removed_best}"
        )
        return TrainResult(
            bird=job.bird,
            trial_num=int(job.trial_num),
            method=job.method,
            nframes=int(job.nframes),
            config_path=job.config_path,
            status="ok",
            elapsed_seconds=float(elapsed),
            latest_snapshot=str(latest_snapshot),
            removed_best_checkpoints=int(removed_best),
            error="",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"[train-job error] {job_tag} | elapsed={elapsed/60.0:.2f} min | error={exc}")
        return TrainResult(
            bird=job.bird,
            trial_num=int(job.trial_num),
            method=job.method,
            nframes=int(job.nframes),
            config_path=job.config_path,
            status="error",
            elapsed_seconds=float(elapsed),
            latest_snapshot="",
            removed_best_checkpoints=0,
            error=str(exc),
        )


def _json_or_none(payload: str | None) -> dict[str, Any] | None:
    if payload is None or str(payload).strip() == "":
        return None
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("--train-network-kwargs-json must decode to a JSON object")
    return loaded


def _run_serial(
    jobs: list[TrainJob],
    *,
    epochs: int,
    train_network_kwargs: dict[str, Any] | None,
    modelprefix_template: str | None,
) -> list[TrainResult]:
    out: list[TrainResult] = []
    suite_t0 = time.perf_counter()
    for idx, j in enumerate(jobs, start=1):
        modelprefix = None
        if modelprefix_template:
            modelprefix = modelprefix_template.format(
                bird=j.bird,
                trial=j.trial_num,
                method=j.method,
                nframes=j.nframes,
            )
        out.append(
            _train_one_job(
                j,
                epochs=epochs,
                train_network_kwargs=train_network_kwargs,
                modelprefix=modelprefix,
                assigned_gpu=None,
                job_index=idx,
                total_jobs=len(jobs),
            )
        )

        n_done = len(out)
        n_ok = sum(1 for r in out if r.status == "ok")
        n_error = n_done - n_ok
        elapsed_suite = time.perf_counter() - suite_t0
        print(
            f"[suite-progress] completed={n_done}/{len(jobs)} ok={n_ok} error={n_error} "
            f"elapsed={elapsed_suite/60.0:.2f} min"
        )
    return out


def _run_parallel(
    jobs: list[TrainJob],
    *,
    epochs: int,
    train_network_kwargs: dict[str, Any] | None,
    modelprefix_template: str | None,
    max_workers: int,
    gpus: list[str] | None,
) -> list[TrainResult]:
    results: list[TrainResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for idx, j in enumerate(jobs):
            assigned_gpu = None
            if gpus:
                assigned_gpu = gpus[idx % len(gpus)]

            modelprefix = None
            if modelprefix_template:
                modelprefix = modelprefix_template.format(
                    bird=j.bird,
                    trial=j.trial_num,
                    method=j.method,
                    nframes=j.nframes,
                )

            fut = ex.submit(
                _train_one_job,
                j,
                epochs=epochs,
                train_network_kwargs=train_network_kwargs,
                modelprefix=modelprefix,
                assigned_gpu=assigned_gpu,
                job_index=idx + 1,
                total_jobs=len(jobs),
            )
            futures[fut] = j

        for fut in as_completed(futures):
            results.append(fut.result())

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all generated DLC models from manifest configs")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Data") / "training_set_build_manifest.csv",
        help="Path to training_set_build_manifest.csv",
    )
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "displacement", "dino"],
        help="Methods to train",
    )
    parser.add_argument(
        "--nframes",
        nargs="*",
        type=int,
        default=[100, 50],
        help="Frame-count variants to train (empty = all)",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids for worker assignment, e.g. 0,1",
    )
    parser.add_argument(
        "--modelprefix-template",
        type=str,
        default="",
        help="Optional format template for DLC modelprefix (empty disables modelprefix)",
    )
    parser.add_argument(
        "--train-network-kwargs-json",
        type=str,
        default="",
        help="JSON object forwarded to dlcs.create_and_train(..., train_network_kwargs=...)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print jobs and exit without training",
    )

    args = parser.parse_args()

    methods = {m.strip().lower() for m in args.methods}
    nframes = set(args.nframes) if args.nframes else None
    train_network_kwargs = _json_or_none(args.train_network_kwargs_json)

    jobs = _load_jobs(args.manifest, methods=methods, nframes=nframes)
    if len(jobs) == 0:
        raise RuntimeError("No trainable jobs found from manifest with the selected filters")

    print(f"Loaded {len(jobs)} training jobs")
    print("Applying bird bodypart sync to all target configs before training...")
    summary_df = _apply_bodypart_sync_per_bird(jobs)
    print(summary_df.to_string(index=False))

    if args.dry_run:
        print("Dry run enabled. Jobs:")
        for j in jobs:
            print(asdict(j))
        return

    gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip() != ""]

    if int(args.max_workers) <= 1:
        results = _run_serial(
            jobs,
            epochs=int(args.epochs),
            train_network_kwargs=train_network_kwargs,
            modelprefix_template=args.modelprefix_template,
        )
    else:
        results = _run_parallel(
            jobs,
            epochs=int(args.epochs),
            train_network_kwargs=train_network_kwargs,
            modelprefix_template=args.modelprefix_template,
            max_workers=int(args.max_workers),
            gpus=gpus if gpus else None,
        )

    out_df = pd.DataFrame([asdict(r) for r in results]).sort_values(
        ["status", "bird", "trial_num", "method", "nframes"],
        ascending=[True, True, True, True, False],
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.manifest.parent / f"training_run_manifest_{stamp}.csv"
    out_df.to_csv(out_path, index=False)

    print("\nTraining run summary:")
    print(out_df.to_string(index=False))
    print(f"\nSaved run manifest: {out_path}")


if __name__ == "__main__":
    main()
