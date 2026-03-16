"""Command-line rigid body denoising for transform CSV files.

Usage:
    python Code-Testing/Denoise_Rigid_Body_CLI.py <input_csv> <cutoff_hz> [--sample-rate 750]

Behavior matches the GUI "Save Filtered CSV" flow:
- Accepts input as (frames x 16) or (16 x frames)
- Filters each channel independently using FFT low-pass on contiguous non-NaN runs
- Preserves CSV shape/orientation and column names
- Saves to: <stem>_Filtered<HzTag>.csv, auto-incrementing _v2, _v3, ... if needed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def coerce_to_frame_major_16(data: np.ndarray) -> np.ndarray:
    """Return array shaped (n_frames, 16)."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}.")

    rows, cols = data.shape
    if cols == 16:
        return data
    if rows == 16:
        return data.T

    raise ValueError(
        f"CSV must have 16 columns (frames x 16) or 16 rows (16 x frames). Got {data.shape}."
    )


def lowpass_one_channel(y: np.ndarray, cutoff_hz: float, sample_rate_hz: float) -> np.ndarray:
    """FFT low-pass on contiguous non-NaN runs while preserving NaN positions."""
    if y.ndim != 1:
        raise ValueError("Expected 1D channel.")

    finite_mask = np.isfinite(y)
    if not np.any(finite_mask):
        return y.copy()

    out = y.copy()
    nyquist = sample_rate_hz / 2.0
    effective_cutoff_hz = min(cutoff_hz, nyquist)

    valid_idx = np.flatnonzero(finite_mask)
    split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
    runs = np.split(valid_idx, split_points)

    for run in runs:
        if run.size == 0:
            continue

        start = int(run[0])
        end = int(run[-1]) + 1
        segment = y[start:end]

        if segment.size < 3:
            out[start:end] = segment
            continue

        seg_fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate_hz)
        seg_fft[freqs > effective_cutoff_hz] = 0.0
        out[start:end] = np.fft.irfft(seg_fft, n=segment.size)

    return out


def filtered_all_channels(data: np.ndarray, cutoff_hz: float, sample_rate_hz: float) -> np.ndarray:
    out = np.empty_like(data)
    for idx in range(data.shape[1]):
        out[:, idx] = lowpass_one_channel(data[:, idx], cutoff_hz, sample_rate_hz)
    return out


def build_output_dataframe(
    filtered_frame_major: np.ndarray,
    source_numeric_df: pd.DataFrame,
    input_is_frames_by_16: bool,
) -> pd.DataFrame:
    if input_is_frames_by_16:
        out_arr = filtered_frame_major
    else:
        out_arr = filtered_frame_major.T

    if out_arr.shape != source_numeric_df.shape:
        raise ValueError(
            "Filtered output shape does not match source CSV shape: "
            f"{out_arr.shape} vs {source_numeric_df.shape}."
        )

    return pd.DataFrame(out_arr, columns=source_numeric_df.columns)


def format_hz_tag(hz: float) -> str:
    return f"{hz:g}".replace(".", "p")


def next_output_path(input_csv: Path, effective_cutoff_hz: float) -> Path:
    hz_tag = format_hz_tag(effective_cutoff_hz)
    base = input_csv.with_name(f"{input_csv.stem}_Filtered{hz_tag}{input_csv.suffix}")
    out_path = base
    counter = 2
    while out_path.exists():
        out_path = input_csv.with_name(
            f"{input_csv.stem}_Filtered{hz_tag}_v{counter}{input_csv.suffix}"
        )
        counter += 1
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Low-pass denoise a rigid body transform CSV and save it with the GUI-style filename."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to rigid body transform CSV")
    parser.add_argument("cutoff_hz", type=float, help="Low-pass cutoff in Hz")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=750.0,
        help="Sample rate in Hz (default: 750)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv: Path = args.input_csv
    cutoff_hz: float = args.cutoff_hz
    sample_rate_hz: float = args.sample_rate

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    if cutoff_hz <= 0:
        raise ValueError("Cutoff must be > 0 Hz.")
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be > 0 Hz.")

    df = pd.read_csv(input_csv)
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    data = df_numeric.to_numpy(dtype=float)

    if data.shape[1] == 16:
        input_is_frames_by_16 = True
    elif data.shape[0] == 16:
        input_is_frames_by_16 = False
    else:
        raise ValueError(f"CSV must be (frames x 16) or (16 x frames). Got {data.shape}.")

    data_frame_major = coerce_to_frame_major_16(data)
    filtered = filtered_all_channels(data_frame_major, cutoff_hz, sample_rate_hz)

    out_df = build_output_dataframe(filtered, df_numeric, input_is_frames_by_16)
    effective_cutoff_hz = min(cutoff_hz, sample_rate_hz / 2.0)
    out_path = next_output_path(input_csv, effective_cutoff_hz)

    out_df.to_csv(out_path, index=False, na_rep="NaN")

    print(f"Input: {input_csv}")
    print(f"Cutoff requested: {cutoff_hz:g} Hz")
    print(f"Sample rate: {sample_rate_hz:g} Hz")
    if effective_cutoff_hz != cutoff_hz:
        print(
            f"Cutoff clipped to Nyquist: {effective_cutoff_hz:g} Hz "
            f"(Nyquist={sample_rate_hz/2.0:g} Hz)"
        )
    print(f"Saved filtered CSV: {out_path}")


if __name__ == "__main__":
    main()
