---
name: plot-cam-overlays
description: 'Plot marker positions over synchronized Cam1/Cam2 image folders from a CSV (DLC or XMALab), with a slider to scrub frames. Use for alignment QA, missing-marker debugging, and frame-index offset checks.'
argument-hint: 'cam1 folder, cam2 folder, csv path, optional start offset'
user-invocable: true
---

# Plot Cam Overlays With Frame Scrubber

Build an interactive viewer that overlays marker points from a CSV on top of images from two camera folders.

Expected input format:
- Cam1 image folder
- Cam2 image folder
- CSV file in DLC or XMALab-style format

Use this skill when you need to verify image/point alignment, inspect offsets, or browse selected frames visually.

## Outcome
Create or update a Python notebook cell (or script) that:
1. Loads images from Cam1 and Cam2 folders.
2. Loads and normalizes marker columns from DLC or XMALab-style CSV.
3. Shows overlays for Cam1, Cam2, or both.
4. Provides slider-based frame scrubbing.
5. Provides marker-size control.
6. Prints diagnostics so invisible markers can be debugged quickly.

## Detection Rules
Use the following schema detection order.

1. Wide cam-suffix schema (preferred)
- Columns match:
  - `<marker>_cam1_X`, `<marker>_cam1_Y`
  - `<marker>_cam2_X`, `<marker>_cam2_Y`
- Common in XMALab exports and converted DLC tables.

2. DLC multi-row header schema
- Typical top rows: scorer/bodyparts/coords.
- Flatten to wide columns and convert into the same canonical suffix form above.

3. Fallback
- If only one camera exists in CSV, still render available camera and show explicit warning for missing camera columns.

## Procedure
1. Validate inputs
- Confirm cam1 folder exists and has images.
- Confirm cam2 folder exists and has images.
- Confirm CSV exists and is readable.
- Collect sorted image paths for each camera (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`).

2. Load and normalize CSV
- Read CSV into pandas.
- Detect schema using Detection Rules.
- Convert to canonical columns (`*_cam1_X/Y`, `*_cam2_X/Y`) where possible.
- Preserve original row order; this is the default frame index.

3. Build frame mapping
- Primary mapping: row index to sorted image list index.
- Optional `index_offset` for known shifts.
- For each frame `i`:
  - Cam1 image index: `i + index_offset`
  - Cam2 image index: `i + index_offset`
- Clamp out-of-range accesses and report when an image is missing.

4. Build interactive UI
- Add controls:
  - `view`: `cam1`, `cam2`, `both`
  - `frame_slider`: 0..N-1
  - `marker_size_slider`
  - optional `offset_slider`
- Render selected frame(s) with matplotlib scatter overlays.
- Invert no axes unless user explicitly requests it.

5. Overlay logic
- For each camera, find all valid `(X, Y)` pairs in canonical columns.
- Convert values with `pd.to_numeric(..., errors="coerce")`.
- Plot only finite points.
- Use high-contrast defaults (for example `lime` with black edge).

6. Diagnostics (required)
Print diagnostics every render:
- current frame index and mapped image indices
- image path used for each camera
- number of marker pairs detected per camera
- number of finite points plotted per camera
- CSV schema detected
- any missing image/column warnings

## Decision Points
- If markers are not visible:
  - Increase marker size.
  - Print finite point count and coordinate ranges.
  - Verify image bounds and whether points lie outside bounds.
- If points are consistently shifted:
  - Adjust `index_offset` and re-check 3-5 known frames.
- If only one camera overlays correctly:
  - Re-check camera-specific column detection and folder contents.

## Completion Checks
A run is complete only when all checks pass:
1. Slider updates image and points without exceptions.
2. Cam1 overlays show non-zero finite points on at least one frame.
3. Cam2 overlays show non-zero finite points on at least one frame.
4. Both-view mode renders two panels correctly.
5. Diagnostics confirm expected marker-pair counts and valid image paths.
6. User can scrub frames and visually confirm alignment quality.

## Suggested Prompt Invocations
- `/plot-cam-overlays cam1 folder=<path> cam2 folder=<path> csv=<path> format=auto`
- `/plot-cam-overlays use existing displacement csv and show both cameras with marker size slider`
- `/plot-cam-overlays debug why markers are invisible and print finite-point diagnostics`

## Notes
- Do not recompute sampling/displacement inside this workflow unless explicitly requested.
- Prefer reviewing existing saved outputs when present.
