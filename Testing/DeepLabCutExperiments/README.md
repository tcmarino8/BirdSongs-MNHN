# DeepLabCut Experiments

Concise, reproducible workspace for out-of-the-box DeepLabCut evaluation and deterministic data splitting.

## What Is Here

- `OutOfTheBox_Experimentation.ipynb`: Main FAIR-style experiment notebook.
- `dataset_splitter.py`: Deterministic split utility for each `{Bird}_{Trial}` folder.
- `Data/`: Trial folders and generated split artifacts.
- `README_Questions.md`: Prompt log (numbered prompts).
- `requirements.txt`: Running dependency list for this experiment folder.

## Expected Trial Structure

Each trial folder in `Data/` should contain:

- `CompleteSet/Cam1`
- `CompleteSet/Cam2`
- `CompleteSet/labeledBodyPartsCoordinates.csv`

The splitter generates:

- `train/` and `test/` with mirrored structure
- `split_metadata.json` per trial
- `Data/split_manifest.csv` global summary

## Reproducible Setup

1. Open a terminal in `DeepLabCutExperiments`.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run deterministic split (`seed=42`, `test_frame_count=300`):

```powershell
python -c "from pathlib import Path; from dataset_splitter import split_all_trials; split_all_trials(Path('Data'), test_frame_count=300, seed=42, overwrite=True)"
```

4. Open and run `OutOfTheBox_Experimentation.ipynb` top-to-bottom.

## Training Set Construction (Pre-Training Stage)

After split generation, build three model-input training sets per trial from `train/`:

- `random_train`
- `displacement_train`
- `dino_train`

Implementation lives in `training_sets/dataset_builder.py` and mirrors `train_update_model` setup up to dataset creation (stops before model training):

- Bird/trial parsed from folder names like `DB_T15`, `Tulio_T10`
- Mapping: `DB -> DavidBowie`, `Tulio -> Tulio`
- Fixed training epoch plan recorded in metadata: `epochs=125`
- Frame-count variants built for every method: `nframes in [100, 50]`
- Displacement selection uses zone size `150`
- Random method metadata sets `update_set=Random`

Run all trials:

```powershell
python -c "from pathlib import Path; from training_sets.dataset_builder import build_three_training_sets_for_all_trials; df = build_three_training_sets_for_all_trials(Path('Data'), nframes_values=(100,50), epochs=125, frame_selection_seed=42, zone_size=150); print(df.to_string(index=False))"
```

Outputs include per-method folders in each trial and a global manifest at `Data/training_set_build_manifest.csv`.

If `deeplabcut` is not installed, the subset folders and metadata are still created and manifest rows are marked `partial`.
If `torch`/`torchvision` are not installed, DINO rows are marked `error` and an error metadata file is written in `dino_train/nframes_*`.

## Update Rules

- Add new packages to `requirements.txt` whenever notebook or scripts gain dependencies.
- Append new user prompts to `README_Questions.md` as the next numbered prompt.
- Number every prompt entry explicitly as `Prompt N:` and keep wording unchanged except numbering.
- Keep this README concise and focused on run steps and file navigation.
