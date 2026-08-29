# Machine Learning Experimentation Overview

This overview maps the full XROMM-to-DeepLabCut-to-XROMM workflow in this project, starting with data conversion and trial splitting, then moving through training-set construction, model training, inference, 
and post-analysis correction. It identifies the scripts that enforce coordinate formatting and conversion into XMALab-compatible outputs, alongside notebooks used to test frame-selection and model-updating strategies. 
It also separates reusable pipeline utilities from experiment-specific analysis notebooks so you can quickly trace where operational code ends and where exploratory evaluation and figure building begin. Use this document 
as a navigation layer when onboarding, debugging save/output behavior, or deciding which files to run for preprocessing, training, prediction, evaluation, and final reporting.

## Core Post-Analysis Scripts

- [Testing/Code-Testing/PostAnalysis_review.py](Testing/Code-Testing/PostAnalysis_review.py): Main interactive post-analysis review workflow for prediction correction and update preparation.
- [Testing/Code-Testing/PostAnalysis_review-updating.py](Testing/Code-Testing/PostAnalysis_review-updating.py): Extended post-analysis workflow with active-update exports, correction autosave, model retraining hooks, and updated prediction generation, working on linux machine.

## test_updated_* Notebooks

These notebooks are focused on testing update strategies and training-update behavior.

- [Testing/Code-Testing/test_updated_experiments.ipynb](Testing/Code-Testing/test_updated_experiments.ipynb)
- [Testing/Code-Testing/test_updated_experiments(FrameSelectionForUpdate).ipynb](Testing/Code-Testing/test_updated_experiments(FrameSelectionForUpdate).ipynb)
- [Testing/Code-Testing/test_updated_experiments(FrameSelectionForUpdate)-ManyAmounts.ipynb](Testing/Code-Testing/test_updated_experiments(FrameSelectionForUpdate)-ManyAmounts.ipynb)
- [Testing/Code-Testing/test_updated_experiments(ModelUpdating).ipynb](Testing/Code-Testing/test_updated_experiments(ModelUpdating).ipynb)
- [Testing/Code-Testing/test_updated_experiments(ModelUpdating)30.ipynb](Testing/Code-Testing/test_updated_experiments(ModelUpdating)30.ipynb)
- [Testing/Code-Testing/test_updated_experiments(ModelUpdating)various.ipynb](Testing/Code-Testing/test_updated_experiments(ModelUpdating)various.ipynb)
- [Testing/Code-Testing/test_updated_experiments(subepochs).ipynb](Testing/Code-Testing/test_updated_experiments(subepochs).ipynb)

## xrommtools_copy

- [Testing/Code-Testing/xrommtools_copy.py](Testing/Code-Testing/xrommtools_copy.py): Local XROMM utility module for converting between DeepLabCut outputs and XMALab-style coordinate formats, plus related data utilities.
- [Testing/DeepLabCutExperiments/xrommtools_copy.py](Testing/DeepLabCutExperiments/xrommtools_copy.py): Experiment-side copy of the same toolset used directly in the DeepLabCutExperiments pipeline.

## DLC Support and Data Conversion Utilities

- [Testing/Code-Testing/DLCsupport.py](Testing/Code-Testing/DLCsupport.py): DeepLabCut helper library for project setup, bodypart consistency checks, training orchestration, and reproducible workflow utilities.
- [Testing/Code-Testing/data-converters.py](Testing/Code-Testing/data-converters.py): CLI and utility functions for image format conversion and TIFF/JPG stack to AVI conversion.

## All Python and Notebook Files in DeepLabCutExperiments

- [Testing/DeepLabCutExperiments/dataset_splitter.py](Testing/DeepLabCutExperiments/dataset_splitter.py): Deterministic train/test splitting utilities with split metadata and manifests.
- [Testing/DeepLabCutExperiments/Figures/FigureDevelopment.ipynb](Testing/DeepLabCutExperiments/Figures/FigureDevelopment.ipynb): Figure creation notebook for paper visuals, model comparisons, and experiment plots.
- [Testing/DeepLabCutExperiments/OutOfTheBox_Experimentation.ipynb](Testing/DeepLabCutExperiments/OutOfTheBox_Experimentation.ipynb): End-to-end out-of-the-box evaluation notebook for XROMM DeepLabCut benchmarking.
- [Testing/DeepLabCutExperiments/Predictions.py](Testing/DeepLabCutExperiments/Predictions.py): Prediction and evaluation utilities, including coordinate scoring and experiment-level metrics.
- [Testing/DeepLabCutExperiments/training_sets/dataset_builder.py](Testing/DeepLabCutExperiments/training_sets/dataset_builder.py): Training-set construction logic for random, displacement, and embedding-based frame selection.
- [Testing/DeepLabCutExperiments/training_sets/train_all_models.py](Testing/DeepLabCutExperiments/training_sets/train_all_models.py): Batch training runner for generated manifests and multi-job execution.
- [Testing/DeepLabCutExperiments/training_sets/__init__.py](Testing/DeepLabCutExperiments/training_sets/__init__.py): Package exports for training-set builder utilities.
- [Testing/DeepLabCutExperiments/xrommtools_copy.py](Testing/DeepLabCutExperiments/xrommtools_copy.py): XROMM/XMALab conversion helper used from the experiments folder.

## How These Pieces Connect

1. Data conversion and preprocessing are handled in [Testing/Code-Testing/data-converters.py](Testing/Code-Testing/data-converters.py) and [Testing/DeepLabCutExperiments/dataset_splitter.py](Testing/DeepLabCutExperiments/dataset_splitter.py).
2. Dataset construction and model training are orchestrated by [Testing/Code-Testing/DLCsupport.py](Testing/Code-Testing/DLCsupport.py), [Testing/DeepLabCutExperiments/training_sets/dataset_builder.py](Testing/DeepLabCutExperiments/training_sets/dataset_builder.py), and [Testing/DeepLabCutExperiments/training_sets/train_all_models.py](Testing/DeepLabCutExperiments/training_sets/train_all_models.py).
3. Prediction conversion to XMALab-compatible formats relies on [Testing/Code-Testing/xrommtools_copy.py](Testing/Code-Testing/xrommtools_copy.py) and [Testing/DeepLabCutExperiments/xrommtools_copy.py](Testing/DeepLabCutExperiments/xrommtools_copy.py).
4. Interactive correction and update flows are centered in [Testing/Code-Testing/PostAnalysis_review.py](Testing/Code-Testing/PostAnalysis_review.py) and [Testing/Code-Testing/PostAnalysis_review-updating.py](Testing/Code-Testing/PostAnalysis_review-updating.py).
5. Notebook-level analysis and figure generation are captured in [Testing/DeepLabCutExperiments/Figures/FigureDevelopment.ipynb](Testing/DeepLabCutExperiments/Figures/FigureDevelopment.ipynb), [Testing/DeepLabCutExperiments/OutOfTheBox_Experimentation.ipynb](Testing/DeepLabCutExperiments/OutOfTheBox_Experimentation.ipynb), and the [Testing/Code-Testing/test_updated_experiments.ipynb](Testing/Code-Testing/test_updated_experiments.ipynb) family.
