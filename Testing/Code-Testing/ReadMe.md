# Bird DLC Analysis Toolkit

This repository contains a complete workflow for running DeepLabCut-based bird tracking, inference, and analysis using pretrained models.

The project is designed to be portable. As long as the project directory structure is preserved, all models and configuration files are located automatically using relative paths.

---

# Project Structure

```
BirdProject/
│
├── environment.yml          # Conda environment
├── README.md
│
├── Code/
│   ├── main.py
│   ├── DLCsupport.py
│   ├── xrommtools_copy.py
│   └── ...
│
├── Model_Zoo/
│   ├── DB_15_17_Trained/
│   ├── Tulio_10_05_Trained/
│   ├── Miguel_06_Trained/
│   └── Endive_42_Trained/
│
└── Data/
```

Do **not** rename or move the `Code`, `Model_Zoo`, or `Data` folders unless you also update the paths in the source code.

---

# Requirements

* Anaconda or Miniconda
* Python 3.11 (installed automatically by the environment)
* NVIDIA GPU (recommended for inference and training)

---

# Installation

Open an **Anaconda Prompt** (Windows) or terminal (Linux/macOS).

Navigate to the project directory:

```bash
cd path/to/BirdProject
```

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate birddlc
```

---

# Running the Software

Navigate to the `Code` directory if necessary:

```bash
cd Code
```

Run the main program:

```bash
python postanalysis_review.py
```

---

# Models

Pretrained DeepLabCut models are stored in the `Model_Zoo` directory.

The software automatically selects the correct model and configuration file for each bird using the mappings defined in `DLCsupport.py`.

If you train new models, update the model paths in `DLCsupport.py` accordingly.

---

# Data

Place videos, images, or other datasets inside the `Data` directory (or wherever your workflow expects them).

The repository is designed so that project-relative paths work regardless of where the project folder is located on your computer.

---

# Updating Models

If a new DeepLabCut model is trained:

1. Copy the trained model into `Model_Zoo`.
2. Update the corresponding entries in `DLCsupport.py`.
3. No other code changes should be necessary.

---

# Troubleshooting

### Conda environment not found

Make sure the environment has been created:

```bash
conda env list
```

If `birddlc` is missing, recreate it:

```bash
conda env create -f environment.yml
```

### Missing Python package

Activate the environment before running the software:

```bash
conda activate birddlc
```

### CUDA/GPU issues

If PyTorch cannot detect a GPU, verify that:

* NVIDIA drivers are installed.
* CUDA is compatible with the installed PyTorch version.
* `torch.cuda.is_available()` returns `True`.

The software can still run on CPU, although inference and training will be slower.

---

# License

This repository is intended for research and educational use. Please ensure that any datasets or pretrained models included with the project are distributed in accordance with their respective licenses.

---

# Contact

For questions, bug reports, or suggestions, please contact the project maintainer (Tyler Marino).
