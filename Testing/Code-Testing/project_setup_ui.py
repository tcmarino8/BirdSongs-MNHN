"""
Simple project setup UI for DeepLabCut combined-project bootstrapping.

What it does:
1) Ask for Task, Experimenter, and combined project root name.
2) Ask for the parent folder where the combined project root should be created.
3) Ask for the folder that contains dummy video(s) and let the user pick one .avi.
4) Create/reuse a DLC project via DLCsupport.create_combined_project_if_missing.
5) Build and display a starter dictionary with key paths.
6) Optionally save the dictionary as JSON.

Run:
    python project_setup_ui.py
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:  # pragma: no cover - optional dependency
    FigureCanvasTkAgg = None

from DLCsupport import (
    BIRD_BODYPARTS,
    build_combined_dataset,
    create_combined_project_if_missing,
    ensure_config_scorer_matches_data,
    find_latest_snapshot,
    set_net_type,
)

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageTk = None


class ProjectSetupUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DLC Project Setup (Starter)")
        self.root.geometry("980x700")
        self.active_mode = "home"

        self.project_parent_dir = tk.StringVar()
        self.dummy_video_dir = tk.StringVar()

        self.task_var = tk.StringVar(value="Canari")
        self.experimenter_var = tk.StringVar(value="Tyler")
        self.project_root_name_var = tk.StringVar(value="Canari_combined_training")
        self.selected_dummy_video_var = tk.StringVar()
        self.existing_config_var = tk.StringVar()

        self.current_project_dict: dict | None = None
        self.current_config_path: Path | None = None

        self.step2_frame: ttk.LabelFrame | None = None
        self.bird_suggestion_var = tk.StringVar()
        self.bodyparts_text: tk.Text | None = None
        self.config_preview_text: tk.Text | None = None

        self.step3_frame: ttk.LabelFrame | None = None
        self.dataset_data_path_var = tk.StringVar()
        self.dataset_name_var = tk.StringVar(value="combined_train")
        self.frame_counts_var = tk.StringVar(value="200,400,800,1400")
        self.dataset_seed_var = tk.StringVar(value="42")
        self.dataset_summary_text: tk.Text | None = None
        self.xy_long_text: tk.Text | None = None
        self.image_preview_label: ttk.Label | None = None
        self.image_info_label: ttk.Label | None = None
        self._preview_photo = None

        self.step4_frame: ttk.LabelFrame | None = None
        self.train_mode_var = tk.StringVar(value="testtrain")
        self.log_interval_var = tk.StringVar(value="10")
        self.train_status_var = tk.StringVar(value="Idle")
        self.train_progress: ttk.Progressbar | None = None
        self.train_logs_text: tk.Text | None = None
        self._training_in_progress = False
        self._training_start_time = 0.0

        self.step5_frame: ttk.LabelFrame | None = None
        self.eval_config_var = tk.StringVar()
        self.eval_model_var = tk.StringVar()
        self.eval_snapshot_path_var = tk.StringVar(value="No snapshot selected")
        self.eval_model_combo: ttk.Combobox | None = None
        self.eval_snapshot_text: tk.Text | None = None
        self.eval_config_text: tk.Text | None = None
        self.eval_layers_text: tk.Text | None = None
        self._eval_snapshot_map: dict[str, Path] = {}

        self.step6_frame: ttk.LabelFrame | None = None
        self.eval_video_folder_var = tk.StringVar()
        self.eval_existing_data_folder_var = tk.StringVar()
        self.eval_truth_csv_var = tk.StringVar()
        self.eval_status_var = tk.StringVar(value="Idle")
        self.eval_plot_type_var = tk.StringVar(value="likelihood_over_time")
        self.eval_color_mode_var = tk.StringVar(value="likelihood")
        self.eval_bodypart_filter_var = tk.StringVar(value="all")
        self.eval_camera_filter_var = tk.StringVar(value="all")
        self.eval_min_likelihood_var = tk.StringVar(value="0.6")
        self.eval_bodypart_combo: ttk.Combobox | None = None
        self.eval_camera_combo: ttk.Combobox | None = None
        self.eval_logs_text: tk.Text | None = None
        self.eval_plot_host: ttk.Frame | None = None
        self._eval_canvas = None
        self._eval_loaded_data: pd.DataFrame | None = None
        self._eval_inference_in_progress = False

        self._build_start_page()

    def _clear_root_children(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()

    def _build_start_page(self) -> None:
        self.active_mode = "home"
        self._clear_root_children()

        frame = ttk.Frame(self.root, padding=24)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="DLC Workflow Launcher",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="center", pady=(50, 10))

        ttk.Label(
            frame,
            text="Choose a workflow to continue.",
            font=("Segoe UI", 11),
        ).pack(anchor="center", pady=(0, 24))

        buttons = ttk.Frame(frame)
        buttons.pack(anchor="center")
        ttk.Button(buttons, text="Train model", command=self._open_train_mode).pack(side="left", padx=10)
        ttk.Button(buttons, text="Test model", command=self._open_test_mode).pack(side="left", padx=10)

    def _open_train_mode(self) -> None:
        self.active_mode = "train"
        self._build_layout()

    def _open_test_mode(self) -> None:
        self.active_mode = "test"
        self._build_test_layout()

    def _build_test_layout(self) -> None:
        self._clear_root_children()

        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.page = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=self.page, anchor="nw")

        def _on_page_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        self.page.bind("<Configure>", _on_page_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        title = ttk.Label(
            self.page,
            text="DLC Setup Wizard (Testing)",
            font=("Segoe UI", 14, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 12))
        ttk.Button(self.page, text="Back", command=self._build_start_page).grid(row=0, column=1, sticky="e", pady=(0, 12))

        self.page.columnconfigure(0, weight=1)
        self._ensure_step5_section()
        self._ensure_step6_section()

    def _build_layout(self) -> None:
        self._clear_root_children()
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.page = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=self.page, anchor="nw")

        def _on_page_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        self.page.bind("<Configure>", _on_page_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        title = ttk.Label(
            self.page,
            text="DLC Setup Wizard (Training)",
            font=("Segoe UI", 14, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 12))
        ttk.Button(self.page, text="Back", command=self._build_start_page).grid(row=0, column=1, sticky="e", pady=(0, 12))

        step1 = ttk.LabelFrame(self.page, text="Step 1: Create project + starter dictionary", padding=10)
        step1.grid(row=1, column=0, sticky="nsew")
        step1.columnconfigure(1, weight=1)
        step1.columnconfigure(3, weight=1)

        ttk.Label(step1, text="Task").grid(row=0, column=0, sticky="w")
        ttk.Entry(step1, textvariable=self.task_var, width=35).grid(row=0, column=1, sticky="we", padx=(8, 16), pady=4)

        ttk.Label(step1, text="Experimenter").grid(row=0, column=2, sticky="w")
        ttk.Entry(step1, textvariable=self.experimenter_var, width=35).grid(row=0, column=3, sticky="we", padx=(8, 0), pady=4)

        ttk.Label(step1, text="Combined Project Root Name").grid(row=1, column=0, sticky="w")
        ttk.Entry(step1, textvariable=self.project_root_name_var, width=35).grid(row=1, column=1, sticky="we", padx=(8, 16), pady=4)

        ttk.Label(step1, text="Project Parent Folder").grid(row=2, column=0, sticky="w")
        ttk.Entry(step1, textvariable=self.project_parent_dir, width=70).grid(
            row=2, column=1, columnspan=2, sticky="we", padx=(8, 8), pady=4
        )
        ttk.Button(step1, text="Browse...", command=self._pick_project_parent).grid(row=2, column=3, sticky="e", pady=4)

        ttk.Label(step1, text="Dummy Video Folder").grid(row=3, column=0, sticky="w")
        ttk.Entry(step1, textvariable=self.dummy_video_dir, width=70).grid(
            row=3, column=1, columnspan=2, sticky="we", padx=(8, 8), pady=4
        )
        ttk.Button(step1, text="Browse...", command=self._pick_dummy_video_folder).grid(row=3, column=3, sticky="e", pady=4)

        ttk.Label(step1, text="Dummy Video (.avi)").grid(row=4, column=0, sticky="w")
        self.video_combo = ttk.Combobox(step1, textvariable=self.selected_dummy_video_var, state="readonly", width=90)
        self.video_combo.grid(row=4, column=1, columnspan=3, sticky="we", padx=(8, 0), pady=4)

        ttk.Separator(step1, orient="horizontal").grid(row=5, column=0, columnspan=4, sticky="we", pady=(10, 8))
        ttk.Label(step1, text="Already created project? (config.yaml)").grid(row=6, column=0, sticky="w")
        ttk.Entry(step1, textvariable=self.existing_config_var, width=70).grid(
            row=6, column=1, columnspan=2, sticky="we", padx=(8, 8), pady=4
        )
        existing_row = ttk.Frame(step1)
        existing_row.grid(row=6, column=3, sticky="e")
        ttk.Button(existing_row, text="Browse...", command=self._pick_existing_config).pack(side="left")
        ttk.Button(existing_row, text="Load", command=self._load_existing_project).pack(side="left", padx=(8, 0))

        action_row = ttk.Frame(step1)
        action_row.grid(row=7, column=0, columnspan=4, sticky="we", pady=(10, 8))

        ttk.Button(action_row, text="Create Project + Build Dictionary", command=self._create_project_and_build_dict).pack(side="left")
        ttk.Button(action_row, text="Save Dictionary As JSON", command=self._save_dict_json).pack(side="left", padx=10)
        ttk.Button(action_row, text="Clear Output", command=self._clear_output).pack(side="left")

        ttk.Label(step1, text="Dictionary Output", font=("Segoe UI", 11, "bold")).grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(8, 4)
        )

        self.output = tk.Text(step1, wrap="word", height=16)
        self.output.grid(row=9, column=0, columnspan=4, sticky="nsew")
        self.output.tag_configure("dict_key", font=("Segoe UI", 9, "bold"))

        out_scroll = ttk.Scrollbar(step1, orient="vertical", command=self.output.yview)
        out_scroll.grid(row=9, column=4, sticky="ns")
        self.output.configure(yscrollcommand=out_scroll.set)

        step1.rowconfigure(9, weight=1)
        self.page.columnconfigure(0, weight=1)

    def _pick_existing_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select existing DLC config.yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            self.existing_config_var.set(selected)

    def _build_starter_from_config(
        self,
        config_path: Path,
        *,
        task: str,
        experimenter: str,
        combined_project_root: Path,
        dummy_video: str,
    ) -> dict:
        project_dir = config_path.parent
        return {
            "project": {
                "task": task,
                "experimenter": experimenter,
                "combined_project_root": str(combined_project_root),
                "config_path": str(config_path),
                "project_dir": str(project_dir),
                "dummy_video": str(dummy_video),
                "dummy_video_folder": str(Path(dummy_video).parent) if dummy_video else "",
            },
            "paths": {
                "videos_path": str(project_dir / "videos"),
                "labeled_data_path": str(project_dir / "labeled-data"),
                "training_datasets_path": str(project_dir / "training-datasets"),
                "dlc_models_path": str(project_dir / "dlc-models-pytorch"),
            },
            "next_fields_to_fill": {
                "bird_name": "",
                "bodyparts": [],
                "data_path": "",
                "test_videos": [],
                "cam1_config": "",
                "cam2_config": "",
            },
        }

    def _load_existing_project(self) -> None:
        config_path = Path(self.existing_config_var.get().strip())
        if not config_path.exists() or config_path.name.lower() not in {"config.yaml", "config.yml"}:
            messagebox.showerror("Invalid config", "Please choose an existing project config.yaml file.")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}

            project_dir = config_path.parent
            combined_root = project_dir.parent

            task = str(cfg.get("Task", self.task_var.get())).strip()
            dummy_video = ""
            video_sets = cfg.get("video_sets", {})
            if isinstance(video_sets, dict) and video_sets:
                dummy_video = str(next(iter(video_sets.keys())))

            # Try parsing experimenter from DLC project folder name: Task-Experimenter-date
            experimenter = self.experimenter_var.get().strip()
            parts = project_dir.name.split("-")
            if len(parts) >= 2:
                task = parts[0] or task
                experimenter = parts[1] or experimenter

            self.task_var.set(task)
            self.experimenter_var.set(experimenter)
            self.project_root_name_var.set(combined_root.name)
            self.project_parent_dir.set(str(combined_root.parent))
            if dummy_video:
                self.selected_dummy_video_var.set(dummy_video)

            self.current_config_path = config_path
            starter = self._build_starter_from_config(
                config_path,
                task=task,
                experimenter=experimenter,
                combined_project_root=combined_root,
                dummy_video=dummy_video,
            )
            self.current_project_dict = starter
            self._write_output_with_bold_keys(starter)

            self._ensure_step2_section()
            self._load_step2_from_config()
            self._ensure_step3_section()
            self._ensure_step4_section()

            messagebox.showinfo("Loaded", "Existing project loaded. You can continue from Step 2.")
        except Exception as exc:
            details = traceback.format_exc()
            messagebox.showerror("Load failed", f"Could not load existing project:\n{exc}")
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, details)

    def _pick_project_parent(self) -> None:
        selected = filedialog.askdirectory(title="Select parent folder for combined project root")
        if selected:
            self.project_parent_dir.set(selected)

    def _pick_dummy_video_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select folder containing dummy .avi files")
        if not selected:
            return

        self.dummy_video_dir.set(selected)
        self._refresh_dummy_video_list()

    def _refresh_dummy_video_list(self) -> None:
        folder = Path(self.dummy_video_dir.get().strip())
        if not folder.exists():
            self.video_combo["values"] = []
            self.selected_dummy_video_var.set("")
            return

        videos = sorted([str(p) for p in folder.glob("*.avi")])
        self.video_combo["values"] = videos

        if videos:
            self.selected_dummy_video_var.set(videos[0])
        else:
            self.selected_dummy_video_var.set("")
            messagebox.showwarning(
                "No AVI files found",
                "No .avi files were found in this folder.\nPick another folder or add a dummy .avi file.",
            )

    def _validate_inputs(self) -> tuple[str, str, Path, Path, Path] | None:
        task = self.task_var.get().strip()
        experimenter = self.experimenter_var.get().strip()
        root_name = self.project_root_name_var.get().strip()

        parent = Path(self.project_parent_dir.get().strip())
        dummy_video = Path(self.selected_dummy_video_var.get().strip())

        if not task:
            messagebox.showerror("Missing value", "Please enter Task.")
            return None
        if not experimenter:
            messagebox.showerror("Missing value", "Please enter Experimenter.")
            return None
        if not root_name:
            messagebox.showerror("Missing value", "Please enter Combined Project Root Name.")
            return None
        if not parent.exists():
            messagebox.showerror("Invalid folder", "Project Parent Folder does not exist.")
            return None
        if not dummy_video.exists():
            messagebox.showerror("Invalid dummy video", "Please select a valid .avi dummy video.")
            return None

        combined_project_root = parent / root_name
        return task, experimenter, combined_project_root, dummy_video, parent

    def _create_project_and_build_dict(self) -> None:
        validated = self._validate_inputs()
        if validated is None:
            return

        task, experimenter, combined_project_root, dummy_video, _parent = validated

        try:
            config_path = create_combined_project_if_missing(
                task=task,
                experimenter=experimenter,
                combined_project_root=combined_project_root,
                dummy_video=dummy_video
            )
            self.current_config_path = Path(config_path)
            starter = self._build_starter_from_config(
                Path(config_path),
                task=task,
                experimenter=experimenter,
                combined_project_root=combined_project_root,
                dummy_video=str(dummy_video),
            )

            self.current_project_dict = starter
            self._write_output_with_bold_keys(starter)
            self._ensure_step2_section()
            self._load_step2_from_config()
            self._ensure_step3_section()
            self._ensure_step4_section()
            messagebox.showinfo("Success", "Project initialized and starter dictionary created.")

        except Exception as exc:
            details = traceback.format_exc()
            messagebox.showerror("Failed", f"Failed to create project/dictionary:\n{exc}")
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, details)

    def _write_output(self, data: dict) -> None:
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, json.dumps(data, indent=2))

    def _write_output_with_bold_keys(self, data: dict) -> None:
        self.output.delete("1.0", tk.END)
        text = json.dumps(data, indent=2)
        self.output.insert(tk.END, text)

        idx = "1.0"
        while True:
            key_start = self.output.search('"', idx, stopindex="end")
            if not key_start:
                break
            key_end = self.output.search('"', f"{key_start}+1c", stopindex="end")
            if not key_end:
                break
            after_key = self.output.get(f"{key_end}+1c", f"{key_end}+2c")
            if after_key == ":":
                self.output.tag_add("dict_key", key_start, f"{key_end}+1c")
            idx = f"{key_end}+1c"

    def _save_dict_json(self) -> None:
        if not self.current_project_dict:
            messagebox.showwarning("Nothing to save", "Create a project dictionary first.")
            return

        target = filedialog.asksaveasfilename(
            title="Save starter dictionary as JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not target:
            return

        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.current_project_dict, fh, indent=2)

        messagebox.showinfo("Saved", f"Dictionary saved to:\n{target}")

    def _clear_output(self) -> None:
        self.output.delete("1.0", tk.END)

    def _ensure_step2_section(self) -> None:
        if self.step2_frame is not None:
            return

        self.step2_frame = ttk.LabelFrame(
            self.page,
            text='Step 2: Relabel body parts and view config',
            padding=10,
        )
        self.step2_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        self.step2_frame.columnconfigure(0, weight=1)
        self.step2_frame.columnconfigure(1, weight=1)

        helper = ttk.Label(
            self.step2_frame,
            text=(
                "Paste bodyparts as comma-separated, dash-separated, or one per line, then save to config.yaml."
            ),
        )
        helper.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        left = ttk.Frame(self.step2_frame)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        suggestion_row = ttk.Frame(left)
        suggestion_row.grid(row=0, column=0, sticky="we", pady=(0, 6))
        suggestion_row.columnconfigure(1, weight=1)

        ttk.Label(suggestion_row, text="Suggested Bird").grid(row=0, column=0, sticky="w")
        bird_options = sorted(BIRD_BODYPARTS.keys())
        bird_combo = ttk.Combobox(
            suggestion_row,
            textvariable=self.bird_suggestion_var,
            values=bird_options,
            state="readonly",
            width=24,
        )
        bird_combo.grid(row=0, column=1, sticky="we", padx=(8, 0))
        bird_combo.bind("<<ComboboxSelected>>", self._on_bird_suggestion_selected)

        ttk.Label(left, text="Editable Bodyparts (one per line)", font=("Segoe UI", 10, "bold")).grid(
            row=1, column=0, sticky="w"
        )

        self.bodyparts_text = tk.Text(left, wrap="none", height=16)
        self.bodyparts_text.grid(row=2, column=0, sticky="nsew", pady=(4, 6))
        left.rowconfigure(2, weight=1)

        btn_row = ttk.Frame(left)
        btn_row.grid(row=3, column=0, sticky="w")
        ttk.Button(btn_row, text="Reload Bodyparts From Config", command=self._load_step2_from_config).pack(side="left")
        ttk.Button(btn_row, text="Save Bodyparts To Config", command=self._save_bodyparts_to_config).pack(side="left", padx=8)

        if bird_options:
            self.bird_suggestion_var.set(bird_options[0])
            self._apply_bird_bodypart_suggestion(bird_options[0])

        right = ttk.Frame(self.step2_frame)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Config Preview (config.yaml)", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.config_preview_text = tk.Text(right, wrap="none", height=16)
        self.config_preview_text.grid(row=1, column=0, sticky="nsew", pady=(4, 6))

        preview_btn_row = ttk.Frame(right)
        preview_btn_row.grid(row=2, column=0, sticky="w")
        ttk.Button(preview_btn_row, text="Refresh Config Preview", command=self._refresh_config_preview).pack(side="left")

    def _ensure_step3_section(self) -> None:
        if self.step3_frame is not None:
            return

        self.step3_frame = ttk.LabelFrame(
            self.page,
            text="Step 3: Create training datasets",
            padding=10,
        )
        self.step3_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        self.step3_frame.columnconfigure(0, weight=1)
        self.step3_frame.columnconfigure(1, weight=1)

        helper = ttk.Label(
            self.step3_frame,
            text=(
                "Mimics notebook Step 6 for one project: build dataset(s), count frames, "
                "show first training image, and display first-row XY labels in long format."
            ),
        )
        helper.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        controls = ttk.Frame(self.step3_frame)
        controls.grid(row=1, column=0, columnspan=2, sticky="we")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Data Path (/trainingdata)").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.dataset_data_path_var).grid(row=0, column=1, sticky="we", padx=(8, 8), pady=3)
        ttk.Button(controls, text="Browse...", command=self._pick_dataset_data_path).grid(row=0, column=2, sticky="e", pady=3)

        ttk.Label(controls, text="Dataset Name (Typically Canari)").grid(row=1, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.dataset_name_var).grid(row=1, column=1, sticky="we", padx=(8, 8), pady=3)

        ttk.Label(controls, text="Frame Counts").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.frame_counts_var).grid(row=2, column=1, sticky="we", padx=(8, 8), pady=3)

        ttk.Label(controls, text="Selection Seed").grid(row=3, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.dataset_seed_var).grid(row=3, column=1, sticky="we", padx=(8, 8), pady=3)

        action = ttk.Frame(self.step3_frame)
        action.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 8))
        ttk.Button(action, text="Create Training Dataset(s)", command=self._create_training_datasets).pack(side="left")

        left = ttk.Frame(self.step3_frame)
        left.grid(row=3, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Build Summary", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.dataset_summary_text = tk.Text(left, wrap="word", height=16)
        self.dataset_summary_text.grid(row=1, column=0, sticky="nsew", pady=(4, 8))

        ttk.Label(left, text="First Row XY (Long Format)", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
        self.xy_long_text = tk.Text(left, wrap="none", height=16)
        self.xy_long_text.grid(row=3, column=0, sticky="nsew", pady=(4, 0))
        left.rowconfigure(1, weight=1)
        left.rowconfigure(3, weight=1)

        right = ttk.Frame(self.step3_frame)
        right.grid(row=3, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="First Training Image", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.image_preview_label = ttk.Label(right, text="No preview yet", anchor="center")
        self.image_preview_label.grid(row=1, column=0, sticky="nsew", pady=(4, 6))
        self.image_info_label = ttk.Label(right, text="", wraplength=360, justify="left")
        self.image_info_label.grid(row=2, column=0, sticky="w")

    def _ensure_step4_section(self) -> None:
        if self.step4_frame is not None:
            return

        self.step4_frame = ttk.LabelFrame(
            self.page,
            text="Step 4: Train model (ResNet50)",
            padding=10,
        )
        self.step4_frame.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        self.step4_frame.columnconfigure(0, weight=1)

        helper = ttk.Label(
            self.step4_frame,
            text=(
                "Follows notebook Cell 7 pattern: set net type to resnet_50 and train with mode "
                "fulltrain|testtrain. Progress is logged at selected checkpoints."
            ),
        )
        helper.grid(row=0, column=0, sticky="w", pady=(0, 8))

        controls = ttk.Frame(self.step4_frame)
        controls.grid(row=1, column=0, sticky="we")
        controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="Train Mode").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.train_mode_var,
            values=["fulltrain", "testtrain"],
            state="readonly",
            width=12,
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=(8, 16))

        ttk.Label(controls, text="Log Interval (epochs)").grid(row=0, column=2, sticky="w")
        interval_combo = ttk.Combobox(
            controls,
            textvariable=self.log_interval_var,
            values=["10"],
            state="readonly",
            width=10,
        )
        interval_combo.grid(row=0, column=3, sticky="w", padx=(8, 16))

        ttk.Button(controls, text="Train Model", command=self._start_training).grid(row=0, column=4, sticky="w")

        status_row = ttk.Frame(self.step4_frame)
        status_row.grid(row=2, column=0, sticky="we", pady=(8, 4))
        status_row.columnconfigure(1, weight=1)
        ttk.Label(status_row, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_row, textvariable=self.train_status_var).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.train_progress = ttk.Progressbar(self.step4_frame, orient="horizontal", mode="determinate")
        self.train_progress.grid(row=3, column=0, sticky="we", pady=(0, 8))

        ttk.Label(self.step4_frame, text="Training Logs", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, sticky="w")
        self.train_logs_text = tk.Text(self.step4_frame, wrap="word", height=12)
        self.train_logs_text.grid(row=5, column=0, sticky="nsew", pady=(4, 0))

    def _ensure_step5_section(self) -> None:
        if self.step5_frame is not None:
            return

        self.step5_frame = ttk.LabelFrame(
            self.page,
            text="Step 5: Model evaluation / inference (setup)",
            padding=10,
        )
        self.step5_frame.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        self.step5_frame.columnconfigure(0, weight=1)
        self.step5_frame.columnconfigure(1, weight=1)

        helper = ttk.Label(
            self.step5_frame,
            text=(
                "Iterative setup: choose a model config, select a discovered snapshot, "
                "and review snapshot + config side-by-side before adding image/AVI inference."
            ),
        )
        helper.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        controls = ttk.Frame(self.step5_frame)
        controls.grid(row=1, column=0, columnspan=2, sticky="we")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Model Config (config.yaml)").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.eval_config_var).grid(row=0, column=1, sticky="we", padx=(8, 8), pady=3)
        btns = ttk.Frame(controls)
        btns.grid(row=0, column=2, sticky="e")
        ttk.Button(btns, text="Browse...", command=self._pick_eval_config).pack(side="left")
        ttk.Button(btns, text="Load Models", command=self._load_eval_models).pack(side="left", padx=(8, 0))

        ttk.Label(controls, text="Model Snapshot").grid(row=1, column=0, sticky="w")
        self.eval_model_combo = ttk.Combobox(
            controls,
            textvariable=self.eval_model_var,
            state="readonly",
            width=90,
        )
        self.eval_model_combo.grid(row=1, column=1, columnspan=2, sticky="we", padx=(8, 0), pady=3)
        self.eval_model_combo.bind("<<ComboboxSelected>>", self._on_eval_model_selected)

        left = ttk.Frame(self.step5_frame)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Selected Snapshot Path", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.eval_snapshot_text = tk.Text(left, wrap="word", height=12)
        self.eval_snapshot_text.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        right = ttk.Frame(self.step5_frame)
        right.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Config Preview", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.eval_config_text = tk.Text(right, wrap="none", height=12)
        self.eval_config_text.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        layers = ttk.Frame(self.step5_frame)
        layers.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        layers.columnconfigure(0, weight=1)
        layers.rowconfigure(1, weight=1)

        ttk.Label(layers, text="Checkpoint Layer Table", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.eval_layers_text = tk.Text(layers, wrap="none", height=16)
        self.eval_layers_text.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        if self.current_config_path is not None:
            self.eval_config_var.set(str(self.current_config_path))
            self._load_eval_models()

    def _pick_eval_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select model config.yaml for evaluation/inference",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            self.eval_config_var.set(selected)

    def _load_eval_models(self) -> None:
        config_path = Path(self.eval_config_var.get().strip())
        if not config_path.exists() or config_path.name.lower() not in {"config.yaml", "config.yml"}:
            messagebox.showerror("Invalid config", "Select a valid config.yaml file first.")
            return

        self._eval_snapshot_map = self._discover_snapshot_options(config_path)
        labels = list(self._eval_snapshot_map.keys())

        if self.eval_model_combo is not None:
            self.eval_model_combo["values"] = labels

        if labels:
            self.eval_model_var.set(labels[-1])
            self._set_selected_snapshot_text(labels[-1])
        else:
            self.eval_model_var.set("")
            self.eval_snapshot_path_var.set("No snapshots found under dlc-models-pytorch.")
            self._set_selected_snapshot_text(None)
            self._set_checkpoint_layers_text(None)

        self._set_config_preview(config_path)

    def _discover_snapshot_options(self, config_path: Path) -> dict[str, Path]:
        project_dir = config_path.parent
        snapshot_paths = sorted(
            project_dir.glob("dlc-models-pytorch/*/*/train/snapshot-*.pt")
        )

        options: dict[str, Path] = {}
        for snap in snapshot_paths:
            train_set_name = snap.parent.parent.name
            label = f"{train_set_name} | {snap.name}"
            options[label] = snap
        return options

    def _on_eval_model_selected(self, _event: tk.Event) -> None:
        label = self.eval_model_var.get().strip()
        self._set_selected_snapshot_text(label)

    def _set_selected_snapshot_text(self, selected_label: str | None) -> None:
        if self.eval_snapshot_text is None:
            return

        self.eval_snapshot_text.delete("1.0", tk.END)

        if selected_label is None:
            self.eval_snapshot_text.insert(tk.END, "No snapshot selected.")
            self._set_checkpoint_layers_text(None)
            return

        snapshot_path = self._eval_snapshot_map.get(selected_label)
        if snapshot_path is None:
            self.eval_snapshot_text.insert(tk.END, "No snapshot selected.")
            self._set_checkpoint_layers_text(None)
            return

        self.eval_snapshot_path_var.set(str(snapshot_path))
        self.eval_snapshot_text.insert(tk.END, str(snapshot_path))
        self._set_checkpoint_layers_text(snapshot_path)

    def _set_checkpoint_layers_text(self, snapshot_path: Path | None) -> None:
        if self.eval_layers_text is None:
            return

        self.eval_layers_text.delete("1.0", tk.END)
        if snapshot_path is None:
            self.eval_layers_text.insert(tk.END, "No checkpoint selected.")
            return

        if not snapshot_path.exists():
            self.eval_layers_text.insert(tk.END, f"Checkpoint not found:\n{snapshot_path}")
            return

        try:
            import torch  # noqa: PLC0415
        except Exception as exc:
            self.eval_layers_text.insert(
                tk.END,
                "PyTorch is required to inspect checkpoint tensors.\n"
                f"Import error: {exc}",
            )
            return

        try:
            try:
                checkpoint_obj = torch.load(str(snapshot_path), map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint_obj = torch.load(str(snapshot_path), map_location="cpu")
            state_dict = self._extract_state_dict(checkpoint_obj)

            if not state_dict:
                self.eval_layers_text.insert(
                    tk.END,
                    "No tensors were extracted from this checkpoint with current rules.\n\n"
                    "Checkpoint structure preview:\n"
                    f"{self._describe_checkpoint_structure(checkpoint_obj)}",
                )
                return

            rows: list[tuple[int, str, str, int, str, str]] = []
            total_tensors = 0
            total_parameters = 0
            total_elements = 0

            for idx, (name, tensor) in enumerate(state_dict.items(), start=1):
                if not hasattr(tensor, "shape"):
                    continue

                shape = self._format_shape(getattr(tensor, "shape", ()))
                numel = int(getattr(tensor, "numel", lambda: 0)())
                dtype = str(getattr(tensor, "dtype", "unknown"))
                kind = self._tensor_kind_from_name(name)

                rows.append((idx, name, shape, numel, dtype, kind))
                total_tensors += 1
                total_elements += numel
                if kind == "param":
                    total_parameters += numel

            header = (
                f"Checkpoint: {snapshot_path}\n"
                f"Tensor entries: {total_tensors} | Parameter elements: {total_parameters:,} | Total elements: {total_elements:,}\n\n"
            )

            col_header = "#   key                                             shape                 numel        dtype           kind"
            self.eval_layers_text.insert(tk.END, header)
            self.eval_layers_text.insert(tk.END, col_header + "\n")
            self.eval_layers_text.insert(tk.END, "-" * len(col_header) + "\n")

            for idx, name, shape, numel, dtype, kind in rows:
                line = (
                    f"{idx:>3} "
                    f"{name[:46]:<46} "
                    f"{shape:<20} "
                    f"{numel:>10,} "
                    f"{dtype[:14]:<14} "
                    f"{kind}"
                )
                self.eval_layers_text.insert(tk.END, line + "\n")
        except Exception:
            self.eval_layers_text.insert(tk.END, traceback.format_exc())

    def _extract_state_dict(self, checkpoint_obj: object) -> dict[str, object]:
        preferred_keys = (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
        )

        if isinstance(checkpoint_obj, dict):
            for key in preferred_keys:
                candidate = checkpoint_obj.get(key)
                if candidate is None:
                    continue
                extracted = self._collect_tensor_entries(candidate, prefix=str(key))
                if extracted:
                    return extracted

        # Fallback: deep recursive extraction across the full object.
        return self._collect_tensor_entries(checkpoint_obj, prefix="")

    def _collect_tensor_entries(self, obj: object, prefix: str) -> dict[str, object]:
        entries: dict[str, object] = {}
        seen_ids: set[int] = set()

        def _is_tensor_like(value: object) -> bool:
            return hasattr(value, "shape") and callable(getattr(value, "numel", None))

        def _walk(value: object, key_prefix: str, depth: int) -> None:
            if value is None:
                return
            if depth > 8:
                return

            obj_id = id(value)
            if obj_id in seen_ids:
                return
            seen_ids.add(obj_id)

            if _is_tensor_like(value):
                key = key_prefix or "<root>"
                entries[key] = value
                return

            if isinstance(value, dict):
                for key, child in value.items():
                    child_key = str(key)
                    next_prefix = f"{key_prefix}.{child_key}" if key_prefix else child_key
                    _walk(child, next_prefix, depth + 1)
                return

            if isinstance(value, (list, tuple)):
                for idx, child in enumerate(value):
                    next_prefix = f"{key_prefix}[{idx}]" if key_prefix else f"[{idx}]"
                    _walk(child, next_prefix, depth + 1)
                return

            # Some checkpoints may store an nn.Module-like object directly.
            if hasattr(value, "state_dict") and callable(getattr(value, "state_dict", None)):
                try:
                    sd = value.state_dict()
                except Exception:
                    sd = None
                if sd is not None:
                    next_prefix = f"{key_prefix}.state_dict" if key_prefix else "state_dict"
                    _walk(sd, next_prefix, depth + 1)

        _walk(obj, prefix, 0)
        return entries

    def _describe_checkpoint_structure(self, checkpoint_obj: object) -> str:
        lines: list[str] = []

        def _typename(value: object) -> str:
            return type(value).__name__

        if isinstance(checkpoint_obj, dict):
            lines.append(f"Top-level type: dict ({len(checkpoint_obj)} keys)")
            for key in sorted(checkpoint_obj.keys(), key=lambda x: str(x))[:60]:
                value = checkpoint_obj[key]
                if isinstance(value, dict):
                    lines.append(f"- {key}: dict ({len(value)} keys)")
                elif isinstance(value, (list, tuple)):
                    lines.append(f"- {key}: {_typename(value)} (len={len(value)})")
                else:
                    lines.append(f"- {key}: {_typename(value)}")
            if len(checkpoint_obj) > 60:
                lines.append("- ...")
        else:
            lines.append(f"Top-level type: {_typename(checkpoint_obj)}")
            attrs = [name for name in ("state_dict", "model", "net", "module") if hasattr(checkpoint_obj, name)]
            if attrs:
                lines.append("Attributes present: " + ", ".join(attrs))

        return "\n".join(lines)

    def _format_shape(self, shape_obj: object) -> str:
        try:
            dims = [int(v) for v in shape_obj]
            return "[" + ", ".join(str(v) for v in dims) + "]"
        except Exception:
            return "[]"

    def _tensor_kind_from_name(self, tensor_name: str) -> str:
        lower = tensor_name.lower()
        if lower.endswith("running_mean") or lower.endswith("running_var") or lower.endswith("num_batches_tracked"):
            return "buffer"
        return "param"

    def _set_config_preview(self, config_path: Path) -> None:
        if self.eval_config_text is None:
            return
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.eval_config_text.delete("1.0", tk.END)
            self.eval_config_text.insert(tk.END, text)
        except Exception as exc:
            self.eval_config_text.delete("1.0", tk.END)
            self.eval_config_text.insert(tk.END, f"Failed to read config:\n{exc}")

    def _ensure_step6_section(self) -> None:
        if self.step6_frame is not None:
            return

        self.step6_frame = ttk.LabelFrame(
            self.page,
            text="Step 6: Run inference + interactive diagnostics",
            padding=10,
        )
        self.step6_frame.grid(row=6, column=0, sticky="nsew", pady=(12, 0))
        self.step6_frame.columnconfigure(0, weight=1)
        self.step6_frame.columnconfigure(1, weight=1)

        helper = ttk.Label(
            self.step6_frame,
            text=(
                "Run inference on a folder of videos using the selected model/snapshot, or load an existing output folder. "
                "Then filter plots by bodypart and minimum likelihood, and color by likelihood/bodypart/truth-distance."
            ),
        )
        helper.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        controls = ttk.Frame(self.step6_frame)
        controls.grid(row=1, column=0, columnspan=2, sticky="we")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Videos Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.eval_video_folder_var).grid(row=0, column=1, sticky="we", padx=(8, 8), pady=3)
        ttk.Button(controls, text="Browse...", command=self._pick_eval_video_folder).grid(row=0, column=2, sticky="e", pady=3)

        ttk.Label(controls, text="Existing Data Folder").grid(row=1, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.eval_existing_data_folder_var).grid(row=1, column=1, sticky="we", padx=(8, 8), pady=3)
        ttk.Button(controls, text="Browse...", command=self._pick_eval_existing_data_folder).grid(row=1, column=2, sticky="e", pady=3)

        ttk.Label(controls, text="Truth CSV (optional)").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.eval_truth_csv_var).grid(row=2, column=1, sticky="we", padx=(8, 8), pady=3)
        ttk.Button(controls, text="Browse...", command=self._pick_eval_truth_csv).grid(row=2, column=2, sticky="e", pady=3)

        action_row = ttk.Frame(self.step6_frame)
        action_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 6))
        ttk.Button(action_row, text="Run Inference", command=self._start_step6_inference).pack(side="left")
        ttk.Button(action_row, text="Load Existing Data", command=self._load_step6_existing_data).pack(side="left", padx=(8, 0))

        status_row = ttk.Frame(self.step6_frame)
        status_row.grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 8))
        status_row.columnconfigure(1, weight=1)
        ttk.Label(status_row, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_row, textvariable=self.eval_status_var).grid(row=0, column=1, sticky="w", padx=(8, 0))

        plot_controls = ttk.Frame(self.step6_frame)
        plot_controls.grid(row=4, column=0, columnspan=2, sticky="we", pady=(0, 8))
        plot_controls.columnconfigure(7, weight=1)

        ttk.Label(plot_controls, text="Plot").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            plot_controls,
            textvariable=self.eval_plot_type_var,
            values=["likelihood_over_time", "xy_over_time", "distance_over_time", "displacement_over_time"],
            state="readonly",
            width=24,
        ).grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(plot_controls, text="Bodypart").grid(row=0, column=2, sticky="w")
        self.eval_bodypart_combo = ttk.Combobox(
            plot_controls,
            textvariable=self.eval_bodypart_filter_var,
            values=["all"],
            state="readonly",
            width=20,
        )
        self.eval_bodypart_combo.grid(row=0, column=3, sticky="w", padx=(6, 12))

        ttk.Label(plot_controls, text="Camera").grid(row=0, column=4, sticky="w")
        self.eval_camera_combo = ttk.Combobox(
            plot_controls,
            textvariable=self.eval_camera_filter_var,
            values=["all"],
            state="readonly",
            width=12,
        )
        self.eval_camera_combo.grid(row=0, column=5, sticky="w", padx=(6, 12))

        ttk.Label(plot_controls, text="Min likelihood").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(plot_controls, textvariable=self.eval_min_likelihood_var, width=10).grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(6, 0))

        ttk.Label(plot_controls, text="Color by").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Combobox(
            plot_controls,
            textvariable=self.eval_color_mode_var,
            values=["likelihood", "bodypart", "distance_from_truth"],
            state="readonly",
            width=20,
        ).grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(6, 0))

        ttk.Button(plot_controls, text="Refresh Plot", command=self._refresh_step6_plot).grid(row=1, column=4, sticky="w", pady=(6, 0))

        self.eval_plot_host = ttk.Frame(self.step6_frame)
        self.eval_plot_host.grid(row=5, column=0, sticky="nsew", padx=(0, 8))
        self.eval_plot_host.columnconfigure(0, weight=1)
        self.eval_plot_host.rowconfigure(0, weight=1)
        placeholder = ttk.Label(self.eval_plot_host, text="No plot yet. Run inference or load existing data, then click Refresh Plot.")
        placeholder.grid(row=0, column=0, sticky="nsew")

        logs = ttk.Frame(self.step6_frame)
        logs.grid(row=5, column=1, sticky="nsew", padx=(8, 0))
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(1, weight=1)

        ttk.Label(logs, text="Step 6 Logs", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.eval_logs_text = tk.Text(logs, wrap="word", height=20)
        self.eval_logs_text.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

    def _pick_eval_video_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select folder containing videos for inference")
        if selected:
            self.eval_video_folder_var.set(selected)

    def _pick_eval_existing_data_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select existing post-processed data folder")
        if selected:
            self.eval_existing_data_folder_var.set(selected)

    def _pick_eval_truth_csv(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select truth CSV (optional)",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.eval_truth_csv_var.set(selected)

    def _append_eval_log(self, message: str) -> None:
        if self.eval_logs_text is None:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.eval_logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.eval_logs_text.see(tk.END)

    def _start_step6_inference(self) -> None:
        if self._eval_inference_in_progress:
            messagebox.showwarning("Inference running", "Inference is already in progress.")
            return

        config_path = Path(self.eval_config_var.get().strip())
        if not config_path.exists():
            messagebox.showerror("Missing config", "Select a valid config.yaml in Step 5 first.")
            return

        selected_label = self.eval_model_var.get().strip()
        snapshot_path = self._eval_snapshot_map.get(selected_label)
        if snapshot_path is None:
            messagebox.showerror("Missing snapshot", "Select a model snapshot in Step 5 first.")
            return

        video_folder = Path(self.eval_video_folder_var.get().strip())
        if not video_folder.exists():
            messagebox.showerror("Missing videos folder", "Select a valid folder containing videos.")
            return

        video_exts = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}
        videos = sorted([p for p in video_folder.iterdir() if p.is_file() and p.suffix.lower() in video_exts])
        if not videos:
            messagebox.showerror("No videos", "No supported video files were found in the selected folder.")
            return

        model_slug = self._safe_name_for_folder(selected_label.split("|")[0].strip() or "model")
        snap_num = self._extract_snapshot_num(snapshot_path.name)
        out_folder = video_folder / f"post_processed_data_{model_slug}_{snap_num}"
        out_folder.mkdir(parents=True, exist_ok=True)
        self.eval_existing_data_folder_var.set(str(out_folder))

        self._eval_inference_in_progress = True
        self.eval_status_var.set("Running inference...")
        self._append_eval_log(f"Output folder: {out_folder}")
        self._append_eval_log(f"Videos found: {len(videos)}")
        self._append_eval_log(
            "Using selected snapshot metadata for output naming. "
            "Actual checkpoint routing depends on installed DeepLabCut API behavior."
        )

        thread = threading.Thread(
            target=self._run_step6_inference_worker,
            args=(config_path, videos, out_folder),
            daemon=True,
        )
        thread.start()

    def _run_step6_inference_worker(self, config_path: Path, videos: list[Path], out_folder: Path) -> None:
        try:
            import deeplabcut  # noqa: PLC0415

            config_str = str(config_path)
            video_strs = [str(v) for v in videos]

            self.root.after(0, lambda: self._append_eval_log("Running analyze_videos..."))
            deeplabcut.analyze_videos(
                config=config_str,
                videos=video_strs,
                destfolder=str(out_folder),
                videotype="",
                save_as_csv=True,
            )

            self.root.after(0, lambda: self._append_eval_log("Running filterpredictions..."))
            try:
                deeplabcut.filterpredictions(
                    config=config_str,
                    videos=video_strs,
                    destfolder=str(out_folder),
                    videotype="",
                    save_as_csv=True,
                )
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._append_eval_log(f"filterpredictions skipped: {e}"))

            self.root.after(0, lambda: self._append_eval_log("Creating labeled videos..."))
            deeplabcut.create_labeled_video(
                config=config_str,
                videos=video_strs,
                destfolder=str(out_folder),
                videotype="",
            )

            self.root.after(0, lambda: self._append_eval_log("Creating trajectory plots..."))
            deeplabcut.plot_trajectories(
                config=config_str,
                videos=video_strs,
                destfolder=str(out_folder),
                videotype="",
            )

            self.root.after(0, lambda: self._on_step6_inference_complete(out_folder))
        except Exception:
            details = traceback.format_exc()
            self.root.after(0, lambda d=details: self._on_step6_inference_failed(d))

    def _on_step6_inference_complete(self, out_folder: Path) -> None:
        self._eval_inference_in_progress = False
        self.eval_status_var.set("Inference complete")
        self._append_eval_log("Inference complete.")
        self._append_eval_log(f"Saved outputs in: {out_folder}")
        self._load_step6_data_from_folder(out_folder)
        messagebox.showinfo("Step 6", f"Inference complete.\n\nOutputs saved in:\n{out_folder}")

    def _on_step6_inference_failed(self, details: str) -> None:
        self._eval_inference_in_progress = False
        self.eval_status_var.set("Inference failed")
        self._append_eval_log("Inference failed. See traceback below.")
        self._append_eval_log(details)
        messagebox.showerror("Step 6", "Inference failed. Check Step 6 logs.")

    def _load_step6_existing_data(self) -> None:
        folder = Path(self.eval_existing_data_folder_var.get().strip())
        if not folder.exists():
            messagebox.showerror("Missing folder", "Select a valid existing data folder first.")
            return
        self._load_step6_data_from_folder(folder)

    def _load_step6_data_from_folder(self, folder: Path) -> None:
        try:
            df = self._build_step6_long_dataframe(folder)
            if df.empty:
                messagebox.showwarning("No data", "No prediction data was found in this folder.")
                return

            self._eval_loaded_data = df
            self._update_step6_filter_options(df)
            self.eval_status_var.set(f"Loaded {len(df):,} points from {folder.name}")
            self._append_eval_log(f"Loaded prediction rows: {len(df):,}")

            labeled_outputs = sorted(folder.rglob("*labeled*.*"))
            traj_dirs = sorted([p for p in folder.rglob("*plot-poses*") if p.is_dir()])
            self._append_eval_log(f"Labeled video/image outputs found: {len(labeled_outputs)}")
            if labeled_outputs:
                self._append_eval_log(f"Example labeled output: {labeled_outputs[0]}")
            self._append_eval_log(f"Trajectory plot directories found: {len(traj_dirs)}")

            saved = self._save_step6_dataframe(df, folder)
            self._append_eval_log(f"Saved long-format data: {saved}")
            self._refresh_step6_plot()
        except Exception as exc:
            self._append_eval_log(traceback.format_exc())
            messagebox.showerror("Load failed", f"Could not load existing analysis data:\n{exc}")

    def _build_step6_long_dataframe(self, folder: Path) -> pd.DataFrame:
        pred_csvs = sorted(folder.glob("*DLC*.csv"))
        if not pred_csvs:
            pred_csvs = sorted(folder.rglob("*DLC*.csv"))
        if not pred_csvs:
            return pd.DataFrame()

        long_parts = [self._prediction_csv_to_long_for_step6(p) for p in pred_csvs]
        df = pd.concat(long_parts, ignore_index=True)

        truth_path = Path(self.eval_truth_csv_var.get().strip()) if self.eval_truth_csv_var.get().strip() else None
        if truth_path is not None and truth_path.exists():
            truth_long = self._truth_csv_to_long_for_step6(truth_path)
            if not truth_long.empty:
                df = df.merge(truth_long, on=["frame_id", "bodypart", "camera"], how="left")
                if {"x_true", "y_true"}.issubset(df.columns):
                    df["distance_px"] = ((df["x_pred"] - df["x_true"]) ** 2 + (df["y_pred"] - df["y_true"]) ** 2) ** 0.5

        df = df.sort_values(["camera", "bodypart", "frame_id"]).reset_index(drop=True)
        df["dx"] = df.groupby(["camera", "bodypart"])["x_pred"].diff()
        df["dy"] = df.groupby(["camera", "bodypart"])["y_pred"].diff()
        df["euclidean_displacement_t-1_to_t"] = (df["dx"] ** 2 + df["dy"] ** 2) ** 0.5
        return df

    def _prediction_csv_to_long_for_step6(self, pred_csv_path: Path) -> pd.DataFrame:
        wide, bp_level, coord_level = self._read_prediction_wide_for_step6(pred_csv_path)
        camera = self._infer_camera_from_name(pred_csv_path.name)
        bodyparts = pd.Index(wide.columns.get_level_values(bp_level)).unique()

        frame_ids = [self._normalize_frame_id(v) for v in wide.index]
        rows: list[pd.DataFrame] = []

        for bp in bodyparts:
            bp_cols = wide.xs(bp, axis=1, level=bp_level, drop_level=False)
            coords = bp_cols.columns.get_level_values(coord_level).astype(str).str.lower().tolist()
            if "x" not in coords or "y" not in coords:
                continue

            xcol = bp_cols.columns[coords.index("x")]
            ycol = bp_cols.columns[coords.index("y")]
            if "likelihood" in coords:
                lcol = bp_cols.columns[coords.index("likelihood")]
                like = pd.to_numeric(bp_cols[lcol], errors="coerce").to_numpy(dtype=float)
            else:
                like = pd.Series([float("nan")] * bp_cols.shape[0]).to_numpy(dtype=float)

            rows.append(
                pd.DataFrame(
                    {
                        "frame_id": frame_ids,
                        "bodypart": str(bp),
                        "camera": camera,
                        "x_pred": pd.to_numeric(bp_cols[xcol], errors="coerce").to_numpy(dtype=float),
                        "y_pred": pd.to_numeric(bp_cols[ycol], errors="coerce").to_numpy(dtype=float),
                        "likelihood": like,
                        "source_csv": pred_csv_path.name,
                    }
                )
            )

        if not rows:
            raise ValueError(f"No usable x/y columns in prediction file: {pred_csv_path}")
        return pd.concat(rows, ignore_index=True)

    def _read_prediction_wide_for_step6(self, pred_csv_path: Path) -> tuple[pd.DataFrame, int, int]:
        for header_rows in ([0, 1, 2], [0, 1, 2, 3], [0, 1]):
            try:
                wide = pd.read_csv(pred_csv_path, header=header_rows, index_col=0)
            except Exception:
                continue

            levels = self._detect_bodypart_coord_levels(wide)
            if levels is not None:
                bp_level, coord_level = levels
                return wide, bp_level, coord_level

        # Fallback: try paired H5 file when CSV formatting is unusual.
        h5_path = pred_csv_path.with_suffix(".h5")
        if h5_path.exists():
            try:
                wide_h5 = pd.read_hdf(h5_path)
                levels = self._detect_bodypart_coord_levels(wide_h5)
                if levels is not None:
                    bp_level, coord_level = levels
                    return wide_h5, bp_level, coord_level
            except Exception:
                pass

        raise ValueError(f"Could not detect bodypart/x/y columns in: {pred_csv_path}")

    def _detect_bodypart_coord_levels(self, wide: pd.DataFrame) -> tuple[int, int] | None:
        if not isinstance(wide.columns, pd.MultiIndex):
            return None

        nlevels = wide.columns.nlevels
        coord_candidates = {"x", "y", "likelihood"}

        for coord_level in range(nlevels):
            values = {
                str(v).strip().lower()
                for v in wide.columns.get_level_values(coord_level)
            }
            if "x" in values and "y" in values and len(values.intersection(coord_candidates)) >= 2:
                bp_level = max(0, coord_level - 1)
                return bp_level, coord_level

        return None

    def _truth_csv_to_long_for_step6(self, truth_csv_path: Path) -> pd.DataFrame:
        truth_df = pd.read_csv(truth_csv_path)
        parsed: list[tuple[str, str, str, str]] = []
        for col in truth_df.columns:
            match = re.match(r"(?P<bodypart>.+)_cam(?P<cam>[12])_(?P<coord>[XY])$", str(col))
            if match is None:
                continue
            parsed.append((match.group("bodypart"), f"cam{match.group('cam')}", match.group("coord").lower(), str(col)))

        if not parsed:
            return pd.DataFrame()

        meta = pd.DataFrame(parsed, columns=["bodypart", "camera", "coord", "column"])
        xmeta = meta[meta["coord"] == "x"].rename(columns={"column": "xcol"})
        ymeta = meta[meta["coord"] == "y"].rename(columns={"column": "ycol"})
        pairs = xmeta.merge(ymeta[["bodypart", "camera", "ycol"]], on=["bodypart", "camera"], how="inner")

        frame_ids = [self._normalize_frame_id(v) for v in range(1, len(truth_df) + 1)]
        out_parts: list[pd.DataFrame] = []
        for _, row in pairs.iterrows():
            out_parts.append(
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
        return pd.concat(out_parts, ignore_index=True)

    def _update_step6_filter_options(self, df: pd.DataFrame) -> None:
        bodyparts = sorted({str(v) for v in df["bodypart"].dropna().unique()})
        cameras = sorted({str(v) for v in df["camera"].dropna().unique()})

        bp_values = ["all"] + bodyparts
        cam_values = ["all"] + cameras

        if self.eval_bodypart_combo is not None:
            self.eval_bodypart_combo["values"] = bp_values
        if self.eval_camera_combo is not None:
            self.eval_camera_combo["values"] = cam_values

        if self.eval_bodypart_filter_var.get() not in bp_values:
            self.eval_bodypart_filter_var.set("all")
        if self.eval_camera_filter_var.get() not in cam_values:
            self.eval_camera_filter_var.set("all")

    def _save_step6_dataframe(self, df: pd.DataFrame, folder: Path) -> Path:
        target = folder / "analysis_long_format.csv"
        df.to_csv(target, index=False)
        return target

    def _refresh_step6_plot(self) -> None:
        if self._eval_loaded_data is None or self._eval_loaded_data.empty:
            messagebox.showwarning("No data", "Load inference outputs first.")
            return

        try:
            min_likelihood = float(self.eval_min_likelihood_var.get().strip())
        except Exception:
            messagebox.showerror("Invalid threshold", "Minimum likelihood must be a numeric value.")
            return

        df = self._eval_loaded_data.copy()
        if "likelihood" in df.columns:
            df = df[df["likelihood"].fillna(-1.0) >= min_likelihood]

        bodypart = self.eval_bodypart_filter_var.get().strip()
        if bodypart and bodypart != "all":
            df = df[df["bodypart"] == bodypart]

        camera = self.eval_camera_filter_var.get().strip()
        if camera and camera != "all":
            df = df[df["camera"] == camera]

        if df.empty:
            self._append_eval_log("Plot skipped: no rows after filters.")
            messagebox.showwarning("No rows", "No data left after applying filters.")
            return

        fig = self._build_step6_figure(df)
        self._show_step6_figure(fig)

        data_folder = Path(self.eval_existing_data_folder_var.get().strip()) if self.eval_existing_data_folder_var.get().strip() else None
        if data_folder is not None and data_folder.exists():
            plot_name = f"plot_{self.eval_plot_type_var.get()}_{self.eval_color_mode_var.get()}_{bodypart}_{camera}.png"
            plot_name = self._safe_name_for_folder(plot_name).replace(".png", "") + ".png"
            plot_path = data_folder / plot_name
            fig.savefig(plot_path, dpi=170, bbox_inches="tight")
            self._append_eval_log(f"Saved plot: {plot_path}")

    def _build_step6_figure(self, df: pd.DataFrame):
        plot_type = self.eval_plot_type_var.get().strip()
        color_mode = self.eval_color_mode_var.get().strip()

        if plot_type == "xy_over_time":
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            self._scatter_time_series(axes[0], df, y_col="x_pred", ylabel="x (px)", color_mode=color_mode)
            self._scatter_time_series(axes[1], df, y_col="y_pred", ylabel="y (px)", color_mode=color_mode)
            axes[1].set_xlabel("frame_id")
            fig.suptitle("X/Y marker positions over time")
            fig.tight_layout()
            return fig

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        if plot_type == "likelihood_over_time":
            self._scatter_time_series(ax, df, y_col="likelihood", ylabel="likelihood", color_mode=color_mode)
            ax.set_title("Likelihood over time")
        elif plot_type == "distance_over_time":
            if "distance_px" not in df.columns:
                raise ValueError("distance_from_truth plot requires truth CSV data.")
            self._scatter_time_series(ax, df, y_col="distance_px", ylabel="distance to truth (px)", color_mode=color_mode)
            ax.set_title("Distance-to-truth over time")
        else:
            self._scatter_time_series(
                ax,
                df.dropna(subset=["euclidean_displacement_t-1_to_t"]),
                y_col="euclidean_displacement_t-1_to_t",
                ylabel="euclidean displacement (t-1 to t) [px]",
                color_mode=color_mode,
            )
            ax.set_title("Displacement over time")

        ax.set_xlabel("frame_id")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    def _scatter_time_series(self, ax, df: pd.DataFrame, y_col: str, ylabel: str, color_mode: str) -> None:
        data = df.dropna(subset=["frame_id", y_col]).copy()
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        if color_mode == "likelihood" and "likelihood" in data.columns:
            sc = ax.scatter(data["frame_id"], data[y_col], c=data["likelihood"], cmap="viridis", s=10, alpha=0.6)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("likelihood")
        elif color_mode == "distance_from_truth" and "distance_px" in data.columns:
            sc = ax.scatter(data["frame_id"], data[y_col], c=data["distance_px"], cmap="magma", s=10, alpha=0.7)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("distance to truth (px)")
        else:
            for bodypart, group in data.groupby("bodypart"):
                ax.scatter(group["frame_id"], group[y_col], s=10, alpha=0.5, label=str(bodypart))
            if data["bodypart"].nunique() <= 12:
                ax.legend(fontsize=8, loc="best")

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    def _show_step6_figure(self, fig) -> None:
        if self.eval_plot_host is None:
            return

        for child in self.eval_plot_host.winfo_children():
            child.destroy()

        if FigureCanvasTkAgg is None:
            plt.show(block=False)
            fallback = ttk.Label(self.eval_plot_host, text="Plot opened in external matplotlib window.")
            fallback.grid(row=0, column=0, sticky="nsew")
            return

        canvas = FigureCanvasTkAgg(fig, master=self.eval_plot_host)
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        canvas.draw()
        self._eval_canvas = canvas

    def _infer_camera_from_name(self, name: str) -> str:
        low = name.lower()
        if "cam2" in low:
            return "cam2"
        return "cam1"

    def _normalize_frame_id(self, raw_value: object) -> float:
        try:
            return float(int(raw_value))
        except Exception:
            pass
        match = re.search(r"(\d+)(?!.*\d)", str(raw_value))
        if match is None:
            return float("nan")
        return float(int(match.group(1)))

    def _safe_name_for_folder(self, text: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
        safe = safe.strip("._")
        return safe or "model"

    def _extract_snapshot_num(self, snapshot_name: str) -> str:
        match = re.search(r"snapshot-(\d+)", snapshot_name)
        if match is None:
            return "unknown"
        return match.group(1)

    def _epochs_for_mode(self) -> int:
        mode = self.train_mode_var.get().strip().lower()
        if mode == "fulltrain":
            return 200
        return 2

    def _interval_epochs(self, total_epochs: int) -> int:
        try:
            interval_choice = int(self.log_interval_var.get().strip())
        except (TypeError, ValueError):
            interval_choice = 10
        return max(1, min(interval_choice, total_epochs))

    def _build_epoch_checkpoints(self, total_epochs: int, interval_epochs: int) -> list[int]:
        checkpoints = list(range(interval_epochs, total_epochs + 1, interval_epochs))
        if not checkpoints or checkpoints[-1] != total_epochs:
            checkpoints.append(total_epochs)
        return checkpoints

    def _append_train_log(self, message: str) -> None:
        if self.train_logs_text is None:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.train_logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.train_logs_text.see(tk.END)

    def _start_training(self) -> None:
        if self.current_config_path is None:
            messagebox.showwarning("Missing config", "Please complete Step 1 first.")
            return
        if self._training_in_progress:
            messagebox.showwarning("Training running", "Training is already in progress.")
            return

        labeled_dir = self.current_config_path.parent / "labeled-data"
        has_labels = bool(list(labeled_dir.rglob("CollectedData_*.h5")) or list(labeled_dir.rglob("CollectedData_*.csv")))
        if not has_labels:
            messagebox.showerror(
                "Missing labeled data",
                "No CollectedData files found under labeled-data. Run Step 3 first.",
            )
            return

        total_epochs = self._epochs_for_mode()
        interval_epochs = self._interval_epochs(total_epochs)
        checkpoints = self._build_epoch_checkpoints(total_epochs, interval_epochs)

        self._training_in_progress = True
        self._training_start_time = time.perf_counter()

        if self.train_logs_text is not None:
            self.train_logs_text.delete("1.0", tk.END)

        if self.train_progress is not None:
            self.train_progress.configure(maximum=total_epochs, value=0)

        self.train_status_var.set(
            f"Training started ({self.train_mode_var.get()}, epochs={total_epochs}, interval={interval_epochs})"
        )
        self._append_train_log("Preparing ResNet50 training...")
        self._append_train_log(f"Checkpoints: {checkpoints}")

        thread = threading.Thread(
            target=self._run_training_worker,
            args=(self.current_config_path, checkpoints),
            daemon=True,
        )
        thread.start()

    def _run_training_worker(self, config_path: Path, checkpoints: list[int]) -> None:
        try:
            import deeplabcut  # noqa: PLC0415

            set_net_type(config_path, "resnet_50")
            self.root.after(0, lambda: self._append_train_log("Set net_type to resnet_50."))

            ensure_config_scorer_matches_data(config_path)
            self.root.after(0, lambda: self._append_train_log("Scorer synchronized with labeled-data."))

            deeplabcut.create_training_dataset(str(config_path))
            self.root.after(0, lambda: self._append_train_log("Created training dataset."))

            snapshot_path: str | None = None
            for target_epoch in checkpoints:
                train_kwargs = {"epochs": target_epoch}
                if snapshot_path is not None:
                    train_kwargs["snapshot_path"] = snapshot_path

                deeplabcut.train_network(str(config_path), **train_kwargs)
                try:
                    snapshot_path = find_latest_snapshot(config_path)
                except FileNotFoundError:
                    # In very short runs (e.g., 1-2 epochs), DLC may not emit a snapshot yet.
                    snapshot_path = None

                self.root.after(
                    0,
                    lambda e=target_epoch, s=snapshot_path: self._on_training_checkpoint(e, s),
                )

            elapsed = time.perf_counter() - self._training_start_time
            self.root.after(0, lambda: self._on_training_finished(snapshot_path, elapsed))
        except Exception:
            details = traceback.format_exc()
            self.root.after(0, lambda: self._on_training_failed(details))

    def _on_training_checkpoint(self, epoch: int, snapshot_path: str | None) -> None:
        if self.train_progress is not None:
            self.train_progress.configure(value=epoch)
        self.train_status_var.set(f"Checkpoint reached: epoch {epoch}")
        if snapshot_path:
            self._append_train_log(f"Reached epoch {epoch}. Latest snapshot: {snapshot_path}")
        else:
            self._append_train_log(
                f"Reached epoch {epoch}. No snapshot file yet (normal for short/test runs)."
            )

    def _on_training_finished(self, snapshot_path: str | None, elapsed_s: float) -> None:
        self._training_in_progress = False
        total_epochs = self._epochs_for_mode()
        if self.train_progress is not None:
            self.train_progress.configure(value=total_epochs)
        self.train_status_var.set(f"Training complete in {elapsed_s/60:.1f} min")
        if snapshot_path:
            self._append_train_log(f"Training complete. Latest snapshot: {snapshot_path}")
            done_text = f"Model training finished.\n\nLatest snapshot:\n{snapshot_path}"
        else:
            self._append_train_log(
                "Training complete. No snapshot file was created at this epoch count."
            )
            done_text = (
                "Model training finished.\n\n"
                "No snapshot file was created yet (common in very short test runs).\n"
                "Try fulltrain or more epochs if you need a snapshot artifact."
            )
        messagebox.showinfo(
            "Training complete",
            done_text,
        )

    def _on_training_failed(self, details: str) -> None:
        self._training_in_progress = False
        self.train_status_var.set("Training failed")
        self._append_train_log("Training failed. See traceback below.")
        if self.train_logs_text is not None:
            self.train_logs_text.insert(tk.END, "\n" + details)
            self.train_logs_text.see(tk.END)
        messagebox.showerror("Training failed", "Training failed. See logs in Step 4.")

    def _pick_dataset_data_path(self) -> None:
        selected = filedialog.askdirectory(title="Select source data path for dataset build")
        if selected:
            self.dataset_data_path_var.set(selected)

    def _parse_frame_counts(self, raw_text: str) -> list[int]:
        counts: list[int] = []
        for token in re.split(r"[\s,;]+", raw_text.strip()):
            if not token:
                continue
            value = int(token)
            if value <= 0:
                raise ValueError("Frame counts must be positive integers.")
            counts.append(value)

        if not counts:
            raise ValueError("Please provide at least one frame count.")
        return counts

    def _create_training_datasets(self) -> None:
        if self.current_config_path is None:
            messagebox.showwarning("Missing config", "Please complete Step 1 first.")
            return

        data_path = Path(self.dataset_data_path_var.get().strip())
        dataset_name = self.dataset_name_var.get().strip()

        if not data_path.exists():
            messagebox.showerror("Invalid path", "Data Path does not exist.")
            return
        if not dataset_name:
            messagebox.showerror("Missing value", "Please enter Dataset Name.")
            return

        try:
            frame_counts = self._parse_frame_counts(self.frame_counts_var.get())
            seed_value = int(self.dataset_seed_var.get().strip())
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        total_available, per_trial_counts = self._estimate_available_frames(data_path)
        if total_available <= 0:
            messagebox.showerror(
                "No usable frames",
                "No valid frames were found in trial CSV files under Data Path. "
                "Expected per trial: 2 AVI files + 1 CSV with labeled 2D points.",
            )
            return

        too_large = [n for n in frame_counts if n > total_available]
        if too_large:
            per_trial_text = "\n".join(
                f"- {trial}: {count} valid frames"
                for trial, count in per_trial_counts.items()
            )
            messagebox.showerror(
                "Not enough frames",
                f"Requested frame counts {too_large} exceed available valid frames ({total_available}).\n\n"
                f"Per trial:\n{per_trial_text}",
            )
            return

        try:
            results: list[dict] = []
            for nframes in frame_counts:
                dataset_name_for_n = f"{dataset_name}_n{nframes}"
                build_combined_dataset(
                    combined_config=self.current_config_path,
                    data_path=data_path,
                    dataset_name=dataset_name_for_n,
                    experimenter=self.experimenter_var.get().strip(),
                    nframes=nframes,
                    frame_selection_seed=seed_value,
                )
                row = self._collect_dataset_frame_stats(
                    config_path=self.current_config_path,
                    nframes=nframes,
                    dataset_name=dataset_name_for_n,
                    total_available_frames=total_available,
                )
                results.append(row)

            if self.dataset_summary_text is not None:
                df = pd.DataFrame(results)
                self.dataset_summary_text.delete("1.0", tk.END)
                self.dataset_summary_text.insert(tk.END, df.to_string(index=False))

            self._populate_first_image_preview(self.current_config_path)
            self._populate_first_xy_long(self.current_config_path)
            messagebox.showinfo("Success", "Training dataset creation complete.")
        except Exception as exc:
            details = traceback.format_exc()
            messagebox.showerror("Build failed", f"Could not build training datasets:\n{exc}")
            if self.dataset_summary_text is not None:
                self.dataset_summary_text.delete("1.0", tk.END)
                self.dataset_summary_text.insert(tk.END, details)

    def _estimate_available_frames(self, data_path: Path) -> tuple[int, dict[str, int]]:
        per_trial_counts: dict[str, int] = {}
        total_available = 0

        trial_dirs = sorted([p for p in data_path.iterdir() if p.is_dir()])
        for trial_dir in trial_dirs:
            csv_candidates = sorted(trial_dir.glob("*.csv"))
            if not csv_candidates:
                continue

            # xrommtools reads the first CSV and treats row 0 as header-like labels.
            df = pd.read_csv(csv_candidates[0], sep=",", header=None)
            if df.empty or len(df) < 2:
                per_trial_counts[trial_dir.name] = 0
                continue

            df_points = df.loc[1:, :].reset_index(drop=True)
            ncol = df_points.shape[1]
            valid_rows = (~pd.isnull(df_points)).sum(axis=1) >= ncol / 2
            valid_count = int(valid_rows.sum())

            per_trial_counts[trial_dir.name] = valid_count
            total_available += valid_count

        return total_available, per_trial_counts

    def _collect_dataset_frame_stats(
        self,
        config_path: Path,
        nframes: int,
        dataset_name: str,
        total_available_frames: int,
    ) -> dict:
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        labeled_dir = Path(config_path).parent / "labeled-data"
        dataset_dir = labeled_dir / dataset_name
        image_paths = [
            p for p in dataset_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        ]

        unique_image_files = {
            p.relative_to(dataset_dir).as_posix()
            for p in image_paths
        }
        unique_logical_frames = {
            p.stem.replace("_cam1_", "_cam_").replace("_cam2_", "_cam_")
            for p in image_paths
        }

        return {
            "dataset_name": dataset_name,
            "nframes_requested": nframes,
            "total_available_frames": total_available_frames,
            "unique_image_files": len(unique_image_files),
            "unique_logical_frames": len(unique_logical_frames),
            "dataset_dir": str(dataset_dir),
        }

    def _populate_first_image_preview(self, config_path: Path) -> None:
        if self.image_preview_label is None or self.image_info_label is None:
            return

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        labeled_dir = Path(config_path).parent / "labeled-data"
        image_paths = sorted(
            [p for p in labeled_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
        )

        if not image_paths:
            self.image_preview_label.configure(text="No training image found.", image="")
            self.image_info_label.configure(text="")
            self._preview_photo = None
            return

        first_image = image_paths[0]
        self.image_info_label.configure(
            text=f"Path: {first_image}\nRelative: {first_image.relative_to(labeled_dir).as_posix()}"
        )

        if Image is None or ImageTk is None:
            self.image_preview_label.configure(text="Pillow is not available for image preview.", image="")
            self._preview_photo = None
            return

        pil_img = Image.open(first_image)
        pil_img.thumbnail((360, 360))
        self._preview_photo = ImageTk.PhotoImage(pil_img)
        self.image_preview_label.configure(image=self._preview_photo, text="")

    def _populate_first_xy_long(self, config_path: Path) -> None:
        if self.xy_long_text is None:
            return

        labeled_dir = Path(config_path).parent / "labeled-data"
        csv_candidates = sorted(labeled_dir.rglob("CollectedData_*.csv"))
        if not csv_candidates:
            self.xy_long_text.delete("1.0", tk.END)
            self.xy_long_text.insert(tk.END, "No CollectedData_*.csv found in labeled-data.")
            return

        first_csv = csv_candidates[0]
        df = pd.read_csv(first_csv, header=[0, 1, 2], index_col=0)
        if df.empty:
            self.xy_long_text.delete("1.0", tk.END)
            self.xy_long_text.insert(tk.END, f"No label rows found in: {first_csv}")
            return

        first_row = df.iloc[0]
        long_rows: list[dict] = []

        if isinstance(df.columns, pd.MultiIndex):
            bp_level = 1 if df.columns.nlevels >= 2 else 0
            coord_level = df.columns.nlevels - 1
            bodyparts = [str(bp) for bp in df.columns.get_level_values(bp_level).unique()]

            for bp in bodyparts:
                x_val = None
                y_val = None
                for col in df.columns:
                    if str(col[bp_level]) != bp:
                        continue
                    coord = str(col[coord_level]).lower()
                    value = first_row[col]
                    if coord == "x":
                        x_val = value
                    elif coord == "y":
                        y_val = value
                long_rows.append({"bodypart": bp, "x": x_val, "y": y_val})
        else:
            # Fallback for unexpected flat columns.
            for col in df.columns:
                long_rows.append({"bodypart": str(col), "x": first_row[col], "y": None})

        long_df = pd.DataFrame(long_rows)
        self.xy_long_text.delete("1.0", tk.END)
        self.xy_long_text.insert(tk.END, f"CSV: {first_csv}\n")
        self.xy_long_text.insert(tk.END, f"Row ID: {df.index[0]}\n\n")
        self.xy_long_text.insert(tk.END, long_df.to_string(index=False))

    def _load_step2_from_config(self) -> None:
        if self.current_config_path is None or self.bodyparts_text is None:
            return

        try:
            with open(self.current_config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            parts = cfg.get("bodyparts", [])
            if not isinstance(parts, list):
                parts = []

            self.bodyparts_text.delete("1.0", tk.END)
            if parts:
                self.bodyparts_text.insert(tk.END, "\n".join(str(p) for p in parts))

            self._refresh_config_preview()
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not load bodyparts from config:\n{exc}")

    def _on_bird_suggestion_selected(self, _event: tk.Event) -> None:
        selected = self.bird_suggestion_var.get().strip()
        self._apply_bird_bodypart_suggestion(selected)

    def _apply_bird_bodypart_suggestion(self, bird_name: str) -> None:
        if self.bodyparts_text is None:
            return

        parts = BIRD_BODYPARTS.get(bird_name, [])
        if not parts:
            return

        self.bodyparts_text.delete("1.0", tk.END)
        self.bodyparts_text.insert(tk.END, "\n".join(str(p) for p in parts))

        if self.current_project_dict is not None:
            self.current_project_dict.setdefault("next_fields_to_fill", {})["bird_name"] = bird_name
            self.current_project_dict.setdefault("next_fields_to_fill", {})["bodyparts"] = list(parts)
            self._write_output_with_bold_keys(self.current_project_dict)

    def _save_bodyparts_to_config(self) -> None:
        if self.current_config_path is None or self.bodyparts_text is None:
            messagebox.showwarning("No config", "Create or load a project first.")
            return

        try:
            with open(self.current_config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}

            raw_bodyparts = self.bodyparts_text.get("1.0", tk.END)
            bodyparts = self._parse_bodyparts_input(raw_bodyparts)

            if not bodyparts:
                messagebox.showwarning(
                    "No bodyparts found",
                    "Could not parse any bodyparts. Use commas, dash-separated items, or one item per line.",
                )
                return

            cfg["bodyparts"] = bodyparts

            with open(self.current_config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)

            if self.current_project_dict is not None:
                self.current_project_dict.setdefault("next_fields_to_fill", {})["bodyparts"] = bodyparts
                self._write_output_with_bold_keys(self.current_project_dict)

            self._refresh_config_preview()
            messagebox.showinfo("Saved", "Bodyparts were saved to config.yaml")
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save bodyparts to config:\n{exc}")

    def _parse_bodyparts_input(self, raw_text: str) -> list[str]:
        # Accept three common formats:
        # 1) comma-separated: "beak, wing, tail"
        # 2) dash-separated: "beak - wing - tail"
        # 3) bullet/line-based: "- beak" or one bodypart per line
        items: list[str] = []

        for line in raw_text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue

            # Remove leading bullet marker if present.
            if cleaned.startswith("-"):
                cleaned = cleaned[1:].strip()

            # Split on commas first, then on spaced dash delimiters.
            comma_parts = [part.strip() for part in cleaned.split(",") if part.strip()]
            for part in comma_parts:
                dash_parts = [seg.strip() for seg in re.split(r"\s+-\s+", part) if seg.strip()]
                if dash_parts:
                    items.extend(dash_parts)

        # Preserve order while removing accidental duplicates.
        seen: set[str] = set()
        ordered_unique: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered_unique.append(item)

        return ordered_unique

    def _refresh_config_preview(self) -> None:
        if self.current_config_path is None or self.config_preview_text is None:
            return

        try:
            with open(self.current_config_path, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.config_preview_text.delete("1.0", tk.END)
            self.config_preview_text.insert(tk.END, text)
        except Exception as exc:
            messagebox.showerror("Preview failed", f"Could not read config file:\n{exc}")


def main() -> None:
    root = tk.Tk()
    app = ProjectSetupUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
