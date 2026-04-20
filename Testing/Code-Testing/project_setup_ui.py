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

import pandas as pd
import yaml

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
        self._eval_snapshot_map: dict[str, Path] = {}

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
            return

        snapshot_path = self._eval_snapshot_map.get(selected_label)
        if snapshot_path is None:
            self.eval_snapshot_text.insert(tk.END, "No snapshot selected.")
            return

        self.eval_snapshot_path_var.set(str(snapshot_path))
        self.eval_snapshot_text.insert(tk.END, str(snapshot_path))

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
