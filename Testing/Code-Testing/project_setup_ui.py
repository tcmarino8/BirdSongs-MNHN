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
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml

from DLCsupport import create_combined_project_if_missing


class ProjectSetupUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DLC Project Setup (Starter)")
        self.root.geometry("980x700")

        self.project_parent_dir = tk.StringVar()
        self.dummy_video_dir = tk.StringVar()

        self.task_var = tk.StringVar(value="Canari")
        self.experimenter_var = tk.StringVar(value="Tyler")
        self.project_root_name_var = tk.StringVar(value="Canari_combined_training")
        self.selected_dummy_video_var = tk.StringVar()

        self.current_project_dict: dict | None = None
        self.current_config_path: Path | None = None

        self.step2_frame: ttk.LabelFrame | None = None
        self.bodyparts_text: tk.Text | None = None
        self.config_preview_text: tk.Text | None = None

        self._build_layout()

    def _build_layout(self) -> None:
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
            text="DLC Setup Wizard (Progressive Steps)",
            font=("Segoe UI", 14, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 12))

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

        action_row = ttk.Frame(step1)
        action_row.grid(row=5, column=0, columnspan=4, sticky="we", pady=(10, 8))

        ttk.Button(action_row, text="Create Project + Build Dictionary", command=self._create_project_and_build_dict).pack(side="left")
        ttk.Button(action_row, text="Save Dictionary As JSON", command=self._save_dict_json).pack(side="left", padx=10)
        ttk.Button(action_row, text="Clear Output", command=self._clear_output).pack(side="left")

        ttk.Label(step1, text="Dictionary Output", font=("Segoe UI", 11, "bold")).grid(
            row=6, column=0, columnspan=4, sticky="w", pady=(8, 4)
        )

        self.output = tk.Text(step1, wrap="word", height=16)
        self.output.grid(row=7, column=0, columnspan=4, sticky="nsew")
        self.output.tag_configure("dict_key", font=("Segoe UI", 9, "bold"))

        out_scroll = ttk.Scrollbar(step1, orient="vertical", command=self.output.yview)
        out_scroll.grid(row=7, column=4, sticky="ns")
        self.output.configure(yscrollcommand=out_scroll.set)

        step1.rowconfigure(7, weight=1)
        self.page.columnconfigure(0, weight=1)

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
            project_dir = Path(config_path).parent
            self.current_config_path = Path(config_path)

            starter = {
                "project": {
                    "task": task,
                    "experimenter": experimenter,
                    "combined_project_root": str(combined_project_root),
                    "config_path": str(config_path),
                    "project_dir": str(project_dir),
                    "dummy_video": str(dummy_video),
                    "dummy_video_folder": str(dummy_video.parent),
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

            self.current_project_dict = starter
            self._write_output_with_bold_keys(starter)
            self._ensure_step2_section()
            self._load_step2_from_config()
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

        ttk.Label(left, text="Editable Bodyparts (one per line)", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.bodyparts_text = tk.Text(left, wrap="none", height=16)
        self.bodyparts_text.grid(row=1, column=0, sticky="nsew", pady=(4, 6))
        left.rowconfigure(1, weight=1)

        btn_row = ttk.Frame(left)
        btn_row.grid(row=2, column=0, sticky="w")
        ttk.Button(btn_row, text="Reload Bodyparts From Config", command=self._load_step2_from_config).pack(side="left")
        ttk.Button(btn_row, text="Save Bodyparts To Config", command=self._save_bodyparts_to_config).pack(side="left", padx=8)

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
