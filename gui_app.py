import atexit
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, colorchooser
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk, ImageSequence

from helper_functions import (
    read_image_rgb,
    fitted_image,
    image_to_palette,
    palette_square_padded,
    palletized_colorspace_fast,
    format_eta,
    is_video_file,
    is_gif_file,
    compute_clamped_dimensions,
)

FULL_DIMENSIONS = 100000000000000
MAX_HISTORY_STATES = 50

THUMB_W = 155
THUMB_H = 100
HISTORY_THUMB_MAX_DIM = 320


class MediaState:
    """
    Represents a single undo/redo state in the history timeline.
    Supports:
      - Static images (kind="image" or 1-frame "gif", backed in memory or temp_frames)
      - Animated GIFs (kind="gif", backed by lossless PNG frames in temp_dir)
      - Videos (kind="video", backed either by source_video with target clamped dimensions
        or by lossless PNG frames in temp_dir after palletizing/resizing)
    Full video frames are never kept as a list in memory; inactive history states hold only
    a small thumbnail (`thumb_rgb`) in RAM while full frames live on disk.
    """

    def __init__(
        self,
        uid: int,
        kind: str,
        backing: str,
        width: int,
        height: int,
        frame_count: int = 1,
        fps: float = 30.0,
        image_rgb: np.ndarray | None = None,
        folder_path: str | None = None,
        source_path: str | None = None,
        thumb_rgb: np.ndarray | None = None,
        durations_ms: list[int] | None = None,
        first_frame_rgb: np.ndarray | None = None,
    ):
        self.uid = uid
        self.kind = kind  # "image", "gif", or "video"
        self.backing = backing  # "memory", "temp_frames", or "source_video"
        self.width = int(width)
        self.height = int(height)
        self.frame_count = max(1, int(frame_count))
        self.fps = float(fps) if (fps and not math.isnan(fps) and 1.0 <= fps <= 240.0) else 30.0
        self.image_rgb = image_rgb
        self.folder_path = folder_path
        self.source_path = source_path
        self.durations_ms = durations_ms

        if thumb_rgb is not None:
            self.thumb_rgb = thumb_rgb
        elif image_rgb is not None:
            self.thumb_rgb = fitted_image(image_rgb, HISTORY_THUMB_MAX_DIM)
        elif first_frame_rgb is not None:
            self.thumb_rgb = fitted_image(first_frame_rgb, HISTORY_THUMB_MAX_DIM)
        else:
            self.thumb_rgb = np.zeros((max(1, min(height, 100)), max(1, min(width, 100)), 3), dtype=np.uint8)

        self._lock = threading.Lock()
        self._cap: cv.VideoCapture | None = None
        self._cap_pos: int = 0
        self._cached_idx: int = 0 if first_frame_rgb is not None else -1
        self._cached_frame: np.ndarray | None = first_frame_rgb

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    @property
    def is_multiframe(self) -> bool:
        return self.kind == "video" or self.frame_count > 1

    @classmethod
    def from_image(cls, uid: int, image_rgb: np.ndarray, kind: str = "image") -> "MediaState":
        h, w = image_rgb.shape[:2]
        return cls(
            uid=uid,
            kind=kind,
            backing="memory",
            width=w,
            height=h,
            frame_count=1,
            fps=30.0,
            image_rgb=image_rgb,
            thumb_rgb=fitted_image(image_rgb, HISTORY_THUMB_MAX_DIM),
        )

    @classmethod
    def from_source_video(
        cls,
        uid: int,
        source_path: str,
        width: int,
        height: int,
        frame_count: int,
        fps: float,
        thumb_rgb: np.ndarray,
        first_frame_rgb: np.ndarray | None = None,
    ) -> "MediaState":
        return cls(
            uid=uid,
            kind="video",
            backing="source_video",
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            source_path=source_path,
            thumb_rgb=thumb_rgb,
            first_frame_rgb=first_frame_rgb,
        )

    @classmethod
    def from_temp_folder(
        cls,
        uid: int,
        kind: str,
        folder_path: str,
        width: int,
        height: int,
        frame_count: int,
        fps: float,
        thumb_rgb: np.ndarray,
        durations_ms: list[int] | None = None,
        first_frame_rgb: np.ndarray | None = None,
    ) -> "MediaState":
        return cls(
            uid=uid,
            kind=kind,
            backing="temp_frames",
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            folder_path=folder_path,
            thumb_rgb=thumb_rgb,
            durations_ms=durations_ms,
            first_frame_rgb=first_frame_rgb,
        )

    def get_frame_delay_ms(self, idx: int) -> int:
        if self.durations_ms and 0 <= idx < len(self.durations_ms):
            return max(15, int(self.durations_ms[idx]))
        return max(15, int(round(1000.0 / max(1.0, self.fps))))

    def get_frame(self, idx: int) -> np.ndarray:
        """Fetches a single RGB uint8 frame by index on demand."""
        idx = max(0, min(int(idx), self.frame_count - 1))

        if self.backing == "memory" and self.image_rgb is not None:
            return self.image_rgb

        with self._lock:
            if self._cached_idx == idx and self._cached_frame is not None:
                return self._cached_frame

            frame_rgb = self._read_frame_unlocked(idx)
            self._cached_idx = idx
            self._cached_frame = frame_rgb
            return frame_rgb

    def _read_frame_unlocked(self, idx: int) -> np.ndarray:
        if self.backing == "temp_frames" and self.folder_path:
            frame_path = os.path.join(self.folder_path, f"frame_{idx:06d}.png")
            if os.path.exists(frame_path):
                bgr = cv.imread(frame_path, cv.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                    if (rgb.shape[1], rgb.shape[0]) != (self.width, self.height):
                        rgb = cv.resize(rgb, (self.width, self.height), interpolation=cv.INTER_NEAREST)
                    return rgb

        elif self.backing == "source_video" and self.source_path:
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv.VideoCapture(self.source_path)
                self._cap_pos = 0

            if self._cap.isOpened():
                if idx != self._cap_pos:
                    self._cap.set(cv.CAP_PROP_POS_FRAMES, idx)
                    self._cap_pos = idx

                ret, bgr = self._cap.read()
                if not ret or bgr is None:
                    # Fallback if container metadata over-reported total frame count
                    if idx > 0:
                        self.frame_count = max(1, idx)
                    self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    ret, bgr = self._cap.read()
                    self._cap_pos = 1 if ret else 0
                else:
                    self._cap_pos = idx + 1

                if ret and bgr is not None:
                    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                    if (rgb.shape[1], rgb.shape[0]) != (self.width, self.height):
                        rgb = cv.resize(rgb, (self.width, self.height), interpolation=cv.INTER_NEAREST)
                    return rgb

        # Safe fallback using thumbnail scaled to target dimensions
        return cv.resize(self.thumb_rgb, (self.width, self.height), interpolation=cv.INTER_NEAREST)

    def iter_frames_for_worker(self):
        """
        Thread-safe generator yielding (frame_idx, frame_rgb) sequentially for background workers.
        Uses its own dedicated file reader so it never collides with main-thread playback.
        """
        if self.backing == "memory" and self.image_rgb is not None:
            yield 0, self.image_rgb.copy()
            return

        if self.backing == "temp_frames" and self.folder_path:
            for idx in range(self.frame_count):
                frame_path = os.path.join(self.folder_path, f"frame_{idx:06d}.png")
                if not os.path.exists(frame_path):
                    break
                bgr = cv.imread(frame_path, cv.IMREAD_COLOR)
                if bgr is None:
                    break
                rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                if (rgb.shape[1], rgb.shape[0]) != (self.width, self.height):
                    rgb = cv.resize(rgb, (self.width, self.height), interpolation=cv.INTER_NEAREST)
                yield idx, rgb
            return

        if self.backing == "source_video" and self.source_path:
            cap = cv.VideoCapture(self.source_path)
            try:
                if not cap.isOpened():
                    raise ValueError(f"Could not open video source: {self.source_path}")
                idx = 0
                while idx < self.frame_count:
                    ret, bgr = cap.read()
                    if not ret or bgr is None:
                        break
                    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                    if (rgb.shape[1], rgb.shape[0]) != (self.width, self.height):
                        rgb = cv.resize(rgb, (self.width, self.height), interpolation=cv.INTER_NEAREST)
                    yield idx, rgb
                    idx += 1
            finally:
                cap.release()

    def release_memory_cache(self):
        """Releases cached full-resolution frame and any open VideoCapture handle when state is inactive."""
        with self._lock:
            self._cached_frame = None
            self._cached_idx = -1
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                self._cap_pos = 0

    def detach_source_to_temp(self, temp_dir: str):
        """
        If this state references an external source_path that is about to be overwritten by Save As,
        copies the source file into temp_dir and repoints this state to the temp copy.
        """
        if self.backing != "source_video" or not self.source_path:
            return
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                self._cap_pos = 0
            ext = os.path.splitext(self.source_path)[1] or ".mp4"
            backup_path = os.path.join(temp_dir, f"source_backup_{self.uid}{ext}")
            if not os.path.exists(backup_path):
                shutil.copy2(self.source_path, backup_path)
            self.source_path = backup_path

    def cleanup(self, keep_folders: set[str] | None = None):
        """Deletes temp disk files when this state is permanently evicted from history."""
        self.release_memory_cache()
        if self.backing == "temp_frames" and self.folder_path:
            if keep_folders is not None and self.folder_path in keep_folders:
                return
            if os.path.exists(self.folder_path):
                shutil.rmtree(self.folder_path, ignore_errors=True)


class PaletteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image & Video Palette Transfer Studio")
        self.root.geometry("1220x960")
        self.root.minsize(1020, 860)

        # Session temporary folder for disk-backed Ctrl+Z video/GIF history states
        self.temp_dir = tempfile.mkdtemp(prefix="palette_stealer_hist_")
        self._state_uid_counter = 0
        atexit.register(self._cleanup_temp_dir)
        self.root.protocol("WM_DELETE_WINDOW", self._on_app_close)

        # Optional cross-platform window icon
        for icon_candidate in ("icon_image.ico", "glitchy.ico"):
            if os.path.exists(icon_candidate):
                try:
                    icon_img = ImageTk.PhotoImage(Image.open(icon_candidate))
                    self.root.iconphoto(True, icon_img)
                    break
                except Exception:
                    pass

        # Image & Media state variables
        self.raw_original_image = None
        self.palette_image = None
        self.palette = None
        self.palette_square_img = None
        self.main_canvas_image = None
        self.secondary_thumb_image = None

        # Video / Animated GIF Playback state
        self.current_frame_idx = 0
        self.is_playing = False
        self._playback_after_id = None
        self._updating_timeline_programmatically = False
        self._is_worker_busy = False
        self._last_previewed_frame_idx = -1

        # Main Canvas Interactive Zoom & Pan State
        self.zoom_factor = 1.0  # 1.0 = Perfectly fitted to canvas
        self.max_zoom = 25.0
        self.img_offset_x = 0.0
        self.img_offset_y = 0.0
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._did_drag = False
        self._main_tk_img = None

        # Timeline & Cancellation state (Main Palletizer / Video Worker)
        self.history: list[MediaState] = []
        self.history_index = -1
        self.cancel_requested = False
        self.active_task_id = 0
        self._main_shared_state = {}

        # Cancellation state (Palette Extraction)
        self.pal_cancel_requested = False
        self.active_pal_task_id = 0
        self._pal_shared_state = {}

        self.color_spaces = {
            "LAB": cv.COLOR_RGB2LAB,
            "HSV": cv.COLOR_RGB2HSV,
            "RGB": None,
            "HLS": cv.COLOR_RGB2HLS,
            "LUV": cv.COLOR_RGB2LUV,
            "YUV": cv.COLOR_RGB2YUV,
            "YCrCb": cv.COLOR_RGB2YCrCb,
            "XYZ": cv.COLOR_RGB2XYZ,
        }

        self.dither_modes = [
            "None",
            "Bayer 2x2",
            "Bayer 4x4",
            "Bayer 8x8",
            "White Noise",
            "Floyd-Steinberg",
            "Atkinson",
        ]

        # Load saved weight presets from disk
        self.presets_file = self._get_presets_file_path()
        self.presets = self._load_presets_from_disk()

        # Load UI color themes from disk
        self.style = ttk.Style(self.root)
        self.themes_file = self._get_themes_file_path()
        self.ui_themes, self.active_theme_name = self._load_themes_from_disk()
        self.current_theme = self.ui_themes[self.active_theme_name].copy()

        self._build_ui()
        self._bind_shortcuts()
        self.apply_ui_theme()

    # ==========================================
    # SESSION TEMP DIRECTORY & CLEANUP
    # ==========================================
    def _next_state_uid(self) -> int:
        self._state_uid_counter += 1
        return self._state_uid_counter

    @property
    def current_media_state(self) -> MediaState | None:
        if 0 <= self.history_index < len(self.history):
            return self.history[self.history_index]
        return None

    @property
    def previous_media_state(self) -> MediaState | None:
        if self.history_index > 0 and (self.history_index - 1) < len(self.history):
            return self.history[self.history_index - 1]
        return None

    def _cleanup_temp_dir(self):
        for state in getattr(self, "history", []):
            try:
                state.release_memory_cache()
            except Exception:
                pass
        temp_dir = getattr(self, "temp_dir", None)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _on_app_close(self):
        self.pause_playback()
        self.cancel_requested = True
        self.pal_cancel_requested = True
        self.active_task_id += 1
        self.active_pal_task_id += 1
        self._cleanup_temp_dir()
        try:
            self.root.destroy()
        except Exception:
            pass

    # ==========================================
    # PRESET PERSISTENCE MANAGEMENT
    # ==========================================
    def _get_presets_file_path(self) -> str:
        """Returns path to weight_presets.json next to the executable or script."""
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        primary_path = os.path.join(base_dir, "weight_presets.json")
        if os.access(base_dir, os.W_OK):
            return primary_path
        return os.path.join(os.path.expanduser("~"), ".palette_studio_presets.json")

    def _load_presets_from_disk(self) -> dict:
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def _save_presets_to_disk(self):
        try:
            with open(self.presets_file, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            messagebox.showerror("Preset Save Error", f"Could not save presets to disk:\n{e}")

    def save_current_preset(self):
        """Saves current Color Space and Channel Weights under a custom name."""
        name = self.preset_var.get().strip()
        if not name:
            name = simpledialog.askstring(
                "Save Weight Preset",
                "Enter a name for this weight configuration:",
                parent=self.root
            )
            if not name or not name.strip():
                return
            name = name.strip()

        self.presets[name] = {
            "color_space": self.color_space_var.get(),
            "w1": round(float(self.w1_var.get()), 4),
            "w2": round(float(self.w2_var.get()), 4),
            "w3": round(float(self.w3_var.get()), 4),
        }
        self._save_presets_to_disk()
        self.preset_dropdown.config(values=list(self.presets.keys()))
        self.preset_var.set(name)

    def delete_selected_preset(self):
        """Deletes the currently selected preset from memory and disk."""
        name = self.preset_var.get().strip()
        if not name or name not in self.presets:
            messagebox.showwarning("Delete Preset", "Please select an existing saved preset to delete.")
            return

        if messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"):
            del self.presets[name]
            self._save_presets_to_disk()
            remaining = list(self.presets.keys())
            self.preset_dropdown.config(values=remaining)
            self.preset_var.set("")

    def _on_preset_selected(self, event=None):
        """Loads weights and color space when a preset is chosen from the dropdown."""
        name = self.preset_var.get().strip()
        if name not in self.presets:
            return

        cfg = self.presets[name]
        if "color_space" in cfg and cfg["color_space"] in self.color_spaces:
            self.color_space_var.set(cfg["color_space"])
        if "w1" in cfg:
            self.w1_var.set(float(cfg["w1"]))
        if "w2" in cfg:
            self.w2_var.set(float(cfg["w2"]))
        if "w3" in cfg:
            self.w3_var.set(float(cfg["w3"]))

    def _get_themes_file_path(self) -> str:
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        primary_path = os.path.join(base_dir, "ui_themes.json")
        if os.access(base_dir, os.W_OK):
            return primary_path
        return os.path.join(os.path.expanduser("~"), ".palette_studio_themes.json")

    def _default_themes(self) -> dict:
        return {
            "Classic Dark": {
                "bg_main": "#222222",
                "fg_text": "#e6e6e6",
                "bg_input": "#333333",
                "fg_input": "#ffffff",
                "accent": "#3a7ebf",
                "canvas_bg": "#181818",
                "thumb_bg": "#2b2b2b",
            },
            "Classic Light": {
                "bg_main": "#f0f0f0",
                "fg_text": "#111111",
                "bg_input": "#ffffff",
                "fg_input": "#111111",
                "accent": "#0078d7",
                "canvas_bg": "#1e1e1e",
                "thumb_bg": "#d9d9d9",
            },
        }

    def _load_themes_from_disk(self) -> tuple[dict, str]:
        defaults = self._default_themes()
        if os.path.exists(self.themes_file):
            try:
                with open(self.themes_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                themes = data.get("themes", {})
                active = data.get("active_theme", "Classic Dark")
                if isinstance(themes, dict) and themes:
                    for k, v in defaults.items():
                        themes.setdefault(k, v)
                    if active not in themes:
                        active = next(iter(themes.keys()))
                    return themes, active
            except Exception:
                pass
        return defaults, "Classic Dark"

    def _save_themes_to_disk(self):
        try:
            payload = {
                "active_theme": self.active_theme_name,
                "themes": self.ui_themes,
            }
            with open(self.themes_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            messagebox.showerror("Theme Save Error", f"Could not save UI themes:\n{e}")

    def apply_ui_theme(self, theme_dict: dict | None = None):
        """Applies the color dictionary to all ttk and tk widgets in the app."""
        if theme_dict is not None:
            self.current_theme = theme_dict.copy()

        t = self.current_theme
        bg_main = t.get("bg_main", "#222222")
        fg_text = t.get("fg_text", "#e6e6e6")
        bg_input = t.get("bg_input", "#333333")
        fg_input = t.get("fg_input", "#ffffff")
        accent = t.get("accent", "#3a7ebf")
        canvas_bg = t.get("canvas_bg", "#181818")
        thumb_bg = t.get("thumb_bg", "#2b2b2b")

        # 'clam' allows full programmatic color customization across OSes
        self.style.theme_use("clam")

        self.root.configure(bg=bg_main)
        self.style.configure(".", background=bg_main, foreground=fg_text)
        self.style.configure("TFrame", background=bg_main)
        self.style.configure("TLabelframe", background=bg_main, bordercolor=bg_input)
        self.style.configure("TLabelframe.Label", background=bg_main, foreground=fg_text)
        self.style.configure("TLabel", background=bg_main, foreground=fg_text)
        self.style.configure("TSeparator", background=bg_input)

        # Buttons
        self.style.configure("TButton", background=bg_input, foreground=fg_input, bordercolor=bg_input)
        self.style.map(
            "TButton",
            background=[("active", accent), ("disabled", bg_main)],
            foreground=[("active", "#ffffff"), ("disabled", "#777777")],
        )

        # Entries & Comboboxes
        self.style.configure("TEntry", fieldbackground=bg_input, foreground=fg_input, insertcolor=fg_input)
        self.style.configure("TCombobox", fieldbackground=bg_input, background=bg_input, foreground=fg_input,
                             arrowcolor=fg_input)
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", bg_input)],
            foreground=[("readonly", fg_input)],
            selectbackground=[("readonly", accent)],
            selectforeground=[("readonly", "#ffffff")],
        )
        self.root.option_add("*TCombobox*Listbox*Background", bg_input)
        self.root.option_add("*TCombobox*Listbox*Foreground", fg_input)
        self.root.option_add("*TCombobox*Listbox*selectBackground", accent)
        self.root.option_add("*TCombobox*Listbox*selectForeground", "#ffffff")

        # Sliders & Progressbars
        self.style.configure("Horizontal.TScale", background=bg_main, troughcolor=bg_input)
        self.style.configure("Horizontal.TProgressbar", background=accent, troughcolor=bg_input, bordercolor=bg_input)

        # Standard tk widgets (Thumbnails & Main Canvas)
        for box in (self.pal_source_box, self.pal_square_box, self.prev_image_box):
            box.configure(bg=thumb_bg)
        for thumb in (self.pal_source_thumb, self.pal_square_thumb, self.prev_image_thumb):
            thumb.configure(bg=thumb_bg, fg=fg_text)

        self.main_canvas.configure(bg=canvas_bg)
        self._render_main_canvas()

    def open_theme_settings(self):
        """Opens a popup window to customize, save, and delete UI color setups."""
        win = tk.Toplevel(self.root)
        win.title("UI Theme Settings")
        win.geometry("380x430")
        win.resizable(False, False)
        win.configure(bg=self.current_theme.get("bg_main", "#222222"))
        win.transient(self.root)

        container = ttk.Frame(win, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        # 1. Preset Selector Row
        top_row = ttk.LabelFrame(container, text="Saved Color Setups", padding=8)
        top_row.pack(fill=tk.X, pady=(0, 10))

        theme_var = tk.StringVar(value=self.active_theme_name)
        theme_cb = ttk.Combobox(top_row, textvariable=theme_var, values=list(self.ui_themes.keys()), width=16)
        theme_cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        # 2. Color Pickers Section
        colors_frame = ttk.LabelFrame(container, text="Customize Colors", padding=8)
        colors_frame.pack(fill=tk.BOTH, expand=True)

        color_fields = [
            ("bg_main", "Window & Panel Background"),
            ("fg_text", "Main Text Color"),
            ("bg_input", "Button & Input Background"),
            ("fg_input", "Button & Input Text"),
            ("accent", "Highlight / Progress Bar"),
            ("canvas_bg", "Main Canvas Background"),
            ("thumb_bg", "Thumbnail Box Background"),
        ]

        swatch_buttons = {}

        def _pick_color(key):
            initial = self.current_theme.get(key, "#222222")
            chosen = colorchooser.askcolor(color=initial, title=f"Choose {key}", parent=win)
            if chosen and chosen[1]:
                hex_col = chosen[1]
                self.current_theme[key] = hex_col
                swatch_buttons[key].configure(bg=hex_col)
                self.apply_ui_theme(self.current_theme)
                win.configure(bg=self.current_theme["bg_main"])

        for key, label_desc in color_fields:
            row = ttk.Frame(colors_frame)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label_desc).pack(side=tk.LEFT)

            col_val = self.current_theme.get(key, "#222222")
            btn = tk.Button(
                row, bg=col_val, width=6, relief=tk.SOLID, bd=1, cursor="hand2",
                command=lambda k=key: _pick_color(k)
            )
            btn.pack(side=tk.RIGHT)
            swatch_buttons[key] = btn

        def _refresh_swatches():
            for k, btn in swatch_buttons.items():
                btn.configure(bg=self.current_theme.get(k, "#222222"))
            win.configure(bg=self.current_theme["bg_main"])

        def _on_select_theme(event=None):
            name = theme_var.get().strip()
            if name in self.ui_themes:
                self.active_theme_name = name
                self.apply_ui_theme(self.ui_themes[name])
                _refresh_swatches()
                self._save_themes_to_disk()

        theme_cb.bind("<<ComboboxSelected>>", _on_select_theme)

        def _save_theme():
            name = theme_var.get().strip()
            if not name:
                name = simpledialog.askstring("Save Theme", "Enter a name for this color setup:", parent=win)
                if not name or not name.strip():
                    return
                name = name.strip()

            self.ui_themes[name] = self.current_theme.copy()
            self.active_theme_name = name
            self._save_themes_to_disk()
            theme_cb.config(values=list(self.ui_themes.keys()))
            theme_var.set(name)

        def _delete_theme():
            name = theme_var.get().strip()
            if name not in self.ui_themes:
                return
            if len(self.ui_themes) <= 1:
                messagebox.showwarning("Delete Theme", "You must keep at least one theme.", parent=win)
                return
            if messagebox.askyesno("Delete Theme", f"Delete color setup '{name}'?", parent=win):
                del self.ui_themes[name]
                self.active_theme_name = next(iter(self.ui_themes.keys()))
                self.apply_ui_theme(self.ui_themes[self.active_theme_name])
                self._save_themes_to_disk()
                theme_cb.config(values=list(self.ui_themes.keys()))
                theme_var.set(self.active_theme_name)
                _refresh_swatches()

        ttk.Button(top_row, text="Save", width=5, command=_save_theme).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(top_row, text="Del", width=4, command=_delete_theme).pack(side=tk.RIGHT)

    def _bind_shortcuts(self):
        self.root.bind_all("<Control-Shift-Z>", self.redo)
        self.root.bind_all("<Control-Shift-z>", self.redo)
        self.root.bind_all("<Control-z>", self._handle_ctrl_z)
        self.root.bind_all("<Control-Z>", self._handle_ctrl_z)
        self.root.bind_all("<space>", self._handle_spacebar)

    def _handle_ctrl_z(self, event):
        if event.state & 0x0001:
            self.redo()
        else:
            self.undo()

    def _handle_spacebar(self, event):
        # Do not intercept spacebar when typing in an Entry or Combobox
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        state = self.current_media_state
        if state is not None and state.is_multiframe and not self._is_worker_busy:
            self.toggle_playback()

    def _build_ui(self):
        self.sidebar = ttk.Frame(self.root, width=395, padding=10)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.main_area = ttk.Frame(self.root, padding=10)
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. BASE IMAGE / VIDEO & CANVAS SIZE
        orig_frame = ttk.LabelFrame(self.sidebar, text="1. Base Image / Video & Canvas Size", padding=8)
        orig_frame.pack(fill=tk.X, pady=(0, 6))

        self.browse_base_btn = ttk.Button(
            orig_frame, text="Browse Base Image / Video...", command=self.load_original_image
        )
        self.browse_base_btn.pack(fill=tk.X)

        dim_row = ttk.Frame(orig_frame)
        dim_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(dim_row, text="Max Dim (0 = Full):").pack(side=tk.LEFT)
        self.orig_max_dim_var = tk.StringVar(value="0")
        ttk.Entry(dim_row, textvariable=self.orig_max_dim_var, width=8).pack(side=tk.LEFT, padx=4)
        self.resize_curr_btn = ttk.Button(
            dim_row, text="Resize Current", command=self.resize_current_canvas_image
        )
        self.resize_curr_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 2. PALETTE SETUP (IMAGES ONLY)
        pal_frame = ttk.LabelFrame(self.sidebar, text="2. Palette Setup (Image Only)", padding=8)
        pal_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(pal_frame, text="Browse Palette Image...", command=self.load_palette_image).pack(fill=tk.X)

        pal_ctrl_row = ttk.Frame(pal_frame)
        pal_ctrl_row.pack(fill=tk.X, pady=4)
        ttk.Label(pal_ctrl_row, text="Samples / Dim (0 = Full):").pack(side=tk.LEFT)
        self.pal_max_dim_var = tk.StringVar(value="0")
        ttk.Entry(pal_ctrl_row, textvariable=self.pal_max_dim_var, width=8).pack(side=tk.RIGHT)

        # Extract Palette & Cancel Buttons
        pal_btn_row = ttk.Frame(pal_frame)
        pal_btn_row.pack(fill=tk.X, pady=(2, 2))

        self.extract_pal_btn = ttk.Button(pal_btn_row, text="Extract Palette", command=self.start_palette_extract_thread)
        self.extract_pal_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.cancel_pal_btn = ttk.Button(pal_btn_row, text="Cancel", command=self.cancel_palette_task, state=tk.DISABLED)
        self.cancel_pal_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Palette Extraction Progress Bar & ETA Label
        self.pal_progress_var = tk.DoubleVar(value=0.0)
        self.pal_progress_bar = ttk.Progressbar(pal_frame, variable=self.pal_progress_var, maximum=100.0, mode="determinate")
        self.pal_progress_bar.pack(fill=tk.X, pady=(2, 1))

        self.pal_progress_label = ttk.Label(pal_frame, text="Ready", font=("", 8), anchor=tk.CENTER)
        self.pal_progress_label.pack(fill=tk.X, pady=(0, 4))

        thumbs_row = ttk.Frame(pal_frame)
        thumbs_row.pack(fill=tk.X)

        src_col = ttk.Frame(thumbs_row)
        src_col.pack(side=tk.LEFT, expand=True)
        self.pal_source_title_lbl = ttk.Label(src_col, text="Palette Source", font=("", 8))
        self.pal_source_title_lbl.pack()
        self.pal_source_box = tk.Frame(src_col, width=THUMB_W, height=THUMB_H, bg="#2b2b2b", relief=tk.SUNKEN, bd=1)
        self.pal_source_box.pack(pady=2)
        self.pal_source_box.pack_propagate(False)
        self.pal_source_thumb = tk.Label(self.pal_source_box, text="No Image\n(Click to Zoom)", bg="#2b2b2b", fg="#888888", cursor="hand2")
        self.pal_source_thumb.pack(fill=tk.BOTH, expand=True)
        self.pal_source_thumb.bind("<Button-1>", lambda e: self.open_popup(self.palette_image, "Palette Source Image"))

        sq_col = ttk.Frame(thumbs_row)
        sq_col.pack(side=tk.RIGHT, expand=True)
        self.pal_square_title_lbl = ttk.Label(sq_col, text="Extracted Square", font=("", 8))
        self.pal_square_title_lbl.pack()
        self.pal_square_box = tk.Frame(sq_col, width=THUMB_W, height=THUMB_H, bg="#2b2b2b", relief=tk.SUNKEN, bd=1)
        self.pal_square_box.pack(pady=2)
        self.pal_square_box.pack_propagate(False)
        self.pal_square_thumb = tk.Label(self.pal_square_box, text="Not Extracted\n(Click to Zoom)", bg="#2b2b2b", fg="#888888", cursor="hand2")
        self.pal_square_thumb.pack(fill=tk.BOTH, expand=True)
        self.pal_square_thumb.bind("<Button-1>", lambda e: self.open_popup(self.palette_square_img, "Extracted Palette Square", nearest=True))

        # 3. COLOR SPACE, WEIGHT SLIDERS & DITHERING
        param_frame = ttk.LabelFrame(self.sidebar, text="3. Color Space, Weights & Dithering", padding=8)
        param_frame.pack(fill=tk.X, pady=(0, 6))

        # Preset Manager Row
        preset_row = ttk.Frame(param_frame)
        preset_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(preset_row, text="Preset:").pack(side=tk.LEFT)

        self.preset_var = tk.StringVar()
        self.preset_dropdown = ttk.Combobox(
            preset_row,
            textvariable=self.preset_var,
            values=list(self.presets.keys()),
            width=14
        )
        self.preset_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.preset_dropdown.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Button(preset_row, text="Save", width=5, command=self.save_current_preset).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(preset_row, text="Del", width=4, command=self.delete_selected_preset).pack(side=tk.RIGHT)

        cs_row = ttk.Frame(param_frame)
        cs_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(cs_row, text="Color Space:").pack(side=tk.LEFT)
        self.color_space_var = tk.StringVar(value="LAB")
        cs_dropdown = ttk.Combobox(cs_row, textvariable=self.color_space_var, values=list(self.color_spaces.keys()), state="readonly", width=18)
        cs_dropdown.pack(side=tk.RIGHT)

        self.w1_var = tk.DoubleVar(value=1.00)
        self.w2_var = tk.DoubleVar(value=0.25)
        self.w3_var = tk.DoubleVar(value=0.25)

        self._create_slider_row(param_frame, "Ch 1 Weight:", self.w1_var, 0.0, 2.0)
        self._create_slider_row(param_frame, "Ch 2 Weight:", self.w2_var, 0.0, 2.0)
        self._create_slider_row(param_frame, "Ch 3 Weight:", self.w3_var, 0.0, 2.0)

        ttk.Separator(param_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        dither_row = ttk.Frame(param_frame)
        dither_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(dither_row, text="Dithering:").pack(side=tk.LEFT)
        self.dither_type_var = tk.StringVar(value="None")
        dither_dropdown = ttk.Combobox(dither_row, textvariable=self.dither_type_var, values=self.dither_modes, state="readonly", width=18)
        dither_dropdown.pack(side=tk.RIGHT)

        self.dither_strength_var = tk.DoubleVar(value=16.0)
        self._create_slider_row(param_frame, "Dither Spread:", self.dither_strength_var, 0.0, 64.0, trough_step=4.0)

        # Action & Progress Buttons Container
        btn_row = ttk.Frame(param_frame)
        btn_row.pack(fill=tk.X, pady=(5, 2))

        self.run_btn = ttk.Button(btn_row, text="Apply Palette", command=self.start_palletizer_thread)
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.cancel_btn = ttk.Button(btn_row, text="Cancel", command=self.cancel_task, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Progress Bar 1: Overall Progress (Current Object in Main Canvas)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(param_frame, variable=self.progress_var, maximum=100.0, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(2, 1))

        self.progress_label = ttk.Label(param_frame, text="Overall: Ready", font=("", 8), anchor=tk.CENTER)
        self.progress_label.pack(fill=tk.X, pady=(0, 2))

        # Progress Bar 2: Individual Video/GIF Frame Progress
        self.frame_progress_var = tk.DoubleVar(value=0.0)
        self.frame_progress_bar = ttk.Progressbar(
            param_frame, variable=self.frame_progress_var, maximum=100.0, mode="determinate"
        )
        self.frame_progress_bar.pack(fill=tk.X, pady=(1, 1))

        self.frame_progress_label = ttk.Label(
            param_frame, text="Frame: Ready", font=("", 8), anchor=tk.CENTER
        )
        self.frame_progress_label.pack(fill=tk.X, pady=(0, 4))

        self.save_btn = ttk.Button(
            param_frame, text="Save Current Main Image / Video As...", command=self.save_result_manual
        )
        self.save_btn.pack(fill=tk.X)

        # 4. HISTORY & PREVIOUS MEDIA SECTION
        self.prev_frame = ttk.LabelFrame(self.sidebar, text="4. History & Previous State (Ctrl+Z)", padding=8)
        self.prev_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        hist_bar = ttk.Frame(self.prev_frame)
        hist_bar.pack(fill=tk.X, pady=(0, 6))

        self.undo_btn = ttk.Button(hist_bar, text="Undo (Ctrl+Z)", command=self.undo, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        self.hist_status_lbl = ttk.Label(hist_bar, text="0 / 0", width=7, anchor=tk.CENTER)
        self.hist_status_lbl.pack(side=tk.LEFT, padx=2)

        self.redo_btn = ttk.Button(hist_bar, text="Redo (Ctrl+Shift+Z)", command=self.redo, state=tk.DISABLED)
        self.redo_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))

        self.prev_image_box = tk.Frame(self.prev_frame, bg="#2b2b2b", relief=tk.SUNKEN, bd=1)
        self.prev_image_box.pack(fill=tk.BOTH, expand=True)
        self.prev_image_box.pack_propagate(False)

        self.prev_image_thumb = tk.Label(
            self.prev_image_box,
            text="No previous state\nin history",
            bg="#2b2b2b",
            fg="#888888",
            cursor="hand2"
        )
        self.prev_image_thumb.pack(fill=tk.BOTH, expand=True)
        self.prev_image_thumb.bind(
            "<Button-1>",
            lambda e: self.open_popup(
                self.previous_media_state,
                "Previous History State (Ctrl+Z)",
                start_frame_idx=self.current_frame_idx,
            ),
        )
        self.prev_image_thumb.bind("<Configure>", self._on_prev_thumb_resize)

        # UI THEME SETTINGS BUTTON
        ttk.Button(
            self.sidebar, text="UI Theme Settings...", command=self.open_theme_settings
        ).pack(fill=tk.X, pady=(0, 4))

        # 5. MAIN CANVAS SECTION (Interactive Zoom, Pan & Video Playback)
        self.main_frame = ttk.LabelFrame(
            self.main_area,
            text="Main Visualization Canvas (Scroll to Zoom, Drag to Pan, Click to Popup)",
            padding=8
        )
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Video / Animated GIF Playback Controls Bar (docked at bottom of Main Visualization Canvas)
        self.playback_frame = ttk.Frame(self.main_frame, padding=(2, 6, 2, 0))
        self.playback_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.play_pause_btn = ttk.Button(
            self.playback_frame, text="▶ Play", width=9, command=self.toggle_playback, state=tk.DISABLED
        )
        self.play_pause_btn.pack(side=tk.LEFT, padx=(0, 3))

        self.prev_frame_btn = ttk.Button(
            self.playback_frame, text="◀ Frame", width=8, command=self.step_prev_frame, state=tk.DISABLED
        )
        self.prev_frame_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.next_frame_btn = ttk.Button(
            self.playback_frame, text="Frame ▶", width=8, command=self.step_next_frame, state=tk.DISABLED
        )
        self.next_frame_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_slider = ttk.Scale(
            self.playback_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.timeline_var,
            command=self._on_timeline_scrub,
        )
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.timeline_slider.bind("<ButtonPress-1>", self._on_timeline_trough_click)
        self.timeline_slider.state(["disabled"])

        self.frame_info_label = ttk.Label(
            self.playback_frame, text="No Media Loaded", width=30, anchor=tk.E, font=("", 8)
        )
        self.frame_info_label.pack(side=tk.RIGHT, padx=(6, 0))

        self.main_canvas = tk.Canvas(self.main_frame, bg="#1e1e1e", highlightthickness=0, cursor="hand2")
        self.main_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bindings for Resize, Scroll-Zoom, Panning, and Click-Popup
        self.main_canvas.bind("<Configure>", self._on_main_canvas_resize)
        self.main_canvas.bind("<MouseWheel>", self._on_main_canvas_scroll)       # Windows / macOS
        self.main_canvas.bind("<Button-4>", self._on_main_canvas_scroll)         # Linux Scroll Up
        self.main_canvas.bind("<Button-5>", self._on_main_canvas_scroll)         # Linux Scroll Down
        self.main_canvas.bind("<ButtonPress-1>", self._on_canvas_mouse_down)
        self.main_canvas.bind("<B1-Motion>", self._on_canvas_mouse_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self._on_canvas_mouse_up)

    # ==========================================
    # VIDEO & GIF PLAYBACK CONTROLS
    # ==========================================
    def toggle_playback(self):
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        state = self.current_media_state
        if state is None or not state.is_multiframe or self._is_worker_busy:
            return
        if state.frame_count <= 1:
            return

        self.is_playing = True
        self.play_pause_btn.config(text="⏸ Pause")
        self._schedule_next_playback_frame(state.get_frame_delay_ms(self.current_frame_idx))

    def pause_playback(self):
        self.is_playing = False
        if self._playback_after_id is not None:
            try:
                self.root.after_cancel(self._playback_after_id)
            except Exception:
                pass
            self._playback_after_id = None
        if hasattr(self, "play_pause_btn"):
            self.play_pause_btn.config(text="▶ Play")

    def _schedule_next_playback_frame(self, delay_ms: int):
        if self._playback_after_id is not None:
            try:
                self.root.after_cancel(self._playback_after_id)
            except Exception:
                pass
        self._playback_after_id = self.root.after(max(10, int(delay_ms)), self._playback_tick)

    def _playback_tick(self):
        self._playback_after_id = None
        if not self.is_playing or self._is_worker_busy:
            self.pause_playback()
            return

        state = self.current_media_state
        if state is None or not state.is_multiframe or state.frame_count <= 1:
            self.pause_playback()
            return

        t0 = time.perf_counter()
        next_idx = (self.current_frame_idx + 1) % state.frame_count
        self._show_media_frame(next_idx)

        target_delay = state.get_frame_delay_ms(self.current_frame_idx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        wait_ms = max(10, int(round(target_delay - elapsed_ms)))
        if self.is_playing:
            self._schedule_next_playback_frame(wait_ms)

    def step_prev_frame(self):
        state = self.current_media_state
        if state is None or not state.is_multiframe or self._is_worker_busy:
            return
        self.pause_playback()
        prev_idx = (self.current_frame_idx - 1) % max(1, state.frame_count)
        self._show_media_frame(prev_idx)

    def step_next_frame(self):
        state = self.current_media_state
        if state is None or not state.is_multiframe or self._is_worker_busy:
            return
        self.pause_playback()
        next_idx = (self.current_frame_idx + 1) % max(1, state.frame_count)
        self._show_media_frame(next_idx)

    def _on_timeline_trough_click(self, event):
        state = self.current_media_state
        if state is None or not state.is_multiframe or state.frame_count <= 1 or self._is_worker_busy:
            return
        if "slider" in str(self.timeline_slider.identify(event.x, event.y)):
            return
        width = max(1, self.timeline_slider.winfo_width())
        frac = max(0.0, min(1.0, event.x / float(width)))
        target_idx = int(round(frac * (state.frame_count - 1)))
        self.pause_playback()
        self._show_media_frame(target_idx)
        return "break"

    def _on_timeline_scrub(self, val_str):
        if self._updating_timeline_programmatically or self._is_worker_busy:
            return
        state = self.current_media_state
        if state is None or not state.is_multiframe or state.frame_count <= 1:
            return
        try:
            target_idx = int(round(float(val_str)))
        except ValueError:
            return
        target_idx = max(0, min(target_idx, state.frame_count - 1))
        if target_idx != self.current_frame_idx:
            self._show_media_frame(target_idx)

    def _show_media_frame(self, frame_idx: int):
        state = self.current_media_state
        if state is None:
            return
        frame_idx = max(0, min(int(frame_idx), state.frame_count - 1))
        self.current_frame_idx = frame_idx
        self.main_canvas_image = state.get_frame(frame_idx)
        self._update_playback_controls_ui()
        self._render_main_canvas()

    def _update_playback_controls_ui(self, rendering_info: tuple[int, int] | None = None):
        if not hasattr(self, "play_pause_btn"):
            return

        if rendering_info is not None:
            cur_f, tot_f = rendering_info
            self.play_pause_btn.config(state=tk.DISABLED)
            self.prev_frame_btn.config(state=tk.DISABLED)
            self.next_frame_btn.config(state=tk.DISABLED)
            self.timeline_slider.state(["disabled"])
            self._updating_timeline_programmatically = True
            try:
                self.timeline_slider.config(from_=0.0, to=float(max(1, tot_f - 1)))
                self.timeline_var.set(float(cur_f))
            finally:
                self._updating_timeline_programmatically = False
            self.frame_info_label.config(text=f"Rendering Frame {cur_f + 1} / {tot_f}")
            return

        state = self.current_media_state
        if state is None:
            self.play_pause_btn.config(state=tk.DISABLED, text="▶ Play")
            self.prev_frame_btn.config(state=tk.DISABLED)
            self.next_frame_btn.config(state=tk.DISABLED)
            self.timeline_slider.state(["disabled"])
            self.frame_info_label.config(text="No Media Loaded")
            return

        if not state.is_multiframe or state.frame_count <= 1:
            self.pause_playback()
            self.play_pause_btn.config(state=tk.DISABLED, text="▶ Play")
            self.prev_frame_btn.config(state=tk.DISABLED)
            self.next_frame_btn.config(state=tk.DISABLED)
            self.timeline_slider.state(["disabled"])
            self._updating_timeline_programmatically = True
            try:
                self.timeline_slider.config(from_=0.0, to=1.0)
                self.timeline_var.set(0.0)
            finally:
                self._updating_timeline_programmatically = False
            tag = "GIF (1 frame)" if state.kind == "gif" else "Static Image (1 / 1)"
            self.frame_info_label.config(text=tag)
            return

        # Multi-frame video or animated GIF
        ctrl_state = tk.DISABLED if self._is_worker_busy else tk.NORMAL
        self.play_pause_btn.config(state=ctrl_state, text="⏸ Pause" if self.is_playing else "▶ Play")
        self.prev_frame_btn.config(state=ctrl_state)
        self.next_frame_btn.config(state=ctrl_state)
        if self._is_worker_busy:
            self.timeline_slider.state(["disabled"])
        else:
            self.timeline_slider.state(["!disabled"])

        max_idx = max(1, state.frame_count - 1)
        self._updating_timeline_programmatically = True
        try:
            self.timeline_slider.config(from_=0.0, to=float(max_idx))
            self.timeline_var.set(float(min(self.current_frame_idx, max_idx)))
        finally:
            self._updating_timeline_programmatically = False

        fps_val = max(1.0, state.fps)
        cur_sec = self.current_frame_idx / fps_val
        tot_sec = state.frame_count / fps_val
        kind_tag = "GIF" if state.kind == "gif" else "VID"
        self.frame_info_label.config(
            text=f"[{kind_tag}] Frame {self.current_frame_idx + 1}/{state.frame_count} ({cur_sec:.1f}s/{tot_sec:.1f}s • {fps_val:.1f}fps)"
        )

    # ==========================================
    # MAIN CANVAS INTERACTIVE ZOOM & PAN
    # ==========================================
    def _update_main_canvas_title(self):
        if not hasattr(self, "main_frame"):
            return
        if self.main_canvas_image is not None:
            ih, iw = self.main_canvas_image.shape[:2]
            zoom_pct = int(round(self.zoom_factor * 100))
            state = self.current_media_state
            if state is not None and state.is_multiframe:
                kind_label = "GIF" if state.kind == "gif" else "Video"
                self.main_frame.config(
                    text=(
                        f"Main Visualization Canvas — {kind_label} ({iw}x{ih}, "
                        f"Frame {self.current_frame_idx + 1}/{state.frame_count}) — "
                        f"Zoom: {zoom_pct}% (Scroll to Zoom, Drag to Pan, Click to Popup)"
                    )
                )
            else:
                self.main_frame.config(
                    text=f"Main Visualization Canvas ({iw}x{ih}) — Zoom: {zoom_pct}% (Scroll to Zoom, Drag to Pan, Click to Popup)"
                )
        else:
            self.main_frame.config(
                text="Main Visualization Canvas (Scroll to Zoom, Drag to Pan, Click to Popup)"
            )

    def _get_base_fit_scale(self, cw, ch, iw, ih):
        avail_w = max(cw - 10, 50)
        avail_h = max(ch - 10, 50)
        return min(avail_w / iw, avail_h / ih)

    def _clamp_canvas_offsets(self, cw, ch, disp_w, disp_h):
        if disp_w <= cw:
            self.img_offset_x = (cw - disp_w) / 2.0
        else:
            self.img_offset_x = min(0.0, max(cw - disp_w, self.img_offset_x))

        if disp_h <= ch:
            self.img_offset_y = (ch - disp_h) / 2.0
        else:
            self.img_offset_y = min(0.0, max(ch - disp_h, self.img_offset_y))

    def _render_main_canvas(self):
        self._update_main_canvas_title()
        self.main_canvas.delete("all")
        cw = self.main_canvas.winfo_width()
        ch = self.main_canvas.winfo_height()

        if self.main_canvas_image is None:
            if cw > 50 and ch > 50:
                self.main_canvas.create_text(
                    cw // 2, ch // 2,
                    text="Select a Base Image or Video to begin",
                    fill="#aaaaaa", font=("", 10)
                )
            return

        if cw < 50 or ch < 50:
            return

        arr = self._ensure_uint8(self.main_canvas_image)
        ih, iw = arr.shape[:2]

        base_scale = self._get_base_fit_scale(cw, ch, iw, ih)
        eff_scale = base_scale * self.zoom_factor

        disp_w = iw * eff_scale
        disp_h = ih * eff_scale

        self._clamp_canvas_offsets(cw, ch, disp_w, disp_h)

        vis_x0 = max(0.0, self.img_offset_x)
        vis_y0 = max(0.0, self.img_offset_y)
        vis_x1 = min(float(cw), self.img_offset_x + disp_w)
        vis_y1 = min(float(ch), self.img_offset_y + disp_h)

        if vis_x1 <= vis_x0 or vis_y1 <= vis_y0:
            return

        src_x0 = max(0, int(np.floor((vis_x0 - self.img_offset_x) / eff_scale)))
        src_y0 = max(0, int(np.floor((vis_y0 - self.img_offset_y) / eff_scale)))
        src_x1 = min(iw, int(np.ceil((vis_x1 - self.img_offset_x) / eff_scale)))
        src_y1 = min(ih, int(np.ceil((vis_y1 - self.img_offset_y) / eff_scale)))

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return

        cropped_arr = arr[src_y0:src_y1, src_x0:src_x1]
        target_w = max(1, int(round((src_x1 - src_x0) * eff_scale)))
        target_h = max(1, int(round((src_y1 - src_y0) * eff_scale)))

        resample_mode = Image.Resampling.NEAREST if eff_scale > 1.5 else Image.Resampling.LANCZOS
        pil_patch = Image.fromarray(cropped_arr).resize((target_w, target_h), resample_mode)

        draw_x = int(round(self.img_offset_x + src_x0 * eff_scale))
        draw_y = int(round(self.img_offset_y + src_y0 * eff_scale))

        self._main_tk_img = ImageTk.PhotoImage(pil_patch)
        self.main_canvas.create_image(draw_x, draw_y, anchor=tk.NW, image=self._main_tk_img)

    def _on_main_canvas_scroll(self, event):
        if self.main_canvas_image is None:
            return

        cw = self.main_canvas.winfo_width()
        ch = self.main_canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            step = 1.2
        elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            step = 1.0 / 1.2
        else:
            return

        old_zoom = self.zoom_factor
        new_zoom = max(1.0, min(self.max_zoom, old_zoom * step))

        if new_zoom < 1.02:
            new_zoom = 1.0

        if new_zoom == old_zoom:
            return

        ih, iw = self.main_canvas_image.shape[:2]
        base_scale = self._get_base_fit_scale(cw, ch, iw, ih)
        old_eff_scale = base_scale * old_zoom
        new_eff_scale = base_scale * new_zoom

        mx, my = event.x, event.y
        img_x = (mx - self.img_offset_x) / old_eff_scale
        img_y = (my - self.img_offset_y) / old_eff_scale

        self.zoom_factor = new_zoom
        self.img_offset_x = mx - img_x * new_eff_scale
        self.img_offset_y = my - img_y * new_eff_scale

        self._render_main_canvas()

    def _on_canvas_mouse_down(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._did_drag = False

    def _on_canvas_mouse_drag(self, event):
        if self.main_canvas_image is None:
            return

        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y

        if abs(dx) > 3 or abs(dy) > 3:
            self._did_drag = True

        if self.zoom_factor > 1.0:
            self.img_offset_x += dx
            self.img_offset_y += dy
            self._drag_start_x = event.x
            self._drag_start_y = event.y
            self._render_main_canvas()

    def _on_canvas_mouse_up(self, event):
        if not self._did_drag and self.main_canvas_image is not None:
            if self._is_worker_busy:
                self.open_popup(self.main_canvas_image, "Main Canvas Live Frame Preview")
            else:
                target = self.current_media_state if self.current_media_state is not None else self.main_canvas_image
                self.open_popup(target, "Main Canvas Media", start_frame_idx=self.current_frame_idx)

    def _reset_zoom(self):
        self.zoom_factor = 1.0
        self.img_offset_x = 0.0
        self.img_offset_y = 0.0
        self._update_main_canvas_title()

    def _create_slider_row(self, parent, label_text, double_var, min_val, max_val, trough_step=0.25):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label_text, width=13).pack(side=tk.LEFT)

        entry_str = tk.StringVar(value=f"{double_var.get():.2f}")
        entry = ttk.Entry(row, textvariable=entry_str, width=6)
        entry.pack(side=tk.RIGHT, padx=(4, 0))

        def _on_var_write(*_args):
            try:
                entry_str.set(f"{float(double_var.get()):.2f}")
            except (ValueError, tk.TclError):
                pass

        # Automatically syncs text box whenever slider moves OR preset is loaded
        double_var.trace_add("write", _on_var_write)

        slider = ttk.Scale(row, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=double_var)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        def _on_trough_click(event):
            # If the user clicked directly on the knob ('slider'), allow normal smooth dragging
            if "slider" in str(slider.identify(event.x, event.y)):
                return

            # If the user clicked on the bar (trough), step by trough_step (0.25)
            width = max(1, slider.winfo_width())
            click_val = min_val + (event.x / width) * (max_val - min_val)
            current = float(double_var.get())

            if click_val >= current:
                new_val = min(max_val, current + trough_step)
            else:
                new_val = max(min_val, current - trough_step)

            double_var.set(round(new_val, 4))
            return "break"  # Prevents Tkinter's default +/- 1.0 trough jump

        slider.bind("<ButtonPress-1>", _on_trough_click)

        def _on_entry_commit(event=None):
            try:
                val = float(entry_str.get())
                double_var.set(val)
            except ValueError:
                entry_str.set(f"{double_var.get():.2f}")

        entry.bind("<Return>", _on_entry_commit)
        entry.bind("<FocusOut>", _on_entry_commit)

    # ==========================================
    # PALETTE EXTRACTION THREADING & PROGRESS
    # ==========================================
    def _set_pal_progress_safe(self, task_id: int, percent: float, status_text: str):
        if task_id != self.active_pal_task_id:
            return
        self.pal_progress_var.set(percent)
        self.pal_progress_label.config(text=f"{percent:.0f}% — {status_text}")

    def cancel_palette_task(self):
        self.pal_cancel_requested = True
        self.active_pal_task_id += 1
        self.extract_pal_btn.config(state=tk.NORMAL)
        self.cancel_pal_btn.config(state=tk.DISABLED)
        if not self._is_worker_busy:
            self.root.config(cursor="")
        self.pal_progress_var.set(0.0)
        self.pal_progress_label.config(text="0% — Cancelled")

    def start_palette_extract_thread(self):
        if self.palette_image is None:
            messagebox.showwarning("Missing Palette Image", "Please select a Palette Image first.")
            return

        try:
            max_dim = self._parse_dimension(self.pal_max_dim_var.get(), "Palette Max Dimension / Samples")
        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))
            return

        self.pal_cancel_requested = False
        self.active_pal_task_id += 1
        task_id = self.active_pal_task_id

        # Shared state polled strictly by the main thread
        self._pal_shared_state = {
            "pct": 0.0,
            "status": "Starting extraction...",
            "done": False,
            "error": None,
            "palette": None,
            "sq_img": None,
        }

        self.extract_pal_btn.config(state=tk.DISABLED)
        self.cancel_pal_btn.config(state=tk.NORMAL)
        self.root.config(cursor="watch")
        self._set_pal_progress_safe(task_id, 0.0, "Starting extraction...")

        threading.Thread(
            target=self._run_palette_extract_worker, args=(task_id, max_dim), daemon=True
        ).start()
        self.root.after(40, lambda: self._poll_pal_worker(task_id))

    def _run_palette_extract_worker(self, task_id: int, max_dim: int):
        try:
            pal_img_copy = self.palette_image.copy()

            def progress_cb(pct, status):
                if task_id == self.active_pal_task_id:
                    self._pal_shared_state["pct"] = pct
                    self._pal_shared_state["status"] = status

            def is_cancelled():
                return self.pal_cancel_requested or (task_id != self.active_pal_task_id)

            extracted_pal = image_to_palette(
                pal_img_copy,
                max_dim,
                progress_callback=progress_cb,
                cancel_check=is_cancelled,
            )

            if extracted_pal is not None and not is_cancelled():
                sq_img = palette_square_padded(extracted_pal)
                self._pal_shared_state["palette"] = extracted_pal
                self._pal_shared_state["sq_img"] = sq_img

            self._pal_shared_state["done"] = True
        except Exception as e:
            self._pal_shared_state["error"] = str(e)
            self._pal_shared_state["done"] = True

    def _poll_pal_worker(self, task_id: int):
        if task_id != self.active_pal_task_id:
            return

        st = self._pal_shared_state
        self._set_pal_progress_safe(task_id, st["pct"], st["status"])

        if not st["done"]:
            self.root.after(40, lambda: self._poll_pal_worker(task_id))
            return

        if st["error"]:
            self._handle_pal_worker_error(task_id, st["error"])
        else:
            self._finalize_palette_extract(task_id, st["palette"], st["sq_img"])

    def _finalize_palette_extract(self, task_id: int, extracted_pal, sq_img):
        if task_id != self.active_pal_task_id:
            return

        self.extract_pal_btn.config(state=tk.NORMAL)
        self.cancel_pal_btn.config(state=tk.DISABLED)
        if not self._is_worker_busy:
            self.root.config(cursor="")

        if extracted_pal is not None and sq_img is not None and not self.pal_cancel_requested:
            self.palette = extracted_pal
            self.palette_square_img = sq_img
            num_colors = len(self.palette)
            self.pal_square_title_lbl.config(text=f"Extracted ({num_colors} px)")
            self._display_on_label(self.pal_square_thumb, self.palette_square_img, THUMB_W - 6, THUMB_H - 6, nearest=True)
            self._set_pal_progress_safe(task_id, 100.0, f"Done ({num_colors} colors)")
        else:
            self._set_pal_progress_safe(task_id, 0.0, "Cancelled")

    def _handle_pal_worker_error(self, task_id: int, err_msg: str):
        if task_id != self.active_pal_task_id:
            return
        self.extract_pal_btn.config(state=tk.NORMAL)
        self.cancel_pal_btn.config(state=tk.DISABLED)
        if not self._is_worker_busy:
            self.root.config(cursor="")
        self._set_pal_progress_safe(task_id, 0.0, "Error")
        messagebox.showerror("Palette Extraction Error", err_msg)

    # ==========================================
    # MAIN PALLETIZER & VIDEO WORKER THREADING
    # ==========================================
    def _set_worker_busy_ui(self, busy: bool):
        self._is_worker_busy = busy
        if busy:
            self.pause_playback()
            self.run_btn.config(state=tk.DISABLED)
            self.cancel_btn.config(state=tk.NORMAL)
            self.browse_base_btn.config(state=tk.DISABLED)
            self.resize_curr_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.undo_btn.config(state=tk.DISABLED)
            self.redo_btn.config(state=tk.DISABLED)
            self.root.config(cursor="watch")
        else:
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            self.browse_base_btn.config(state=tk.NORMAL)
            self.resize_curr_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            total = len(self.history)
            self.undo_btn.config(state=tk.NORMAL if self.history_index > 0 else tk.DISABLED)
            self.redo_btn.config(state=tk.NORMAL if self.history_index < total - 1 else tk.DISABLED)
            self.root.config(cursor="")
        self._update_playback_controls_ui()

    def _set_progress_safe(
        self,
        task_id: int,
        overall_pct: float,
        overall_status: str,
        frame_pct: float = 0.0,
        frame_status: str | None = None,
    ):
        if task_id != self.active_task_id:
            return
        self.progress_var.set(overall_pct)
        self.progress_label.config(text=f"Overall: {overall_pct:.0f}% — {overall_status}")

        if frame_status is not None:
            self.frame_progress_var.set(frame_pct)
            self.frame_progress_label.config(text=f"Frame: {frame_pct:.0f}% — {frame_status}")

    def cancel_task(self):
        self.cancel_requested = True
        self.active_task_id += 1
        self._set_worker_busy_ui(False)
        self.progress_var.set(0.0)
        self.progress_label.config(text="Overall: 0% — Cancelled")
        self.frame_progress_var.set(0.0)
        self.frame_progress_label.config(text="Frame: 0% — Cancelled")

        # Restore main canvas to active history state if a partial video render was previewed
        if self.current_media_state is not None:
            if (
                self.main_canvas_image is not None
                and self.main_canvas_image.shape[:2] != self.current_media_state.shape[:2]
            ):
                self._reset_zoom()
            self._show_media_frame(min(self.current_frame_idx, self.current_media_state.frame_count - 1))

    def start_palletizer_thread(self):
        active_state = self.current_media_state
        if active_state is None or self.main_canvas_image is None:
            messagebox.showwarning("Missing Canvas Media", "Please load a Base Image or Video onto the Main Canvas first.")
            return
        if self.palette is None:
            messagebox.showwarning("Missing Extracted Palette", "Please load a palette image and click 'Extract Palette' first.")
            return

        self.cancel_requested = False
        self.active_task_id += 1
        task_id = self.active_task_id
        self._last_previewed_frame_idx = -1

        is_multi = active_state.is_multiframe
        init_frame_status = "Starting frame 1..." if is_multi else "N/A (Static Image)"

        self._main_shared_state = {
            "pct": 0.0,
            "status": "Starting...",
            "frame_pct": 0.0,
            "frame_status": init_frame_status,
            "is_multiframe": is_multi,
            "latest_frame": None,
            "latest_frame_idx": -1,
            "total_frames": active_state.frame_count,
            "done": False,
            "error": None,
            "result_state": None,
        }

        self._set_worker_busy_ui(True)
        self._set_progress_safe(task_id, 0.0, "Starting...", 0.0, init_frame_status)

        threading.Thread(
            target=self._run_palletizer_worker,
            args=(task_id, active_state),
            daemon=True,
        ).start()
        self.root.after(40, lambda: self._poll_main_worker(task_id))

    def _run_palletizer_worker(self, task_id: int, source_state: MediaState):
        out_folder = None
        try:
            w1 = float(self.w1_var.get())
            w2 = float(self.w2_var.get())
            w3 = float(self.w3_var.get())
            color_space_code = self.color_spaces[self.color_space_var.get()]
            dither_type = self.dither_type_var.get()
            dither_strength = float(self.dither_strength_var.get())
            palette_copy = self.palette.copy()

            def is_cancelled():
                return self.cancel_requested or (task_id != self.active_task_id)

            # -------------------------------------------------
            # CASE 1: SINGLE-FRAME STATIC IMAGE (OR 1-FRAME GIF)
            # -------------------------------------------------
            if not source_state.is_multiframe:
                source_canvas_img = self._ensure_uint8(source_state.get_frame(0)).copy()

                def single_progress_cb(pct, status):
                    if task_id == self.active_task_id:
                        self._main_shared_state["pct"] = pct
                        self._main_shared_state["status"] = status

                result_img = palletized_colorspace_fast(
                    source_canvas_img,
                    palette_copy,
                    25000,
                    color_space_code,
                    w1,
                    w2,
                    w3,
                    dither_type=dither_type,
                    dither_strength=dither_strength,
                    progress_callback=single_progress_cb,
                    cancel_check=is_cancelled,
                )

                if result_img is not None and not is_cancelled():
                    new_state = MediaState.from_image(
                        uid=self._next_state_uid(),
                        image_rgb=self._ensure_uint8(result_img),
                        kind=source_state.kind,
                    )
                    self._main_shared_state["result_state"] = new_state

                self._main_shared_state["done"] = True
                return

            # -------------------------------------------------
            # CASE 2: MULTI-FRAME VIDEO OR ANIMATED GIF
            # -------------------------------------------------
            total_frames = max(1, source_state.frame_count)
            state_uid = self._next_state_uid()
            out_folder = os.path.join(self.temp_dir, f"state_{state_uid}")
            os.makedirs(out_folder, exist_ok=True)

            video_start_time = time.time()
            actual_written = 0
            first_thumb = None

            for frame_idx, frame_rgb in source_state.iter_frames_for_worker():
                if is_cancelled():
                    break

                self._main_shared_state["frame_pct"] = 0.0
                self._main_shared_state["frame_status"] = f"({frame_idx + 1}/{total_frames}) Starting..."

                def frame_progress_cb(f_pct, f_status, idx=frame_idx):
                    if task_id != self.active_task_id:
                        return
                    self._main_shared_state["frame_pct"] = f_pct
                    self._main_shared_state["frame_status"] = f"({idx + 1}/{total_frames}) {f_status}"

                    overall_frac = (idx + (f_pct / 100.0)) / float(total_frames)
                    overall_pct = min(99.9, overall_frac * 100.0)
                    elapsed = time.time() - video_start_time
                    overall_eta = (
                        (elapsed / overall_frac) * (1.0 - overall_frac)
                        if overall_frac > 0.002
                        else 0.0
                    )
                    self._main_shared_state["pct"] = overall_pct
                    self._main_shared_state["status"] = (
                        f"Frame {idx + 1}/{total_frames} (ETA: {format_eta(overall_eta)})"
                    )

                rendered_frame = palletized_colorspace_fast(
                    frame_rgb,
                    palette_copy,
                    25000,
                    color_space_code,
                    w1,
                    w2,
                    w3,
                    dither_type=dither_type,
                    dither_strength=dither_strength,
                    progress_callback=frame_progress_cb,
                    cancel_check=is_cancelled,
                )

                if rendered_frame is None or is_cancelled():
                    break

                rendered_uint8 = self._ensure_uint8(rendered_frame)

                # Save rendered frame directly to temporary disk folder (not kept in RAM list)
                frame_file = os.path.join(out_folder, f"frame_{actual_written:06d}.png")
                ok = cv.imwrite(
                    frame_file,
                    cv.cvtColor(rendered_uint8, cv.COLOR_RGB2BGR),
                    [cv.IMWRITE_PNG_COMPRESSION, 1],
                )
                if not ok:
                    raise IOError(f"Failed to write rendered frame to temporary storage: {frame_file}")

                if first_thumb is None:
                    first_thumb = fitted_image(rendered_uint8, HISTORY_THUMB_MAX_DIM)

                # Publish only the latest frame for live main-canvas display
                self._main_shared_state["latest_frame"] = rendered_uint8
                self._main_shared_state["latest_frame_idx"] = actual_written
                actual_written += 1

                done_frac = actual_written / float(total_frames)
                elapsed = time.time() - video_start_time
                overall_eta = (elapsed / done_frac) * (1.0 - done_frac) if done_frac > 0.002 else 0.0
                self._main_shared_state["pct"] = min(99.9, done_frac * 100.0)
                self._main_shared_state["status"] = (
                    f"Frame {actual_written}/{total_frames} (ETA: {format_eta(overall_eta)})"
                )
                self._main_shared_state["frame_pct"] = 100.0
                self._main_shared_state["frame_status"] = f"({actual_written}/{total_frames}) Frame complete!"

            if is_cancelled() or actual_written == 0:
                if out_folder and os.path.exists(out_folder):
                    shutil.rmtree(out_folder, ignore_errors=True)
                if not is_cancelled() and actual_written == 0:
                    self._main_shared_state["error"] = "No frames could be processed from the video."
            else:
                durations = (
                    source_state.durations_ms[:actual_written]
                    if source_state.durations_ms
                    else None
                )
                result_state = MediaState.from_temp_folder(
                    uid=state_uid,
                    kind=source_state.kind,
                    folder_path=out_folder,
                    width=source_state.width,
                    height=source_state.height,
                    frame_count=actual_written,
                    fps=source_state.fps,
                    thumb_rgb=first_thumb,
                    durations_ms=durations,
                )
                self._main_shared_state["result_state"] = result_state

            self._main_shared_state["done"] = True

        except Exception as e:
            if out_folder and os.path.exists(out_folder):
                shutil.rmtree(out_folder, ignore_errors=True)
            self._main_shared_state["error"] = str(e)
            self._main_shared_state["done"] = True

    def _poll_main_worker(self, task_id: int):
        if task_id != self.active_task_id:
            return

        st = self._main_shared_state
        self._set_progress_safe(
            task_id,
            st["pct"],
            st["status"],
            st.get("frame_pct", 0.0),
            st.get("frame_status", "Ready"),
        )

        # Live preview each video/GIF frame on the main canvas as it finishes rendering
        latest_idx = st.get("latest_frame_idx", -1)
        latest_frame = st.get("latest_frame", None)
        if latest_frame is not None and latest_idx > self._last_previewed_frame_idx:
            self._last_previewed_frame_idx = latest_idx
            self.current_frame_idx = latest_idx
            self.main_canvas_image = latest_frame
            self._update_playback_controls_ui(
                rendering_info=(latest_idx, max(1, st.get("total_frames", 1)))
            )
            self._render_main_canvas()

        if not st["done"]:
            self.root.after(40, lambda: self._poll_main_worker(task_id))
            return

        if st["error"]:
            self._handle_worker_error(task_id, st["error"])
        else:
            self._finalize_palletizer(task_id, st["result_state"])

    def _finalize_palletizer(self, task_id: int, result_state: MediaState | None):
        if task_id != self.active_task_id:
            return

        self._set_worker_busy_ui(False)

        if result_state is not None and not self.cancel_requested:
            preserve_idx = self._main_shared_state.get("preserve_frame_idx", True)
            if self._main_shared_state.get("reset_zoom_on_finish", False):
                self._reset_zoom()
            self._push_media_to_history(result_state, preserve_frame_idx=preserve_idx)
            if result_state.is_multiframe:
                self._set_progress_safe(
                    task_id,
                    100.0,
                    f"Done! ({result_state.frame_count} frames)",
                    100.0,
                    f"({result_state.frame_count}/{result_state.frame_count}) Done!",
                )
            else:
                self._set_progress_safe(
                    task_id,
                    100.0,
                    "Done!",
                    0.0,
                    "N/A (Static Image)",
                )
        else:
            self._set_progress_safe(task_id, 0.0, "Cancelled", 0.0, "Cancelled")
            if self.current_media_state is not None:
                if (
                    self.main_canvas_image is not None
                    and self.main_canvas_image.shape[:2] != self.current_media_state.shape[:2]
                ):
                    self._reset_zoom()
                self._show_media_frame(min(self.current_frame_idx, self.current_media_state.frame_count - 1))

    def _handle_worker_error(self, task_id: int, err_msg: str):
        if task_id != self.active_task_id:
            return
        self._set_worker_busy_ui(False)
        self._set_progress_safe(task_id, 0.0, "Error", 0.0, "Error")
        if self.current_media_state is not None:
            self._show_media_frame(min(self.current_frame_idx, self.current_media_state.frame_count - 1))
        messagebox.showerror("Processing Error", err_msg)

    # ==========================================
    # HISTORY / TIMELINE MANAGEMENT (Ctrl+Z)
    # ==========================================
    def _active_temp_folders(self) -> set[str]:
        return {s.folder_path for s in self.history if s.folder_path}

    def _push_to_history(self, img_array: np.ndarray, kind: str = "image"):
        if img_array is None:
            return
        state = MediaState.from_image(
            uid=self._next_state_uid(),
            image_rgb=self._ensure_uint8(img_array).copy(),
            kind=kind,
        )
        self._push_media_to_history(state)

    def _push_media_to_history(self, new_state: MediaState, preserve_frame_idx: bool = False):
        if new_state is None:
            return

        self.pause_playback()

        # Clean up any truncated redo branch states
        if self.history_index < len(self.history) - 1:
            truncated = self.history[self.history_index + 1:]
            self.history = self.history[: self.history_index + 1]
            keep = self._active_temp_folders()
            if new_state.folder_path:
                keep.add(new_state.folder_path)
            for old_st in truncated:
                old_st.cleanup(keep_folders=keep)

        prev_active = self.current_media_state
        if prev_active is not None and new_state.shape[:2] != prev_active.shape[:2]:
            self._reset_zoom()
        elif self.main_canvas_image is not None and new_state.shape[:2] != self.main_canvas_image.shape[:2]:
            self._reset_zoom()

        # Release full-frame memory cache on previous active state before switching
        if prev_active is not None:
            prev_active.release_memory_cache()

        self.history.append(new_state)

        if len(self.history) > MAX_HISTORY_STATES:
            evicted = self.history.pop(0)
            evicted.cleanup(keep_folders=self._active_temp_folders())

        self.history_index = len(self.history) - 1
        if not preserve_frame_idx:
            self.current_frame_idx = 0
        else:
            self.current_frame_idx = max(0, min(self.current_frame_idx, new_state.frame_count - 1))

        self._sync_history_views()

    def undo(self, event=None):
        if self._is_worker_busy:
            return
        if self.history_index > 0:
            self.pause_playback()
            curr_st = self.history[self.history_index]
            prev_shape = curr_st.shape[:2]
            curr_st.release_memory_cache()

            self.history_index -= 1
            target_st = self.history[self.history_index]
            if target_st.shape[:2] != prev_shape:
                self._reset_zoom()
            self.current_frame_idx = max(0, min(self.current_frame_idx, target_st.frame_count - 1))
            self._sync_history_views()

    def redo(self, event=None):
        if self._is_worker_busy:
            return
        if self.history_index < len(self.history) - 1:
            self.pause_playback()
            curr_st = self.history[self.history_index]
            prev_shape = curr_st.shape[:2]
            curr_st.release_memory_cache()

            self.history_index += 1
            target_st = self.history[self.history_index]
            if target_st.shape[:2] != prev_shape:
                self._reset_zoom()
            self.current_frame_idx = max(0, min(self.current_frame_idx, target_st.frame_count - 1))
            self._sync_history_views()

    def _sync_history_views(self):
        total = len(self.history)

        if 0 <= self.history_index < total:
            active_state = self.history[self.history_index]
            self.current_frame_idx = max(0, min(self.current_frame_idx, active_state.frame_count - 1))
            self.main_canvas_image = active_state.get_frame(self.current_frame_idx)
            self._update_playback_controls_ui()
            self.root.update_idletasks()
            self._render_main_canvas()

        if self.history_index > 0:
            prev_state = self.history[self.history_index - 1]
            self.secondary_thumb_image = prev_state.thumb_rgb
            if prev_state.is_multiframe:
                kind_str = "GIF" if prev_state.kind == "gif" else "Video"
                self.prev_frame.config(
                    text=f"4. History & Previous {kind_str} ({prev_state.width}x{prev_state.height}, {prev_state.frame_count}f)"
                )
            else:
                self.prev_frame.config(
                    text=f"4. History & Previous Image ({prev_state.width}x{prev_state.height})"
                )
            self.root.update_idletasks()
            pw = max(self.prev_image_box.winfo_width() - 10, 180)
            ph = max(self.prev_image_box.winfo_height() - 10, 110)
            self._display_on_label(self.prev_image_thumb, self.secondary_thumb_image, pw, ph)
        else:
            self.secondary_thumb_image = None
            self.prev_frame.config(text="4. History & Previous State (Ctrl+Z)")
            self.prev_image_thumb.config(image="", text="Start of history\n(No previous state)")
            self.prev_image_thumb.image = None

        if total > 0:
            self.hist_status_lbl.config(text=f"{self.history_index + 1} / {total}")
        else:
            self.hist_status_lbl.config(text="0 / 0")

        if not self._is_worker_busy:
            self.undo_btn.config(state=tk.NORMAL if self.history_index > 0 else tk.DISABLED)
            self.redo_btn.config(state=tk.NORMAL if self.history_index < total - 1 else tk.DISABLED)

    # ==========================================
    # IMAGE CONVERSION & DISPLAY HELPERS
    # ==========================================
    def _ensure_uint8(self, img_array):
        if img_array is None:
            return None
        arr = np.asarray(img_array)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _display_on_label(self, label_widget, img_array, max_w, max_h, nearest=False):
        if img_array is None:
            return

        arr = self._ensure_uint8(img_array)
        pil_img = Image.fromarray(arr)

        max_w = max(max_w, 50)
        max_h = max(max_h, 50)

        orig_w, orig_h = pil_img.size
        ratio = min(max_w / orig_w, max_h / orig_h)
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))

        resample_mode = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
        pil_img = pil_img.resize((new_w, new_h), resample_mode)

        tk_img = ImageTk.PhotoImage(pil_img)
        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img

    def _on_main_canvas_resize(self, event):
        if event.width > 50 and event.height > 50:
            self._render_main_canvas()

    def _on_prev_thumb_resize(self, event):
        if self.secondary_thumb_image is not None and event.width > 50 and event.height > 50:
            self._display_on_label(
                self.prev_image_thumb,
                self.secondary_thumb_image,
                event.width - 10,
                event.height - 10
            )

    # ==========================================
    # CORE WORKFLOW ACTIONS (LOAD, RESIZE, SAVE)
    # ==========================================
    def _probe_video_frame_count(self, file_path: str, reported_count: int) -> int:
        if reported_count > 0:
            return reported_count
        cap = cv.VideoCapture(file_path)
        count = 0
        try:
            while cap.isOpened():
                if not cap.grab():
                    break
                count += 1
        finally:
            cap.release()
        return max(1, count)

    def load_original_image(self):
        if self._is_worker_busy:
            return

        file_path = filedialog.askopenfilename(
            title="Select Base Image, GIF, or Video",
            filetypes=[
                (
                    "Media Files (Images, GIFs & Videos)",
                    "*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif *.gif *.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.m4v *.mpg *.mpeg",
                ),
                ("Image & GIF Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif *.gif"),
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.m4v *.mpg *.mpeg"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            max_dim = self._parse_dimension(self.orig_max_dim_var.get(), "Max Dimension")
        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))
            return

        self.pause_playback()

        try:
            # 1. VIDEO FILE
            if is_video_file(file_path):
                self._load_video_file(file_path, max_dim)
                return

            # 2. GIF OR ANIMATED WEBP
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".gif", ".webp"):
                with Image.open(file_path) as pil_img:
                    n_frames = getattr(pil_img, "n_frames", 1)
                    is_anim = getattr(pil_img, "is_animated", False) and n_frames > 1

                if is_anim:
                    self._start_gif_load_thread(file_path, max_dim)
                    return
                elif ext == ".gif":
                    # Single-frame GIF
                    with Image.open(file_path) as pil_img:
                        rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                    clamped = fitted_image(rgb_arr, max_dim)
                    self.raw_original_image = None
                    self._reset_zoom()
                    self._push_to_history(clamped, kind="gif")
                    return

            # 3. STANDARD STATIC IMAGE
            self.raw_original_image = read_image_rgb(file_path)
            if max_dim == FULL_DIMENSIONS:
                fitted_orig = self.raw_original_image.copy()
            else:
                fitted_orig = fitted_image(self.raw_original_image, max_dim)

            self._reset_zoom()
            self._push_to_history(fitted_orig, kind="image")

        except Exception as e:
            messagebox.showerror("Error Loading Media", str(e))

    def _load_video_file(self, file_path: str, max_dim: int):
        cap = cv.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video file:\n{file_path}")

        ret, first_bgr = cap.read()
        if not ret or first_bgr is None:
            cap.release()
            raise ValueError(f"Could not decode video frames from:\n{file_path}")

        orig_h, orig_w = first_bgr.shape[:2]
        fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
        if math.isnan(fps) or fps < 1.0 or fps > 240.0:
            fps = 30.0
        raw_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        frame_count = self._probe_video_frame_count(file_path, raw_count)
        target_w, target_h = compute_clamped_dimensions(orig_w, orig_h, max_dim)

        first_rgb = cv.cvtColor(first_bgr, cv.COLOR_BGR2RGB)
        if (target_w, target_h) != (orig_w, orig_h):
            first_rgb = cv.resize(first_rgb, (target_w, target_h), interpolation=cv.INTER_NEAREST)

        thumb_rgb = fitted_image(first_rgb, HISTORY_THUMB_MAX_DIM)
        state = MediaState.from_source_video(
            uid=self._next_state_uid(),
            source_path=file_path,
            width=target_w,
            height=target_h,
            frame_count=frame_count,
            fps=fps,
            thumb_rgb=thumb_rgb,
            first_frame_rgb=first_rgb,
        )
        self.raw_original_image = None
        self._reset_zoom()
        self._push_media_to_history(state)

    def _start_gif_load_thread(self, file_path: str, max_dim: int):
        self.cancel_requested = False
        self.active_task_id += 1
        task_id = self.active_task_id
        self._last_previewed_frame_idx = -1

        self._main_shared_state = {
            "pct": 0.0,
            "status": "Loading animated GIF...",
            "frame_pct": 0.0,
            "frame_status": "Decoding frames...",
            "is_multiframe": True,
            "preserve_frame_idx": False,
            "reset_zoom_on_finish": True,
            "latest_frame": None,
            "latest_frame_idx": -1,
            "total_frames": 1,
            "done": False,
            "error": None,
            "result_state": None,
        }

        self._set_worker_busy_ui(True)
        self._set_progress_safe(task_id, 0.0, "Loading animated GIF...", 0.0, "Decoding frames...")

        threading.Thread(
            target=self._run_gif_load_worker,
            args=(task_id, file_path, max_dim),
            daemon=True,
        ).start()
        self.root.after(40, lambda: self._poll_main_worker(task_id))

    def _run_gif_load_worker(self, task_id: int, file_path: str, max_dim: int):
        out_folder = None
        try:
            def is_cancelled():
                return self.cancel_requested or (task_id != self.active_task_id)

            state_uid = self._next_state_uid()
            out_folder = os.path.join(self.temp_dir, f"state_{state_uid}")
            os.makedirs(out_folder, exist_ok=True)

            durations_ms = []
            actual_written = 0
            first_thumb = None
            target_w, target_h = 1, 1

            with Image.open(file_path) as pil_img:
                total_frames = max(1, getattr(pil_img, "n_frames", 1))
                self._main_shared_state["total_frames"] = total_frames
                orig_w, orig_h = pil_img.size
                target_w, target_h = compute_clamped_dimensions(orig_w, orig_h, max_dim)

                for idx, frame in enumerate(ImageSequence.Iterator(pil_img)):
                    if is_cancelled():
                        break

                    dur = int(frame.info.get("duration", 100) or 100)
                    durations_ms.append(max(20, dur))

                    rgb_arr = np.array(frame.convert("RGB"), dtype=np.uint8)
                    if (rgb_arr.shape[1], rgb_arr.shape[0]) != (target_w, target_h):
                        rgb_arr = cv.resize(rgb_arr, (target_w, target_h), interpolation=cv.INTER_NEAREST)

                    frame_file = os.path.join(out_folder, f"frame_{actual_written:06d}.png")
                    cv.imwrite(
                        frame_file,
                        cv.cvtColor(rgb_arr, cv.COLOR_RGB2BGR),
                        [cv.IMWRITE_PNG_COMPRESSION, 1],
                    )

                    if first_thumb is None:
                        first_thumb = fitted_image(rgb_arr, HISTORY_THUMB_MAX_DIM)

                    actual_written += 1
                    pct = (actual_written / float(total_frames)) * 100.0
                    self._main_shared_state["pct"] = pct
                    self._main_shared_state["status"] = f"Loading GIF frame {actual_written}/{total_frames}"
                    self._main_shared_state["frame_pct"] = 100.0
                    self._main_shared_state["frame_status"] = f"Loaded {actual_written}/{total_frames}"

            if is_cancelled() or actual_written == 0:
                if out_folder and os.path.exists(out_folder):
                    shutil.rmtree(out_folder, ignore_errors=True)
            else:
                avg_dur = sum(durations_ms) / max(1, len(durations_ms))
                fps = 1000.0 / max(15.0, avg_dur)
                result_state = MediaState.from_temp_folder(
                    uid=state_uid,
                    kind="gif",
                    folder_path=out_folder,
                    width=target_w,
                    height=target_h,
                    frame_count=actual_written,
                    fps=fps,
                    thumb_rgb=first_thumb,
                    durations_ms=durations_ms,
                )
                self.raw_original_image = None
                self._main_shared_state["result_state"] = result_state

            self._main_shared_state["done"] = True
        except Exception as e:
            if out_folder and os.path.exists(out_folder):
                shutil.rmtree(out_folder, ignore_errors=True)
            self._main_shared_state["error"] = str(e)
            self._main_shared_state["done"] = True

    def resize_current_canvas_image(self):
        if self._is_worker_busy:
            return
        active_state = self.current_media_state
        if active_state is None or self.main_canvas_image is None:
            messagebox.showwarning("No Media", "Load an image or video onto the Main Canvas first.")
            return

        try:
            max_dim = self._parse_dimension(self.orig_max_dim_var.get(), "Max Dimension")
        except ValueError as ve:
            messagebox.showerror("Resize Error", str(ve))
            return

        self.pause_playback()

        # 1. VIDEO OR GIF (1-frame or multi-frame):
        # Clamp dimensions down so biggest dimension (height or width) is <= Max Dim.
        # If selected Max Dim is bigger than current biggest dimension (or 0 = Full), keep current dimensions.
        if active_state.kind in ("video", "gif"):
            cur_w, cur_h = active_state.width, active_state.height
            new_w, new_h = compute_clamped_dimensions(cur_w, cur_h, max_dim)

            if (new_w, new_h) == (cur_w, cur_h):
                # Selected dimension is >= current biggest dimension; clamped size equals current size
                self._reset_zoom()
                self._render_main_canvas()
                return

            if not active_state.is_multiframe:
                resized_img = fitted_image(active_state.get_frame(0).copy(), max_dim)
                self._reset_zoom()
                self._push_to_history(resized_img, kind=active_state.kind)
                return

            if active_state.backing == "source_video" and active_state.source_path:
                first_frame = cv.resize(
                    active_state.get_frame(0), (new_w, new_h), interpolation=cv.INTER_NEAREST
                )
                new_state = MediaState.from_source_video(
                    uid=self._next_state_uid(),
                    source_path=active_state.source_path,
                    width=new_w,
                    height=new_h,
                    frame_count=active_state.frame_count,
                    fps=active_state.fps,
                    thumb_rgb=fitted_image(first_frame, HISTORY_THUMB_MAX_DIM),
                    first_frame_rgb=first_frame,
                )
                self._reset_zoom()
                self._push_media_to_history(new_state, preserve_frame_idx=True)
                return

            # Backed by temp_frames (animated GIF or already-processed video): resize all frames in worker
            self._start_multiframe_resize_thread(active_state, new_w, new_h)
            return

        # 2. STATIC IMAGE
        try:
            if max_dim == FULL_DIMENSIONS:
                source = self.raw_original_image if self.raw_original_image is not None else self.main_canvas_image
                resized_img = source.copy()
            else:
                resized_img = fitted_image(self.main_canvas_image.copy(), max_dim)
            self._reset_zoom()
            self._push_to_history(resized_img, kind="image")
        except Exception as e:
            messagebox.showerror("Resize Error", str(e))

    def _start_multiframe_resize_thread(self, source_state: MediaState, new_w: int, new_h: int):
        self.cancel_requested = False
        self.active_task_id += 1
        task_id = self.active_task_id
        self._last_previewed_frame_idx = -1

        self._reset_zoom()
        self._main_shared_state = {
            "pct": 0.0,
            "status": f"Resizing to {new_w}x{new_h}...",
            "frame_pct": 0.0,
            "frame_status": "Resizing frames...",
            "is_multiframe": True,
            "preserve_frame_idx": True,
            "reset_zoom_on_finish": True,
            "latest_frame": None,
            "latest_frame_idx": -1,
            "total_frames": source_state.frame_count,
            "done": False,
            "error": None,
            "result_state": None,
        }

        self._set_worker_busy_ui(True)
        self._set_progress_safe(task_id, 0.0, f"Resizing to {new_w}x{new_h}...", 0.0, "Starting...")

        threading.Thread(
            target=self._run_multiframe_resize_worker,
            args=(task_id, source_state, new_w, new_h),
            daemon=True,
        ).start()
        self.root.after(40, lambda: self._poll_main_worker(task_id))

    def _run_multiframe_resize_worker(self, task_id: int, source_state: MediaState, new_w: int, new_h: int):
        out_folder = None
        try:
            def is_cancelled():
                return self.cancel_requested or (task_id != self.active_task_id)

            total_frames = max(1, source_state.frame_count)
            state_uid = self._next_state_uid()
            out_folder = os.path.join(self.temp_dir, f"state_{state_uid}")
            os.makedirs(out_folder, exist_ok=True)

            actual_written = 0
            first_thumb = None

            for idx, frame_rgb in source_state.iter_frames_for_worker():
                if is_cancelled():
                    break

                resized_rgb = cv.resize(frame_rgb, (new_w, new_h), interpolation=cv.INTER_NEAREST)
                frame_file = os.path.join(out_folder, f"frame_{actual_written:06d}.png")
                cv.imwrite(
                    frame_file,
                    cv.cvtColor(resized_rgb, cv.COLOR_RGB2BGR),
                    [cv.IMWRITE_PNG_COMPRESSION, 1],
                )

                if first_thumb is None:
                    first_thumb = fitted_image(resized_rgb, HISTORY_THUMB_MAX_DIM)

                self._main_shared_state["latest_frame"] = resized_rgb
                self._main_shared_state["latest_frame_idx"] = actual_written
                actual_written += 1

                pct = (actual_written / float(total_frames)) * 100.0
                self._main_shared_state["pct"] = pct
                self._main_shared_state["status"] = f"Resized {actual_written}/{total_frames} frames ({new_w}x{new_h})"
                self._main_shared_state["frame_pct"] = 100.0
                self._main_shared_state["frame_status"] = f"Frame {actual_written}/{total_frames} resized"

            if is_cancelled() or actual_written == 0:
                if out_folder and os.path.exists(out_folder):
                    shutil.rmtree(out_folder, ignore_errors=True)
            else:
                durations = (
                    source_state.durations_ms[:actual_written]
                    if source_state.durations_ms
                    else None
                )
                result_state = MediaState.from_temp_folder(
                    uid=state_uid,
                    kind=source_state.kind,
                    folder_path=out_folder,
                    width=new_w,
                    height=new_h,
                    frame_count=actual_written,
                    fps=source_state.fps,
                    thumb_rgb=first_thumb,
                    durations_ms=durations,
                )
                self._main_shared_state["result_state"] = result_state

            self._main_shared_state["done"] = True
        except Exception as e:
            if out_folder and os.path.exists(out_folder):
                shutil.rmtree(out_folder, ignore_errors=True)
            self._main_shared_state["error"] = str(e)
            self._main_shared_state["done"] = True

    def load_palette_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Palette Image (Images Only)",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        if is_video_file(file_path):
            messagebox.showwarning(
                "Images Only",
                "The palette feature only accepts static images from which to extract colors."
            )
            return

        try:
            if is_gif_file(file_path):
                with Image.open(file_path) as pil_img:
                    if getattr(pil_img, "n_frames", 1) > 1:
                        messagebox.showwarning(
                            "Static Images Only",
                            "Please select a static image for Palette extraction."
                        )
                        return
                    raw_pal = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            else:
                raw_pal = read_image_rgb(file_path)

            self.palette_image = fitted_image(raw_pal, FULL_DIMENSIONS, cv.INTER_CUBIC)
            ph, pw = self.palette_image.shape[:2]
            self.pal_source_title_lbl.config(text=f"Palette Source ({pw}x{ph})")
            self._display_on_label(self.pal_source_thumb, self.palette_image, THUMB_W - 6, THUMB_H - 6)

            self.palette = None
            self.palette_square_img = None
            self.pal_square_title_lbl.config(text="Extracted Square")
            self.pal_square_thumb.config(image="", text="Click 'Extract Palette'")
            self.pal_square_thumb.image = None
            self.pal_progress_var.set(0.0)
            self.pal_progress_label.config(text="Ready")
        except Exception as e:
            messagebox.showerror("Error Loading Palette Image", str(e))

    def save_result_manual(self):
        if self._is_worker_busy:
            return
        state = self.current_media_state
        if state is None or self.main_canvas_image is None:
            messagebox.showwarning("No Media", "There is no image or video on the main canvas to save.")
            return

        self.pause_playback()

        if not state.is_multiframe:
            save_path = filedialog.asksaveasfilename(
                title="Save Image As",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("GIF", "*.gif")]
            )
            if save_path:
                arr = self._ensure_uint8(self.main_canvas_image)
                Image.fromarray(arr).save(save_path)
                messagebox.showinfo("Saved", f"Image saved successfully to:\n{save_path}")
            return

        # Multi-frame video or animated GIF save options
        if state.kind == "gif":
            default_ext = ".gif"
            ftypes = [
                ("Animated GIF", "*.gif"),
                ("MP4 Video", "*.mp4"),
                ("AVI Video", "*.avi"),
                ("Current Frame as PNG", "*.png"),
                ("Current Frame as JPEG", "*.jpg"),
            ]
        else:
            default_ext = ".mp4"
            ftypes = [
                ("MP4 Video", "*.mp4"),
                ("AVI Video", "*.avi"),
                ("Animated GIF", "*.gif"),
                ("Current Frame as PNG", "*.png"),
                ("Current Frame as JPEG", "*.jpg"),
            ]

        save_path = filedialog.asksaveasfilename(
            title="Save Video / Animation As",
            defaultextension=default_ext,
            filetypes=ftypes,
        )
        if not save_path:
            return

        ext = os.path.splitext(save_path)[1].lower()
        try:
            self.root.config(cursor="watch")
            self.root.update_idletasks()

            # Static single-frame export of current frame
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"):
                arr = self._ensure_uint8(self.main_canvas_image)
                Image.fromarray(arr).save(save_path)
                self.root.config(cursor="")
                messagebox.showinfo("Saved", f"Current frame saved to:\n{save_path}")
                return

            # Protect any history state referencing the same external file path from overwrite lock
            abs_save = os.path.abspath(save_path)
            for hist_st in self.history:
                if hist_st.backing == "source_video" and hist_st.source_path:
                    if os.path.abspath(hist_st.source_path) == abs_save:
                        hist_st.detach_source_to_temp(self.temp_dir)

            # Animated GIF export
            if ext == ".gif":
                pil_frames = []
                for _, frame_rgb in state.iter_frames_for_worker():
                    pil_frames.append(Image.fromarray(self._ensure_uint8(frame_rgb)))
                if not pil_frames:
                    raise ValueError("No frames available to save.")
                durations = (
                    state.durations_ms[: len(pil_frames)]
                    if state.durations_ms
                    else [state.get_frame_delay_ms(0)] * len(pil_frames)
                )
                pil_frames[0].save(
                    save_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=durations,
                    loop=0,
                )
                self.root.config(cursor="")
                messagebox.showinfo("Saved", f"Animated GIF saved successfully to:\n{save_path}")
                return

            # Video container export (.mp4, .avi, etc.)
            if ext == ".avi":
                codec_candidates = ["XVID", "MJPG", "DIVX"]
            else:
                codec_candidates = ["mp4v", "avc1", "XVID", "MJPG"]

            writer = None
            for codec in codec_candidates:
                fourcc = cv.VideoWriter_fourcc(*codec)
                candidate = cv.VideoWriter(save_path, fourcc, state.fps, (state.width, state.height))
                if candidate.isOpened():
                    writer = candidate
                    break
                candidate.release()

            if writer is None:
                raise RuntimeError(f"Could not initialize video encoder for '{save_path}'.")

            try:
                for _, frame_rgb in state.iter_frames_for_worker():
                    bgr = cv.cvtColor(self._ensure_uint8(frame_rgb), cv.COLOR_RGB2BGR)
                    writer.write(bgr)
            finally:
                writer.release()

            self.root.config(cursor="")
            messagebox.showinfo("Saved", f"Video saved successfully to:\n{save_path}")
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Save Error", f"Could not save media:\n{e}")

    # ==========================================
    # INTERACTIVE POPUP ZOOM, PAN & VIDEO PLAYER
    # ==========================================
    def open_popup(self, media_or_img, title="Image Preview", nearest=False, start_frame_idx=0):
        if media_or_img is None:
            return

        is_state = isinstance(media_or_img, MediaState)
        if is_state:
            media_state: MediaState | None = media_or_img
            cur_idx = max(0, min(int(start_frame_idx), media_state.frame_count - 1))
            arr = self._ensure_uint8(media_state.get_frame(cur_idx))
        else:
            media_state = None
            cur_idx = 0
            arr = self._ensure_uint8(media_or_img)

        if arr is None:
            return

        ih, iw = arr.shape[:2]

        popup = tk.Toplevel(self.root)
        popup.geometry("840x700")
        popup.minsize(420, 340)
        c_bg = self.current_theme.get("canvas_bg", "#1e1e1e")
        popup.configure(bg=c_bg)

        state = {
            "zoom": 1.0,
            "max_zoom": 30.0,
            "ox": 0.0,
            "oy": 0.0,
            "drag_x": 0,
            "drag_y": 0,
            "tk_img": None,
            "frame_idx": cur_idx,
            "frame_arr": arr,
            "playing": False,
            "after_id": None,
            "updating_slider": False,
        }

        # Optional playback bar inside popup when inspecting a multi-frame video or GIF
        if media_state is not None and media_state.is_multiframe and media_state.frame_count > 1:
            p_bar = ttk.Frame(popup, padding=(8, 4, 8, 8))
            p_bar.pack(side=tk.BOTTOM, fill=tk.X)

            p_play_btn = ttk.Button(p_bar, text="▶ Play", width=9)
            p_play_btn.pack(side=tk.LEFT, padx=(0, 3))

            p_prev_btn = ttk.Button(p_bar, text="◀ Frame", width=8)
            p_prev_btn.pack(side=tk.LEFT, padx=(0, 2))

            p_next_btn = ttk.Button(p_bar, text="Frame ▶", width=8)
            p_next_btn.pack(side=tk.LEFT, padx=(0, 6))

            p_var = tk.DoubleVar(value=float(cur_idx))
            p_slider = ttk.Scale(
                p_bar,
                from_=0.0,
                to=float(max(1, media_state.frame_count - 1)),
                orient=tk.HORIZONTAL,
                variable=p_var,
            )
            p_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

            p_lbl = ttk.Label(p_bar, text="", width=22, anchor=tk.E, font=("", 8))
            p_lbl.pack(side=tk.RIGHT, padx=(6, 0))
        else:
            p_play_btn = None
            p_slider = None
            p_lbl = None

        canvas = tk.Canvas(popup, bg=c_bg, highlightthickness=0, cursor="fleur")
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        def _update_popup_title():
            zoom_pct = int(round(state["zoom"] * 100))
            if media_state is not None and media_state.is_multiframe:
                popup.title(
                    f"{title} ({iw}x{ih}, Frame {state['frame_idx'] + 1}/{media_state.frame_count}) — "
                    f"Zoom: {zoom_pct}% (Scroll to Zoom, Drag to Pan)"
                )
                if p_lbl is not None:
                    p_lbl.config(
                        text=f"Frame {state['frame_idx'] + 1} / {media_state.frame_count} ({media_state.fps:.1f} fps)"
                    )
            else:
                popup.title(f"{title} ({iw}x{ih}) — Zoom: {zoom_pct}% (Scroll to Zoom, Drag to Pan)")

        def _render_popup():
            if not popup.winfo_exists():
                return
            _update_popup_title()
            canvas.delete("all")
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 40 or ch < 40:
                return

            cur_arr = state["frame_arr"]
            base_scale = min(max(cw - 10, 40) / iw, max(ch - 10, 40) / ih)
            eff_scale = base_scale * state["zoom"]

            disp_w = iw * eff_scale
            disp_h = ih * eff_scale

            if disp_w <= cw:
                state["ox"] = (cw - disp_w) / 2.0
            else:
                state["ox"] = min(0.0, max(cw - disp_w, state["ox"]))

            if disp_h <= ch:
                state["oy"] = (ch - disp_h) / 2.0
            else:
                state["oy"] = min(0.0, max(ch - disp_h, state["oy"]))

            vis_x0 = max(0.0, state["ox"])
            vis_y0 = max(0.0, state["oy"])
            vis_x1 = min(float(cw), state["ox"] + disp_w)
            vis_y1 = min(float(ch), state["oy"] + disp_h)

            if vis_x1 <= vis_x0 or vis_y1 <= vis_y0:
                return

            src_x0 = max(0, int(np.floor((vis_x0 - state["ox"]) / eff_scale)))
            src_y0 = max(0, int(np.floor((vis_y0 - state["oy"]) / eff_scale)))
            src_x1 = min(iw, int(np.ceil((vis_x1 - state["ox"]) / eff_scale)))
            src_y1 = min(ih, int(np.ceil((vis_y1 - state["oy"]) / eff_scale)))

            if src_x1 <= src_x0 or src_y1 <= src_y0:
                return

            cropped = cur_arr[src_y0:src_y1, src_x0:src_x1]
            target_w = max(1, int(round((src_x1 - src_x0) * eff_scale)))
            target_h = max(1, int(round((src_y1 - src_y0) * eff_scale)))

            use_nearest = nearest or (eff_scale > 1.5)
            resample_mode = Image.Resampling.NEAREST if use_nearest else Image.Resampling.LANCZOS
            pil_patch = Image.fromarray(cropped).resize((target_w, target_h), resample_mode)

            draw_x = int(round(state["ox"] + src_x0 * eff_scale))
            draw_y = int(round(state["oy"] + src_y0 * eff_scale))

            state["tk_img"] = ImageTk.PhotoImage(pil_patch)
            canvas.create_image(draw_x, draw_y, anchor=tk.NW, image=state["tk_img"])

        def _set_popup_frame(idx: int):
            if media_state is None:
                return
            idx = max(0, min(int(idx), media_state.frame_count - 1))
            state["frame_idx"] = idx
            state["frame_arr"] = self._ensure_uint8(media_state.get_frame(idx))
            if p_slider is not None:
                state["updating_slider"] = True
                try:
                    p_var.set(float(idx))
                finally:
                    state["updating_slider"] = False
            _render_popup()

        def _stop_popup_play():
            state["playing"] = False
            if state["after_id"] is not None:
                try:
                    popup.after_cancel(state["after_id"])
                except Exception:
                    pass
                state["after_id"] = None
            if p_play_btn is not None and popup.winfo_exists():
                p_play_btn.config(text="▶ Play")

        def _popup_tick():
            state["after_id"] = None
            if not state["playing"] or not popup.winfo_exists() or media_state is None:
                return
            t0 = time.perf_counter()
            nxt = (state["frame_idx"] + 1) % media_state.frame_count
            _set_popup_frame(nxt)
            delay = media_state.get_frame_delay_ms(state["frame_idx"])
            elapsed = (time.perf_counter() - t0) * 1000.0
            wait_ms = max(10, int(round(delay - elapsed)))
            if state["playing"] and popup.winfo_exists():
                state["after_id"] = popup.after(wait_ms, _popup_tick)

        def _toggle_popup_play():
            if state["playing"]:
                _stop_popup_play()
            else:
                if media_state is None or media_state.frame_count <= 1:
                    return
                state["playing"] = True
                if p_play_btn is not None:
                    p_play_btn.config(text="⏸ Pause")
                state["after_id"] = popup.after(
                    media_state.get_frame_delay_ms(state["frame_idx"]), _popup_tick
                )

        if p_play_btn is not None and media_state is not None:
            p_play_btn.config(command=_toggle_popup_play)
            p_prev_btn.config(
                command=lambda: (_stop_popup_play(), _set_popup_frame((state["frame_idx"] - 1) % media_state.frame_count))
            )
            p_next_btn.config(
                command=lambda: (_stop_popup_play(), _set_popup_frame((state["frame_idx"] + 1) % media_state.frame_count))
            )

            def _on_p_scrub(val_str):
                if state["updating_slider"]:
                    return
                try:
                    idx = int(round(float(val_str)))
                except ValueError:
                    return
                if idx != state["frame_idx"]:
                    _set_popup_frame(idx)

            p_slider.config(command=_on_p_scrub)

        def _on_popup_close():
            _stop_popup_play()
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _on_popup_close)

        def _on_popup_scroll(event):
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 40 or ch < 40:
                return

            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                step = 1.2
            elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
                step = 1.0 / 1.2
            else:
                return

            old_zoom = state["zoom"]
            new_zoom = max(1.0, min(state["max_zoom"], old_zoom * step))
            if new_zoom < 1.02:
                new_zoom = 1.0
            if new_zoom == old_zoom:
                return

            base_scale = min(max(cw - 10, 40) / iw, max(ch - 10, 40) / ih)
            old_eff = base_scale * old_zoom
            new_eff = base_scale * new_zoom

            mx, my = event.x, event.y
            img_x = (mx - state["ox"]) / old_eff
            img_y = (my - state["oy"]) / old_eff

            state["zoom"] = new_zoom
            state["ox"] = mx - img_x * new_eff
            state["oy"] = my - img_y * new_eff
            _render_popup()

        def _on_popup_down(event):
            state["drag_x"] = event.x
            state["drag_y"] = event.y

        def _on_popup_drag(event):
            if state["zoom"] > 1.0:
                state["ox"] += event.x - state["drag_x"]
                state["oy"] += event.y - state["drag_y"]
                state["drag_x"] = event.x
                state["drag_y"] = event.y
                _render_popup()

        canvas.bind("<Configure>", lambda e: _render_popup())
        canvas.bind("<MouseWheel>", _on_popup_scroll)
        canvas.bind("<Button-4>", _on_popup_scroll)
        canvas.bind("<Button-5>", _on_popup_scroll)
        canvas.bind("<ButtonPress-1>", _on_popup_down)
        canvas.bind("<B1-Motion>", _on_popup_drag)

    @staticmethod
    def _parse_dimension(val_str, field_name):
        val_str = val_str.strip()
        if not val_str:
            return FULL_DIMENSIONS
        try:
            val = int(val_str)
            if val < 0:
                raise ValueError
            return FULL_DIMENSIONS if val == 0 else val
        except ValueError:
            raise ValueError(f"'{field_name}' must be 0 (for full resolution) or a positive integer.")


if __name__ == "__main__":
    root = tk.Tk()
    app = PaletteApp(root)
    root.mainloop()
