"""Live Sudoku solver desktop app.

Usage:
    python live_solver.py
    python live_solver.py --model models/digit_cnn.pth --camera 0 --size 600
    python live_solver.py --smoke
    python live_solver.py --smoke path/to/image.jpg

A Tkinter desktop app that runs the existing Sudoku solver live on a webcam
feed, with an upload-image path, pause/resume, snapshot, and status panels.
"""
import argparse
import os
import queue
import sys
import threading
import time
from tkinter import Tk, ttk, Frame, Label, filedialog, messagebox

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

import sudoku_core as sc
from digit_cnn import load_digit_model, load_temperature, predict_cells_probs, \
    classify_preprocessed


DEFAULT_MODEL = "models/digit_cnn.pth"
DEFAULT_CAMERA = 0
DEFAULT_SIZE = 600
SMOKE_IMAGE = os.path.join(
    "benchmark_data", "hf_test_sample", "images", "0t1g4k7u4lec1.jpeg")
DISPLAY_BG = "#1a1a1a"
STATUS_BG = "#f5f5f5"
FILL_COLOR = "#00897b"          # teal for solver-filled cells
RECOG_COLOR = "#212121"         # dark gray for recognized givens


def imread_any(path):
    """cv2.imread with a Pillow fallback for formats cv2 cannot decode."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image as PilImage
        with PilImage.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
    except (OSError, ValueError):
        return None
    return arr[:, :, ::-1].copy()


def fit_size(src_w, src_h, max_w, max_h):
    """Return an aspect-preserving (w, h) that fits inside max_w x max_h."""
    scale = min(max_w / max(src_w, 1), max_h / max(src_h, 1), 1.0)
    return int(src_w * scale), int(src_h * scale)


def annotate_frame(frame, quad_pts, grid, solved, ok):
    """Draw the detected quad and recognized/solved digits on the BGR frame."""
    annotated = frame.copy()
    if quad_pts is None:
        return annotated

    # Closed green polyline around the detected grid.
    pts = quad_pts.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(annotated, [np.vstack([pts, pts[0:1]])], True,
                  (0, 255, 0), 3, cv2.LINE_AA)

    # Canonicalize corner order so (u, v) maps cleanly to rows/columns.
    P = sc.order_points(quad_pts).astype(np.float32)
    p0, p1, p2, p3 = P[0], P[1], P[2], P[3]

    # Approximate cell width in screen pixels for font sizing.
    cell_w = float(np.linalg.norm(p1 - p0)) / 9.0
    font_scale = max(0.4, cell_w / 54.0)
    thickness = max(1, int(round(font_scale * 2)))

    for r in range(9):
        for c in range(9):
            given = int(grid[r, c])
            filled = ok and solved[r, c] != 0 and given == 0
            if given == 0 and not filled:
                continue

            u = (c + 0.5) / 9.0
            v = (r + 0.5) / 9.0
            pt = ((1 - u) * (1 - v) * p0 +
                  u * (1 - v) * p1 +
                  u * v * p2 +
                  (1 - u) * v * p3)
            x, y = int(round(pt[0])), int(round(pt[1]))

            text = str(given) if given != 0 else str(int(solved[r, c]))
            color = (255, 255, 255) if given != 0 else (0, 255, 255)

            # Dark outline for readability on any background.
            cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

    return annotated


class ModelThread(threading.Thread):
    """Load the CNN and its temperature sidecar in the background."""
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app

    def run(self):
        try:
            model = load_digit_model(self.app.model_path, device=self.app.device)
            temperature = load_temperature(self.app.model_path)
            self.app.status_queue.put(("model_ready", model, temperature, None))
        except Exception as exc:
            self.app.status_queue.put(("model_ready", None, None, str(exc)))


class CameraThread(threading.Thread):
    """Capture webcam frames in a background thread."""
    MAX_WIDTH = 1280
    MAX_HEIGHT = 720

    def __init__(self, index):
        super().__init__(daemon=True)
        self.index = index
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self.camera_ok = False
        self.cap = None

    @staticmethod
    def _open(index):
        """Open the capture, preferring DirectShow on Windows.

        MSMF (the OpenCV default) is known to leak frame buffers with some
        webcam drivers; DirectShow reuses them. Falls back to the default
        backend when neither named backend opens the device.
        """
        if os.name == "nt":
            for backend in (getattr(cv2, "CAP_DSHOW", None),
                            getattr(cv2, "CAP_MSMF", None)):
                if backend is None:
                    continue
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    return cap
                cap.release()
        return cv2.VideoCapture(index)

    @staticmethod
    def _fit(frame):
        """Downscale oversized frames to bound per-frame memory.

        The solver only needs ~600 px for the warp, so a 4K camera frame is
        pure overhead (11 MB/frame vs ~2.8 MB at 720p).
        """
        h, w = frame.shape[:2]
        scale = min(CameraThread.MAX_WIDTH / w,
                    CameraThread.MAX_HEIGHT / h, 1.0)
        if scale >= 1.0:
            return frame
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def run(self):
        self.cap = self._open(self.index)
        self.camera_ok = self.cap.isOpened()
        if not self.camera_ok:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.MAX_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.MAX_HEIGHT)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            if not self.paused.is_set():
                with self._lock:
                    self._latest = self._fit(frame)
        self.cap.release()
        self.cap = None

    def get_frame(self):
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    def stop(self):
        self.stop_event.set()


class SolverThread(threading.Thread):
    """Run detection -> recognition -> solving on frames from a queue."""
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.stop_event = threading.Event()
        self.queue = queue.Queue()
        self.busy = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            job = self.queue.get()
            if job is None or self.stop_event.is_set():
                break
            # Drain queue so only the newest job is processed.
            newest = job
            while True:
                try:
                    newest = self.queue.get_nowait()
                except queue.Empty:
                    break
            kind, frame = newest
            self.busy.set()
            try:
                result = self._process(frame, kind)
            except Exception as exc:
                result = {
                    "kind": kind,
                    "frame": frame,
                    "detection": "error",
                    "grid": np.zeros((9, 9), dtype=int),
                    "solved": np.zeros((9, 9), dtype=int),
                    "ok": False,
                    "nodes": 0,
                    "resensed": 0,
                    "error": str(exc),
                }
            self.app.results_queue.put(result)
            # Throttle live passes to keep the UI responsive.
            if kind == "live" and not self.stop_event.is_set():
                time.sleep(0.15)
            self.busy.clear()

    def _process(self, frame, kind):
        size = self.app.size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        quad, _ = sc.detect_grid_contour(gray)
        quad_pts = None
        detection = "none"
        warped = None

        if quad is not None:
            warped = sc.four_point_transform(gray, quad, size)
            quad_pts = quad.reshape(4, 2)
            detection = "contour"
        else:
            warped = sc.line_grid_quad(gray, size)
            if warped is not None:
                detection = "lines"

        grid = np.zeros((9, 9), dtype=int)
        solved = np.zeros((9, 9), dtype=int)
        ok = False
        nodes = 0
        resensed = 0
        limit_hit = False

        if warped is not None:
            cells = sc.extract_cells(warped, size)
            probs = predict_cells_probs(
                cells, self.app.model, device=self.app.device,
                temperature=self.app.temperature)
            grid = probs.argmax(axis=1).reshape(9, 9)
            stats = {}
            solved, ok, resensed = sc.solve_with_resensing(
                grid, probs, cells,
                lambda views: classify_preprocessed(
                    views, self.app.model, self.app.device),
                stats=stats)
            nodes = stats.get("nodes", 0)
            limit_hit = stats.get("limit_hit", False)

        annotated = annotate_frame(frame, quad_pts, grid, solved, ok)
        return {
            "kind": kind,
            "frame": annotated,
            "detection": detection,
            "grid": grid,
            "solved": solved,
            "ok": ok,
            "nodes": nodes,
            "resensed": resensed,
            "limit_hit": limit_hit,
            "error": None,
        }

    def submit(self, kind, frame):
        self.queue.put((kind, frame))

    def drain(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        self.stop_event.set()
        self.queue.put(None)


class LiveSolverApp:
    def __init__(self, root, args):
        self.root = root
        self.root.title("Sudoku Live Solver")
        self.root.minsize(1100, 700)
        self.root.configure(bg=DISPLAY_BG)

        self.model_path = args.model
        self.camera_index = args.camera
        self.size = args.size
        self.smoke = args.smoke is not False
        self.smoke_image = args.smoke if isinstance(args.smoke, str) else SMOKE_IMAGE
        self.smoke_done = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.temperature = 1.0
        self.model_ready = False

        self.mode = "camera"          # "camera" or "upload"
        self.has_camera = False
        self.available_cameras = []
        self.camera_running = False
        self.paused = False
        self.upload_frame = None
        self.latest_result = None
        self._smoke_watchdog_id = None

        self.status_queue = queue.Queue()
        self.results_queue = queue.Queue()

        self.camera_thread = None
        self.solver_thread = SolverThread(self)
        self.model_thread = ModelThread(self)

        self._build_ui()
        self._reset_grids()

        # Start the solver and model threads immediately; camera starts on demand.
        self.solver_thread.start()
        self.model_thread.start()

        if not self.smoke:
            self._probe_camera()

        self._poll_id = self.root.after(33, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Main layout: left display, right panel, bottom toolbar.
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Left display
        display_frame = Frame(self.root, bg=DISPLAY_BG, bd=0)
        display_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)

        # Explicit width/height pin the label's REQUESTED size to a small
        # constant: without it, each displayed image would resize the label,
        # push the window's requested size past its actual size, and Tk would
        # grow the window a bit every pass (measured: +40 px / 2 s in camera
        # mode). sticky="nsew" still stretches the label to fill its cell,
        # so the display looks identical - only the geometry feedback is cut.
        self.video_label = Label(
            display_frame, text="Waiting for camera…", fg="white",
            bg=DISPLAY_BG, font=("Segoe UI", 14), width=2, height=1)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self._display_photo = None

        # Right panel
        right = Frame(self.root, bg=STATUS_BG, bd=1, relief="solid")
        right.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(2, weight=1)
        right.grid_rowconfigure(3, weight=0)

        Label(right, text="Sudoku Live Solver", bg=STATUS_BG,
              font=("Segoe UI", 18, "bold"), fg=RECOG_COLOR).grid(
            row=0, column=0, sticky="ew", pady=(12, 8), padx=12)

        grids = Frame(right, bg=STATUS_BG)
        grids.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        grids.grid_columnconfigure(0, weight=1)
        grids.grid_columnconfigure(1, weight=1)

        self.rec_title = Label(grids, text="Recognized", bg=STATUS_BG,
                               font=("Segoe UI", 11, "bold"), fg=RECOG_COLOR)
        self.rec_title.grid(row=0, column=0, pady=(0, 4))
        self.sol_title = Label(grids, text="Solution", bg=STATUS_BG,
                               font=("Segoe UI", 11, "bold"), fg=RECOG_COLOR)
        self.sol_title.grid(row=0, column=1, pady=(0, 4))

        self.rec_frame, self.rec_labels = self._build_grid(grids)
        self.rec_frame.grid(row=1, column=0, padx=6)
        self.sol_frame, self.sol_labels = self._build_grid(grids)
        self.sol_frame.grid(row=1, column=1, padx=6)

        # Status block
        status = Frame(right, bg=STATUS_BG)
        status.grid(row=2, column=0, sticky="ew", padx=12, pady=8)
        status.grid_columnconfigure(0, weight=1)

        Label(status, text="Status", bg=STATUS_BG, font=("Segoe UI", 11, "bold"),
              fg=RECOG_COLOR).grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.mode_lbl = Label(status, text="Loading model…", bg=STATUS_BG,
                              fg="#616161", font=("Segoe UI", 10))
        self.mode_lbl.grid(row=1, column=0, sticky="w", pady=1)
        self.detect_lbl = Label(status, text="Detection: —", bg=STATUS_BG,
                                fg="#616161", font=("Segoe UI", 10))
        self.detect_lbl.grid(row=2, column=0, sticky="w", pady=1)
        self.solve_lbl = Label(status, text="Solve: —", bg=STATUS_BG,
                               fg="#616161", font=("Segoe UI", 10))
        self.solve_lbl.grid(row=3, column=0, sticky="w", pady=1)
        self.model_lbl = Label(status, text=f"Model: {self.model_path} ({self.device})",
                               bg=STATUS_BG, fg="#616161", font=("Segoe UI", 10))
        self.model_lbl.grid(row=4, column=0, sticky="w", pady=1)

        # Toolbar
        toolbar = Frame(self.root, bg=STATUS_BG, bd=1, relief="solid")
        toolbar.grid(row=2, column=0, columnspan=2, sticky="ew",
                     padx=10, pady=(0, 10))
        toolbar.grid_columnconfigure(0, weight=1)

        inner = Frame(toolbar, bg=STATUS_BG)
        inner.grid(row=0, column=0, pady=10)

        Label(inner, text="Camera:", bg=STATUS_BG, fg="#616161",
              font=("Segoe UI", 10)).grid(row=0, column=0, padx=(4, 2))
        self.camera_combo = ttk.Combobox(inner, state="readonly", width=4,
                                         values=[], font=("Segoe UI", 10))
        self.camera_combo.grid(row=0, column=1, padx=(0, 6))
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_selected)

        self.start_btn = ttk.Button(inner, text="Start Camera",
                                    command=self._toggle_camera)
        self.start_btn.grid(row=0, column=2, padx=4)
        self.pause_btn = ttk.Button(inner, text="Pause",
                                    command=self._toggle_pause)
        self.pause_btn.grid(row=0, column=3, padx=4)
        self.snapshot_btn = ttk.Button(
            inner, text="Snapshot & Solve", command=self._snapshot)
        self.snapshot_btn.grid(row=0, column=4, padx=4)
        self.upload_btn = ttk.Button(
            inner, text="Upload Image…", command=self._upload_image)
        self.upload_btn.grid(row=0, column=5, padx=4)
        self.quit_btn = ttk.Button(inner, text="Quit", command=self._on_close)
        self.quit_btn.grid(row=0, column=6, padx=4)

        self._set_controls(enabled=False)

    def _build_grid(self, parent):
        """Create a 9x9 grid of labels inside 3x3 boxes. Returns (frame, labels).

        Labels are appended in ROW-MAJOR order so label index i corresponds
        to grid.flat[i] (recognized/solved grids are row-major); the visual
        layout keeps the 3x3 box structure.
        """
        outer = Frame(parent, bg="black", bd=2, relief="solid")
        font = ("Consolas", 16, "bold")
        boxes = {}
        for br in range(3):
            for bc in range(3):
                box = Frame(outer, bg="black", bd=1, relief="solid")
                box.grid(row=br, column=bc, padx=1, pady=1)
                boxes[(br, bc)] = box
        labels = []
        for r in range(9):
            for c in range(9):
                lbl = Label(boxes[(r // 3, c // 3)], width=2, height=1,
                            font=font, bg="white", fg=RECOG_COLOR,
                            relief="solid", bd=1)
                lbl.grid(row=r % 3, column=c % 3, padx=1, pady=1)
                labels.append(lbl)
        return outer, labels

    # ---------------------------------------------------------------- state
    def _enumerate_cameras(self, limit=4):
        """Probe indices 0..limit-1 for openable capture devices."""
        found = []
        for i in range(limit):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append(i)
            cap.release()
        return found

    def _probe_camera(self):
        """Enumerate available cameras and configure the picker."""
        self.available_cameras = self._enumerate_cameras()
        self.has_camera = len(self.available_cameras) > 0
        if not self.available_cameras:
            self.mode = "upload"
            self.mode_lbl.config(
                text="No camera detected — use Upload image")
            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="disabled")
            self.snapshot_btn.config(state="disabled")
            self.camera_combo.config(state="disabled")
            return

        self.camera_combo.config(
            values=[str(i) for i in self.available_cameras])
        chosen = (self.camera_index
                  if self.camera_index in self.available_cameras
                  else self.available_cameras[0])
        self.camera_index = chosen
        self.camera_combo.set(str(chosen))

    def _on_camera_selected(self, event=None):
        """Picker changed: switch the active camera (restart if running)."""
        try:
            idx = int(self.camera_combo.get())
        except (ValueError, TypeError):
            return
        if idx == self.camera_index:
            return
        self.camera_index = idx
        if self.camera_running:
            self._stop_camera()
            self._start_camera()
        else:
            self.mode_lbl.config(
                text=f"Camera {idx}: selected", fg="#616161")

    def _set_controls(self, enabled):
        """Enable/disable solver-related controls while the model loads."""
        state = "normal" if enabled else "disabled"
        self.upload_btn.config(state=state)
        if enabled and self.has_camera and not self.camera_running:
            self.start_btn.config(state="normal")
        else:
            self.start_btn.config(state="disabled")
        if enabled and self.camera_running:
            self.pause_btn.config(state="normal")
            self.snapshot_btn.config(state="normal")
        else:
            self.pause_btn.config(state="disabled")
            self.snapshot_btn.config(state="disabled")

    def _sync_buttons(self):
        """Refresh button labels/states from current mode and camera state."""
        if self.mode == "upload":
            self.start_btn.config(text="Back to Camera")
            if self.has_camera:
                self.start_btn.config(state="normal")
            else:
                self.start_btn.config(state="disabled")
            self.pause_btn.config(text="Pause", state="disabled")
            self.snapshot_btn.config(state="disabled")
            self.upload_btn.config(state="disabled")
            return

        # camera mode
        self.upload_btn.config(state="normal")
        self.start_btn.config(text="Stop Camera" if self.camera_running else "Start Camera")
        if not self.has_camera:
            self.start_btn.config(state="disabled")
        elif self.model_ready:
            self.start_btn.config(state="normal")

        if self.camera_running and self.model_ready:
            self.pause_btn.config(state="normal",
                                  text="Resume" if self.paused else "Pause")
            self.snapshot_btn.config(state="normal")
        else:
            self.pause_btn.config(state="disabled", text="Pause")
            self.snapshot_btn.config(state="disabled")

    def _reset_grids(self):
        for lbl in self.rec_labels:
            lbl.config(text="", bg="white", fg=RECOG_COLOR)
        for lbl in self.sol_labels:
            lbl.config(text="", bg="white", fg=RECOG_COLOR)

    # ---------------------------------------------------------------- model
    def _on_model_ready(self, model, temperature, error):
        if error is not None:
            self.model_lbl.config(text=f"Model: error loading {self.model_path}")
            self.mode_lbl.config(text=f"Model error: {error}", fg="#c62828")
            if self.smoke:
                print(f"SMOKE error: {error}", file=sys.stderr)
                self._on_close(exit_code=1)
            return

        self.model = model
        self.temperature = temperature
        self.model_ready = True
        self.model_lbl.config(
            text=f"Model: {os.path.basename(self.model_path)} ({self.device})")
        self.mode_lbl.config(text="Ready — start camera or upload an image")

        if self.smoke:
            self._run_smoke()
        else:
            self._sync_buttons()

    def _run_smoke(self):
        img = imread_any(self.smoke_image)
        if img is None:
            print(f"SMOKE error: could not read {self.smoke_image}", file=sys.stderr)
            self._on_close(exit_code=1)
            return
        self.mode = "upload"
        self.upload_frame = img
        self.solver_thread.submit("solve", img)
        self._smoke_watchdog_id = self.root.after(30000, self._smoke_timeout)

    def _smoke_timeout(self):
        if not self.smoke_done:
            print("SMOKE error: solve pass timed out", file=sys.stderr)
            self._on_close(exit_code=1)

    # ---------------------------------------------------------------- camera
    def _toggle_camera(self):
        if self.mode == "upload":
            # "Back to Camera" was pressed.
            self.mode = "camera"
            self.upload_frame = None
            self._reset_grids()
            self._start_camera()
            self._sync_buttons()
            return

        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()
        self._sync_buttons()

    def _start_camera(self):
        if self.camera_thread is not None:
            return
        self.camera_thread = CameraThread(self.camera_index)
        self.camera_thread.start()
        self.camera_running = True
        self.paused = False
        self.mode = "camera"
        self.mode_lbl.config(
            text=f"Camera {self.camera_index}: running",
            fg="#616161")

    def _stop_camera(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.join(timeout=1.0)
            self.camera_thread = None
        self.camera_running = False
        self.paused = False
        self.mode_lbl.config(
            text=f"Camera {self.camera_index}: stopped", fg="#616161")

    def _toggle_pause(self):
        if not self.camera_running or self.camera_thread is None:
            return
        self.paused = not self.paused
        self.camera_thread.paused.set() if self.paused else self.camera_thread.paused.clear()
        self.mode_lbl.config(
            text=(f"Camera {self.camera_index}: paused" if self.paused
                  else f"Camera {self.camera_index}: running"),
            fg="#616161")
        self._sync_buttons()

    def _snapshot(self):
        if not self.camera_running or self.camera_thread is None:
            return
        self.paused = True
        self.camera_thread.paused.set()
        frame = self.camera_thread.get_frame()
        if frame is not None:
            self.solver_thread.drain()
            self.solver_thread.submit("solve", frame)
        self.mode_lbl.config(
            text=f"Camera {self.camera_index}: paused (snapshot)", fg="#616161")
        self._sync_buttons()

    # ---------------------------------------------------------------- upload
    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select a Sudoku image",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tiff;*.tif"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*"),
            ])
        if not path:
            return
        img = imread_any(path)
        if img is None:
            messagebox.showerror("Could not read image",
                                 f"The file could not be opened as an image:\n{path}")
            self.mode_lbl.config(text="Upload failed: unreadable image", fg="#c62828")
            return

        # Switch to upload mode.
        if self.camera_running:
            self._stop_camera()
        self.mode = "upload"
        self.upload_frame = img
        self.paused = False
        self.mode_lbl.config(text=f"Uploaded image: {os.path.basename(path)}",
                             fg="#616161")
        self._sync_buttons()
        self.solver_thread.drain()
        self.solver_thread.submit("solve", img)

    # ---------------------------------------------------------------- display
    def _show_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w = max(self.video_label.winfo_width(), 320)
        max_h = max(self.video_label.winfo_height(), 240)
        new_w, new_h = fit_size(w, h, max_w, max_h)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.video_label.config(image=photo, text="")
        self._display_photo = photo

    def _update_grids(self, grid, solved, ok):
        # Recognized grid.
        for i, lbl in enumerate(self.rec_labels):
            v = int(grid.flat[i])
            lbl.config(text=str(v) if v else "", fg=RECOG_COLOR, bg="white")

        # Solution grid.
        if ok:
            self.sol_title.config(text="Solution")
            for i, lbl in enumerate(self.sol_labels):
                given = int(grid.flat[i])
                fill = int(solved.flat[i])
                if given:
                    lbl.config(text=str(given), fg=RECOG_COLOR, bg="white")
                elif fill:
                    lbl.config(text=str(fill), fg=FILL_COLOR, bg="#e0f2f1")
                else:
                    lbl.config(text="", bg="white")
        else:
            self.sol_title.config(text="No solution found")
            for lbl in self.sol_labels:
                lbl.config(text="", bg="white")

    # ---------------------------------------------------------------- poll
    def _poll(self):
        # Drain status queue.
        while True:
            try:
                item = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "model_ready":
                _, model, temperature, error = item
                self._on_model_ready(model, temperature, error)

        # Drain result queue.
        new_result = None
        while True:
            try:
                new_result = self.results_queue.get_nowait()
            except queue.Empty:
                break

        if new_result is not None:
            self.latest_result = new_result
            self._show_frame(new_result["frame"])
            self._update_grids(new_result["grid"], new_result["solved"],
                               new_result["ok"])
            detection = new_result["detection"]
            self.detect_lbl.config(
                text=f"Grid detected: {detection}",
                fg="#2e7d32" if detection != "none" else "#c62828")

            if new_result.get("error"):
                self.solve_lbl.config(
                    text=f"Solve error: {new_result['error']}", fg="#c62828")
            elif new_result["detection"] == "none":
                self.solve_lbl.config(text="Solve: no grid detected")
            elif new_result["ok"]:
                self.solve_lbl.config(
                    text=f"Solved — {new_result['nodes']:,} nodes, "
                         f"{new_result['resensed']} re-sensed",
                    fg="#2e7d32")
            else:
                reason = "node limit" if new_result["nodes"] >= 190_000 else "no valid solution"
                self.solve_lbl.config(
                    text=f"No solution — {reason} ({new_result['nodes']:,} nodes)",
                    fg="#c62828")

            if self.smoke and not self.smoke_done:
                self.smoke_done = True
                if self._smoke_watchdog_id is not None:
                    self.root.after_cancel(self._smoke_watchdog_id)
                    self._smoke_watchdog_id = None
                print(
                    f"SMOKE: detected={new_result['detection']} "
                    f"solved={new_result['ok']} nodes={new_result['nodes']}")
                self.root.after(8000, lambda: self._on_close(exit_code=0))

        # Feed live frames to the solver when appropriate. The busy gate
        # ensures at most ONE frame is in flight/pending per pass, so the
        # queue can never accumulate frames while a (slow) pass runs.
        if (self.model_ready and self.mode == "camera" and
                self.camera_running and not self.paused and
                self.camera_thread is not None and
                not self.solver_thread.busy.is_set()):
            frame = self.camera_thread.get_frame()
            if frame is not None:
                self.solver_thread.submit("live", frame)

        self._poll_id = self.root.after(33, self._poll)

    # ---------------------------------------------------------------- close
    def _on_close(self, exit_code=0):
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        if self._smoke_watchdog_id is not None:
            self.root.after_cancel(self._smoke_watchdog_id)
            self._smoke_watchdog_id = None

        if self.solver_thread is not None:
            self.solver_thread.stop()
            self.solver_thread.join(timeout=1.0)
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.join(timeout=1.0)

        self.root.destroy()
        if self.smoke:
            sys.exit(exit_code if self.smoke_done else 1)


def main():
    parser = argparse.ArgumentParser(description="Live Sudoku solver")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="CNN weights")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA,
                        help="webcam index")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="warped grid size")
    parser.add_argument("--smoke", nargs="?", const=True, default=False,
                        help="self-test mode; optionally pass an image path")
    args = parser.parse_args()

    root = Tk()
    app = LiveSolverApp(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
