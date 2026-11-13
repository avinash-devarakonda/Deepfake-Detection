import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as tkfont

# ── Attempt to import detection dependencies ─────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.xception import preprocess_input
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Model Loading ─────────────────────────────────────────────────────────────
custom_model = False
model = None

def load_detection_model():
    global model, custom_model
    if not TF_AVAILABLE:
        return
    try:
        model = load_model('deepfake_detection_model.h5')
        custom_model = True
    except Exception:
        model = Xception(weights='imagenet', include_top=False, pooling='avg')
        custom_model = False

def preprocess_image(img_path):
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.xception import preprocess_input
    img = keras_image.load_img(img_path, target_size=(299, 299))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_deepfake(img_path):
    if not TF_AVAILABLE or model is None:
        score = np.random.rand()  # Demo mode
    elif custom_model:
        img_array = preprocess_image(img_path)
        score = float(model.predict(img_array)[0][0])
    else:
        score = np.random.rand()
    label = "REAL" if score < 0.5 else "FAKE"
    return label, score

# ── Color Palette ─────────────────────────────────────────────────────────────
BG        = "#0d0d0f"
SURFACE   = "#16161a"
PANEL     = "#1c1c22"
BORDER    = "#2a2a35"
ACCENT    = "#00e5ff"
ACCENT2   = "#7c3aed"
REAL_CLR  = "#00e676"
FAKE_CLR  = "#ff1744"
TEXT      = "#e8e8f0"
SUBTEXT   = "#7a7a90"
WHITE     = "#ffffff"

# ── Main Application ──────────────────────────────────────────────────────────
class DeepfakeDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Deepfake Detector")
        self.geometry("960x700")
        self.minsize(800, 600)
        self.configure(bg=BG)
        self.resizable(True, True)

        self._img_path = None
        self._photo   = None
        self._running = False

        self._setup_fonts()
        self._build_ui()
        self._load_model_async()

    # ── Fonts ─────────────────────────────────────────────────────────────────
    def _setup_fonts(self):
        self.f_title  = tkfont.Font(family="Courier New", size=20, weight="bold")
        self.f_label  = tkfont.Font(family="Courier New", size=10)
        self.f_mono   = tkfont.Font(family="Courier New", size=11, weight="bold")
        self.f_score  = tkfont.Font(family="Courier New", size=36, weight="bold")
        self.f_sub    = tkfont.Font(family="Courier New", size=9)
        self.f_badge  = tkfont.Font(family="Courier New", size=13, weight="bold")

    # ── UI Build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=SURFACE, height=64)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        inner_hdr = tk.Frame(hdr, bg=SURFACE)
        inner_hdr.pack(fill="both", expand=True, padx=24)

        tk.Label(inner_hdr, text="◈ DEEPFAKE DETECTOR", font=self.f_title,
                 bg=SURFACE, fg=ACCENT).pack(side="left", pady=16)

        self.model_badge = tk.Label(inner_hdr, text="● LOADING MODEL",
                                    font=self.f_sub, bg=SURFACE, fg=SUBTEXT)
        self.model_badge.pack(side="right", pady=16)

        # ── Separator ──
        tk.Frame(self, bg=ACCENT, height=1).pack(fill="x")

        # ── Body ──
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=20)

        # Left: Image panel
        self.left = tk.Frame(body, bg=PANEL, bd=0, highlightthickness=1,
                              highlightbackground=BORDER)
        self.left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self._build_drop_zone()

        # Right: Controls + Results
        right = tk.Frame(body, bg=BG, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_controls(right)
        self._build_results(right)
        self._build_log(right)

        # ── Status bar ──
        bar = tk.Frame(self, bg=SURFACE, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready. Select an image to begin analysis.")
        tk.Label(bar, textvariable=self.status_var, font=self.f_sub,
                 bg=SURFACE, fg=SUBTEXT, anchor="w").pack(side="left", padx=16, pady=6)

    def _build_drop_zone(self):
        self.drop_frame = tk.Frame(self.left, bg=PANEL)
        self.drop_frame.pack(fill="both", expand=True, padx=2, pady=2)

        self.img_label = tk.Label(
            self.drop_frame,
            text="[ NO IMAGE LOADED ]\n\nClick 'Open Image' to begin",
            font=self.f_label, bg=PANEL, fg=SUBTEXT,
            justify="center", cursor="hand2"
        )
        self.img_label.pack(fill="both", expand=True)
        self.img_label.bind("<Button-1>", lambda e: self._open_image())

        # Overlay label (shown on image hover)
        self.img_label.bind("<Enter>", self._on_img_hover)
        self.img_label.bind("<Leave>", self._on_img_leave)

    def _build_controls(self, parent):
        cframe = tk.Frame(parent, bg=PANEL, bd=0, highlightthickness=1,
                          highlightbackground=BORDER)
        cframe.pack(fill="x", pady=(0, 12))

        tk.Label(cframe, text="CONTROLS", font=self.f_sub,
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=14, pady=(12, 4))

        tk.Frame(cframe, bg=BORDER, height=1).pack(fill="x", padx=14)

        btn_cfg = dict(font=self.f_mono, relief="flat", bd=0,
                       activebackground=ACCENT2, cursor="hand2",
                       padx=10, pady=10)

        self.open_btn = tk.Button(
            cframe, text="⊕  OPEN IMAGE",
            bg=ACCENT2, fg=WHITE, activeforeground=WHITE,
            command=self._open_image, **btn_cfg
        )
        self.open_btn.pack(fill="x", padx=14, pady=(14, 6))
        self._add_hover(self.open_btn, ACCENT2, "#6d28d9")

        self.analyze_btn = tk.Button(
            cframe, text="▶  ANALYZE",
            bg=BORDER, fg=SUBTEXT, activeforeground=WHITE,
            state="disabled", command=self._run_analysis, **btn_cfg
        )
        self.analyze_btn.pack(fill="x", padx=14, pady=(0, 14))

        self.progress = ttk.Progressbar(cframe, mode="indeterminate",
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=14, pady=(0, 14))
        self._style_progressbar()

    def _style_progressbar(self):
        s = ttk.Style(self)
        s.theme_use("default")
        s.configure("Accent.Horizontal.TProgressbar",
                    troughcolor=BORDER, background=ACCENT,
                    thickness=3)

    def _build_results(self, parent):
        rframe = tk.Frame(parent, bg=PANEL, bd=0, highlightthickness=1,
                          highlightbackground=BORDER)
        rframe.pack(fill="x", pady=(0, 12))

        tk.Label(rframe, text="ANALYSIS RESULT", font=self.f_sub,
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=14, pady=(12, 4))
        tk.Frame(rframe, bg=BORDER, height=1).pack(fill="x", padx=14)

        self.verdict_var = tk.StringVar(value="—")
        self.verdict_label = tk.Label(
            rframe, textvariable=self.verdict_var,
            font=self.f_score, bg=PANEL, fg=SUBTEXT
        )
        self.verdict_label.pack(pady=(16, 4))

        self.conf_var = tk.StringVar(value="confidence: —")
        tk.Label(rframe, textvariable=self.conf_var, font=self.f_sub,
                 bg=PANEL, fg=SUBTEXT).pack()

        # Confidence bar
        bar_bg = tk.Frame(rframe, bg=BORDER, height=6)
        bar_bg.pack(fill="x", padx=14, pady=(12, 16))
        self.conf_bar = tk.Frame(bar_bg, bg=ACCENT2, height=6, width=0)
        self.conf_bar.place(x=0, y=0, relheight=1, width=0)
        self._bar_width = 0
        bar_bg.bind("<Configure>", self._update_bar_bg)

    def _build_log(self, parent):
        lframe = tk.Frame(parent, bg=PANEL, bd=0, highlightthickness=1,
                          highlightbackground=BORDER)
        lframe.pack(fill="both", expand=True)

        tk.Label(lframe, text="LOG", font=self.f_sub,
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=14, pady=(12, 4))
        tk.Frame(lframe, bg=BORDER, height=1).pack(fill="x", padx=14)

        self.log_text = tk.Text(
            lframe, bg=BG, fg=SUBTEXT, font=self.f_sub,
            relief="flat", bd=0, state="disabled",
            wrap="word", insertbackground=ACCENT
        )
        self.log_text.pack(fill="both", expand=True, padx=14, pady=12)

    # ── Interactions ──────────────────────────────────────────────────────────
    def _add_hover(self, btn, normal, hover):
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=normal))

    def _on_img_hover(self, e):
        if self._img_path:
            self.img_label.configure(text="Click to change image")

    def _on_img_leave(self, e):
        if self._img_path and not self._photo:
            self.img_label.configure(text="")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        self._img_path = path
        self._display_image(path)
        self._reset_result()
        self.analyze_btn.configure(state="normal", bg=ACCENT, fg=BG,
                                   activebackground=ACCENT)
        self._add_hover(self.analyze_btn, ACCENT, "#00bcd4")
        self.status_var.set(f"Image loaded: {os.path.basename(path)}")
        self._log(f"Loaded: {os.path.basename(path)}")

    def _display_image(self, path):
        try:
            pil_img = Image.open(path)
            w = self.left.winfo_width() or 560
            h = self.left.winfo_height() or 540
            pil_img.thumbnail((w - 20, h - 20), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(pil_img)
            self.img_label.configure(image=self._photo, text="",
                                     bg=PANEL, compound="center")
        except Exception as ex:
            self._log(f"Error loading image: {ex}")

    def _reset_result(self):
        self.verdict_var.set("—")
        self.verdict_label.configure(fg=SUBTEXT)
        self.conf_var.set("confidence: —")
        self._animate_bar(0, ACCENT2)

    def _run_analysis(self):
        if self._running or not self._img_path:
            return
        self._running = True
        self.analyze_btn.configure(state="disabled", bg=BORDER, fg=SUBTEXT)
        self.progress.start(12)
        self.status_var.set("Analyzing image…")
        self._log("Starting analysis…")
        threading.Thread(target=self._analyze_worker, daemon=True).start()

    def _analyze_worker(self):
        try:
            label, score = predict_deepfake(self._img_path)
            self.after(0, self._show_result, label, score)
        except Exception as ex:
            self.after(0, self._analysis_error, str(ex))

    def _show_result(self, label, score):
        self.progress.stop()
        self._running = False
        color = REAL_CLR if label == "REAL" else FAKE_CLR
        self.verdict_var.set(label)
        self.verdict_label.configure(fg=color)
        pct = score if label == "FAKE" else (1 - score)
        self.conf_var.set(f"confidence: {pct * 100:.1f}%")
        self._animate_bar(pct, color)
        self.analyze_btn.configure(state="normal", bg=ACCENT, fg=BG)
        self.status_var.set(f"Result: {label}  |  Confidence: {pct*100:.1f}%")
        self._log(f"Prediction: {label}  (score={score:.4f})")

    def _analysis_error(self, msg):
        self.progress.stop()
        self._running = False
        self.analyze_btn.configure(state="normal", bg=ACCENT, fg=BG)
        self.status_var.set("Error during analysis.")
        self._log(f"ERROR: {msg}")

    # ── Confidence Bar ────────────────────────────────────────────────────────
    def _update_bar_bg(self, event):
        self._bar_total_width = event.width
        self._animate_bar(self._bar_width, self.conf_bar.cget("bg"))

    def _animate_bar(self, fraction, color, step=0):
        target = fraction
        total  = getattr(self, "_bar_total_width", 252)
        target_px = int(total * target)
        current_px = int(self.conf_bar.winfo_width()) if step > 0 else 0
        if step == 0:
            self.conf_bar.place_configure(width=0)
            self._bar_width = fraction
            self.conf_bar.configure(bg=color)
            self.after(16, lambda: self._animate_bar(target, color, 1))
            return
        delta = target_px - current_px
        if abs(delta) < 2:
            self.conf_bar.place_configure(width=target_px)
            return
        new_px = current_px + max(1, abs(delta) // 6) * (1 if delta > 0 else -1)
        self.conf_bar.place_configure(width=new_px)
        self.after(16, lambda: self._animate_bar(target, color, step + 1))

    # ── Log ───────────────────────────────────────────────────────────────────
    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"> {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ── Model Loading ─────────────────────────────────────────────────────────
    def _load_model_async(self):
        self.model_badge.configure(text="● LOADING MODEL", fg=SUBTEXT)
        self._log("Loading detection model…")
        threading.Thread(target=self._model_worker, daemon=True).start()

    def _model_worker(self):
        load_detection_model()
        self.after(0, self._model_ready)

    def _model_ready(self):
        if not TF_AVAILABLE:
            self.model_badge.configure(text="● DEMO MODE (TF missing)", fg=FAKE_CLR)
            self._log("TensorFlow not found. Running in demo (random) mode.")
        elif custom_model:
            self.model_badge.configure(text="● CUSTOM MODEL", fg=REAL_CLR)
            self._log("Custom deepfake model loaded successfully.")
        else:
            self.model_badge.configure(text="● PLACEHOLDER MODEL", fg="#ffab00")
            self._log("Custom model not found. Using placeholder (random scores).")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = DeepfakeDetectorApp()
    app.mainloop()
