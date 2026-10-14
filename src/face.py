"""Face tracking + cropping, shared by offline preprocessing and the live demo.

The same code must run in both places: if training crops and live crops differ
(box size, smoothing, color order), the network sees inputs it was never trained on.
"""
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe's C++ logging

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")


def ensure_face_model():
    """Download Google's MediaPipe face detector (230 KB) on first use, so a fresh clone runs."""
    if MODEL_PATH.exists():
        return
    import urllib.request

    print(f"Downloading the MediaPipe face detector (230 KB) to {MODEL_PATH} ...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(MODEL_PATH)
CROP_SIZE = 72


class FaceTracker:
    """Detects the face every `detect_every` frames and smooths the box between detections.

    Box format: (center_x, center_y, side) of a square, in pixels.
    - scale: the detector's box is tight around eyes/mouth; enlarge it to include
      forehead and cheeks, where most of the pulse signal is.
    - alpha: exponential smoothing; a jittery box adds pixel-shift noise that can be
      larger than the pulse color change itself.
    """

    def __init__(self, detect_every=5, scale=1.5, alpha=0.3, min_confidence=0.5):
        ensure_face_model()
        self.detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        ))
        self.detect_every = detect_every
        self.scale = scale
        self.alpha = alpha
        self.box = None
        self.found = False  # did the most recent detection succeed?
        self.i = 0

    def _detect(self, frame_rgb):
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
        if not result.detections:
            return None
        best = max(result.detections, key=lambda d: d.categories[0].score)
        b = best.bounding_box
        side = max(b.width, b.height) * self.scale
        return np.array([b.origin_x + b.width / 2, b.origin_y + b.height / 2, side], dtype=np.float64)

    def update(self, frame_rgb):
        """Feed one RGB frame; returns the current smoothed box, or None if no face yet."""
        if self.box is None or self.i % self.detect_every == 0:
            target = self._detect(frame_rgb)
            self.found = target is not None
            if target is not None:
                self.box = target if self.box is None else self.alpha * target + (1 - self.alpha) * self.box
        self.i += 1
        return None if self.box is None else self.box.copy()

    def close(self):
        self.detector.close()


def crop_face(frame_rgb, box, size=CROP_SIZE):
    """Cut the square box out of the frame (zero-padding past the edges) and resize."""
    cx, cy, side = box
    half = side / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    h, w = frame_rgb.shape[:2]
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        frame_rgb = cv2.copyMakeBorder(frame_rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    # INTER_AREA averages pixels when shrinking, which also reduces camera noise.
    return cv2.resize(frame_rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
