"""Loader for the UBFC-rPPG (Dataset 2) folder layout:

    data/raw/UBFC-rPPG/subjectN/vid.avi
    data/raw/UBFC-rPPG/subjectN/ground_truth.txt   # 3 rows: PPG, HR (oximeter), timestamps (s)
"""
import re
from pathlib import Path

import cv2
import numpy as np

UBFC_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "UBFC-rPPG"


def list_subjects(root=UBFC_ROOT):
    dirs = [d for d in Path(root).iterdir() if d.is_dir() and d.name.startswith("subject")]
    return sorted(dirs, key=lambda d: int(re.findall(r"\d+", d.name)[0]))


def load_ground_truth(subject_dir):
    rows = [np.array(line.split(), dtype=np.float64) for line in open(Path(subject_dir) / "ground_truth.txt")]
    ppg, hr, ts = rows[0], rows[1], rows[2]
    # Timestamps jitter frame to frame, so use the average rate over the whole recording.
    fs = (len(ts) - 1) / (ts[-1] - ts[0])
    return {"ppg": ppg, "hr": hr, "ts": ts, "fs": fs}


def resample_uniform(ppg, ts, fs):
    """Ground-truth samples have jittery timestamps; put them on an even grid (one per frame)."""
    grid = ts[0] + np.arange(len(ts)) / fs
    return np.interp(grid, ts, ppg)


def video_info(subject_dir):
    cap = cv2.VideoCapture(str(Path(subject_dir) / "vid.avi"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": n,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    # Decode the first and last frame to make sure the file isn't truncated/corrupt.
    ok_first, _ = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok_last, _ = cap.read()
    info["readable"] = bool(ok_first and ok_last)
    cap.release()
    return info
