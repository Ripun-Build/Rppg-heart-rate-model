"""PyTorch Dataset: cuts preprocessed UBFC subjects into normalized 160-frame clips.

Each item is a dict:
  x       (C, T, 72, 72) float32  network input
  y       (T,)           float32  target PPG (same normalization family as x)
  fs      float                   frame rate, needed to turn a waveform into BPM
  subject str, start int          where the clip came from (for per-video evaluation)

Input types (`input_type`):
  "diff"  DiffNormalized frames: (x[t+1] - x[t]) / (x[t+1] + x[t]). Removes the static
          face and keeps the tiny frame-to-frame color change; 3 channels. Label is the
          normalized derivative of the PPG to match. This is PhysNet's default in rPPG-Toolbox.
  "std"   Standardized frames (zero mean, unit std); 3 channels, standardized PPG label.
  "both"  diff + std stacked; 6 channels, diff label.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "UBFC-rPPG"
SPLIT_FILE = ROOT / "configs" / "split_ubfc.json"
EPS = 1e-7


def make_split(n_val=6, n_test=6, seed=42):
    """Split by SUBJECT (never by clip): the same face must not be in train and test."""
    names = sorted((p.stem for p in PROCESSED.glob("subject*.npz")), key=lambda s: int(s[7:]))
    rng = np.random.default_rng(seed)
    names = list(rng.permutation(names))
    split = {
        "seed": seed,
        "test": sorted(names[:n_test], key=lambda s: int(s[7:])),
        "val": sorted(names[n_test : n_test + n_val], key=lambda s: int(s[7:])),
        "train": sorted(names[n_test + n_val :], key=lambda s: int(s[7:])),
    }
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=2))
    return split


def load_split():
    return json.loads(SPLIT_FILE.read_text()) if SPLIT_FILE.exists() else make_split()


def diff_normalize(x):
    """Frame-difference normalization along axis 0; output has one fewer step."""
    d = (x[1:] - x[:-1]) / (x[1:] + x[:-1] + EPS)
    return d / (d.std() + EPS)


def diff_label(y):
    """Normalized first difference of the PPG. Unlike pixels, the PPG is centered on zero,
    so dividing by (y[t+1] + y[t]) as in diff_normalize would blow up near zero crossings."""
    d = np.diff(y)
    return d / (d.std() + EPS)


def standardize(x):
    return (x - x.mean()) / (x.std() + EPS)


class RPPGDataset(Dataset):
    def __init__(self, subjects, clip_len=160, stride=None, input_type="diff", augment=False,
                 min_face_ratio=0.9, root=PROCESSED):
        self.clip_len = clip_len
        self.input_type = input_type
        self.augment = augment
        stride = stride or clip_len
        self.data, self.index = {}, []
        for name in subjects:
            d = np.load(Path(root) / f"{name}.npz")
            self.data[name] = {k: d[k] for k in ("frames", "ppg", "fs", "found")}
            n = len(d["ppg"])
            # Need clip_len + 1 frames: frame differences turn T+1 frames into T steps.
            for start in range(0, n - clip_len, stride):
                if d["found"][start : start + clip_len + 1].mean() >= min_face_ratio:
                    self.index.append((name, start))

    @property
    def in_channels(self):
        return 6 if self.input_type == "both" else 3

    def __len__(self):
        return len(self.index)

    def _read(self, name, start, rng):
        frames, ppg = self.data[name]["frames"], self.data[name]["ppg"]
        n_out = self.clip_len + 1
        speed = rng.uniform(0.8, 1.2) if self.augment else 1.0
        # Speed augmentation: read more/fewer source frames and squeeze them into n_out.
        # The label is resampled identically, so the target HR changes by the same factor.
        n_src = min(int(round(self.clip_len * speed)) + 1, len(ppg) - start)
        if n_src == n_out:
            return frames[start : start + n_out].astype(np.float32), ppg[start : start + n_out].astype(np.float32)
        pos = np.linspace(0, n_src - 1, n_out)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, n_src - 1)
        w = (pos - i0).astype(np.float32)
        src = frames[start : start + n_src].astype(np.float32)
        x = src[i0] * (1 - w)[:, None, None, None] + src[i1] * w[:, None, None, None]
        y = np.interp(pos, np.arange(n_src), ppg[start : start + n_src]).astype(np.float32)
        return x, y

    def _augment_frames(self, x, rng):
        if rng.random() < 0.5:
            x = x[:, :, ::-1]  # horizontal flip
        # Brightness / contrast change for the whole clip (simulates different lighting).
        x = np.clip(x * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15), 0, 255)
        # Small spatial shift (simulates an imperfect face box).
        dy, dx = rng.integers(-4, 5, size=2)
        x = np.roll(x, (dy, dx), axis=(1, 2))
        return np.ascontiguousarray(x)

    def __getitem__(self, i):
        name, start = self.index[i]
        rng = np.random.default_rng()
        x, y = self._read(name, start, rng)
        if self.augment:
            x = self._augment_frames(x, rng)

        if self.input_type == "std":
            inp, label = standardize(x[:-1]), standardize(y[:-1])
        else:
            inp, label = diff_normalize(x), diff_label(y)
            if self.input_type == "both":
                inp = np.concatenate([inp, standardize(x[:-1])], axis=-1)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(inp.transpose(3, 0, 1, 2))),  # (C, T, H, W)
            "y": torch.from_numpy(label.astype(np.float32)),
            "fs": float(self.data[name]["fs"]),
            "subject": name,
            "start": start,
        }
