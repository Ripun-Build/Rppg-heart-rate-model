"""Phase 2 check: are the face crops good, and does the Dataset produce sane clips?

Checks per subject:
  - data_ok: face found in (almost) every frame and the face box is stable
    (small frame-to-frame jitter, in pixels of the original video). This is the real
    pass/fail test of the preprocessing pipeline.
  - the pulse survived cropping: POS (classical method) run on the crops should give
    roughly the same HR as the ground-truth PPG in each 160-frame window.
    hard_for_pos flags subjects where POS struggles (e.g. head motion). That is a
    property of the video, NOT a pipeline failure; these clips stay in the dataset.
Then builds the train/val/test Datasets and checks shapes, NaNs, and loading speed.

Outputs (outputs/phase2/):
  summary.csv, summary.png      per-subject crop quality + POS HR agreement
  dataset_sample.png            what one training clip and its label look like (no faces)
  private/crops_grid.png        one crop per subject  (shows faces -> git-ignored)

Run:  python -m src.check_preprocess
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import PROCESSED, RPPGDataset, load_split
from src.pulse import estimate_bpm, pos

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase2"
WINDOW = 160
MIN_FACE_FOUND_PCT = 99.0
MAX_JITTER_PX = 2.0
HARD_POS_MAE = 10.0  # BPM; above this the subject is flagged "hard for POS" (not a failure)


def skin_rgb(frames):
    """Mean RGB of the central part of the crop (mostly skin, less hair/background)."""
    return frames[:, 18:54, 18:54].reshape(len(frames), -1, 3).mean(axis=1)


def check_subject(path):
    d = np.load(path)
    frames, ppg, fs, boxes, found = d["frames"], d["ppg"], float(d["fs"]), d["boxes"], d["found"]
    centers = boxes[found, :2]
    jitter = np.linalg.norm(np.diff(centers, axis=0), axis=1)

    pulse = pos(skin_rgb(frames), fs)
    pos_hr, gt_hr, motion = [], [], []
    for s in range(0, len(ppg) - WINDOW + 1, WINDOW):
        pos_hr.append(estimate_bpm(pulse[s : s + WINDOW], fs))
        gt_hr.append(estimate_bpm(ppg[s : s + WINDOW], fs))
        motion.append(np.ptp(boxes[s : s + WINDOW, :2], axis=0).max())  # how far the head travelled
    pos_hr, gt_hr = np.array(pos_hr), np.array(gt_hr)

    row = {
        "subject": path.stem,
        "frames": len(frames),
        "face_found_pct": round(100 * found.mean(), 1),
        "box_side_px": round(float(np.median(boxes[found, 2])), 1),
        "jitter_px_mean": round(float(jitter.mean()), 2),
        "jitter_px_max": round(float(jitter.max()), 1),
        "max_head_motion_px": round(float(max(motion)), 1),
        "pos_mae": round(float(np.mean(np.abs(pos_hr - gt_hr))), 2),
        "pos_within_5bpm_pct": round(100 * float(np.mean(np.abs(pos_hr - gt_hr) <= 5)), 0),
    }
    row["data_ok"] = row["face_found_pct"] > MIN_FACE_FOUND_PCT and row["jitter_px_mean"] < MAX_JITTER_PX
    row["hard_for_pos"] = row["pos_mae"] >= HARD_POS_MAE
    return row, pos_hr, gt_hr, frames[len(frames) // 2]


def plot_summary(df, pos_all, gt_all, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(df))
    names = df["subject"].str.replace("subject", "")
    axes[0].bar(x, df["jitter_px_mean"], color=np.where(df["data_ok"], "C0", "C3"))
    axes[0].set_xticks(x, names)
    axes[0].set(title=f"Face box jitter (red = data problem, limit {MAX_JITTER_PX} px)",
                xlabel="subject", ylabel="mean frame-to-frame move (pixels)")
    axes[0].tick_params(axis="x", labelsize=7, rotation=90)

    axes[1].scatter(gt_all, pos_all, s=10, alpha=0.6)
    axes[1].plot([40, 160], [40, 160], "k--", lw=1, label="perfect agreement")
    within = np.mean(np.abs(pos_all - gt_all) <= 5) * 100
    axes[1].set(title=f"POS on our crops vs ground truth ({within:.0f}% of windows within 5 BPM)",
                xlabel="ground-truth PPG HR (BPM)", ylabel="POS HR from crops (BPM)", xlim=(40, 160), ylim=(40, 160))
    axes[1].legend()

    axes[2].bar(x, df["pos_mae"], color=np.where(df["hard_for_pos"], "C1", "C0"))
    axes[2].axhline(df["pos_mae"].mean(), color="k", ls="--", lw=1, label=f"mean {df['pos_mae'].mean():.2f} BPM")
    axes[2].axhline(HARD_POS_MAE, color="C1", ls=":", lw=1, label=f"orange = hard for POS (>= {HARD_POS_MAE:.0f} BPM)")
    axes[2].set_xticks(x, names)
    axes[2].set(title="POS HR error per subject (160-frame windows)", xlabel="subject", ylabel="MAE (BPM)")
    axes[2].tick_params(axis="x", labelsize=7, rotation=90)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_crops(names, crops, out_path):
    cols = 7
    rows = int(np.ceil(len(crops) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, name, crop in zip(axes.flat, names, crops):
        ax.imshow(crop)
        ax.set_title(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_dataset_sample(ds, out_path):
    item = ds[0]
    x, y, fs = item["x"].numpy(), item["y"].numpy(), item["fs"]
    t = np.arange(len(y)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    # Spatial average of each input channel = what the network "sees" over time, without showing a face.
    for c, color in zip(range(3), "rgb"):
        axes[0].plot(t, x[c].mean(axis=(1, 2)), color=color, lw=1, label=f"input channel {'RGB'[c]} (spatial mean)")
    axes[0].set(title=f"One training clip: {item['subject']} from frame {item['start']} "
                      f"(input {tuple(x.shape)}, type '{ds.input_type}')", xlabel="time (s)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].plot(t, y, color="k")
    axes[1].set(title=f"Target label {tuple(y.shape)}  ->  {estimate_bpm(y, fs):.1f} BPM", xlabel="time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def check_datasets():
    split = load_split()
    print(f"\nSplit (by subject): train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])}")
    print(f"  val:  {', '.join(split['val'])}\n  test: {', '.join(split['test'])}")
    datasets = {
        "train": RPPGDataset(split["train"], stride=80, augment=True),   # overlapping clips + augmentation
        "val": RPPGDataset(split["val"]),
        "test": RPPGDataset(split["test"]),
    }
    for name, ds in datasets.items():
        t0 = time.time()
        items = [ds[i] for i in range(0, len(ds), max(1, len(ds) // 20))]
        dt = (time.time() - t0) / len(items)
        bad = sum(not (np.isfinite(it["x"].numpy()).all() and np.isfinite(it["y"].numpy()).all()) for it in items)
        print(f"  {name:>5}: {len(ds):4d} clips, x {tuple(items[0]['x'].shape)}, y {tuple(items[0]['y'].shape)}, "
              f"NaN/inf clips {bad}, {1000 * dt:.0f} ms/clip")
    return datasets


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "private").mkdir(exist_ok=True)
    paths = sorted(PROCESSED.glob("subject*.npz"), key=lambda p: int(p.stem[7:]))
    print(f"Checking {len(paths)} preprocessed subjects\n")

    results = [check_subject(p) for p in paths]
    df = pd.DataFrame([r[0] for r in results])
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nData OK (face found + stable box): {df['data_ok'].sum()}/{len(df)}")
    hard = df.loc[df["hard_for_pos"], "subject"].tolist()
    print(f"Hard for POS (video property, not a failure; kept for training): {', '.join(hard) or 'none'}")
    print(f"Face found: {df['face_found_pct'].mean():.2f}% of frames | box jitter: {df['jitter_px_mean'].mean():.2f} px/frame")
    pos_all = np.concatenate([r[1] for r in results])
    gt_all = np.concatenate([r[2] for r in results])
    print(f"POS on crops vs ground truth: MAE {np.mean(np.abs(pos_all - gt_all)):.2f} BPM, "
          f"{100 * np.mean(np.abs(pos_all - gt_all) <= 5):.0f}% of windows within 5 BPM")

    plot_summary(df, pos_all, gt_all, OUT_DIR / "summary.png")
    plot_crops(df["subject"], [r[3] for r in results], OUT_DIR / "private" / "crops_grid.png")

    datasets = check_datasets()
    plot_dataset_sample(datasets["val"], OUT_DIR / "dataset_sample.png")
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
