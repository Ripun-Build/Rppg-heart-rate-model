"""Phase 5: evaluate trained models on the held-out UBFC test subjects.

For every model checkpoint (and the classical baselines GREEN / CHROM / POS on the
same clips) this reports the standard rPPG metrics at two levels:
  clip          one HR per 160-frame clip (~5.4 s), what the live demo will show
  video_fft     one HR per test subject: all clips concatenated, then one FFT
                (rPPG-Toolbox convention). Unreliable when HR changes a lot within the
                video (several spectral peaks of similar height, e.g. subject24).
  video_median  one HR per test subject: median of its clip HRs (robust to HR changes)

It also checks the confidence score: is the SNR of the predicted waveform low exactly
when the BPM is wrong? SNR is computed on validation AND test, so thresholds can be
chosen on validation only.

The first --checkpoint is the main model (chosen on validation in Phase 3). Others are
reported as an ablation, never used to pick the model.

Outputs (outputs/phase5/):
  metrics.csv                  all methods x {clip, video} on the test subjects
  <run>/clips_val.csv, clips_test.csv   per-clip HR, error, SNR (for confidence tuning)
  <run>/bland_altman.png       agreement plot + error vs heart rate
  <run>/per_subject.png        clip MAE per test subject, PhysNet vs baselines
  <run>/confidence.png         |error| vs SNR on validation and test, with the quality thresholds
  quality.csv                  accuracy per confidence label (Good / Fair / Poor), val and test
  <run>/test_waveforms.png     predicted vs true pulse for a few test clips

Run:  python -m src.evaluate                       (main model only)
      python -m src.evaluate --ablation            (+ runs 2 and 3)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.check_preprocess import skin_rgb
from src.dataset import RPPGDataset, load_split
from src.model import PhysNet
from src.pulse import SNR_FAIR, SNR_GOOD, chrom, estimate_bpm, green, pos, quality_label, snr_db, to_pulse
from src.train import plot_examples, predict

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "phase5"
MAIN = ROOT / "checkpoints" / "physnet_ubfc" / "best.pt"
ABLATION = [ROOT / "checkpoints" / "physnet_ubfc_reg" / "best.pt",
            ROOT / "checkpoints" / "physnet_ubfc_ema" / "best.pt"]
BASELINES = {"GREEN": green, "CHROM": chrom, "POS": pos}


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = PhysNet(cfg["model"]["frames"], in_channels=6 if cfg["data"]["input_type"] == "both" else 3,
                    dropout=cfg["model"].get("dropout", 0.0))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck["epoch"]


def metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    err = pred - true
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAPE%": 100 * np.mean(np.abs(err) / true),
        "r": np.corrcoef(pred, true)[0, 1] if len(err) > 2 else np.nan,
        "within5%": 100 * np.mean(np.abs(err) <= 5),
        "n": len(err),
    }


def z(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def clip_table(model, ds, device, batch_size=4):
    """One row per clip: true HR, PhysNet HR + SNR, and each baseline's HR."""
    out = predict(model, DataLoader(ds, batch_size=batch_size), device)
    # Diff-trained models output pulse derivatives: integrate predictions AND labels back to
    # pulse waves once here, so every HR / SNR / plot below works on the same kind of signal.
    is_diff = ds.input_type != "std"
    out["pred"] = np.stack([to_pulse(p, is_diff) for p in out["pred"]])
    out["label"] = np.stack([to_pulse(y, is_diff) for y in out["label"]])
    rows = []
    for i, (subj, start) in enumerate(zip(out["subject"], out["start"])):
        fs = out["fs"][i]
        rgb = skin_rgb(ds.data[subj]["frames"][start : start + ds.clip_len + 1].astype(np.float64))
        row = {"subject": subj, "start": start, "fs": fs,
               "true_hr": estimate_bpm(out["label"][i], fs),
               "PhysNet_hr": estimate_bpm(out["pred"][i], fs),
               "snr_db": snr_db(out["pred"][i], fs)}
        for name, fn in BASELINES.items():
            row[f"{name}_hr"] = estimate_bpm(fn(rgb, fs), fs)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["PhysNet_err"] = df["PhysNet_hr"] - df["true_hr"]
    return df, out


def video_table(ds, out):
    """One row per subject: FFT over all of that subject's clips concatenated in time."""
    rows = []
    for subj in sorted(set(out["subject"]), key=lambda s: int(s[7:])):
        idx = sorted((i for i, s in enumerate(out["subject"]) if s == subj), key=lambda i: out["start"][i])
        fs = out["fs"][idx[0]]
        label = np.concatenate([z(out["label"][i]) for i in idx])
        pred = np.concatenate([z(out["pred"][i]) for i in idx])
        first, last = out["start"][idx[0]], out["start"][idx[-1]] + ds.clip_len + 1
        rgb = skin_rgb(ds.data[subj]["frames"][first:last].astype(np.float64))
        row = {"subject": subj, "minutes": len(label) / fs / 60,
               "true_hr": estimate_bpm(label, fs), "PhysNet_hr": estimate_bpm(pred, fs)}
        for name, fn in BASELINES.items():
            row[f"{name}_hr"] = estimate_bpm(fn(rgb, fs), fs)
        rows.append(row)
    return pd.DataFrame(rows)


def video_median_table(clips):
    """One row per subject: median of the clip-level HRs (true and every method)."""
    cols = ["true_hr", "PhysNet_hr"] + [f"{m}_hr" for m in BASELINES]
    return clips.groupby("subject")[cols].median().reset_index()


def quality_table(clips, name):
    """Accuracy of PhysNet per confidence label."""
    rows = []
    for q in ("Good", "Fair", "Poor"):
        d = clips[clips["quality"] == q]
        e = d["PhysNet_err"].abs()
        rows.append({"set": name, "quality": q, "clips": len(d), "share%": 100 * len(d) / len(clips),
                     "MAE": e.mean() if len(d) else np.nan,
                     "within5%": 100 * (e <= 5).mean() if len(d) else np.nan})
    return rows


def plot_bland_altman(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    t, p = df["true_hr"], df["PhysNet_hr"]
    mean, diff = (t + p) / 2, p - t
    bias, sd = diff.mean(), diff.std()
    axes[0].scatter(mean, diff, s=14, alpha=0.7)
    for y, ls, lab in [(bias, "-", f"bias {bias:+.2f}"), (bias + 1.96 * sd, "--", f"+1.96 SD {bias + 1.96 * sd:+.1f}"),
                       (bias - 1.96 * sd, "--", f"-1.96 SD {bias - 1.96 * sd:+.1f}")]:
        axes[0].axhline(y, color="C3", ls=ls, lw=1, label=lab)
    axes[0].set(title="Bland-Altman: PhysNet vs ground truth (test clips)",
                xlabel="mean of predicted and true HR (BPM)", ylabel="predicted - true (BPM)")
    axes[0].legend(fontsize=8)

    lo, hi = min(t.min(), p.min()) - 5, max(t.max(), p.max()) + 5
    axes[1].scatter(t, p, s=14, alpha=0.7, label="PhysNet")
    axes[1].scatter(t, df["POS_hr"], s=10, alpha=0.4, marker="x", color="0.4", label="POS")
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].set(title="Predicted vs true HR (test clips)", xlabel="true HR (BPM)", ylabel="predicted HR (BPM)",
                xlim=(lo, hi), ylim=(lo, hi))
    axes[1].legend(fontsize=8)

    for col, color, lab in [("PhysNet_hr", "C0", "PhysNet"), ("POS_hr", "0.4", "POS")]:
        e = df[col] - t
        axes[2].scatter(t, e, s=12, alpha=0.6, color=color, label=lab)
        ok = np.abs(e) < 20  # fit the trend on non-outliers so one gross miss doesn't dominate
        slope, icpt = np.polyfit(t[ok], e[ok], 1)
        xs = np.array([t.min(), t.max()])
        axes[2].plot(xs, slope * xs + icpt, color=color, lw=2,
                     label=f"{lab} trend: {slope * 10:+.2f} BPM error per +10 BPM")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set(title="Error vs true heart rate (does it grow at high HR?)", xlabel="true HR (BPM)",
                ylabel="predicted - true (BPM)", ylim=(-25, 25))
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_subject(df, path):
    subjects = sorted(df["subject"].unique(), key=lambda s: int(s[7:]))
    methods = ["PhysNet"] + list(BASELINES)
    x = np.arange(len(subjects))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, m in enumerate(methods):
        maes = [np.mean(np.abs(df.loc[df.subject == s, f"{m}_hr"] - df.loc[df.subject == s, "true_hr"])) for s in subjects]
        ax.bar(x + (k - (len(methods) - 1) / 2) * w, maes, w, label=m)
    ax.set_xticks(x, subjects)
    ax.set(title="Clip-level MAE per test subject", ylabel="MAE (BPM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confidence(val, test, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, df, name in [(axes[0], val, "validation"), (axes[1], test, "test")]:
        bad = df["PhysNet_err"].abs() > 5
        ax.scatter(df.loc[~bad, "snr_db"], df.loc[~bad, "PhysNet_err"].abs(), s=14, label="error <= 5 BPM")
        ax.scatter(df.loc[bad, "snr_db"], df.loc[bad, "PhysNet_err"].abs(), s=24, color="C3", label="error > 5 BPM")
        ax.axhline(5, color="k", ls=":", lw=1)
        ax.axvspan(df["snr_db"].min() - 1, SNR_FAIR, color="C3", alpha=0.08, label=f"Poor (< {SNR_FAIR:g} dB)")
        ax.axvspan(SNR_FAIR, SNR_GOOD, color="C1", alpha=0.08, label="Fair")
        ax.axvspan(SNR_GOOD, df["snr_db"].max() + 1, color="C2", alpha=0.08, label=f"Good (>= {SNR_GOOD:g} dB)")
        ax.set(title=f"Confidence check ({name}, {len(df)} clips)", xlabel="SNR of predicted waveform (dB)")
        ax.set_yscale("symlog", linthresh=5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|predicted - true| (BPM)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_checkpoint(path, split, device, main):
    model, cfg, epoch = load_model(path, device)
    name = cfg["name"]
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]
    test_ds = RPPGDataset(split["test"], clip_len=d["clip_len"], input_type=d["input_type"])
    test_clips, test_out = clip_table(model, test_ds, device)
    test_videos = video_table(test_ds, test_out)
    test_median = video_median_table(test_clips)
    test_clips["quality"] = [quality_label(s) for s in test_clips["snr_db"]]
    test_clips.to_csv(out_dir / "clips_test.csv", index=False)
    test_videos.to_csv(out_dir / "videos_test.csv", index=False)

    if main:
        val_ds = RPPGDataset(split["val"], clip_len=d["clip_len"], input_type=d["input_type"])
        val_clips, _ = clip_table(model, val_ds, device)
        val_clips["quality"] = [quality_label(s) for s in val_clips["snr_db"]]
        val_clips.to_csv(out_dir / "clips_val.csv", index=False)
        pd.DataFrame(quality_table(val_clips, "val") + quality_table(test_clips, "test")).to_csv(
            OUT_DIR / "quality.csv", index=False)
        plot_bland_altman(test_clips, out_dir / "bland_altman.png")
        plot_per_subject(test_clips, out_dir / "per_subject.png")
        plot_confidence(val_clips, test_clips, out_dir / "confidence.png")
        plot_examples(test_out, test_clips["PhysNet_hr"].to_numpy(), test_clips["true_hr"].to_numpy(),
                      out_dir / "test_waveforms.png", n=6)

    label = f"PhysNet ({name}, epoch {epoch})" + ("" if main else " [ablation]")
    rows = [{"method": label, "level": lvl, **metrics(df["PhysNet_hr"], df["true_hr"])}
            for lvl, df in [("clip", test_clips), ("video_fft", test_videos), ("video_median", test_median)]]
    return rows, test_clips, test_videos, test_median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(MAIN), help="main model (chosen on validation)")
    parser.add_argument("--ablation", action="store_true", help="also score runs 2 and 3 on test")
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    split = load_split()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Test subjects: {', '.join(split['test'])}\n")

    rows, test_clips, test_videos, test_median = evaluate_checkpoint(Path(args.checkpoint), split, device, main=True)
    for name in BASELINES:  # baselines don't depend on the model; take them from the main run's tables
        for lvl, df in [("clip", test_clips), ("video_fft", test_videos), ("video_median", test_median)]:
            rows.append({"method": name, "level": lvl, **metrics(df[f"{name}_hr"], df["true_hr"])})
    if args.ablation:
        for path in ABLATION:
            if path.exists():
                rows += evaluate_checkpoint(path, split, device, main=False)[0]

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "metrics.csv", index=False)
    pd.set_option("display.width", 160)
    for lvl in ("clip", "video_fft", "video_median"):
        t = table[table.level == lvl].drop(columns="level").set_index("method").round(2)
        print(f"=== {lvl}-level, test set ===\n{t.to_string()}\n")

    # Is there a heart-rate-dependent bias? (fit on non-outliers)
    ok = test_clips["PhysNet_err"].abs() < 20
    slope = np.polyfit(test_clips.loc[ok, "true_hr"], test_clips.loc[ok, "PhysNet_err"], 1)[0]
    print(f"PhysNet bias {test_clips['PhysNet_err'].mean():+.2f} BPM; error trend {slope * 10:+.2f} BPM per +10 BPM true HR")
    print(f"\nPer-video results, whole-video FFT (main model):\n{test_videos.round(1).to_string(index=False)}")
    q = pd.read_csv(OUT_DIR / "quality.csv").round(2)
    print(f"\nConfidence labels (thresholds chosen on validation: Good >= {SNR_GOOD:g} dB, Poor < {SNR_FAIR:g} dB):\n"
          f"{q.to_string(index=False)}")
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
