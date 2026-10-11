"""Phase 1: sanity-check UBFC-rPPG before any training.

Checks per subject:
  - video opens, first/last frame decode, frame count == number of PPG samples
  - video FPS agrees with the ground-truth timestamps
  - the PPG label is self-consistent: in each 160-frame window (the clip size we train on)
    HR from the FFT agrees with HR from counting individual beats (peak detection)
  - for reference only: agreement with the oximeter's displayed HR, which lags ~5 s
    and is smoothed, so it is NOT used for pass/fail

Outputs (outputs/phase1/):
  summary.csv            one row per subject
  all_subjects.png       per-window HR agreement across all subjects
  subject_<N>.png        detailed plots for one subject

Run:  python -m src.check_dataset [--subject 1]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.pulse import MAX_HZ, MIN_HZ, _spectrum, bandpass, estimate_bpm, snr_db
from src.ubfc import list_subjects, load_ground_truth, resample_uniform, video_info

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase1"
WINDOW = 160
VALID_HR = (40, 200)  # oximeter readings outside this are dropouts
MAX_LABEL_MAE = 5.0   # BPM; FFT-vs-peaks disagreement above this means a noisy label


def oximeter_hr(hr):
    valid = hr[(hr >= VALID_HR[0]) & (hr <= VALID_HR[1])]
    return float(np.median(valid)) if len(valid) else np.nan


def detect_beats(ppg, fs):
    """Indices of systolic peaks (one per heartbeat) in the filtered PPG."""
    filtered = bandpass(ppg, fs)
    peaks, _ = find_peaks(filtered, distance=fs / MAX_HZ, prominence=0.3 * filtered.std())
    return peaks


def peak_hr(peaks, start, end, fs):
    """HR from the mean beat-to-beat interval of the peaks inside [start, end)."""
    p = peaks[(peaks >= start) & (peaks < end)]
    return 60 * fs / np.mean(np.diff(p)) if len(p) >= 3 else np.nan


def windowed_hr(ppg, hr, fs):
    """Per non-overlapping 160-frame window: FFT HR, beat-count HR, oximeter HR, SNR."""
    peaks = detect_beats(ppg, fs)
    out = {"fft": [], "peaks": [], "oxi": [], "snr": []}
    for start in range(0, len(ppg) - WINDOW + 1, WINDOW):
        clip = ppg[start : start + WINDOW]
        out["fft"].append(estimate_bpm(clip, fs))
        out["peaks"].append(peak_hr(peaks, start, start + WINDOW, fs))
        out["oxi"].append(oximeter_hr(hr[start : start + WINDOW]))
        out["snr"].append(snr_db(clip, fs))
    return {k: np.array(v) for k, v in out.items()}


def mae(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[ok] - b[ok])))


def check_subject(subject_dir):
    gt = load_ground_truth(subject_dir)
    vid = video_info(subject_dir)
    ppg = resample_uniform(gt["ppg"], gt["ts"], gt["fs"])
    w = windowed_hr(ppg, gt["hr"], gt["fs"])
    row = {
        "subject": subject_dir.name,
        "frames": vid["frames"],
        "ppg_samples": len(gt["ppg"]),
        "duration_s": round(gt["ts"][-1] - gt["ts"][0], 1),
        "video_fps": round(vid["fps"], 2),
        "gt_fs": round(gt["fs"], 2),
        "readable": vid["readable"],
        "oximeter_hr": round(oximeter_hr(gt["hr"]), 1),
        "fft_hr": round(estimate_bpm(ppg, gt["fs"]), 1),
        "hr_dropout_pct": round(100 * np.mean((gt["hr"] < VALID_HR[0]) | (gt["hr"] > VALID_HR[1])), 1),
        "n_windows": len(w["fft"]),
        "label_mae": round(mae(w["fft"], w["peaks"]), 2),     # FFT vs beat counting (pass/fail)
        "label_snr_db": round(float(np.median(w["snr"])), 1),
        "oximeter_mae": round(mae(w["fft"], w["oxi"]), 2),    # reference only
    }
    row["frames_match"] = row["frames"] == row["ppg_samples"]
    row["fps_match"] = abs(row["video_fps"] - row["gt_fs"]) < 1.0
    row["label_ok"] = row["label_mae"] < MAX_LABEL_MAE
    row["PASS"] = row["readable"] and row["frames_match"] and row["fps_match"] and row["label_ok"]
    return row, w


def plot_subject(subject_dir, out_path):
    gt = load_ground_truth(subject_dir)
    fs = gt["fs"]
    ppg = resample_uniform(gt["ppg"], gt["ts"], fs)
    filtered = bandpass(ppg, fs)
    peaks = detect_beats(ppg, fs)
    t = np.arange(len(ppg)) / fs
    freqs, power = _spectrum(filtered, fs)
    w = windowed_hr(ppg, gt["hr"], fs)
    w_t = (np.arange(len(w["fft"])) * WINDOW + WINDOW / 2) / fs

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    first = t <= 10
    p10 = peaks[t[peaks] <= 10]
    axes[0].plot(t[first], ppg[first], color="0.6", label="raw PPG")
    axes[0].plot(t[first], filtered[first], color="C3", label=f"bandpass {MIN_HZ}-{MAX_HZ} Hz")
    axes[0].plot(t[p10], filtered[p10], "kv", label="detected beats")
    axes[0].set(title=f"{subject_dir.name}: ground-truth PPG (first 10 s)", xlabel="time (s)")
    axes[0].legend(loc="upper right")

    axes[1].plot(freqs * 60, power / power.max(), color="C0")
    axes[1].axvline(freqs[np.argmax(power)] * 60, color="C3", ls="--",
                    label=f"FFT peak {freqs[np.argmax(power)] * 60:.1f} BPM")
    axes[1].set(title="Spectrum of whole-video PPG (several peaks = HR changed during the video)",
                xlabel="BPM", ylabel="normalized power")
    axes[1].legend(loc="upper right")

    axes[2].plot(gt["ts"], np.where(gt["hr"] >= VALID_HR[0], gt["hr"], np.nan), color="C2",
                 alpha=0.7, label="oximeter display HR (lagged, reference)")
    axes[2].plot(w_t, w["fft"], "o-", color="C3", label="PPG: FFT HR")
    axes[2].plot(w_t, w["peaks"], "s--", color="k", label="PPG: beat-count HR")
    axes[2].set(title=f"Heart rate per {WINDOW}-frame window (label MAE {mae(w['fft'], w['peaks']):.2f} BPM)",
                xlabel="time (s)", ylabel="BPM")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_all(df, windows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fft = np.concatenate([w["fft"] for w in windows])
    pk = np.concatenate([w["peaks"] for w in windows])
    ok = ~np.isnan(pk)
    lo, hi = 40, 160
    axes[0].scatter(pk[ok], fft[ok], s=10, alpha=0.6)
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect agreement")
    axes[0].set(title=f"All {ok.sum()} windows: two HR estimates from the PPG label",
                xlabel="beat-count HR (BPM)", ylabel="FFT HR (BPM)", xlim=(lo, hi), ylim=(lo, hi))
    axes[0].legend()

    x = np.arange(len(df))
    colors = np.where(df["PASS"], "C0", "C3")
    axes[1].bar(x - 0.2, df["label_mae"], 0.4, color=colors, label="label: FFT vs beat count (pass/fail)")
    axes[1].bar(x + 0.2, df["oximeter_mae"], 0.4, color="0.75", label="FFT vs oximeter (reference)")
    axes[1].axhline(MAX_LABEL_MAE, color="C3", ls="--", lw=1, label=f"fail threshold {MAX_LABEL_MAE} BPM")
    axes[1].set_xticks(x, df["subject"].str.replace("subject", ""))
    axes[1].set(title=f"Per-subject mean error over {WINDOW}-frame windows", xlabel="subject", ylabel="MAE (BPM)")
    axes[1].tick_params(axis="x", labelsize=7, rotation=90)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1, help="subject number for the detailed plot")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = list_subjects()
    print(f"Found {len(subjects)} subjects\n")
    rows, windows = zip(*[check_subject(s) for s in subjects])
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)

    cols = ["subject", "frames", "video_fps", "gt_fs", "fft_hr", "label_mae", "label_snr_db",
            "oximeter_mae", "hr_dropout_pct", "PASS"]
    print(df[cols].to_string(index=False))
    print(f"\nPassed: {df['PASS'].sum()}/{len(df)}")
    print(f"Label check (FFT vs beat count), mean MAE: {df['label_mae'].mean():.2f} BPM")
    print(f"Reference (FFT vs oximeter), mean MAE:    {df['oximeter_mae'].mean():.2f} BPM")
    print(f"Frame rates: {sorted(df['video_fps'].round().astype(int).unique().tolist())} fps")
    print(f"Total: {df['n_windows'].sum()} windows, {df['duration_s'].sum() / 60:.1f} min")

    plot_all(df, windows, OUT_DIR / "all_subjects.png")
    detail = next(s for s in subjects if s.name == f"subject{args.subject}")
    plot_subject(detail, OUT_DIR / f"subject_{args.subject}.png")
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
