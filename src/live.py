"""Phase 6: live heart rate from a webcam (or a video file, for testing).

Per frame:   face tracker -> 72x72 crop -> ring buffer of (timestamp, crop)
Every ~1 s:  last 161 crops resampled to exactly 30 fps -> DiffNormalized -> PhysNet
             -> integrate (to_pulse) -> FFT BPM + SNR -> Good / Fair / Poor
Display:     median (= mean) of the last 2 non-Poor estimates. Chosen on the validation videos:
             a median of 5 lagged 2-3 s behind fast HR changes (val MAE 2.42 vs 1.87 BPM);
             'Good only' scored well but would freeze the number whenever light is only 'Fair'.

The crop, normalization and BPM/SNR code are the same functions used for training and
evaluation, so the network sees exactly what it was trained on.

Measurement mode (default): record --duration seconds (30), then show ONE final heart rate =
median of the ~1 s window estimates, with a range and a confidence level. Each measurement is
logged (numbers only) to outputs/phase6/sessions_webcam.csv, optionally with a reference reading
typed on the result screen (smartwatch / pulse oximeter) to measure real-world accuracy.

Run:
  python -m src.live                                   # webcam 0, 30 s measurement
  python -m src.live --duration 20                     # shorter measurement
  python -m src.live --continuous                      # live BPM every second instead
  python -m src.live --video data/raw/UBFC-rPPG/subject8/vid.avi            # replay a video
  python -m src.live --test-sessions                   # headless: simulated 30 s measurements on UBFC
  python -m src.live --diagnose                        # your webcam log vs UBFC reference values
  python -m src.live --test-ubfc                       # headless: continuous mode on UBFC test videos
Keys: r = measure again, q / Esc (or the window's close button) = quit; on the result screen
type a reference BPM from a smartwatch / oximeter (optional) before pressing r or q.
"""
import argparse
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from src.dataset import diff_normalize
from src.evaluate import MAIN, load_model
from src.face import FaceTracker, crop_face
from src.pulse import SNR_GOOD, bandpass, estimate_bpm, pos, quality_label, snr_db, to_pulse

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "phase6"
TARGET_FS = 30.0          # UBFC training videos are ~29-30 fps; resample everything to this
CLIP_LEN = 160            # frames per network input (161 crops -> 160 differences)
QUALITY_COLORS = {"Good": (60, 180, 60), "Fair": (0, 165, 255), "Poor": (60, 60, 230)}


def resample_crops(times, crops, grid):
    """Linearly interpolate crops captured at `times` onto the time points `grid`."""
    idx = np.clip(np.searchsorted(times, grid) - 1, 0, len(times) - 2)
    w = np.clip((grid - times[idx]) / (times[idx + 1] - times[idx] + 1e-9), 0, 1).astype(np.float32)
    return crops[idx] * (1 - w)[:, None, None, None] + crops[idx + 1] * w[:, None, None, None]


def poor_reason(snr, found, luma):
    """Why an estimate is not 'Good', in words for the user (empty if it is Good)."""
    if found < 0.9:
        return "face lost - look at the camera"
    if luma < 40:
        return "too dark - add light on your face"
    if luma > 220:
        return "too bright - avoid direct light"
    if snr < SNR_GOOD:
        return "weak pulse signal - keep still, even light on face"
    return ""


class PulseMonitor:
    """Turns a stream of (frame, timestamp) into heart-rate estimates."""

    def __init__(self, checkpoint=MAIN, device=None, infer_every_s=1.0, smooth_n=2, background=True,
                 keep_all=False):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model, _, _ = load_model(checkpoint, self.device)
        self.tracker = FaceTracker()
        self.window_s = (CLIP_LEN + 1) / TARGET_FS          # 5.37 s of video per estimate
        self.buf = deque()                                  # (t, crop uint8, face_found)
        self.infer_every_s = infer_every_s
        self.last_infer_t = -np.inf
        self.history = deque(maxlen=smooth_n)               # recent non-Poor BPMs (display smoothing)
        self.result = None                                  # latest estimate (dict)
        self.results = []                                   # every estimate made so far
        self.keep_all = keep_all                            # measurement mode: keep every crop
        self.all_frames = []                                # (t, crop, face_found) for the whole session
        self.box_sides = []                                 # face box side (px) per stored frame
        self.frame_w = None
        self.background = background
        self._busy = False
        self._thread = None
        self._lock = threading.Lock()

    def add_frame(self, frame_bgr, t):
        """Feed one frame captured at time t (seconds). Returns the current face box or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # trained on RGB crops
        box = self.tracker.update(rgb)
        if box is not None:
            item = (t, crop_face(rgb, box), self.tracker.found)
            self.buf.append(item)
            if self.keep_all:
                self.all_frames.append(item)
                self.box_sides.append(float(box[2]))
                self.frame_w = frame_bgr.shape[1]
        while self.buf and self.buf[0][0] < t - self.window_s - 1.0:
            self.buf.popleft()

        ready = self.buf and self.buf[-1][0] - self.buf[0][0] >= self.window_s
        if ready and t - self.last_infer_t >= self.infer_every_s and not self._busy:
            self.last_infer_t = t
            clip, found, luma = self._resampled_clip()
            if self.background:
                self._busy = True
                self._thread = threading.Thread(target=self._infer, args=(clip, found, luma, t), daemon=True)
                self._thread.start()
            else:
                self._infer(clip, found, luma, t)
        return box

    def progress(self):
        """0..1, how full the buffer is before the first estimate."""
        if not self.buf:
            return 0.0
        return min(1.0, (self.buf[-1][0] - self.buf[0][0]) / self.window_s)

    def _resampled_clip(self):
        """Linearly interpolate the buffered crops onto an even 30 fps grid ending now.
        Webcams drift (e.g. 15 fps in dim light); the network expects ~30 fps."""
        times = np.array([b[0] for b in self.buf])
        crops = np.stack([b[1] for b in self.buf]).astype(np.float32)
        grid = times[-1] - np.arange(CLIP_LEN, -1, -1) / TARGET_FS
        clip = resample_crops(times, crops, grid)
        recent = times >= grid[0]
        found = float(np.mean([b[2] for b, r in zip(self.buf, recent) if r]))
        luma = float(np.mean(clip @ np.array([0.299, 0.587, 0.114], np.float32)))
        return clip, found, luma

    def _infer(self, clip, found, luma, t):
        try:
            x = torch.from_numpy(diff_normalize(clip).transpose(3, 0, 1, 2)[None].copy()).to(self.device)
            with torch.no_grad():
                pred = self.model(x)[0].cpu().numpy()
            pulse = to_pulse(pred, is_diff=True)
            bpm, snr = estimate_bpm(pulse, TARGET_FS), snr_db(pulse, TARGET_FS)
            quality = quality_label(snr, found, luma)
            with self._lock:
                if quality != "Poor":
                    self.history.append(bpm)
                self.result = {
                    "t": t, "bpm_raw": bpm, "snr_db": snr, "quality": quality, "reason": poor_reason(snr, found, luma),
                    "face_found": found, "luma": luma,
                    "bpm": float(np.median(self.history)) if self.history else None,
                    "wave": bandpass(pulse, TARGET_FS),
                }
                self.results.append(self.result)
        finally:
            self._busy = False

    def wait(self):
        """Block until the background estimate (if any) has finished."""
        if self._thread is not None:
            self._thread.join(timeout=5)

    def close(self):
        self.wait()  # never tear down while the model is still running in the worker thread
        self.tracker.close()


def session_pulse(mon):
    """Pulse wave for the whole session: every crop resampled to 30 fps, cut into consecutive
    161-frame clips, run through the network, integrated and joined end to end."""
    if len(mon.all_frames) < 2:
        return None
    times = np.array([f[0] for f in mon.all_frames])
    crops = np.stack([f[1] for f in mon.all_frames]).astype(np.float32)
    grid = np.arange(times[0], times[-1], 1 / TARGET_FS)
    n_clips = (len(grid) - 1) // CLIP_LEN
    if n_clips == 0:
        return None
    frames = resample_crops(times, crops, grid[: n_clips * CLIP_LEN + 1])
    x = np.stack([diff_normalize(frames[k * CLIP_LEN : (k + 1) * CLIP_LEN + 1]).transpose(3, 0, 1, 2)
                  for k in range(n_clips)])
    with torch.no_grad():
        pred = mon.model(torch.from_numpy(x).to(mon.device)).cpu().numpy()
    parts = [to_pulse(q, is_diff=True) for q in pred]
    return np.concatenate([(q - q.mean()) / (q.std() + 1e-8) for q in parts])


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())])


def final_candidates(mon):
    """Different ways to turn one measurement session into a single BPM (compared on validation)."""
    res = mon.results
    if not res:
        return {}
    bpm = np.array([r["bpm_raw"] for r in res])
    snr = np.array([r["snr_db"] for r in res])
    q = np.array([r["quality"] for r in res])
    ok = q != "Poor"
    good = q == "Good"
    out = {
        "median of all windows": float(np.median(bpm)),
        "median of non-Poor windows": float(np.median(bpm[ok])) if ok.any() else float(np.median(bpm)),
        "median of Good windows": float(np.median(bpm[good])) if good.any() else float(np.median(bpm[ok] if ok.any() else bpm)),
        "SNR-weighted median": weighted_median(bpm, 10 ** (snr / 10)),
    }
    pulse = session_pulse(mon)
    if pulse is not None:
        out["one FFT over whole session"] = estimate_bpm(pulse, TARGET_FS)
    return out


def session_diagnostics(mon):
    """Numbers that explain a weak signal (all also computed on UBFC for comparison):
      face_px          face box side in pixels (larger face = more skin pixels = less noise)
      face_frac        face box side / frame width
      light_var_pct    std of the skin brightness over the session, % of its mean
                       (camera auto-exposure 'hunting' or flickering light make this large)
      pos_snr_db       SNR of the classical POS method on the SAME crops. Model-independent:
                       if POS is weak too, the camera/lighting signal is weak; if POS is fine
                       but PhysNet is weak, the model doesn't transfer to this camera.
      pos_bpm          median POS heart rate over the session (second opinion)
    """
    from src.check_preprocess import skin_rgb

    if len(mon.all_frames) < CLIP_LEN + 1:
        return {}
    times = np.array([f[0] for f in mon.all_frames])
    rgb = skin_rgb(np.stack([f[1] for f in mon.all_frames]).astype(np.float64))
    grid = np.arange(times[0], times[-1], 1 / TARGET_FS)
    rgb30 = np.stack([np.interp(grid, times, rgb[:, c]) for c in range(3)], axis=1)
    pulse = pos(rgb30, TARGET_FS)
    starts = range(0, len(pulse) - CLIP_LEN, int(TARGET_FS))
    pos_bpm = [estimate_bpm(pulse[k : k + CLIP_LEN + 1], TARGET_FS) for k in starts]
    pos_snr = [snr_db(pulse[k : k + CLIP_LEN + 1], TARGET_FS) for k in starts]
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    face_px = float(np.median(mon.box_sides)) if mon.box_sides else np.nan
    return {
        "face_px": face_px,
        "face_frac": face_px / mon.frame_w if mon.frame_w else np.nan,
        "light_var_pct": float(100 * luma.std() / luma.mean()),
        "pos_snr_db": float(np.median(pos_snr)) if pos_snr else np.nan,
        "pos_bpm": float(np.median(pos_bpm)) if pos_bpm else np.nan,
    }


def simulate_sessions(split_name="val", duration=30.0, checkpoint=MAIN):
    """Simulate measurement sessions on UBFC videos: a fresh PulseMonitor per session (tracking
    from scratch), sessions starting at 0 s and 30 s. Truth = average HR from beat counting on
    the ground-truth PPG over the same span."""
    import pandas as pd

    from src.check_dataset import detect_beats
    from src.dataset import load_split
    from src.ubfc import UBFC_ROOT, load_ground_truth, resample_uniform

    rows = []
    for subj in load_split()[split_name]:
        gt = load_ground_truth(UBFC_ROOT / subj)
        fs = gt["fs"]
        ppg = resample_uniform(gt["ppg"], gt["ts"], fs)
        beats = detect_beats(ppg, fs)
        for start in (0.0, 30.0):
            f0, f1 = int(start * fs), int((start + duration) * fs)
            if f1 > len(ppg):
                continue
            cap = cv2.VideoCapture(str(UBFC_ROOT / subj / "vid.avi"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            mon = PulseMonitor(checkpoint, background=False, keep_all=True)
            for i in range(f1 - f0):
                ok, frame = cap.read()
                if not ok:
                    break
                mon.add_frame(frame, i / fs)
            cap.release()
            b = beats[(beats >= f0) & (beats < f1)]
            true = 60 * fs * (len(b) - 1) / (b[-1] - b[0])
            row = {"subject": subj, "start_s": start, "true_hr": true, "n_estimates": len(mon.results),
                   **{f"pct_{k}": 100 * np.mean([r["quality"] == k for r in mon.results]) for k in ("Good", "Fair", "Poor")},
                   **final_candidates(mon),
                   "median_snr_db": float(np.median([r["snr_db"] for r in mon.results])),
                   **session_diagnostics(mon)}
            rows.append(row)
            mon.close()
    return pd.DataFrame(rows)


def draw(frame, box, mon, fps):
    h, w = frame.shape[:2]
    if box is not None:
        cx, cy, side = box
        cv2.rectangle(frame, (int(cx - side / 2), int(cy - side / 2)), (int(cx + side / 2), int(cy + side / 2)),
                      (255, 255, 255), 1)
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 92), (0, 0, 0), -1)
    cv2.rectangle(panel, (0, h - 80), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)

    r = mon.result
    if box is None:
        cv2.putText(frame, "No face found", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 230), 2)
    elif r is None:
        cv2.putText(frame, f"Measuring... {int(100 * mon.progress())}%  (hold still)", (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        color = QUALITY_COLORS[r["quality"]]
        txt = f"{r['bpm']:.0f} BPM" if r["bpm"] is not None else "-- BPM"
        cv2.putText(frame, txt, (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        label = r["quality"] + ("  - hold still / more light" if r["quality"] == "Poor" else "")
        cv2.putText(frame, label, (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"SNR {r['snr_db']:+.1f} dB   raw {r['bpm_raw']:.0f}", (250, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        wave = r["wave"]
        wave = (wave - wave.min()) / (np.ptp(wave) + 1e-9)
        pts = np.stack([np.linspace(10, w - 10, len(wave)), h - 12 - wave * 56], axis=1).astype(np.int32)
        cv2.polylines(frame, [pts], False, color, 2)
    cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "Not a medical device", (w - 185, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


WINDOW = "rPPG heart rate"


def open_source(source, is_file):
    cap = cv2.VideoCapture(source) if is_file else cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. On macOS: System Settings > Privacy & Security > Camera, "
                         "allow the app you are running this from (e.g. Visual Studio Code), then restart it.")
    return cap


def window_closed():
    """True if the user clicked the window's red close button."""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def shutdown(cap, mon=None):
    """Clean exit: finish the background estimate, free the camera, and really close the window.
    On macOS destroyAllWindows() only takes effect while window events are processed, so pump
    them a few times; otherwise the window stays on screen frozen ('not responding')."""
    if mon is not None:
        mon.close()
    cap.release()
    cv2.destroyAllWindows()
    for _ in range(10):
        cv2.waitKey(1)


def run_camera(source, checkpoint, is_file):
    cap = open_source(source, is_file)
    file_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else None
    mon = PulseMonitor(checkpoint)
    stamps, i, t0 = deque(maxlen=30), 0, time.perf_counter()
    print("Press q or Esc to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Webcam: real capture time. Video file: its own timeline, so replay speed doesn't matter.
            t = i / file_fps if is_file else time.perf_counter() - t0
            i += 1
            stamps.append(time.perf_counter())
            box = mon.add_frame(frame, t)
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0] + 1e-9) if len(stamps) > 1 else 0.0
            draw(frame, box, mon, fps)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(int(1000 / file_fps) if is_file else 1) & 0xFF
            if key in (ord("q"), 27) or window_closed():
                break
    finally:
        shutdown(cap, mon)

def final_estimate(mon):
    """The single number shown at the end of a measurement.

    Median of all ~1 s window estimates in the session: best on the UBFC validation sessions
    (MAE 0.94 BPM vs 0.95 non-Poor only, 1.01 SNR-weighted, 1.47 one FFT over the session);
    the median already ignores the few bad windows. The spread of the window estimates gives
    the range and a confidence level (heuristic thresholds, to be checked on webcam sessions).
    """
    res = mon.results
    if not res:
        return None
    bpm = np.array([r["bpm_raw"] for r in res])
    q = np.array([r["quality"] for r in res])
    p25, p50, p75 = np.percentile(bpm, [25, 50, 75])
    spread, good = p75 - p25, float(np.mean(q == "Good"))
    confidence = "High" if spread <= 5 and good >= 0.5 else "Medium" if spread <= 10 else "Low"
    reasons = [r["reason"] for r in res if r["reason"]]
    return {"bpm": float(p50), "low": float(p25), "high": float(p75), "spread": float(spread),
            "confidence": confidence, "n": len(res),
            **{f"pct_{k}": 100 * float(np.mean(q == k)) for k in ("Good", "Fair", "Poor")},
            "median_snr_db": float(np.median([r["snr_db"] for r in res])),
            "mean_luma": float(np.mean([r["luma"] for r in res])),
            "face_found": float(np.mean([r["face_found"] for r in res])),
            "main_issue": max(set(reasons), key=reasons.count) if reasons else "",
            **session_diagnostics(mon)}


CONF_COLORS = {"High": (60, 180, 60), "Medium": (0, 165, 255), "Low": (60, 60, 230)}


def _overlay(frame, top, bottom):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, top), (0, 0, 0), -1)
    if bottom:
        cv2.rectangle(panel, (0, h - bottom), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(panel, 0.6, frame, 0.4, 0)


def draw_measure(frame, box, mon, elapsed, duration):
    h, w = frame.shape[:2]
    if box is not None:
        cx, cy, side = box
        cv2.rectangle(frame, (int(cx - side / 2), int(cy - side / 2)), (int(cx + side / 2), int(cy + side / 2)),
                      (255, 255, 255), 1)
    _overlay(frame, 92, 80)
    if box is None:
        cv2.putText(frame, "Look at the camera", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        return
    cv2.putText(frame, f"Measuring... {int(elapsed)} / {int(duration)} s   keep still", (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, (16, 56), (w - 16, 70), (90, 90, 90), -1)
    cv2.rectangle(frame, (16, 56), (16 + int((w - 32) * min(1.0, elapsed / duration)), 70), (60, 180, 60), -1)
    r = mon.result
    if r is not None:
        status = "signal OK" if not r["reason"] else r["reason"]
        cv2.putText(frame, status, (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, QUALITY_COLORS[r["quality"]], 1)
        wave = r["wave"]
        wave = (wave - wave.min()) / (np.ptp(wave) + 1e-9)
        pts = np.stack([np.linspace(10, w - 10, len(wave)), h - 12 - wave * 56], axis=1).astype(np.int32)
        cv2.polylines(frame, [pts], False, (200, 200, 200), 2)


def draw_result(frame, est):
    h, w = frame.shape[:2]
    _overlay(frame, h, 0)
    if est is None:
        cv2.putText(frame, "No measurement - face not found", (16, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (60, 60, 230), 2)
    else:
        cv2.putText(frame, "Your heart rate", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        cv2.putText(frame, f"{est['bpm']:.0f} BPM", (16, 160), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6)
        cv2.putText(frame, f"range {est['low']:.0f} - {est['high']:.0f} BPM", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        cv2.putText(frame, f"confidence: {est['confidence']}", (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    CONF_COLORS[est["confidence"]], 2)
        if est["confidence"] != "High" and est["main_issue"]:
            cv2.putText(frame, f"tip: {est['main_issue']}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 1)
    cv2.putText(frame, "r = measure again    q = quit", (16, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Not a medical device", (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def save_session(est):
    """Append one measurement to the webcam log right away (numbers only, no images)."""
    import pandas as pd

    log = OUT_DIR / "sessions_webcam.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([est])
    # Rewrite instead of appending, so rows logged before new columns were added stay aligned.
    df = pd.concat([pd.read_csv(log), row], ignore_index=True) if log.exists() else row
    df.to_csv(log, index=False)


def result_screen(last, est):
    """Show the final heart rate; the user may type a reference BPM (smartwatch / oximeter).
    Returns ('again' | 'quit', reference_bpm or None)."""
    typed = ""
    while True:
        shown = last.copy()
        draw_result(shown, est)
        if est is not None:
            cv2.putText(shown, f"watch / oximeter BPM (optional): {typed}_", (20, 330), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            cv2.putText(shown, "type digits, Backspace to fix", (20, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1)
        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(50) & 0xFF
        ref = float(typed) if typed else None
        if window_closed() or key in (ord("q"), 27):
            return "quit", ref
        if key == ord("r"):
            return "again", ref
        if chr(key).isdigit() and len(typed) < 3:
            typed += chr(key)
        elif key in (8, 127):  # Backspace / Delete
            typed = typed[:-1]


def run_measure(source, checkpoint, is_file, duration):
    """Measurement mode: record `duration` seconds, then show one final heart rate."""
    import datetime

    cap = open_source(source, is_file)
    file_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else None
    i, t0, n_saved, mon = 0, time.perf_counter(), 0, None
    print(f"Measurement mode: {duration:.0f} s per measurement. Press q or Esc to quit.")
    try:
        while True:
            mon = PulseMonitor(checkpoint, keep_all=True)
            t_start, frame_times, last, quit_now = None, [], None, False
            while True:  # measuring
                ok, frame = cap.read()
                if not ok:
                    break
                t = i / file_fps if is_file else time.perf_counter() - t0
                i += 1
                box = mon.add_frame(frame, t)
                if box is not None and t_start is None:
                    t_start = t  # the clock starts once a face is found
                elapsed = 0.0 if t_start is None else t - t_start
                frame_times.append(t)
                last = frame.copy()  # clean frame (no overlay) for the result screen
                draw_measure(frame, box, mon, elapsed, duration)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(int(1000 / file_fps) if is_file else 1) & 0xFF
                if key in (ord("q"), 27) or window_closed():
                    quit_now = True
                    break
                if elapsed >= duration:
                    break
            if quit_now or last is None:
                break  # quit mid-measurement: nothing to report
            mon.wait()  # let the last background estimate finish
            est = final_estimate(mon)
            mon.close()
            mon = None
            action, ref = result_screen(last, est)
            if est is not None:
                ft = np.diff(frame_times)
                est.update({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "source": "video" if is_file else f"camera {source}", "duration_s": duration,
                            "mean_fps": float(1 / np.mean(ft)) if len(ft) else np.nan,
                            "reference_bpm": ref if ref is not None else np.nan})
                save_session(est)
                n_saved += 1
                print(f"  measurement {n_saved}: {est['bpm']:.0f} BPM (range {est['low']:.0f}-{est['high']:.0f}), "
                      f"confidence {est['confidence']}, reference {ref if ref is not None else '-'}, "
                      f"Good/Fair/Poor {est['pct_Good']:.0f}/{est['pct_Fair']:.0f}/{est['pct_Poor']:.0f}%, "
                      f"SNR {est['median_snr_db']:+.1f} dB, light {est['mean_luma']:.0f}, "
                      f"face {100 * est['face_found']:.0f}%, {est['mean_fps']:.0f} fps")
                print_diagnostics(est)
            if action == "quit":
                break
    finally:
        shutdown(cap, mon)
    if n_saved:
        print(f"Saved {n_saved} measurement(s) to {OUT_DIR / 'sessions_webcam.csv'} (numbers only, no images).")

UBFC_REF_FILE = OUT_DIR / "ubfc_reference_diagnostics.csv"


def ubfc_reference():
    """Median diagnostics over the simulated 30 s UBFC test sessions (computed once, cached)."""
    import pandas as pd

    if not UBFC_REF_FILE.exists():
        print("Computing UBFC reference values once (about 1 minute)...")
        df = simulate_sessions("test")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "sessions_test_ubfc.csv", index=False)
        df[["median_snr_db", "face_px", "face_frac", "light_var_pct", "pos_snr_db"]].median().to_frame("ubfc").T.to_csv(
            UBFC_REF_FILE, index=False)
    return pd.read_csv(UBFC_REF_FILE).iloc[0]


def print_diagnostics(est):
    """One-line-per-cue comparison of a measurement with the UBFC training-style videos."""
    try:
        ref = ubfc_reference()
    except Exception as e:  # never let diagnostics break a measurement
        print(f"    (diagnostics unavailable: {e})")
        return
    if "face_px" not in est:
        return
    rows = [
        ("PhysNet signal (SNR)", f"{est['median_snr_db']:+.1f} dB", f"{ref['median_snr_db']:+.1f} dB"),
        ("POS signal, same frames", f"{est['pos_snr_db']:+.1f} dB", f"{ref['pos_snr_db']:+.1f} dB"),
        ("face size", f"{est['face_px']:.0f} px ({100 * est['face_frac']:.0f}% of width)",
         f"{ref['face_px']:.0f} px ({100 * ref['face_frac']:.0f}%)"),
        ("brightness variation", f"{est['light_var_pct']:.1f} %", f"{ref['light_var_pct']:.1f} %"),
    ]
    print(f"    {'diagnostic':24s} {'this measurement':24s} UBFC typical")
    for name, mine, theirs in rows:
        print(f"    {name:24s} {mine:24s} {theirs}")
    print(f"    POS (classical) says {est['pos_bpm']:.0f} BPM, PhysNet says {est['bpm']:.0f} BPM")


def diagnose():
    """Summarize every logged webcam measurement next to UBFC reference values."""
    import pandas as pd

    log = OUT_DIR / "sessions_webcam.csv"
    if not log.exists():
        raise SystemExit("No webcam measurements logged yet - run a measurement first.")
    d = pd.read_csv(log)
    ref = ubfc_reference()
    cols = ["time", "bpm", "confidence", "reference_bpm", "median_snr_db", "pos_snr_db", "pos_bpm",
            "face_px", "light_var_pct", "mean_luma", "face_found", "mean_fps"]
    pd.set_option("display.width", 250)
    print(d[[c for c in cols if c in d]].round(2).to_string(index=False))
    print("\nUBFC typical:", ", ".join(f"{k} {v:.2f}" for k, v in ref.items()))
    has = d.dropna(subset=["reference_bpm"]) if "reference_bpm" in d else d.iloc[0:0]
    if len(has):
        e = (has.bpm - has.reference_bpm).abs()
        print(f"\nAccuracy vs your reference readings ({len(has)}): MAE {e.mean():.1f} BPM, max {e.max():.1f}")
        if "pos_bpm" in has and has.pos_bpm.notna().any():
            print(f"  POS on the same frames: MAE {(has.pos_bpm - has.reference_bpm).abs().mean():.1f} BPM")
    if "pos_snr_db" in d and d.pos_snr_db.notna().any():
        recent = d.dropna(subset=["pos_snr_db"])
        weak_pos = recent.pos_snr_db.median() < ref["pos_snr_db"] - 2
        weak_net = recent.median_snr_db.median() < ref["median_snr_db"] - 2
        print("\nReading:", "camera/lighting signal itself is weak (POS weak too)" if weak_pos and weak_net else
              "model-specific: POS is fine but PhysNet is weak on this camera" if weak_net else
              "signal strength is close to UBFC")


def test_ubfc(checkpoint, split_name="test"):
    """Run the live pipeline (no display, synchronous) on the held-out UBFC test videos and
    compare every displayed BPM with the ground-truth PPG over the same 5.4 s window."""
    import pandas as pd

    from src.dataset import load_split
    from src.ubfc import UBFC_ROOT, load_ground_truth, resample_uniform

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for subj in load_split()[split_name]:
        gt = load_ground_truth(UBFC_ROOT / subj)
        ppg = resample_uniform(gt["ppg"], gt["ts"], gt["fs"])
        cap = cv2.VideoCapture(str(UBFC_ROOT / subj / "vid.avi"))
        fps = gt["fs"]
        mon, i, t_start = PulseMonitor(checkpoint, background=False), 0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            prev = mon.result
            mon.add_frame(frame, t)
            if mon.result is not prev:  # a new estimate was made on this frame
                r = mon.result
                n = int(round(mon.window_s * fps))
                true = estimate_bpm(ppg[max(0, i + 1 - n) : i + 1], fps)
                rows.append({"subject": subj, "t": t, "true_hr": true, "bpm_raw": r["bpm_raw"],
                             "bpm_shown": r["bpm"], "quality": r["quality"], "snr_db": r["snr_db"]})
            i += 1
        cap.release()
        mon.close()
        d = pd.DataFrame([r for r in rows if r["subject"] == subj])
        print(f"{subj}: {len(d)} estimates, raw MAE {np.mean(np.abs(d.bpm_raw - d.true_hr)):.2f}, "
              f"shown MAE {np.nanmean(np.abs(d.bpm_shown - d.true_hr)):.2f} BPM "
              f"({i / (time.perf_counter() - t_start):.0f} frames/s processed)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"live_{split_name}_ubfc.csv", index=False)
    return df


def plot_test(df, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = list(dict.fromkeys(df.subject))
    fig, axes = plt.subplots(len(subjects), 1, figsize=(11, 2.3 * len(subjects)), sharex=True)
    for ax, s in zip(axes, subjects):
        d = df[df.subject == s]
        ax.plot(d.t, d.true_hr, "k-", lw=2, label="ground truth (PPG, same 5.4 s window)")
        ax.plot(d.t, d.bpm_shown, "-", color="C0", lw=2, label="shown on screen (last 2 non-Poor)")
        for q, c in [("Good", "C2"), ("Fair", "C1"), ("Poor", "C3")]:
            dq = d[d.quality == q]
            ax.scatter(dq.t, dq.bpm_raw, s=12, color=c, label=f"single estimate: {q}", zorder=3)
        mae = np.nanmean(np.abs(d.bpm_shown - d.true_hr))
        ax.set_title(f"{s}: shown-BPM MAE {mae:.2f}", fontsize=9)
        ax.set_ylabel("BPM")
    axes[0].legend(fontsize=7, ncol=3, loc="upper left")
    axes[-1].set_xlabel("time in video (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="replay a video file instead of the webcam")
    parser.add_argument("--test-ubfc", action="store_true", help="headless accuracy check on UBFC test videos")
    parser.add_argument("--checkpoint", default=str(MAIN))
    parser.add_argument("--duration", type=float, default=30.0, help="seconds per measurement")
    parser.add_argument("--continuous", action="store_true", help="old mode: live BPM updated every second")
    parser.add_argument("--diagnose", action="store_true",
                        help="compare your logged webcam measurements with UBFC reference values")
    parser.add_argument("--test-sessions", action="store_true",
                        help="headless check: simulated 30 s measurements on UBFC val + test videos")
    args = parser.parse_args()

    if args.test_ubfc:
        df = test_ubfc(args.checkpoint)
        plot_test(df, OUT_DIR / "live_vs_truth.png")
        e_raw, e_shown = (df.bpm_raw - df.true_hr).abs(), (df.bpm_shown - df.true_hr).abs()
        print(f"\nAll test videos: {len(df)} estimates (one per second)")
        print(f"  single 5.4 s estimate: MAE {e_raw.mean():.2f} BPM, within 5 BPM {100 * (e_raw <= 5).mean():.0f}%")
        print(f"  shown (last 2 non-Poor): MAE {e_shown.mean():.2f} BPM, within 5 BPM {100 * (e_shown <= 5).mean():.0f}%")
        for q in ("Good", "Fair", "Poor"):
            d = df[df.quality == q]
            if len(d):
                print(f"  {q:4s}: {100 * len(d) / len(df):4.0f}% of estimates, single-estimate MAE "
                      f"{(d.bpm_raw - d.true_hr).abs().mean():.2f}")
        print(f"\nSaved {OUT_DIR / 'live_test_ubfc.csv'} and live_vs_truth.png")
    elif args.diagnose:
        diagnose()
    elif args.test_sessions:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("val", "test"):
            df = simulate_sessions(name, args.duration, args.checkpoint)
            df.to_csv(OUT_DIR / f"sessions_{name}_ubfc.csv", index=False)
            e = (df["median of all windows"] - df.true_hr).abs()
            print(f"{name}: {len(df)} sessions of {args.duration:.0f} s, final-BPM MAE {e.mean():.2f}, "
                  f"max error {e.max():.1f}, within 3 BPM {100 * (e <= 3).mean():.0f}%")
    elif args.continuous:
        run_camera(args.video or args.camera, args.checkpoint, is_file=bool(args.video))
    else:
        run_measure(args.video or args.camera, args.checkpoint, is_file=bool(args.video), duration=args.duration)


if __name__ == "__main__":
    main()
