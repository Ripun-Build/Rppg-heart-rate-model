"""Phase 2: turn each UBFC video into 72x72 face crops + an aligned PPG label.

For every subject this saves data/processed/UBFC-rPPG/subjectN.npz with:
  frames  (T, 72, 72, 3) uint8 RGB face crops, one per video frame
  ppg     (T,)  float32 ground-truth PPG on an even time grid (one sample per frame)
  fs      ()    frame rate (Hz) from the ground-truth timestamps
  boxes   (T, 3) face box (cx, cy, side) in original-video pixels
  found   (T,)  bool, False where the face detector had lost the face

Clips are NOT cut here; src/dataset.py does that so the clip length / stride /
normalization can change without re-running this slow step.

Run:  python -m src.preprocess [--workers 6] [--overwrite]
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from src.face import FaceTracker, crop_face
from src.ubfc import list_subjects, load_ground_truth, resample_uniform

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "UBFC-rPPG"


def process_subject(subject_dir):
    t0 = time.time()
    gt = load_ground_truth(subject_dir)
    tracker = FaceTracker()
    cap = cv2.VideoCapture(str(Path(subject_dir) / "vid.avi"))
    frames, boxes, found = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV decodes as BGR; the network trains on RGB
        box = tracker.update(rgb)
        if box is None:
            frames.append(np.zeros((72, 72, 3), np.uint8))
            boxes.append(np.zeros(3))
        else:
            frames.append(crop_face(rgb, box))
            boxes.append(box)
        found.append(box is not None and tracker.found)
    cap.release()
    tracker.close()

    n = len(frames)
    ppg = resample_uniform(gt["ppg"], gt["ts"], gt["fs"])[:n]
    np.savez(
        OUT_DIR / f"{subject_dir.name}.npz",
        frames=np.stack(frames),
        ppg=ppg.astype(np.float32),
        fs=np.float32(gt["fs"]),
        boxes=np.stack(boxes).astype(np.float32),
        found=np.array(found),
    )
    return subject_dir.name, n, len(gt["ppg"]), float(np.mean(found)), time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true", help="re-process subjects already saved")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = list_subjects()
    todo = [s for s in subjects if args.overwrite or not (OUT_DIR / f"{s.name}.npz").exists()]
    print(f"{len(subjects)} subjects, {len(todo)} to process with {args.workers} workers -> {OUT_DIR}\n")

    t0 = time.time()
    with Pool(args.workers) as pool:
        for name, n, n_ppg, found, dt in pool.imap_unordered(process_subject, todo):
            flag = "" if n == n_ppg and found > 0.99 else "  <-- CHECK"
            print(f"{name:>10}: {n} frames (ppg {n_ppg}), face found {100 * found:.1f}%, {dt:.0f}s{flag}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
