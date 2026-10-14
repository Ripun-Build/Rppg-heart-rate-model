# RPPG – Heart rate from a face camera

PhysNet (3D-CNN) trained on UBFC-rPPG, tested on PURE, then run live on a webcam.
Outputs BPM plus a signal-quality label (Good / Fair / Poor). Not a medical device.

## Setup
```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
# MediaPipe face detector (230 KB, Google's official model)
curl -L -o models/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
```

## Layout
| Path | What |
|---|---|
| `src/model.py` | PhysNet: `(B,3,160,72,72)` clip -> `(B,160)` pulse waveform |
| `src/losses.py` | Negative Pearson loss |
| `src/pulse.py` | Bandpass, FFT -> BPM, `to_pulse` (integrate diff outputs), SNR, quality label, GREEN/CHROM/POS |
| `src/ubfc.py` | UBFC-rPPG loader (ground truth, video info) |
| `src/check_dataset.py` | Phase 1 dataset sanity check + plots |
| `src/face.py` | Face tracker (MediaPipe + smoothing) and 72x72 crop, shared with the live demo |
| `src/preprocess.py` | Videos -> face crops + aligned PPG in `data/processed/` |
| `src/dataset.py` | PyTorch Dataset: 160-frame clips, DiffNormalized input, augmentation, subject split |
| `src/check_preprocess.py` | Phase 2 check: crop quality, POS sanity check, Dataset shapes |
| `configs/split_ubfc.json` | Fixed train/val/test split by subject (30/6/6) |
| `src/train.py` | Training loop, overfit sanity test, best-checkpoint saving, TensorBoard |
| `configs/physnet_ubfc.yaml` | Training settings |
| `src/evaluate.py` | Phase 5 test-set evaluation vs GREEN/CHROM/POS, Bland-Altman, confidence check |
| `src/live.py` | Phase 6 webcam app: 30 s measurement -> one BPM + range + confidence, diagnostics |
| `src/export.py` | *(todo)* ONNX export |
| `data/raw/` | Downloaded datasets (git-ignored) |

## Datasets (request access – Phase 1)
- **UBFC-rPPG** (Dataset 2): https://sites.google.com/view/ybenezeth/ubfcrppg
  Extract to `data/raw/UBFC-rPPG/subject*/{vid.avi,ground_truth.txt}`
- **PURE**: TU Ilmenau "Pulse Rate Detection Dataset" (search "PURE dataset TU Ilmenau"; request by email)
  Extract to `data/raw/PURE/`

Reference implementation: https://github.com/ubicomplab/rPPG-Toolbox

## Progress
- [x] Phase 0 – env, model, loss, BPM/SNR utils (smoke-tested on MPS)
- [x] Phase 1 – UBFC-rPPG ground-truth check: 42/42 subjects pass, PPG label FFT-vs-beat-count MAE 1.01 BPM
  (`python -m src.check_dataset`, results in `outputs/phase1/`)
- [x] Phase 2 – preprocessing: face found in 100% of frames, box jitter 0.07 px/frame,
  POS on crops 2.88 BPM MAE (pulse preserved); 685 train / 69 val / 68 test clips
  (`python -m src.preprocess`, `python -m src.check_preprocess`, results in `outputs/phase2/`)
- [~] Phase 3 – training (first run): overfit test passes; best val MAE **1.79 BPM** (epoch 5).
  Stopped at epoch 11/30: train loss kept falling while val loss rose (overfitting).
  Next: weight decay, lower LR, stronger augmentation, early stopping.
  (`python -m src.train --overfit 4`, `python -m src.train`, results in `outputs/phase3/`)
- [x] Phase 4/5 – evaluation on the 6 held-out UBFC test subjects (run 1 `best.pt`, chosen on validation):

  | Method | Clip MAE (BPM) | Clips within 5 BPM | Per-video MAE (median of clips) |
  |---|---|---|---|
  | **PhysNet** | **1.54** | **98.5%** | **0.95** |
  | POS | 1.63 | 95.6% | 1.33 |
  | CHROM | 2.06 | 92.6% | 1.43 |
  | GREEN | 7.32 | 76.5% | 2.35 |

  Bias -0.67 BPM, no heart-rate-dependent error. Confidence (SNR thresholds chosen on validation):
  Good 87% of test clips, MAE 1.27; Poor 4%, MAE 5.50.
  Bug fixed: HR was read from the network's *differenced* output, whose 2nd harmonic can win
  at low HR (55 BPM read as 110); outputs are now integrated first (`to_pulse`).
  (`python -m src.evaluate`, results in `outputs/phase5/`)
- [x] Phase 6 – webcam app (`python -m src.live`): measurement mode records 30 s, then shows one
  heart rate = median of the ~1 s window estimates, with a range and a confidence level.
  Same crop / normalization / BPM code as training; frames resampled to 30 fps from timestamps.
  Simulated 30 s measurements on the UBFC test videos: **MAE 0.86 BPM**, worst 2.3 BPM (12 sessions);
  final-BPM method chosen on validation (median of windows 0.94 vs one long FFT 1.47 BPM).
  Webcam: face found ~100%, but pulse SNR ~-2 dB vs +1.9 dB on UBFC -> mostly Low confidence.
  Diagnostics (face size, brightness variation, POS on the same frames) added to find out whether
  the camera/lighting or the model is the cause; real-world accuracy vs a reference still to be measured.
  (`--continuous` for live BPM every second, `--diagnose`, `--test-sessions`; results in `outputs/phase6/`)
- [ ] Phase 7 – ONNX export
