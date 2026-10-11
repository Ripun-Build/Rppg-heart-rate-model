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
| `src/pulse.py` | Bandpass, FFT -> BPM, spectral SNR, quality label |
| `src/ubfc.py` | UBFC-rPPG loader (ground truth, video info) |
| `src/check_dataset.py` | Phase 1 dataset sanity check + plots |
| `src/face.py` | Face tracker (MediaPipe + smoothing) and 72x72 crop, shared with the live demo |
| `src/preprocess.py` | Videos -> face crops + aligned PPG in `data/processed/` |
| `src/dataset.py` | PyTorch Dataset: 160-frame clips, DiffNormalized input, augmentation, subject split |
| `src/check_preprocess.py` | Phase 2 check: crop quality, POS sanity check, Dataset shapes |
| `configs/split_ubfc.json` | Fixed train/val/test split by subject (30/6/6) |
| `src/train.py`, `src/evaluate.py` | *(todo)* |
| `src/live.py`, `src/export.py` | *(todo)* webcam demo, ONNX export |
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
- [ ] Phase 3 – training
- [ ] Phase 4/5 – evaluation + confidence tuning
- [ ] Phase 6 – live webcam demo
- [ ] Phase 7 – ONNX export
