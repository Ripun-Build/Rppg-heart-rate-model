# RPPG – Heart rate from a face camera

PhysNet (3D-CNN) trained on UBFC-rPPG, tested on PURE, then run live on a webcam.
Outputs BPM plus a signal-quality label (Good / Fair / Poor). Not a medical device.

## Setup
```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Layout
| Path | What |
|---|---|
| `src/model.py` | PhysNet: `(B,3,160,72,72)` clip -> `(B,160)` pulse waveform |
| `src/losses.py` | Negative Pearson loss |
| `src/pulse.py` | Bandpass, FFT -> BPM, spectral SNR, quality label |
| `src/preprocess.py` | *(todo)* face crop + 160-frame clips |
| `src/dataset.py`, `src/train.py`, `src/evaluate.py` | *(todo)* |
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
- [ ] Phase 1 – dataset download + ground-truth check
- [ ] Phase 2 – preprocessing
- [ ] Phase 3 – training
- [ ] Phase 4/5 – evaluation + confidence tuning
- [ ] Phase 6 – live webcam demo
- [ ] Phase 7 – ONNX export
