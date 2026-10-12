"""Phase 3: train PhysNet on UBFC-rPPG.

Every epoch: train on the train subjects, then measure heart-rate error on the
validation subjects. The checkpoint with the lowest validation HR error is kept as
best.pt; the test subjects are not touched here (that's Phase 5).

Outputs:
  checkpoints/<name>/best.pt, last.pt     model weights (+ optimizer state for resuming)
  runs/<name>/                            TensorBoard logs
  outputs/phase3/<name>/history.csv       per-epoch loss and HR metrics
  outputs/phase3/<name>/curves.png        training curves
  outputs/phase3/<name>/val_examples.png  predicted vs true waveform for a few val clips

Run:
  python -m src.train --overfit 4     # sanity check: memorize 4 clips, loss should go to ~0
  python -m src.train                 # full training (configs/physnet_ubfc.yaml)
  python -m src.train --resume        # continue from checkpoints/<name>/last.pt
"""
import argparse
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.dataset import RPPGDataset, load_split
from src.losses import NegPearsonLoss
from src.model import PhysNet
from src.pulse import estimate_bpm

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device):
    """Run the model over a loader; returns predicted and true waveforms plus clip info."""
    model.eval()
    preds, labels, fs, subjects, starts = [], [], [], [], []
    for batch in loader:
        preds.append(model(batch["x"].to(device)).cpu().numpy())
        labels.append(batch["y"].numpy())
        fs += [float(f) for f in batch["fs"]]
        subjects += list(batch["subject"])
        starts += [int(s) for s in batch["start"]]
    return {"pred": np.concatenate(preds), "label": np.concatenate(labels),
            "fs": np.array(fs), "subject": subjects, "start": starts}


def hr_metrics(out):
    """Clip-level heart rate: FFT of predicted waveform vs FFT of true waveform."""
    pred_hr = np.array([estimate_bpm(p, f) for p, f in zip(out["pred"], out["fs"])])
    true_hr = np.array([estimate_bpm(y, f) for y, f in zip(out["label"], out["fs"])])
    err = pred_hr - true_hr
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "pearson": float(np.corrcoef(pred_hr, true_hr)[0, 1]) if len(err) > 1 else float("nan"),
        "within5_pct": float(100 * np.mean(np.abs(err) <= 5)),
    }, pred_hr, true_hr


def plot_curves(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(hist["epoch"], hist["train_loss"], "o-", label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], "o-", label="validation")
    axes[0].set(title="Loss (1 - Pearson correlation of waveforms)", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    axes[1].plot(hist["epoch"], hist["val_mae"], "o-", color="C3", label="validation MAE")
    best = hist.loc[hist["val_mae"].idxmin()]
    axes[1].axhline(best["val_mae"], color="k", ls="--", lw=1,
                    label=f"best {best['val_mae']:.2f} BPM (epoch {int(best['epoch'])})")
    axes[1].set(title="Validation heart-rate error", xlabel="epoch", ylabel="MAE (BPM)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_examples(out, pred_hr, true_hr, path, n=4):
    idx = np.linspace(0, len(pred_hr) - 1, n).astype(int)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n))
    for ax, i in zip(axes, idx):
        t = np.arange(out["pred"].shape[1]) / out["fs"][i]
        z = lambda v: (v - v.mean()) / (v.std() + 1e-8)
        ax.plot(t, z(out["label"][i]), color="k", label=f"true ({true_hr[i]:.1f} BPM)")
        ax.plot(t, z(out["pred"][i]), color="C3", alpha=0.8, label=f"predicted ({pred_hr[i]:.1f} BPM)")
        ax.set_title(f"{out['subject'][i]}, frames {out['start'][i]}-{out['start'][i] + len(t)}", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def overfit_test(cfg, n_clips, device, steps=150):
    """Train on a handful of clips with no augmentation. If the loss doesn't drop near 0,
    something in the data/model/loss wiring is broken."""
    split = load_split()
    ds = RPPGDataset(split["train"], clip_len=cfg["data"]["clip_len"], input_type=cfg["data"]["input_type"])
    batch = next(iter(DataLoader(Subset(ds, range(0, len(ds), len(ds) // n_clips)[:n_clips]), batch_size=n_clips)))
    model = PhysNet(cfg["model"]["frames"], in_channels=ds.in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = NegPearsonLoss()
    x, y = batch["x"].to(device), batch["y"].to(device)
    print(f"Overfit test: {n_clips} clips from {sorted(set(batch['subject']))}, {steps} steps")
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 25 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    for p, t, f in zip(pred, batch["y"].numpy(), batch["fs"]):
        print(f"  true {estimate_bpm(t, float(f)):6.1f} BPM   predicted {estimate_bpm(p, float(f)):6.1f} BPM")
    print("PASS" if loss.item() < 0.1 else "FAIL: loss did not approach 0, check the pipeline")


def train(cfg, device, resume=False):
    name = cfg["name"]
    ckpt_dir, out_dir = ROOT / "checkpoints" / name, ROOT / "outputs" / "phase3" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    d, t = cfg["data"], cfg["train"]
    split = load_split()
    train_ds = RPPGDataset(split["train"], clip_len=d["clip_len"], stride=d["train_stride"],
                           input_type=d["input_type"], augment=d["augment"])
    val_ds = RPPGDataset(split["val"], clip_len=d["clip_len"], input_type=d["input_type"])
    # num_workers=0: loading a clip takes ~10 ms vs ~1 s per training step, and on macOS
    # worker processes would each copy the whole dataset into memory.
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=t["batch_size"])

    model = PhysNet(cfg["model"]["frames"], in_channels=train_ds.in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    loss_fn = NegPearsonLoss()

    start_epoch, best_mae, history = 1, float("inf"), []
    if resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best_mae, history = ck["epoch"] + 1, ck["best_mae"], ck["history"]
        print(f"Resumed from epoch {ck['epoch']} (best val MAE {best_mae:.2f})")

    writer = SummaryWriter(ROOT / "runs" / name)
    print(f"Training {name} on {device}: {len(train_ds)} train clips, {len(val_ds)} val clips, "
          f"{len(train_dl)} steps/epoch, {t['epochs']} epochs")

    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for i, batch in enumerate(train_dl, 1):
            loss = loss_fn(model(batch["x"].to(device)), batch["y"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if i % 50 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_dl)} loss {np.mean(losses[-50:]):.4f}", flush=True)
        sched.step()

        out = predict(model, val_dl, device)
        val_loss = float(loss_fn(torch.from_numpy(out["pred"]), torch.from_numpy(out["label"])))
        m, pred_hr, true_hr = hr_metrics(out)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               "val_mae": m["mae"], "val_rmse": m["rmse"], "val_pearson": m["pearson"],
               "val_within5_pct": m["within5_pct"], "lr": opt.param_groups[0]["lr"],
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        for k in ("train_loss", "val_loss", "val_mae", "val_rmse", "val_pearson", "lr"):
            writer.add_scalar(k, row[k], epoch)

        improved = m["mae"] < best_mae
        if improved:
            best_mae = m["mae"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_mae": best_mae},
                       ckpt_dir / "best.pt")
            plot_examples(out, pred_hr, true_hr, out_dir / "val_examples.png")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best_mae": best_mae, "history": history, "config": cfg},
                   ckpt_dir / "last.pt")

        hist = pd.DataFrame(history)
        hist.to_csv(out_dir / "history.csv", index=False)
        plot_curves(hist, out_dir / "curves.png")
        print(f"Epoch {epoch:2d}/{t['epochs']}  train {row['train_loss']:.4f}  val {val_loss:.4f}  "
              f"val MAE {m['mae']:5.2f} BPM  RMSE {m['rmse']:5.2f}  r {m['pearson']:.3f}  "
              f"within 5 BPM {m['within5_pct']:3.0f}%  ({row['minutes']:.1f} min){'  * best' if improved else ''}",
              flush=True)

    writer.close()
    print(f"\nDone. Best validation MAE {best_mae:.2f} BPM -> {ckpt_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "physnet_ubfc.yaml"))
    parser.add_argument("--overfit", type=int, default=0, help="sanity check on N clips instead of training")
    parser.add_argument("--epochs", type=int, help="override number of epochs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    device = cfg["train"]["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    set_seed(cfg["train"]["seed"])

    if args.overfit:
        overfit_test(cfg, args.overfit, device)
    else:
        train(cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
