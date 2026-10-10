"""Loss functions for rPPG waveform regression."""
import torch
import torch.nn as nn


class NegPearsonLoss(nn.Module):
    """1 - Pearson correlation, averaged over the batch.

    Only the *shape* of the waveform matters (timing of each beat), not its
    scale or offset, which is exactly what we need to read heart rate.
    Range: 0 (perfect) .. 2 (perfectly inverted).
    """

    def forward(self, pred, target):
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        num = (pred * target).sum(dim=1)
        den = torch.sqrt((pred**2).sum(dim=1) * (target**2).sum(dim=1)) + 1e-8
        return (1 - num / den).mean()
