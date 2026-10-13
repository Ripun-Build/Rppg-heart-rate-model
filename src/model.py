"""PhysNet: 3D-CNN that maps a face video clip to a pulse waveform.

Reference: Yu et al., "Remote Photoplethysmograph Signal Measurement from
Facial Videos Using Spatio-Temporal Networks", BMVC 2019.

Input:  (B, 3, T, H, W)  e.g. (B, 3, 160, 72, 72)
Output: (B, T)           predicted rPPG waveform, one value per frame
"""
import torch
import torch.nn as nn


def conv_block(c_in, c_out, kernel, padding):
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel, stride=1, padding=padding),
        nn.BatchNorm3d(c_out),
        nn.ReLU(inplace=True),
    )


class PhysNet(nn.Module):
    def __init__(self, frames=160, in_channels=3, dropout=0.0):
        super().__init__()
        self.frames = frames
        # Drops whole feature maps during training so the decoder can't depend on any
        # single memorized feature; does nothing at inference.
        self.dropout = nn.Dropout3d(dropout)

        # Encoder: spatial pooling shrinks the face, temporal pooling halves time twice.
        self.conv1 = conv_block(in_channels, 16, (1, 5, 5), (0, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        self.conv2 = conv_block(16, 32, 3, 1)
        self.conv3 = conv_block(32, 64, 3, 1)
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=2)  # T/2

        self.conv4 = conv_block(64, 64, 3, 1)
        self.conv5 = conv_block(64, 64, 3, 1)
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=2)  # T/4

        self.conv6 = conv_block(64, 64, 3, 1)
        self.conv7 = conv_block(64, 64, 3, 1)
        self.pool4 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        self.conv8 = conv_block(64, 64, 3, 1)
        self.conv9 = conv_block(64, 64, 3, 1)

        # Decoder: transpose convs restore the temporal length T/4 -> T.
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.head = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv3(self.conv2(x)))
        x = self.pool3(self.conv5(self.conv4(x)))
        x = self.pool4(self.conv7(self.conv6(x)))
        x = self.dropout(self.conv9(self.conv8(x)))
        x = self.up2(self.up1(x))
        # Collapse space to a single value per frame (AdaptiveAvgPool3d isn't supported on MPS).
        x = self.head(x.mean(dim=(3, 4), keepdim=True))
        return x.view(x.shape[0], -1)  # (B, T)


if __name__ == "__main__":
    model = PhysNet()
    clip = torch.randn(2, 3, 160, 72, 72)
    out = model(clip)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output {tuple(out.shape)}, params {n_params / 1e6:.2f}M")
