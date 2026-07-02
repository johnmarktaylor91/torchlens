# SOURCE: vendored from SJTU-WirelessAI-Lab/CFARNet @ main (train.py, class IndexPredictionCNN)
# https://github.com/SJTU-WirelessAI-Lab/CFARNet
# Paper: "CFARNet: Learning-Based High-Resolution Multi-Target Detection for Rainbow Beam
# Radar" (arXiv:2505.10150). CFARNet replaces the traditional CFAR (Constant False Alarm Rate)
# detector with a CNN that predicts target peak indices directly in the angle-Doppler domain
# from complex radar echo data (a Doppler FFT + log-magnitude front end followed by a 2D-conv
# feature extractor and a 1D-conv prediction head). Vendored verbatim from the official repo's
# `train.py` (the `IndexPredictionCNN` model class); no architectural changes. Only the
# unrelated dataset/training/argparse code from `train.py` was dropped -- the model class body
# itself is untouched.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from train.py (SJTU-WirelessAI-Lab/CFARNet @ main)
# ---------------------------------------------------------------------------
class IndexPredictionCNN(nn.Module):
    def __init__(self, M_plus_1, Ns, hidden_dim=512, dropout=0.2):
        super().__init__()
        C = 512
        self.feat = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, (2, 1), 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, (2, 1), 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, C, 3, (2, 1), 1, bias=False),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
        )
        self.pred = nn.Sequential(
            nn.Conv1d(C, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, 1),
        )

    def forward(self, x):
        # Input: [B, Ns, M] Complex -> Log Magnitude [B, 1, Ns, M]
        x_fft = torch.fft.fftshift(torch.fft.fft(x, dim=1), dim=1)
        x = torch.log1p(torch.abs(x_fft)).unsqueeze(1)
        x = self.feat(x)
        x = torch.max(x, dim=2)[0]  # Max pool over Doppler dim
        return self.pred(x).squeeze(1)


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# tiny config: small angle-Doppler grid (repo trains with much larger M_plus_1/Ns/hidden_dim)
_M_PLUS_1, _NS, _HIDDEN_DIM = 33, 16, 32


def build_cfarnet():
    return IndexPredictionCNN(_M_PLUS_1, _NS, hidden_dim=_HIDDEN_DIM, dropout=0.0)


def example_input_cfarnet():
    # complex radar echo tensor [B, Ns, M_plus_1], as consumed by ChunkedEchoDataset/train.py
    return torch.randn(1, _NS, _M_PLUS_1, dtype=torch.complex64)


MENAGERIE_ENTRIES = [
    ("CFARNet", "build_cfarnet", "example_input_cfarnet", 2025, "SOURCE_AVAILABLE"),
]
