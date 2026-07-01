# SOURCE: vendored from Yuqiu-Yang/pMTnet_Omni @ main
# (pMTnet_Omni/models/cdr3_beta_encoder_model.py)
#
# Original file:
#   https://raw.githubusercontent.com/Yuqiu-Yang/pMTnet_Omni/main/pMTnet_Omni/models/cdr3_beta_encoder_model.py
#
# This is the newer, actively-maintained successor of the original queue-row target
# tianshilu/pMTnet (which is a pure TensorFlow/Keras script, not a PyTorch nn.Module,
# and is explicitly referenced in the queue notes as superseded by "pMTnet Omni at
# Yuqiu-Yang/pMTnet_Omni"). pMTnet Omni is a real PyTorch reimplementation; this file
# vendors its CDR3-beta encoder, a multi-scale 1D-conv-over-2D-grid + Transformer VAE
# that encodes a padded (25 x 5) Atchley-factor CDR3-beta sequence representation into
# a 5-dim latent embedding used downstream by the TCR-pMHC binding classifier.
#
# No architectural changes were made; only the import block was minimally trimmed to
# the two used symbols.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class cdr3VAEb(nn.Module):
    def __init__(self):
        super(cdr3VAEb, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 150, (1, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 150, (2, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 2 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 150, (3, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 3 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1, 150, (4, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 4 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1, 150, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 5 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1, 150, (6, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 6 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(1, 150, (7, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 7 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(1, 150, (8, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 8 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(1, 150, (9, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 9 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(1, 150, (10, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 10 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(1, 150, (11, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 11 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(1, 150, (12, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 12 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(1, 150, (13, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 13 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(1, 150, (14, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 14 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(1, 150, (15, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 15 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(1, 150, (16, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 16 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer17 = nn.Sequential(
            nn.Conv2d(1, 150, (17, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 17 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer18 = nn.Sequential(
            nn.Conv2d(1, 150, (18, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 18 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer19 = nn.Sequential(
            nn.Conv2d(1, 150, (19, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 19 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer20 = nn.Sequential(
            nn.Conv2d(1, 150, (20, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 20 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer21 = nn.Sequential(
            nn.Conv2d(1, 150, (21, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 21 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer22 = nn.Sequential(
            nn.Conv2d(1, 150, (22, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 22 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer23 = nn.Sequential(
            nn.Conv2d(1, 150, (23, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 23 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer24 = nn.Sequential(
            nn.Conv2d(1, 150, (24, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 24 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.layer25 = nn.Sequential(
            nn.Conv2d(1, 150, (25, 5)),
            nn.ReLU(),
            nn.MaxPool2d((25 - 25 + 1, 1)),
            nn.Flatten(),
            nn.Linear(150, int(150 / 2)),
            nn.ReLU(),
            nn.Linear(int(150 / 2), 3),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=5, nhead=5), num_layers=6
        )
        self.linear1 = nn.Linear(in_features=3 * 25, out_features=30)
        self.linear2 = nn.Linear(in_features=3 * 25, out_features=30)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=30, out_features=int(150 * 25 / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(150 * 25 / 2), out_features=150 * 25),
            nn.Unflatten(1, (25, 150, 1)),
            nn.Conv2d(
                in_channels=25, out_channels=1, kernel_size=(150 - 25 + 1, 5), padding=(0, 4)
            ),
        )

    def forward(self, x):
        padding_mask = torch.sum(x.squeeze(dim=1), dim=2) == 0
        x = x.permute(2, 1, 0, 3)
        x = torch.squeeze(x, dim=1)
        x = self.transformer_encoder(src=x, src_key_padding_mask=padding_mask)
        x = torch.unsqueeze(x, dim=1)
        x = x.permute(2, 1, 0, 3)
        f1 = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(x)
        f5 = self.layer5(x)
        f6 = self.layer6(x)
        f7 = self.layer7(x)
        f8 = self.layer8(x)
        f9 = self.layer9(x)
        f10 = self.layer10(x)
        f11 = self.layer11(x)
        f12 = self.layer12(x)
        f13 = self.layer13(x)
        f14 = self.layer14(x)
        f15 = self.layer15(x)
        f16 = self.layer16(x)
        f17 = self.layer17(x)
        f18 = self.layer18(x)
        f19 = self.layer19(x)
        f20 = self.layer20(x)
        f21 = self.layer21(x)
        f22 = self.layer22(x)
        f23 = self.layer23(x)
        f24 = self.layer24(x)
        f25 = self.layer25(x)
        mu = self.linear1(
            torch.cat(
                (
                    f1,
                    f2,
                    f3,
                    f4,
                    f5,
                    f6,
                    f7,
                    f8,
                    f9,
                    f10,
                    f11,
                    f12,
                    f13,
                    f14,
                    f15,
                    f16,
                    f17,
                    f18,
                    f19,
                    f20,
                    f21,
                    f22,
                    f23,
                    f24,
                    f25,
                ),
                1,
            )
        )
        logvar = self.linear2(
            torch.cat(
                (
                    f1,
                    f2,
                    f3,
                    f4,
                    f5,
                    f6,
                    f7,
                    f8,
                    f9,
                    f10,
                    f11,
                    f12,
                    f13,
                    f14,
                    f15,
                    f16,
                    f17,
                    f18,
                    f19,
                    f20,
                    f21,
                    f22,
                    f23,
                    f24,
                    f25,
                ),
                1,
            )
        )
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        encoded = mu + eps * std
        decoded = self.decoder(encoded)
        return encoded, decoded, mu, logvar


# --- staging entry points ----------------------------------------------------


def build_pmtnet_omni_cdr3_beta():
    return cdr3VAEb()


def example_input_pmtnet_omni_cdr3_beta():
    # (batch, 1, 25, 5) -- 25 padded CDR3-beta AA positions x 5 Atchley factors.
    return torch.rand(2, 1, 25, 5)


MENAGERIE_ENTRIES = [
    (
        "pMTnet-Omni-CDR3-beta-VAE",
        build_pmtnet_omni_cdr3_beta,
        example_input_pmtnet_omni_cdr3_beta,
        2024,
        "SOURCE_AVAILABLE",
    ),
]
