# SOURCE: vendored from ZhonghaoZ/AMP-Net_TIP @ main
# https://github.com/ZhonghaoZ/AMP-Net_TIP/blob/main/train_AMP_Net.py
# The `Denoiser` and `AMP_net_Deblock` classes (AMP-Net-K: an unrolled Approximate
# Message Passing network for compressive image sensing -- learned linear sampling
# matrix `A`, a learned initial-reconstruction matrix `Q`, and `layer_num` unrolled
# AMP iterations each doing a linear gradient step (`block1`) followed by a learned
# CNN denoiser correction; IEEE Trans. Image Processing 2021) are transcribed
# VERBATIM (layer-for-layer, same forward-pass control flow) from train_AMP_Net.py.
# Only changes: (1) the hardcoded `torch.eye(33 * 33).float().cuda()` inside
# `forward` is replaced with a device-matching `torch.eye(...)` built from the
# input tensor's device (no `.cuda()` requirement so it runs on CPU); (2) unused
# training-only functions (`compute_loss`, `get_final_loss`, `get_loss`, `train`,
# `get_val_result`, `load_sampling_matrix`, `get_Q`) and the `if __name__ ==
# "__main__"` training script are dropped -- only the two `nn.Module` classes are
# needed to construct and trace the network. No architectural layer or unrolled-AMP
# mechanism was added, removed, or altered.
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import Module

MENAGERIE_ZOO = "vendored-pytorch"


# --- train_AMP_Net.py (verbatim architecture) ---
class Denoiser(Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1, bias=False),
        )

    def forward(self, inputs):
        inputs = torch.unsqueeze(torch.reshape(torch.transpose(inputs, 0, 1), [-1, 33, 33]), dim=1)
        output = self.D(inputs)
        # output=inputs-output
        output = torch.transpose(torch.reshape(torch.squeeze(output), [-1, 33 * 33]), 0, 1)
        return output


class AMP_net_Deblock(Module):
    def __init__(self, layer_num, A):
        super().__init__()
        self.layer_num = layer_num
        self.denoisers = []
        self.steps = []
        self.register_parameter("A", nn.Parameter(torch.from_numpy(A).float(), requires_grad=False))
        self.register_parameter(
            "Q", nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True)
        )
        for n in range(layer_num):
            self.denoisers.append(Denoiser())
            self.register_parameter(
                "step_" + str(n + 1), nn.Parameter(torch.tensor(1.0), requires_grad=False)
            )
            self.steps.append(eval("self.step_" + str(n + 1)))
        for n, denoiser in enumerate(self.denoisers):
            self.add_module("denoiser_" + str(n + 1), denoiser)

    def forward(self, inputs, output_layers):
        H = int(inputs.shape[2] / 33)
        L = int(inputs.shape[3] / 33)
        S = inputs.shape[0]

        y = self.sampling(inputs)
        X = torch.matmul(self.Q, y)
        for n in range(output_layers):
            step = self.steps[n]
            denoiser = self.denoisers[n]

            z = self.block1(X, y, step)
            noise = denoiser(X)
            eye = torch.eye(33 * 33, dtype=torch.float32, device=inputs.device)
            X = z - torch.matmul(
                (step * torch.matmul(torch.transpose(self.A, 0, 1), self.A)) - eye, noise
            )

            X = self.together(X, S, H, L)
            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=1), dim=0)
            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=2), dim=0)
            X = torch.transpose(torch.reshape(X, [-1, 33 * 33]), 0, 1)

        X = self.together(X, S, H, L)
        return torch.unsqueeze(X, dim=1)

    def sampling(self, inputs):
        inputs = torch.squeeze(inputs, dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=1), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=2), dim=0)
        inputs = torch.transpose(torch.reshape(inputs, [-1, 33 * 33]), 0, 1)
        outputs = torch.matmul(self.A, inputs)
        return outputs

    def block1(self, X, y, step):
        outputs = torch.matmul(torch.transpose(self.A, 0, 1), y - torch.matmul(self.A, X))
        outputs = step * outputs + X
        return outputs

    def together(self, inputs, S, H, L):
        inputs = torch.reshape(torch.transpose(inputs, 0, 1), [-1, 33, 33])
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H * S, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
        return inputs


# --- staging entry points ---
def build_amp_net():
    rng = np.random.default_rng(0)
    # Real repo loads a precomputed Gaussian random sampling matrix (e.g.
    # dataset/sampling_matrix/25.mat, CS_ratio=25% of 33*33=1089 rows). A small
    # random Gaussian matrix of the same shape convention reproduces the real
    # constructor's sampling-matrix argument without requiring the .mat asset.
    cs_ratio = 25
    n = 33 * 33
    m = int(n * cs_ratio / 100)
    A = rng.standard_normal((m, n)).astype(np.float32)
    model = AMP_net_Deblock(layer_num=2, A=A)
    model.eval()
    return model


def example_input_amp_net():
    # A single 33x33 image block (H=1, L=1 grid of blocks), batch size 1, 1 channel,
    # plus the number of unrolled AMP layers to run (forward's second positional arg).
    return (torch.randn(1, 1, 33, 33), 2)


MENAGERIE_ENTRIES = [
    ("AMP-Net (AMP-Net-K)", "build_amp_net", "example_input_amp_net", 2021, MENAGERIE_ZOO),
]
