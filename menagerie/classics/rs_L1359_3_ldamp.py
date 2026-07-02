# FAITHFUL PORT of ricedsp/D-AMP_Toolbox @ master (original framework: TensorFlow 1.x)
# https://github.com/ricedsp/D-AMP_Toolbox/blob/master/LDAMP_TensorFlow/LearnedDAMP.py
#
# Metzler, Mousavi & Baraniuk, "Learned D-AMP: Principled Neural Network based Compressive
# Image Recovery" (LDAMP), NeurIPS 2017. The original repo (LDAMP_TensorFlow/LearnedDAMP.py)
# is Python 2 + TensorFlow 1.x graph-mode code (`print 'x = ', x` statement syntax,
# tf.variable_scope, tf.placeholder) and cannot run in a modern base torch env, so this is a
# faithful architecture transcription, not a vendor-as-is. Transcribed 1:1 from the real repo:
#   - LDAMP(...) [line ~229]: the outer AMP recurrence -- r = xhat + A^T @ z, rvar from the
#     residual z, denoise via DnCNN_outer_wrapper, then the Onsager-corrected residual update
#     z = y - A @ xhat + (n/m) * dxdr * z.
#   - DnCNN_wrapper(...) [line ~514]: wraps the denoiser with a Monte-Carlo estimate of the
#     divergence dx/dr (Stein's-Unbiased-Risk-Estimate-style random perturbation: perturb r by
#     eta*epsilon, re-denoise, dxdr = mean(eta*(xhat_perturbed - xhat)) / epsilon). This IS
#     part of the model's forward computation in the original code (used to compute the Onsager
#     term z), so it is preserved in this port rather than dropped as a training-only detail.
#   - DnCNN(...) [line ~535] / init_vars_DnCNN(...) [line ~362]: the residual DnCNN denoiser --
#     first layer conv+ReLU (no bias, matching the original which only ever defines `weights`,
#     never `biases`, for DnCNN -- see the original's commented-out `#biases[...] = ...` lines),
#     n_DnCNN_layers-2 middle layers of conv+BatchNorm+ReLU (no bias, matching
#     init_vars_DnCNN's per-layer `weights[l]` only), a final conv layer (no bias, no
#     activation), and a residual output `x_hat = r - final_conv_output`.
#   - This port uses `tie=True` (the untied-per-iteration-weights branch is a config flag in
#     the original, not a different architecture) with n_DAMP_layers unrolled AMP iterations
#     reusing one DnCNN denoiser instance, matching `DnCNN_outer_wrapper`'s `if tie:` branch.
#   - The Gaussian sensing operator (A_handle/At_handle for mode='gaussian' in
#     GenerateMeasurementOperators) is used: A is a fixed (non-trainable) buffer, matching how
#     the original treats the measurement matrix as constant problem data (A_val placeholder),
#     not a learned parameter.

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """Residual image denoiser, transcribed from LearnedDAMP.py::DnCNN /
    ::init_vars_DnCNN. Operates on a flattened (n, batch) tensor representing a
    channel_img x height_img x width_img image, exactly as the original does via its
    internal reshape/transpose. No biases anywhere (the original repo defines weights
    only -- see the commented-out bias lines in init_vars_DnCNN)."""

    def __init__(
        self,
        height_img,
        width_img,
        channel_img,
        filter_height,
        filter_width,
        num_filters,
        n_dncnn_layers,
    ):
        super().__init__()
        self.height_img = height_img
        self.width_img = width_img
        self.channel_img = channel_img
        self.n_dncnn_layers = n_dncnn_layers

        self.conv_first = nn.Conv2d(
            channel_img,
            num_filters,
            kernel_size=(filter_height, filter_width),
            padding="same",
            bias=False,
        )
        self.mid_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    num_filters,
                    num_filters,
                    kernel_size=(filter_height, filter_width),
                    padding="same",
                    bias=False,
                )
                for _ in range(n_dncnn_layers - 2)
            ]
        )
        self.mid_bns = nn.ModuleList(
            [nn.BatchNorm2d(num_filters) for _ in range(n_dncnn_layers - 2)]
        )
        self.conv_last = nn.Conv2d(
            num_filters,
            channel_img,
            kernel_size=(filter_height, filter_width),
            padding="same",
            bias=False,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (n, batch) with n = height_img * width_img * channel_img
        orig_shape = r.shape
        batch = orig_shape[1]
        x = r.transpose(0, 1).reshape(batch, self.channel_img, self.height_img, self.width_img)

        x = torch.relu(self.conv_first(x))
        for conv, bn in zip(self.mid_convs, self.mid_bns):
            x = torch.relu(bn(conv(x)))
        x = self.conv_last(x)

        x = x.reshape(batch, -1).transpose(0, 1)
        return r - x


class LDAMP(nn.Module):
    """Learned Denoising-based AMP network (Metzler, Mousavi & Baraniuk 2017), faithfully
    ported from LearnedDAMP.py::LDAMP + ::DnCNN_wrapper with tie=True (a single shared DnCNN
    denoiser reused across all unrolled AMP iterations)."""

    def __init__(
        self,
        height_img=8,
        width_img=8,
        channel_img=1,
        m=32,
        filter_height=3,
        filter_width=3,
        num_filters=8,
        n_dncnn_layers=4,
        n_damp_layers=3,
    ):
        super().__init__()
        n = height_img * width_img * channel_img
        self.n = n
        self.m = m
        self.n_damp_layers = n_damp_layers

        A = torch.randn(m, n) / (m**0.5)  # mode='gaussian' sensing operator
        self.register_buffer("A", A)

        self.denoiser = DnCNN(
            height_img,
            width_img,
            channel_img,
            filter_height,
            filter_width,
            num_filters,
            n_dncnn_layers,
        )

    def denoise_with_divergence(self, r: torch.Tensor, eta: torch.Tensor):
        """Transcribed from DnCNN_wrapper: Monte Carlo estimate of dxhat/dr via a random
        perturbation, used to form the Onsager correction term."""
        xhat = self.denoiser(r)
        r_abs = r.abs()
        epsilon = torch.clamp(0.001 * r_abs.amax(dim=0), min=0.00001)
        r_perturbed = r + eta * epsilon
        xhat_perturbed = self.denoiser(r_perturbed)
        eta_dx = eta * (xhat_perturbed - xhat)
        mean_eta_dx = eta_dx.mean(dim=0)
        dxdr = mean_eta_dx / epsilon
        return xhat, dxdr

    def forward(self, y: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        # y: (m, batch) noisy compressive measurements
        # eta: (n_damp_layers, n, batch) pre-drawn standard-normal perturbation noise, one
        # draw per unrolled iteration (kept as an explicit input rather than sampled inside
        # forward() so the op graph TorchLens captures is deterministic/traceable).
        batch = y.shape[1]
        z = y
        xhat = torch.zeros(self.n, batch, dtype=y.dtype, device=y.device)

        for it in range(self.n_damp_layers):
            r = xhat + self.A.T @ z
            rvar = (1.0 / self.m) * torch.sum(torch.square(z.abs()), dim=0)  # noqa: F841 (unused in original too; matches upstream)
            xhat, dxdr = self.denoise_with_divergence(r, eta[it])
            z = y - self.A @ xhat + (self.n / self.m) * dxdr * z

        return xhat


def build_ldamp():
    return LDAMP(
        height_img=8,
        width_img=8,
        channel_img=1,
        m=32,
        filter_height=3,
        filter_width=3,
        num_filters=8,
        n_dncnn_layers=4,
        n_damp_layers=3,
    )


def example_input_ldamp():
    net = build_ldamp()
    batch = 2
    y = torch.randn(net.m, batch)
    eta = torch.randn(net.n_damp_layers, net.n, batch)
    return (y, eta)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    ("LDAMP (Learned D-AMP)", "build_ldamp", "example_input_ldamp", 2017, "ported-pytorch"),
]
