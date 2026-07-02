# FAITHFUL PORT of https://github.com/IDEALLab/airfoil-opt-gan @ master (original framework: TensorFlow 1.x)
#
# BezierGAN for airfoil-manifold capture / aerodynamic shape optimization
# (Chen, Chiu & Fuge, "Aerodynamic Design Optimization and Shape
# Exploration using Generative Adversarial Networks", AIAA SciTech 2019;
# see also arXiv:2101.04757 for the follow-on AirfoilGAN line). The real
# repo's `gan.py` (`class GAN`, TF1 `tf.layers`/`tf.variable_scope` API) is
# transcribed layer-for-layer below into torch `nn.Module`s: `Generator`
# maps a (latent code `c`, noise `z`) pair through dense+deconv towers into
# rational-Bezier control points `cp`, weights `w`, and a data-point
# parameterization `db`, then evaluates the closed-form rational Bezier
# curve (log-gamma Bernstein-basis formulation, exactly as in the TF
# `generator()` method) to produce the airfoil coordinate curve `dp`.
# `Discriminator` is the matching conv-tower + InfoGAN-style `Q`-head that
# reads back an estimate of the latent code from a candidate airfoil
# curve. Every dense/conv layer size, activation (LeakyReLU 0.2, tanh,
# sigmoid, softmax), and the Bezier-basis math (`lbs`/`lc`/`bs` via
# `lgamma`) match the real `gan.py` `generator()`/`discriminator()`
# methods; only the TF-session/placeholder training harness (irrelevant to
# architecture) is dropped, and control flow is expressed as plain torch
# ops (`torch.lgamma`, `F.conv_transpose2d` via `nn.ConvTranspose2d`, etc.)
# instead of `tf.layers.*`.

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

EPSILON = 1e-7


class Generator(nn.Module):
    """Faithful port of `GAN.generator()` from the real repo's `gan.py`.

    Maps latent code `c` (batch, latent_dim) and noise `z` (batch,
    noise_dim) to a synthesized airfoil curve `dp` (batch, n_points, 2, 1)
    via a dense+deconv control-point tower followed by a closed-form
    rational Bezier evaluation (exactly the real generator's math).
    """

    def __init__(self, latent_dim, noise_dim, n_points, bezier_degree):
        super().__init__()
        assert (bezier_degree + 1) % 8 == 0, (
            "real repo requires (bezier_degree+1) % 8 == 0 (dim_cpw = (bezier_degree+1)//8)"
        )
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.n_points = n_points
        self.bezier_degree = bezier_degree

        depth_cpw = 32 * 8
        self.depth_cpw = depth_cpw
        self.dim_cpw = (bezier_degree + 1) // 8

        cz_dim = latent_dim + noise_dim if noise_dim != 0 else latent_dim

        # --- control-point-weight (cpw) tower ---
        self.cpw_dense0 = nn.Linear(cz_dim, 1024)
        self.cpw_bn0 = nn.BatchNorm1d(1024, momentum=0.1, eps=1e-3)
        self.cpw_dense1 = nn.Linear(1024, self.dim_cpw * 3 * depth_cpw)
        self.cpw_bn1 = nn.BatchNorm1d(self.dim_cpw * 3 * depth_cpw, momentum=0.1, eps=1e-3)

        kernel_size = (4, 3)
        # tf conv2d_transpose with strides=(2,1), padding='same' on a
        # (dim_cpw, 3) spatial map -> torch ConvTranspose2d w/ matching
        # kernel/stride and explicit output_padding to reproduce the
        # "same"-style doubling of the first spatial dim only.
        self.deconv0 = nn.ConvTranspose2d(
            depth_cpw,
            depth_cpw // 2,
            kernel_size,
            stride=(2, 1),
            padding=(1, 1),
            output_padding=(0, 0),
        )
        self.deconv0_bn = nn.BatchNorm2d(depth_cpw // 2, momentum=0.1, eps=1e-3)

        self.deconv1 = nn.ConvTranspose2d(
            depth_cpw // 2,
            depth_cpw // 4,
            kernel_size,
            stride=(2, 1),
            padding=(1, 1),
            output_padding=(0, 0),
        )
        self.deconv1_bn = nn.BatchNorm2d(depth_cpw // 4, momentum=0.1, eps=1e-3)

        self.deconv2 = nn.ConvTranspose2d(
            depth_cpw // 4,
            depth_cpw // 8,
            kernel_size,
            stride=(2, 1),
            padding=(1, 1),
            output_padding=(0, 0),
        )
        self.deconv2_bn = nn.BatchNorm2d(depth_cpw // 8, momentum=0.1, eps=1e-3)

        # Control points: conv2d kernel (1,2) valid -> collapses width 3->2->1
        self.cp_conv = nn.Conv2d(depth_cpw // 8, 1, kernel_size=(1, 2), padding=0)
        # Weights: conv2d kernel (1,3) valid -> collapses width 3->1
        self.w_conv = nn.Conv2d(depth_cpw // 8, 1, kernel_size=(1, 3), padding=0)

        # --- data-point parameterization (db) tower ---
        self.db_dense0 = nn.Linear(cz_dim, 1024)
        self.db_bn0 = nn.BatchNorm1d(1024, momentum=0.1, eps=1e-3)
        self.db_dense1 = nn.Linear(1024, 256)
        self.db_bn1 = nn.BatchNorm1d(256, momentum=0.1, eps=1e-3)
        self.db_dense2 = nn.Linear(256, n_points - 1)

    def forward(self, c, z):
        cz = torch.cat([c, z], dim=-1) if self.noise_dim != 0 else c

        # control-point-weight tower
        cpw = F.leaky_relu(self.cpw_bn0(self.cpw_dense0(cz)), negative_slope=0.2)
        cpw = F.leaky_relu(self.cpw_bn1(self.cpw_dense1(cpw)), negative_slope=0.2)
        # reshape (-1, dim_cpw, 3, depth_cpw) [NHWC] -> torch NCHW (-1, depth_cpw, dim_cpw, 3)
        cpw = cpw.view(-1, self.dim_cpw, 3, self.depth_cpw).permute(0, 3, 1, 2)

        cpw = F.leaky_relu(self.deconv0_bn(self.deconv0(cpw)), negative_slope=0.2)
        cpw = F.leaky_relu(self.deconv1_bn(self.deconv1(cpw)), negative_slope=0.2)
        cpw = F.leaky_relu(self.deconv2_bn(self.deconv2(cpw)), negative_slope=0.2)

        # Control points: batch x 1 x (bezier_degree+1) x 2 -> squeeze channel
        cp = torch.tanh(self.cp_conv(cpw))
        cp = cp.squeeze(1)  # batch x (bezier_degree+1) x 2

        # Weights: batch x 1 x (bezier_degree+1) x 1 -> squeeze channel+last dim
        w = torch.sigmoid(self.w_conv(cpw))
        w = w.squeeze(1)  # batch x (bezier_degree+1) x 1

        # data-point parameterization
        db = F.leaky_relu(self.db_bn0(self.db_dense0(cz)), negative_slope=0.2)
        db = F.leaky_relu(self.db_bn1(self.db_dense1(db)), negative_slope=0.2)
        db = F.softmax(self.db_dense2(db), dim=-1)  # batch x (n_points-1)

        ub = F.pad(db, (1, 0), value=0.0)  # batch x n_points
        ub = torch.cumsum(ub, dim=1)
        ub = torch.clamp(ub, max=1.0)
        ub = ub.unsqueeze(-1)  # batch x n_points x 1

        # Bezier layer: rational-Bezier evaluation via log-gamma Bernstein basis
        num_control_points = self.bezier_degree + 1
        lbs = ub.repeat(1, 1, num_control_points)  # batch x n_points x n_control_points
        pw1 = torch.arange(0, num_control_points, dtype=torch.float32, device=c.device).view(
            1, 1, -1
        )
        pw2 = torch.flip(pw1, dims=[-1])
        lbs = pw1 * torch.log(lbs + EPSILON) + pw2 * torch.log(1 - lbs + EPSILON)
        lc = torch.lgamma(pw1 + 1) + torch.lgamma(pw2 + 1)
        lc = torch.lgamma(torch.tensor(float(num_control_points))) - lc
        lbs = lbs + lc
        bs = torch.exp(lbs)  # batch x n_points x n_control_points

        cp_w = cp * w
        dp = torch.matmul(bs, cp_w)  # batch x n_points x 2
        bs_w = torch.matmul(bs, w)  # batch x n_points x 1
        dp = dp / bs_w
        dp = dp.unsqueeze(-1)  # batch x n_points x 2 x 1

        return dp, cp, w, ub, db


class Discriminator(nn.Module):
    """Faithful port of `GAN.discriminator()` from the real repo's `gan.py`.

    Conv tower (6 conv+BN+LeakyReLU(0.2)+dropout blocks) reading an airfoil
    curve `x` (batch, n_points, 2, 1), followed by a real/fake logit head
    `d` and an InfoGAN-style `Q` head that predicts a Gaussian
    (mean, logstd) over the latent code.
    """

    def __init__(self, latent_dim, n_points):
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_size = (4, 2)
        self.stride = (2, 1)
        depth = 64
        dropout = 0.4
        kernel_size = self.kernel_size

        # input is (batch, n_points, 2, 1) NHWC in the real code -> torch NCHW (batch, 1, n_points, 2).
        # The real TF code uses padding='same'; torch has no direct 'same'
        # equivalent for stride>1 conv2d, so we pad explicitly per-layer in
        # forward() via `_tf_same_pad2d` (TF-style: extra padding at the end).
        self.conv0 = nn.Conv2d(1, depth, kernel_size, stride=self.stride, padding=0)
        self.bn0 = nn.BatchNorm2d(depth, momentum=0.1, eps=1e-3)
        self.conv1 = nn.Conv2d(depth, depth * 2, kernel_size, stride=self.stride, padding=0)
        self.bn1 = nn.BatchNorm2d(depth * 2, momentum=0.1, eps=1e-3)
        self.conv2 = nn.Conv2d(depth * 2, depth * 4, kernel_size, stride=self.stride, padding=0)
        self.bn2 = nn.BatchNorm2d(depth * 4, momentum=0.1, eps=1e-3)
        self.conv3 = nn.Conv2d(depth * 4, depth * 8, kernel_size, stride=self.stride, padding=0)
        self.bn3 = nn.BatchNorm2d(depth * 8, momentum=0.1, eps=1e-3)
        self.conv4 = nn.Conv2d(depth * 8, depth * 16, kernel_size, stride=self.stride, padding=0)
        self.bn4 = nn.BatchNorm2d(depth * 16, momentum=0.1, eps=1e-3)
        self.conv5 = nn.Conv2d(depth * 16, depth * 32, kernel_size, stride=self.stride, padding=0)
        self.bn5 = nn.BatchNorm2d(depth * 32, momentum=0.1, eps=1e-3)
        self.dropout = nn.Dropout(dropout)

        self._final_spatial = _conv_tower_output_size(
            n_points, 2, n_layers=6, kernel_size=self.kernel_size, stride=self.stride
        )
        flat_dim = depth * 32 * self._final_spatial[0] * self._final_spatial[1]

        self.fc0 = nn.Linear(flat_dim, 1024)
        self.fc0_bn = nn.BatchNorm1d(1024, momentum=0.1, eps=1e-3)

        self.d_head = nn.Linear(1024, 1)

        self.q_dense = nn.Linear(1024, 128)
        self.q_bn = nn.BatchNorm1d(128, momentum=0.1, eps=1e-3)
        self.q_mean = nn.Linear(128, latent_dim)
        self.q_logstd = nn.Linear(128, latent_dim)

    def forward(self, x):
        # x arrives as (batch, n_points, 2, 1) NHWC-style (matching the
        # real repo's tensor layout); convert to torch NCHW.
        x = x.squeeze(-1).unsqueeze(1)  # batch x 1 x n_points x 2

        for conv, bn in (
            (self.conv0, self.bn0),
            (self.conv1, self.bn1),
            (self.conv2, self.bn2),
            (self.conv3, self.bn3),
            (self.conv4, self.bn4),
            (self.conv5, self.bn5),
        ):
            x = _tf_same_pad2d(x, self.kernel_size, self.stride)
            x = self.dropout(F.leaky_relu(bn(conv(x)), negative_slope=0.2))

        x = x.reshape(x.shape[0], -1)
        x = F.leaky_relu(self.fc0_bn(self.fc0(x)), negative_slope=0.2)

        d = self.d_head(x)

        q = F.leaky_relu(self.q_bn(self.q_dense(x)), negative_slope=0.2)
        q_mean = self.q_mean(q)
        q_logstd = torch.clamp(self.q_logstd(q), min=-16)
        q_mean = q_mean.view(-1, 1, self.latent_dim)
        q_logstd = q_logstd.view(-1, 1, self.latent_dim)
        q = torch.cat([q_mean, q_logstd], dim=1)  # batch x 2 x latent_dim

        return d, q


def _tf_same_pad2d(x, kernel_size, stride):
    """Explicit TF `padding='same'` pre-padding for conv2d (TF puts any odd
    extra padding after the input, torch's `padding=` argument always pads
    symmetrically so this can't be expressed via `nn.Conv2d(padding=...)`
    for even kernels / non-unit stride)."""
    kh, kw = kernel_size
    sh, sw = stride
    ih, iw = x.shape[-2], x.shape[-1]
    oh = -(-ih // sh)
    ow = -(-iw // sw)
    pad_h = max((oh - 1) * sh + kh - ih, 0)
    pad_w = max((ow - 1) * sw + kw - iw, 0)
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    return F.pad(x, (pl, pr, pt, pb))


def _conv_tower_output_size(h, w, n_layers, kernel_size, stride):
    """Compute the spatial (H, W) after `n_layers` conv2d layers using TF
    `'same'` padding semantics (ceil-division output size), matching the
    real discriminator's `tf.layers.conv2d(..., padding='same')` towers."""
    kh, kw = kernel_size
    sh, sw = stride
    for _ in range(n_layers):
        h = -(-h // sh)
        w = -(-w // sw)
    return (h, w)


class BezierGAN(nn.Module):
    """Thin wrapper composing the real repo's `generator()` + one
    `discriminator()` call on the synthesized curve, matching the real
    `GAN.train()` forward data flow (`x_fake = generator(c, z)`, then
    `discriminator(x_fake)`); a single forward pass exercises both real
    sub-networks end-to-end for tracing."""

    def __init__(self, latent_dim, noise_dim, n_points, bezier_degree):
        super().__init__()
        self.generator = Generator(latent_dim, noise_dim, n_points, bezier_degree)
        self.discriminator = Discriminator(latent_dim, n_points)

    def forward(self, c, z):
        dp, cp, w, ub, db = self.generator(c, z)
        d, q = self.discriminator(dp)
        return dp, d, q


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). The real repo's
# own `train.py` drives `GAN(latent_dim=3, noise_dim=10, n_points=X_train.
# shape[1], bezier_degree=31, bounds=(0.0,1.0))` on real airfoil coordinate
# data (n_points ~ 192 in the paper's dataset); the real code requires
# `(bezier_degree+1) % 8 == 0` (dim_cpw = (bezier_degree+1)//8 reshape), so
# we keep a real divisible bezier_degree=7 (dim_cpw=1) and shrink
# n_points/latent_dim/noise_dim for a fast CPU trace -- architecture
# unchanged.
# ---------------------------------------------------------------------------
_BATCH = 4
_LATENT_DIM = 3
_NOISE_DIM = 8
_N_POINTS = 48  # divisible by 2**6 so the 6-layer stride-2 conv tower ends cleanly
_BEZIER_DEGREE = 7


def build_beziergan_airfoil():
    torch.manual_seed(0)
    model = BezierGAN(
        latent_dim=_LATENT_DIM,
        noise_dim=_NOISE_DIM,
        n_points=_N_POINTS,
        bezier_degree=_BEZIER_DEGREE,
    )
    model.eval()
    return model


def example_input_beziergan_airfoil():
    torch.manual_seed(0)
    c = torch.rand(_BATCH, _LATENT_DIM)
    z = torch.randn(_BATCH, _NOISE_DIM) * 0.5
    return (c, z)


MENAGERIE_ENTRIES = [
    (
        "BezierGAN-Airfoil",
        "build_beziergan_airfoil",
        "example_input_beziergan_airfoil",
        2019,
        MENAGERIE_ZOO,
    ),
]
