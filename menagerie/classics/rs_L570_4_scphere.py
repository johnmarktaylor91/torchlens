# FAITHFUL PORT of klarman-cell-observatory/scPhere @ master (original framework: TensorFlow 1.x / tensorflow_probability)
# https://github.com/klarman-cell-observatory/scPhere/blob/master/scphere/model/vae.py
# https://github.com/klarman-cell-observatory/scPhere/blob/master/scphere/distributions/von_mises_fisher.py
# https://github.com/klarman-cell-observatory/scPhere/blob/master/scphere/ops/ive.py
# https://github.com/klarman-cell-observatory/scPhere/blob/master/scphere/util/util.py
#
# scPhere: a single-cell RNA-seq VAE whose headline contribution is a
# hyperspherical (von Mises-Fisher) latent space instead of a Gaussian one.
# The original code is TF1-graph-mode (tf.compat.v1.placeholder/Session,
# tf.contrib, tensorflow_probability distributions) and cannot run in a
# base torch env, so this ports the `latent_dist='vmf'`,
# `observation_dist='nb'` path of `SCPHERE._encoder` / `SCPHERE._decoder`
# (Dense+BatchNorm MLP stacks, faithfully translated layer-for-layer) and
# the `VonMisesFisher` reparameterized-rejection sampler (`_sample_n`,
# `__sample_w3`, `__sample_w_rej`, `__householder_rotation`) from
# von_mises_fisher.py, translated op-for-op from TF ops to the torch
# equivalents. `ive` (exponentially-scaled modified Bessel function of the
# first kind) is ported from a TF custom_gradient wrapping
# `scipy.special.ive` into a torch autograd.Function wrapping the same
# scipy call, preserving the analytic gradient
# `d/dz ive(v,z) = ive(v-1,z) - ive(v,z)*(v+z)/z` from ive.py. The NB
# log-likelihood is ported from util.py's `log_likelihood_nb`. Training-time
# pieces (the depth regularizer, the TF1 Saver/Session helpers, the
# `OptimizerVAE` gradient-clipping optimizer, and the `normal`/`wn`
# (hyperbolic wrapped-normal) latent variants) are outside the traced
# forward path and are omitted; this ports the model's namesake vMF path.

import math

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F


MENAGERIE_ZOO = "ported-pytorch"

EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10


# --- ported from ops/ive.py ----------------------------------------------


class _IveFunction(torch.autograd.Function):
    """Exponentially scaled modified Bessel function of the first kind.

    Ported from the TF `@custom_gradient` `ive(v, z)` in ive.py, which wraps
    `scipy.special.ive` (special-cased for v==0/v==1 via i0e/i1e, matching
    the source's `np.select`) with the analytic gradient
    `d/dz ive(v, z) = ive(v-1, z) - ive(v, z) * (v + z) / z`.
    """

    @staticmethod
    def forward(ctx, v, z):
        z_np = z.detach().cpu().numpy()
        v_val = float(v)
        if v_val == 0:
            out_np = scipy.special.i0e(z_np)
        elif v_val == 1:
            out_np = scipy.special.i1e(z_np)
        else:
            out_np = scipy.special.ive(v_val, z_np)
        out = torch.as_tensor(out_np, dtype=z.dtype, device=z.device)
        ctx.save_for_backward(z, out)
        ctx.v = v_val
        return out

    @staticmethod
    def backward(ctx, grad_output):
        z, out = ctx.saved_tensors
        v = ctx.v
        ive_v_minus_1 = ive(v - 1, z)
        grad_z = grad_output * (ive_v_minus_1 - out * (v + z) / z)
        return None, grad_z


def ive(v, z):
    return _IveFunction.apply(v, z)


# --- ported from distributions/von_mises_fisher.py -----------------------


class VonMisesFisher:
    """The von-Mises-Fisher distribution with location `loc` (unit vector)
    and concentration `scale`, ported op-for-op from `VonMisesFisher` in
    von_mises_fisher.py (the `hyperspherical_vae`-derived rejection sampler:
    `_sample_n` dispatches to `__sample_w3` for m==3 and `__sample_w_rej`
    (Wood 1994 rejection sampling via a Beta proposal) otherwise, then
    applies the same Householder rotation to move the sampled point from
    the standard direction e1 to `loc`).
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self.m = loc.shape[-1]
        self.mf = float(self.m)
        e1 = torch.zeros(1, self.m, dtype=loc.dtype, device=loc.device)
        e1[0, 0] = 1.0
        self.e1 = e1

    def sample(self):
        batch_shape = self.scale.shape[:-1]
        n = batch_shape[0] if len(batch_shape) > 0 else 1

        if self.m == 3:
            w = self.__sample_w3(n)
        else:
            w = self.__sample_w_rej(n)

        w = torch.clamp(w, -1 + 1e-6, 1 - 1e-6)
        v_raw = torch.randn(
            self.scale.shape[:-1] + (self.m,), dtype=self.scale.dtype, device=self.scale.device
        )
        v = F.normalize(v_raw[..., 1:], p=2, dim=-1)

        tmp = torch.sqrt(1.0 + w) * torch.sqrt(1.0 - w)
        x = torch.cat((w, tmp * v), dim=-1)
        z = self.__householder_rotation(x)

        return z

    def __sample_w3(self, n):
        shape = self.scale.shape
        u = torch.rand(shape, dtype=self.scale.dtype, device=self.scale.device)
        u = torch.clamp(u, 1e-16, 1 - 1e-16)

        logsumexp_input = torch.stack([torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0)
        w = 1 + torch.logsumexp(logsumexp_input, dim=0) / self.scale
        return w

    def __sample_w_rej(self, n):
        tmp = torch.sqrt((4 * (self.scale**2)) + (self.mf - 1) ** 2)
        b = (self.mf - 1.0) / (2.0 * self.scale + tmp)

        w = self.__rejection_loop(b)
        return w

    def __rejection_loop(self, b0):
        shape = b0.shape
        w = torch.zeros_like(b0)
        mask = torch.ones_like(b0, dtype=torch.bool)
        b = b0

        max_iters = 100
        for _ in range(max_iters):
            if not mask.any():
                break
            conc = (self.mf - 1.0) / 2.0
            e = torch.distributions.Beta(
                torch.full_like(b, conc), torch.full_like(b, conc)
            ).sample()
            u = torch.rand(shape, dtype=self.scale.dtype, device=self.scale.device)
            w_candidate = (1.0 - (1.0 + b) * e) / (1.0 - (1.0 - b) * e)
            x = (1.0 - b) / (1.0 + b)
            c = self.scale * x + (self.mf - 1) * torch.log1p(-(x**2))

            tmp = torch.clamp(x * w_candidate, 0, 1 - 1e-16)
            reject = (
                (self.mf - 1.0) * torch.log(1.0 - tmp) + self.scale * w_candidate - c
            ) < torch.log(u)
            accept = ~reject

            update = mask & accept
            w = torch.where(update, w_candidate, w)
            mask = torch.where(update, reject, mask)

        return w

    def __householder_rotation(self, x):
        u = F.normalize(self.e1 - self.loc, p=2, dim=-1)
        z = x - 2 * torch.sum(u * x, dim=-1, keepdim=True) * u
        return z

    def entropy(self):
        return (
            -(
                self.scale * ive(self.mf / 2.0, self.scale) / ive(self.mf / 2.0 - 1, self.scale)
            ).reshape(self.scale.shape[:-1])
            - self._log_normalization()
        )

    def _log_normalization(self):
        output = (
            (self.mf / 2.0 - 1) * torch.log(self.scale)
            - (self.mf / 2.0) * math.log(2 * math.pi)
            - (self.scale + torch.log(ive(self.mf / 2.0 - 1, self.scale)))
        )
        return output.reshape(self.scale.shape[:-1])


# --- ported from util/util.py --------------------------------------------


def log_likelihood_nb(x, mu, sigma, eps=1e-16):
    log_mu_sigma = torch.log(mu + sigma + eps)

    ll = (
        torch.lgamma(x + sigma)
        - torch.lgamma(sigma)
        - torch.lgamma(x + 1)
        + sigma * torch.log(sigma + eps)
        - sigma * log_mu_sigma
        + x * torch.log(mu + eps)
        - x * log_mu_sigma
    )

    return torch.sum(ll, dim=-1)


# --- ported from model/vae.py (SCPHERE, vmf + nb path) -------------------


class SCPHERE(nn.Module):
    """Faithful port of `SCPHERE.__init__/_encoder/_decoder` for
    `latent_dist='vmf'`, `observation_dist='nb'`, `batch_invariant=False`,
    single batch covariate (`n_batch` an int, not a list). Encoder/decoder
    MLPs use Linear+BatchNorm1d+activation stacks matching the original's
    `tf.keras.layers.Dense`+`BatchNormalization` chain layer-for-layer.
    """

    def __init__(
        self, n_gene, n_batch, z_dim=2, encoder_layer=None, decoder_layer=None, activation=F.elu
    ):
        super().__init__()

        if encoder_layer is None:
            encoder_layer = [128, 64, 32]
        if decoder_layer is None:
            decoder_layer = [32, 128]

        self.n_input_feature = n_gene
        self.n_batch = n_batch
        # latent_dist == 'vmf': z_dim += 1 (extra unit-sphere dimension)
        self.z_dim = z_dim + 1
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.activation = activation

        # --- encoder net ---
        enc_in = n_gene + n_batch
        self.enc_linears = nn.ModuleList()
        self.enc_bns = nn.ModuleList()
        prev = enc_in
        for width in encoder_layer:
            self.enc_linears.append(nn.Linear(prev, width))
            self.enc_bns.append(nn.BatchNorm1d(width))
            prev = width

        # vmf latent heads
        self.z_sigma_head = nn.Linear(prev, 1)
        self.z_mu_head = nn.Linear(prev, self.z_dim)

        # --- decoder net ---
        dec_in = self.z_dim + n_batch
        self.dec_linears = nn.ModuleList()
        self.dec_bns = nn.ModuleList()
        prev = dec_in
        for width in decoder_layer:
            self.dec_linears.append(nn.Linear(prev, width))
            self.dec_bns.append(nn.BatchNorm1d(width))
            prev = width

        # nb observation heads
        self.mu_head = nn.Linear(prev, n_gene)
        self.sigma_sq_head = nn.Linear(prev, n_gene)

    def _encoder(self, x, batch):
        # observation_dist == 'nb': log1p, then l2-normalize (vmf branch)
        x = torch.log1p(x)
        x = F.normalize(x, p=2, dim=-1)

        x = torch.cat([x, batch], dim=1)

        h = x
        for linear, bn in zip(self.enc_linears, self.enc_bns):
            h = self.activation(linear(h))
            h = bn(h)

        z_sigma_square = F.softplus(self.z_sigma_head(h)) + 1
        z_sigma_square = torch.clamp(z_sigma_square, 1, 10000)

        z_mu = F.normalize(self.z_mu_head(h), p=2, dim=-1)

        return z_mu, z_sigma_square

    def _decoder(self, z, batch, library_size):
        z = torch.cat([z, batch], dim=1)

        h = z
        for linear, bn in zip(self.dec_linears, self.dec_bns):
            h = self.activation(linear(h))
            h = bn(h)

        mu = F.softmax(self.mu_head(h), dim=-1)
        mu = mu * library_size

        sigma_square = F.softplus(self.sigma_sq_head(h))
        sigma_square = torch.mean(sigma_square, dim=0)

        sigma_square = torch.clamp(sigma_square, EPS, MAX_SIGMA_SQUARE)

        return mu, sigma_square

    def forward(self, x, batch_id):
        # one-hot the batch id (multi_one_hot with a single n_batch entry)
        batch = F.one_hot(batch_id, self.n_batch).to(x.dtype)

        library_size = torch.sum(x, dim=1, keepdim=True)

        z_mu, z_sigma_square = self._encoder(x, batch)

        q_z = VonMisesFisher(z_mu, z_sigma_square)
        z = q_z.sample()

        mu, sigma_square = self._decoder(z, batch, library_size)

        log_likelihood = torch.mean(log_likelihood_nb(x, mu, sigma_square, eps=1e-10))

        return mu, sigma_square, z, log_likelihood


def build_scphere():
    return SCPHERE(
        n_gene=48,
        n_batch=3,
        z_dim=2,
        encoder_layer=[32, 16],
        decoder_layer=[16, 32],
    )


def example_input_scphere():
    batch_size = 8
    n_gene = 48
    n_batch = 3
    x = torch.rand(batch_size, n_gene) * 5.0
    batch_id = torch.randint(0, n_batch, (batch_size,))
    return (x, batch_id)


MENAGERIE_ENTRIES = [
    ("scPhere", "build_scphere", "example_input_scphere", 2021, "ported-pytorch"),
]
