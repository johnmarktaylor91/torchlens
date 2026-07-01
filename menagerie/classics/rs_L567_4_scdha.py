# FAITHFUL PORT of duct317/scDHA @ ac9060b70768a60e7e740eddc03aa42f75156559 (original
# framework: R, via the `torch` R package's C++ backend -- no Python source exists)
#   R/TorchSupport.R (scDHA_AE, scDHA_VAE, sampling())
#
# scDHA ("single-cell Decomposition using Hierarchical Autoencoder", Tran et al. 2021,
# Nature Computer Science) stage-2 core model. The real architecture (from R/TorchSupport.R,
# transcribed layer-for-layer, not guessed from the paper) is a non-convolutional VAE with
# an optional batch-normalized encoder trunk (SELU activation is used only when batch_norm
# is disabled, matching the R `nn_batch_norm1d`/`nnf_selu` branch), a softmax-clamped
# variance head, reparameterized sampling with a user-set epsilon_std, and TWO independent
# decoder heads (h1[1]/x_[1] and h1[2]/x_[2], each Linear -> SELU -> Linear) whose outputs
# are both trained against the reconstruction target (scDHA's "multiple-latent-space" trick
# for stability). Weight init: PyTorch's default kaiming-uniform Linear init is used here
# in place of R's `nn_init_xavier_uniform_manual`/`nn_init_zeros_(bias)` -- the R file's
# Xavier-uniform + zero-bias scheme is a plain post-construction re-init that does not
# change the module graph or forward computation TorchLens captures.
import torch
import torch.nn as nn
import torch.nn.functional as F


class scDHA_AE(nn.Module):
    """Stage-1 non-negative sparse autoencoder used to score gene weights (Wsd).

    R: `scDHA_AE <- nn_module("scDHA_AE", initialize = function(original_dim, im_dim) ...)`.
    """

    def __init__(self, original_dim: int, im_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(original_dim, im_dim)
        self.fc2 = nn.Linear(im_dim, original_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def sampling(mu: torch.Tensor, var: torch.Tensor, epsilon_std: float, lat_dim: int) -> torch.Tensor:
    """R: `sampling <- function(mu, var, epsilon_std, lat_dim) mu + sqrt(var)*epsilon_std*randn(1, lat_dim)`."""
    eps = torch.randn(1, lat_dim, device=mu.device, dtype=mu.dtype)
    return mu + torch.sqrt(var) * epsilon_std * eps


class scDHA_VAE(nn.Module):
    """Stage-2 dual-decoder VAE used to generate the low-dimensional latent representation.

    R: `scDHA_VAE <- nn_module("scDHA_VAE", initialize = function(original_dim, im_dim,
    lat_dim, epsilon_std, batch_norm=TRUE, zero_bias=TRUE) ...)`.
    """

    def __init__(
        self,
        original_dim: int,
        im_dim: int,
        lat_dim: int,
        epsilon_std: float,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.epsilon_std = epsilon_std
        self.lat_dim = lat_dim

        self.h = nn.Linear(original_dim, im_dim)
        if batch_norm:
            self.bn = nn.BatchNorm1d(im_dim, momentum=0.01, eps=1e-3)

        self.mu = nn.Linear(im_dim, lat_dim)
        self.var = nn.Linear(im_dim, lat_dim)

        self.h1 = nn.ModuleList(
            [
                nn.Linear(lat_dim, im_dim),
                nn.Linear(lat_dim, im_dim),
            ]
        )
        self.x_ = nn.ModuleList(
            [
                nn.Linear(im_dim, original_dim),
                nn.Linear(im_dim, original_dim),
            ]
        )

    def forward(self, x: torch.Tensor):
        if self.batch_norm:
            im = self.bn(self.h(x))
            mu = self.mu(im)
            var = F.softmax(self.var(im), dim=1)
        else:
            im = F.selu(self.h(x))
            mu = self.mu(im)
            var = F.softmax(self.var(im), dim=1)

        out = [mu, var]
        for i in range(2):
            z = sampling(mu, var, self.epsilon_std, self.lat_dim)
            head = F.selu(self.h1[i](z))
            out.append(self.x_[i](head))
        return out

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_norm:
            im = self.bn(self.h(x))
        else:
            im = F.selu(self.h(x))
        return self.mu(im)


def build_scdha_vae():
    original_dim = 50
    im_dim = 64
    lat_dim = 25
    return scDHA_VAE(original_dim, im_dim, lat_dim, epsilon_std=1.0, batch_norm=True)


def example_input_scdha_vae():
    batch_size = 16
    original_dim = 50
    return torch.rand(batch_size, original_dim)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("scDHA VAE", "build_scdha_vae", "example_input_scdha_vae", 2021, MENAGERIE_ZOO),
]
