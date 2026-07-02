# SOURCE: vendored from qu-gg/torch-neural-ssm @ main
#   https://github.com/qu-gg/torch-neural-ssm
#   Files combined: models/CommonVAE.py (LatentStateEncoder, EmissionDecoder),
#   utils/layers.py (Gaussian, Flatten, UnFlatten), models/group_b2/RGN.py
#   (RGNResFunction, RGNRes -- "Residual Recurrent Generative Network" latent
#   dynamics function). The original RGNRes/LatentDynamicsModel subclasses
#   pytorch_lightning.LightningModule purely for the train/val-step + hydra-cfg
#   boilerplate; that boilerplate is stripped here (not architectural) and the
#   real nn.Module encoder/decoder/dynamics-function code is kept verbatim
#   (only replacing the hydra DictConfig with a plain SimpleNamespace of the
#   same field names/defaults used in the repo's example configs).
#
# Architecture: Neural State-Space Model (Neural SSM), "system identification"
# style -- a convolutional VAE-style encoder produces an initial latent state
# z0 from a short window of input frames, a residual MLP dynamics function
# rolls z0 forward autoregressively in latent space (used here as the ODE
# right-hand-side network, matching group_b2/RGN.py's RGNResFunction), and a
# transposed-convolutional decoder emits per-timestep frame reconstructions.

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

MENAGERIE_ZOO = "vendored-pytorch"


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, input):
        nc = input[0].numel() // (self.w**2)
        return input.view(input.size(0), nc, self.w, self.w)


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim, fix_variance=False):
        super().__init__()
        self.fix_variance = fix_variance
        self.mu = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(in_dim // 2, out_dim),
        )
        self.logvar = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(in_dim // 2, out_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + (noise * std)

    def forward(self, x):
        mu = self.mu(x)
        if self.fix_variance:
            logvar = torch.full_like(mu, fill_value=0.1)
        else:
            logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class LatentStateEncoder(nn.Module):
    """Convolutional encoder q(z0 | x_{0:z_amort}) -> initial latent state."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.z_amort = cfg.z_amort_train

        self.initial_encoder = nn.Sequential(
            nn.Conv2d(self.z_amort, cfg.num_filters, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(cfg.num_filters),
            nn.ReLU(),
            nn.Conv2d(
                cfg.num_filters, cfg.num_filters * 2, kernel_size=5, stride=2, padding=(2, 2)
            ),
            nn.BatchNorm2d(cfg.num_filters * 2),
            nn.ReLU(),
            nn.Conv2d(
                cfg.num_filters * 2, cfg.num_filters * 8, kernel_size=5, stride=2, padding=(2, 2)
            ),
            nn.BatchNorm2d(cfg.num_filters * 8),
            nn.ReLU(),
            nn.AvgPool2d(4),
            Flatten(),
        )

        self.stochastic_out = Gaussian(cfg.num_filters * 8, cfg.latent_dim)
        self.deterministic_out = nn.Linear(cfg.num_filters * 8, cfg.latent_dim)
        self.out_act = nn.Tanh()

        self.z_means = None
        self.z_logvs = None

    def kl_z_term(self):
        if self.cfg.stochastic is False:
            return 0.0
        batch_size = self.z_means.shape[0]
        mus, logvars = self.z_means.view([-1]), self.z_logvs.view([-1])
        q = Normal(mus, torch.exp(0.5 * logvars))
        n = Normal(
            torch.zeros(len(mus), device=mus.device), torch.ones(len(mus), device=mus.device)
        )
        return kl(q, n).view([batch_size, -1]).sum([1]).mean()

    def forward(self, x):
        z0 = self.initial_encoder(x[:, : self.z_amort])
        if self.cfg.stochastic is True:
            self.z_means, self.z_logvs, z0 = self.stochastic_out(z0)
        else:
            z0 = self.deterministic_out(z0)
        return self.out_act(z0)


class EmissionDecoder(nn.Module):
    """Transposed-convolutional decoder z_i -> x_i."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv_dim = cfg.num_filters * 4**3

        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, self.conv_dim),
            UnFlatten(4),
            nn.ConvTranspose2d(
                self.conv_dim // 16, cfg.num_filters * 4, kernel_size=4, stride=1, padding=(0, 0)
            ),
            nn.BatchNorm2d(cfg.num_filters * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                cfg.num_filters * 4, cfg.num_filters * 2, kernel_size=5, stride=2, padding=(1, 1)
            ),
            nn.BatchNorm2d(cfg.num_filters * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                cfg.num_filters * 2,
                cfg.num_filters,
                kernel_size=5,
                stride=2,
                padding=(1, 1),
                output_padding=(1, 1),
            ),
            nn.BatchNorm2d(cfg.num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(
                cfg.num_filters, cfg.num_channels, kernel_size=5, stride=1, padding=(2, 2)
            ),
            nn.Sigmoid(),
        )

    def forward(self, zts):
        x_rec = self.decoder(zts.contiguous().view([zts.shape[0] * zts.shape[1], -1]))
        x_rec = x_rec.view(
            [zts.shape[0], x_rec.shape[0] // zts.shape[0], self.cfg.dim, self.cfg.dim]
        )
        return x_rec


class RGNResFunction(nn.Module):
    """Standard Residual Recurrent Generative Network dynamics function."""

    def __init__(self, cfg):
        super().__init__()
        dynamics_network = [nn.Linear(cfg.latent_dim, cfg.num_hidden), nn.Tanh()]
        for _ in range(cfg.num_layers - 1):
            dynamics_network.extend([nn.Linear(cfg.num_hidden, cfg.num_hidden), nn.Tanh()])
        dynamics_network.extend([nn.Linear(cfg.num_hidden, cfg.latent_dim), nn.Tanh()])
        self.dynamics_network = nn.Sequential(*dynamics_network)

    def forward(self, t, z):
        return self.dynamics_network(z)


class RGNRes(nn.Module):
    """Latent dynamics as parameterized by a residual recurrent generative network.

    Ported from models.group_b2.RGN.RGNRes; the original subclassed
    models.CommonDynamics.LatentDynamicsModel (a pytorch_lightning.LightningModule)
    purely for training-loop boilerplate. Here it is a plain nn.Module holding the
    same encoder/decoder/dynamics_func submodules and the same forward() rollout.
    """

    def __init__(self, cfg, generation_len=4):
        super().__init__()
        self.cfg = cfg
        # generation_len is a fixed rollout-length hyperparameter in the
        # original repo's call sites (cfg.generation_len / cfg.generation_validation_len);
        # stored as a plain int attribute here so the module is a single-tensor-input
        # callable for tracing (the original forward(x, generation_len) took it as a
        # second positional int argument, not a tensor).
        self.generation_len = generation_len
        self.encoder = LatentStateEncoder(cfg)
        self.decoder = EmissionDecoder(cfg)
        self.dynamics_func = RGNResFunction(cfg)

    def forward(self, x):
        z_init = self.encoder(x)
        z_cur = z_init
        zts = [z_init]
        for _ in range(self.generation_len - 1):
            z_cur = self.dynamics_func(None, z_cur)
            zts.append(z_cur)
        zt = torch.stack(zts, dim=1)
        x_rec = self.decoder(zt)
        return x_rec, zt


def _default_cfg():
    # Matches the field names/defaults used throughout the repo's example
    # configs (cfg/model/*.yaml), sized down for a fast trace.
    return SimpleNamespace(
        z_amort_train=3,
        num_filters=4,
        latent_dim=8,
        stochastic=False,
        num_hidden=16,
        num_layers=2,
        num_channels=1,
        dim=32,
    )


def build_rgnres():
    cfg = _default_cfg()
    return RGNRes(cfg, generation_len=4)


def example_input_rgnres():
    cfg = _default_cfg()
    generation_len = 4
    # [BatchSize, GenerationLen, NumChannels, H, W] -- consumed as
    # x[:, :z_amort] inside LatentStateEncoder (channel dim doubles as time).
    x = torch.randn(2, generation_len, cfg.dim, cfg.dim)
    return (x,)


MENAGERIE_ENTRIES = [
    (
        "Neural State-Space Model (RGNRes)",
        build_rgnres,
        example_input_rgnres,
        2022,
        "vendored-pytorch",
    ),
]
