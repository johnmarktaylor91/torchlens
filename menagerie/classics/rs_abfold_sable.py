# FAITHFUL REIMPLEMENTATION from arXiv:2202.06159 (no public code) -- A/B sonnet
"""SABLE: Sequential VAE with gradient-reversal adversarial training for
cross-session neural-population alignment (arXiv:2202.06159).

Distinctive mechanism: a single-cell-resolution sequential VAE (LFADS-style
initial-condition generator) is trained with a **gradient reversal layer (GRL)**
placed between the encoder and the *neural reconstruction* decoder only (not the
behavior decoder). This makes the encoder adversarially maximize neural
reconstruction loss (i.e. try to make its latent code uninformative about
session-specific neural idiosyncrasies) while the separate behavior decoder
pulls the same latent code to be *maximally* informative about behavior. No
explicit discriminator network exists -- "session-invariance" falls directly out
of the GRL wired straight into the reconstruction pathway (paper Section 3):
"we reverse the backpropagation gradient between the neural reconstruction
decoder and the encoder... producing a latent space separable by behaviour but
not by session."
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity in the forward pass; negates (and scales) the gradient on backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_r: float) -> torch.Tensor:
        ctx.lambda_r = lambda_r
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_r * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_r: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_r)


class SABLE(nn.Module):
    """Sequential VAE + GRL cross-session alignment.

    Dims per paper Appendix B: encoder BiGRU 512 units x 3 layers, latent dim 64,
    neural-decoder GRU 256 units (single layer), behavior-decoder GRU 256 units
    x 2 layers, W_enc 512, W_fac 128, W_beh 512 (reduced for toy scale here, see
    spec card ASSUMPTIONS).
    """

    def __init__(
        self,
        num_neurons: int = 20,
        behavior_dim: int = 2,
        enc_hidden: int = 32,
        enc_layers: int = 3,
        latent_dim: int = 8,
        neural_dec_hidden: int = 16,
        behavior_dec_hidden: int = 16,
        behavior_dec_layers: int = 2,
        w_enc: int = 32,
        w_fac: int = 12,
        w_beh: int = 32,
        lambda_r: float = 1.0,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.latent_dim = latent_dim
        self.neural_dec_hidden = neural_dec_hidden
        self.behavior_dec_hidden = behavior_dec_hidden
        self.behavior_dec_layers = behavior_dec_layers
        self.lambda_r = lambda_r

        # --- Encoder: bidirectional GRU (Section 3, Eq. 1) ---
        self.encoder_gru = nn.GRU(
            input_size=num_neurons,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.w_enc = nn.Linear(2 * enc_hidden, w_enc)
        self.to_mu = nn.Linear(w_enc, latent_dim)
        self.to_logvar = nn.Linear(w_enc, latent_dim)

        # --- Bridges from latent initial condition z to each generator's initial state ---
        self.bridge_neural = nn.Linear(latent_dim, neural_dec_hidden)
        self.bridge_behavior = nn.Linear(latent_dim, behavior_dec_hidden * behavior_dec_layers)

        # --- Neural reconstruction decoder: autonomous GRU generator (Eq. 2) ---
        # Fed zeros at every step; all temporal structure comes from the initial
        # condition z (LFADS-style controller-free generator).
        self.neural_decoder_gru = nn.GRU(
            input_size=1, hidden_size=neural_dec_hidden, num_layers=1, batch_first=True
        )
        self.w_fac = nn.Linear(neural_dec_hidden, w_fac)
        self.rate_readout = nn.Linear(w_fac, num_neurons)

        # --- Behavior decoder: separate autonomous GRU generator (Eq. 3), NOT behind GRL ---
        self.behavior_decoder_gru = nn.GRU(
            input_size=1,
            hidden_size=behavior_dec_hidden,
            num_layers=behavior_dec_layers,
            batch_first=True,
        )
        self.w_beh = nn.Linear(behavior_dec_hidden, w_beh)
        self.behavior_readout = nn.Linear(w_beh, behavior_dim)

    def encode(self, spikes: torch.Tensor):
        # spikes: (batch, T, num_neurons)
        _, h_n = self.encoder_gru(spikes)
        # h_n: (num_layers*2, batch, enc_hidden) -- take last layer's fwd+bwd concat.
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        h_enc = torch.tanh(self.w_enc(h_cat))
        mu = self.to_mu(h_enc)
        logvar = self.to_logvar(h_enc)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, spikes: torch.Tensor):
        batch, T, _ = spikes.shape
        mu, logvar = self.encode(spikes)
        z = self.reparameterize(mu, logvar)

        # --- Neural reconstruction path: GRL sits between z and this decoder only ---
        z_neural = gradient_reversal(z, self.lambda_r)
        g0_neural = torch.tanh(self.bridge_neural(z_neural)).unsqueeze(0)  # (1, batch, H)
        zero_input = torch.zeros(batch, T, 1, device=spikes.device, dtype=spikes.dtype)
        g_neural, _ = self.neural_decoder_gru(zero_input, g0_neural)
        factors = self.w_fac(g_neural)
        rates = torch.nn.functional.softplus(self.rate_readout(factors))  # Poisson rate, Eq. 4-5

        # --- Behavior decoding path: normal gradient flow, NOT behind GRL ---
        g0_beh_flat = torch.tanh(self.bridge_behavior(z))  # (batch, H*layers)
        g0_beh = g0_beh_flat.view(batch, self.behavior_dec_layers, self.behavior_dec_hidden)
        g0_beh = g0_beh.permute(1, 0, 2).contiguous()
        g_beh, _ = self.behavior_decoder_gru(zero_input, g0_beh)
        beh_feat = self.w_beh(g_beh)
        behavior_pred = self.behavior_readout(beh_feat)

        return rates, behavior_pred, mu, logvar


def build_sable() -> SABLE:
    return SABLE(num_neurons=20, behavior_dim=2, enc_hidden=32, latent_dim=8)


def example_input_sable() -> torch.Tensor:
    return torch.rand(4, 25, 20)  # (batch, T, num_neurons) spike-count-like input


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    ("SABLE", "build_sable", "example_input_sable", 2022, "REIMPL"),
]
