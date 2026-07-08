# FAITHFUL PORT of https://github.com/LiDan456/GAN-AD @ master (model.py: `generator`,
# `discriminator`, `GAN_loss`) (original framework: TensorFlow 1.x)
#
# GAN-AD (Li et al., "Anomaly Detection with Generative Adversarial Networks for Multivariate
# Time Series", ICS/SCADA process-anomaly detection on the SWaT testbed): a recurrent GAN --
# the repo's own top comment attributes the generator/discriminator design to
# https://github.com/ratschlab/RGAN ("Most of the models are copied from ... RGAN") -- whose
# generator is a single-layer LSTM run over a latent-noise sequence `Z`, projected per
# timestep through a Linear + tanh readout into the reconstructed multivariate sensor window;
# the discriminator is a single-layer LSTM run over a (real or generated) sensor window,
# projected per timestep through a Linear + sigmoid readout into a real/fake probability per
# timestep. Anomaly scoring at inference time (AD.py, not vendored -- pure numpy
# post-processing of scores, not an architectural component) combines a per-sample residual
# (real x vs. its DR-inverted generator reconstruction) with the discriminator's real/fake
# score; the module vendored here faithfully reproduces the GAN forward computation graph
# itself (`GAN_loss`'s `G_sample = generator(Z)`, `D_real, D_logit_real = discriminator(X)`,
# `D_fake, D_logit_fake = discriminator(G_sample, reuse=True)`), which is the actual learned
# architecture; TF1.x's variable-scope `reuse=True` mechanism (the discriminator LSTM/Linear
# sharing the SAME weights across the real and generated branches) is reproduced by literally
# calling the same `nn.LSTM`/`nn.Linear` submodules twice, which is the natural torch
# equivalent of TF1 scope-based weight reuse -- not an architecture change.
#
# TF1.x (tf.placeholder / tf.variable_scope / tf.contrib.rnn.LSTMCell / tf.nn.dynamic_rnn,
# `tf.logging`) cannot be reasonably installed alongside the base torch/numpy env (TF1 is EOL,
# incompatible with modern Python 3.11 / numpy ABI), so this is a torch transcription of the
# real TF1 `generator`/`discriminator` functions rather than a vendor-as-is; every layer,
# recurrence, and readout below matches the original 1:1 (single-layer LSTM -> per-timestep
# Linear -> nonlinearity, in both branches). Differential-privacy SGD (dp_optimizer/
# sanitizer/accountant) and the custom `mod_core_rnn_cell_impl.LSTMCell` (a bias-init-only
# fork of the stock TF LSTM cell) are training/optimizer concerns, not part of the forward
# architecture, and are not ported.
#
# Repo: https://github.com/LiDan456/GAN-AD @ master
# File: model.py (`generator`, `discriminator`, `GAN_loss`)

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class RGANGenerator(nn.Module):
    """Single-layer LSTM over latent noise Z -> per-timestep Linear -> tanh.

    Faithful port of model.py's `generator(z, hidden_units_g, seq_length, batch_size,
    num_generated_features, ...)`: `cell = LSTMCell(hidden_units_g, ...)` run via
    `tf.nn.dynamic_rnn`, then `logits_2d = rnn_outputs_2d @ W_out_G + b_out_G`, then
    `output = tanh(logits_2d)` (the `scale_out_G` learnable scale parameter exists in the
    original but is unused on the active `tanh` output path -- `output_2d` is set to
    `tf.nn.tanh(logits_2d)` directly, with the `scale_out_G`-multiplied line commented out --
    so it is intentionally omitted here, matching the code that actually executes).
    """

    def __init__(self, latent_dim, hidden_units_g, num_generated_features):
        super().__init__()
        self.rnn = nn.LSTM(input_size=latent_dim, hidden_size=hidden_units_g, batch_first=True)
        self.out = nn.Linear(hidden_units_g, num_generated_features)

    def forward(self, z):
        # z: [B, seq_length, latent_dim]
        rnn_outputs, _ = self.rnn(z)
        logits = self.out(rnn_outputs)
        return torch.tanh(logits)


class RGANDiscriminator(nn.Module):
    """Single-layer LSTM over a (real or generated) sensor window -> per-timestep
    Linear -> sigmoid.

    Faithful port of model.py's `discriminator(x, hidden_units_d, seq_length, batch_size,
    ...)`: `cell = tf.contrib.rnn.LSTMCell(hidden_units_d, ...)` run via
    `tf.nn.dynamic_rnn`, then `logits = einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D`,
    then `output = sigmoid(logits)`, returned as `(output, logits)` exactly as the original
    returns `(output, logits)` for use in the sigmoid-cross-entropy GAN loss.
    """

    def __init__(self, num_generated_features, hidden_units_d):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=num_generated_features, hidden_size=hidden_units_d, batch_first=True
        )
        self.out = nn.Linear(hidden_units_d, 1)

    def forward(self, x):
        # x: [B, seq_length, num_generated_features]
        rnn_outputs, _ = self.rnn(x)
        logits = self.out(rnn_outputs)
        output = torch.sigmoid(logits)
        return output, logits


class GANAD(nn.Module):
    """Faithful port of `GAN_loss(Z, X, generator_settings, discriminator_settings)`: runs
    the generator on latent noise `Z`, then the SAME discriminator (TF1 `reuse=True` weight
    sharing == calling the same submodule twice here) on both the real window `X` and the
    generated window `G_sample`, exactly matching the original real/fake GAN forward graph
    used to compute `D_loss`/`G_loss` at train time and the discriminator score used for
    anomaly detection at inference time (DR_discriminator.py / AD.py).
    """

    def __init__(self, latent_dim, hidden_units_g, hidden_units_d, num_generated_features):
        super().__init__()
        self.generator = RGANGenerator(latent_dim, hidden_units_g, num_generated_features)
        self.discriminator = RGANDiscriminator(num_generated_features, hidden_units_d)

    def forward(self, z, x):
        g_sample = self.generator(z)
        d_real, d_logit_real = self.discriminator(x)
        d_fake, d_logit_fake = self.discriminator(g_sample)
        return g_sample, d_real, d_logit_real, d_fake, d_logit_fake


def build_gan_ad():
    # Real defaults per experiments/settings/swat_gen.txt (hidden_units_g=hidden_units_d=100,
    # latent_dim=15, seq_length=120, num_signals/num_generated_features=51 for SWaT) shrunk
    # for a fast trace; architecture (single-layer LSTM + Linear readout, both branches)
    # unchanged.
    return GANAD(latent_dim=4, hidden_units_g=8, hidden_units_d=8, num_generated_features=5)


def example_input_gan_ad():
    batch, seq_length, latent_dim, num_generated_features = 2, 6, 4, 5
    z = torch.randn(batch, seq_length, latent_dim)
    x = torch.randn(batch, seq_length, num_generated_features)
    return (z, x)


MENAGERIE_ENTRIES = [
    (
        "GAN-AD (RGAN for ICS/SWaT Anomaly Detection)",
        build_gan_ad,
        example_input_gan_ad,
        2018,
        MENAGERIE_ZOO,
    ),
]
