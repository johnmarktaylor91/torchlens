# FAITHFUL PORT of scvae/scvae @ 8bf173865658dd4e44d2e369bfdffc8981db5332 (original framework: TensorFlow 1.x + TensorFlow Probability)
# Original: scvae/models/variational_autoencoder.py (VariationalAutoencoder._setup_model_graph)
#   + scvae/models/utilities.py (dense_layer / dense_layers)
#   + scvae/distributions/utilities.py (DISTRIBUTIONS["gaussian"], DISTRIBUTIONS["poisson"])
# The original graph-mode TF1 + TFP code (tf.placeholder / tf.variable_scope /
# tf.Session / tfp.distributions) cannot run in this base torch env and TF1's
# legacy graph API plus tensorflow-probability are not reasonably installable
# alongside torch here, so the architecture is transcribed faithfully into
# base-env torch below, scoped to scVAE's default configuration:
#   inference_architecture="MLP", generative_architecture="MLP",
#   hidden_sizes=[100], latent_size=2, latent_distribution="gaussian",
#   reconstruction_distribution="poisson", minibatch_normalisation=True,
#   batch_correction=False, count_sum=False, number_of_samples=1 (single
#   reparameterized draw, matching the default IW/MC sample count).
"""scVAE: a variational autoencoder for single-cell RNA-seq count data.

scVAE parameterises a Gaussian approximate posterior q(z|x) over a latent
code via an MLP encoder, reparameterizes to draw z, then decodes z through
a mirrored MLP into the rate of a Poisson reconstruction distribution
(log_lambda, clipped to the original model's [-10, 10] support before the
exp()). Both encoder and decoder dense layers apply BatchNorm1d between the
linear projection and the ReLU activation, matching the original
`dense_layer` helper's minibatch-normalisation insertion point (the default
config has `minibatch_normalisation=True`).
Reference: https://github.com/scvae/scvae
"""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


# --- ported from scvae/models/utilities.py: dense_layer/dense_layers ---


class DenseLayer(nn.Module):
    """Linear -> BatchNorm1d -> activation, matching dense_layer()'s
    minibatch_normalisation=True insertion point (norm before activation)."""

    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU() if activation else None

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DenseLayers(nn.Module):
    """Stack of DenseLayer, matching dense_layers()'s ENCODER/DECODER scope."""

    def __init__(self, in_features, hidden_sizes):
        super().__init__()
        dims = [in_features] + list(hidden_sizes)
        self.layers = nn.ModuleList(
            [DenseLayer(dims[i], dims[i + 1]) for i in range(len(hidden_sizes))]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# --- ported from scvae/models/variational_autoencoder.py: _setup_model_graph ---


class ScVae(nn.Module):
    def __init__(self, feature_size, latent_size=2, hidden_sizes=(100,)):
        super().__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size

        # ENCODER (inference_architecture="MLP").
        self.encoder = DenseLayers(feature_size, hidden_sizes)
        encoder_out_dim = hidden_sizes[-1] if hidden_sizes else feature_size

        # Posterior q(z|x) parameters: "gaussian" latent distribution uses
        # identity activation for both mu and log_sigma (dense_layer with no
        # activation_fn, i.e. a bare affine "parameter_variable" projection).
        self.mu_layer = nn.Linear(encoder_out_dim, latent_size)
        self.log_sigma_layer = nn.Linear(encoder_out_dim, latent_size)

        # DECODER (generative_architecture="MLP"); dense_layers(reverse_order=True)
        # over the same hidden_sizes, from the latent code (no batch_correction /
        # count_sum concatenation branches under the default config).
        self.decoder = DenseLayers(latent_size, tuple(reversed(hidden_sizes)))
        decoder_out_dim = hidden_sizes[0] if hidden_sizes else latent_size

        # Reconstruction distribution parameterisation: "poisson" has a
        # single "log_lambda" parameter, identity activation, support
        # [-10, 10] (clipped before rate = exp(log_lambda)).
        self.log_lambda_layer = nn.Linear(decoder_out_dim, feature_size)

    def forward(self, x):
        # ENCODER -> posterior parameters.
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        log_sigma = torch.clamp(self.log_sigma_layer(encoded), -3, 3)

        # Reparameterization trick, single sample (number_of_samples=1).
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        # DECODER -> reconstruction distribution parameters.
        decoded = self.decoder(z)
        log_lambda = torch.clamp(self.log_lambda_layer(decoded), -10, 10)
        reconstruction_rate = torch.exp(log_lambda)

        return reconstruction_rate, mu, log_sigma, z


# --- staging harness ---

_FEATURE_SIZE = 32
_LATENT_SIZE = 2
_HIDDEN_SIZES = (16,)


def build_scvae():
    model = ScVae(
        feature_size=_FEATURE_SIZE,
        latent_size=_LATENT_SIZE,
        hidden_sizes=_HIDDEN_SIZES,
    )
    model.eval()
    return model


def example_input_scvae():
    # (batch, feature_size) non-negative RNA-seq count matrix.
    return torch.poisson(torch.ones(4, _FEATURE_SIZE) * 3.0)


MENAGERIE_ENTRIES = [
    (
        "scVAE",
        "build_scvae",
        "example_input_scvae",
        2020,
        "ported-pytorch",
    ),
]
