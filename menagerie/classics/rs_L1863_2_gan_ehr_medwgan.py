# FAITHFUL PORT of baowaly/SynthEHR @ master (model.py `MEDGAN`/`MEDWGAN`
# classes, original framework: TensorFlow 1.x / tf.contrib.slim, cannot be
# installed alongside the modern base env -- tf.contrib was removed in
# TensorFlow 2). Transcribes the MEDWGAN generative-adversarial architecture
# (the improved WGAN-GP variant of medGAN, per the queue notes) mechanism-
# for-mechanism from the real code:
#   - autoencoder: `buildAutoencoder` -- stack of compress Linear+tanh/relu
#     layers down to `embeddingDim`, then decompress Linear+tanh/relu layers
#     back to `inputDim` (the decoder half is shared/reused by the generator
#     path, matching `decodeVariables` reuse in the original).
#   - generator: `buildGenerator` -- residual MLP blocks
#     (`tempVec = h3 + tempVec`) with BatchNorm1d between each Linear +
#     activation (tanh for binary EHR features, relu otherwise), operating in
#     the autoencoder's embedding space, followed by decoding via the shared
#     autoencoder decoder to produce `inputDim`-wide synthetic records.
#   - discriminator: `getDiscriminatorResults` -- minibatch-averaging feature
#     (`inputMean` concatenated with the record) fed through Linear+ReLU+
#     Dropout layers to a scalar WGAN critic score (no final sigmoid, per
#     MEDWGAN's `getDiscriminatorResults` override that removes
#     `tf.nn.sigmoid`).
# Gradient-penalty term (`ddx`) and the tf.Session-based training loop are
# training-time-only machinery and are intentionally not ported; forward
# structure (autoencoder decode + generator + discriminator) is preserved.
"""MEDWGAN: WGAN-GP synthetic EHR generator (medGAN family, WGAN-GP variant).

From Baowaly et al. 2019, "Synthesizing Electronic Health Records Using
Improved Generative Adversarial Networks" (JAMIA), extending Choi et al.'s
medGAN with an improved Wasserstein GAN + gradient penalty training scheme.
Architecture: pretrained autoencoder defines an embedding space; the
generator produces samples in that space via residual MLP blocks with batch
normalization, decoded through the (shared) autoencoder decoder; the critic
scores real vs. decoded-fake records via a minibatch-discrimination MLP.
"""

import torch
import torch.nn as nn


MENAGERIE_ZOO = "ported-pytorch"


class Autoencoder(nn.Module):
    """Compress/decompress MLP stack (`buildAutoencoder` in the original)."""

    def __init__(
        self, input_dim, embedding_dim, compress_dims, decompress_dims, data_type="binary"
    ):
        super().__init__()
        self.data_type = data_type
        self.activation = torch.tanh if data_type == "binary" else torch.relu

        compress_dims = list(compress_dims) + [embedding_dim]
        decompress_dims = list(decompress_dims) + [input_dim]

        compress_layers = []
        prev_dim = input_dim
        for dim in compress_dims:
            compress_layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.compress_layers = nn.ModuleList(compress_layers)

        decompress_layers = []
        prev_dim = embedding_dim
        for dim in decompress_dims:
            decompress_layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.decompress_layers = nn.ModuleList(decompress_layers)

    def encode(self, x):
        h = x
        for layer in self.compress_layers:
            h = self.activation(layer(h))
        return h

    def decode(self, z):
        h = z
        n = len(self.decompress_layers)
        for i, layer in enumerate(self.decompress_layers):
            h = layer(h)
            if i == n - 1:
                # final decompress layer uses sigmoid for binary, relu otherwise
                h = torch.sigmoid(h) if self.data_type == "binary" else torch.relu(h)
            else:
                h = self.activation(h)
        return h

    def forward(self, x):
        return self.decode(self.encode(x))


class ResidualGeneratorBlock(nn.Module):
    """One `tempVec = h3 + tempVec` residual block from `buildGenerator`."""

    def __init__(self, dim, activation):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.activation = activation

    def forward(self, x):
        h = self.linear(x)
        h = self.bn(h)
        h = self.activation(h)
        return h + x


class Generator(nn.Module):
    """`buildGenerator`: residual MLP blocks ending in a non-residual output layer."""

    def __init__(self, random_dim, generator_dims, data_type="binary"):
        super().__init__()
        activation = torch.tanh if data_type == "binary" else torch.relu
        self.data_type = data_type
        dims = list(generator_dims)
        assert dims[0] == random_dim, (
            "first generator hidden dim must equal random_dim (residual blocks)"
        )

        blocks = []
        for dim in dims[:-1]:
            blocks.append(ResidualGeneratorBlock(dim, activation))
        self.blocks = nn.ModuleList(blocks)

        out_dim = dims[-1]
        self.out_linear = nn.Linear(dims[-2] if len(dims) > 1 else random_dim, out_dim, bias=False)
        self.out_bn = nn.BatchNorm1d(out_dim)
        self.final_activation = torch.tanh if data_type == "binary" else torch.relu

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        out = self.out_linear(h)
        out = self.out_bn(out)
        out = self.final_activation(out)
        return out + h


class Discriminator(nn.Module):
    """`getDiscriminatorResults` (MEDWGAN override: no output sigmoid)."""

    def __init__(self, input_dim, discriminator_dims):
        super().__init__()
        dims = list(discriminator_dims)
        layers = []
        prev_dim = input_dim * 2  # concatenated with minibatch-mean feature
        for dim in dims[:-1]:
            layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.hidden_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(prev_dim, dims[-1])

    def forward(self, x):
        batch_mean = x.mean(dim=0, keepdim=True).expand_as(x)
        h = torch.cat([x, batch_mean], dim=1)
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
            h = self.dropout(h)
        # MEDWGAN critic score: no sigmoid (WGAN uses raw scores)
        return self.out(h).squeeze(-1)


class MEDWGAN(nn.Module):
    """Full MEDWGAN forward graph: generator -> decoder -> discriminator."""

    def __init__(
        self,
        input_dim=64,
        embedding_dim=16,
        random_dim=16,
        generator_dims=(16, 16, 16),
        discriminator_dims=(32, 16, 1),
        compress_dims=(),
        decompress_dims=(),
        data_type="binary",
    ):
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim, embedding_dim, compress_dims, decompress_dims, data_type
        )
        self.generator = Generator(random_dim, list(generator_dims) + [embedding_dim], data_type)
        self.discriminator = Discriminator(input_dim, discriminator_dims)

    def forward(self, x_real, x_random):
        fake_embedding = self.generator(x_random)
        x_fake = self.autoencoder.decode(fake_embedding)
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)
        return y_real, y_fake


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny dims, matches the repo's config family
# scaled down for fast tracing; data_type='binary' matches the paper's
# primary EHR diagnosis-code setting).
# ---------------------------------------------------------------------------


def build_gan_ehr_medwgan():
    model = MEDWGAN(
        input_dim=48,
        embedding_dim=12,
        random_dim=12,
        generator_dims=(12, 12),
        discriminator_dims=(24, 12, 1),
        data_type="binary",
    )
    model.eval()
    return model


def example_input_gan_ehr_medwgan():
    x_real = torch.rand(8, 48).round()
    x_random = torch.randn(8, 12)
    return (x_real, x_random)


MENAGERIE_ENTRIES = [
    (
        "GAN-EHR (MEDWGAN)",
        "build_gan_ehr_medwgan",
        "example_input_gan_ehr_medwgan",
        2019,
        "ported-pytorch",
    ),
]
