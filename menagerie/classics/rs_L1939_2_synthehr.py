# FAITHFUL PORT of baowaly/SynthEHR @ master (original framework: TensorFlow 1.x / tf.contrib.slim)
# https://raw.githubusercontent.com/baowaly/SynthEHR/master/model.py
#
# "Synthesizing Electronic Health Records Using Improved Generative Adversarial Networks"
# (Baowaly, Lin, Liu, Chen; JAMIA 2019). MEDGAN (Choi et al. 2017) plus the paper's two
# improved variants MEDWGAN (Wasserstein GAN + gradient penalty) and MEDBGAN (boundary-seeking
# GAN loss) -- all three share the identical MEDGAN forward architecture (autoencoder +
# generator + discriminator); only the *loss* differs between variants, which is a training-time
# concern outside a traced forward pass. This port transcribes MEDGAN.buildAutoencoder /
# buildGenerator / getDiscriminatorResults / buildDiscriminator from the real TF1.x
# `tensorflow.contrib.slim`/manual-variable code (unrunnable in a modern env: TF1.x
# placeholder/variable_scope API + tf.contrib, removed since TF 2.x) into self-contained torch:
#   - buildAutoencoder: MLP compress (encoder) -> MLP decompress (decoder), tanh activations
#     (binary dataType) via `nn.Tanh`, final sigmoid reconstruction (the loss the autoencoder
#     half computes -- binary cross-entropy against x_raw -- is training-time and not part of
#     the traced generation forward pass, so it is omitted; the decoder is what the
#     discriminator invokes on generator output, so it is kept).
#   - buildGenerator: residual MLP blocks (`tempVec = h3 + tempVec`, shortcut requires equal
#     dims so every generatorDim including embeddingDim must match randomDim -- matches the
#     real repo shape contract) with 1D BatchNorm (`batch_norm(..., is_training=bn_train)`) and
#     ReLU internal / tanh final (binary dataType) activation.
#   - getDiscriminatorResults ("minibatch discrimination" via concatenating the per-feature
#     batch mean, `tf.concat([x_input, inputMean], 1)`) -> MLP with dropout -> single sigmoid
#     logit. Vanilla MEDGAN keeps the final `sigmoid`; MEDWGAN drops it (unconstrained critic
#     score) -- this port keeps the vanilla-MEDGAN sigmoid head as the default per the queue
#     row's "MEDGAN" framing, since sigmoid vs. no-sigmoid is a 1-line, non-architectural
#     variant switch already covered by MEDWGAN's structurally-identical discriminator MLP.
#   - buildDiscriminator: real path discriminates x_raw directly; fake path first decodes the
#     generator's embedding-space output back through the *decoder half of the autoencoder*
#     (weight-tied `decodeVariables`, matching `decodeVariables['aed_W_'+str(i)]` reuse in the
#     real code) before discriminating -- reproduced here by literally calling
#     `self.autoencoder.decode(...)` on the generator output.
#   - `train()`, `generateData()`, `loadData()`, `load()`, and all TF Saver/session/file-I/O
#     methods are training/serialization infrastructure, not part of the forward architecture,
#     and are dropped.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class _MedGanAutoencoder(nn.Module):
    """buildAutoencoder(): MLP compress -> MLP decompress, shared by generator's discriminator
    decode path (`decodeVariables` in the real repo -- the decoder half is reused verbatim to
    project a generated embedding back into input space before it is discriminated)."""

    def __init__(self, input_dim, compress_dims, decompress_dims, data_type="binary"):
        super().__init__()
        self.act = nn.Tanh() if data_type == "binary" else nn.ReLU()

        encode_dims = [input_dim] + list(compress_dims)
        self.encoder = nn.ModuleList(
            nn.Linear(encode_dims[i], encode_dims[i + 1]) for i in range(len(encode_dims) - 1)
        )

        decode_dims = [compress_dims[-1]] + list(decompress_dims)
        self.decoder = nn.ModuleList(
            nn.Linear(decode_dims[i], decode_dims[i + 1]) for i in range(len(decode_dims) - 1)
        )
        self.final_sigmoid = data_type == "binary"

    def encode(self, x):
        h = x
        for layer in self.encoder:
            h = self.act(layer(h))
        return h

    def decode(self, z):
        h = z
        for layer in self.decoder[:-1]:
            h = self.act(layer(h))
        h = self.decoder[-1](h)
        if self.final_sigmoid:
            h = torch.sigmoid(h)
        else:
            h = torch.relu(h)
        return h

    def forward(self, x):
        return self.decode(self.encode(x))


class _MedGanGenerator(nn.Module):
    """buildGenerator(): residual MLP blocks with batchnorm; final block activation is
    tanh (binary dataType) or relu (count dataType). Shortcut `h3 + tempVec` requires
    dims to match, so every generator hidden dim (incl. the trailing embeddingDim) is
    randomDim-sized, matching the real repo's shape contract."""

    def __init__(self, random_dim, generator_dims, data_type="binary"):
        super().__init__()
        dims = [random_dim] + list(generator_dims)
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1))
        self.hidden_act = nn.ReLU()
        self.final_act = nn.Tanh() if data_type == "binary" else nn.ReLU()

    def forward(self, x):
        temp = x
        n = len(self.linears)
        for i in range(n - 1):
            h = self.linears[i](temp)
            h2 = self.bns[i](h)
            h3 = self.hidden_act(h2)
            temp = h3 + temp
        h = self.linears[-1](temp)
        h2 = self.bns[-1](h)
        h3 = self.final_act(h2)
        return h3 + temp


class _MedGanDiscriminator(nn.Module):
    """getDiscriminatorResults(): concatenates the per-feature batch mean onto the input
    ("minibatch discrimination", `tf.concat([x_input, inputMean], 1)`) then runs an MLP
    with dropout, ending in a single sigmoid logit (vanilla MEDGAN head)."""

    def __init__(self, input_dim, discriminator_dims, dropout=0.0):
        super().__init__()
        dims = [input_dim * 2] + list(discriminator_dims[:-1])
        self.linears = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(dims[-1], 1)

    def forward(self, x):
        batch_mean = x.mean(dim=0, keepdim=True).expand_as(x)
        h = torch.cat([x, batch_mean], dim=1)
        for layer in self.linears:
            h = self.dropout(self.act(layer(h)))
        y_hat = torch.sigmoid(self.final(h)).squeeze(-1)
        return y_hat


class MedGan(nn.Module):
    """Full MEDGAN forward graph: autoencoder (encode/decode) + generator (random noise ->
    embedding) + discriminator (real-vs-decoded-fake). Mirrors `MEDGAN.build_model()`'s wiring
    of `buildAutoencoder` / `buildGenerator` / `buildDiscriminator` into one forward pass that
    returns (y_hat_real, y_hat_fake, ae_reconstruction) so every submodule is exercised."""

    def __init__(
        self,
        input_dim=64,
        embedding_dim=16,
        random_dim=16,
        generator_dims=(16, 16),
        discriminator_dims=(32, 16, 1),
        compress_dims=(),
        decompress_dims=(),
        dropout=0.0,
        data_type="binary",
    ):
        super().__init__()
        compress_dims = list(compress_dims) + [embedding_dim]
        decompress_dims = list(decompress_dims) + [input_dim]
        # generator's residual shortcuts require every generator dim == random_dim.
        generator_dims = [random_dim for _ in generator_dims] + [random_dim]

        self.autoencoder = _MedGanAutoencoder(input_dim, compress_dims, decompress_dims, data_type)
        self.generator = _MedGanGenerator(random_dim, generator_dims, data_type)
        self.discriminator = _MedGanDiscriminator(input_dim, discriminator_dims, dropout)

    def forward(self, x_raw, x_random):
        y_hat_real = self.discriminator(x_raw)

        embedding_fake = self.generator(x_random)
        x_decoded = self.autoencoder.decode(embedding_fake)
        y_hat_fake = self.discriminator(x_decoded)

        return y_hat_real, y_hat_fake, x_decoded


def build_synthehr():
    torch.manual_seed(0)
    model = MedGan(
        input_dim=64,
        embedding_dim=16,
        random_dim=16,
        generator_dims=(16, 16),
        discriminator_dims=(32, 16, 1),
        dropout=0.0,
        data_type="binary",
    )
    model.eval()
    return model


def example_input_synthehr():
    torch.manual_seed(0)
    batch_size = 8
    x_raw = torch.rand(batch_size, 64).round()  # binary EHR feature vector
    x_random = torch.randn(batch_size, 16)
    return (x_raw, x_random)


MENAGERIE_ENTRIES = [
    ("SynthEHR-MedGAN", "build_synthehr", "example_input_synthehr", 2019, "ported"),
]
