# FAITHFUL PORT of eBay/RANSynCoders @ main (original framework: TensorFlow/Keras)
#
# The repo's `models.py` is TensorFlow (`tensorflow.python.keras.layers.Layer` /
# `Dense`) with no PyTorch implementation, and its pinned deps (`tensorflow==2.7.2`,
# `keras==2.3.1`) are not part of the base env, so per the menagerie ladder this is
# transcribed FAITHFULLY into self-contained torch rather than vendored as-is.
#
# Ported here: `RANCoders` (== `RANSynCoders`'s core anomaly-detection ensemble --
# the `self.rancoders` model built in `RANSynCoders.build()`), with its `Encoder`
# and `Decoder` layers. Every mechanism from the real `models.py` RANCODER section
# is preserved:
#   - `n_estimators` independent bootstrap-feature-subset encoder/decoder pairs
#     (`randsamples`: a random `max_features`-sized column subset per estimator,
#     fixed at build time -- ported as a non-trainable buffer of gathered indices).
#   - Each `Encoder` is a stack of `encoding_depth` Dense layers, each halving
#     width (`input_shape[-1] / 2**(i+1)`), followed by a `latent_dim` Dense.
#   - Each `Decoder` (used twice per estimator: "hi" and "lo") is a stack of
#     `decoding_depth` Dense layers growing back up (`output_dim / 2**(depth-i)`),
#     followed by a `restored` Dense back to `output_dim`.
#   - `call()` gathers each estimator's feature subset, encodes to the shared
#     latent, and decodes through the upper (`decoders_upper`) and lower
#     (`decoders_lower`) branches, returning `(o_hi, o_lo)` stacked over the
#     estimator axis -- i.e. per-estimator predicted upper/lower quantile bounds
#     used downstream for KPI anomaly-interval scoring.
#
# Repo: https://github.com/eBay/RANSynCoders

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class Encoder(nn.Module):
    """Port of RANCODER `Encoder` (models.py): `depth` halving-width Dense layers
    + a `latent_dim` Dense, each followed by `activation`."""

    def __init__(self, in_features, latent_dim, activation=nn.Identity, depth=2):
        super().__init__()
        self.depth = depth
        layers = []
        cur_in = in_features
        for i in range(depth):
            out_features = int(in_features / (2 ** (i + 1)))
            layers.append(nn.Linear(cur_in, out_features))
            layers.append(activation())
            cur_in = out_features
        self.hidden = nn.Sequential(*layers)
        self.latent = nn.Sequential(nn.Linear(cur_in, latent_dim), activation())

    def forward(self, x):
        x = self.hidden(x)
        return self.latent(x)


class Decoder(nn.Module):
    """Port of RANCODER `Decoder` (models.py): `depth` growing-width Dense layers
    + a `restored` Dense back to `output_dim`, each followed by `activation`
    (except the final `restored` layer, which uses `output_activation`)."""

    def __init__(
        self,
        in_features,
        output_dim,
        activation=nn.Identity,
        output_activation=nn.Identity,
        depth=2,
    ):
        super().__init__()
        self.depth = depth
        layers = []
        cur_in = in_features
        for i in range(depth):
            out_features = int(output_dim / (2 ** (depth - i)))
            layers.append(nn.Linear(cur_in, out_features))
            layers.append(activation())
            cur_in = out_features
        self.hidden = nn.Sequential(*layers)
        self.restored = nn.Sequential(nn.Linear(cur_in, output_dim), output_activation())

    def forward(self, x):
        x = self.hidden(x)
        return self.restored(x)


class RANCoders(nn.Module):
    """Port of RANCODER `RANCoders` (models.py): `n_estimators` independent
    Encoder/(Decoder_hi, Decoder_lo) triplets over a random `max_features`-sized
    bootstrap feature subset per estimator."""

    def __init__(
        self,
        n_features: int,
        n_estimators: int = 100,
        max_features: int = 3,
        encoding_depth: int = 2,
        latent_dim: int = 2,
        decoding_depth: int = 2,
        activation=nn.Identity,
        output_activation=nn.Identity,
        seed: int = 0,
    ):
        super().__init__()
        if n_features <= max_features:
            raise ValueError("n_features must exceed max_features")
        self.n_estimators = n_estimators
        self.max_features = max_features

        self.encoders = nn.ModuleList(
            [
                Encoder(max_features, latent_dim, activation, encoding_depth)
                for _ in range(n_estimators)
            ]
        )
        self.decoders_upper = nn.ModuleList(
            [
                Decoder(latent_dim, max_features, activation, output_activation, decoding_depth)
                for _ in range(n_estimators)
            ]
        )
        self.decoders_lower = nn.ModuleList(
            [
                Decoder(latent_dim, max_features, activation, output_activation, decoding_depth)
                for _ in range(n_estimators)
            ]
        )

        # The repo's `randsamples`: a fixed (n_estimators, max_features) bootstrap
        # feature-index table sampled once at `build()` time, stored as a
        # non-trainable weight. Ported as a registered (non-trainable) buffer.
        g = torch.Generator().manual_seed(seed)
        randsamples = torch.stack(
            [torch.randperm(n_features, generator=g)[:max_features] for _ in range(n_estimators)]
        )
        self.register_buffer("randsamples", randsamples)

    def forward(self, x):
        o_hi = []
        o_lo = []
        for i in range(self.n_estimators):
            xi = x.index_select(-1, self.randsamples[i])
            z = self.encoders[i](xi)
            o_hi.append(self.decoders_upper[i](z).unsqueeze(0))
            o_lo.append(self.decoders_lower[i](z).unsqueeze(0))
        return torch.cat(o_hi, dim=0), torch.cat(o_lo, dim=0)


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
# n_features=38 mirrors the repo's default SMD (machine-1-*) KPI count; n_estimators
# kept at the repo default of 100, max_features at the repo default of 3.
_N_FEATURES = 38


def build_ransyncoders():
    return RANCoders(n_features=_N_FEATURES, n_estimators=100, max_features=3)


def example_input_ransyncoders():
    return torch.randn(8, _N_FEATURES)


MENAGERIE_ENTRIES = [
    (
        "RANSynCoders",
        build_ransyncoders,
        example_input_ransyncoders,
        2021,
        "PORT",
    ),
]
