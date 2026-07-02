# FAITHFUL PORT of MadhumitaSushil/SDAE @ master (original framework: Keras 1.x /
# Theano)
#
# The repo's `sdae.py` hardcodes `KERAS_BACKEND=theano` and uses the Keras-1.x
# functional API (`Dense(output_dim=..., init=...)`), which is not installable in
# the base env, so per the menagerie ladder this is transcribed FAITHFULLY into
# self-contained torch rather than vendored as-is.
#
# `StackedDenoisingAE.get_pretrained_sda()` (sdae.py) greedily pretrains `n_layers`
# single-hidden-layer denoising autoencoders (Dropout -> Dense(enc_act) ->
# Dense(dec_act), one per layer, each trained on the previous layer's hidden
# output), then `_build_model_from_encoders()` assembles the trained encoder
# halves into one feed-forward `Sequential(Dropout, encoder_0, encoder_1, ...)` --
# that assembled Sequential is the traceable inference-time model, ported here as
# `SDAEEncoderStack`. Every architectural element is preserved: masking dropout on
# the input of the first layer, one Dense-with-activation encoder layer per stack
# level, and the corresponding Dense-with-activation decoder used only during the
# (untraced) per-layer pretraining phase -- both encoder and decoder halves are
# ported below so `StackedDenoisingAE` mirrors the real class, but only the encoder
# stack is the forward-traceable inference model per `get_pretrained_sda`'s own
# `get_enc_model=True` default return path.
#
# Repo: https://github.com/MadhumitaSushil/SDAE

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

_ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class _DenoisingAELayer(nn.Module):
    """Port of one per-layer autoencoder built inside
    `StackedDenoisingAE.get_pretrained_sda()`: Dropout(input) -> Dense(encoder_layer,
    enc_act) -> Dense(decoder_layer, dec_act)."""

    def __init__(self, in_features, n_hid, dropout, enc_act, dec_act, bias=True):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.encoder_layer = nn.Linear(in_features, n_hid, bias=bias)
        self.encoder_act = _ACTIVATIONS[enc_act]()
        self.decoder_layer = nn.Linear(n_hid, in_features, bias=bias)
        self.decoder_act = _ACTIVATIONS[dec_act]()

    def forward(self, x):
        x = self.dropout_layer(x)
        encoded = self.encoder_act(self.encoder_layer(x))
        decoded = self.decoder_act(self.decoder_layer(encoded))
        return encoded, decoded


class StackedDenoisingAE(nn.Module):
    """Port of `StackedDenoisingAE` (sdae.py). Builds `n_layers` per-layer
    denoising autoencoders (as `_build_layers` mirrors the real
    `get_pretrained_sda` construction loop), matching the real class's per-layer
    hidden/dropout/activation-list broadcasting semantics."""

    def __init__(
        self,
        n_layers=1,
        n_hid=(500,),
        dropout=(0.05,),
        enc_act=("sigmoid",),
        dec_act=("linear",),
        bias=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hid, self.dropout, self.enc_act, self.dec_act = self._assert_input(
            n_layers, list(n_hid), list(dropout), list(enc_act), list(dec_act)
        )
        self.bias = bias
        self.layers = None  # built lazily once input dim is known, as in get_pretrained_sda

    def _assert_input(self, n_layers, n_hid, dropout, enc_act, dec_act):
        if len(n_hid) == 1:
            n_hid = n_hid * n_layers
        if len(dropout) == 1:
            dropout = dropout * n_layers
        if len(enc_act) == 1:
            enc_act = enc_act * n_layers
        if len(dec_act) == 1:
            dec_act = dec_act * n_layers
        assert n_layers == len(n_hid) == len(dropout) == len(enc_act) == len(dec_act)
        return n_hid, dropout, enc_act, dec_act

    def build_layers(self, n_in):
        """Mirrors the per-layer construction loop inside
        `get_pretrained_sda`: layer 0 sees the raw input width; layer i>0 sees the
        previous layer's hidden width (the real code re-derives this by replacing
        `data_in` with the previous layer's hidden output each iteration)."""
        modules = []
        cur_in = n_in
        for cur_layer in range(self.n_layers):
            modules.append(
                _DenoisingAELayer(
                    cur_in,
                    self.n_hid[cur_layer],
                    self.dropout[cur_layer],
                    self.enc_act[cur_layer],
                    self.dec_act[cur_layer],
                    self.bias,
                )
            )
            cur_in = self.n_hid[cur_layer]
        self.layers = nn.ModuleList(modules)
        return self

    def forward(self, x):
        """Full stacked-autoencoder forward: chains each layer's (encoded,
        decoded) pair, feeding the encoded hidden representation forward as the
        next layer's input -- matching how `get_pretrained_sda` iteratively
        replaces `data_in` with the prior layer's hidden output."""
        encodeds = []
        decodeds = []
        h = x
        for layer in self.layers:
            h, d = layer(h)
            encodeds.append(h)
            decodeds.append(d)
        return encodeds, decodeds


class SDAEEncoderStack(nn.Module):
    """Port of `StackedDenoisingAE._build_model_from_encoders()`: the deployed
    inference model is `Sequential(Dropout(dropout[0]), encoder_layer_0,
    activation_0, encoder_layer_1, activation_1, ...)` built from the pretrained
    per-layer encoder halves. This is the model `get_pretrained_sda(...,
    get_enc_model=True)` returns by default and the one used downstream for
    low-dimensional representation extraction."""

    def __init__(self, sdae: StackedDenoisingAE):
        super().__init__()
        assert sdae.layers is not None, "call StackedDenoisingAE.build_layers() first"
        self.input_dropout = nn.Dropout(sdae.dropout[0])
        self.encoder_layers = nn.ModuleList([layer.encoder_layer for layer in sdae.layers])
        self.encoder_acts = nn.ModuleList([layer.encoder_act for layer in sdae.layers])

    def forward(self, x):
        x = self.input_dropout(x)
        for linear, act in zip(self.encoder_layers, self.encoder_acts):
            x = act(linear(x))
        return x


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_N_IN = 64


def build_sdae():
    sdae = StackedDenoisingAE(
        n_layers=3,
        n_hid=(256, 128, 64),
        dropout=(0.05,),
        enc_act=("sigmoid",),
        dec_act=("linear",),
        bias=True,
    ).build_layers(_N_IN)
    model = SDAEEncoderStack(sdae)
    model.eval()
    return model


def example_input_sdae():
    return torch.randn(4, _N_IN)


MENAGERIE_ENTRIES = [
    (
        "SDAE (Stacked Denoising Autoencoder)",
        build_sdae,
        example_input_sdae,
        2016,
        "PORT",
    ),
]
