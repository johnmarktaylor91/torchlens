# SOURCE: vendored from https://github.com/mdribeiro/DeepCFD @ master
# (src/deepcfd/models/UNetEx.py, src/deepcfd/models/AutoEncoder.py)
#
# DeepCFD (Ribeiro, Rehman, Ahmed, Dengel 2020, arXiv:2004.08826) predicts
# steady-state laminar-flow fields (Ux, Uy, p) over 2D channel geometries
# directly from a signed-distance-function input encoding, replacing a CFD
# solver. `UNetEx` is the flagship architecture used for the paper's main
# results: a single shared weight-normed/batch-normed convolutional U-Net
# encoder feeding THREE independent decoder branches (one per output
# field: Ux, Uy, p), each performing its own max-unpool-based upsampling
# with skip connections from the shared encoder. Copied verbatim below
# (only the relative `from .AutoEncoder import create_layer` import was
# flattened into this single file); no architecture was altered.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# From src/deepcfd/models/AutoEncoder.py (create_layer helper, verbatim).
# ---------------------------------------------------------------------------
def create_layer(
    in_channels,
    out_channels,
    kernel_size,
    wn=True,
    bn=True,
    activation=nn.ReLU,
    convolution=nn.Conv2d,
):
    assert kernel_size % 2 == 1
    layer = []
    conv = convolution(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    if wn:
        conv = weight_norm(conv)
    layer.append(conv)
    if activation is not None:
        layer.append(activation())
    if bn:
        layer.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layer)


# ---------------------------------------------------------------------------
# From src/deepcfd/models/UNetEx.py, verbatim.
# ---------------------------------------------------------------------------
def create_encoder_block(
    in_channels, out_channels, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2
):
    encoder = []
    for i in range(layers):
        _in = out_channels
        _out = out_channels
        if i == 0:
            _in = in_channels
        encoder.append(create_layer(_in, _out, kernel_size, wn, bn, activation, nn.Conv2d))
    return nn.Sequential(*encoder)


def create_decoder_block(
    in_channels,
    out_channels,
    kernel_size,
    wn=True,
    bn=True,
    activation=nn.ReLU,
    layers=2,
    final_layer=False,
):
    decoder = []
    for i in range(layers):
        _in = in_channels
        _out = in_channels
        _bn = bn
        _activation = activation
        if i == 0:
            _in = in_channels * 2
        if i == layers - 1:
            _out = out_channels
            if final_layer:
                _bn = False
                _activation = None
        decoder.append(
            create_layer(_in, _out, kernel_size, wn, _bn, _activation, nn.ConvTranspose2d)
        )
    return nn.Sequential(*decoder)


def create_encoder(
    in_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2
):
    encoder = []
    for i in range(len(filters)):
        if i == 0:
            encoder_layer = create_encoder_block(
                in_channels, filters[i], kernel_size, wn, bn, activation, layers
            )
        else:
            encoder_layer = create_encoder_block(
                filters[i - 1], filters[i], kernel_size, wn, bn, activation, layers
            )
        encoder = encoder + [encoder_layer]
    return nn.Sequential(*encoder)


def create_decoder(
    out_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2
):
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            decoder_layer = create_decoder_block(
                filters[i], out_channels, kernel_size, wn, bn, activation, layers, final_layer=True
            )
        else:
            decoder_layer = create_decoder_block(
                filters[i],
                filters[i - 1],
                kernel_size,
                wn,
                bn,
                activation,
                layers,
                final_layer=False,
            )
        decoder = [decoder_layer] + decoder
    return nn.Sequential(*decoder)


class UNetEx(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        filters=[16, 32, 64],
        layers=3,
        weight_norm=True,
        batch_norm=True,
        activation=nn.ReLU,
        final_activation=None,
    ):
        super().__init__()
        assert len(filters) > 0
        self.final_activation = final_activation
        self.encoder = create_encoder(
            in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers
        )
        decoders = []
        for i in range(out_channels):
            decoders.append(
                create_decoder(1, filters, kernel_size, weight_norm, batch_norm, activation, layers)
            )
        self.decoders = nn.Sequential(*decoders)

    def encode(self, x):
        tensors = []
        indices = []
        sizes = []
        for encoder in self.encoder:
            x = encoder(x)
            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode(self, _x, _tensors, _indices, _sizes):
        y = []
        for _decoder in self.decoders:
            x = _x
            tensors = _tensors[:]
            indices = _indices[:]
            sizes = _sizes[:]
            for decoder in _decoder:
                tensor = tensors.pop()
                size = sizes.pop()
                ind = indices.pop()
                x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
                x = torch.cat([tensor, x], dim=1)
                x = decoder(x)
            y.append(x)
        return torch.cat(y, dim=1)

    def forward(self, x):
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). Real-paper defaults
# are in_channels=3 (SDF channels), out_channels=3 (Ux, Uy, p), filters=
# [8, 16, 32, 32], kernel_size=5, layers=3 -- shrunk here to a tiny 3-filter
# stack + small spatial grid for a fast CPU trace, same architecture shape.
# ---------------------------------------------------------------------------
def build_deepcfd_unetex():
    torch.manual_seed(0)
    model = UNetEx(in_channels=3, out_channels=3, kernel_size=3, filters=[8, 16, 32], layers=2)
    model.eval()
    return model


def example_input_deepcfd_unetex():
    torch.manual_seed(0)
    # Spatial dims must be divisible by 2**len(filters) (here 2**3=8) for the
    # max_pool2d/max_unpool2d round-trip to reconstruct exact sizes.
    return torch.randn(2, 3, 16, 16)


MENAGERIE_ENTRIES = [
    ("DeepCFD-UNetEx", "build_deepcfd_unetex", "example_input_deepcfd_unetex", 2020, MENAGERIE_ZOO),
]
