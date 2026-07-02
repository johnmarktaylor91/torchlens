# SOURCE: vendored from https://github.com/pakheiyeung/ImplicitVol @ main (model.py)
# ImplicitVol: implicit neural representation for sensorless 3D ultrasound-freehand
# reconstruction (Yeung et al., arXiv:2109.12108). The classes below (`Sine`,
# `SirenLayer`, `OfficialNerf_siren`) plus the `encode_position` positional-encoding
# helper are copied verbatim from the official repo's model.py (MIT-style research
# code). `OfficialNerf_siren` is the actual per-slice implicit-volume network the
# repo trains in train.py: a SIREN-activated NeRF-style MLP mapping encoded 3D
# positions to intensity. Construction below (`pos_in_dims=63`, `D=128`) matches
# train.py's own `OfficialNerf_siren(pos_in_dims=63, D=128)` call, where
# pos_in_dims=63 comes from encode_position(..., levels=10, inc_input=True) on a
# 3-channel position (3 * (2*10 + 1) = 63). Only the CUDA `.cuda()` call and the
# LearnPose class (a separate camera-pose-learning module, not part of the implicit
# volume network) were dropped; the network architecture is untouched.
"""Vendored ImplicitVol SIREN-NeRF model definition (OfficialNerf_siren)."""

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


def encode_position(input, levels, inc_input):
    """
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.
    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    """

    # this is already doing 'log_sampling' in the official code.
    result_list = [input] if inc_input else []
    for i in range(levels):
        temp = 2.0**i * input  # (..., C)
        result_list.append(torch.sin(temp))  # (..., C)
        result_list.append(torch.cos(temp))  # (..., C)

    result_list = torch.cat(
        result_list, dim=-1
    )  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    return result_list  # (..., C*(2L+1))


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, w0=1.0, is_first=False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.activation = Sine(w0)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = w0
        self.c = 6
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.layer.weight.uniform_(-w_std, w_std)
            if self.layer.bias is not None:
                self.layer.bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.layer(x)
        out = self.activation(out)
        return out


class OfficialNerf_siren(nn.Module):
    def __init__(self, pos_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(OfficialNerf_siren, self).__init__()

        self.pos_in_dims = pos_in_dims

        self.layers0 = nn.Sequential(
            SirenLayer(pos_in_dims, D, use_bias=True, w0=30.0, is_first=True),
            SirenLayer(D, D, use_bias=True, w0=1.0, is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1.0, is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1.0, is_first=False),
        )

        self.layers1 = nn.Sequential(
            SirenLayer(D + pos_in_dims, D, use_bias=True, w0=1.0, is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1.0, is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1.0, is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1.0, is_first=False),
        )

        # self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.img_layers = SirenLayer(D, D // 2, use_bias=True, w0=1.0, is_first=False)
        self.fc_img = nn.Linear(D // 2, 1)

        # self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_img.bias.data = torch.tensor([0.02]).float()

    def forward(self, pos_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :return: rgb_density (H, W, N_sample, 1)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=-1)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        # x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.img_layers(feat)  # (H, W, N_sample, D/2)
        img = self.fc_img(x)  # (H, W, N_sample, 1)

        return img


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny config, matches train.py's own
# OfficialNerf_siren(pos_in_dims=63, D=128) construction; example input is a
# small grid of already-encoded 3D positions).
# ---------------------------------------------------------------------------

_D = 128
_LEVELS = 10
_POS_IN_DIMS = 3 * (2 * _LEVELS + 1)  # 63, matches train.py


def build_implicitvol():
    return OfficialNerf_siren(pos_in_dims=_POS_IN_DIMS, D=_D)


def example_input_implicitvol():
    # Small (H, W, N_sample, 3) grid of raw 3D positions, encoded the same way
    # train.py encodes sampled query points before feeding the network.
    raw_pos = torch.rand(4, 4, 2, 3)
    pos_enc = encode_position(raw_pos, levels=_LEVELS, inc_input=True)
    return (pos_enc,)


MENAGERIE_ENTRIES = [
    ("ImplicitVol", "build_implicitvol", "example_input_implicitvol", 2021, "vendored-pytorch"),
]
