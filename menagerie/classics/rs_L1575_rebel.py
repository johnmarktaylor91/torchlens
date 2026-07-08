# SOURCE: vendored from facebookresearch/rebel @ 7960a42750f3407ea9eb2c3333d4c2a7961f6df4
# File: cfvpy/models.py
# ReBeL (Recursive Belief-based Learning), NeurIPS 2020, arxiv 2007.13544.
# Net2 is the real counterfactual-value network used for Liar's Dice self-play
# training in the official repo (public belief state -> per-action CFV output).

from __future__ import annotations

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


def build_mlp(
    *,
    n_in,
    n_hidden,
    n_layers,
    out_size=None,
    act=None,
    use_layer_norm=False,
    dropout=0,
):
    if act is None:
        act = GELU()
    build_norm_layer = (  # noqa: E731
        lambda: nn.LayerNorm(n_hidden) if use_layer_norm else nn.Sequential()
    )
    build_dropout_layer = (  # noqa: E731
        lambda: nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
    )

    last_size = n_in
    vals_net = []
    for _ in range(n_layers):
        vals_net.extend(
            [
                nn.Linear(last_size, n_hidden),
                build_norm_layer(),
                act,
                build_dropout_layer(),
            ]
        )
        last_size = n_hidden
    if out_size is not None:
        vals_net.append(nn.Linear(last_size, out_size))
    return nn.Sequential(*vals_net)


def input_size(num_faces, num_dice):
    return 1 + 1 + (2 * num_faces * num_dice + 1) + 2 * output_size(num_faces, num_dice)


def output_size(num_faces, num_dice):
    return num_faces**num_dice


class Net2(nn.Module):
    def __init__(
        self,
        *,
        num_faces,
        num_dice,
        n_hidden=256,
        use_layer_norm=False,
        dropout=0,
        n_layers=3,
    ):
        super().__init__()

        n_in = input_size(num_faces, num_dice)
        self.body = build_mlp(
            n_in=n_in,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.output = nn.Linear(
            n_hidden if n_layers > 0 else n_in, output_size(num_faces, num_dice)
        )
        # Make initial predictions closer to 0.
        with torch.no_grad():
            self.output.weight.data *= 0.01
            self.output.bias *= 0.01

    def forward(self, packed_input: torch.Tensor):
        return self.output(self.body(packed_input))


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)


_NUM_FACES = 6
_NUM_DICE = 1


def build_rebel_net2():
    return Net2(
        num_faces=_NUM_FACES,
        num_dice=_NUM_DICE,
        n_hidden=32,
        n_layers=2,
    )


def example_input_rebel_net2():
    return torch.randn(2, input_size(_NUM_FACES, _NUM_DICE))


MENAGERIE_ENTRIES = [
    ("ReBeL_Net2", "build_rebel_net2", "example_input_rebel_net2", 2020, "vendored-pytorch"),
]
