# FAITHFUL PORT of mbchang/dynamics @ master (original framework: Torch7/Lua)
#   src/lua/npe.lua      (init_network: encoder + shared pairwise layers + decoder)
#   src/lua/modules.lua  (init_object_encoder, init_object_decoder_with_identity)
# "A Compositional Object-Based Approach to Learning Physical Dynamics"
# (Chang, Ullman, Torralba, Tenenbaum, ICLR 2017) -- the Neural Physics Engine (NPE).
# The original code is Torch7/Lua (requires `nn`, `rnn`, `nngraph`, luarocks) and cannot
# run in a base torch env, so the architecture is transcribed faithfully from
# src/lua/npe.lua + src/lua/modules.lua into torch:
#   - `init_object_encoder`: a shared-weight two-branch MLP that separately embeds the
#     focus ("this") object state and each context-object state, then concatenates them
#     (JoinTable) into a single pairwise embedding.
#   - the pairwise embedding is passed through `params.layers` shared-weight
#     Linear+ReLU blocks (nn.Sequencer over the shared `layer`, applied once per
#     context object -- same weights, not shared state across steps).
#   - the per-context-object encoded vectors are summed (CAddTable) -- this is the
#     compositional pairwise-interaction aggregation central to NPE.
#   - `init_object_decoder_with_identity`: the summed interaction vector is
#     concatenated with the focus object's own raw past state (its "identity") and
#     decoded by a small MLP into the predicted future object state.
"""FAITHFUL PORT of the Neural Physics Engine (Chang et al., ICLR 2017)."""

from typing import List, Tuple

import torch
import torch.nn as nn


class ObjectEncoder(nn.Module):
    """Port of modules.lua::init_object_encoder.

    Separately embeds the focus ("this") object state and one context object's
    state with two independent Linear+ReLU branches, then concatenates the two
    half-dimension embeddings into a single `rnn_inp_dim` vector.
    """

    def __init__(self, input_dim: int, rnn_inp_dim: int, bias: bool = True) -> None:
        super().__init__()
        assert rnn_inp_dim % 2 == 0
        self.this_branch = nn.Linear(input_dim, rnn_inp_dim // 2, bias=bias)
        self.context_branch = nn.Linear(input_dim, rnn_inp_dim // 2, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, this_state: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        this_out = self.relu(self.this_branch(this_state))
        context_out = self.relu(self.context_branch(context_state))
        return torch.cat([this_out, context_out], dim=-1)


class ObjectDecoderWithIdentity(nn.Module):
    """Port of modules.lua::init_object_decoder_with_identity.

    Concatenates the aggregated pairwise-interaction vector with the focus
    object's own raw past state ("identity"), then applies a stack of
    Linear+ReLU layers (final layer has no activation) to predict the future
    object state.
    """

    def __init__(
        self,
        rnn_hid_dim: int,
        num_layers: int,
        identity_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        decoder_in_dim = identity_dim + rnn_hid_dim

        if num_layers <= 1:
            self.decoder_net: nn.Module = nn.Linear(decoder_in_dim, out_dim)
        else:
            layers: List[nn.Module] = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(decoder_in_dim, rnn_hid_dim))
                    layers.append(nn.ReLU())
                elif i == num_layers - 1:
                    layers.append(nn.Linear(rnn_hid_dim, out_dim))
                else:
                    layers.append(nn.Linear(rnn_hid_dim, rnn_hid_dim))
                    layers.append(nn.ReLU())
            self.decoder_net = nn.Sequential(*layers)

    def forward(self, pairwise_summed: torch.Tensor, this_past: torch.Tensor) -> torch.Tensor:
        decoder_in = torch.cat([pairwise_summed, this_past], dim=-1)
        return self.decoder_net(decoder_in)


class NeuralPhysicsEngine(nn.Module):
    """Port of npe.lua::init_network (params.model == 'npe').

    For a focus object and a variable number of context objects, encodes each
    (this, context) pair with a shared `ObjectEncoder`, passes each pairwise
    embedding through `layers` shared-weight Linear+ReLU transforms (the
    `layer:clone()` stack, cloned per depth but NOT per context object --
    weights are shared across all context objects at a given depth, matching
    the Lua `nn.Sequencer(step)` applied over the table of pairwise inputs),
    sums the resulting per-context vectors (CAddTable), and decodes the sum
    together with the focus object's raw past state into a predicted future
    state.
    """

    def __init__(
        self,
        object_dim: int = 22,
        num_past: int = 2,
        num_future: int = 1,
        rnn_dim: int = 24,
        layers: int = 5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.object_dim = object_dim
        self.num_past = num_past
        self.num_future = num_future
        self.rnn_dim = rnn_dim
        self.n_layers = layers

        input_dim = object_dim * num_past
        out_dim = object_dim * num_future

        self.encoder = ObjectEncoder(input_dim, rnn_dim, bias=bias)

        self.shared_layers = nn.ModuleList(
            [nn.Linear(rnn_dim, rnn_dim, bias=bias) for _ in range(layers)]
        )
        self.relu = nn.ReLU()

        self.decoder = ObjectDecoderWithIdentity(
            rnn_hid_dim=rnn_dim,
            num_layers=layers,
            identity_dim=input_dim,
            out_dim=out_dim,
        )

    def _step(self, this_state: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        h = self.encoder(this_state, context_state)
        for lin in self.shared_layers:
            h = self.relu(lin(h))
        return h

    def forward(self, this_past: torch.Tensor, context_past: torch.Tensor) -> torch.Tensor:
        """
        this_past: (batch, object_dim * num_past) -- focus object's flattened past state
        context_past: (batch, n_context, object_dim * num_past) -- each context object's
            flattened past state
        returns: (batch, object_dim * num_future) -- predicted focus object future state
        """
        n_context = context_past.shape[1]
        pairwise_sum = None
        for c in range(n_context):
            step_out = self._step(this_past, context_past[:, c, :])
            pairwise_sum = step_out if pairwise_sum is None else pairwise_sum + step_out
        if pairwise_sum is None:
            pairwise_sum = this_past.new_zeros(this_past.shape[0], self.rnn_dim)
        return self.decoder(pairwise_sum, this_past)


_OBJECT_DIM = 22
_NUM_PAST = 2
_NUM_FUTURE = 1
_RNN_DIM = 24
_LAYERS = 3  # reduced from paper default of 5 for a small trace-validation instance
_N_CONTEXT = 3


def build_npe() -> nn.Module:
    model = NeuralPhysicsEngine(
        object_dim=_OBJECT_DIM,
        num_past=_NUM_PAST,
        num_future=_NUM_FUTURE,
        rnn_dim=_RNN_DIM,
        layers=_LAYERS,
        bias=True,
    )
    model.eval()
    return model


def example_input_npe() -> Tuple[torch.Tensor, torch.Tensor]:
    batch = 2
    this_past = torch.randn(batch, _OBJECT_DIM * _NUM_PAST)
    context_past = torch.randn(batch, _N_CONTEXT, _OBJECT_DIM * _NUM_PAST)
    return (this_past, context_past)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    (
        "Neural Physics Engine",
        build_npe,
        example_input_npe,
        2017,
        MENAGERIE_ZOO,
    ),
]
