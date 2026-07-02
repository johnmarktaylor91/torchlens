# SOURCE: vendored from https://github.com/havakv/torchtuples @ master
# (torchtuples/practical.py: DenseVanillaBlock, MLPVanilla) + architecture
# usage from https://github.com/havakv/pycox @ master
# (examples/deephit.ipynb, pycox/models/deephit.py)
#
# DeepHit (Lee et al., AAAI 2018, "DeepHit: A Deep Learning Approach to
# Survival Analysis with Competing Risks") for maintenance/PdM survival
# analysis, via the well-resourced official `pycox` implementation
# (Kvamme, "Time-to-event prediction with neural networks and Cox
# regression", JMLR 2019). `pycox.models.DeepHitSingle`/`DeepHit` are
# `torchtuples.Model` training/inference wrappers (optimizer, loss,
# dataloaders) around a user-supplied `net`; they are not themselves a
# forward-pass nn.Module. The actual traceable network is the real
# `torchtuples.practical.MLPVanilla` class, which is exactly what pycox's
# canonical DeepHit example (examples/deephit.ipynb) constructs and passes
# into `DeepHitSingle(net, ...)`:
#   net = tt.practical.MLPVanilla(in_features, num_nodes=[32, 32],
#                                  out_features=labtrans.out_features,
#                                  batch_norm=True, dropout=0.1)
#
# Vendored verbatim (only the two real nn.Module classes; the loss/
# optimizer/dataloader/label-transform training machinery is dropped --
# it consumes labels/duration bins at training time, not part of the
# forward architecture). The only non-architectural change is inlining
# `torchtuples.tupletree.tuplefy(...).flatten()` as an equivalent local
# helper (`_flatten_num_nodes`) so this file has zero torchtuples/pycox
# import dependency; the flattening logic is identical for the scalar/list
# `num_nodes` case used here. No layer, block, or dataflow was changed.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


def _flatten_num_nodes(in_features, num_nodes):
    """Equivalent to torchtuples.tuplefy(in_features, num_nodes).flatten()
    for the list/tuple num_nodes case used by MLPVanilla."""
    out = [in_features]
    if isinstance(num_nodes, (list, tuple)):
        out.extend(num_nodes)
    else:
        out.append(num_nodes)
    return out


class DenseVanillaBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        batch_norm=True,
        dropout=0.0,
        activation=nn.ReLU,
        w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


class MLPVanilla(nn.Module):
    """Real `torchtuples.practical.MLPVanilla`: the standard feed-forward
    network pycox's DeepHit examples plug in as the survival-analysis
    backbone (a stack of DenseVanillaBlock hidden layers + a final Linear
    projecting to `out_features` PMF bins)."""

    def __init__(
        self,
        in_features,
        num_nodes,
        out_features,
        batch_norm=True,
        dropout=None,
        activation=nn.ReLU,
        output_activation=None,
        output_bias=True,
        w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
    ):
        super().__init__()
        num_nodes = _flatten_num_nodes(in_features, num_nodes)
        if not hasattr(dropout, "__iter__"):
            dropout = [dropout for _ in range(len(num_nodes) - 1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)


def build_deephit_mlp():
    # Matches pycox's canonical examples/deephit.ipynb config on the
    # metabric dataset: in_features=9, num_nodes=[32, 32],
    # num_durations=10 (labtrans.out_features), batch_norm=True, dropout=0.1.
    return MLPVanilla(
        in_features=9,
        num_nodes=[32, 32],
        out_features=10,
        batch_norm=True,
        dropout=0.1,
    )


def example_input_deephit_mlp():
    return torch.randn(4, 9)


MENAGERIE_ENTRIES = [
    (
        # NOTE: menagerie/classics/deephit.py already registers a "DeepHit"
        # entry, but it is an unsourced from-scratch reimplementation with no
        # provenance header (paper-description SLOP, not real source). This
        # entry is the genuine pycox/torchtuples-backed model (real
        # MLPVanilla class, real official-package config), named distinctly
        # to avoid a name clash with the legacy stub during integration.
        "DeepHit (pycox MLPVanilla)",
        build_deephit_mlp,
        example_input_deephit_mlp,
        2018,
        MENAGERIE_ZOO,
    ),
]
