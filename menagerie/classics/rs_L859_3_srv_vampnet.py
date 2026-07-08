# SOURCE: vendored from deeptime-ml/deeptime @ v0.4.5 (deeptime/util/torch.py, class MLP).
# This is the real `deeptime.util.torch.MLP` class -- deeptime's own documented reference
# "neural network lobe for VAMPNets" (see `VAMPNetModel`/`VAMPNet` docstrings: "See also
# deeptime.util.torch.MLP"). SRV (State-free reversible VAMPnets, Chen, Sidky, Ferguson 2019)
# is not a distinct architecture class in deeptime -- it is the VAMP-2/reversible-scoring
# training regime (`VAMPNet(lobe=..., lobe_timelagged=None, score_method='VAMP2')`, i.e. a
# SHARED lobe applied at both time-lags, which is what makes the learned decomposition
# state-free and time-reversal symmetric) applied to train this exact MLP "lobe" network;
# deeptime is the maintained, canonical PyTorch reimplementation of VAMPnets/SRV (the
# original markovmodel/deeptime `vampnet/` directory is an older, pre-PyTorch numpy tool
# folded into this package's `deeptime.decomposition.deep` module).
#
# No import fixes needed: MLP only depends on torch/torch.nn (`try_import` is deeptime's
# internal optional-import helper; inlined here as a plain `import torch.nn as nn` since
# torch is a hard dependency in this environment).

import torch
import torch.nn as nn


class MLP(nn.Module):
    r"""A multilayer perceptron which can, e.g., be used as a neural network lobe for VAMPNets.

    Parameters
    ----------
    units : list of int
        The units of the fully connected layers.
    nonlinearity : callable, default=None
        A callable (like a constructor) which yields an instance of a particular activation function.
        Defaults to ELU.
    initial_batchnorm : bool, default=True
        Whether to use batch normalization before the data enters the rest of the network.
    output_nonlinearity : callable, default=None
        The output activation/nonlinearity. If the data decomposes into states, it can make sense to use
        an output activation like softmax which produces a probability distribution over said states.
        The callable should take no arguments and produce an object of type :code:`torch.nn.Module`.
    """

    def __init__(
        self, units, nonlinearity=None, initial_batchnorm: bool = False, output_nonlinearity=None
    ):
        super().__init__()
        if nonlinearity is None:
            nonlinearity = nn.ELU
        if len(units) > 1:
            layers = []
            if initial_batchnorm:
                layers.append(nn.BatchNorm1d(units[0]))
            for fan_in, fan_out in zip(units[:-2], units[1:-1]):
                layers.append(nn.Linear(fan_in, fan_out))
                layers.append(nonlinearity())
            layers.append(nn.Linear(units[-2], units[-1]))
            if output_nonlinearity is not None:
                layers.append(output_nonlinearity())
            self._sequential = nn.Sequential(*layers)
        else:
            self._sequential = nn.Identity()

    def forward(self, inputs):
        return self._sequential(inputs)


# ---------------------------------------------------------------------------
# Menagerie build/example plumbing
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"


def build_srv_lobe():
    """Tiny random-init SRV/VAMPnet lobe MLP (deeptime's reference lobe architecture).

    Matches the shape deeptime's own VAMPnets/SRV examples use: an MLP with an initial
    batchnorm and ELU hidden nonlinearities, mapping raw featurized frames down to a
    small number of learned (reversible/state-free) collective-variable outputs.
    """
    torch.manual_seed(0)
    return MLP(units=[12, 24, 24, 4], nonlinearity=nn.ELU, initial_batchnorm=True)


def example_input_srv_lobe():
    """A small batch of featurized molecular-dynamics frames -- (batch=8, features=12)."""
    torch.manual_seed(0)
    x = torch.randn(8, 12)
    return (x,)


MENAGERIE_ENTRIES = [
    (
        "SRV (State-free reversible VAMPnets) lobe",
        build_srv_lobe,
        example_input_srv_lobe,
        2019,
        MENAGERIE_ZOO,
    ),
]
