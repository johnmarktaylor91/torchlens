# SOURCE: vendored from gao-lab/Cell_BLAST @ master
#   (Cell_BLAST/rebuild.py: Linear, MLP; Cell_BLAST/latent.py: Regularizer, Latent, Gau)
#
# Cell BLAST / DIRECTi (Cao et al., Nature Communications 2020): a semi-supervised
# deep generative model for scRNA-seq cell embedding and batch-effect removal. The
# top-level `DIRECTi` class (directi.py) is a training orchestrator -- it composes a
# `latent_module` (encoder) and `prob_module` (generative decoder) but exposes no single
# `forward()` of its own; training/inference route through `.fit()` / `.inference()`,
# which call custom per-submodule optimizers (the module's own `rebuild.RMSprop`), not a
# plain forward pass. The actual traceable neural-net component -- the cell embedding
# encoder used by default -- is `Gau`, the Gaussian latent module: an MLP
# (Linear->BatchNorm1d->LeakyReLU stack, via the module's own from-scratch `rebuild.Linear`
# / `rebuild.MLP`, deliberately re-implemented in torch to reproduce original-TensorFlow
# init/behavior) followed by a linear projection to the latent cell-embedding space, plus
# an adversarial `Regularizer` (an MLP + sigmoid discriminator used during GAN-style
# latent regularization). Copied verbatim from the real repo's Linear/MLP (rebuild.py)
# and Regularizer/Latent/Gau (latent.py) classes, aside from collapsing the two-file
# module split and dropping fine-tune-only state (`save_origin_state`/`deviation_loss`)
# and training-loop-only methods (`fetch_grad`, `d_loss`, `g_loss`, `init_loss_record`,
# `parameters_reg`, `parameters_fit`) that are not part of the forward architecture.
import typing

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features,
        out_features: int,
        bias: bool = True,
        init_std: float = 0.01,
        trunc: bool = True,
    ) -> "Linear":
        if not isinstance(in_features, list) and not isinstance(in_features, tuple):
            in_features = [in_features]

        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.ParameterList()
        for _in_features in in_features:
            self.weight.append(nn.Parameter(torch.Tensor(out_features, _in_features)))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.init_std = init_std
        self.trunc = trunc
        self.reset_parameters()

    def reset_parameters(self):
        if self.trunc:
            for _weight in self.weight:
                nn.init.trunc_normal_(
                    _weight,
                    std=self.init_std,
                    a=-2 * self.init_std,
                    b=2 * self.init_std,
                )
        else:
            for _weight in self.weight:
                nn.init.normal_(_weight, std=self.init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if not isinstance(input, list) and not isinstance(input, tuple):
            input = [input]
        for i, (_input, _weight) in enumerate(zip(input, self.weight)):
            if i:
                result = result + F.linear(_input, _weight, None)  # noqa: F821
            else:
                result = F.linear(_input, _weight, self.bias)
        return result

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class MLP(nn.Module):
    def __init__(
        self,
        i_dim,
        o_dim,
        dropout,
        bias: bool = True,
        batch_normalization: bool = False,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.i_dim = i_dim
        self.o_dim = o_dim
        self.dropout = dropout
        self.bias = bias
        self.batch_normalization = batch_normalization
        self.activation = activation

        self.hiddens = nn.ModuleList()
        module_seq = []

        for _i_dim, _o_dim, _dropout in zip(i_dim, o_dim, dropout):
            if _dropout > 0:
                module_seq.append(nn.Dropout(dropout))

            hidden = Linear(_i_dim, _o_dim, bias=bias)
            self.hiddens.append(hidden)
            module_seq.append(hidden)

            if batch_normalization:
                module_seq.append(nn.BatchNorm1d(_o_dim, eps=0.001, momentum=0.01))

            if activation:
                module_seq.append(nn.LeakyReLU(0.2))

        self.layer_seq = nn.Sequential(*module_seq)

        self.first_layer_trainable = True

    def forward(self, x: torch.Tensor):
        return self.layer_seq(x)

    @property
    def first_layer_trainable(self):
        return self._first_layer_trainable

    @first_layer_trainable.setter
    def first_layer_trainable(self, flag: bool):
        self._first_layer_trainable = flag
        if len(self.hiddens) > 0:
            self.hiddens[0].requires_grad_(flag)


class Regularizer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        name: str = "Reg",
        _class: str = "Regularizer",
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.name = name
        self._class = _class

        i_dim = [latent_dim] + [h_dim] * (depth - 1) if depth > 0 else []
        o_dim = [h_dim] * depth
        dropout = [dropout] * depth
        if depth > 0:
            dropout[0] = 0.0
        self.mlp = MLP(i_dim, o_dim, dropout)
        self.output = Linear(h_dim, 1) if depth > 0 else Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.output(self.mlp(x)))


class Latent(nn.Module):
    r"""
    Abstract base class for latent variable modules.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.0,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "Latent",
        _class: str = "Latent",
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.fine_tune = fine_tune
        self.deviation_reg = deviation_reg
        self.name = name
        self._class = _class
        self.record_prefix = "discriminator"


class Gau(Latent):
    r"""
    Gaussian latent module. The Gaussian latent variable is used as cell embedding
    (the default DIRECTi encoder in Cell BLAST).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.001,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "Gau",
        _class: str = "Gau",
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim,
            latent_dim,
            h_dim,
            depth,
            dropout,
            lambda_reg,
            fine_tune,
            deviation_reg,
            name,
            _class,
            **kwargs,
        )

        self.gau_reg = Regularizer(latent_dim, h_dim, depth, dropout, name="gau")
        self.gaup_sampler = D.Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))

        i_dim = [input_dim] + [h_dim] * (depth - 1) if depth > 0 else []
        o_dim = [h_dim] * depth
        dropout = [dropout] * depth
        self.mlp = MLP(i_dim, o_dim, dropout, bias=False, batch_normalization=True)
        self.gau = Linear(h_dim, latent_dim) if depth > 0 else Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gau = self.gau(self.mlp(x))
        return gau, gau

    def fetch_latent(self, x: torch.Tensor) -> torch.Tensor:
        gau = self.gau(self.mlp(x))
        return gau


def build_cellblast_gau():
    # Real usage: input_dim = number of genes (typically ~2000 HVGs), latent_dim ~= 10.
    # Shrink both for tracing speed while keeping the real architecture (2-layer MLP
    # encoder + adversarial Regularizer head) untouched.
    return Gau(input_dim=64, latent_dim=10, h_dim=32, depth=2, dropout=0.0)


def example_input_cellblast_gau():
    # Real input: per-cell log-library-size-normalized expression vector, batched.
    # batch_size > 1 required: MLP uses BatchNorm1d.
    return torch.randn(4, 64)


MENAGERIE_ENTRIES = [
    (
        "Cell BLAST DIRECTi (Gau)",
        build_cellblast_gau,
        example_input_cellblast_gau,
        2020,
        "SOURCE_AVAILABLE",
    ),
]
