# SOURCE: vendored from broadinstitute/CellBender @ c5f5d9f41a2926a0b46515ccc4a3383d52a9ffa9
# (cellbender/remove_background/vae/base.py, decoder.py, encoder.py -- EncodeZ only)
#
# CellBender remove-background: a VAE that separates ambient-RNA background from
# real single-cell gene expression (Fleming et al., Nat Methods 2023). The full
# generative model (`RemoveBackgroundPyroModel` in model.py) is a Pyro
# probabilistic program (pyro.sample/pyro.plate/pyro.param throughout its
# model()/guide() methods) -- Pyro is a real, non-base dependency that is
# architecturally load-bearing there (not an import-only fix), so that top-level
# class is out of scope for vendoring into a base-env torch trace.
#
# However CellBender's core neural-network components -- the fully-connected
# gene-expression encoder (`EncodeZ`) and decoder (`Decoder`), built on the
# repo's own `FullyConnectedNetwork`/`FullyConnectedLayer` primitives -- have
# ZERO pyro dependency and are transcribed verbatim below (only pyro-coupled
# sibling classes in encoder.py, e.g. `EncodeNonZLatents`/`CompositeEncoder`,
# were dropped; the transcribed classes themselves are untouched). `CellBenderVAECore`
# is a thin new wrapper (not present upstream) that chains the real encoder and
# real decoder in one forward pass, matching the encode(z)->decode(chi) path used
# by the full Pyro model's `model()`/`guide()`, so the pair can be traced as a
# single-input nn.Module.

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchlens as tl  # noqa: F401  (import parity with real usage context)

MENAGERIE_ZOO = "vendored-pytorch"


# --- vae/base.py ---
class FullyConnectedLayer(torch.nn.Module):
    """Neural network unit made of a fully connected linear layer, but
    customizable including shapes, activations, batch norm, layer norm, and
    dropout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation=torch.nn.ReLU,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        modules: list = []
        if dropout_rate is not None:
            modules.append(torch.nn.Dropout(p=dropout_rate))
        modules.append(torch.nn.Linear(in_features=input_dim, out_features=output_dim))
        if use_batch_norm:
            modules.append(torch.nn.BatchNorm1d(num_features=output_dim, momentum=0.01, eps=0.001))
        if use_layer_norm:
            modules.append(
                torch.nn.LayerNorm(normalized_shape=output_dim, elementwise_affine=False)
            )
        if activation is not None:
            modules.append(activation() if isinstance(activation, type) else activation)

        self.layer = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FullyConnectedNetwork(torch.nn.Module):
    """Neural network made of fully connected linear layers, FullyConnectedLayer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
        output_activation: Optional[torch.nn.Module] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        norm_output: bool = False,
        dropout_rate: Optional[float] = None,
        dropout_input: bool = False,
    ):
        super().__init__()

        if use_layer_norm and use_batch_norm:
            raise UserWarning(
                "You are trying to use both batch norm and layer norm. That's probably too much norm."
            )

        dim_ins_and_outs = zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        n_layers = 1 + len(hidden_dims)
        layers = [
            FullyConnectedLayer(
                input_dim=i,
                output_dim=j,
                activation=hidden_activation if (layer < n_layers - 1) else output_activation,
                use_batch_norm=use_batch_norm if ((layer < n_layers - 1) or norm_output) else False,
                use_layer_norm=use_layer_norm if ((layer < n_layers - 1) or norm_output) else False,
                dropout_rate=None if ((layer == 0) and not dropout_input) else dropout_rate,
            )
            for layer, (i, j) in enumerate(dim_ins_and_outs)
        ]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Any:
        return self.network(x)


# --- vae/decoder.py ---
class Decoder(FullyConnectedNetwork):
    """Decoder module transforms latent representation into gene expression (output on a simplex via softmax)."""

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)
        self.input_dim = input_dim
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.network(z))


# --- vae/encoder.py (pyro-free EncodeZ + its transform_input helper only) ---
def transform_input(x: torch.Tensor, transform: Optional[str], eps: float = 1e-5) -> torch.Tensor:
    """Transform input to encoder."""
    if transform is None:
        return x
    elif transform == "log":
        x = x.log1p()
        return x
    elif transform == "normalize":
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        return x
    elif transform == "normalize_log":
        x = x.log1p()
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        return x
    elif transform == "log_normalize":
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x.log1p()
        return x
    else:
        raise NotImplementedError(
            "Specified an input transform that is not supported.  Choose from 'log' or 'normalize'."
        )


class EncodeZ(FullyConnectedNetwork):
    """Encoder module transforms gene expression into latent representation (loc, scale)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        input_transform: Optional[str] = None,
        **kwargs,
    ):
        assert len(hidden_dims) > 0, "EncodeZ needs to have at least one hidden layer"
        super(EncodeZ, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            hidden_activation=nn.Softplus(),
            output_activation=nn.Softplus(),
            norm_output=True,
            **kwargs,
        )
        self.transform = input_transform
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loc_out = nn.Linear(hidden_dims[-1], output_dim)
        self.sig_out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Transform input.
        x = x.reshape(-1, self.input_dim)
        x_ = transform_input(x, self.transform)

        # Obtain last hidden layer.
        hidden = self.network(x_)

        # Compute the outputs: loc is any real number, scale must be positive.
        loc = self.loc_out(hidden)
        scale = torch.exp(self.sig_out(hidden))

        return {"loc": loc.squeeze(), "scale": scale.squeeze()}


class CellBenderVAECore(nn.Module):
    """Thin wrapper (not upstream) chaining the real EncodeZ -> Decoder pair for single-input tracing."""

    def __init__(self, n_genes: int, z_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.encoder = EncodeZ(
            input_dim=n_genes,
            hidden_dims=hidden_dims,
            output_dim=z_dim,
            input_transform="normalize",
        )
        self.decoder = Decoder(
            input_dim=z_dim, hidden_dims=list(reversed(hidden_dims)), output_dim=n_genes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        z = out["loc"]
        return self.decoder(z)


def build_cellbender():
    model = CellBenderVAECore(n_genes=50, z_dim=8, hidden_dims=[32, 16])
    model.eval()
    return model


def example_input_cellbender():
    return torch.rand(4, 50)


MENAGERIE_ENTRIES = [
    ("CellBender", "build_cellbender", "example_input_cellbender", 2023, "vendored-pytorch"),
]
