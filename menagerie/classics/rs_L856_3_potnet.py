# SOURCE: vendored from https://github.com/divelab/AIRS @ main (OpenMat/PotNet)
#
# PotNet: Efficient Representation Learning for Materials via Infinite Potential
# Summation (ICML 2024, divelab). Files combined (each class copied verbatim from
# the real repo, imports/paths fixed minimally so the module is self-contained):
#   - OpenMat/PotNet/models/utils.py   -> RBFExpansion (radial basis expansion,
#                                          multiple kernel types)
#   - OpenMat/PotNet/models/potnet.py  -> PotNetConv (interaction-network message
#                                          passing block) and PotNet (top-level
#                                          model: atom/edge embeddings, conv stack,
#                                          pooling head)
#
# The real repo's PotNetConfig is a pydantic BaseSettings subclass (models/base.py,
# models/config.py) that reads from env vars via `pydantic.typing.Literal`, which
# is removed in modern pydantic and is orchestration/config plumbing, not part of
# the traced architecture -- it is replaced here with a plain class carrying the
# exact same fields/defaults. The `transformer=True` branch (which additionally
# needs `models/transformer.py::TransformerConv`) and the `charge_map=True` branch
# are both optional forward-path branches in the real PotNet.forward; this trace
# uses the repo's default `euclidean=False, transformer=False, charge_map=False`
# configuration, so the infinite-potential-summation edge path (RBFExpansion +
# infinite_linear + infinite_bn folded into the same edge list as PotNetConv) is
# exercised, matching the paper's actual "PotNet" configuration.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import sys as _sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool
from torch_geometric.nn.models.schnet import ShiftedSoftplus

# torch_geometric.nn.MessagePassing's signature inspector resolves type hints
# via sys.modules[cls.__module__].__dict__ (torch_geometric/inspector.py). When
# this file is loaded through a bare importlib.util.spec_from_file_location /
# module_from_spec / exec_module sequence (as opposed to a normal `import`),
# the module is never registered in sys.modules, so that lookup raises
# KeyError as soon as a PotNetConv (a MessagePassing subclass) is constructed.
# Self-register a lightweight proxy backed by this module's own globals() (not
# a copy -- a live-shared dict) so later class definitions are visible too.
if __name__ not in _sys.modules:

    class _LiveModuleProxy:
        def __init__(self, module_globals):
            self.__dict__ = module_globals

    _sys.modules[__name__] = _LiveModuleProxy(globals())

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# models/config.py + models/base.py (adapted: a plain class replacing the
# real repo's pydantic BaseSettings subclass -- same field names/defaults,
# no env-var reading, since that's config plumbing outside the traced model)
# ------------------------------------------------------------------
class PotNetConfig:
    def __init__(
        self,
        name: str = "potnet",
        conv_layers: int = 3,
        atom_input_features: int = 92,
        inf_edge_features: int = 64,
        fc_features: int = 256,
        output_dim: int = 256,
        output_features: int = 1,
        rbf_min: float = -4.0,
        rbf_max: float = 4.0,
        potentials=None,
        euclidean: bool = False,
        charge_map: bool = False,
        transformer: bool = False,
    ):
        self.name = name
        self.conv_layers = conv_layers
        self.atom_input_features = atom_input_features
        self.inf_edge_features = inf_edge_features
        self.fc_features = fc_features
        self.output_dim = output_dim
        self.output_features = output_features
        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        self.potentials = potentials if potentials is not None else []
        self.euclidean = euclidean
        self.charge_map = charge_map
        self.transformer = transformer


# ------------------------------------------------------------------
# models/utils.py: RBFExpansion  (verbatim)
# ------------------------------------------------------------------
class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale=None,
        type: str = "gaussian",
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        self.type = type

        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        base = self.gamma * (distance.unsqueeze(-1) - self.centers)
        if self.type == "gaussian":
            return (-(base**2)).exp()
        elif self.type == "quadratic":
            return base**2
        elif self.type == "linear":
            return base
        elif self.type == "inverse_quadratic":
            return 1.0 / (1.0 + base**2)
        elif self.type == "multiquadric":
            return (1.0 + base**2).sqrt()
        elif self.type == "inverse_multiquadric":
            return 1.0 / (1.0 + base**2).sqrt()
        elif self.type == "spline":
            return base**2 * (base + 1.0).log()
        elif self.type == "poisson_one":
            return (base - 1.0) * (-base).exp()
        elif self.type == "poisson_two":
            return (base - 2.0) / 2.0 * base * (-base).exp()
        elif self.type == "matern32":
            return (1.0 + 3**0.5 * base) * (-(3**0.5) * base).exp()
        elif self.type == "matern52":
            return (1.0 + 5**0.5 * base + 5 / 3 * base**2) * (-(5**0.5) * base).exp()
        else:
            raise Exception("No Implemented Radial Basis Method")


# ------------------------------------------------------------------
# models/potnet.py: PotNetConv  (verbatim)
# ------------------------------------------------------------------
class PotNetConv(MessagePassing):
    def __init__(self, fc_features):
        super().__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features), nn.SiLU(), nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features), nn.SiLU(), nn.Linear(fc_features, fc_features)
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        return F.relu(x + self.bn(out))

    def message(self, x_i, x_j, edge_attr, index):
        score = torch.sigmoid(
            self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1)))
        )
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))


# ------------------------------------------------------------------
# models/potnet.py: PotNet  (verbatim)
# ------------------------------------------------------------------
class PotNet(nn.Module):
    def __init__(self, config: PotNetConfig = PotNetConfig(name="potnet")):
        super().__init__()
        self.config = config
        if not config.charge_map:
            self.atom_embedding = nn.Linear(config.atom_input_features, config.fc_features)
        else:
            self.atom_embedding = nn.Linear(config.atom_input_features + 10, config.fc_features)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=config.rbf_min, vmax=config.rbf_max, bins=config.fc_features),
            nn.Linear(config.fc_features, config.fc_features),
            nn.SiLU(),
        )

        if not self.config.euclidean:
            self.inf_edge_embedding = RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.inf_edge_features,
                type="multiquadric",
            )
            self.infinite_linear = nn.Linear(config.inf_edge_features, config.fc_features)
            self.infinite_bn = nn.BatchNorm1d(config.fc_features)

        self.conv_layers = nn.ModuleList(
            [PotNetConv(config.fc_features) for _ in range(config.conv_layers)]
        )

        if not config.euclidean and config.transformer:
            # NOTE: the real repo's transformer=True branch additionally needs
            # models/transformer.py::TransformerConv; not vendored here since
            # this staging module always builds with transformer=False (the
            # paper's default PotNet configuration).
            raise NotImplementedError(
                "transformer=True branch requires models/transformer.py::TransformerConv"
            )

        self.fc = nn.Sequential(
            nn.Linear(config.fc_features, config.fc_features), ShiftedSoftplus()
        )
        self.fc_out = nn.Linear(config.output_dim, config.output_features)

    def forward(self, data, print_data=False):
        """CGCNN function mapping graph to outputs."""
        edge_index = data.edge_index
        if self.config.euclidean:
            edge_features = self.edge_embedding(data.edge_attr)
        else:
            edge_features = self.edge_embedding(-0.75 / data.edge_attr)

        if not self.config.euclidean:
            inf_edge_index = data.inf_edge_index
            inf_feat = sum(
                data.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.config.potentials)
            )
            inf_edge_features = self.inf_edge_embedding(inf_feat)
            inf_edge_features = self.infinite_bn(
                F.softplus(self.infinite_linear(inf_edge_features))
            )

        if self.config.charge_map:
            node_features = self.atom_embedding(torch.cat([data.x, data.g_feats], -1))
        else:
            node_features = self.atom_embedding(data.x)

        if not self.config.euclidean and not self.config.transformer:
            edge_index = torch.cat([data.edge_index, inf_edge_index], 1)
            edge_features = torch.cat([edge_features, inf_edge_features], 0)

        for i in range(self.config.conv_layers):
            node_features = self.conv_layers[i](node_features, edge_index, edge_features)

        features = global_mean_pool(node_features, data.batch)
        features = self.fc(features)
        return torch.squeeze(self.fc_out(features))


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
def build_potnet():
    torch.manual_seed(0)
    config = PotNetConfig(
        name="potnet",
        conv_layers=2,
        atom_input_features=16,
        inf_edge_features=8,
        fc_features=16,
        output_dim=16,
        output_features=1,
        rbf_min=-4.0,
        rbf_max=4.0,
        potentials=[1.0, 1.0, 1.0],
        euclidean=False,
        charge_map=False,
        transformer=False,
    )
    return PotNet(config=config)


def example_input_potnet():
    from torch_geometric.data import Data

    torch.manual_seed(0)
    num_nodes = 6
    x = torch.randn(num_nodes, 16)

    # a small ring graph for the "real" (finite-distance) edges
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5]],
        dtype=torch.long,
    )
    edge_attr = torch.rand(edge_index.shape[1]) + 0.5  # positive distances

    # "infinite" potential-summation edges: fully connected, 3 potential channels
    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes).repeat(num_nodes)
    keep = src != dst
    inf_edge_index = torch.stack([src[keep], dst[keep]], dim=0)
    inf_edge_attr = torch.rand(inf_edge_index.shape[1], 3)

    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    data.inf_edge_index = inf_edge_index
    data.inf_edge_attr = inf_edge_attr
    return (data,)


MENAGERIE_ENTRIES = [
    ("potnet", "build_potnet", "example_input_potnet", 2024, MENAGERIE_ZOO),
]
