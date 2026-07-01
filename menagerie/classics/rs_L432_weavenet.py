# SOURCE: vendored from deepchem/deepchem @ master
# https://raw.githubusercontent.com/deepchem/deepchem/master/deepchem/models/torch_models/weavemodel_pytorch.py
# https://raw.githubusercontent.com/deepchem/deepchem/master/deepchem/models/torch_models/layers.py (WeaveLayer, WeaveGather)
# https://raw.githubusercontent.com/deepchem/deepchem/master/deepchem/utils/pytorch_utils.py (get_activation)
#
# Kearnes, McCloskey, Berndl, Pande, Riley 2016 "Molecular graph convolutions:
# moving beyond fingerprints" -- the "Weave" graph-convolution architecture for
# molecular property prediction (deepchem's `dc.models.torch_models.WeaveModel`).
# Weave convolutions model bond (pair) features explicitly alongside atom
# features: each `WeaveLayer` performs 4 typed transforms (atom->atom "AA",
# pair->atom "PA", atom->pair "AP", pair->pair "PP", each a Linear+BatchNorm1d+
# activation), fuses the two atom-side branches into new atom features and the
# two pair-side branches into new pair features, and this is stacked `n_weave`
# times. A final graph-level `WeaveGather` layer (Gaussian-histogram-expand +
# per-molecule segment-sum over `atom_split`) produces a fixed-size molecule
# embedding, followed by dense layers to task outputs (classification/
# regression). This is the real deepchem PyTorch model class (`Weave`, the
# `nn.Module` used inside `WeaveModel(TorchModel)`) -- not a reimplementation.
#
# deepchem itself is not an installed base lib (pip package `deepchem`, not in
# the base env), so instead of installing the whole package this vendors only
# the actual architecture-defining pieces the `Weave` nn.Module depends on:
# `WeaveLayer`, `WeaveGather` (from `deepchem/models/torch_models/layers.py`)
# and `get_activation` (from `deepchem/utils/pytorch_utils.py`). All three are
# copied verbatim (no architectural changes). The surrounding deepchem-specific
# machinery (`TorchModel` training wrapper, `Dataset`, `WeaveFeaturizer`,
# losses, metrics) is deliberately NOT vendored -- it is data/training
# plumbing, not part of the `Weave` architecture itself. `Weave.__init__`/
# `forward` are also copied verbatim from `weavemodel_pytorch.py`.
#
# Example input below constructs a tiny batch of 2 "molecules" (4 atoms total,
# mirroring the real docstring's `smiles = ["CCC", "C"]` example) with random
# atom/pair features, matching the exact `[atom_feat, pair_feat, pair_split,
# atom_split, atom_to_pair]` input contract documented in the real
# `Weave.forward`/`WeaveLayer.forward`/`WeaveGather.forward` docstrings.

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from collections.abc import Sequence as SequenceCollection

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init as initializers


def get_activation(fn: Union[Callable, str]):
    """Get a PyTorch activation function, specified either directly or as a string.

    Verbatim from deepchem.utils.pytorch_utils.get_activation.
    """
    if isinstance(fn, str):
        return getattr(torch.nn.functional, fn)
    return fn


class WeaveLayer(nn.Module):
    """Core Weave convolution (verbatim from deepchem.models.torch_models.layers.WeaveLayer).

    Expects 4 inputs `[atom_features, pair_features, pair_split, atom_to_pair]`.
    """

    def __init__(
        self,
        n_atom_input_feat: int = 75,
        n_pair_input_feat: int = 14,
        n_atom_output_feat: int = 50,
        n_pair_output_feat: int = 50,
        n_hidden_AA: int = 50,
        n_hidden_PA: int = 50,
        n_hidden_AP: int = 50,
        n_hidden_PP: int = 50,
        update_pair: bool = True,
        init_: str = "xavier_uniform_",
        activation: str = "relu",
        batch_normalize: bool = True,
        **kwargs,
    ):
        super(WeaveLayer, self).__init__(**kwargs)
        self.init: str = init_
        self.activation: str = activation
        self.activation_fn: torch.nn.Module = get_activation(activation)
        self.update_pair: bool = update_pair
        self.n_hidden_AA: int = n_hidden_AA
        self.n_hidden_PA: int = n_hidden_PA
        self.n_hidden_AP: int = n_hidden_AP
        self.n_hidden_PP: int = n_hidden_PP
        self.n_hidden_A: int = n_hidden_AA + n_hidden_PA
        self.n_hidden_P: int = n_hidden_AP + n_hidden_PP
        self.batch_normalize: bool = batch_normalize

        self.n_atom_input_feat: int = n_atom_input_feat
        self.n_pair_input_feat: int = n_pair_input_feat
        self.n_atom_output_feat: int = n_atom_output_feat
        self.n_pair_output_feat: int = n_pair_output_feat

        init = getattr(initializers, self.init)
        self.W_AA: torch.Tensor = init(torch.empty(self.n_atom_input_feat, self.n_hidden_AA))
        self.b_AA: torch.Tensor = torch.zeros((self.n_hidden_AA,))
        self.AA_bn: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=self.n_hidden_AA,
            eps=1e-3,
            momentum=0.99,
            affine=True,
            track_running_stats=True,
        )

        self.W_PA: torch.Tensor = init(torch.empty(self.n_pair_input_feat, self.n_hidden_PA))
        self.b_PA: torch.Tensor = torch.zeros((self.n_hidden_PA,))
        self.PA_bn: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=self.n_hidden_PA,
            eps=1e-3,
            momentum=0.99,
            affine=True,
            track_running_stats=True,
        )

        self.W_A: torch.Tensor = init(torch.empty(self.n_hidden_A, self.n_atom_output_feat))
        self.b_A: torch.Tensor = torch.zeros((self.n_atom_output_feat,))
        self.A_bn: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=self.n_atom_output_feat,
            eps=1e-3,
            momentum=0.99,
            affine=True,
            track_running_stats=True,
        )

        if self.update_pair:
            self.W_AP: torch.Tensor = init(
                torch.empty(self.n_atom_input_feat * 2, self.n_hidden_AP)
            )
            self.b_AP: torch.Tensor = torch.zeros((self.n_hidden_AP,))
            self.AP_bn: nn.BatchNorm1d = nn.BatchNorm1d(
                num_features=self.n_hidden_AP,
                eps=1e-3,
                momentum=0.99,
                affine=True,
                track_running_stats=True,
            )
            self.W_PP: torch.Tensor = init(torch.empty(self.n_pair_input_feat, self.n_hidden_PP))
            self.b_PP: torch.Tensor = torch.zeros((self.n_hidden_PP,))
            self.PP_bn: nn.BatchNorm1d = nn.BatchNorm1d(
                num_features=self.n_hidden_PP,
                eps=1e-3,
                momentum=0.99,
                affine=True,
                track_running_stats=True,
            )

            self.W_P: torch.Tensor = init(torch.empty(self.n_hidden_P, self.n_pair_output_feat))
            self.b_P: torch.Tensor = torch.zeros((self.n_pair_output_feat,))
            self.P_bn: nn.BatchNorm1d = nn.BatchNorm1d(
                num_features=self.n_pair_output_feat,
                eps=1e-3,
                momentum=0.99,
                affine=True,
                track_running_stats=True,
            )
        self.built = True

    def forward(
        self, inputs: List[Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> List[Union[torch.Tensor, torch.Tensor]]:
        atom_features: torch.Tensor = torch.as_tensor(inputs[0])
        pair_features: torch.Tensor = torch.as_tensor(inputs[1])

        pair_split: torch.Tensor = torch.as_tensor(inputs[2])
        atom_to_pair: torch.Tensor = torch.as_tensor(inputs[3])

        activation = self.activation_fn

        AA: torch.Tensor = torch.matmul(atom_features.type(torch.float32), self.W_AA) + self.b_AA
        if self.batch_normalize:
            self.AA_bn.eval()
            AA = self.AA_bn(AA)
        AA = activation(AA)
        PA: torch.Tensor = torch.matmul(pair_features.type(torch.float32), self.W_PA) + self.b_PA
        if self.batch_normalize:
            self.PA_bn.eval()
            PA = self.PA_bn(PA)
        PA = activation(PA)

        t_grp: Dict[Tensor, Tensor] = {}
        idx: int = 0
        for i, s_id in enumerate(pair_split):
            s_id = s_id.item()
            if s_id in t_grp:
                t_grp[s_id] = t_grp[s_id] + PA[idx]
            else:
                t_grp[s_id] = PA[idx]
            idx = i + 1

            lst = list(t_grp.values())
            tensor = torch.stack(lst)
        PA = tensor

        A: torch.Tensor = torch.matmul(torch.concat([AA, PA], 1), self.W_A) + self.b_A
        if self.batch_normalize:
            self.A_bn.eval()
            A = self.A_bn(A)
        A = activation(A)

        if self.update_pair:
            AP_ij: torch.Tensor = (
                torch.matmul(
                    torch.reshape(
                        atom_features[atom_to_pair], [-1, 2 * self.n_atom_input_feat]
                    ).type(torch.float32),
                    self.W_AP,
                )
                + self.b_AP
            )
            if self.batch_normalize:
                self.AP_bn.eval()
                AP_ij = self.AP_bn(AP_ij)
            AP_ij = activation(AP_ij)
            AP_ji: torch.Tensor = (
                torch.matmul(
                    torch.reshape(
                        atom_features[torch.flip(atom_to_pair, [1])],
                        [-1, 2 * self.n_atom_input_feat],
                    ).type(torch.float32),
                    self.W_AP,
                )
                + self.b_AP
            )
            if self.batch_normalize:
                self.AP_bn.eval()
                AP_ji = self.AP_bn(AP_ji)
            AP_ji = activation(AP_ji)
            PP: torch.Tensor = (
                torch.matmul(pair_features.type(torch.float32), self.W_PP) + self.b_PP
            )
            if self.batch_normalize:
                self.PP_bn.eval()
                PP = self.PP_bn(PP)
            PP = activation(PP)
            P: torch.Tensor = (
                torch.matmul(torch.concat([AP_ij + AP_ji, PP], 1).type(torch.float32), self.W_P)
                + self.b_P
            )
            if self.batch_normalize:
                self.P_bn.eval()
                P = self.P_bn(P)
            P = activation(P)
        else:
            P = pair_features

        return [A, P]


class WeaveGather(nn.Module):
    """Weave-gathering layer (verbatim from deepchem.models.torch_models.layers.WeaveGather).

    Expects 2 inputs `[atom_features, atom_split]`.
    """

    def __init__(
        self,
        batch_size: int,
        n_input: int = 128,
        gaussian_expand: bool = True,
        compress_post_gaussian_expansion: bool = False,
        init_: str = "xavier_uniform_",
        activation: str = "tanh",
        **kwargs,
    ):
        super(WeaveGather, self).__init__(**kwargs)
        self.n_input: int = n_input
        self.batch_size: int = batch_size
        self.gaussian_expand: bool = gaussian_expand
        self.compress_post_gaussian_expansion: bool = compress_post_gaussian_expansion
        self.init: str = init_
        self.activation: str = activation
        self.activation_fn: torch.nn.Module = get_activation(activation)

        if self.compress_post_gaussian_expansion:
            init = getattr(initializers, self.init)
            self.W: torch.Tensor = init(torch.empty([self.n_input * 11, self.n_input]))
            self.b: torch.Tensor = torch.zeros((self.n_input,))
        self.built = True

    def forward(self, inputs: List[Union[np.ndarray, np.ndarray]]) -> torch.Tensor:
        outputs: torch.Tensor = torch.as_tensor(inputs[0])
        atom_split: torch.Tensor = torch.as_tensor(inputs[1])

        if self.gaussian_expand:
            outputs = self.gaussian_histogram(outputs)

        t_grp: Dict[Tensor, Tensor] = {}
        idx: int = 0
        for i, s_id in enumerate(atom_split):
            s_id = s_id.item()
            if s_id in t_grp:
                t_grp[s_id] = t_grp[s_id] + outputs[idx]
            else:
                t_grp[s_id] = outputs[idx]
            idx = i + 1

            lst = list(t_grp.values())
            tensor = torch.stack(lst)
        output_molecules: torch.Tensor = tensor

        if self.compress_post_gaussian_expansion:
            output_molecules = torch.matmul(output_molecules.type(torch.float32), self.W) + self.b
            output_molecules = self.activation_fn(output_molecules)

        return output_molecules

    def gaussian_histogram(self, x: torch.Tensor) -> torch.Tensor:
        import torch.distributions as dist

        gaussian_memberships: List[Tuple[float, float]] = [
            (-1.645, 0.283),
            (-1.080, 0.170),
            (-0.739, 0.134),
            (-0.468, 0.118),
            (-0.228, 0.114),
            (0.0, 0.114),
            (0.228, 0.114),
            (0.468, 0.118),
            (0.739, 0.134),
            (1.080, 0.170),
            (1.645, 0.283),
        ]

        distributions: List[dist.Normal] = [
            dist.Normal(torch.tensor(p[0]), torch.tensor(p[1])) for p in gaussian_memberships
        ]
        dist_max: List[torch.Tensor] = [
            distributions[i].log_prob(torch.tensor(gaussian_memberships[i][0])).exp()
            for i in range(11)
        ]

        outputs: List[torch.Tensor] = [
            distributions[i].log_prob(torch.as_tensor(x)).exp() / dist_max[i] for i in range(11)
        ]
        output: torch.Tensor = torch.stack(outputs, dim=2)
        output = output / torch.sum(output, dim=2, keepdim=True)
        output = output.view(-1, self.n_input * 11)
        return output


OneOrMany = Union
ActivationFn = Union[Callable, str]


class Weave(nn.Module):
    """Graph convolutional network for molecular property prediction (Weave-style).

    Verbatim from deepchem.models.torch_models.weavemodel_pytorch.Weave (the
    `nn.Module` powering deepchem's `WeaveModel`). Sequence of layers: Weave
    feature modules -> final convolution -> Weave Gather -> dense layers ->
    softmax (classification) or linear (regression).
    """

    def __init__(
        self,
        n_tasks: int,
        n_atom_feat=75,
        n_pair_feat=14,
        n_hidden: int = 50,
        n_graph_feat: int = 128,
        n_weave: int = 2,
        fully_connected_layer_sizes: List[int] = [2000, 100],
        conv_weight_init_stddevs=0.03,
        weight_init_stddevs=0.01,
        bias_init_consts=0.0,
        dropouts=0.25,
        final_conv_activation_fn=F.tanh,
        activation_fns="relu",
        batch_normalize: bool = True,
        gaussian_expand: bool = True,
        compress_post_gaussian_expansion: bool = False,
        mode: str = "classification",
        n_classes: int = 2,
        batch_size: int = 100,
    ):
        super(Weave, self).__init__()
        if mode not in ["classification", "regression"]:
            raise ValueError("mode must be either 'classification' or 'regression'")

        if not isinstance(n_atom_feat, SequenceCollection):
            n_atom_feat = [n_atom_feat] * n_weave
        if not isinstance(n_pair_feat, SequenceCollection):
            n_pair_feat = [n_pair_feat] * n_weave
        n_layers = len(fully_connected_layer_sizes)
        if not isinstance(conv_weight_init_stddevs, SequenceCollection):
            conv_weight_init_stddevs = [conv_weight_init_stddevs] * n_weave
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(activation_fns, str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers

        self.n_tasks = n_tasks
        self.n_atom_feat = n_atom_feat
        self.n_pair_feat = n_pair_feat
        self.n_hidden = n_hidden
        self.n_graph_feat = n_graph_feat
        self.mode = mode
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.fully_connected_layer_sizes = fully_connected_layer_sizes
        self.weight_init_stddevs = weight_init_stddevs
        self.bias_init_consts = bias_init_consts
        self.dropouts = dropouts
        self.activation_fns = [get_activation(i) for i in activation_fns]
        self.batch_normalize = batch_normalize
        self.n_weave = n_weave

        torch.manual_seed(22)
        self.layers: nn.ModuleList = nn.ModuleList()
        for ind in range(n_weave):
            n_atom: int = self.n_atom_feat[ind]
            n_pair: int = self.n_pair_feat[ind]
            if ind < n_weave - 1:
                n_atom_next: int = self.n_atom_feat[ind + 1]
                n_pair_next: int = self.n_pair_feat[ind + 1]
            else:
                n_atom_next = n_hidden
                n_pair_next = n_hidden
            weave_layer = WeaveLayer(
                n_atom_input_feat=n_atom,
                n_pair_input_feat=n_pair,
                n_atom_output_feat=n_atom_next,
                n_pair_output_feat=n_pair_next,
                batch_normalize=batch_normalize,
            )
            nn.init.trunc_normal_(weave_layer.W_AA, 0, std=conv_weight_init_stddevs[ind])
            nn.init.trunc_normal_(weave_layer.W_PA, 0, std=conv_weight_init_stddevs[ind])
            nn.init.trunc_normal_(weave_layer.W_A, 0, std=conv_weight_init_stddevs[ind])
            if weave_layer.update_pair:
                nn.init.trunc_normal_(weave_layer.W_AP, 0, std=conv_weight_init_stddevs[ind])
                nn.init.trunc_normal_(weave_layer.W_PP, 0, std=conv_weight_init_stddevs[ind])
                nn.init.trunc_normal_(weave_layer.W_P, 0, std=conv_weight_init_stddevs[ind])
            self.layers.append(weave_layer)

        self.dense1: nn.Linear = nn.Linear(n_hidden, self.n_graph_feat)
        self.dense1_act = final_conv_activation_fn
        self.dense1_bn: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=self.n_graph_feat,
            eps=1e-3,
            momentum=0.99,
            affine=True,
            track_running_stats=True,
        )

        self.weave_gather = WeaveGather(
            batch_size,
            n_input=self.n_graph_feat,
            gaussian_expand=gaussian_expand,
            compress_post_gaussian_expansion=compress_post_gaussian_expansion,
        )

        if n_layers > 0:
            self.layers2: nn.ModuleList = nn.ModuleList()
            in_size = self.n_graph_feat * 11
            for ind, layer_size, weight_stddev, bias_const, dropout, activation_fn in zip(
                [0, 1],
                fully_connected_layer_sizes,
                weight_init_stddevs,
                bias_init_consts,
                dropouts,
                self.activation_fns,
            ):
                self.layer: nn.Linear = nn.Linear(in_size, layer_size)
                nn.init.trunc_normal_(self.layer.weight, 0, std=weight_stddev)
                if self.layer.bias is not None:
                    self.layer.bias = nn.Parameter(torch.full(self.layer.bias.shape, bias_const))
                self.layer.layer_bn = nn.BatchNorm1d(
                    num_features=layer_size,
                    eps=1e-3,
                    momentum=0.99,
                    affine=True,
                    track_running_stats=True,
                )
                self.layer.weight_stddev = weight_stddev
                self.layer.bias_const = bias_const
                self.layer.dropout = nn.Dropout(dropout)
                self.layer.layer_act = activation_fn
                self.layers2.append(self.layer)
                in_size = layer_size

        n_tasks = self.n_tasks
        if self.mode == "classification":
            n_classes = self.n_classes
            self.layer_2 = nn.Linear(fully_connected_layer_sizes[1], n_tasks * n_classes)
        else:
            self.layer_2 = nn.Linear(fully_connected_layer_sizes[1], n_tasks)

    def forward(self, inputs):
        input1: List[np.ndarray] = [
            np.array(inputs[0]),
            np.array(inputs[1]),
            np.array(inputs[2]),
            np.array(inputs[4]),
        ]
        for ind in range(self.n_weave):
            weave_layer_ind_A, weave_layer_ind_P = self.layers[ind](input1)
            input1 = [
                weave_layer_ind_A,
                weave_layer_ind_P,
                np.array(inputs[2]),
                np.array(inputs[4]),
            ]

        dense1: torch.Tensor = self.dense1(weave_layer_ind_A)
        dense1 = self.dense1_act(dense1)
        if self.batch_normalize:
            self.dense1_bn.eval()
            dense1 = self.dense1_bn(dense1)

        weave_gather: torch.Tensor = self.weave_gather([dense1, inputs[3]])
        if self.n_layers > 0:
            input_layer: torch.Tensor = weave_gather
            for ind, dropout in zip([0, 1], self.dropouts):
                dense2 = self.layers2[ind]
                layer = self.layers2[ind](input_layer)
                if dropout > 0.0:
                    dense2.dropout.eval()
                    layer = dense2.dropout(layer)
                if self.batch_normalize:
                    dense2.layer_bn.eval()
                    layer = dense2.layer_bn(layer)
                layer = dense2.layer_act(layer)
                input_layer = layer
            output: torch.Tensor = input_layer
        else:
            output = weave_gather

        n_tasks = self.n_tasks
        if self.mode == "classification":
            n_classes = self.n_classes
            logits: torch.Tensor = torch.reshape(self.layer_2(output), (-1, n_tasks, n_classes))
            output = F.softmax(logits, dim=2)
            outputs: List[torch.Tensor] = [output, logits]
        else:
            output = self.layer_2(output)
            outputs = [output]

        return outputs


def build_weavenet():
    # n_tasks=1, small fully_connected_layer_sizes/n_hidden/n_graph_feat to keep
    # the trace lightweight; batch_size matches the 2-molecule example input.
    return Weave(
        n_tasks=1,
        n_atom_feat=75,
        n_pair_feat=14,
        n_hidden=16,
        n_graph_feat=32,
        n_weave=2,
        fully_connected_layer_sizes=[64, 32],
        mode="classification",
        n_classes=2,
        batch_size=2,
    )


def example_input_weavenet():
    # Mirrors the real Weave/WeaveLayer/WeaveGather docstring example: 2
    # molecules ("CCC", "C"), 4 atoms total (3 + 1), 10 pairs total (9 + 1).
    total_n_atoms = 4
    n_atom_feat = 75
    n_pair_feat = 14

    atom_feat = torch.rand(total_n_atoms, n_atom_feat)
    pair_feat = torch.rand(10, n_pair_feat)
    pair_split = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], dtype=torch.long)
    atom_split = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    atom_to_pair = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
            [3, 3],
        ],
        dtype=torch.long,
    )

    return ([atom_feat, pair_feat, pair_split, atom_split, atom_to_pair],)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("WeaveNet", "build_weavenet", "example_input_weavenet", 2016, "vendored"),
]
