# SOURCE: vendored from deepchem/deepchem @ 1bf68943aa870592085386712d3c3c7d00e91a43
# (deepchem/models/torch_models/graphconvmodel.py::_GraphConvTorchModel +
#  deepchem/models/torch_models/layers.py::GraphConv/GraphPool/GraphGather +
#  deepchem/utils/pytorch_utils.py::get_activation/unsorted_segment_sum/unsorted_segment_max)
"""DeepChem GraphConvModel: the PyTorch port of Duvenaud et al. 2015 "Convolutional
networks on graphs for learning molecular fingerprints" (neural graph fingerprints),
NeurIPS 2015, https://arxiv.org/abs/1509.09292.

``_GraphConvTorchModel`` is the real ``nn.Module`` DeepChem's ``GraphConvModel``
(a ``TorchModel`` training-loop wrapper) wraps for the actual forward pass; we trace
that inner module directly since the outer ``TorchModel`` class only adds
fit/predict/save-load plumbing (dataset iteration, loss selection, checkpointing),
not architecture. Its ``GraphConv``/``GraphPool``/``GraphGather`` layers, plus the
small ``get_activation``/``unsorted_segment_sum``/``unsorted_segment_max`` helpers
they depend on, are copied verbatim below (only the ``deepchem.utils.pytorch_utils``
and ``deepchem.models.torch_models.layers`` import paths are collapsed into this
single file; no logic changed). ``deepchem`` itself is not installed in this env, so
the model is vendored rather than imported from the package directly.

The real input to this model is DeepChem's ``ConvMol.agglomerate_mols`` batched
molecular-graph representation: per-atom features sorted by node degree, a
``deg_slice`` giving the (start, count) atom range for each degree bucket, a
``membership`` vector mapping atoms to molecules, and one neighbor-adjacency-index
tensor per degree bucket (``deg_adj_lists``). Building that representation normally
goes through RDKit + ``dc.feat.ConvMolFeaturizer``, which are not installed here;
instead we construct a tiny hand-built 2-molecule batch (a 2-atom bonded pair, both
atoms degree 1, plus a 1-atom lone molecule, degree 0) that matches the exact
``ConvMol.agglomerate_mols`` output layout, so ``_GraphConvTorchModel.forward`` runs
unmodified on real (if synthetic) graph-conv input tensors.
"""

from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as initializers

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from deepchem/utils/pytorch_utils.py --------------------------


def get_activation(fn: Union[Callable, str]):
    """Get a PyTorch activation function, specified either directly or as a string."""
    if isinstance(fn, str):
        return getattr(torch.nn.functional, fn)
    return fn


def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum."""
    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")
    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    s = torch.prod(torch.tensor(data.shape[1:])).long()
    segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape
    shape: List[int] = [num_segments] + list(data.shape[1:])
    tensor: torch.Tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def unsorted_segment_max(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Computes the maximum along segments of a tensor. Analogous to tf.unsorted_segment_max."""
    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have to be a 1-D tensor")
    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.full(shape, float("-inf"), dtype=data.dtype)

    expanded_segment_ids = segment_ids.unsqueeze(-1).expand(-1, *data.shape[1:])

    for i in range(num_segments):
        mask = expanded_segment_ids == i
        tensor[i] = torch.max(data.masked_fill(~mask, float("-inf")), dim=0)[0]

    return tensor


# --- vendored from deepchem/models/torch_models/layers.py -------------------


class GraphConv(nn.Module):
    """Graph Convolutional Layer (Duvenaud et al. 2015 neural fingerprints)."""

    def __init__(
        self,
        out_channel: int,
        number_input_features: int,
        min_deg: int = 0,
        max_deg: int = 10,
        activation_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super(GraphConv, self).__init__(**kwargs)
        self.out_channel: int = out_channel
        self.min_degree: int = min_deg
        self.max_degree: int = max_deg
        self.number_input_features: int = number_input_features
        self.activation_fn: Optional[Callable] = activation_fn

        num_deg: int = 2 * self.max_degree + (1 - self.min_degree)
        self.W_list: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(
                    getattr(initializers, "xavier_uniform_")(
                        torch.empty(number_input_features, self.out_channel)
                    )
                )
                for k in range(num_deg)
            ]
        )
        self.b_list: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(
                    getattr(initializers, "zeros_")(
                        torch.empty(
                            self.out_channel,
                        )
                    )
                )
                for k in range(num_deg)
            ]
        )
        self.built = True

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        atom_features: torch.Tensor = inputs[0]
        deg_slice: torch.Tensor = inputs[1]
        deg_adj_lists: List[torch.Tensor] = inputs[3:]

        W = iter(self.W_list)
        b = iter(self.b_list)

        deg_summed: List = self.sum_neigh(atom_features, deg_adj_lists)

        new_rel_atoms_collection = []

        split_features = torch.split(atom_features, (deg_slice[:, 1]).tolist())
        for deg in range(1, self.max_degree + 1):
            rel_atoms: torch.Tensor = torch.from_numpy(deg_summed[deg - 1])
            self_atoms: torch.Tensor = split_features[deg - self.min_degree]

            rel_out: torch.Tensor = torch.matmul(rel_atoms.type(torch.float32), next(W)) + next(b)
            self_out: torch.Tensor = torch.matmul(self_atoms.type(torch.float32), next(W)) + next(b)
            out: torch.Tensor = rel_out + self_out
            new_rel_atoms_collection.append(torch.from_numpy(out.detach().numpy()))

        if self.min_degree == 0:
            self_atoms = split_features[0]
            out = torch.matmul(self_atoms.type(torch.float32), next(W)) + next(b)
            new_rel_atoms_collection.insert(0, torch.from_numpy(out.detach().numpy()))

        atom_features = torch.concat(new_rel_atoms_collection, 0)

        if self.activation_fn is not None:
            atom_features = self.activation_fn(atom_features)

        return atom_features

    def sum_neigh(self, atoms: torch.Tensor, deg_adj_lists) -> List:
        deg_summed = []
        for deg in range(1, self.max_degree + 1):
            gathered_atoms: torch.Tensor = atoms[deg_adj_lists[deg - 1]]
            summed_atoms: torch.Tensor = torch.sum(gathered_atoms, 1)
            deg_summed.append(summed_atoms.detach().numpy())
        return deg_summed


class GraphPool(nn.Module):
    """A GraphPool gathers data from local neighborhoods of a graph (max-pool)."""

    def __init__(self, min_degree: int = 0, max_degree: int = 10, **kwargs):
        super(GraphPool, self).__init__(**kwargs)
        self.min_degree: int = min_degree
        self.max_degree: int = max_degree

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        atom_features: torch.Tensor = inputs[0]
        deg_slice: torch.Tensor = inputs[1]
        deg_adj_lists: List[torch.Tensor] = inputs[3:]

        deg_maxed = []

        split_features = torch.split(atom_features, (deg_slice[:, 1]).tolist())
        for deg in range(1, self.max_degree + 1):
            self_atoms: torch.Tensor = split_features[deg - self.min_degree]

            if deg_adj_lists[deg - 1].shape[0] == 0:
                maxed_atoms: torch.Tensor = torch.zeros((0, self_atoms.shape[-1]))
                deg_maxed.append(maxed_atoms)
            else:
                self_atoms = torch.unsqueeze(self_atoms, 1)
                gathered_atoms: torch.Tensor = atom_features[deg_adj_lists[deg - 1]]
                gathered_atoms = torch.concat([self_atoms, gathered_atoms], 1)

                max_atoms = torch.max(gathered_atoms, 1)
                deg_maxed.append(max_atoms[0])

        if self.min_degree == 0:
            self_atoms = split_features[0]
            deg_maxed.insert(0, self_atoms)

        return torch.concat(deg_maxed, 0)


class GraphGather(nn.Module):
    """Pools node-level feature vectors to create per-molecule feature vectors."""

    def __init__(self, batch_size: int, activation_fn: Optional[Callable] = None, **kwargs):
        super(GraphGather, self).__init__(**kwargs)
        self.batch_size: int = batch_size
        self.activation_fn: Optional[Callable] = activation_fn

    def forward(self, inputs: List[torch.Tensor]):
        atom_features: torch.Tensor = inputs[0]
        membership: torch.Tensor = inputs[2].to(torch.int64)

        assert self.batch_size > 1, "graph_gather requires batches larger than 1"

        sparse_reps: torch.Tensor = unsorted_segment_sum(atom_features, membership, self.batch_size)
        max_reps: torch.Tensor = unsorted_segment_max(atom_features, membership, self.batch_size)
        mol_features: torch.Tensor = torch.concat([sparse_reps, max_reps], 1)

        if self.activation_fn is not None:
            mol_features = self.activation_fn(mol_features)
        return mol_features


class TrimGraphOutput(nn.Module):
    """Trim the output to the correct number of samples (GraphGather always
    outputs fixed-size batches)."""

    def __init__(self, **kwargs):
        super(TrimGraphOutput, self).__init__(**kwargs)

    def forward(self, inputs):
        n_samples = torch.squeeze(inputs[1])
        return inputs[0][0:n_samples]


# --- vendored from deepchem/models/torch_models/graphconvmodel.py -----------


class _GraphConvTorchModel(nn.Module):
    """Graph Convolutional Model (Duvenaud et al. 2015 neural fingerprints).

    Per-atom feature descriptors are combined and recombined over stacked
    GraphConv/GraphPool layers, then gathered into a per-molecule "neural
    fingerprint" for classification or regression.
    """

    def __init__(
        self,
        n_tasks: int,
        number_input_features: List[int],
        graph_conv_layers: List[int] = [64, 64],
        dense_layer_size: int = 128,
        dropout=0.0,
        mode: str = "classification",
        number_atom_features: int = 75,
        n_classes: int = 2,
        batch_normalize: bool = True,
        uncertainty: bool = False,
        batch_size: int = 100,
    ):
        super(_GraphConvTorchModel, self).__init__()
        if mode not in ["classification", "regression"]:
            raise ValueError("mode must be either 'classification' or 'regression'")
        self.n_tasks: int = n_tasks
        self.n_classes: int = n_classes
        self.mode: str = mode
        self.uncertainty: bool = uncertainty

        if not isinstance(dropout, (list, tuple)):
            dropout = [dropout] * (len(graph_conv_layers) + 1)
        if len(dropout) != len(graph_conv_layers) + 1:
            raise ValueError("Wrong number of dropout probabilities provided")
        if uncertainty:
            if mode != "regression":
                raise ValueError("Uncertainty is only supported in regression mode")
            if any(d == 0.0 for d in dropout):
                raise ValueError("Dropout must be included in every layer to predict uncertainty")

        self.graph_convs: nn.ModuleList = nn.ModuleList(
            [
                GraphConv(layer_size, input_size, activation_fn=get_activation("relu"))
                for layer_size, input_size in zip(graph_conv_layers, number_input_features)
            ]
        )

        self.batch_norms: nn.ModuleList = nn.ModuleList(
            [
                nn.BatchNorm1d(
                    num_features=64, eps=1e-3, momentum=0.99, affine=True, track_running_stats=True
                )
                if batch_normalize
                else nn.Identity()
                for _ in range(len(graph_conv_layers))
            ]
        )
        self.batch_norms.append(
            nn.BatchNorm1d(
                num_features=dense_layer_size,
                eps=1e-3,
                momentum=0.99,
                affine=True,
                track_running_stats=True,
            )
            if batch_normalize
            else nn.Identity()
        )
        self.dropouts: nn.ModuleList = nn.ModuleList(
            [nn.Dropout(rate) if rate > 0.0 else nn.Identity() for rate in dropout]
        )
        self.graph_pools: nn.ModuleList = nn.ModuleList([GraphPool() for _ in graph_conv_layers])
        self.dense: nn.Linear = nn.Linear(64, dense_layer_size)
        self.dense_act = F.relu
        self.graph_gather = GraphGather(batch_size=batch_size, activation_fn=get_activation("tanh"))
        self.trim = TrimGraphOutput()
        if self.mode == "classification":
            self.reshape_dense: nn.Linear = nn.Linear(dense_layer_size * 2, n_tasks * n_classes)
        else:
            self.regression_dense: nn.Linear = nn.Linear(dense_layer_size * 2, n_tasks)
            if self.uncertainty:
                self.uncertainty_dense: nn.Linear = nn.Linear(dense_layer_size * 2, n_tasks)
                self.uncertainty_trim = TrimGraphOutput()

    def forward(self, inputs, training=False) -> List[torch.Tensor]:
        atom_features: torch.Tensor = inputs[0]
        degree_slice: torch.Tensor = inputs[1]
        membership: torch.Tensor = inputs[2].to(torch.int64)
        n_samples: torch.Tensor = inputs[3].to(torch.int64)
        deg_adjs: List[torch.Tensor] = [deg_adj.to(torch.int64) for deg_adj in inputs[4:]]

        in_layer: torch.Tensor = atom_features
        for i in range(len(self.graph_convs)):
            gc_in: List[torch.Tensor] = [in_layer, degree_slice, membership] + deg_adjs
            gc1: torch.Tensor = self.graph_convs[i](gc_in)
            if self.batch_norms[i] is not None:
                gc1 = self.batch_norms[i](gc1)
            if training and self.dropouts[i] is not None:
                gc1 = self.dropouts[i](gc1)
            gp_in: List[torch.Tensor] = [gc1, degree_slice, membership] + deg_adjs
            in_layer = self.graph_pools[i](gp_in)
        dense: torch.Tensor = self.dense(in_layer)
        denseact: torch.Tensor = self.dense_act(dense)
        if self.batch_norms[-1] is not None:
            denseact = self.batch_norms[-1](denseact)
        if training and self.dropouts[-1] is not None:
            denseact = self.dropouts[-1](denseact)
        neural_fingerprint: torch.Tensor = self.graph_gather(
            [denseact, degree_slice, membership] + deg_adjs
        )
        if self.mode == "classification":
            logits: torch.Tensor = torch.reshape(
                self.reshape_dense(neural_fingerprint), (-1, self.n_tasks, self.n_classes)
            )
            logits = self.trim([logits, n_samples])
            output: torch.Tensor = F.softmax(logits, dim=2)
            outputs: List[torch.Tensor] = [output, logits, neural_fingerprint]
        else:
            output = self.regression_dense(neural_fingerprint)
            output = self.trim([output, n_samples])
            if self.uncertainty:
                log_var: torch.Tensor = self.uncertainty_dense(neural_fingerprint)
                log_var = self.uncertainty_trim([log_var, n_samples])
                var: torch.Tensor = torch.exp(log_var)
                outputs = [output, var, output, log_var, neural_fingerprint]
            else:
                outputs = [output, neural_fingerprint]

        return outputs


# --- staging harness -------------------------------------------------------


def build_graphconvmodel():
    # batch_size must match the number of molecules in the example batch
    # (GraphGather asserts batch_size > 1 and unsorted_segment_{sum,max} use it
    # as num_segments over the membership vector).
    return _GraphConvTorchModel(
        n_tasks=1,
        number_input_features=[75, 64],
        graph_conv_layers=[64, 64],
        dense_layer_size=32,
        dropout=0.0,
        mode="classification",
        number_atom_features=75,
        n_classes=2,
        batch_normalize=False,
        uncertainty=False,
        batch_size=2,
    )


def example_input_graphconvmodel():
    # Hand-built 2-molecule batch matching ConvMol.agglomerate_mols' layout:
    # molecule 0 = a 2-atom bonded pair (both atoms degree 1, bonded to each
    # other), molecule 1 = a single unbonded atom (degree 0). Atoms are sorted
    # by degree (degree-0 atoms first, then degree-1, ...), matching the real
    # featurizer's atom ordering.
    num_atom_features = 75
    max_degree = 10

    # 3 total atoms: atom 0 = the degree-0 lone atom (molecule 1), atoms 1,2 =
    # the degree-1 bonded pair (molecule 0), local-indexed within the degree-1
    # bucket as [0, 1].
    atom_features = torch.rand(3, num_atom_features)

    # deg_slice[deg] = (start, count) in the degree-sorted atom_features array.
    deg_slice = torch.zeros(max_degree + 1, 2, dtype=torch.int64)
    deg_slice[0] = torch.tensor([0, 1])  # 1 atom of degree 0
    deg_slice[1] = torch.tensor([1, 2])  # 2 atoms of degree 1

    # membership[i] = which molecule (in the degree-sorted ordering) atom i
    # belongs to: atom 0 (degree-0) -> molecule 1; atoms 1,2 (degree-1,
    # molecule 0's pair) -> molecule 0.
    membership = torch.tensor([1, 0, 0], dtype=torch.int64)

    n_samples = torch.tensor(2, dtype=torch.int64)

    # deg_adjs[deg-1] for deg=1..max_degree: neighbor-index array of shape
    # (num_atoms_of_that_degree, deg), indexing into the *degree-1-local*
    # atom slice for GraphConv.sum_neigh's ``atoms[deg_adj_lists[deg-1]]``
    # gather (see GraphConv.forward: it indexes the raw ``atom_features``
    # tensor passed in, which for the first layer is the full 3-atom tensor).
    # Degree 1: 2 atoms, each with exactly 1 neighbor (each other), indices
    # into the full atom_features tensor (positions 1 and 2).
    deg1_adj = torch.tensor([[2], [1]], dtype=torch.int64)
    deg_adjs = [deg1_adj] + [
        torch.zeros((0, d), dtype=torch.int64) for d in range(2, max_degree + 1)
    ]

    inputs = [atom_features, deg_slice, membership, n_samples] + deg_adjs
    return (inputs,)


MENAGERIE_ENTRIES = [
    ("GraphConvModel", "build_graphconvmodel", "example_input_graphconvmodel", 2015, "vendored"),
]
