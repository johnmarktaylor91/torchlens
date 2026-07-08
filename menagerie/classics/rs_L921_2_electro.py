# SOURCE: vendored from john-bradshaw/electro @ master
#   (repo: https://github.com/john-bradshaw/electro)
#   rxn_steps/model/graph_models.py (NodeEmbedder, GraphAggregator) +
#   rxn_steps/model/action_selector.py (ActionSelector, ActionSelectorInputs) +
#   rxn_steps/model/electro_model.py (Electro) + rxn_steps/data/graph_ds.py
#   (TorchGraph, LogitNodeGraph, ActionSelectorGraphIds), plus its pinned git
#   submodule john-bradshaw/GNN @ 69981455180d4fe43794236edd3e11a019c775aa
#   (.gitmodules / `gh api .../submodules` resolved this exact commit; the
#   submodule has since drifted on `master` -- e.g. renamed
#   `GraphAsAdjList`->`DirectedGraphAsAdjList`, added a `cuda_details`-less
#   `ggnn_sparse.py`, moved `GGNNParams.cuda_details` -- so the pin, not
#   `master`, is the version electro's code actually imports against):
#   graph_neural_networks/core/{utils,mlp,nd_ten_ops,data_types}.py +
#   graph_neural_networks/ggnn_general/{ggnn_base,graph_tops}.py +
#   graph_neural_networks/sparse_pattern/{ggnn_sparse,graph_as_adj_list}.py.
# ELECTRO (Bradshaw, Kusner, Paige, Segler & Hernandez-Lobato, "A Model to
# Search for Synthesizable Molecules", NeurIPS 2019 / arXiv:1906.05221) is an
# autoregressive electron-pushing-step predictor over a molecular graph: a
# sparse Gated Graph Neural Network (GGNN, Li et al. 2015) embeds the reactant
# graph into node embeddings, then per-step MLPs ("action selectors") score
# which atom to select next (initial/remove/add electron-pushing arrows) plus
# a stop probability, optionally conditioned on a reagent-graph context vector
# aggregated the same way. Code copied verbatim from both real repos; only
# imports/relative-paths were flattened into this single file, and the
# `graph_tops.py` import of the (dead-for-our-purposes, unused-by-the-classes
# vendored here) `node_stack_pattern` submodule was dropped -- `mlp` module
# is aliased `gnn_mlp` here to avoid a name clash with torch's own `mlp`
# convention used elsewhere in this file's docstrings.
#
# rdkit is used ONLY for offline SMILES->graph featurization (building the
# node-feature/adjacency-list tensors from a molecule) in the real repo's
# `rxn_steps/data/rdkit_ops/rdkit_featurization_ops.py`; the model's forward
# pass itself (this file) never calls rdkit. The example input below builds a
# tiny hand-written batch of 2 toy "molecules" (adjacency lists) directly,
# matching AtomFeatParams' real feature width (71 atom types + 9 degree bins +
# 12 explicit-valence bins + 8 misc = 100) and real bond-type edge names
# (single/double/triple), bypassing the rdkit featurization step entirely.

import typing
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim from graph_neural_networks/core/utils.py (pinned GNN commit)
# ---------------------------------------------------------------------------


class CudaDetails:
    def __init__(self, use_cuda: bool, gpu_id=None):
        self.use_cuda = use_cuda
        self.gpu_id = gpu_id

    def return_cudafied(self, arg):
        if self.use_cuda:
            arg = arg.cuda(self.gpu_id)
        return arg

    @property
    def r_mod(self):
        return torch.cuda if self.use_cuda else torch

    @property
    def device_str(self):
        return "cuda:0" if self.use_cuda else "cpu"


def from_np_to_cuda(numpy_array, cuda_details: "CudaDetails"):
    return cuda_details.return_cudafied(torch.from_numpy(numpy_array))


# ---------------------------------------------------------------------------
# Verbatim from graph_neural_networks/core/data_types.py (pinned GNN commit)
# ---------------------------------------------------------------------------

TORCH_FLT = torch.float32
NP_LONG = np.int64


# ---------------------------------------------------------------------------
# Verbatim from graph_neural_networks/core/nd_ten_ops.py (pinned GNN commit)
# ---------------------------------------------------------------------------

import enum  # noqa: E402


class NdTensor(enum.Enum):
    NUMPY = "numpy"
    TORCH = "torch"


def work_out_nd_or_tensor(var) -> NdTensor:
    if isinstance(var, torch.Tensor):
        return NdTensor.TORCH
    elif isinstance(var, np.ndarray):
        return NdTensor.NUMPY
    else:
        raise RuntimeError


Nd_Ten = typing.Union[np.ndarray, torch.Tensor]
Op_Nd_Ten = typing.Union[np.ndarray, torch.Tensor, None]


def concatenate(variables: typing.List[Nd_Ten], axis=0) -> Nd_Ten:
    variant = work_out_nd_or_tensor(variables[0])
    if variant is NdTensor.NUMPY:
        return np.concatenate(variables, axis=axis)
    else:
        return torch.cat(variables, dim=axis)


# ---------------------------------------------------------------------------
# Verbatim from graph_neural_networks/core/mlp.py (pinned GNN commit)
# ---------------------------------------------------------------------------


class GnnMlpParams(typing.NamedTuple):
    input_dim: int
    output_dim: int
    hidden_sizes: typing.List[int]


class GnnMLP(nn.Module):
    def __init__(self, params: GnnMlpParams):
        super().__init__()
        self.params = params
        layer_sizes = [self.params.input_dim] + self.params.hidden_sizes + [self.params.output_dim]
        layer_dims = zip(layer_sizes[:-1], layer_sizes[1:])
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for input_dim, output_dim in layer_dims]
        )

    def forward(self, input_tensor):
        hidden = input_tensor
        for i, layer in enumerate(self.linears):
            hidden = layer(hidden)
            if i < self.num_layers - 1:
                hidden = torch.relu(hidden)
        return hidden

    @property
    def num_layers(self):
        return len(self.linears)


# ---------------------------------------------------------------------------
# Verbatim from graph_neural_networks/ggnn_general/ggnn_base.py (pinned commit)
# ---------------------------------------------------------------------------


class GGNNParams(typing.NamedTuple):
    hlayer_size: int
    edge_names: typing.List[str]
    cuda_details: CudaDetails
    num_layers: int


APPENDER_TO_HIDDEN_NAMES = "_bond_proj_"


class GGNNBase(nn.Module):
    """Gated Graph Neural Network (node features). Li et al. 2015, arXiv:1511.05493."""

    def __init__(self, params: GGNNParams):
        super().__init__()
        self.params = params
        self.GRU_hidden = nn.GRUCell(self.params.hlayer_size, self.params.hlayer_size)
        self.A_hidden = nn.ModuleDict(
            {
                k + APPENDER_TO_HIDDEN_NAMES: nn.Linear(
                    self.params.hlayer_size, self.params.hlayer_size
                )
                for k in self.params.edge_names
            }
        )

    def get_edge_names_and_projections(self):
        return ((k[: -len(APPENDER_TO_HIDDEN_NAMES)], v) for k, v in self.A_hidden.items())

    def forward(self, *input):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Verbatim from graph_neural_networks/ggnn_general/graph_tops.py (pinned
# commit) -- only GraphFeaturesFromStackedNodeFeaturesBase and
# GraphFeaturesStackIndexAdd, the two classes ELECTRO actually uses; the
# module-level `from graph_neural_networks.node_stack_pattern import
# stacked_nodes` import (needed by sibling classes GraphFeaturesStackCS /
# GraphFeaturesStackPad, not by StackIndexAdd) is dropped.
# ---------------------------------------------------------------------------


class GraphFeaturesFromStackedNodeFeaturesBase(nn.Module):
    def __init__(self, mlp_project_up, mlp_gate, mlp_func, cuda_details=None):
        super().__init__()
        self.mlp_project_up = mlp_project_up
        self.mlp_gate = mlp_gate
        self.mlp_func = mlp_func
        self.cuda_details = cuda_details


class GraphFeaturesStackIndexAdd(GraphFeaturesFromStackedNodeFeaturesBase):
    """Do the sum by Pytorch's index_add method."""

    def forward(self, node_features, node_to_graph_id):
        proj_up = self.mlp_project_up(node_features)  # [v*, j]
        gate_logit = self.mlp_gate(node_features)  # [v*, 1]
        gate = torch.sigmoid(gate_logit)  # [v*, j]
        gated_vals = gate * proj_up

        num_graphs = node_to_graph_id.max() + 1
        graph_sums = torch.zeros(
            num_graphs, gated_vals.shape[1], device=self.cuda_details.device_str, dtype=TORCH_FLT
        )  # [g, j]
        graph_sums.index_add_(0, node_to_graph_id, gated_vals)

        result = self.mlp_func(graph_sums)  # [g, q]
        return result


# ---------------------------------------------------------------------------
# Verbatim from graph_neural_networks/sparse_pattern/graph_as_adj_list.py
# (pinned GNN commit)
# ---------------------------------------------------------------------------


class GraphAsAdjList:
    def __init__(
        self,
        node_features: Nd_Ten,
        edge_type_to_adjacency_list_map: typing.Mapping[str, Nd_Ten],
        node_to_graph_id: Nd_Ten,
    ):
        self.node_features = node_features
        self.edge_type_to_adjacency_list_map = edge_type_to_adjacency_list_map
        self.node_to_graph_id = node_to_graph_id
        self.max_num_graphs = self.node_to_graph_id.max() + 1
        self.edge_type_to_adjacency_list_directed_map = None

    def do_lazy_ops(self):
        if self.edge_type_to_adjacency_list_directed_map is None:
            new = {}
            for key, value in self.edge_type_to_adjacency_list_map.items():
                if value.shape[0] == 0:
                    new[key] = None
                else:
                    new[key] = concatenate([value, value[::-1]], axis=1)
            self.edge_type_to_adjacency_list_directed_map = new

    @property
    def variant(self) -> NdTensor:
        return work_out_nd_or_tensor(self.node_features)

    def to_torch(self, cuda_details: CudaDetails):
        self.do_lazy_ops()

        def func_to_map(x):
            return None if (x is None or x.size == 0) else from_np_to_cuda(x, cuda_details)

        self._map_all_props(func_to_map)
        return self

    def _map_all_props(self, func):
        self.node_features = func(self.node_features)
        self.node_to_graph_id = func(self.node_to_graph_id)
        self._map_over_adjacency_list_for_all_edges_both_directed_and_undirected(func)

    def _map_over_adjacency_list_for_all_edges_both_directed_and_undirected(self, func):
        self.edge_type_to_adjacency_list_map = {
            k: func(v) for k, v in self.edge_type_to_adjacency_list_map.items()
        }
        self.edge_type_to_adjacency_list_directed_map = {
            k: func(v) for k, v in self.edge_type_to_adjacency_list_directed_map.items()
        }


# ---------------------------------------------------------------------------
# Verbatim from graph_neural_networks/sparse_pattern/ggnn_sparse.py (pinned
# GNN commit)
# ---------------------------------------------------------------------------


class GGNNSparse(GGNNBase):
    def forward(self, graphs: GraphAsAdjList):
        hidden = graphs.node_features
        num_nodes = hidden.shape[0]

        for _t in range(self.params.num_layers):
            message = torch.zeros(
                num_nodes,
                self.params.hlayer_size,
                device=self.params.cuda_details.device_str,
                dtype=TORCH_FLT,
            )

            for edge_name, projection in self.get_edge_names_and_projections():
                adj_list = graphs.edge_type_to_adjacency_list_directed_map[edge_name]
                if adj_list is None:
                    continue  # no edges of this type
                projected_feats = projection(hidden)
                message.index_add_(0, adj_list[0], projected_feats.index_select(0, adj_list[1]))

            hidden = self.GRU_hidden(message, hidden)

        return GraphAsAdjList(
            hidden, graphs.edge_type_to_adjacency_list_map, graphs.node_to_graph_id
        )


# ---------------------------------------------------------------------------
# Verbatim from rxn_steps/data/graph_ds.py
# ---------------------------------------------------------------------------

TORCH_INT = torch.int64


def isin(ar1, ar2):
    """same as numpy.isin -- https://github.com/pytorch/pytorch/issues/3025"""
    return (ar1[..., None] == ar2).any(-1)


class TorchGraph:
    """Stacks node features belonging to all graphs one on top of each other
    (no padding needed for graphs with different node counts). Nodes
    belonging to each graph must be grouped together; graph ids consecutive
    starting from 0."""

    def __init__(
        self,
        node_features: torch.Tensor,
        node_to_graphid: torch.Tensor,
        num_nodes_per_graph: typing.Optional[torch.Tensor] = None,
    ):
        self.node_features = node_features  # [v*, h]
        self.node_to_graphid = node_to_graphid  # [v*]
        self.empty_graph = self.node_features.shape[0] == 0

        if not self.empty_graph:
            if num_nodes_per_graph is None:
                self.num_nodes_per_graph = self._get_the_number_of_nodes_per_graph()
            else:
                self.num_nodes_per_graph = num_nodes_per_graph
            self.graph_offsets = torch.cat(
                [
                    torch.tensor([0], dtype=TORCH_INT, device=self.dev_str_n2gid),
                    torch.cumsum(self.num_nodes_per_graph[:-1], 0),
                ]
            )

    @property
    def max_num_graphs(self):
        return self.node_to_graphid.max() + 1

    @property
    def dev_str_n2gid(self):
        return str(self.node_to_graphid.device)

    @property
    def dev_str_feats(self):
        return str(self.node_features.device)

    def __getitem__(self, graph_ids_of_interest):
        no_graphs_selected = graph_ids_of_interest.shape[0] == 0
        if self.empty_graph or no_graphs_selected:
            return None
        else:
            nodes_to_use_mask = isin(self.node_to_graphid, graph_ids_of_interest)
            new_node_features = self.node_features[nodes_to_use_mask, :]
            num_nodes_per_graph = self.num_nodes_per_graph[graph_ids_of_interest]

            new_graph_ids = torch.zeros(
                self.max_num_graphs, device=self.dev_str_n2gid, dtype=TORCH_INT
            )
            new_graph_ids[graph_ids_of_interest] = torch.arange(
                0, graph_ids_of_interest.shape[0], device=self.dev_str_n2gid, dtype=TORCH_INT
            )
            old_node_to_graph_ids_of_those_using = self.node_to_graphid[nodes_to_use_mask]
            new_node_to_graph_id = new_graph_ids[old_node_to_graph_ids_of_those_using]

            return TorchGraph(new_node_features, new_node_to_graph_id, num_nodes_per_graph)

    def _get_the_number_of_nodes_per_graph(self):
        dev_str = self.dev_str_n2gid
        ones = torch.ones(self.node_features.shape[0], device=dev_str, dtype=TORCH_INT)
        graph_sums = torch.zeros(self.max_num_graphs, device=dev_str, dtype=TORCH_INT)
        graph_sums.index_add_(0, self.node_to_graphid, ones)
        return graph_sums


class LogitNodeGraph(TorchGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.empty_graph:
            assert self.node_features.shape[1] == 1, "The node features should now be logits"

    @property
    def squeezed_logits(self):
        return torch.squeeze(self.node_features, dim=1)


@dataclass
class ActionSelectorGraphIds:
    graphs_ids: Nd_Ten
    prev_action_per_graph: Op_Nd_Ten
    reagent_context_ids: Op_Nd_Ten


# ---------------------------------------------------------------------------
# Verbatim from rxn_steps/model/graph_models.py
# ---------------------------------------------------------------------------


class NodeEmbedder(nn.Module):
    """Takes node features and graph structure and computes node embeddings."""

    def __init__(
        self,
        hidden_layer_size: int,
        edge_names: typing.List[str],
        embedding_dim: int,
        cuda_details: CudaDetails,
        num_time_steps: int,
    ):
        super().__init__()
        self.ggnn = GGNNSparse(
            GGNNParams(hidden_layer_size, edge_names, cuda_details, num_time_steps)
        )
        self.embedding_dim = embedding_dim

    def forward(self, g_adjlist: GraphAsAdjList):
        g_adjlist = self.ggnn(g_adjlist)
        return TorchGraph(g_adjlist.node_features, node_to_graphid=g_adjlist.node_to_graph_id)


class GraphAggregator(nn.Module):
    """Attention-weighted sum of node embeddings into a graph-level embedding
    (convenience wrapper on GraphFeaturesStackIndexAdd)."""

    def __init__(self, node_feature_dim: int, final_dim: int, cuda_details: CudaDetails):
        super().__init__()
        mlp_up = GnnMLP(GnnMlpParams(node_feature_dim, 2 * node_feature_dim, []))
        mlp_gate = GnnMLP(GnnMlpParams(node_feature_dim, 1, []))
        mlp_func = GnnMLP(GnnMlpParams(2 * node_feature_dim, final_dim, []))
        self.g_top = GraphFeaturesStackIndexAdd(
            mlp_up, mlp_gate, mlp_func, cuda_details=cuda_details
        )

    def forward(self, graphs: TorchGraph):
        return self.g_top(graphs.node_features, graphs.node_to_graphid)


# ---------------------------------------------------------------------------
# Verbatim from rxn_steps/model/action_selector.py
# ---------------------------------------------------------------------------


class ActionSelectorInputs(typing.NamedTuple):
    graphs: TorchGraph
    prev_action_per_graph: typing.Optional[torch.Tensor]
    context_vectors_per_graph: typing.Optional[torch.Tensor]


class ActionSelector(nn.Module):
    def __init__(self, mlp_input_size, hidden_sizes):
        super().__init__()
        self.mlp = GnnMLP(GnnMlpParams(mlp_input_size, 1, hidden_sizes))

    def forward(self, input_: ActionSelectorInputs) -> typing.Optional[LogitNodeGraph]:
        if input_.graphs is None:
            return None

        stacked_nodes = input_.graphs.node_features  # [v*, h]

        if input_.prev_action_per_graph is not None:
            prev_actions = input_.graphs.graph_offsets + input_.prev_action_per_graph
            prev_action_features = stacked_nodes[prev_actions]
            concat_mat = prev_action_features[input_.graphs.node_to_graphid, :]
            stacked_nodes = torch.cat([stacked_nodes, concat_mat], dim=1)

        if input_.context_vectors_per_graph is not None:
            concat_mat = input_.context_vectors_per_graph[input_.graphs.node_to_graphid, :]
            stacked_nodes = torch.cat([stacked_nodes, concat_mat], dim=1)

        stacked_logits = self.mlp(stacked_nodes)  # [v*, h] -> [v*, 1]
        return LogitNodeGraph(stacked_logits, input_.graphs.node_to_graphid)


# ---------------------------------------------------------------------------
# Verbatim from rxn_steps/model/electro_model.py
# ---------------------------------------------------------------------------


class Electro(nn.Module):
    def __init__(
        self,
        stop_net_aggregator: GraphAggregator,
        reagents_net_aggregator: typing.Optional[GraphAggregator],
        initial_select: ActionSelector,
        remove_select: ActionSelector,
        add_select: ActionSelector,
    ):
        super().__init__()
        self.stop_net_aggregator = stop_net_aggregator
        self.reagents_net_aggregator = reagents_net_aggregator
        self.initial_select = initial_select
        self.remove_select = remove_select
        self.add_select = add_select

    def forward(
        self,
        graphs: TorchGraph,
        initial_select_inputs: ActionSelectorGraphIds,
        remove_select_inputs: ActionSelectorGraphIds,
        add_select_inputs: ActionSelectorGraphIds,
        reagent_graphs: typing.Optional[TorchGraph] = None,
    ):
        stop_logits = self.stop_net_aggregator(graphs)
        stop_logits = torch.squeeze(stop_logits, dim=1)

        if self.reagents_net_aggregator is not None:
            assert reagent_graphs is not None
            reagent_context = self.reagents_net_aggregator(reagent_graphs)
        else:
            reagent_context = None

        results = []
        for action_selector_input, selector in [
            (initial_select_inputs, self.initial_select),
            (remove_select_inputs, self.remove_select),
            (add_select_inputs, self.add_select),
        ]:
            graphs_of_interest = graphs[action_selector_input.graphs_ids]

            if graphs_of_interest is None:
                results.append(None)
            else:
                reagents_necessary = (
                    (reagent_context is not None)
                    and (action_selector_input.reagent_context_ids is not None)
                    and (action_selector_input.reagent_context_ids.shape[0] != 0)
                )
                if reagents_necessary:
                    context = reagent_context[action_selector_input.reagent_context_ids]
                else:
                    context = None

                inp = ActionSelectorInputs(
                    graphs_of_interest, action_selector_input.prev_action_per_graph, context
                )
                logits = selector(inp)
                results.append(logits)
        initial_action_logits, remove_action_logits, add_action_logits = results
        return stop_logits, initial_action_logits, remove_action_logits, add_action_logits


class ElectroFullModel(nn.Module):
    """Menagerie wrapper mirroring the real repo's `get_electro.FullModel`
    (NodeEmbedder + Electro composed end-to-end): embed the reactant graph
    and the reagent graph with the same GGNN, then run the electron-pushing
    action selectors over the embedded reactant graph."""

    def __init__(self, ggnn: NodeEmbedder, electro: Electro):
        super().__init__()
        self.ggnn = ggnn
        self.electro = electro

    def forward(
        self,
        reactant_adjlist: GraphAsAdjList,
        reagent_adjlist: GraphAsAdjList,
        initial_select_inputs: ActionSelectorGraphIds,
        remove_select_inputs: ActionSelectorGraphIds,
        add_select_inputs: ActionSelectorGraphIds,
    ):
        embedded_reactants = self.ggnn(reactant_adjlist)
        embedded_reagents = self.ggnn(reagent_adjlist)
        return self.electro(
            embedded_reactants,
            initial_select_inputs,
            remove_select_inputs,
            add_select_inputs,
            embedded_reagents,
        )


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------

# Real AtomFeatParams feature width from rxn_steps/data/rdkit_ops/
# rdkit_featurization_ops.py: 71 atom types + 9 degree bins + 12
# explicit-valence bins + 8 misc one-hot/scalar feats = 100.
_NODE_FEAT_LEN = 100
_EDGE_NAMES = ["single", "double", "triple"]


def build_electro():
    """Tiny-size real ELECTRO (NodeEmbedder GGNN + Electro action-selection
    heads), CPU-only CudaDetails, node feature width matching the real
    AtomFeatParams.atom_feature_length (100) and real bond edge names."""
    cuda_details = CudaDetails(use_cuda=False)
    ggnn = NodeEmbedder(
        hidden_layer_size=_NODE_FEAT_LEN,
        edge_names=_EDGE_NAMES,
        embedding_dim=_NODE_FEAT_LEN,
        cuda_details=cuda_details,
        num_time_steps=2,
    )

    reagent_ctx_size = 20
    stop_net = GraphAggregator(_NODE_FEAT_LEN, 1, cuda_details)
    reagent_context_net = GraphAggregator(_NODE_FEAT_LEN, reagent_ctx_size, cuda_details)
    initial_select = ActionSelector(_NODE_FEAT_LEN + reagent_ctx_size, [50])
    remove_select = ActionSelector(2 * _NODE_FEAT_LEN, [50])
    add_select = ActionSelector(2 * _NODE_FEAT_LEN, [50])

    electro = Electro(stop_net, reagent_context_net, initial_select, remove_select, add_select)
    return ElectroFullModel(ggnn, electro)


def example_input_electro():
    """Two tiny hand-built toy 'molecules' (3-atom and 2-atom graphs) as a
    GraphAsAdjList batch, with a single `single`-bond edge connecting the
    first atom of each graph, matching what a real rdkit-featurized reactant
    graph would look like (bypassing rdkit itself, which is used only for
    offline SMILES->graph featurization in the source repo, never inside the
    model's forward pass). A second identically-shaped batch stands in for
    the reagent graph, plus a real ActionSelectorGraphIds per action head."""
    rng = np.random.RandomState(0)
    n0, n1 = 3, 2
    total_nodes = n0 + n1
    node_to_graph_id_np = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    edge_map_np = {
        "single": np.array([[0, 3], [1, 4]], dtype=np.int64),
        "double": np.zeros((2, 0), dtype=np.int64),
        "triple": np.zeros((2, 0), dtype=np.int64),
    }
    cuda_details = CudaDetails(use_cuda=False)

    reactant_feats = rng.randn(total_nodes, _NODE_FEAT_LEN).astype(np.float32)
    reactant_adjlist = GraphAsAdjList(reactant_feats, edge_map_np, node_to_graph_id_np).to_torch(
        cuda_details
    )

    reagent_feats = rng.randn(total_nodes, _NODE_FEAT_LEN).astype(np.float32)
    reagent_adjlist = GraphAsAdjList(reagent_feats, edge_map_np, node_to_graph_id_np).to_torch(
        cuda_details
    )

    graphs_ids = torch.tensor([0, 1], dtype=torch.int64)
    initial_inputs = ActionSelectorGraphIds(
        graphs_ids=graphs_ids,
        prev_action_per_graph=None,
        reagent_context_ids=torch.tensor([0, 1], dtype=torch.int64),
    )
    remove_inputs = ActionSelectorGraphIds(
        graphs_ids=graphs_ids,
        prev_action_per_graph=torch.tensor([0, 0], dtype=torch.int64),
        reagent_context_ids=None,
    )
    add_inputs = ActionSelectorGraphIds(
        graphs_ids=graphs_ids,
        prev_action_per_graph=torch.tensor([0, 0], dtype=torch.int64),
        reagent_context_ids=None,
    )

    return (reactant_adjlist, reagent_adjlist, initial_inputs, remove_inputs, add_inputs)


MENAGERIE_ENTRIES = [
    (
        "ELECTRO",
        build_electro,
        example_input_electro,
        2019,
        "CODE",
    ),
]
