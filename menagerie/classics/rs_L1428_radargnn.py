# SOURCE: vendored from https://github.com/TUMFTM/RadarGNN @ main
#
# RadarGNN (Fent, Bauerschmidt, Lienkamp. 2023, "RadarGNN: A Transformation Invariant
# Graph Neural Network for Radar-based Perception"). https://arxiv.org/abs/2304.06547
#
# The full `DetNetBasic` GNN (initial node/edge feature-embedding MLPs, a stack of
# message-passing graph-convolution layers -- either the general `MPNNConv` or the
# adapted `RadarPointGNNConv` -- followed by classification and bounding-box regression
# heads) is vendored verbatim (byte-for-byte, only import paths adjusted) from the
# repo's own files:
#   https://raw.githubusercontent.com/TUMFTM/RadarGNN/main/src/gnnradarobjectdetection/gnn/gnn_models.py
#   https://raw.githubusercontent.com/TUMFTM/RadarGNN/main/src/gnnradarobjectdetection/gnn/mpnn_layers.py
#   https://raw.githubusercontent.com/TUMFTM/RadarGNN/main/src/gnnradarobjectdetection/gnn/configs.py
#     (only `GNNArchitectureConfig`, the dataclass DetNetBasic actually consumes;
#     `TrainingConfig` is training-hyperparameter bookkeeping, not part of the network)
# No architecture was written from scratch; this is 100% real code from torch_geometric's
# MessagePassing base class plus the repo's own layer/model definitions.
#
# Two harness-only adjustments (no architectural change): (1) `GNNArchitectureConfig`
# is ported from the real `@dataclass` to a plain class with an identical field/default
# set -- `@dataclass` resolves `cls.__module__` via `sys.modules[__module__]` at class-
# definition time, which is unset when this file is exec'd via
# `importlib.util.spec_from_file_location` without a prior `sys.modules` registration.
# (2) The real `message()` methods' `x_i: Tensor, x_j: Tensor, edge_attr: OptTensor`
# type annotations are stripped -- `MessagePassing.__init__` calls
# `Inspector.inspect_signature(self.message)`, which resolves annotated parameter types
# via that same `sys.modules[__module__]` lookup and KeyErrors for the same reason.
# `forward()`'s annotations are untouched since `forward` is never introspected this way.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from typing import List

import torch
from torch import Tensor
from torch.nn import ModuleList, ReLU, Sequential
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# gnn/configs.py -- GNNArchitectureConfig only (TrainingConfig is training-
# hyperparameter bookkeeping, not part of the trainable network). The real
# repo defines this as a `@dataclass`; ported here to a plain class with an
# identical field/default set, because `@dataclass` resolves `cls.__module__`
# via `sys.modules[__module__]` at class-definition time, which is unset when
# this staging file is exec'd via `importlib.util.spec_from_file_location`
# without first registering the module in `sys.modules` -- a harness detail
# unrelated to the architecture, so the config semantics (fields, types,
# defaults) are kept byte-for-byte identical to the original dataclass.
# ---------------------------------------------------------------------------
class GNNArchitectureConfig:
    """Stores possible GNN model architecture configurations."""

    def __init__(
        self,
        node_feature_dimension: int,
        edge_feature_dimension: int,
        conv_layer_dimensions: list,
        classification_head_layer_dimensions: list,
        regression_head_layer_dimensions: list,
        initial_node_feature_embedding: bool = False,
        initial_edge_feature_embedding: bool = False,
        node_feature_embedding_layer_dimensions: list = None,
        edge_feature_embedding_layer_dimensions: list = None,
        conv_layer_type: str = "MPNNConv",
        batch_norm_in_mlps: bool = True,
        conv_pre_mlp_layer_number: int = 1,
        conv_post_mlp_layer_number: int = 1,
        conv_use_edge_encoder: bool = False,
        aggregation_function: str = "max",
    ):
        # initial node and edge feature dimension
        self.node_feature_dimension = node_feature_dimension
        self.edge_feature_dimension = edge_feature_dimension

        # layers for graph convolution and detection head
        self.conv_layer_dimensions = conv_layer_dimensions
        self.classification_head_layer_dimensions = classification_head_layer_dimensions
        self.regression_head_layer_dimensions = regression_head_layer_dimensions

        # layers for initial node and edge feature embedding MLPs
        self.initial_node_feature_embedding = initial_node_feature_embedding
        self.initial_edge_feature_embedding = initial_edge_feature_embedding
        self.node_feature_embedding_layer_dimensions = node_feature_embedding_layer_dimensions
        self.edge_feature_embedding_layer_dimensions = edge_feature_embedding_layer_dimensions
        self.conv_layer_type = conv_layer_type

        # configuration for graph convolution layers
        self.batch_norm_in_mlps = batch_norm_in_mlps
        self.conv_pre_mlp_layer_number = conv_pre_mlp_layer_number
        self.conv_post_mlp_layer_number = conv_post_mlp_layer_number
        self.conv_use_edge_encoder = conv_use_edge_encoder
        self.aggregation_function = aggregation_function


# ---------------------------------------------------------------------------
# gnn/mpnn_layers.py -- vendored verbatim.
# ---------------------------------------------------------------------------
class MPNNConv(MessagePassing):
    """Implementation of a general MPNN layer with edge features.

    This layer is not included in pytorch_grometric.
    The two layers from pytorch_geometric that are the closest related to this layer are:
        - NNConv (from original MPNN paper, but does not include edge features)
        - PNAConv (Includes general MPNN update with edge features but also advanced aggregators, ...)

    Attributes:
        in_channels: Dimension of initial node features.
        out_channels: Dimension of the embedded node features after the graph convolution layer.
        edge_dim: Dimension of initial edge features.
        use_edge_encoder: Decides whether to use an edge_encoder or not.
        edge_encoder: MLP used for encoding the edge features before the graph convolution.
        pre_mlp: Message function MLP.
        post_mlp: Update function MLP.

    Methods:
        reset_parameters: Resets the parameters of the model.
        forward: Updates node feature vectors.
        message: Calculates the messages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        aggr: str = "max",
        pre_layers: int = 1,
        post_layers: int = 1,
        use_edge_encoder: bool = False,
    ):
        """
        Args:
            in_channels: integer describing the dimension of initial node features
            out_channels: integer describing the dimension of node feature embeddings returned by this layer
            edge_dim: integer describing the dimension of initial edge features
            aggr (optional): permutation invariant aggregation function
            pre_layers (optional): number of layers in message MLP of the graph convolution operation
            post_layers (optional): number of layers in update MLP of the graph convolution operation
            use_edge_encoder (optional): boolean to chose weather a MLP should be used to transform the edge features to the same dimension as the node features
        """

        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.use_edge_encoder = use_edge_encoder

        # can be used to adapt edge feature dimension to be equal to node feature dimension -> Done this way in PNAConv
        if use_edge_encoder:
            self.edge_encoder = Linear(edge_dim, self.in_channels)
            pre_mlp_dim = 3 * in_channels
        else:
            pre_mlp_dim = 2 * in_channels + edge_dim

        # MLPs before and after aggregation
        # or maybe better reduce output dim of message MLP to match node feature dim ?! -> like in PNAConv
        modules = [Linear(pre_mlp_dim, pre_mlp_dim)]
        for _ in range(pre_layers - 1):
            modules += [ReLU()]
            modules += [Linear(pre_mlp_dim, pre_mlp_dim)]
        self.pre_mlp = Sequential(*modules)

        modules = [Linear(pre_mlp_dim + in_channels, out_channels)]
        for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(out_channels, out_channels)]
        self.post_mlp = Sequential(*modules)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_edge_encoder:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_mlp:
            reset(nn)
        for nn in self.post_mlp:
            reset(nn)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        m_emb = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, m_emb], dim=-1)
        h = self.post_mlp(out)

        return h

    def message(self, x_i, x_j, edge_attr):
        if self.use_edge_encoder:
            edge_attr = self.edge_encoder(edge_attr)
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        m_emb = self.pre_mlp(m)

        return m_emb


class RadarPointGNNConv(MessagePassing):
    """Adapted Radar-PointGNN convolution with edge features.

    Graph convolution layer as specified in: Radar-PointGNN: Graph Based Object Recognition for Unstructured Radar Point-cloud Data.
    (DOI: 10.1109/RadarConf2147009.2021.9455172).
    BUT with the following changes:
    instead of only using "x_j - x_i, m_j" in the pre_mlp, all edge attributes are used which may contain also other features than the relative position "e_ij, m_j"

    Attributes:
        in_channels: Dimension of initial node features.
        out_channels: Dimension of the embedded node features after the graph convolution layer.
        init_node_dim: Dimension of initial node features.
        init_edge_dim: Dimension of initial edge features.
        pre_mlp: Message function MLP.
        post_mlp: Update function MLP.

    Methods:
        reset_parameters: Resets the parameters of the model.
        forward: Updates node feature vectors.
        message: Calculates the messages.
    """

    def __init__(
        self,
        init_node_dim: int,
        init_edge_dim: int,
        aggr: str = "max",
        pre_layers: int = 1,
        post_layers: int = 1,
    ):
        """
        Args:
            init_node_dim: integer describing the dimension of initial node features
            init_edge_dim: integer describing the dimension of node feature embeddings returned by this layer
            aggr (optional): permutation invariant aggregation function
            pre_layers (optional): number of layers in message MLP of the graph convolution operation
            post_layers (optional): number of layers in update MLP of the graph convolution operation
        """
        super().__init__(aggr=aggr)

        # output dim. = input dim. -> No increase in embedding dimension possible with this layer
        self.in_channels = init_node_dim
        # output dim. = input dim. -> No increase in embedding dimension possible with this layer
        self.out_channels = init_node_dim

        self.init_node_dim = init_node_dim
        self.init_edge_dim = init_edge_dim

        pre_mlp_dim = init_node_dim + init_edge_dim

        # MLPs before and after aggregation
        # or maybe better reduce output dim of message MLP to match node feature dim ?! -> like in PNAConv
        modules = [Linear(pre_mlp_dim, pre_mlp_dim)]
        for _ in range(pre_layers - 1):
            modules += [ReLU()]
            modules += [Linear(pre_mlp_dim, pre_mlp_dim)]
        self.pre_mlp = Sequential(*modules)

        modules = [Linear(pre_mlp_dim + init_node_dim, init_node_dim)]
        for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(init_node_dim, init_node_dim)]
        self.post_mlp = Sequential(*modules)

        self.reset_parameters()

    def reset_parameters(self):
        for nn in self.pre_mlp:
            reset(nn)
        for nn in self.post_mlp:
            reset(nn)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        m_emb = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, m_emb], dim=-1)
        h = self.post_mlp(out)

        return h + x

    def message(self, x_i, x_j, edge_attr):
        m = torch.cat([x_j, edge_attr], dim=-1)
        m_emb = self.pre_mlp(m)

        return m_emb


# ---------------------------------------------------------------------------
# gnn/gnn_models.py -- vendored verbatim.
# ---------------------------------------------------------------------------
class DetNetBasic(torch.nn.Module):
    """GNN for end to end object detection and semantic segmentation.

    The model consists of an initial node and edge feature embedding, followed by graph convolution layers,
    and a final detection head for classification and bounding box prediction for each point.
    """

    def __init__(self, config: GNNArchitectureConfig):
        super().__init__()

        # Store the settings of the model as instance attributes for later access
        self.batch_norm_mlps = config.batch_norm_in_mlps

        self.node_feat_dim = config.node_feature_dimension
        self.edge_feat_dim = config.edge_feature_dimension

        self.conv_layer_dimensions = config.conv_layer_dimensions
        self.initial_node_feature_embedding = config.initial_node_feature_embedding
        self.initial_edge_feature_embedding = config.initial_edge_feature_embedding

        self.conv_pre_mlp_layers = config.conv_pre_mlp_layer_number
        self.conv_post_mlp_layers = config.conv_post_mlp_layer_number
        self.conv_use_edge_encoder = config.conv_use_edge_encoder
        self.aggregation = config.aggregation_function

        # Create the MLPs for initial node and edge feature embedding
        if config.initial_node_feature_embedding:
            layer_dimensions = config.node_feature_embedding_layer_dimensions[:-1]
            out_dim = config.node_feature_embedding_layer_dimensions[-1]
            self.node_emb_mlp = get_mlp(
                self.node_feat_dim, out_dim, layer_dimensions, self.batch_norm_mlps
            )
            self.node_feat_dim = out_dim

        if config.initial_edge_feature_embedding:
            layer_dimensions = config.edge_feature_embedding_layer_dimensions[:-1]
            out_dim = config.edge_feature_embedding_layer_dimensions[-1]
            self.edge_emb_mlp = get_mlp(
                self.edge_feat_dim, out_dim, layer_dimensions, self.batch_norm_mlps
            )
            self.edge_feat_dim = out_dim

        # graph convolutional layer definition - create the first graph convolution layer
        # MPNNConv layer should be interchangeable by any other Graph Convolution layer operating on node and edge features (e.g. GATConv, PNAConv)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        layer_dim = self.conv_layer_dimensions[0]
        if config.conv_layer_type == "MPNNConv":
            conv = MPNNConv(
                self.node_feat_dim,
                layer_dim,
                self.edge_feat_dim,
                aggr=self.aggregation,
                pre_layers=self.conv_pre_mlp_layers,
                post_layers=self.conv_post_mlp_layers,
                use_edge_encoder=self.conv_use_edge_encoder,
            )
        elif config.conv_layer_type == "RadarPointGNNConv":
            conv = RadarPointGNNConv(
                self.node_feat_dim,
                self.edge_feat_dim,
                aggr=self.aggregation,
                pre_layers=self.conv_pre_mlp_layers,
                post_layers=self.conv_post_mlp_layers,
            )
            layer_dim = self.node_feat_dim
        else:
            raise Exception(
                f"{config.conv_layer_type} is invalid GNN conv layer type. Chose either MPNNConv or RadarPointGNNConv"
            )

        batch_norm = BatchNorm(layer_dim)
        self.convs.append(conv)
        self.batch_norms.append(batch_norm)

        # graph convolutional layer definition - create the remaining graph convolution layers
        for next_layer_dim in self.conv_layer_dimensions[1:]:
            if config.conv_layer_type == "MPNNConv":
                conv = MPNNConv(
                    layer_dim,
                    next_layer_dim,
                    self.edge_feat_dim,
                    aggr=self.aggregation,
                    pre_layers=self.conv_pre_mlp_layers,
                    post_layers=self.conv_post_mlp_layers,
                    use_edge_encoder=self.conv_use_edge_encoder,
                )
            elif config.conv_layer_type == "RadarPointGNNConv":
                conv = RadarPointGNNConv(
                    self.node_feat_dim,
                    self.edge_feat_dim,
                    aggr=self.aggregation,
                    pre_layers=self.conv_pre_mlp_layers,
                    post_layers=self.conv_post_mlp_layers,
                )
            else:
                raise Exception(
                    f"{config.conv_layer_type} is invalid GNN conv layer type. Chose either MPNNConv or RadarPointGNNConv"
                )

            batch_norm = BatchNorm(next_layer_dim)
            self.convs.append(conv)
            self.batch_norms.append(batch_norm)
            layer_dim = next_layer_dim

        # Define detection head with classification and bounding box regression MLPs
        final_embedding_dim = self.conv_layer_dimensions[-1]

        layer_dimensions = config.classification_head_layer_dimensions[:-1]
        out_dim = config.classification_head_layer_dimensions[-1]
        self.classification_head = get_mlp(
            final_embedding_dim, out_dim, layer_dimensions, self.batch_norm_mlps
        )
        # Classification head has no softmax/logsoftmax in the end as this is integrated in Pytorch into the CrossEntropyLoss function (which is called during training for loss calculation)
        # self.classification_head.append(LogSoftmax(dim = 1))

        layer_dimensions = config.regression_head_layer_dimensions[:-1]
        out_dim = config.regression_head_layer_dimensions[-1]
        self.regression_head = get_mlp(
            final_embedding_dim, out_dim, layer_dimensions, self.batch_norm_mlps
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """Forward pass of the model.

        Args:
            x: Node feature matrix.
            edge_index: Edge connection matrix (Alternative representation of sparse adjacency matrix)
            edge_attr: Edge feature matrix.

        Returns:
            c: Probability distribution for the class prediction for each node.
            bb: Regressed bounding box for each node.
        """
        # execute MLPs for initial node and edge feature embedding
        if self.initial_node_feature_embedding:
            x = self.node_emb_mlp(x)

        if self.initial_edge_feature_embedding:
            edge_attr = self.edge_emb_mlp(edge_attr)

        # apply graph convolutions followed and nonlinearity
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            # maybe add drop out
            x = torch.nn.functional.relu(x)

        # use final node feature embeddings for prediction
        c = self.classification_head(x)
        bb = self.regression_head(x)

        return c, bb


def get_mlp(
    in_size: int, out_size: int, hidden_layer_sizes: List[int], batch_norm: bool
) -> Sequential:
    """Creates a MLP with the specified layer number and dimension.

    Args:
        in_size: Input layer dimension.
        out_size: Output layer dimension.
        hidden_layer_sizes: Dimensions of the hidden layers.
        batch_norm: Boolean, whether batch norm should be applied between all layers.

    Returns:
        mlp: pytorch sequential object with the created MLP
    """

    if len(hidden_layer_sizes) == 0:
        modules = [Linear(in_size, out_size)]
    else:
        modules = [Linear(in_size, hidden_layer_sizes[0])]
        in_size = hidden_layer_sizes[0]

        if len(hidden_layer_sizes) == 1:
            layer_size = hidden_layer_sizes[0]
        else:
            for layer_size in hidden_layer_sizes[1:]:
                if batch_norm:
                    modules += [BatchNorm(in_size)]
                    # maybe add dropout
                    # modules += [Dropout()]
                modules += [ReLU()]
                modules += [Linear(in_size, layer_size)]
                in_size = layer_size

        if batch_norm:
            modules += [BatchNorm(layer_size)]
            # maybe add dropout
            # modules += [Dropout()]
        modules += [ReLU()]
        modules += [Linear(layer_size, out_size)]

    mlp = Sequential(*modules)

    return mlp


# ---------------------------------------------------------------------------
# Menagerie build/example-input glue (tiny config, random init).
# ---------------------------------------------------------------------------
def build_radargnn():
    cfg = GNNArchitectureConfig(
        node_feature_dimension=6,
        edge_feature_dimension=5,
        conv_layer_dimensions=[16, 16],
        classification_head_layer_dimensions=[16, 5],
        regression_head_layer_dimensions=[16, 4],
        initial_node_feature_embedding=True,
        initial_edge_feature_embedding=True,
        node_feature_embedding_layer_dimensions=[16, 16],
        edge_feature_embedding_layer_dimensions=[16, 16],
        conv_layer_type="MPNNConv",
    )
    return DetNetBasic(cfg)


def example_input_radargnn():
    # small radar-point-cloud graph: 10 nodes (6 initial node features, matching
    # RadarScenes-style x/y/vr/rcs/... point features), 20 directed edges (5 initial
    # edge features, matching relative-position/az/... edge encoding).
    n_nodes, n_edges = 10, 20
    x = torch.randn(n_nodes, 6)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 5)
    return x, edge_index, edge_attr


MENAGERIE_ENTRIES = [
    ("RadarGNN", "build_radargnn", "example_input_radargnn", 2023, "vendored-pytorch"),
]
