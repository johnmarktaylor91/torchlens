# SOURCE: vendored from fredjo89/heterogeneous-mpnn @ master
# https://raw.githubusercontent.com/fredjo89/heterogeneous-mpnn/master/models_HMPNN_ct.py
#
# Fredheim, Johannessen 2023 (arXiv:2307.13499) "Finding Money Launderers Using
# Heterogeneous Graph Neural Networks" -- a heterogeneous extension of MPNN (Message
# Passing Neural Network, Gilmer et al. 2017 "Neural Message Passing for Quantum
# Chemistry") for anti-money-laundering node classification on a real Norwegian bank
# transaction graph. For each node type, `HMPNN_ct_Layer` runs one edge-conditioned
# `torch_geometric.nn.NNConv` message-passing operator per incoming meta-step (i.e. per
# (src_type, relation, node_type) edge type), wrapped in `HeteroConv(aggr="sum")`; the
# per-meta-step message vectors are sigmoid-activated and concatenated ("ct" = concat),
# then linearly projected (+ sigmoid) to the node type's new representation -- the
# "HMPNN with HMCT aggregation" that gives the paper's method its name. Stacking these
# layers across the graph's node types (all node types updated in parallel at each depth,
# using the previous layer's representations as the next layer's `dim_in`) forms the full
# HMPNN_ct_1Layer..HMPNN_ct_4Layer models for binary node classification on one target
# node type.
#
# `create_dim_in`, `HMPNN_ct_Layer`, `HMPNN_ct_1Layer`, `HMPNN_ct_2Layer`,
# `HMPNN_ct_3Layer`, `HMPNN_ct_4Layer` are copied verbatim from the real
# `models_HMPNN_ct.py` (only whitespace/formatting is unchanged from source; no
# architectural edits). `HMPNN_ct_2Layer` is instantiated here as the representative
# multi-layer variant. The real model is driven from `utils.train_model`, which calls
# `model.forward(data.x_dict, data.edge_index_dict, data.edge_attr_dict)` on a
# `torch_geometric.data.HeteroData` graph loaded from the paper's real bank-transaction
# dataset; we reproduce that exact calling convention with a small synthetic 2-node-type,
# 2-edge-type `HeteroData` graph (same dict-of-tensors shape the real training loop feeds
# in) since the real dataset is not redistributable/available in this environment.

import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, NNConv


################################################################################################
def create_dim_in(data, dim):
    """
    Helper function that creates the "dim_in" dictionary used as input to HMPNN_ct_Layer().
    The keys of the dict are each node type in the graph, and the value is the number "dim" (same for all keys)
    Input:
        data: The graph object
        dim: An integer holding the representation-dimension for each node-types  (which is the same for all node-types)
    Output:
        dim_in: A dictionary that maps each node-type in the graph-object (data) to the same integer (dim)
    """
    dim_in = {}
    for node_type in data.node_types:
        dim_in[node_type] = dim
    return dim_in


################################################################################################


################################################################################################
class HMPNN_ct_Layer(torch.nn.Module):
    """
    This class creates the HMPNN_ct-model for a single node type in a single layer. Assembling multiple objects of this class (that depends on the number of layers in the model and nodes in the graph) will produce a full HMPNN_ct-model.
    In the case of a single-layer model that makes prediction of a single node-type, this will consist only of a single object of this class.

    An object of this class takes as input the graph and the node-representation vectors of all nodes of all node-types, and output the new node node-representations for one specific node-type (specified by the input-paramter node_type).
    The new node-representation is a result of having applied the operation we call "HMPNN with HMCT aggregation".
    This means to perform two main operators:
        1. For each meta_step ending with node_type, apply the MPNN operator. This will produce one vector for each meta-step, each of which can be seen as an incoming message to nodes of type node_type.
        2. Concatenate the message-vectors into one vector, and apply a linear transformation to the result. The output is the new representation of the nodes of type node_type

    Input:
        data: The graph object
        node_type: The node-type to which we will produce new node-representations
        dim_in: A dictionary holding the node representation-dimension of all node types (before applying this layer).
        dim_message: Determines the dimension of the vectors that is the output accross each meta-step (i.e. for each MPNN-operator). This dimension is the same for all meta-steps.
        dim_out: The dimension of the new node-representation for nodes of type node_type (the dimension of the output produced by this operator)

    (Additional) Class Variables:
        num_meta_steps: The number of meta-steps incoming to node_type
        message_passers: List of the MPNN-operators. Has the length num_meta_steps (one for each incoming meta-step)
        linear: THe linear layer that transforms the concatenated output-vectors from message_passers to the final output, which is the new representation of node_type (has dimension dim_out)

    """

    def __init__(
        self,
        data: torch_geometric.data.hetero_data.HeteroData,
        node_type: str = "indivi",
        dim_in: dict = {},
        dim_message: int = -1,
        dim_out: int = 1,
    ):
        super().__init__()

        # Initialize node_type, dim_in, dim_message, dim_out and num_meta_steps
        self.node_type = node_type
        self.dim_in = dim_in
        if len(self.dim_in) == 0:
            for nt in data.node_types:
                self.dim_in[nt] = data[nt].x.shape[
                    1
                ]  # If dim_in is not provided, we set it equal to the feature-dimension of each node_type  # noqa: E701
        self.dim_message = dim_message
        if self.dim_message == -1:
            self.dim_message = data[
                node_type
            ].x.shape[
                1
            ]  # if dim_message is not given, we set it equal to the number of features of the node_type that will receive messages  # noqa: E701
        self.dim_out = dim_out
        self.num_meta_steps = sum(
            [meta_step[2] == node_type for meta_step in data.edge_types]
        )  # num_meta_steps holds the number of meta-steps incoming to node_type

        # Initialize the MPNN-operators for each meta-step ending in node_type, and placing them in a list
        self.message_passers = torch.nn.ModuleList()
        for meta_step in data.edge_types:
            if meta_step[2] == self.node_type:
                num_edge_features = data[meta_step].edge_attr.shape[1]
                message_function = torch.nn.Linear(
                    num_edge_features, self.dim_in[meta_step[0]] * self.dim_message
                )
                mp = {}
                mp[meta_step] = NNConv(
                    (self.dim_in[meta_step[0]], self.dim_in[meta_step[2]]),
                    self.dim_message,
                    message_function,
                )
                self.message_passers.append(HeteroConv(mp, aggr="sum"))

        # Initialize the linear layer that the concatenated message vectors will be transformed by
        self.linear = torch.nn.Linear(len(self.message_passers) * self.dim_message, self.dim_out)

    def print_data(self):
        # A helper-function to inspect the value of the class-variables
        print(f"node_type: {self.node_type}")
        print(f"dim_in: {self.dim_in}")
        print(f"dim_message: {self.dim_message}")
        print(f"dim_out: {self.dim_out}")
        print(f"num_meta_steps: {self.num_meta_steps}")
        print(f"message_passers: {self.message_passers}")
        print(f"linear: {self.linear}")

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Apply message passing accross each meta-step and concatenate the results into a single tensor
        for i in range(self.num_meta_steps):
            msg = self.message_passers[i](x_dict, edge_index_dict, edge_attr_dict)[self.node_type]
            msg = torch.sigmoid(msg)
            if "msg_cat" not in locals():
                msg_cat = msg  # noqa: E701
            else:
                msg_cat = torch.cat((msg_cat, msg), 1)  # noqa: E701
        # Apply linear transformation, followed by the sigmoid function, and return
        return torch.sigmoid(self.linear(msg_cat))


################################################################################################


################################################################################################
class HMPNN_ct_2Layer(torch.nn.Module):
    """
    The class that defines the 2-layer HMPNN_ct-model used for binary node-classification on nodes of type node_type
    Input:
        data: The graph object
        node_type: The node_type to classify (predict which binary class each node of type node_type belongs to)
    """

    def __init__(
        self, data: torch_geometric.data.hetero_data.HeteroData, node_type: str = "indivi"
    ):
        super().__init__()

        self.node_type = node_type
        self.node_types = data.node_types  # All node types in the graph

        self.dim_message_layer_1 = 2  # Choice from thesis: 2
        self.dim_message_layer_2 = 10  # Choice from thesis (final layer): 10
        self.dim_out_layer_1 = 5  # Choice from thesis: 5
        self.dim_out = 1  # Specifies the final dimension of the representation vector for node_type (the final output of the class). Since this is used for binary classification, it is set to 1.

        # dim_in_2 is used as input when creating the second layer "hmct message passing operator".
        #   The variable is used to specify the representation/feature dimension of each node type in the graph after having been transformed by the first layer. This is required because we allow the representation dimension for a node type change from the dimension of its original feature vector after a message passing layer is applied.
        self.dim_in_2 = create_dim_in(data, self.dim_out_layer_1)

        # Creating the HMPNN_ct-operators for the first layer: one operator for each node_type in the graph.
        self.layer_1 = torch.nn.ModuleList()
        for nt in data.node_types:
            self.layer_1.append(
                HMPNN_ct_Layer(
                    data,
                    node_type=nt,
                    dim_message=self.dim_message_layer_1,
                    dim_out=self.dim_out_layer_1,
                )
            )

        # Creating the HMPNN_ct-operators for the output-layer: Only created for node_type (the node to make predictions on)
        self.layer_2 = HMPNN_ct_Layer(
            data,
            node_type=node_type,
            dim_in=self.dim_in_2,
            dim_message=self.dim_message_layer_2,
            dim_out=self.dim_out,
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict_updates = {}
        # Apply layer 1
        for node_update_fun in self.layer_1:
            x_dict_updates[node_update_fun.node_type] = node_update_fun.forward(
                x_dict, edge_index_dict, edge_attr_dict
            )
        # Apply output layer
        return self.layer_2.forward(x_dict_updates, edge_index_dict, edge_attr_dict)


################################################################################################


def _build_synthetic_hetero_data():
    """Small synthetic heterogeneous transaction-style graph: 2 node types
    ("indivi" = individual account nodes, "transfer" = transaction nodes), with an
    incoming meta-step for EVERY node type (as `HMPNN_ct_Layer` is created per node type
    in `HMPNN_ct_2Layer.__init__`, each node type needs >=1 incoming meta-step to have a
    defined `msg_cat`) -- mirroring the real bank-graph's multi-relation structure
    described in the paper. Matches the dict-of-tensors HeteroData shape
    `utils.train_model` feeds into `model.forward(data.x_dict, data.edge_index_dict,
    data.edge_attr_dict)`."""
    torch.manual_seed(0)
    data = HeteroData()
    n_indivi, n_transfer = 12, 15
    data["indivi"].x = torch.randn(n_indivi, 6)
    data["transfer"].x = torch.randn(n_transfer, 4)

    # meta-step 1: transfer -> indivi
    src1 = torch.randint(0, n_transfer, (20,))
    dst1 = torch.randint(0, n_indivi, (20,))
    data["transfer", "sends_to", "indivi"].edge_index = torch.stack([src1, dst1])
    data["transfer", "sends_to", "indivi"].edge_attr = torch.randn(20, 3)

    # meta-step 2: indivi -> indivi
    src2 = torch.randint(0, n_indivi, (18,))
    dst2 = torch.randint(0, n_indivi, (18,))
    data["indivi", "linked_to", "indivi"].edge_index = torch.stack([src2, dst2])
    data["indivi", "linked_to", "indivi"].edge_attr = torch.randn(18, 2)

    # meta-step 3: indivi -> transfer (so "transfer" also has an incoming meta-step)
    src3 = torch.randint(0, n_indivi, (16,))
    dst3 = torch.randint(0, n_transfer, (16,))
    data["indivi", "initiates", "transfer"].edge_index = torch.stack([src3, dst3])
    data["indivi", "initiates", "transfer"].edge_attr = torch.randn(16, 2)

    return data


def build_hmpnn_aml():
    data = _build_synthetic_hetero_data()
    return HMPNN_ct_2Layer(data, node_type="indivi")


def example_input_hmpnn_aml():
    data = _build_synthetic_hetero_data()
    return (data.x_dict, data.edge_index_dict, data.edge_attr_dict)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("HMPNN-AML", "build_hmpnn_aml", "example_input_hmpnn_aml", 2023, "vendored"),
]
