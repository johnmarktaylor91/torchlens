# FAITHFUL PORT of https://github.com/oxpig/DeLinker @ 7ba9334058095bcd60be2e05435576a52236c115
# (original framework: TensorFlow 1.x, tf.contrib.rnn.GRUCell + tf.Variable graph-session
# API)
#
# Transcribed from `DeLinker.py::DenseGGNNChemModel` (the real, only, model class in the
# repo), specifically the trainable neural encoder pipeline that is actually a "gated
# graph neural network" (GGNN, Li et al. 2016) VAE bottleneck:
#
#   - `get_node_embedding_state`            (node-symbol one-hot -> embedding lookup)
#   - `compute_final_node_representations_with_residual`
#         (`self.params["residual_connection_on"] == True` is DeLinker's actual
#          default -- this is the encoder path DeLinker trains with. Per edge type e,
#          message m_e = h @ W_e (+ bias_e); messages are aggregated over the dense
#          adjacency tensor adj[e] @ m_e and summed across edge types; the aggregate,
#          concatenated with residual connections from earlier propagation steps per
#          `params['residual_connections']`, drives a GRUCell update of the node
#          hidden state. Run for `num_timesteps` propagation steps, once for the
#          "in" (fragments) graph and once for the "out" (full molecule) graph,
#          exactly as `make_model()` calls it twice with `_encoder` weights shared
#          between both calls.)
#   - `compute_mean_and_logvariance`        (linear heads -> VAE mu/logvar, per-vertex
#                                             for the fragment graph, graph-mean-pooled
#                                             for the full-molecule graph)
#
# The autoregressive molecule-generation decoder (`generate_new_graphs`, RDKit valence
# checking, BFS path ordering, edge/edge-type/stop-node logit construction) is inference-
# time heuristic scaffolding wrapped AROUND this trained encoder -- not itself a
# trainable-NN forward pass -- and is out of scope for a menagerie trace of the real
# architecture; see `DeLinker.py::construct_logit_matrices`/`generate_new_graphs` in the
# original repo for that machinery. This module ports the trainable GGNN-VAE core.
#
# Default hyperparameters (DeLinker.py::DenseGGNNChemModel.default_params):
#   hidden_size=32, encoding_size=4, num_timesteps=7, use_edge_bias=True,
#   residual_connection_on=True, tie_fwd_bkwd=True (num_edge_types = 3 forward-only
#   bond types: single/double/triple, per utils.py::bond_dict),
#   residual_connections={2:[0], 4:[0,2], 6:[0,2,4], 8:[0,2,4,6], 10:[0,2,4,6,8],
#                          12:[0,2,4,6,8,10], 14:[0,2,4,6,8,10,12]}.
import torch
import torch.nn as nn

_DEFAULT_RESIDUAL_CONNECTIONS = {
    2: [0],
    4: [0, 2],
    6: [0, 2, 4],
    8: [0, 2, 4, 6],
    10: [0, 2, 4, 6, 8],
    12: [0, 2, 4, 6, 8, 10],
    14: [0, 2, 4, 6, 8, 10, 12],
}


class GGNNEncoder(nn.Module):
    """Ports `compute_final_node_representations_with_residual` for one scope
    (encoder weights are shared between the "in"/fragment and "out"/molecule graphs,
    matching DeLinker.py::make_model which calls the same `_encoder`-scoped weights
    on both `initial_state_in` and `initial_state_out`)."""

    def __init__(
        self,
        hidden_size,
        num_edge_types,
        num_timesteps,
        residual_connections=None,
        use_edge_bias=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types
        self.num_timesteps = num_timesteps
        self.residual_connections = residual_connections or _DEFAULT_RESIDUAL_CONNECTIONS
        self.use_edge_bias = use_edge_bias

        # weights['edge_weights_encoder'+str(iter_idx)]: [num_edge_types, h, h]
        self.edge_weights = nn.ParameterList(
            [
                nn.Parameter(_glorot_init((num_edge_types, hidden_size, hidden_size)))
                for _ in range(num_timesteps)
            ]
        )
        if use_edge_bias:
            self.edge_biases = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(num_edge_types, 1, hidden_size))
                    for _ in range(num_timesteps)
                ]
            )
        self.node_grus = nn.ModuleList(
            [
                nn.GRUCell(
                    hidden_size * (1 + len(self.residual_connections.get(i, []))),
                    hidden_size,
                )
                for i in range(num_timesteps)
            ]
        )

    def forward(self, h, adj):
        """
        h:   (B, V, hidden_size) initial node representations
        adj: (num_edge_types, B, V, V) dense adjacency (DeLinker transposes
             `adjacency_matrix_*` from [b,e,v,v] to [e,b,v,v] before this call)
        """
        b, v, hdim = h.shape
        h = h.reshape(b * v, hdim)
        all_hidden_states = [h]

        for iter_idx in range(self.num_timesteps):
            acts = None
            for edge_type in range(self.num_edge_types):
                m = h @ self.edge_weights[iter_idx][edge_type]  # (b*v, h)
                if self.use_edge_bias:
                    m = m + self.edge_biases[iter_idx][edge_type]
                m = m.reshape(b, v, hdim)
                contrib = torch.bmm(adj[edge_type], m)  # (b, v, h)
                acts = contrib if acts is None else acts + contrib
            acts = acts.reshape(b * v, hdim)

            residual_layers = self.residual_connections.get(iter_idx, [])
            residual_states = [all_hidden_states[i] for i in residual_layers]
            gru_input = torch.cat([acts] + residual_states, dim=1)

            h = self.node_grus[iter_idx](gru_input, h)
            all_hidden_states.append(h)

        return all_hidden_states[-1].reshape(b, v, hdim)


class DeLinkerGGNNVAE(nn.Module):
    """Ports the trainable GGNN-VAE core of DeLinker.py::DenseGGNNChemModel:
    node-symbol embedding -> shared-weight GGNN encoder (run once for the input
    fragments graph, once for the target full-molecule graph) -> mean/logvariance
    projection heads."""

    def __init__(
        self,
        num_symbols=8,
        hidden_size=32,
        encoding_size=4,
        num_timesteps=7,
        num_edge_types=3,
        use_edge_bias=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # weights["node_embedding"]: [num_symbols, hidden_size]
        self.node_embedding = nn.Parameter(_glorot_init((num_symbols, hidden_size)))

        self.encoder = GGNNEncoder(
            hidden_size=hidden_size,
            num_edge_types=num_edge_types,
            num_timesteps=num_timesteps,
            use_edge_bias=use_edge_bias,
        )

        # weights['mean_weights'], weights['variance_weights']: per-vertex heads
        # (used on the fragments/"in" graph representation)
        self.mean_head = nn.Linear(hidden_size, hidden_size)
        self.logvariance_head = nn.Linear(hidden_size, hidden_size)

        # weights['mean_weights_out'], weights['variance_weights_out']: graph-level
        # heads (used on the mean-pooled full-molecule/"out" graph representation)
        self.mean_head_out = nn.Linear(hidden_size, encoding_size)
        self.logvariance_head_out = nn.Linear(hidden_size, encoding_size)

    def get_node_embedding_state(self, node_symbols_onehot, node_mask):
        # `tf.argmax(one_hot_state, axis=2)` then embedding_lookup, masked
        idx = node_symbols_onehot.argmax(dim=2)
        state = self.node_embedding[idx]
        return state * node_mask.unsqueeze(-1)

    def forward(
        self,
        node_symbols_in,
        adjacency_in,
        node_mask_in,
        node_symbols_out,
        adjacency_out,
        node_mask_out,
    ):
        """
        node_symbols_in/out: (B, V, num_symbols) one-hot node-type labels for the
            fragments ("in") graph and the full target molecule ("out") graph
        adjacency_in/out:    (num_edge_types, B, V, V) dense per-edge-type adjacency
        node_mask_in/out:    (B, V) validity mask
        """
        h_in0 = self.get_node_embedding_state(node_symbols_in, node_mask_in)
        h_out0 = self.get_node_embedding_state(node_symbols_out, node_mask_out)

        h_in = self.encoder(h_in0, adjacency_in)
        h_out = self.encoder(h_out0, adjacency_out)

        # PER-VERTEX encoding of the fragments graph
        mean = self.mean_head(h_in)
        logvariance = self.logvariance_head(h_in)

        # AVERAGE (graph-level) encoding of the full-molecule graph
        mask_out = node_mask_out.unsqueeze(-1)
        avg_h_out = (h_out * mask_out).sum(dim=1) / mask_out.sum(dim=1).clamp_min(1e-7)
        mean_out = self.mean_head_out(avg_h_out)
        logvariance_out = self.logvariance_head_out(avg_h_out)

        return mean, logvariance, mean_out, logvariance_out


def _glorot_init(shape):
    fan_in, fan_out = shape[-2], shape[-1]
    limit = (6.0 / (fan_in + fan_out)) ** 0.5
    return torch.empty(shape).uniform_(-limit, limit)


def build_delinker():
    return DeLinkerGGNNVAE(
        num_symbols=8,
        hidden_size=16,
        encoding_size=4,
        num_timesteps=3,
        num_edge_types=3,
        use_edge_bias=True,
    ).eval()


def example_input_delinker():
    torch.manual_seed(0)
    b, v, num_symbols, num_edge_types = 1, 6, 8, 3

    def one_hot_graph():
        idx = torch.randint(0, num_symbols, (b, v))
        onehot = torch.zeros(b, v, num_symbols)
        onehot.scatter_(2, idx.unsqueeze(-1), 1.0)
        node_mask = torch.ones(b, v)
        adjacency = torch.rand(num_edge_types, b, v, v)
        return onehot, adjacency, node_mask

    node_symbols_in, adjacency_in, node_mask_in = one_hot_graph()
    node_symbols_out, adjacency_out, node_mask_out = one_hot_graph()
    return (
        node_symbols_in,
        adjacency_in,
        node_mask_in,
        node_symbols_out,
        adjacency_out,
        node_mask_out,
    )


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeLinker", build_delinker, example_input_delinker, 2020, "SOURCE_AVAILABLE"),
]
