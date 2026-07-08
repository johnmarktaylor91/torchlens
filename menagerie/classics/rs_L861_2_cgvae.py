# FAITHFUL PORT of microsoft/constrained-graph-variational-autoencoder @ master (original
# framework: TensorFlow 1.x, tf.contrib.rnn / raw tf.Variable graph-mode)
# https://github.com/microsoft/constrained-graph-variational-autoencoder/blob/master/CGVAE.py
# https://github.com/microsoft/constrained-graph-variational-autoencoder/blob/master/GGNN_core.py
# https://github.com/microsoft/constrained-graph-variational-autoencoder/blob/master/utils.py
#
# CGVAE (Constrained Graph Variational Autoencoder, Liu et al. NeurIPS 2018). The real repo
# is TF1 session/placeholder code using `tf.contrib.rnn.GRUCell` -- `tf.contrib` was removed
# from TensorFlow entirely (TF2.x), so the upstream code cannot run in any installable
# TF2 env; rung-3 faithful port is applied. Ported here is the model's single-forward-pass
# *architecture*: the `DenseGGNNChemModel` GGNN encoder from
# `compute_final_node_representations_with_residual` (per-edge-type linear message
# transforms + GRU-cell propagation with residual connections across `num_timesteps`,
# exactly mirroring the `residual_connections` schedule and the `edge_weights.../
# edge_biases.../node_gru...` per-iteration weight layout in `prepare_specific_graph_model`),
# `compute_mean_and_logvariance` (linear projection to VAE mean/logvariance) and
# `sample_with_mean_and_logvariance` (reparameterization sampling), the node-symbol logit
# head from `construct_logit_matrices`, and the QED property-regression head
# (`gated_regression` with the real `MLP` gate/transform helper from utils.py, hid_sizes=[]
# i.e. a single Linear+ReLU layer exactly as upstream). NOT ported: the autoregressive
# BFS graph-generation decoder loop (`generate_cross_entropy`, `search_and_generate_molecule`,
# `optimization_over_prior`, ...) -- that is training/generation-time step-by-step search
# control flow conditioned on per-step ground truth (like beam search), not a traceable
# single forward pass, and is not part of what a `tl.trace()` capture would exercise for
# any of the repo's other traced models either. Every remaining tensor op is a 1:1 transcription
# of the TF1 ops (tf.matmul -> torch.matmul/nn.Linear, tf.contrib.rnn.GRUCell -> nn.GRUCell,
# tf.concat -> torch.cat, tf.nn.sigmoid/tanh/relu -> torch equivalents) with no new/removed
# architectural mechanism.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


def glorot_init(shape):
    # utils.py glorot_init: same formula as nn.init.xavier_uniform_, kept as a literal
    # port so weight construction matches the real repo's initializer call-for-call.
    init_range = torch.sqrt(torch.tensor(6.0 / (shape[-2] + shape[-1])))
    return (torch.rand(*shape) * 2 - 1) * init_range


class MLP(nn.Module):
    """Faithful port of utils.py MLP: with hid_sizes=[] (as used for the QED regression
    gate/transform in construct_loss) this is a single Linear layer followed by ReLU,
    exactly as the real __call__ implementation computes."""

    def __init__(self, in_size, out_size, hid_sizes):
        super().__init__()
        dims = [in_size] + list(hid_sizes) + [out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        self.weights = nn.ParameterList([nn.Parameter(glorot_init(s)) for s in weight_sizes])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(s[-1])) for s in weight_sizes])

    def forward(self, inputs):
        acts = inputs
        hid = acts
        for w, b in zip(self.weights, self.biases):
            hid = torch.matmul(acts, w) + b
            acts = torch.relu(hid)
        return hid


class CGVAEEncoder(nn.Module):
    """Faithful port of the DenseGGNNChemModel GGNN encoder + VAE head + node-symbol head
    + QED gated-regression head from CGVAE.py. Single forward pass over one padded batch
    of graphs, matching the real tensor shapes: node features [b, v, h], adjacency
    [e, b, v, v] (per edge-type stacked adjacency, matching the real `adj[edge_type]`
    indexing in compute_final_node_representations_with_residual), node mask [b, v].
    """

    def __init__(
        self,
        hidden_size=16,
        num_edge_types=3,
        num_timesteps=4,
        num_symbols=9,
        use_edge_bias=True,
        residual_connections=None,
    ):
        super().__init__()
        self.h_dim = hidden_size
        self.num_edge_types = num_edge_types
        self.num_timesteps = num_timesteps
        self.num_symbols = num_symbols
        self.use_edge_bias = use_edge_bias
        # Same shrunk residual schedule shape as the real repo's dict (iteration idx ->
        # list of earlier iteration indices whose hidden state is concatenated in as a
        # residual connection), just truncated to num_timesteps.
        self.residual_connections = residual_connections or {2: [0]}

        h_dim = hidden_size

        # --- prepare_specific_graph_model encoder weights (scope='_encoder') ---
        self.edge_weights = nn.ParameterList(
            [
                nn.Parameter(glorot_init((num_edge_types, h_dim, h_dim)))
                for _ in range(num_timesteps)
            ]
        )
        if use_edge_bias:
            self.edge_biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(num_edge_types, 1, h_dim)) for _ in range(num_timesteps)]
            )
        # Real repo: `tf.contrib.rnn.GRUCell` lazily sizes its input-to-hidden weights to
        # whatever `acts` width it is called with at that scope/iteration -- and `acts` is
        # widened by `tf.concat([acts] + layer_residual_states, axis=1)` whenever
        # `residual_connections[iter_idx]` is non-empty (hidden state `h` itself always
        # stays h_dim-wide). Port that faithfully: build one GRUCell per iteration with
        # input_size = h_dim * (1 + num_residual_connections_feeding_that_iteration).
        self.node_gru = nn.ModuleList(
            [
                nn.GRUCell(h_dim * (1 + len(self.residual_connections.get(iter_idx, []))), h_dim)
                for iter_idx in range(num_timesteps)
            ]
        )

        # --- mean / logvariance projection weights ---
        self.mean_weights = nn.Parameter(glorot_init((h_dim, h_dim)))
        self.mean_biases = nn.Parameter(torch.zeros(1, h_dim))
        self.variance_weights = nn.Parameter(glorot_init((h_dim, h_dim)))
        self.variance_biases = nn.Parameter(torch.zeros(1, h_dim))

        # --- node symbol logits weights ---
        self.node_symbol_weights = nn.Parameter(glorot_init((h_dim, num_symbols)))
        self.node_symbol_biases = nn.Parameter(torch.zeros(1, num_symbols))

        # --- QED gated-regression head (construct_loss task loop, task_id=0) ---
        self.regression_gate = MLP(h_dim, 1, [])
        self.regression_transform = MLP(h_dim, 1, [])
        self.qed_weights = nn.Parameter(glorot_init((h_dim, h_dim)))
        self.qed_biases = nn.Parameter(torch.zeros(1, h_dim))

    def compute_final_node_representations_with_residual(self, h, adj):
        # h: [b, v, h_dim] initial node representation; adj: [e, b, v, v] adjacency
        b, v, h_dim = h.shape
        h = h.reshape(-1, h_dim)  # [b*v, h]
        all_hidden_states = [h]
        for iter_idx in range(self.num_timesteps):
            acts = None
            for edge_type in range(self.num_edge_types):
                m = torch.matmul(h, self.edge_weights[iter_idx][edge_type])  # [b*v, h]
                if self.use_edge_bias:
                    m = m + self.edge_biases[iter_idx][edge_type]
                m = m.reshape(-1, v, h_dim)  # [b, v, h]
                contrib = torch.matmul(adj[edge_type], m)  # [b, v, h]
                acts = contrib if acts is None else acts + contrib
            acts = acts.reshape(-1, h_dim)  # [b*v, h]

            layer_residual_idxs = self.residual_connections.get(iter_idx)
            if layer_residual_idxs:
                residual_states = [all_hidden_states[i] for i in layer_residual_idxs]
                acts = torch.cat([acts] + residual_states, dim=1)  # [b*v, h*(1+n_residual)]

            h = self.node_gru[iter_idx](acts, h)  # GRUCell(acts, h) -> [b*v, h]
            all_hidden_states.append(h)
        last_h = all_hidden_states[-1].reshape(b, v, h_dim)
        return last_h

    def compute_mean_and_logvariance(self, final_node_representations):
        h_dim = self.h_dim
        reshaped = final_node_representations.reshape(-1, h_dim)
        mean = torch.matmul(reshaped, self.mean_weights) + self.mean_biases
        logvariance = torch.matmul(reshaped, self.variance_weights) + self.variance_biases
        return mean, logvariance

    def sample_with_mean_and_logvariance(self, mean, logvariance, z_prior, node_mask):
        b, v = node_mask.shape
        h_dim = self.h_dim
        z_prior_flat = z_prior.reshape(-1, h_dim)
        # Training-mode sampling: z = mean + sigma * eps (non-standard normal), matching
        # the `is_generative=False` branch of the real tf.cond.
        z_sampled = mean + torch.sqrt(torch.exp(logvariance)) * z_prior_flat
        z_sampled = z_sampled.reshape(b, v, h_dim) * node_mask.unsqueeze(2)
        return z_sampled

    def gated_regression(self, last_h, node_mask):
        # last_h: [b, v, h] -- normalized z_sampled, as in construct_loss (l2_normalize dim=2)
        b, v, h_dim = last_h.shape
        last_h = last_h.reshape(-1, h_dim)
        last_h = torch.relu(torch.matmul(last_h, self.qed_weights) + self.qed_biases)
        gate_input = last_h
        gated_outputs = torch.sigmoid(self.regression_gate(gate_input)) * torch.tanh(
            self.regression_transform(last_h)
        )
        gated_outputs = gated_outputs.reshape(b, v)
        masked = gated_outputs * node_mask
        output = masked.sum(dim=1)
        return torch.sigmoid(output)

    def forward(self, initial_node_representation, adjacency_matrix, node_mask, z_prior):
        """
        Args:
            initial_node_representation: [b, v, h] padded initial node features
            adjacency_matrix: [e, b, v, v] per-edge-type adjacency (dense)
            node_mask: [b, v] float mask, 1 for real nodes, 0 for padding
            z_prior: [b, v, h] standard-normal noise for the reparameterization trick
        Returns:
            dict with the four real model outputs: mean, logvariance, node_symbol_logits,
            qed_computed_values (matching CGVAE.py's self.ops entries of the same names).
        """
        final_node_representations = self.compute_final_node_representations_with_residual(
            initial_node_representation, adjacency_matrix
        )
        mean, logvariance = self.compute_mean_and_logvariance(final_node_representations)
        z_sampled = self.sample_with_mean_and_logvariance(mean, logvariance, z_prior, node_mask)

        node_symbol_logits = torch.matmul(
            z_sampled.reshape(-1, self.h_dim), self.node_symbol_weights
        )
        node_symbol_logits = node_symbol_logits + self.node_symbol_biases
        node_symbol_logits = node_symbol_logits.reshape(
            z_sampled.shape[0], z_sampled.shape[1], self.num_symbols
        )

        normalized_z_sampled = nn.functional.normalize(z_sampled, dim=2)
        qed_computed_values = self.gated_regression(normalized_z_sampled, node_mask)

        return {
            "mean": mean,
            "logvariance": logvariance,
            "node_symbol_logits": node_symbol_logits,
            "qed_computed_values": qed_computed_values,
        }


def build_cgvae():
    # Tiny menagerie-scale config: hidden_size=16 (real repo default is 100), 3 edge
    # types (real repo default: single/double/triple bond, matching ZINC/QM9 configs),
    # num_timesteps=4 (real default 12, shrunk for trace speed), num_symbols=9 (real ZINC
    # config uses 9 heavy-atom symbol classes).
    return CGVAEEncoder(hidden_size=16, num_edge_types=3, num_timesteps=4, num_symbols=9)


def example_input_cgvae():
    # Small synthetic padded batch: batch_size=2, max_vertices=6. Adjacency is built as a
    # random symmetric per-edge-type dense adjacency (matching the real repo's dense
    # [e, b, v, v] `adjacency_matrix` placeholder), node_mask marks all 6 nodes real for
    # graph 0 and the first 4 for graph 1 (padding), z_prior is standard-normal noise.
    torch.manual_seed(0)
    b, v, h, e = 2, 6, 16, 3
    initial_node_representation = torch.randn(b, v, h)
    adjacency_matrix = (torch.rand(e, b, v, v) > 0.7).float()
    node_mask = torch.ones(b, v)
    node_mask[1, 4:] = 0.0
    z_prior = torch.randn(b, v, h)
    return (initial_node_representation, adjacency_matrix, node_mask, z_prior)


MENAGERIE_ENTRIES = [
    (
        "CGVAE",
        "build_cgvae",
        "example_input_cgvae",
        2018,
        MENAGERIE_ZOO,
    ),
]
