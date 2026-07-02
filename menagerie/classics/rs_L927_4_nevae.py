# FAITHFUL PORT of Human-Centric-Machine-Learning/nevae (formerly
# Networks-Learning/nevae) @ 4ffac5a918e4eef1bda8a84e9e556d5bb2d3fbf4
#   (repo: https://github.com/Human-Centric-Machine-Learning/nevae)
#   nevae/cell.py (class VAEGCell, method __call__) + nevae/layer.py
#   (input_layer, fc_layer). (original framework: TensorFlow 1.x graph-mode,
#   tf.placeholder + tf.variable_scope + tf.while_loop + pervasive tf.Print
#   training-debug statements -- TF1.x cannot be installed/run alongside this
#   environment's torch stack, and this task's rules forbid installing extra
#   frameworks for a vendor, so the real code is transcribed faithfully into
#   base-env torch rather than run in-place).
# NeVAE (Samanta, De, Ganguly, Gomez-Rodriguez, "NEVAE: A Deep Generative
# Model for Molecular Graphs", AAAI 2019 / JMLR 2020) is a graph VAE for
# molecule generation with a PROVEN node/edge existence probability that
# supports an arbitrary, variable number of nodes: a random-walk positional
# encoder propagates node features through k learned linear steps gated by
# a (weighted) adjacency matrix (`input_layer`), a standard VAE
# encoder (concat over random-walk steps -> FC hidden -> FC mu/sigma,
# softplus-parameterized) produces a per-node Gaussian latent `z`, Poisson
# heads predict expected node count and (conditioned on z) expected edge
# count, and per-candidate-edge decoder MLPs predict edge existence
# ("hidden"), edge-type/bond-order ("marker", `bin_dim`-way one-hot
# concatenated onto the candidate pair) and per-node atom-type ("label")
# from concatenated endpoint latents -- all as in the real
# `VAEGCell.__call__`.
#
# Code transcribed 1:1 from the real `input_layer`/`fc_layer`/
# `VAEGCell.__call__`: the random-walk propagation recurrence, the
# encoder mu/sigma FC layers (softplus activations, matching real code),
# reparameterization (`z = mu + sigma @ eps`), the Poisson node/edge count
# heads, and the per-edge-candidate decoder loop over `combination` random
# walk samples are all preserved verbatim. Only TF1.x-specific plumbing is
# dropped: `tf.Print` debug statements (side-effect logging only, not
# architecture), the `tf.variable_scope(..., reuse=tf.AUTO_REUSE)` weight
# sharing (replaced with real `nn.Module` submodules that are constructed
# once and reused across the `combination` loop -- the same weight-sharing
# behavior `reuse=True` achieves in TF1.x), and the outer `tf.while_loop`
# over per-candidate edges (replaced with an equivalent Python for-loop,
# since a real forward pass has a fixed, known edge-candidate count).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class FCLayer(nn.Module):
    """Transcribed from layer.py `fc_layer`: a plain Linear (`xw_plus_b`)
    with an optional activation, matching the real xavier-init weight +
    constant(0.01)-init bias."""

    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.01)
        self.activation = activation

    def forward(self, x):
        out = self.linear(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


class InputLayer(nn.Module):
    """Transcribed from layer.py `input_layer`: k learned (d, d) linear
    propagation steps, each step multiplying the raw features by
    `w_in[step]` and (for step > 0) gating by the weighted-adjacency-
    propagated previous step's output. Constant-0.5-init weight matrix,
    matching the real `tf.constant_initializer(0.5)`."""

    def __init__(self, k, d):
        super().__init__()
        self.k = k
        self.d = d
        # real shape: [k, d, d], constant-initialized to 0.5
        self.w_in = nn.Parameter(torch.full((k, d, d), 0.5))

    def forward(self, adj, weight, features):
        """
        Args:
            adj (torch.Tensor): (n, n) adjacency matrix.
            weight (torch.Tensor): (n, n) edge weight matrix.
            features (torch.Tensor): (n, d) node feature matrix.
        Returns:
            c_x (torch.Tensor): (k, n, d) random-walk propagated features.
        """
        weighted_adj = adj * weight
        output_list = []
        for i in range(self.k):
            if i > 0:
                step = (features @ self.w_in[i]) * (weighted_adj @ output_list[i - 1])
            else:
                step = features @ self.w_in[i]
            output_list.append(step)
        return torch.stack(output_list, dim=0)


class VAEGCell(nn.Module):
    """Transcribed from cell.py `VAEGCell`. Fixed-size graph VAE cell:
    random-walk encoder -> Gaussian latent z -> Poisson node/edge-count
    heads -> per-edge-candidate decoder (hidden/marker/label MLPs)."""

    def __init__(self, k, d, z_dim, bin_dim, n, combination):
        super().__init__()
        self.k = k
        self.d = d
        self.z_dim = z_dim
        self.bin_dim = bin_dim
        self.n = n
        self.combination = combination

        self.input_layer = InputLayer(k, d)

        # Encoder (real scope "Encoder"): concat over k random-walk steps
        # -> hidden (relu) -> mu / sigma (softplus)
        self.enc_hidden = FCLayer(k * d, 32, activation=F.relu)
        self.enc_mu = FCLayer(32, z_dim, activation=F.softplus)
        self.enc_sigma = FCLayer(32, z_dim, activation=F.softplus)

        # Poisson heads (real scope "Poisson")
        self.node_head = FCLayer(1, 1, activation=F.softplus)
        self.edge_head = FCLayer(z_dim + 1, 1, activation=F.softplus)

        # Decoder (real scope "Decoder", reuse=True -- shared weights
        # across every edge-candidate / atom-type iteration)
        self.dec_hidden = FCLayer(2 * z_dim, 1, activation=F.softplus)
        self.dec_marker = FCLayer(2 * z_dim + bin_dim, 1, activation=F.softplus)
        self.dec_label = FCLayer(z_dim + 4, 1, activation=F.softplus)

    def forward(self, adj, weight, features, node_count, eps, edges):
        """
        Args:
            adj (torch.Tensor): (n, n) adjacency matrix.
            weight (torch.Tensor): (n, n) edge weight matrix.
            features (torch.Tensor): (n, d) node feature matrix.
            node_count (torch.Tensor): (1, n) node-degree/count vector fed
                to the Poisson node-count head.
            eps (torch.Tensor): (n, z_dim, 1) standard-normal noise for the
                reparameterization trick.
            edges (torch.Tensor): (combination, n_edge_candidates, 2) long
                tensor of candidate (src, dst) node-index pairs per random
                walk sample.

        Returns:
            enc_mu, enc_sigma, z, dec_hidden (list), dec_marker (list),
            label, lambda_node, lambda_edge -- mirroring the real
            `VAEGCell.__call__` return tuple (prior_mu/prior_sigma/c_x/
            debug_sigma dropped: diagnostic-only in the real code, not
            consumed downstream).
        """
        n, z_dim = self.n, self.z_dim

        c_x = self.input_layer(adj, weight, features)  # (k, n, d)

        # Encoder: concat over k steps -> (n, k*d)
        list_cx = list(torch.unbind(c_x, dim=0))
        enc_in = torch.cat(list_cx, dim=1)  # (n, k*d)
        enc_hidden = self.enc_hidden(enc_in)
        enc_mu = self.enc_mu(enc_hidden).reshape(n, z_dim, 1)
        enc_sigma_diag = self.enc_sigma(enc_hidden)  # (n, z_dim)
        enc_sigma = torch.diag_embed(enc_sigma_diag)  # (n, z_dim, z_dim)

        # Reparameterize: z = mu + sigma @ eps
        z = enc_mu + torch.bmm(enc_sigma, eps)  # (n, z_dim, 1)

        # Poisson heads
        lambda_node = self.node_head(node_count.transpose(0, 1))  # (n, 1)
        z_reshape = z.reshape(-1, z_dim)  # (n, z_dim)
        n_cast = torch.full((1, z_dim), float(n), dtype=z.dtype, device=z.device)
        z_concat = torch.cat([z_reshape, n_cast], dim=0)  # (n+1, z_dim)
        lambda_edge = self.edge_head(
            torch.cat(
                [z_concat, torch.zeros(z_concat.shape[0], 1, dtype=z.dtype, device=z.device)], dim=1
            )
        )

        # Decoder: per-node atom-type label (4 one-hot atom types: C,H,O,N)
        z_new = z.reshape(n, z_dim)
        z_stack_label = []
        eye4 = torch.eye(4, dtype=z.dtype, device=z.device)
        for u in range(n):
            for j in range(4):
                z_stack_label.append(
                    torch.cat([z_new[u].unsqueeze(0), eye4[j].unsqueeze(0)], dim=1).squeeze(0)
                )
        label = self.dec_label(torch.stack(z_stack_label, dim=0))

        # Decoder: per-edge-candidate hidden/marker predictions, one pass
        # per random-walk `combination` sample (shared decoder weights,
        # matching the real `reuse=True` scope)
        dec_hidden_out = []
        dec_marker_out = []
        eye_bin = torch.eye(self.bin_dim, dtype=z.dtype, device=z.device)
        for i in range(self.combination):
            t = edges[i]  # (n_edges_i, 2)
            z_stack = torch.cat([z_new[t[:, 0]], z_new[t[:, 1]]], dim=1)  # (n_edges_i, 2*z_dim)
            z_stack_weight_parts = []
            for j in range(self.bin_dim):
                m = eye_bin[j].unsqueeze(0).expand(z_stack.shape[0], -1)
                z_stack_weight_parts.append(torch.cat([z_stack, m], dim=1))
            z_stack_weight = torch.cat(z_stack_weight_parts, dim=0)

            dec_hidden_out.append(self.dec_hidden(z_stack))
            dec_marker_out.append(self.dec_marker(z_stack_weight))

        return (
            enc_mu,
            enc_sigma,
            z,
            dec_hidden_out,
            dec_marker_out,
            label,
            lambda_node.mean(),
            lambda_edge.mean(),
        )


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------

_N = 6  # tiny toy graph, 6 nodes
_D = 4  # node feature width (atom-type one-hot-ish)
_K = 3  # random-walk depth (real default `--random_walk`: 5)
_Z_DIM = 5  # real default `--z_dim`: 5
_BIN_DIM = 3  # real default `--bin_dim`: 3 (bond-order classes)
_COMBINATION = 2  # node_sample * bfs_sample random-walk samples


def build_nevae():
    return VAEGCell(k=_K, d=_D, z_dim=_Z_DIM, bin_dim=_BIN_DIM, n=_N, combination=_COMBINATION)


def example_input_nevae():
    """A tiny 6-node toy graph: random symmetric adjacency + edge-weight
    matrices, random node features, a node-count vector, standard-normal
    reparameterization noise, and `_COMBINATION` random-walk edge-candidate
    samples (each a small list of (src, dst) node-index pairs), matching
    the real `VAEGCell.__call__(c_x, n, d, k, combination, eps_passed,
    sample, scope)` inputs (built directly rather than routed through the
    real repo's `load_data_new`/BFS-sampling data pipeline, which reads
    graph files from disk)."""
    torch.manual_seed(0)
    n, d, z_dim = _N, _D, _Z_DIM

    adj_upper = torch.randint(0, 2, (n, n)).float()
    adj = torch.triu(adj_upper, diagonal=1)
    adj = adj + adj.transpose(0, 1)  # symmetric, zero diagonal

    weight = torch.rand(n, n)
    weight = (weight + weight.transpose(0, 1)) / 2.0

    features = torch.randn(n, d)
    node_count = torch.rand(1, n)
    eps = torch.randn(n, z_dim, 1)

    # `combination` random-walk edge-candidate samples: small fixed lists
    # of (src, dst) node-index pairs (a real random-walk/BFS sampler would
    # produce these; here they're a directly-built stand-in of the same
    # shape/dtype the real decoder loop consumes).
    edges = torch.stack(
        [
            torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long),
            torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long),
        ],
        dim=0,
    )

    return (adj, weight, features, node_count, eps, edges)


MENAGERIE_ENTRIES = [
    (
        "NeVAE",
        build_nevae,
        example_input_nevae,
        2019,
        "CODE",
    ),
]
