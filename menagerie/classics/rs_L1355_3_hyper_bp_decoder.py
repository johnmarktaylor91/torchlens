# FAITHFUL PORT of https://github.com/facebookresearch/HyperNetworkDecoder @ master
#   (original framework: TensorFlow 1.x graph-mode -- `tf.placeholder`/`tf.get_variable`/
#   `tf.contrib.layers.xavier_initializer`; not runnable as-is, and not torch)
#
# Ported file: scripts/model.py -> `HyperNetworkDecoder.__init__` (forward graph),
# `f_hyper`, `mlp_vn`, `arc_tanh_like`, `generate_graph_matrix`. Nachmani & Wolf,
# "Hyper-Graph-Network Decoders for Block Codes" (NeurIPS 2019).
#
# HyperBP is a hypernetwork-conditioned belief-propagation decoder for linear block
# codes (BCH/LDPC/Polar): a small MLP ("f") consumes the current odd(variable)-node
# log-likelihoods and, at every BP iteration, generates the weights of the message
# tensor used by the variable-node update MLP ("g" / `mlp_vn`) -- i.e. the BP
# message-passing weights are *hypernetwork-generated per-iteration* rather than
# static/shared, which is the paper's core contribution over a plain "weighted BP"
# unrolled network. The check-node (odd->even) update itself uses a fixed closed-form
# tanh/arctanh-like message-combination rule (standard soft-BP), gated by the code's
# Tanner-graph connectivity matrices (`W_odd2even`, `W_even2odd`, `W_output`, etc.),
# which are derived once from the code's parity-check matrix via `generate_graph_matrix`
# (transcribed verbatim below, index-for-index, from the real repo).
#
# Faithful-port notes (no architectural changes, only TF1->torch mechanical transcription):
#   - `tf.placeholder`/`tf.Variable`/`tf.get_variable` -> plain torch Tensors / nn.Parameter.
#   - `tf.einsum('aij,bjk->abik', ...)` / `tf.einsum('aij,ajb->aib', ...)` ->
#     `torch.einsum` with the identical subscripts (torch supports the same einsum
#     grammar).
#   - `tf.contrib.layers.xavier_initializer()` -> `nn.init.xavier_uniform_`.
#   - The repo trains `W_odd2even_graphnn_var`/`W_output_var` as free trainable
#     variables masked by the fixed 0/1 connectivity matrices every forward pass
#     (`tf.multiply(W_fixed, W_var)`); this masked-parameter pattern is preserved
#     exactly (`nn.Parameter` initialized to the fixed connectivity matrix, re-masked
#     by the fixed matrix on every forward call, matching the original graph op).
#   - The repo hardcodes `args.batch_size` as a static graph placeholder dimension;
#     the ported module instead reads batch size dynamically from the input tensor
#     (this is the standard "static placeholder -> dynamic first dim" TF1->torch
#     shape change, not an architectural change -- every tensor op and its axis
#     semantics are otherwise identical).
#   - The repo's training/optimizer code (`tf.train.AdamOptimizer`, loss reduction)
#     is not part of the network architecture and is dropped; only the forward
#     decoder graph (`y_output`, i.e. `out_i` of the final iteration) is ported.
#
# Forward flow (see model.py `__init__`, lines ~39-81):
#   x_hv = log((1 + prod_j tanh(0.5*clip(x*W_input))) / (1 - prod_j tanh(...)))   # LLR init
#   for each of (num_hidden_layers - 1) BP iterations:
#       fw1, fw2 = f_hyper(|x_hv|)                        # hypernetwork generates g's weights
#       x_hv_c = einsum('aij,bjk->abik', x_hv[:, None, :], W_odd2even_graphnn * W_var).squeeze(2)
#       x_all = concat([x_hv_c, (x @ W_skipconn2even)[..., None]], dim=2)
#       x_hp = mlp_vn(x_all, {h1: fw1, out: fw2})          # hypernet-weighted VN update
#       x_hv_c = tile(x_hp) * W_even2odd.flatten()          # check-node message
#       x_hv = 2 * arc_tanh_like(prod_j x_hv_c, order=1005)
#       out_i = x + (x_hv @ (W_output * W_output_var))      # marginalization / output LLR
#   return out_i

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def generate_graph_matrix(parity_check_matrix: np.ndarray):
    """Faithful port of HyperNetworkDecoder.generate_graph_matrix (model.py).
    Builds the fixed 0/1 Tanner-graph connectivity matrices from a code's binary
    parity-check matrix H (shape [num_checks, code_n]), index-for-index identical
    to the original numpy logic."""
    H = parity_check_matrix.astype(np.float32)
    code_n = H.shape[1]
    n_odd = int(np.sum(H))
    n_even = n_odd

    W_input = np.zeros((code_n, n_odd), dtype=np.float32)
    W_odd2even = np.zeros((n_odd, n_even), dtype=np.float32)
    W_odd2even_graphnn = np.zeros((n_odd, n_even, n_odd), dtype=np.float32)
    W_skipconn2even = np.zeros((code_n, n_even), dtype=np.float32)
    W_even2odd = np.zeros((n_even, n_odd), dtype=np.float32)
    W_output = np.zeros((n_odd, code_n), dtype=np.float32)

    # init W_input
    k = 0
    for i in range(0, H.shape[0], 1):
        for j in range(0, H.shape[1], 1):
            if H[i, j] == 1:
                vec = H[i, :].copy()
                vec[j] = 0
                W_input[:, k] = vec
                k += 1

    # init W_odd2even & W_skipconn2even
    k = 0
    for j in range(0, H.shape[1], 1):
        for i in range(0, H.shape[0], 1):
            if H[i, j] == 1:
                num_of_conn = np.sum(H[:, j])
                idx = np.argwhere(H[:, j] == 1)
                for ll in range(0, int(num_of_conn), 1):
                    vec_tmp = np.zeros((n_odd,), dtype=np.float32)
                    for r in range(0, H.shape[0], 1):
                        if H[r, j] == 1 and idx[ll][0] != r:
                            idx_vec = np.cumsum(H[r, 0 : j + 1])[-1] - 1
                            vec_tmp[int(idx_vec + np.sum(H[:r, :]))] = 1.0
                    W_odd2even[:, k] = vec_tmp.transpose()
                    k += 1
                break

    # init W_odd2even_graphnn
    for j in range(0, W_odd2even.shape[1], 1):
        for i in range(0, W_odd2even.shape[0], 1):
            W_odd2even_graphnn[j, i, i] = W_odd2even[i, j]

    # init W_even2odd, W_skipconn2even & W_output
    k, m = 0, 0
    for j in range(0, H.shape[1], 1):
        for i in range(0, H.shape[0], 1):
            if H[i, j] == 1:
                idx_row = np.cumsum(H[i, 0 : j + 1])[-1] - 1
                till_d_c = np.sum(H[:i, :])
                this_d_c = np.sum(H[: (i + 1), :])
                W_even2odd[k, int(till_d_c) : int(this_d_c)] = 1.0
                W_even2odd[k, int(till_d_c + idx_row)] = 0.0

                W_skipconn2even[j, k] = 1.0

                idx_row = np.cumsum(H[i, 0 : j + 1])[-1] - 1
                till_d_c = np.sum(H[:i, :])
                W_output[int(till_d_c + idx_row), m] = 1.0

                k += 1
        m += 1

    return {
        "W_input": W_input,
        "W_odd2even": W_odd2even,
        "W_odd2even_graphnn": W_odd2even_graphnn,
        "W_skipconn2even": W_skipconn2even,
        "W_even2odd": W_even2odd,
        "W_output": W_output,
        "n_odd": n_odd,
        "n_even": n_even,
        "code_n": code_n,
    }


class HyperBPDecoder(nn.Module):
    """Faithful port of HyperNetworkDecoder (model.py). A hypernetwork ("f") generates
    the weights of a small per-iteration variable-node update MLP ("g" / `mlp_vn`) from
    the current odd-node LLR magnitudes, conditioning belief propagation over a fixed
    Tanner-graph structure derived from `parity_check_matrix`."""

    def __init__(
        self,
        parity_check_matrix: np.ndarray,
        num_hidden_layers: int = 5,
        n_hidden_1: int = 16,
        n_hidden_2: int = 16,
        sf_n_hidden_1: int = 32,
        sf_n_hidden_2: int = 32,
        sf_n_hidden_3: int = 32,
    ):
        super().__init__()
        g = generate_graph_matrix(parity_check_matrix)
        self.code_n = g["code_n"]
        self.n_odd = g["n_odd"]
        self.n_even = g["n_even"]
        self.n_input = self.n_odd + 1  # +1 for skip connection
        self.num_hidden_layers = num_hidden_layers

        # fixed (non-trainable) Tanner-graph connectivity matrices
        self.register_buffer("W_input", torch.from_numpy(g["W_input"]))
        self.register_buffer("W_odd2even_graphnn", torch.from_numpy(g["W_odd2even_graphnn"]))
        self.register_buffer("W_skipconn2even", torch.from_numpy(g["W_skipconn2even"]))
        self.register_buffer("W_even2odd", torch.from_numpy(g["W_even2odd"]))
        self.register_buffer("W_output", torch.from_numpy(g["W_output"]))

        # trainable variables masked by the fixed connectivity each forward pass,
        # initialized to the fixed matrix itself (as in the TF1 `tf.Variable(W_fixed)` init)
        self.W_odd2even_graphnn_var = nn.Parameter(
            torch.from_numpy(g["W_odd2even_graphnn"]).clone()
        )
        self.W_output_var = nn.Parameter(torch.from_numpy(g["W_output"]).clone())

        # f-network (hypernetwork): consumes |x_hv| (n_odd,) -> generates g's weights
        self.f_h1 = nn.Parameter(torch.empty(self.n_odd, sf_n_hidden_1))
        self.f_h2 = nn.Parameter(torch.empty(sf_n_hidden_1, sf_n_hidden_2))
        self.f_h3 = nn.Parameter(torch.empty(sf_n_hidden_2, sf_n_hidden_3))
        self.f_h4 = nn.Parameter(torch.empty(sf_n_hidden_3, sf_n_hidden_3))
        self.f_head1 = nn.Parameter(torch.empty(sf_n_hidden_3, self.n_input * n_hidden_1))
        self.f_head2 = nn.Parameter(torch.empty(sf_n_hidden_3, n_hidden_2))

        for p in (self.f_h1, self.f_h2, self.f_h3, self.f_h4, self.f_head1, self.f_head2):
            nn.init.xavier_uniform_(p)

        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2

    def f_hyper(self, x: torch.Tensor):
        """Port of model.py `f_hyper`: 4-layer tanh MLP hypernetwork with two output
        heads (fw1 -> g's first-layer weights, fw2 -> g's output-layer weights)."""
        layer_1 = torch.tanh(torch.einsum("aj,jb->ab", x, self.f_h1))
        layer_2 = torch.tanh(torch.einsum("aj,jb->ab", layer_1, self.f_h2))
        layer_3 = torch.tanh(torch.einsum("aj,jb->ab", layer_2, self.f_h3))
        layer_4 = torch.tanh(torch.einsum("aj,jb->ab", layer_3, self.f_h4))
        out_1 = torch.einsum("aj,jb->ab", layer_4, self.f_head1)
        out_2 = torch.einsum("aj,jb->ab", layer_4, self.f_head2)
        return out_1, out_2

    @staticmethod
    def mlp_vn(x: torch.Tensor, w_h1: torch.Tensor, w_out: torch.Tensor):
        """Port of model.py `mlp_vn`: the hypernetwork-weighted variable-node update
        MLP (2-layer tanh MLP whose weights are per-sample, generated by f_hyper)."""
        layer_1 = torch.tanh(torch.einsum("aij,ajb->aib", x, w_h1))
        out_layer = torch.tanh(torch.einsum("aij,ajb->aib", layer_1, w_out))
        return out_layer.squeeze(2)

    @staticmethod
    def arc_tanh_like(x: torch.Tensor, order: int = 1005):
        """Port of model.py `arc_tanh_like`: truncated odd-power series approximation
        of arctanh, used as the soft check-node combination rule."""
        out = x
        for i in range(3, order + 1):
            if (i - 1) % 2 == 0:
                out = out + (1.0 / i) * torch.pow(x, i)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, code_n) received (soft) channel values. Returns (batch, code_n)
        decoded LLR output after `num_hidden_layers - 1` hyper-BP iterations."""
        batch_size = x.shape[0]

        # ---- input layer (initial odd-node LLR from the channel + parity structure) ----
        x_tile = x.repeat(1, self.n_odd)
        W_input_flat = self.W_input.transpose(0, 1).reshape(-1)
        x_tile = x_tile * W_input_flat
        x_tile = x_tile.reshape(batch_size, self.n_odd, self.code_n)
        u_i = torch.tanh(0.5 * torch.clamp(x_tile, min=-10.0, max=10.0))
        u_i = u_i + (1.0 - (torch.abs(u_i) > 0).float())
        z_input = torch.prod(u_i, dim=2)
        x_hv = torch.log((1 + z_input) / (1 - z_input))

        out_i = x
        for _ in range(0, self.num_hidden_layers - 1, 1):
            # ---- hypernetwork generates the VN-update MLP weights for this iteration ----
            fw1, fw2 = self.f_hyper(torch.abs(x_hv))
            x_hv_c = x_hv.unsqueeze(1)  # (batch, 1, n_odd)

            w_h1 = fw1.reshape(batch_size, self.n_input, self.n_hidden_1)
            w_out = fw2.reshape(batch_size, self.n_hidden_2, 1)

            w_odd2even_masked = self.W_odd2even_graphnn * self.W_odd2even_graphnn_var
            x_input_tile = torch.einsum("aij,bjk->abik", x_hv_c, w_odd2even_masked)
            x_hv_c = x_input_tile.squeeze(2)  # (batch, n_odd, n_odd)

            x_sc = torch.matmul(x, self.W_skipconn2even).unsqueeze(2)  # (batch, n_even, 1)
            x_all = torch.cat([x_hv_c, x_sc], dim=2)  # (batch, n_odd, n_input)
            x_hp = self.mlp_vn(x_all, w_h1, w_out)  # (batch, n_odd)

            # ---- check-node update (fixed closed-form soft-BP combination rule) ----
            x_hv_c = x_hp.repeat(1, self.n_odd)
            x_hv_c = x_hv_c * self.W_even2odd.transpose(0, 1).reshape(-1)
            x_hv_c = x_hv_c.reshape(batch_size, self.n_odd, self.n_even)
            x_hv_c = x_hv_c + (1.0 - (torch.abs(x_hv_c) > 0).float())
            x_hv_c = torch.prod(x_hv_c, dim=2)
            x_hv = 2 * self.arc_tanh_like(x_hv_c, order=1005)

            # ---- marginalization / output LLR ----
            w_output_masked = self.W_output * self.W_output_var
            out_i = x + torch.matmul(x_hv, w_output_masked)

        return out_i


# ---- tiny build/example ----
#
# The real repo ships example parity-check matrices for BCH(63,51) (data/BCH_63_51_H.npy);
# for tracing we instead use the textbook Hamming(7,4) parity-check matrix -- a real,
# valid linear-block-code parity-check matrix (not a synthetic/random stand-in for the
# architecture itself, only for the input *size*), which keeps n_odd small (12) while
# exercising the identical hypernetwork-conditioned BP architecture end-to-end.

_HAMMING_7_4_H = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ],
    dtype=np.float32,
)


def build_hyper_bp_decoder():
    model = HyperBPDecoder(
        parity_check_matrix=_HAMMING_7_4_H,
        num_hidden_layers=3,
        n_hidden_1=8,
        n_hidden_2=8,
        sf_n_hidden_1=8,
        sf_n_hidden_2=8,
        sf_n_hidden_3=8,
    )
    model.eval()
    return model


def example_input_hyper_bp_decoder():
    """Matches HyperBPDecoder.forward: (batch, code_n=7) soft channel LLR/received values."""
    return torch.randn((2, 7), dtype=torch.float32)


MENAGERIE_ENTRIES = [
    ("HyperBP", "build_hyper_bp_decoder", "example_input_hyper_bp_decoder", 2019, MENAGERIE_ZOO),
]
