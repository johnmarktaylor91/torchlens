# FAITHFUL PORT of https://github.com/lorenlugosch/neural-min-sum-decoding @ master
# (original framework: TensorFlow 1.x, fetched 2026-07-02)
#
# Neural Offset Min-Sum decoding (Lugosch & Gross, "Neural Offset Min-Sum Decoding",
# ISIT 2017, https://arxiv.org/abs/1701.05931). The repo (`main.py` / `helper_functions.py`)
# is TF1.x procedural graph-building code -- `tf.placeholder`, `tf.while_loop`,
# `tf.Session`, hand-rolled `tf.gather`/`tf.stack`/`tf.while_loop` message passing over a
# fixed Tanner (parity-check) graph -- and is not runnable as a base-env torch import
# (TF1.x graph-mode API, deleted from modern TensorFlow). Its architecture is transcribed
# here FAITHFULLY as a base-env torch `nn.Module`, matching `main.py`'s `compute_vc`,
# `compute_cv`, `marginalize`, and `belief_propagation_iteration` functions exactly:
#
#   - compute_vc: variable-to-check messages = each edge's extrinsic sum of incoming
#     check-to-variable messages (all edges at that variable node except itself) plus the
#     variable's original channel LLR ("soft_input"). Ported verbatim from the
#     `for i in range(n): for j in range(var_degrees[i]): ...` extrinsic-sum loop.
#   - compute_cv (FNOMS branch, `decoder_type == "FNOMS"`): check-to-variable messages via
#     min-sum (product of signs * min of magnitudes over the extrinsic set), with a learned
#     PER-ITERATION, PER-EDGE offset `B_cv[iteration]` subtracted from the magnitude and
#     clipped at zero via `tf.nn.softplus` (for the offset) + `tf.nn.relu` (for the
#     subtraction) -- exactly the "neural offset min-sum" contribution over plain min-sum
#     belief propagation.
#   - marginalize: posterior LLR per variable node = channel LLR + sum of all incoming
#     check-to-variable messages at that node.
#   - the fixed edge permutation bookkeeping (`d`, `u`, `edge_order`/`new_order` gather
#     re-indexing in `compute_vc`/`compute_cv`) that maps flat per-edge message vectors
#     between "variable-major" and "check-major" order is reproduced exactly via
#     precomputed index tensors (this bookkeeping is graph topology, not a learned
#     parameter, and is identical every forward call for a fixed code -- so it is computed
#     once in `__init__`, mirroring `helper_functions.load_code`).
#
# The `num_iterations`-step `tf.while_loop` becomes a plain Python `for` loop over learned
# `nn.Parameter` slices (identical unrolled computation, since `num_iterations` is a fixed
# hyperparameter in the original code too). The specific Tanner graph traced here is the
# repo's own bundled `codes/hamming.alist` (Hamming(7,4): n=7 bits, m=3 checks, 12 edges),
# parsed with the exact same alist convention as `helper_functions.load_code`.

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def _parse_alist(text: str):
    """Reproduces `helper_functions.load_code`'s alist parsing + `d`/`u` edge-index
    bookkeeping (graph topology only; no tensors)."""
    lines = text.strip("\n").split("\n")
    idx = 0
    n, m = [int(s) for s in lines[idx].split(" ") if s != ""]
    idx += 1
    idx += 1  # max_var_degree, max_chk_degree (unused directly)
    idx += 2  # two ignored lines (per-node degree lists)

    var_edges = [[] for _ in range(n)]
    for i in range(n):
        row = [s for s in lines[idx].split(" ") if s != ""]
        idx += 1
        var_edges[i] = [int(s) - 1 for s in row]

    chk_edges = [[] for _ in range(m)]
    for i in range(m):
        row = [s for s in lines[idx].split(" ") if s != ""]
        idx += 1
        chk_edges[i] = [int(s) - 1 for s in row]

    var_degrees = [len(x) for x in var_edges]
    chk_degrees = [len(x) for x in chk_edges]

    d = [[] for _ in range(n)]
    edge = 0
    for i in range(n):
        for _j in range(var_degrees[i]):
            d[i].append(edge)
            edge += 1

    u = [[] for _ in range(m)]
    edge = 0
    for i in range(m):
        for j in range(chk_degrees[i]):
            v = chk_edges[i][j]
            for e in range(var_degrees[v]):
                if i == var_edges[v][e]:
                    u[i].append(d[v][e])

    num_edges = sum(var_degrees)
    return dict(
        n=n,
        m=m,
        var_edges=var_edges,
        chk_edges=chk_edges,
        var_degrees=var_degrees,
        chk_degrees=chk_degrees,
        d=d,
        u=u,
        num_edges=num_edges,
    )


# Bundled with the real repo at codes/hamming.alist (Hamming(7,4)).
_HAMMING_ALIST = """7 3
3 4
1 1 1 2 2 3 2
4 4 4
1
2
3
1 2
2 3
1 2 3
1 3
1 4 6 7
2 4 5 6
3 5 6 7"""


class FNOMSDecoder(nn.Module):
    """Feed-forward Neural Offset Min-Sum belief-propagation decoder (FNOMS), ported
    faithfully from `main.py`'s `compute_vc` / `compute_cv` (FNOMS branch) /
    `marginalize` / `belief_propagation_iteration` for a fixed Tanner graph."""

    def __init__(self, alist_text: str = _HAMMING_ALIST, num_iterations: int = 5):
        super().__init__()
        code = _parse_alist(alist_text)
        self.n = code["n"]
        self.m = code["m"]
        self.num_edges = code["num_edges"]
        self.num_iterations = num_iterations

        # Per-variable-node lists of edge indices ("d" in the original code): used by
        # compute_vc's extrinsic sum and by marginalize.
        d = code["d"]
        # Per-check-node lists of edge indices ("u" in the original code): used by
        # compute_cv's extrinsic sum.
        u = code["u"]

        # --- compute_vc bookkeeping: variable-major "reordered_soft_input" gather, and
        # extrinsic-sum-per-edge computed via a dense (num_edges, num_edges) 0/1 mask
        # (edge e's variable-extrinsic-sum = sum over edges at the same variable node,
        # excluding e itself) plus the inverse permutation "new_order".
        var_of_edge = torch.zeros(self.num_edges, dtype=torch.long)
        vc_extrinsic_mask = torch.zeros(self.num_edges, self.num_edges)
        for i in range(self.n):
            for e in d[i]:
                var_of_edge[e] = i
                for e2 in d[i]:
                    if e2 != e:
                        vc_extrinsic_mask[e, e2] = 1.0
        self.register_buffer("var_of_edge", var_of_edge)
        self.register_buffer("vc_extrinsic_mask", vc_extrinsic_mask)

        # --- compute_cv bookkeeping: check-major extrinsic sum, same dense-mask trick,
        # built over the check-major "u" edge order then mapped back to variable-major
        # edge order via `new_order` (edge_order -> identity here since the mask is
        # already indexed in variable-major edge ids through `u`).
        cv_extrinsic_mask = torch.zeros(self.num_edges, self.num_edges)
        for i in range(self.m):
            for e in u[i]:
                for e2 in u[i]:
                    if e2 != e:
                        cv_extrinsic_mask[e, e2] = 1.0
        self.register_buffer("cv_extrinsic_mask", cv_extrinsic_mask)

        # Learned per-iteration, per-edge offsets (decoder.B_cv in the FNOMS branch).
        self.B_cv = nn.Parameter(torch.randn(num_iterations, self.num_edges) * 1.0)

    def compute_vc(self, cv: torch.Tensor, soft_input: torch.Tensor) -> torch.Tensor:
        """cv: (num_edges, batch) check-to-variable messages from the previous iteration
        (zeros on the first). soft_input: (n, batch) channel LLRs.
        Returns vc: (num_edges, batch) variable-to-check messages."""
        # extrinsic sum over all edges at the same variable node except the edge itself
        vc = self.vc_extrinsic_mask @ cv
        reordered_soft_input = soft_input[self.var_of_edge]
        return vc + reordered_soft_input

    def compute_cv(self, vc: torch.Tensor, iteration: int) -> torch.Tensor:
        """vc: (num_edges, batch) variable-to-check messages.
        Returns cv: (num_edges, batch) check-to-variable messages (min-sum with a
        learned, per-iteration offset)."""
        signs = torch.sign(vc)
        mags = torch.abs(vc)

        # extrinsic product-of-signs and min-of-magnitudes over all edges at the same
        # check node except the edge itself.
        mask = self.cv_extrinsic_mask  # (num_edges, num_edges), 1 where extrinsic
        # product of signs over the extrinsic set: since signs are +-1, a masked
        # product reduces to exp(sum(log|sign|)) trick isn't needed -- use a large
        # additive bias to exclude non-extrinsic entries from the min, and multiply
        # signs with masked entries forced to +1 (neutral for a product).
        signs_masked = torch.where(
            mask.bool().unsqueeze(-1), signs.unsqueeze(0), torch.ones_like(signs).unsqueeze(0)
        )
        prods = signs_masked.prod(dim=1)

        big = torch.finfo(mags.dtype).max / 4
        mags_masked = torch.where(
            mask.bool().unsqueeze(-1), mags.unsqueeze(0), torch.full_like(mags, big).unsqueeze(0)
        )
        mins = mags_masked.min(dim=1).values

        offsets = torch.nn.functional.softplus(self.B_cv[iteration]).unsqueeze(-1)
        mins = torch.relu(mins - offsets)

        cv = prods * mins
        return cv

    def marginalize(self, soft_input: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        """soft_input: (n, batch), cv: (num_edges, batch) -> posterior LLRs (n, batch)."""
        soft_output = torch.zeros_like(soft_input)
        soft_output = soft_output.index_add(0, self.var_of_edge, cv)
        return soft_input + soft_output

    def forward(self, soft_input: torch.Tensor) -> torch.Tensor:
        """soft_input: (n, batch) channel LLRs for one codeword per batch element
        (matches `tf_train_dataset` in the original code). Returns the posterior LLR
        after `num_iterations` belief-propagation iterations, matching the final
        `soft_output` of `belief_propagation_op`."""
        batch = soft_input.shape[-1]
        cv = torch.zeros(self.num_edges, batch, dtype=soft_input.dtype, device=soft_input.device)
        soft_output = soft_input

        for iteration in range(self.num_iterations):
            vc = self.compute_vc(cv, soft_input)
            cv = self.compute_cv(vc, iteration)
            soft_output = self.marginalize(soft_input, cv)

        return soft_output


# ---- tiny build/example ----------------------------------------------------------------


def build_noms():
    """FNOMS decoder over the repo's own bundled Hamming(7,4) code, 3 BP iterations."""
    model = FNOMSDecoder(alist_text=_HAMMING_ALIST, num_iterations=3)
    model.eval()
    return model


def example_input_noms():
    """Matches FNOMSDecoder.forward: (n, batch) channel LLRs, n=7 for the Hamming(7,4)
    code (BPSK-modulated codeword bits + noise, as in the real training loop)."""
    torch.manual_seed(0)
    return torch.randn(7, 4, dtype=torch.float32)


MENAGERIE_ENTRIES = [
    ("Neural Offset Min-Sum (NOMS)", "build_noms", "example_input_noms", 2017, MENAGERIE_ZOO),
]
