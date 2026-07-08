# FAITHFUL PORT of sravan-ankireddy/nams @ main (original framework: PyTorch, script-only)
# https://github.com/sravan-ankireddy/nams
#
# NAMS ("Neural Augmented Min Sum", i.e. Neural Min-Sum / weighted-offset min-sum belief
# propagation for LDPC/BCH decoding) is real, runnable PyTorch code, but in the source
# repo it is not packaged as a self-contained `nn.Module.forward()`: the trainable
# `NeuralNetwork` class in `neural_ms.py` holds only the learnable parameters
# (`W_cv`/`B_cv`/`W_gw`), while the actual belief-propagation message-passing (functions
# `compute_vc`, `compute_cv`, `marginalize`, `belief_propagation_iteration`, `nn_decode`
# in the same file) is written as free functions that close over ~15 module-level globals
# (`args`, `n`, `m`, `H`, `edges`, `extrinsic_edges_vc`, `extrinsic_edges_cv`, `new_order_vc`,
# `new_order_cv`, `edges_m`, `var_degrees`, `chk_degrees`, `num_edges`, `device`, `model`)
# populated by an `argparse` CLI script (`get_args()`) and a Tanner-graph loader
# (`load_code()` in `utils.py`, which itself needs `.alist`/`.gmat` parity-check-matrix
# files plus the `mat73` package for some code paths). None of this can be imported and
# called as `model(x)` without first running the whole CLI script.
#
# This module is a FAITHFUL PORT: every formula and control-flow branch reachable under
# the repo's own DEFAULT CLI arguments (`decoder_type="neural_ms"`, `cv_model=1`,
# `vc_model=0`, `entangle_weights=2`, `nn_eq=0`, `relu=0`) is transcribed unchanged from
# `neural_ms.py` + `utils.py::load_code`, just re-homed into a proper `nn.Module` whose
# `__init__` builds the Tanner-graph edge-index bookkeeping (verbatim port of
# `load_code`'s alist parsing + `new_order_cv`/`new_order_vc` construction) and whose
# `forward` runs `num_iterations` belief-propagation iterations (verbatim port of
# `belief_propagation_iteration` -> `compute_vc` -> `compute_cv` -> `marginalize`, default
# path only: `cv_model=1, vc_model=0` weighted min-sum with `entangle_weights=2`
# per-variable-node weight tying, `nn_eq=0` both W_cv/B_cv trainable, `relu=0`).
# The Tanner graph itself is the repo's own bundled `H_G_mat/hamming.alist` /
# `H_G_mat/hamming.gmat` (the smallest code shipped in the repo, a (7,4) Hamming code),
# copied verbatim below (7 variable nodes / 3 check nodes / 4 message bits). No
# architecture, weighting scheme, or numerical formula was invented or altered; only the
# untraceable free-function/global-closure control flow was re-homed into `__init__`/
# `forward`, `np.int` (removed in modern numpy) was replaced with `int`, and the
# CLI/argparse/training-loop/`.mat`-loading/plotting code (irrelevant to the forward pass)
# was dropped.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# H_G_mat/hamming.alist, verbatim (smallest code bundled in the real repo)
# ---------------------------------------------------------------------------
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


def _load_code(alist_text: str):
    """Faithful port of utils.py::load_code's alist-parsing branch (G_filename == "" path
    is not needed here since NAMS only consumes the Tanner-graph edge bookkeeping, not G)."""
    lines = alist_text.splitlines()
    it = iter(lines)

    n, m = [int(s) for s in next(it).split(" ")]
    var_degrees = [0] * n
    chk_degrees = [0] * m

    next(it)  # max_var_degree, max_chk_degree (unused downstream)
    next(it)  # ignored
    next(it)  # ignored

    var_edges = [[] for _ in range(n)]
    for i in range(n):
        row = next(it).split(" ")
        row = [s for s in row if s != ""]
        var_edges[i] = [int(s) - 1 for s in row]
        var_degrees[i] = len(var_edges[i])

    chk_edges = [[] for _ in range(m)]
    for i in range(m):
        row = next(it).split(" ")
        row = [s for s in row if s != ""]
        chk_edges[i] = [int(s) - 1 for s in row]
        chk_degrees[i] = len(chk_edges[i])

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

    edges = []
    for i in range(n):
        for _j in range(var_degrees[i]):
            edges.append(i)

    edges_m = []
    for i in range(n):
        temp_e = []
        for e in range(var_degrees[i]):
            temp_e.append(d[i][e])
        edges_m.append(temp_e)

    edge_order_vc = []
    extrinsic_edges_vc = []
    for i in range(n):
        for j in range(var_degrees[i]):
            edge_order_vc.append(d[i][j])
            temp_edges = []
            for jj in range(var_degrees[i]):
                if jj != j:
                    temp_edges.append(d[i][jj])
            extrinsic_edges_vc.append(temp_edges)

    edge_order_cv = []
    extrinsic_edges_cv = []
    for i in range(m):
        for j in range(chk_degrees[i]):
            edge_order_cv.append(u[i][j])
            temp_edges = []
            for jj in range(chk_degrees[i]):
                if jj != j:
                    temp_edges.append(u[i][jj])
            extrinsic_edges_cv.append(temp_edges)

    new_order_cv = [0] * num_edges
    for pos, e in enumerate(edge_order_cv):
        new_order_cv[e] = pos

    new_order_vc = [0] * num_edges
    for pos, e in enumerate(edge_order_vc):
        new_order_vc[e] = pos

    return dict(
        n=n,
        m=m,
        var_degrees=var_degrees,
        chk_degrees=chk_degrees,
        num_edges=num_edges,
        edges=edges,
        edges_m=edges_m,
        extrinsic_edges_vc=extrinsic_edges_vc,
        extrinsic_edges_cv=extrinsic_edges_cv,
        new_order_cv=new_order_cv,
        new_order_vc=new_order_vc,
    )


class NAMS(nn.Module):
    """Neural Min-Sum LDPC/BCH belief-propagation decoder (default CLI config:
    decoder_type='neural_ms', cv_model=1, vc_model=0, entangle_weights=2, nn_eq=0,
    relu=0), packaged as a proper nn.Module. See module docstring/header for provenance.

    Input: soft_input, an LLR tensor of shape (n, batch_size) -- one channel LLR per
    variable (codeword bit) node, batched. Output: soft_output, same shape, the decoder's
    refined posterior LLRs after `num_iterations` belief-propagation iterations.
    """

    def __init__(self, alist_text: str = _HAMMING_ALIST, num_iterations: int = 5):
        super().__init__()
        code = _load_code(alist_text)
        self.n = code["n"]
        self.m = code["m"]
        self.num_iterations = num_iterations
        self.num_edges = code["num_edges"]
        self.var_degrees = code["var_degrees"]
        self.chk_degrees = code["chk_degrees"]

        # Long buffers used purely as index tensors (not learnable).
        self.register_buffer("edges", torch.tensor(code["edges"], dtype=torch.long))
        self.register_buffer("new_order_vc", torch.tensor(code["new_order_vc"], dtype=torch.long))
        self.register_buffer("new_order_cv", torch.tensor(code["new_order_cv"], dtype=torch.long))
        self.extrinsic_edges_vc = code["extrinsic_edges_vc"]  # list[list[int]], ragged
        self.extrinsic_edges_cv = code["extrinsic_edges_cv"]  # list[list[int]], ragged
        self.edges_m = code["edges_m"]  # list[list[int]], ragged

        # entangle_weights == 2: one weight per variable node (num_w1=1, num_w2=n).
        # cv_model=1, vc_model=0 -> trainable W_cv/B_cv, shape (1, n).
        var_B = 1
        var_W = 1
        B_cv_init = torch.fmod(torch.randn([1, self.n]), 2 * var_B)
        W_cv_init = torch.fmod(torch.randn([1, self.n]), 2 * var_W)
        self.B_cv = nn.Parameter(B_cv_init)
        self.W_cv = nn.Parameter(W_cv_init)

    def _compute_vc(
        self, cv: torch.Tensor, soft_input: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        reordered_soft_input = soft_input[self.edges, :]
        vc_list = []
        count_vc = 0
        for i in range(self.n):
            for _j in range(self.var_degrees[i]):
                ext = self.extrinsic_edges_vc[count_vc]
                if ext:
                    idx = torch.tensor(ext, dtype=torch.long, device=cv.device)
                    temp = cv[idx, :]
                    temp = torch.sum(temp, 0)
                else:
                    temp = torch.zeros([batch_size], device=cv.device)
                vc_list.append(temp)
                count_vc += 1
        vc = torch.stack(vc_list)
        vc = vc[self.new_order_vc, :]
        # cv_model=1, vc_model=0 branch: plain sum (no weighting at the VC step).
        vc = vc + reordered_soft_input
        return vc

    def _compute_cv(self, vc: torch.Tensor, batch_size: int) -> torch.Tensor:
        prod_list = []
        min_list = []
        count_cv = 0
        for i in range(self.m):
            for _j in range(self.chk_degrees[i]):
                ext = self.extrinsic_edges_cv[count_cv]
                if ext:
                    idx = torch.tensor(ext, dtype=torch.long, device=vc.device)
                    temp = vc[idx, :]
                else:
                    temp = torch.zeros([1, batch_size], device=vc.device)
                prod_chk_temp = torch.prod(torch.sign(temp), 0)
                sign_chk_temp, _min_ind = torch.min(torch.abs(temp), 0)
                prod_list.append(prod_chk_temp.float())
                min_list.append(sign_chk_temp.float())
                count_cv += 1
        prods = torch.stack(prod_list)
        mins = torch.stack(min_list)

        # neural_ms + cv_model=1 branch, entangle_weights==2:
        # replicate per-variable-node weights across each var node's edges.
        B_cv_vec = torch.cat(
            [
                self.B_cv[0, im].reshape(1, 1).repeat(1, self.var_degrees[im])
                for im in range(self.n)
            ],
            dim=1,
        )
        W_cv_vec = torch.cat(
            [
                self.W_cv[0, im].reshape(1, 1).repeat(1, self.var_degrees[im])
                for im in range(self.n)
            ],
            dim=1,
        )
        offsets = B_cv_vec[0].reshape(-1, 1).repeat(1, batch_size)
        scaling = W_cv_vec[0].reshape(-1, 1).repeat(1, batch_size)
        # relu=0 branch: plain offset-min-sum (no relu clamp on the offset residual).
        cv = scaling * prods * (mins - offsets)
        cv = cv[self.new_order_cv, :]
        return cv

    def _marginalize(
        self, soft_input: torch.Tensor, cv: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        # cv_model=1, vc_model=0 (not the cv_model==0,vc_model==0 "Gaussian" branch), so
        # marginalization is a plain extrinsic sum with no W_gw weighting.
        soft_output_list = []
        for i in range(self.n):
            idx = torch.tensor(self.edges_m[i], dtype=torch.long, device=cv.device)
            temp = cv[idx, :]
            temp = torch.sum(temp, 0)
            soft_output_list.append(temp)
        soft_output = torch.cat(soft_output_list, 0).reshape(soft_input.shape)
        soft_output = soft_input + soft_output
        return soft_output

    def forward(self, soft_input: torch.Tensor) -> torch.Tensor:
        batch_size = soft_input.shape[1]
        cv = torch.zeros([self.num_edges, batch_size], device=soft_input.device)
        soft_output = soft_input
        for _iteration in range(self.num_iterations):
            vc = self._compute_vc(cv, soft_input, batch_size)
            cv = self._compute_cv(vc, batch_size)
            soft_output = self._marginalize(soft_input, cv, batch_size)
        return soft_output


def build_nams():
    return NAMS(num_iterations=5)


def example_input_nams():
    # n=7 codeword-bit LLRs, batch of 4.
    return torch.randn(7, 4)


MENAGERIE_ENTRIES = [
    ("NAMS", "build_nams", "example_input_nams", 2022, "SOURCE_AVAILABLE"),
]
