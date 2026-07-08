# SOURCE: vendored from tsy935/graphs4mer @ main
#   model/graphs4mer.py :: GraphS4mer (+ module-level helper functions)
#   model/s4.py :: S4/S4Model/SSKernel stack (from HazyResearch/state-spaces, re-published
#                   in this repo as a "standalone version" -- Author: albertfgu)
#   model/graph_learner.py :: GraphLearner (adapted from hugochan/IDGL)
#   model/decoders.py :: SequenceDecoder (from HazyResearch/state-spaces)
#
# GraphS4mer (Tang et al., "Modeling Multivariate Biosignals With Graph Neural
# Networks and Structured State Spaces", Best Paper CHIL 2023, arXiv:2211.11176).
# A dynamic-graph architecture for multivariate biosignals (EEG/ECG/PSG): each
# channel is a node whose raw waveform is embedded by an S4 (Structured State
# Space) temporal encoder, a learned self-attention graph (GraphLearner) infers
# time-varying edges between channels (optionally blended with a k-NN cosine
# prior graph), a stack of GINEConv GNN layers propagates information across
# the learned graph per timestep, and a linear classifier reads out the
# temporally + graph-pooled representation.
#
# Vendored verbatim (imports consolidated into one preamble since the four
# source files are merged into a single module here; no architectural changes).
# The upstream module.decoders/module.s4/module.graph_learner cross-imports
# (`from model.s4 import S4Model`, `from model.graph_learner import *`, etc.)
# are replaced with plain in-file definition order since everything now lives
# in one file. The S4 kernel's optional CUDA / pykeops fast paths
# (`extensions.cauchy`, `pykeops`) are absent here exactly as in the real
# code's own try/except fallback -- it transparently falls back to the slow
# (but correct) naive Cauchy/Vandermonde kernels, which is what the real repo
# does on any machine without those optional extensions installed.
# Only the classification variant (`GraphS4mer`) is vendored; the sibling
# `GraphS4mer_Regression` class in the same file is architecturally identical
# modulo its decoder head and is omitted for brevity.
#
# One import gap in the real upstream file is completed here (not an
# architecture change): `graphs4mer.py`'s `temporal_model="gru"` branch calls
# `pack_padded_sequence`/`pad_packed_sequence` without importing them from
# `torch.nn.utils.rnn` -- a pre-existing bug in the real repo. The import is
# added below so the module is importable; that branch is not exercised by
# the default (`temporal_model="s4"`) build/example below regardless.

import logging
import math
from functools import partial

import numpy as np
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from einops import rearrange, reduce, repeat
from pytorch_lightning.utilities import rank_zero_only
from scipy import special as ss
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import (
    GATv2Conv,
    GINEConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

MENAGERIE_ZOO = "vendored-pytorch"

contract = oe.contract
contract_expression = oe.contract_expression


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for lvl in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, lvl, rank_zero_only(getattr(logger, lvl)))

    return logger


log = get_logger(__name__)


class Decoder(nn.Module):
    """This class doesn't do much but just signals the interface that Decoders are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset
        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


class SequenceDecoder(Decoder):
    def __init__(self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == "ragged":
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]  # noqa: E731
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]  # noqa: E731
        elif self.mode == "pool":
            restrict = (
                lambda x: (  # noqa: E731
                    torch.cumsum(x, dim=-2)
                    / torch.arange(1, 1 + x.size(-2), device=x.device, dtype=x.dtype).unsqueeze(-1)
                )[..., -l_output:, :]
            )

            def restrict(x):  # noqa: F811
                L = x.size(-2)
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(-2)
                denom = torch.arange(L - l_output + 1, L + 1, dtype=x.dtype, device=x.device)
                s = s / denom
                return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]  # noqa: E731
            # TODO use same restrict function as pool case
        elif self.mode == "ragged":
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]  # noqa: E731
        else:
            raise NotImplementedError("Mode must be ['last' | 'first' | 'pool' | 'sum']")

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.output_transform(x)


import logging  # noqa: E402
import torch.nn as nn  # noqa: E402
import opt_einsum as oe  # noqa: E402

contract = oe.contract
contract_expression = oe.contract_expression


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)

""" Cauchy and Vandermonde kernels """

try:  # Try CUDA extension
    from extensions.cauchy.cauchy import cauchy_mult

    has_cauchy_extension = True
except:  # noqa: E722
    # log.warn(
    #     "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    # )
    has_cauchy_extension = False

try:  # Try pykeops
    import pykeops
    from pykeops.torch import Genred

    has_pykeops = True
    log.info("Pykeops installation found.")

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [
            tensor.view((1,) * (max_dim - len(tensor.shape)) + tensor.shape) for tensor in tensors
        ]
        return tensors

    def cauchy_conj(v, z, w):
        """Pykeops version"""
        expr_num = "z * ComplexReal(v) - Real2Complex(Sum(v * w))"
        expr_denom = "ComplexMult(z-w, z-Conj(w))"

        cauchy_mult = Genred(
            f"ComplexDivide({expr_num}, {expr_denom})",
            [
                "v = Vj(2)",
                "z = Vi(2)",
                "w = Vj(2)",
            ],
            reduction_op="Sum",
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2 * cauchy_mult(v, z, w, backend="GPU")
        return _r2c(r)

    def log_vandermonde(v, x, L):
        expr = "ComplexMult(v, ComplexExp(ComplexMult(x, l)))"
        vandermonde_mult = Genred(
            expr,
            [
                "v = Vj(2)",
                "x = Vj(2)",
                "l = Vi(2)",
            ],
            reduction_op="Sum",
            axis=1,
        )

        l = torch.arange(L).to(x)  # noqa: E741
        v, x, l = _broadcast_dims(v, x, l)  # noqa: E741
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)  # noqa: E741

        r = vandermonde_mult(v, x, l, backend="GPU")
        return 2 * _r2c(r).real

    def log_vandermonde_transpose(u, v, x, L):
        """
        u: ... H L
        v: ... H N
        x: ... H N
        Returns: ... H N

        V = Vandermonde(a, L) : (H N L)
        contract_L(V * u * v)
        """
        expr = "ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))"
        vandermonde_mult = Genred(
            expr,
            [
                "u = Vj(2)",
                "v = Vi(2)",
                "x = Vi(2)",
                "l = Vj(2)",
            ],
            reduction_op="Sum",
            axis=1,
        )

        l = torch.arange(L).to(x)  # noqa: E741
        u, v, x, l = _broadcast_dims(u, v, x, l)  # noqa: E741
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)  # noqa: E741

        r = vandermonde_mult(u, v, x, l, backend="GPU")
        return _r2c(r)

except ImportError:
    has_pykeops = False
    if not has_cauchy_extension:
        log.warning(
            "Falling back on slow Cauchy kernel. Install at least one of pykeops or the CUDA extension for efficiency."
        )

        def cauchy_naive(v, z, w):
            """
            v, w: (..., N)
            z: (..., L)
            returns: (..., L)
            """
            cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))  # (... N L)
            return torch.sum(cauchy_matrix, dim=-2)

    # Vandermonde functions
    log.error(
        "Falling back on slow Vandermonde kernel. Install pykeops for improved memory efficiency."
    )

    def log_vandermonde(v, x, L):
        """
        v: (..., N)
        x: (..., N)
        returns: (..., L) \sum v x^l
        """
        vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x))  # (... N L)
        vandermonde_prod = contract("... n, ... n l -> ... l", v, vandermonde_matrix)  # (... L)
        return 2 * vandermonde_prod.real

    def log_vandermonde_transpose(u, v, x, L):
        vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x))  # (... N L)
        vandermonde_prod = contract(
            "... l, ... n, ... n l -> ... n", u.to(x), v.to(x), vandermonde_matrix
        )  # (... L)
        return vandermonde_prod


_conj = lambda x: torch.cat([x, x.conj()], dim=-1)  # noqa: E731
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()  # noqa: E731
else:
    _resolve_conj = lambda x: x.conj()  # noqa: E731


""" Simple nn.Module components """


def Activation(activation=None, dim=-1):
    if activation in [None, "id", "identity", "linear"]:
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation in ["swish", "silu"]:
        return nn.SiLU()
    elif activation == "glu":
        return nn.GLU(dim=dim)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def LinearActivation(
    d_input,
    d_output,
    bias=True,
    transposed=False,
    activation=None,
    activate=False,  # Apply activation as part of this module
    **kwargs,
):
    """Returns a linear nn.Module with control over axes order, initialization, and activation"""

    # Construct core module
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation == "glu":
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)"""
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            return X
        return X


""" Misc functional utilities """


def power(L, A, v=None):
    """Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A)  # , dtype=A.dtype, device=A.device)  # noqa: E741

    powers = [A]
    l = 1  # noqa: E741
    while True:
        if L % 2 == 1:
            I = powers[-1] @ I  # noqa: E741
        L //= 2
        if L == 0:
            break
        l *= 2  # noqa: E741
        powers.append(powers[-1] @ powers[-1])

    if v is None:
        return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, "... (z l) -> ... z l", z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


""" HiPPO utilities """


def transition(measure, N):
    """A, B transition matrices for different measures"""
    # Legendre (translated)
    if measure == "legt":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # Legendre (scaled)
    elif measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = (
            B.copy()
        )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == "legsd":
        # Essentially equivalent to S4D-LegS
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = (
            B.copy()
        )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        A += 0.5 * B * B[None, :, 0]
        B = B / 2.0
    elif measure in ["fourier_diag", "foud"]:
        # Essentially equivalent to S4D-Lin
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        A = A - 0.5 * np.eye(N)
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1
        B = B[:, None]
    elif measure in ["fourier", "fout"]:
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    else:
        raise NotImplementedError

    return A, B


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """Return low-rank matrix L such that A + L is normal"""

    if measure == "legs":
        assert rank >= 1
        P = torch.sqrt(0.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)  # (1 N)
    elif measure == "legt":
        assert rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype))  # (N)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)  # (2 N)
        P *= 2 ** (-0.5)  # Halve the rank correct just like the original matrix was halved
    elif measure in ["fourier", "fout"]:
        P = torch.zeros(N)
        P[0::2] = 2**0.5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure in ["fourier_diag", "foud", "legsd"]:
        P = torch.zeros(1, N, dtype=dtype)
    else:
        raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank - d, N, dtype=dtype)], dim=0)  # (rank N)
    return P


def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True):
    """Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype)  # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]  # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)  # (r N)
    AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    err = torch.sum((_A - _A[0, 0] * torch.eye(N)) ** 2) / N
    if (
        err > 1e-5
    ):  # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)

    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision:
        AP = AP.to(torch.double)
    w_im, V = torch.linalg.eigh(AP * -1j)  # (..., N) (..., N, N)
    if diagonalize_precision:
        w_im, V = w_im.to(cdtype), V.to(cdtype)
    w = w_re + 1j * w_im
    # Check: V w V^{-1} = A
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    # Only keep half of each conjugate pair
    _, idx = torch.sort(w.imag)
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, : N // 2]
    w = w_sorted[: N // 2]
    assert w[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if w[-1].abs() < 1e-4:
        V[:, -1] = 0.0
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)
    err = torch.sum((2 * _AP.real - AP) ** 2) / N
    if err > 1e-5:
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    B = contract("ij, j -> i", V_inv, B.to(V))  # V^* B
    P = contract("ij, ...j -> ...i", V_inv, P.to(V))  # V^* P

    return w, P, B, V


def dplr(
    scaling,
    N,
    rank=1,
    H=1,
    dtype=torch.float,
    real_scale=1.0,
    imag_scale=1.0,
    random_real=False,
    random_imag=False,
    normalize=False,
    diagonal=True,
    random_B=False,
):
    assert dtype == torch.float or torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)
    if random_real:
        real_part = torch.rand(H, N // 2)
    else:
        real_part = 0.5 * torch.ones(H, N // 2)
    if random_imag:
        imag_part = N // 2 * torch.rand(H, N // 2)
    else:
        imag_part = repeat(torch.arange(N // 2), "n -> h n", h=H)

    real_part = real_scale * real_part
    if scaling == "random":
        imag_part = torch.randn(H, N // 2)
    elif scaling == "real":
        imag_part = 0 * imag_part
        real_part = 1 + repeat(torch.arange(N // 2), "n -> h n", h=H)
    elif scaling in ["linear", "lin"]:
        imag_part = pi * imag_part
    elif scaling in [
        "inverse",
        "inv",
    ]:  # Based on asymptotics of the default HiPPO matrix
        imag_part = 1 / pi * N * (N / (1 + 2 * imag_part) - 1)
    elif scaling in ["inverse2", "inv2"]:
        imag_part = 1 / pi * N * (N / (1 + imag_part) - 1)
    elif scaling in ["quadratic", "quad"]:
        imag_part = 1 / pi * (1 + 2 * imag_part) ** 2
    elif scaling in ["legs", "hippo"]:
        w, _, _, _ = nplr("legsd", N)
        imag_part = w.imag

    else:
        raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part

    # Initialize B
    if random_B:
        B = torch.randn(H, N // 2, dtype=dtype)
    else:
        B = torch.ones(H, N // 2, dtype=dtype)

    if normalize:
        norm = -B / w  # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2 * torch.sum(
            torch.abs(norm) ** 2, dim=-1, keepdim=True
        )  # Variance with a random C vector
        B = B / zeta**0.5

    P = torch.randn(rank, H, N // 2, dtype=dtype)
    if diagonal:
        P = P * 0.0
    V = torch.eye(N, dtype=dtype)[:: N // 2]  # Only used in testing
    V = repeat(V, "n m -> h n m", h=H)

    return w, P, B, V


def ssm(measure, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization

    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    if measure == "dplr":
        w, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    elif measure.startswith("diag"):
        args = measure.split("-")
        assert args[0] == "diag" and len(args) > 1
        scaling = args[1]
        w, P, B, V = dplr(scaling=scaling, N=N, rank=R, H=H, diagonal=True, **ssm_args)
    else:
        w, P, B, V = nplr(measure, N, R, **ssm_args)
        w = repeat(w, "n -> s n", s=H)
        P = repeat(P, "r n -> r s n", s=H)
        B = repeat(B, "n -> s n", s=H)
        V = repeat(V, "n m -> s n m", s=H)
    return w, P, B, V


combinations = {
    "hippo": ["legs", "fourier"],
    "diag": ["diag-inv", "diag-lin"],
    "all": ["legs", "fourier", "diag-inv", "diag-lin"],
}


def combination(measures, N, R, S, **ssm_args):
    if isinstance(measures, str):
        measures = combinations[measures] if measures in combinations else [measures]

    assert S % len(measures) == 0, (
        f"{S} independent trainable SSM copies must be multiple of {len(measures)} different measures"
    )
    w, P, B, V = zip(*[ssm(measure, N, R, S // len(measures), **ssm_args) for measure in measures])
    w = torch.cat(w, dim=0)  # (S N)
    P = torch.cat(P, dim=1)  # (R S N)
    B = torch.cat(B, dim=0)  # (S N)
    V = torch.cat(V, dim=0)  # (S N N)
    return w, P, B, V


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters"""

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class SSKernelNPLR(OptimModule):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)"""

    @torch.no_grad()
    def _setup_C(self, L):
        """Construct C~ from C

        Two modes are supported: go directly to length L if self.L is 1, or length is doubled
        """

        if self.L.item() == 0:
            if self.verbose:
                log.info(f"S4: Initializing kernel to length {L}")
            double_length = False
        elif L > self.L.item():  # 2*int(self.L) == L:
            if self.verbose:
                log.info(f"S4: Doubling length from L = {self.L.item()} to {2 * self.L.item()}")
            double_length = True
            L = self.L.item()  # Convenience for the math below
        else:
            return

        C = _r2c(self.C)
        dA, _ = self._setup_state()
        dA_L = power(L, dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length:
            prod = -prod  # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., : self.N]  # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        self.L = 2 * self.L if double_length else self.L + L  # Preserve type/device

    def _omega(self, L, dtype, device, cache=True):
        """Calculate (and cache) FFT nodes and their "unprocessed" version with the bilinear transform
        This should be called everytime the internal length self.L changes"""

        # Use cached if available
        if cache and hasattr(self, "omega") and self.omega.size(-1) == L // 2 + 1:
            return self.omega, self.z

        omega = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=dtype, device=device)  # \omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)

        # Cache if necessary
        if cache:
            self.omega = omega
            self.z = z
        return omega, z

    def __init__(
        self,
        w,
        P,
        B,
        C,
        log_dt,
        L=None,  # starting/maximum length of kernel
        lr=None,
        verbose=False,
        keops=False,
        real_type="exp",  # ['none' | 'exp' | 'relu' | sigmoid']
        real_tolerance=1e-3,
        bandlimit=None,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        A is represented by diag(w) - PP^*
        w: (S, N) diagonal part
        P: (R, S, N) low-rank part

        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature
        lr: [dict | float | None] hook to set lr of special parameters (A, B, dt)

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        R (or rank): rank of low-rank part
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.verbose = verbose
        self.keops = keops
        self.bandlimit = bandlimit
        self.real_type = real_type
        self.real_tolerance = real_tolerance

        # Rank of low-rank correction
        self.rank = P.shape[-3]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)

        # Check different SSM inits
        assert w.size(-2) == P.size(-2) == B.size(-2)  # n_ssm
        assert self.H % w.size(0) == 0
        self.n_ssm = w.size(0)
        self.broadcast = self.H // w.size(
            0
        )  # Each trainable SSM needs to be duplicated this many times

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))  # (C, H, N)
        B = B.unsqueeze(0)  # (1, 1, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None
        self.register("log_dt", log_dt, lr_dict.get("dt", lr))
        self.register("B", _c2r(B), lr_dict.get("B", lr))
        self.register("P", _c2r(P), lr_dict.get("A", lr))
        self.register("inv_w_real", self._w_init(w.real), lr_dict.get("A", lr))
        self.register("w_imag", w.imag, lr_dict.get("A", lr))

        self.l_max = L
        self.register_buffer("L", torch.tensor(0))  # Internal length

    def _w_init(self, w_real):
        w_real = torch.clamp(w_real, max=-self.real_tolerance)
        if self.real_type == "none":
            return -w_real
        elif self.real_type == "exp":
            return torch.log(-w_real)  # Some of the HiPPO methods have real part 0
        elif self.real_type == "relu":
            return -w_real
        elif self.real_type == "sigmoid":
            return torch.logit(-w_real)
        elif self.real_type == "softplus":
            return torch.log(torch.exp(-w_real) - 1)
        else:
            raise NotImplementedError

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.real_type == "none":
            w_real = -self.inv_w_real
        elif self.real_type == "exp":
            w_real = -torch.exp(self.inv_w_real)
        elif self.real_type == "relu":
            w_real = -F.relu(self.inv_w_real)
        elif self.real_type == "sigmoid":
            w_real = -F.sigmoid(self.inv_w_real)
        elif self.real_type == "softplus":
            w_real = -F.softplus(self.inv_w_real)
        else:
            raise NotImplementedError
        w = w_real + 1j * self.w_imag
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """

        # Initialize C~ if necessary (done in forward pass so it's on the correct device)
        if self.L.item() == 0 and self.l_max is not None and self.l_max > 0:
            self._setup_C(self.l_max)

        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) frequency rate
        if L is None:
            L = round(self.L.item() / rate)

        # Increase the internal length if needed
        continuous_L = round(rate * L)
        while continuous_L > self.L.item():
            self._setup_C(continuous_L)
        discrete_L = round(self.L.item() / rate)

        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj()
        w = self._w()  # (n_ssm, N)

        # Address bandlimiting
        if self.bandlimit is not None:
            freqs = w.imag.abs() / (2 * math.pi)  # (H, N)
            freqs = dt[:, None] / rate * freqs  # (H, N)
            mask = torch.where(freqs < self.bandlimit * 0.5, 1, 0)
            C = C * mask

        # Get FFT nodes of right length
        omega, z = self._omega(discrete_L, dtype=w.dtype, device=w.device, cache=(rate == 1.0))

        # Broadcast parameters to same hidden features H
        B = repeat(B, "1 t n -> 1 (v t) n", v=self.broadcast)
        P = repeat(P, "r t n -> r (v t) n", v=self.broadcast)
        Q = repeat(Q, "r t n -> r (v t) n", v=self.broadcast)
        w = repeat(w, "t n -> (v t) n", v=self.broadcast)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state  # (B H N)
            sA = s * _conj(w) - contract(  # (B H N)
                "bhm, rhm, rhn -> bhn", s, _conj(Q), _conj(P)
            )
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., : self.N]

            B = torch.cat([s, B], dim=-3)  # (B+1, H, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3)  # (B+1+R, H, N)
        C = torch.cat([C, Q], dim=-3)  # (C+R, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (B+1+R, C+R, H, N)

        # Calculate resolvent at omega
        if has_cauchy_extension and z.dtype == torch.cfloat and not self.keops:
            r = cauchy_mult(v, z, w, symmetric=True)
        elif has_pykeops:
            r = cauchy_conj(v, z, w)
        else:
            r = cauchy_naive(v, z, w)
        r = r * dt[None, None, :, None]  # (B+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (
                1 + r[-1:, -1:, :, :]
            )
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[
                1:, :1, :, :
            ]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=discrete_L)  # (B+1, C, H, L)

        # # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (B, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :]  # (C H L)

        return k_B, k_state

    @torch.no_grad()
    def _setup_linear(self):
        """Create parameters that allow fast linear stepping of state"""
        w = self._w()
        B = _r2c(self.B)  # (H N)
        P = _r2c(self.P)
        Q = P.conj()

        # Repeat w shape properly
        B = repeat(B, "1 t n -> 1 (v t) n", v=self.broadcast)
        P = repeat(P, "r t n -> r (v t) n", v=self.broadcast)
        Q = repeat(Q, "r t n -> r (v t) n", v=self.broadcast)
        w = repeat(w, "t n -> (v t) n", v=self.broadcast)

        # Prepare Linear stepping
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        R = (
            torch.eye(self.rank, dtype=w.dtype, device=w.device)
            + 2 * contract("r h n, h n, s h n -> h r s", Q, D, P).real
        )  # (H R R)
        Q_D = rearrange(Q * D, "r h n -> h r n")
        try:
            R = torch.linalg.solve(R, Q_D)  # (H R N)
        except:  # noqa: E722
            R = torch.tensor(
                np.linalg.solve(
                    R.to(Q_D).contiguous().detach().cpu(),
                    Q_D.contiguous().detach().cpu(),
                )
            ).to(Q_D)
        R = rearrange(R, "h r n -> r h n")

        self.step_params = {
            "D": D,  # (H N)
            "R": R,  # (R H N)
            "P": P,  # (R H N)
            "Q": Q,  # (R H N)
            "B": B,  # (1 H N)
            "E": 2.0 / dt.unsqueeze(-1) + w,  # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C)  # View used for dtype/device

        if u is None:  # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:  # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if (
            state.size(-1) == self.N
        ):  # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = (
                lambda p, x, y: contract(  # noqa: E731
                    "r h n, r h m, ... h m -> ... h n", _conj(p), _conj(x), _conj(y)
                )[..., : self.N]
            )  # inner outer product
        else:
            assert state.size(-1) == 2 * self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract(  # noqa: E731
                "r h n, r h m, ... h m -> ... h n", p, x, y
            )  # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (R H N)
        P = step_params["P"]  # (R H N)
        Q = step_params["Q"]  # (R H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state)  # (B H N)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """Construct dA and dB for discretized state equation"""

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C)  # Just returns a view that we use for finding dtype/device

        state = torch.eye(2 * self.N, dtype=C.dtype, device=C.device).unsqueeze(-2)  # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        dB = rearrange(dB, "1 h n -> h n")  # (H N)
        return dA, dB

    def _step_state(self, u, state):
        """Must be called after self.default_state() is used to construct an initial state!"""
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state

    def _setup_step(self, mode="dense"):
        """Set up dA, dB, dC discretized parameters for stepping"""
        self.dA, self.dB = self._setup_state()

        # Calculate original C
        C = _conj(_r2c(self.C))  # (H C N)
        if self.L.item() == 0:
            dC = C
        else:
            # self.C represents C_tilde
            dA_L = power(self.L.item(), self.dA)
            I = torch.eye(self.dA.size(-1)).to(dA_L)  # noqa: E741

            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2),
                C.unsqueeze(-1),
            ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == "linear":
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2 * self.dC[:, :, : self.N]
        elif mode == "diagonal":
            # Eigendecomposition of the A matrix
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print(
                    "Diagonalization error:",
                    torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA),
                )

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract("h n m, h m -> h n", V_inv, self.dB)
            self.dC = contract("h n m, c h n -> c h m", V, self.dC)

        elif mode == "dense":
            pass
        else:
            raise NotImplementedError(
                "NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}"
            )

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        step_mode = getattr(
            self, "_step_mode", "dense"
        )  # Used in default_state, which is called without _setup_step() in forward_state()
        if step_mode != "linear":
            N *= 2

            if step_mode == "diagonal":
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N),  # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N),  # self.dC.shape
            batch_shape + (H, N),
        )

        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """Must have called self._setup_step() and created state with self.default_state() before calling this"""

        if self._step_mode == "linear":
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y.real, new_state


class SSKernelDiag(OptimModule):
    """Version using (complex) diagonal state matrix (S4D)"""

    def __init__(
        self,
        A,
        B,
        C,
        log_dt,
        L=None,
        disc="bilinear",
        real_type="exp",
        lr=None,
        bandlimit=None,
    ):
        super().__init__()
        self.L = L
        self.disc = disc
        self.bandlimit = bandlimit
        self.real_type = real_type

        # Rank of low-rank correction
        assert A.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = A.size(-1)
        assert A.size(-2) == B.size(-2)  # Number of independent SSMs trained
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)

        self.channels = C.shape[0]
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        # Register parameters
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None

        self.register("log_dt", log_dt, lr_dict.get("dt", lr))
        self.register("A", _c2r(A), lr_dict.get("A", lr))
        self.register("B", _c2r(B), lr_dict.get("B", lr))
        self.register("inv_A_real", self._A_init(A.real), lr_dict.get("A", lr))
        self.register("A_imag", A.imag, lr_dict.get("A", lr))

    def _A_init(self, A_real):
        A_real = torch.clamp(A_real, max=-1e-4)
        if self.real_type == "none":
            return -A_real
        elif self.real_type == "exp":
            return torch.log(-A_real)  # Some of the HiPPO methods have real part 0
        elif self.real_type == "relu":
            return -A_real
        elif self.real_type == "sigmoid":
            return torch.logit(-A_real)
        elif self.real_type == "softplus":
            return torch.log(torch.exp(-A_real) - 1)
        else:
            raise NotImplementedError

    def _A(self):
        # Get the internal A (diagonal) parameter
        if self.real_type == "none":
            A_real = -self.inv_A_real
        elif self.real_type == "exp":
            A_real = -torch.exp(self.inv_A_real)
        elif self.real_type == "relu":
            # JAX version seems to NaN if you alloA 0's, although this code Aas fine Aithout it
            A_real = -F.relu(self.inv_A_real) - 1e-4
        elif self.real_type == "sigmoid":
            A_real = -F.sigmoid(self.inv_A_real)
        elif self.real_type == "softplus":
            A_real = -F.softplus(self.inv_A_real)
        else:
            raise NotImplementedError
        A = A_real + 1j * self.A_imag
        return A

    def forward(self, L, state=None, rate=1.0, u=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """

        dt = torch.exp(self.log_dt) * rate  # (H)
        C = _r2c(self.C)  # (C H N)
        A = self._A()  # (H N)

        B = _r2c(self.B)
        B = repeat(B, "t n -> 1 (v t) n", v=self.repeat)

        if self.bandlimit is not None:
            freqs = dt[:, None] / rate * A.imag.abs() / (2 * math.pi)  # (H, N)
            mask = torch.where(freqs < self.bandlimit * 0.5, 1, 0)
            C = C * mask

        # Incorporate dt into A
        A = repeat(A, "t n -> (v t) n", v=self.repeat)
        dtA = A * dt.unsqueeze(-1)  # (H N)

        # Augment B with state
        if state is not None:
            s = state / dt.unsqueeze(-1)
            if self.disc == "bilinear":
                s = s * (1.0 + dtA / 2)
            elif self.disc == "zoh":
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.0)
            B = torch.cat([s, B], dim=-3)  # (1+B H N)

        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)
        if self.disc == "zoh":
            # Power up
            C = C * (torch.exp(dtA) - 1.0) / A
            K = log_vandermonde(C, dtA, L)  # (H L)
        elif self.disc == "bilinear":
            C = C * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)  # or * dtA / A
            dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
            K = log_vandermonde(C, dA.log(), L)
        elif self.disc == "dss":
            # Implementation from DSS meant for case when real eigenvalues can be positive
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device)  # [H N L]
            A_gt_0 = A.real > 0  # [N]
            if A_gt_0.any():
                with torch.no_grad():
                    P_max = dtA * (A_gt_0 * (L - 1))  # [H N]
                P = P - P_max.unsqueeze(-1)  # [H N L]
            S = P.exp()  # [H N L]

            dtA_neg = dtA * (1 - 2 * A_gt_0)  # [H N]
            num = dtA_neg.exp() - 1  # [H N]
            den = (dtA_neg * L).exp() - 1  # [H N]

            # Inline reciprocal function for DSS logic
            x = den * A
            x_conj = _resolve_conj(x)
            r = x_conj / (x * x_conj + 1e-7)

            C = C * num * r  # [C H N]
            K = contract("chn,hnl->chl", C, S).float()
        else:
            assert False, f"{self.disc} not supported"

        K = K.view(-1, self.channels, self.H, L)  # (1+B C H L)
        if state is not None:
            K_state = K[:-1, :, :, :]  # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :]  # (C H L)
        return K, K_state

    def _setup_step(self):
        # These methods are organized like this to be compatible with the NPLR kernel interface
        dt = torch.exp(self.log_dt)  # (H)
        B = _r2c(self.B)  # (H N)
        C = _r2c(self.C)  # (C H N)
        self.dC = C
        A = self._A()  # (H N)

        # Incorporate dt into A
        dtA = A * dt.unsqueeze(-1)  # (H N)
        if self.disc == "zoh":
            self.dA = torch.exp(dtA)  # (H N)
            self.dB = B * (torch.exp(dtA) - 1.0) / A  # (C H N)
        elif self.disc == "bilinear":
            self.dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
            self.dB = B * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)  # or * dtA / A

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) + contract(
            "h n, b h -> b h n", self.dB, u
        )
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2 * y.real, next_state

    def forward_state(self, u, state):
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous()  # (B H L)
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state


class SSKernel(nn.Module):
    """Wrapper around SSKernel parameterizations.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    _setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=None,
        measure="legs",
        rank=1,
        channels=1,
        dt_min=0.001,
        dt_max=0.1,
        deterministic=False,
        lr=None,
        mode="nplr",
        n_ssm=None,
        verbose=False,
        measure_args={},
        **kernel_args,
    ):
        """State Space Kernel which computes the convolution kernel $\\bar{K}$

        H: Number of independent SSM copies; controls the size of the model. Also called d_model in the config.
        N: State size (dimensionality of parameters A, B, C). Also called d_state in the config. Generally shouldn't need to be adjusted and doens't affect speed much.
        L: Maximum length of convolution kernel, if known. Should work in the majority of cases even if not known.
        measure: Options for initialization of (A, B). For NPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
        rank: Rank of low-rank correction for NPLR mode. Needs to be increased for measure "legt"
        channels: C channels turns the SSM from a 1-dim to C-dim map; can think of it having C separate "heads" per SSM. This was partly a feature to make it easier to implement bidirectionality; it is recommended to set channels=1 and adjust H to control parameters instead
        dt_min, dt_max: min and max values for the step size dt (\Delta)
        mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D; 'slow' is a dense version for testing
        n_ssm: Number of independent trainable (A, B) SSMs, e.g. n_ssm=1 means all A/B parameters are tied across the H different instantiations of C. n_ssm=None means all H SSMs are completely independent. Generally, changing this option can save parameters but doesn't affect performance or speed much. This parameter must divide H
        lr: Passing in a number (e.g. 0.001) sets attributes of SSM parameers (A, B, dt). A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        """
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), H))
        else:
            log_dt = torch.rand(self.H, dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)

        # Compute the preprocessed representation
        w, P, B, V = combination(measure, self.N, rank, self.n_ssm, **measure_args)

        # Broadcast C to have H channels
        if deterministic:
            C = torch.zeros(channels, self.H, self.N, dtype=cdtype)
            C[:, :, :1] = 1.0
            C = contract("hmn, chn -> chm", V.conj().transpose(-1, -2), C)  # V^* C
        else:
            C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert (
            self.n_ssm % B.size(-2) == 0
            and self.n_ssm % P.size(-2) == 0
            and self.n_ssm % w.size(-2) == 0
        )
        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, "t n -> (v t) n", v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, "r t n -> r (v t) n", v=self.n_ssm // P.size(-2)).clone().contiguous()
        w = repeat(w, "t n -> (v t) n", v=self.n_ssm // w.size(-2)).clone().contiguous()
        C = C.contiguous()

        if mode == "nplr":
            self.kernel = SSKernelNPLR(
                w,
                P,
                B,
                C,
                log_dt,
                L=L,
                lr=lr,
                verbose=verbose,
                **kernel_args,
            )
        elif mode == "diag":
            C = C * repeat(B, "t n -> (v t) n", v=H // self.n_ssm)
            self.kernel = SSKernelDiag(
                w,
                B,
                C,
                log_dt,
                L=L,
                lr=lr,
                **kernel_args,
            )
        else:
            raise NotImplementedError(f"{mode} is not valid")

    def forward(self, state=None, L=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """

        if hasattr(self.kernel, "forward_state"):
            return self.kernel.forward_state(u, state)

        dA, dB = self.kernel._setup_state()  # Construct dA, dB matrices
        # dA, dB = self.kernel.dA, self.kernel.dB # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj:
            state = _conj(state)

        v = contract(
            "h n, b h l -> b h n l", dB, u.flip(-1)
        )  # dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj:
            next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_step(self, **kwargs):
        # This method is intended to be private so that setting up an S4 module with
        # ```
        # if hasattr(module, 'setup_step'): module.setup_step()
        # ```
        # will not trigger this method multiple times
        self.kernel._setup_step(**kwargs)

    def step(self, u, state, **kwargs):
        y, state = self.kernel.step(u, state, **kwargs)
        return y, state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


class S4(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=None,
        channels=1,
        bidirectional=False,
        # Arguments for position-wise feedforward components
        activation="gelu",
        postact="glu",
        hyper_act=None,
        dropout=0.0,
        tie_dropout=False,
        bottleneck=None,
        gate=None,
        transposed=True,
        verbose=False,
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF
        hyper_act: use a "hypernetwork" multiplication (experimental)
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        gate: add gated activation (GSS)
        bottleneck: reduce SSM dimension (GSS)

        See the class SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.d_model = d_model
        self.H = d_model
        self.N = d_state
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.H = self.H // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.H,
                transposed=self.transposed,
                activation=activation,
                activate=True,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=self.transposed,
                activation=activation,
                activate=True,
            )
            self.output_gate = LinearActivation(
                self.d_model * gate,
                self.d_model,
                transposed=self.transposed,
                activation=None,
                activate=False,
            )

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = SSKernel(
            self.H,
            N=self.N,
            L=self.L,
            channels=channels,
            verbose=verbose,
            **kernel_args,
        )

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.H * self.channels,
            self.d_model * (1 if self.gate is None else self.gate),
            transposed=self.transposed,
            activation=postact,
            activate=True,
        )

    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs):
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Mask out padding tokens
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert (
                isinstance(lengths, torch.Tensor)
                and lengths.ndim == 1
                and lengths.size(0) in [1, u.size(0)]
            )
            mask = torch.where(
                torch.arange(L, device=lengths.device) < lengths[:, None, None],
                1.0,
                0.0,
            )
            u = u * mask

        if self.gate is not None:
            v = self.input_gate(u)
        if self.bottleneck is not None:
            u = self.input_linear(u)

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(L=L_kernel, rate=rate, state=state)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=L_kernel + L)  # (C H L)
        u_f = torch.fft.rfft(u, n=L_kernel + L)  # (B H L)
        y_f = contract("bhl,chl->bchl", u_f, k_f)
        y = torch.fft.irfft(y_f, n=L_kernel + L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract("bhl,ch->bchl", u, self.D)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state  #
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, "b (s c) h l -> s b c h l", s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, "... c h l -> ... (c h) l")

        y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        y = self.output_linear(y)

        if self.gate is not None:
            y = self.output_gate(y * v)

        return y, next_state

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, u, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, "b c h -> b (c h)")
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model


"""
model/pool.py (from https://github.com/HazyResearch/state-spaces), used by S4Model
when pool=True (not exercised by the default build below, but part of the real
vendored source).
"""


class DownAvgPool(nn.Module):
    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        if not self.transposed:
            x = rearrange(x, "b ... d -> b d ...")
        # einops appears slower than F
        if x.ndim == 3:
            x = F.avg_pool1d(x, self.stride, self.stride)
        elif x.ndim == 4:
            x = F.avg_pool2d(x, self.stride, self.stride)
        else:
            # Reduction string e.g. "b d (l1 2) (l2 2) -> b d l1 l2"
            reduce_str = (
                "b d "
                + " ".join([f"(l{i} {self.stride})" for i in range(x.ndim - 2)])
                + " -> b d "
                + " ".join([f"l{i}" for i in range(x.ndim - 2)])
            )
            x = reduce(x, reduce_str, "mean")

        if self.expand > 1:
            x = repeat(x, "b d ... -> b (d e) ...", e=self.expand)
        if not self.transposed:
            x = rearrange(x, "b d ... -> b ... d")
        return x

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand


"""
Adapted from https://github.com/HazyResearch/state-spaces/blob/main/example.py
"""


class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=1,
        d_model=256,
        d_state=64,
        channels=1,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        l_max=1024,
        l_output=1,
        bidirectional=False,
        postact=None,  # none or 'glu'
        add_encoder=True,
        add_decoder=False,
        pool=False,
        pool_stride=1,
        pool_expand=1,
        temporal_pool=None,
        use_lengths=False,
        **kernel_args,
    ):
        super().__init__()
        self.d_input = d_input
        self.prenorm = prenorm
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.pool = pool
        self.pool_stride = pool_stride
        self.pool_expand = pool_expand
        self.temporal_pool = temporal_pool

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        if add_encoder:
            self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(
                    d_model=d_model,
                    l_max=l_max,
                    bidirectional=bidirectional,
                    postact=postact,
                    dropout=dropout,
                    transposed=True,
                    verbose=False,
                    channels=channels,
                    **kernel_args,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        if self.pool:
            self.pool_layer = DownAvgPool(
                d_input=d_model, stride=pool_stride, expand=pool_expand, transposed=True
            )

        # Linear decoder
        if add_decoder:
            if temporal_pool == "mean":
                self.decoder = nn.Linear(d_model, d_output)
            else:
                self.decoder = SequenceDecoder(
                    d_model=d_model,
                    d_output=d_output,
                    l_output=l_output,
                    use_lengths=use_lengths,
                    mode=temporal_pool,
                )

    def forward(self, x, lengths=None):
        """
        x: shape is (B, L, d_input)
        """
        assert x.shape[-1] == self.d_input

        if self.add_encoder:
            x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=lengths)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            if self.pool:
                x = self.pool_layer(x)

        x = x.transpose(-1, -2)  # (B, L, d_model)

        # Decode the outputs
        if self.add_decoder:
            if self.temporal_pool == "mean":
                if lengths is None:
                    x = torch.mean(x, dim=1)  # (B, d_model)
                else:
                    x = torch.stack(
                        [
                            torch.mean(out[:length, :], dim=0)
                            for out, length in zip(torch.unbind(x, dim=0), lengths)
                        ],
                        dim=0,
                    )  # (B, d_model)
                    x = self.decoder(x)
            else:
                x = self.decoder(x)  # (B, L, d_model) -> (B, L_out, d_output)

            if x.shape[1] == 1:
                x = x.squeeze(1)

        return x


VERY_SMALL_NUMBER = 1e-12
INF = 1e20


class GraphLearner(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_nodes,
        num_heads=1,
        embed_dim=10,
        metric_type="self_attention",
    ):
        super(GraphLearner, self).__init__()

        self.num_nodes = num_nodes
        self.metric_type = metric_type

        if metric_type == "weighted_cosine":
            self.weight_tensor = torch.Tensor(num_heads, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

        elif metric_type == "self_attention":
            self.linear_Q = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_K = nn.Linear(input_size, hidden_size, bias=False)

        elif metric_type == "cosine":  # no learnable params
            pass

        elif metric_type == "adaptive":
            # for adaptive GSL, each "head" is for one graph within a temporal resolution
            if num_heads > 1:
                self.E1 = torch.FloatTensor(num_heads, num_nodes, embed_dim)
            else:
                self.E1 = torch.FloatTensor(num_nodes, embed_dim)
            self.E1 = nn.Parameter(nn.init.xavier_uniform_(self.E1))

        else:
            raise ValueError("Unknown metric_type: {}".format(metric_type))

    def forward(self, context, attn_mask=None, batch_size=None):
        """
        Args:
            context: (batch, num_nodes, dim)
            attn_mask: (batch, num_nodes, num_nodes), 0 will be masked out as 0 in attention
        Returns:
            attention: (batch, num_nodes, num_nodes)
        """
        if self.metric_type == "weighted_cosine":
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            attention[attention < 0] = 0

            # optional masking
            markoff_value = 0
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attention = attention.masked_fill_(
                    torch.logical_not(attn_mask.bool()), markoff_value
                )

        elif self.metric_type == "self_attention":
            Q = self.linear_Q(context)
            K = self.linear_K(context)

            attention = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])

            # optional masking
            markoff_value = -INF
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attention = attention.masked_fill_(
                    torch.logical_not(attn_mask.bool()), markoff_value
                )

            attention = torch.softmax(attention, dim=-1)

        elif self.metric_type == "cosine":
            context_norm = F.normalize(context.unsqueeze(0), p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            attention[attention < 0] = 0

            # optional masking
            markoff_value = 0
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attention = attention.masked_fill_(
                    torch.logical_not(attn_mask.bool()), markoff_value
                )

        elif self.metric_type == "adaptive":
            attention = F.leaky_relu(torch.matmul(self.E1, self.E1.transpose(-1, -2)))

            # repeat by batch size
            if len(self.E1.shape) == 3:
                attention = attention.repeat(batch_size, 1, 1, 1)
            else:
                attention = attention.repeat(batch_size, 1, 1)

            # optional masking
            markoff_value = -INF
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attention = attention.masked_fill_(
                    torch.logical_not(attn_mask.bool()), markoff_value
                )

            attention = torch.softmax(attention, dim=-1).reshape(-1, self.num_nodes, self.num_nodes)

        else:
            raise NotImplementedError()

        return attention


def calculate_cosine_decay_weight(max_weight, epoch, epoch_total, min_weight=0):
    """
    Calculate decayed weight (hyperparameter) based on cosine annealing schedule
    Referred to https://arxiv.org/abs/1608.03983
    and https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    """
    curr_weight = min_weight + 0.5 * (max_weight - min_weight) * (
        1 + math.cos(epoch / epoch_total * math.pi)
    )
    return curr_weight


def calculate_normalized_laplacian(adj):
    """
    Args:
        adj: torch tensor, shape (batch, num_nodes, num_nodes)

    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    D = diag(A)
    """

    batch, num_nodes, _ = adj.shape
    d = adj.sum(-1)  # (batch, num_nodes)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (batch, num_nodes, num_nodes)

    identity = (torch.eye(num_nodes).unsqueeze(0).repeat(batch, 1, 1)).to(
        adj.device
    )  # (batch, num_nodes, num_nodes)
    normalized_laplacian = identity - torch.matmul(
        torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt
    )

    return normalized_laplacian


def feature_smoothing(adj, X):
    # normalized laplacian
    L = calculate_normalized_laplacian(adj)

    feature_dim = X.shape[-1]
    mat = torch.matmul(torch.matmul(X.transpose(1, 2), L), X) / (feature_dim**2)
    loss = mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
    return loss


def get_knn_graph(x, k, dist_measure="cosine", undirected=True):
    if dist_measure == "euclidean":
        dist = torch.cdist(x, x, p=2.0)
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        knn_val, knn_ind = torch.topk(dist, k, dim=-1, largest=False)  # smallest distances
    elif dist_measure == "cosine":
        norm = torch.norm(x, dim=-1, p="fro")[:, :, None]
        x_norm = x / norm
        dist = torch.matmul(x_norm, x_norm.transpose(1, 2))
        knn_val, knn_ind = torch.topk(dist, k, dim=-1, largest=True)  # largest similarities
    else:
        raise NotImplementedError

    adj_mat = (torch.ones_like(dist) * 0).scatter_(-1, knn_ind, knn_val).to(x.device)

    adj_mat = torch.clamp(adj_mat, min=0.0)  # remove negatives

    if undirected:
        adj_mat = (adj_mat + adj_mat.transpose(1, 2)) / 2

    # add self-loop
    I = (  # noqa: E741
        torch.eye(adj_mat.shape[-1], adj_mat.shape[-1])
        .unsqueeze(0)
        .repeat(adj_mat.shape[0], 1, 1)
        .to(bool)
    ).to(x.device)
    adj_mat = adj_mat * (~I) + I

    # to sparse graph
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat)

    return edge_index, edge_weight, adj_mat


def prune_adj_mat(adj_mat, num_nodes, method="thresh", edge_top_perc=None, knn=None, thresh=None):
    if method == "thresh":
        sorted, indices = torch.sort(
            adj_mat.reshape(-1, num_nodes * num_nodes),
            dim=-1,
            descending=True,
        )
        K = int((num_nodes**2) * edge_top_perc)
        mask = adj_mat > sorted[:, K].unsqueeze(1).unsqueeze(2)
        adj_mat = adj_mat * mask
    elif method == "knn":
        knn_val, knn_ind = torch.topk(adj_mat, knn, dim=-1, largest=True)
        adj_mat = (torch.ones_like(adj_mat) * 0).scatter_(-1, knn_ind, knn_val).to(adj_mat.device)
    elif method == "thresh_abs":
        mask = (adj_mat > thresh).float()
        adj_mat = adj_mat * mask
    else:
        raise NotImplementedError

    return adj_mat


class GraphS4mer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_nodes,
        dropout,
        num_temporal_layers,
        g_conv,
        num_gnn_layers,
        hidden_dim,
        max_seq_len,
        resolution,
        state_dim=64,
        channels=1,
        temporal_model="s4",
        bidirectional=False,
        prenorm=False,
        postact=None,
        metric="self_attention",
        adj_embed_dim=10,
        gin_mlp=False,
        train_eps=False,
        prune_method="thresh",
        edge_top_perc=0.5,
        thresh=None,
        temporal_pool="mean",
        graph_pool="sum",
        activation_fn="relu",
        num_classes=1,
        undirected_graph=True,
        use_prior=False,
        K=3,
        regularizations=["feature_smoothing", "degree", "sparse"],
        residual_weight=0.0,
        decay_residual_weight=False,
        **kwargs,
    ):
        super().__init__()

        if (resolution is not None) and (max_seq_len % resolution != 0):
            raise ValueError("max_seq_len must be divisible by resolution!")

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.edge_top_perc = edge_top_perc
        self.graph_pool = graph_pool
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.metric = metric
        self.undirected_graph = undirected_graph
        self.use_prior = use_prior
        self.K = K
        self.regularizations = regularizations
        self.residual_weight = residual_weight
        self.temporal_pool = temporal_pool
        self.temporal_model = temporal_model
        self.max_seq_len = max_seq_len
        self.resolution = resolution
        self.prune_method = prune_method
        self.thresh = thresh
        self.decay_residual_weight = decay_residual_weight

        # temporal layer
        if temporal_model == "gru":
            self.t_model = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif temporal_model == "s4":
            self.t_model = S4Model(
                d_input=input_dim,
                d_model=hidden_dim,
                d_state=state_dim,
                channels=channels,
                n_layers=num_temporal_layers,
                dropout=dropout,
                prenorm=prenorm,
                l_max=max_seq_len,
                bidirectional=bidirectional,
                postact=postact,  # none or 'glu'
                add_decoder=False,
                pool=False,  # hard-coded
                temporal_pool=None,
            )

        else:
            raise NotImplementedError

        # graph learning layer
        self.attn_layers = GraphLearner(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_nodes=num_nodes,
            embed_dim=adj_embed_dim,
            metric_type=metric,
        )

        # gnn layers
        self.gnn_layers = nn.ModuleList()
        if g_conv == "graphsage":
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim, **kwargs))
        elif g_conv == "gine":
            for _ in range(num_gnn_layers):
                if gin_mlp:
                    gin_nn = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                else:
                    gin_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
                self.gnn_layers.append(
                    GINEConv(nn=gin_nn, eps=0.0, train_eps=train_eps, edge_dim=1, **kwargs)
                )
        else:
            raise NotImplementedError

        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_fn == "elu":
            self.activation = nn.ELU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        data,
        return_attention=False,
        lengths=None,
        epoch=None,
        epoch_total=None,
    ):
        """
        Args:
            data: torch geometric data object
        """
        x = data.x  # (batch * num_nodes, seq_len, 1)
        batch = x.shape[0] // self.num_nodes
        num_nodes = self.num_nodes
        _, seq_len, _ = x.shape
        batch_idx = data.batch  # noqa: F841

        if lengths is not None:
            lengths = torch.repeat_interleave(lengths, num_nodes, dim=0)

        # temporal layer
        if self.temporal_model == "s4":
            x = self.t_model(x, lengths)  # (batch * num_nodes, seq_len, hidden_dim)
        else:
            if lengths is not None:
                x = pack_padded_sequence(x, lengths, batch_first=True)
            x, _ = self.t_model(x)
            if lengths is not None:
                x, lengths = pad_packed_sequence(x)

        # get output with <resolution> as interval
        if lengths is None:
            x = x.view(batch, num_nodes, seq_len, -1)  # (batch, num_nodes, seq_len, hidden_dim)
            x_tmp = []
            num_dynamic_graphs = self.max_seq_len // self.resolution
            for t in range(num_dynamic_graphs):
                start = t * self.resolution
                stop = start + self.resolution
                curr_x = torch.mean(x[:, :, start:stop, :], dim=2)
                x_tmp.append(curr_x)
            x_tmp = torch.stack(x_tmp, dim=1)  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)
            x = x_tmp.reshape(
                -1, num_nodes, self.hidden_dim
            )  # (batch * num_dynamic_graphs, num_nodes, hidden_dim)
            del x_tmp
        else:  # for variable lengths, mean pool over actual lengths
            x = torch.stack(
                [
                    torch.mean(out[:length, :], dim=0)
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
            x = x.reshape(batch, num_nodes, -1)  # (batch, num_nodes, hidden_dim)
            num_dynamic_graphs = 1

        # get initial adj
        if self.use_prior:
            adj_mat = torch_geometric.utils.to_dense_adj(
                edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr
            )
        else:
            # knn cosine graph
            edge_index, edge_weight, adj_mat = get_knn_graph(
                x,
                self.K,
                dist_measure="cosine",
                undirected=self.undirected_graph,
            )
            edge_index = edge_index.to(x.device)
            edge_weight = edge_weight.to(x.device)
            adj_mat = adj_mat.to(x.device)

        # learn adj mat
        attn_weight = self.attn_layers(x)  # (batch*num_dynamic_graphs, num_nodes, num_nodes)

        # to undirected
        if self.undirected_graph:
            attn_weight = (attn_weight + attn_weight.transpose(1, 2)) / 2
        raw_attn_weight = attn_weight.clone()

        # add residual
        if len(adj_mat.shape) == 2:
            adj_mat = torch.cat([adj_mat] * num_dynamic_graphs * batch, dim=0)
        elif len(adj_mat.shape) == 3 and (adj_mat.shape != attn_weight.shape):
            adj_mat = torch.cat([adj_mat] * num_dynamic_graphs, dim=0)

        # knn graph weight (aka residual weight) decay
        if self.decay_residual_weight:
            assert (epoch is not None) and (epoch_total is not None)
            residual_weight = calculate_cosine_decay_weight(
                max_weight=self.residual_weight, epoch=epoch, epoch_total=epoch_total, min_weight=0
            )
        else:
            residual_weight = self.residual_weight
        # add knn graph
        adj_mat = residual_weight * adj_mat + (1 - residual_weight) * attn_weight

        # prune graph
        adj_mat = prune_adj_mat(
            adj_mat,
            num_nodes,
            method=self.prune_method,
            edge_top_perc=self.edge_top_perc,
            knn=self.K,
            thresh=self.thresh,
        )

        # regularization loss
        reg_losses = self.regularization_loss(x, adj=adj_mat)

        # back to sparse graph
        edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat)

        # add self-loop
        edge_index, edge_weight = torch_geometric.utils.remove_self_loops(
            edge_index=edge_index, edge_attr=edge_weight
        )
        edge_index, edge_weight = torch_geometric.utils.add_self_loops(
            edge_index=edge_index,
            edge_attr=edge_weight,
            fill_value=1,
        )

        x = x.view(
            batch * num_dynamic_graphs * num_nodes, -1
        )  # (batch * num_dynamic_graphs * num_nodes, hidden_dim)
        for i in range(len(self.gnn_layers)):
            # gnn layer
            x = self.gnn_layers[i](x, edge_index=edge_index, edge_attr=edge_weight.reshape(-1, 1))
            x = self.dropout(
                self.activation(x)
            )  # (batch * num_dynamic_graphs * num_nodes, hidden_dim)
        x = x.view(batch * num_dynamic_graphs, num_nodes, -1).view(
            batch, num_dynamic_graphs, num_nodes, -1
        )  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)

        # temporal pool
        if self.temporal_pool == "last":
            x = x[:, -1, :, :]  # (batch, num_nodes, hidden_dim)
        elif self.temporal_pool == "mean":
            x = torch.mean(x, dim=1)
        else:
            raise NotImplementedError

        # graph pool
        if self.graph_pool == "sum":
            x = torch.sum(x, dim=1)  # (batch, hidden_dim)
        elif self.graph_pool == "mean":
            x = torch.mean(x, dim=1)
        elif self.graph_pool == "max":
            x, _ = torch.max(x, dim=1)
        else:
            raise NotImplementedError
        feat = x.clone()

        # classifier
        x = self.classifier(x)

        if return_attention:
            return (
                x,
                reg_losses,
                raw_attn_weight.reshape(batch, num_dynamic_graphs, num_nodes, num_nodes),
                adj_mat.reshape(batch, num_dynamic_graphs, num_nodes, num_nodes),
                feat,
            )
        else:
            return x, reg_losses

    def regularization_loss(self, x, adj, reduce="mean"):
        """
        Referred to https://github.com/hugochan/IDGL/blob/master/src/core/model_handler.py#L1116
        """
        batch, num_nodes, _ = x.shape
        n = num_nodes

        loss = {}

        if "feature_smoothing" in self.regularizations:
            curr_loss = feature_smoothing(adj=adj, X=x) / (n**2)
            if reduce == "mean":
                loss["feature_smoothing"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["feature_smoothing"] = torch.sum(curr_loss)
            else:
                loss["feature_smoothing"] = curr_loss

        if "degree" in self.regularizations:
            ones = torch.ones(batch, num_nodes, 1).to(x.device)
            curr_loss = -(1 / n) * torch.matmul(
                ones.transpose(1, 2), torch.log(torch.matmul(adj, ones))
            ).squeeze(-1).squeeze(-1)
            if reduce == "mean":
                loss["degree"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["degree"] = torch.sum(curr_loss)
            else:
                loss["degree"] = curr_loss

        if "sparse" in self.regularizations:
            curr_loss = 1 / (n**2) * torch.pow(torch.norm(adj, p="fro", dim=(-1, -2)), 2)

            if reduce == "mean":
                loss["sparse"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["sparse"] = torch.sum(curr_loss)
            else:
                loss["sparse"] = curr_loss

        if "symmetric" in self.regularizations and self.undirected_graph:
            curr_loss = torch.norm(adj - adj.transpose(1, 2), p="fro", dim=(-1, -2))
            if reduce == "mean":
                loss["symmetric"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["symmetric"] = torch.sum(curr_loss)
            else:
                loss["symmetric"] = curr_loss

        return loss


"""Model for regression/forecasting task"""


# ---- build/example helpers (staging-only, not part of the vendored source) ----


def build_graphs4mer():
    torch.manual_seed(0)
    return GraphS4mer(
        input_dim=1,
        num_nodes=6,
        dropout=0.0,
        num_temporal_layers=1,
        g_conv="gine",
        num_gnn_layers=1,
        hidden_dim=16,
        max_seq_len=32,
        resolution=16,
        state_dim=8,
        channels=1,
        temporal_model="s4",
        bidirectional=False,
        prenorm=False,
        postact=None,
        metric="self_attention",
        adj_embed_dim=10,
        gin_mlp=False,
        train_eps=False,
        prune_method="thresh",
        edge_top_perc=0.5,
        temporal_pool="mean",
        graph_pool="sum",
        activation_fn="relu",
        num_classes=2,
        undirected_graph=True,
        use_prior=False,
        K=2,
        regularizations=["feature_smoothing", "degree", "sparse"],
        residual_weight=0.0,
    ).eval()


def example_input_graphs4mer():
    torch.manual_seed(0)
    batch, num_nodes, seq_len = 2, 6, 32
    x = torch.randn(batch * num_nodes, seq_len, 1)
    batch_idx = torch.repeat_interleave(torch.arange(batch), num_nodes)
    data = torch_geometric.data.Data(x=x)
    data.batch = batch_idx
    data.num_graphs = batch
    return (data,)


MENAGERIE_ENTRIES = [
    ("GraphS4mer", build_graphs4mer, example_input_graphs4mer, 2023, "vendored-pytorch"),
]
