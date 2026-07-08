# SOURCE: vendored from Xiaoxun-Gong/DeepH-E3 @ main
# Files combined below, each still carrying its own upstream header/attribution where present:
#   deephe3/model.py (Net + submodules: get_gate_nonlin, EquiConv, NodeUpdateBlock, EdgeUpdateBlock)
#   deephe3/e3modules.py (sort_irreps, e3LayerNorm, e3ElementWise, SkipConnection,
#     SeparateWeightTensorProduct, SelfTp, SphericalBasis; Rotate/e3TensorDecomp kept for
#     import-completeness but unused by the forward path exercised here)
#   deephe3/utils.py (only flt2cplx, irreps_from_l1l2 -- the two helpers e3modules.py imports)
#   deephe3/from_dimenet/basis_utils.py (bessel_basis + its sympy helpers)
#   deephe3/from_nequip/cutoffs.py (PolynomialCutoff)
#   deephe3/from_nequip/tp_utils.py (tp_path_exists)
#   deephe3/from_schnetpack/acsf.py (GaussianBasis)
#
# DeepH-E3 (https://github.com/Xiaoxun-Gong/DeepH-E3) is Xiaoxun Gong's separate official
# E(3)-equivariant successor to DeepH-pack: an e3nn tensor-product graph network that predicts
# DFT Hamiltonian matrices with full O(3) equivariance and (optional, not exercised here)
# spin-orbit coupling support. It is architecturally distinct from DeepH-pack's HGNN
# (see L849_deeph.py) -- different repo, different message-passing/equivariance mechanism
# (e3nn irreps + tensor products vs. CGConv/PAINN + scalar features) -- so it is vendored as
# its own module rather than folded into the DeepH-pack one.
#
# Only imports/relative-path fixes were made to let the real classes run standalone in the
# base torchlens env (torch, torch_geometric, torch_scatter, e3nn, sympy, scipy, numpy are all
# installed). No architecture was altered.

import os

import numpy as np
import sympy as sym
import torch
import torch.nn.functional as F
from e3nn.nn import Extract, Gate
from e3nn.o3 import (
    FullyConnectedTensorProduct,
    Irrep,
    Irreps,
    Linear,
    SphericalHarmonics,
    TensorProduct,
    matrix_to_angles,
    wigner_3j,
)
from scipy import special as sp
from scipy.optimize import brentq
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from torch_geometric.utils import degree
from torch_scatter import scatter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from deephe3/from_schnetpack/acsf.py
# ---------------------------------------------------------------------------


class GaussianBasis(nn.Module):
    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super(GaussianBasis, self).__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        if not self.centered:
            coeff = -0.5 / torch.pow(self.width, 2)
            diff = distances[:, None] - self.offsets[None, :]
        else:
            coeff = -0.5 / torch.pow(self.offsets, 2)
            diff = distances[:, None]
        gauss = torch.exp(coeff * torch.pow(diff, 2))
        return gauss


# ---------------------------------------------------------------------------
# from deephe3/from_nequip/cutoffs.py
# ---------------------------------------------------------------------------


class PolynomialCutoff(nn.Module):
    def __init__(self, r_max, p=6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123"""
        super(PolynomialCutoff, self).__init__()
        self.register_buffer("p", torch.Tensor([p]))
        self.register_buffer("r_max", torch.Tensor([r_max]))

    def forward(self, x):
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1.0)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2.0)
        )
        envelope *= (x < self.r_max).float()
        return envelope


# ---------------------------------------------------------------------------
# from deephe3/from_nequip/tp_utils.py
# ---------------------------------------------------------------------------


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


# ---------------------------------------------------------------------------
# from deephe3/from_dimenet/basis_utils.py
# ---------------------------------------------------------------------------


def Jn(r, n):
    """numerical spherical bessel functions of order n"""
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def Jn_zeros(n, k):
    """Compute the first k zeros of the spherical bessel functions up to order n (excluded)"""
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]
    return zerosj


def spherical_bessel_formulas(n):
    """Computes the sympy formulas for the spherical bessel functions up to order n (excluded)"""
    x = sym.symbols("x")
    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """Compute the sympy formulas for the normalized and rescaled spherical bessel functions up
    to order n (excluded) and maximum frequency k (excluded)."""
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


# ---------------------------------------------------------------------------
# from deephe3/utils.py (only the two helpers e3modules.py imports)
# ---------------------------------------------------------------------------


def flt2cplx(flt_dtype):
    if flt_dtype == torch.float32:
        cplx_dtype = torch.complex64
    elif flt_dtype == torch.float64:
        cplx_dtype = torch.complex128
    elif flt_dtype == np.float32:
        cplx_dtype = np.complex64
    elif flt_dtype == np.float64:
        cplx_dtype = np.complex128
    else:
        raise NotImplementedError(f"Unsupported float dtype: {flt_dtype}")
    return cplx_dtype


def irreps_from_l1l2(l1, l2, mul, spinful, no_parity=False):
    r"""non-spinful example: l1=1, l2=2 (1x2) ->
    required_irreps_full=1+2+3, required_irreps=1+2+3, required_irreps_x1=None"""
    p = 1
    if not no_parity:
        p = (-1) ** (l1 + l2)
    required_ls = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = Irreps([(mul, (l, p)) for l in required_ls])
    required_irreps_full = required_irreps
    required_irreps_x1 = None
    if spinful:
        required_irreps_x1 = []
        for _, ir in required_irreps:
            required_ls_irx1 = range(abs(ir.l - 1), ir.l + 1 + 1)
            irx1 = Irreps([(mul, (l, p)) for l in required_ls_irx1])
            required_irreps_x1.append(irx1)
            required_irreps_full += irx1
    return required_irreps_full, required_irreps, required_irreps_x1


# ---------------------------------------------------------------------------
# from deephe3/e3modules.py (Rotate/e3TensorDecomp kept for import-completeness)
# ---------------------------------------------------------------------------


class Rotate:
    def __init__(self, default_dtype_torch, device_torch="cpu", spinful=False):
        sqrt_2 = 1.4142135623730951
        self.spinful = spinful
        if spinful:
            assert default_dtype_torch in [torch.complex64, torch.complex128]
        else:
            assert default_dtype_torch in [torch.float32, torch.float64]

        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch.cfloat, device=device_torch),
            1: torch.tensor(
                [[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]],
                dtype=torch.cfloat,
                device=device_torch,
            ),
            2: torch.tensor(
                [
                    [0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                    [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                    [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                ],
                dtype=torch.cfloat,
                device=device_torch,
            ),
            3: torch.tensor(
                [
                    [0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                    [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                    [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                    [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                    [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                ],
                dtype=torch.cfloat,
                device=device_torch,
            ),
        }
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=default_dtype_torch).to(device=device_torch),
            1: torch.eye(3, dtype=default_dtype_torch)[[1, 2, 0]].to(device=device_torch),
            2: torch.eye(5, dtype=default_dtype_torch)[[2, 4, 0, 3, 1]].to(device=device_torch),
            3: torch.eye(7, dtype=default_dtype_torch)[[6, 4, 2, 0, 1, 3, 5]].to(
                device=device_torch
            ),
        }
        self.Us_wiki2openmx = {k: v.T for k, v in self.Us_openmx2wiki.items()}
        if spinful:
            self.Us_openmx2wiki_sp = {}
            for k, v in self.Us_openmx2wiki.items():
                self.Us_openmx2wiki_sp[k] = torch.block_diag(v, v)

        self.dtype = default_dtype_torch

    def rotate_e3nn_v(self, v, R, l, order_xyz=True):
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        return v @ Irrep(l, 1).D_from_matrix(R_e3nn)

    def rotate_openmx_H(self, H, R, l_left, l_right, order_xyz=True):
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        return (
            self.Us_openmx2wiki[l_left].T
            @ Irrep(l_left, 1).D_from_matrix(R_e3nn).transpose(-1, -2)
            @ self.Us_openmx2wiki[l_left]
            @ H
            @ self.Us_openmx2wiki[l_right].T
            @ Irrep(l_right, 1).D_from_matrix(R_e3nn)
            @ self.Us_openmx2wiki[l_right]
        )

    def wiki2openmx_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]

    def openmx2wiki_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left] @ H @ self.Us_openmx2wiki[l_right].T

    def openmx2wiki_left_right(self, orbital_types_left, orbital_types_right):
        if isinstance(orbital_types_left, int):
            orbital_types_left = [orbital_types_left]
        if isinstance(orbital_types_right, int):
            orbital_types_right = [orbital_types_right]
        openmx2wiki_left = torch.block_diag(*[self.Us_openmx2wiki[l] for l in orbital_types_left])
        openmx2wiki_right = torch.block_diag(*[self.Us_openmx2wiki[l] for l in orbital_types_right])
        if self.spinful:
            openmx2wiki_left = torch.block_diag(openmx2wiki_left, openmx2wiki_left)
            openmx2wiki_right = torch.block_diag(openmx2wiki_right, openmx2wiki_right)
        return openmx2wiki_left, openmx2wiki_right

    def rotate_matrix_convert(self, R):
        return torch.eye(3)[[1, 2, 0]] @ R @ torch.eye(3)[[1, 2, 0]].T


class sort_irreps(torch.nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()

        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = Extract(irreps_in, irreps_out_list, instructions)

        irreps_in_list = [((mul, ir),) for mul, ir in irreps_in]
        instructions_inv = [(i,) for i in sorted_irreps.p]
        self.extr_inv = Extract(sorted_irreps.irreps, irreps_in_list, instructions_inv)

        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps.simplify()

    def forward(self, x):
        extracted = self.extr(x)
        return torch.cat(extracted, dim=-1)

    def inverse(self, x):
        extracted_inv = self.extr_inv(x)
        return torch.cat(extracted_inv, dim=-1)


class e3LayerNorm(nn.Module):
    def __init__(
        self,
        irreps_in,
        eps=1e-5,
        affine=True,
        normalization="component",
        subtract_mean=True,
        divide_norm=False,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.eps = eps

        if affine:
            ib, iw = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar():
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        assert normalization in ["component", "norm"]
        self.normalization = normalization

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            self.weight.data.fill_(1)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64)

        batch_size = int(batch.max()) + 1
        batch_degree = degree(batch, batch_size, dtype=torch.int64).clamp_(min=1).to(dtype=x.dtype)

        out = []
        ix = 0
        for index, (mul, ir) in enumerate(self.irreps_in):
            field = x[:, ix : ix + mul * ir.dim].reshape(-1, mul, ir.dim)

            if self.subtract_mean or ir.l == 0:
                mean = (
                    scatter(field, batch, dim=0, dim_size=batch_size, reduce="add").mean(
                        dim=1, keepdim=True
                    )
                    / batch_degree[:, None, None]
                )
                field = field - mean[batch]

            if self.divide_norm or ir.l == 0:
                norm = scatter(
                    field.abs().pow(2), batch, dim=0, dim_size=batch_size, reduce="mean"
                ).mean(dim=[1, 2], keepdim=True)
                if self.normalization == "norm":
                    norm = norm * ir.dim
                field = field / (norm.sqrt()[batch] + self.eps)

            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]

            out.append(field.reshape(-1, mul * ir.dim))
            ix += mul * ir.dim

        out = torch.cat(out, dim=-1)
        return out


class e3ElementWise:
    def __init__(self, irreps_in):
        self.irreps_in = Irreps(irreps_in)
        len_weight = 0
        for mul, ir in self.irreps_in:
            len_weight += mul
        self.len_weight = len_weight

    def __call__(self, x: torch.Tensor, weight: torch.Tensor):
        ix = 0
        iw = 0
        out = []
        for mul, ir in self.irreps_in:
            field = x[:, ix : ix + mul * ir.dim]
            field = field.reshape(-1, mul, ir.dim)
            field = field * weight[:, iw : iw + mul][:, :, None]
            field = field.reshape(-1, mul * ir.dim)
            ix += mul * ir.dim
            iw += mul
            out.append(field)
        return torch.cat(out, dim=-1)


class SkipConnection(nn.Module):
    def __init__(self, irreps_in, irreps_out, is_complex=False):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        self.sc = None
        if irreps_in == irreps_out:
            self.sc = None
        else:
            self.sc = Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, old, new):
        if self.sc is not None:
            old = self.sc(old)
        return old + new


class SelfTp(nn.Module):
    def __init__(self, irreps_in, irreps_out, **kwargs):
        """z_i = W'_{ij}x_j W''_{ik}x_k (k>=j)"""
        super().__init__()
        assert not kwargs.pop("internal_weights", False)
        assert kwargs.pop("shared_weights", True)

        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)

        instr_tp = []
        weights1, weights2 = [], []
        for i1, (mul1, ir1) in enumerate(irreps_in):
            for i2 in range(i1, len(irreps_in)):
                mul2, ir2 = irreps_in[i2]
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mul_out)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mul_out)))
                        instr_tp.append((i1, i2, i_out, "uvw", True, 1.0))

        self.tp = TensorProduct(
            irreps_in,
            irreps_in,
            irreps_out,
            instr_tp,
            internal_weights=False,
            shared_weights=True,
            **kwargs,
        )
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def forward(self, x):
        weights = []
        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))
        weights = torch.cat(weights)
        return self.tp(x, x, weights)


class SeparateWeightTensorProduct(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        """z_i = W'_{ij}x_j W''_{ik}y_k"""
        super().__init__()
        assert not kwargs.pop("internal_weights", False)
        assert kwargs.pop("shared_weights", True)

        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        instr_tp = []
        weights1, weights2 = [], []
        for i1, (mul1, ir1) in enumerate(irreps_in1):
            for i2, (mul2, ir2) in enumerate(irreps_in2):
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mul_out)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mul_out)))
                        instr_tp.append((i1, i2, i_out, "uvw", True, 1.0))

        self.tp = TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr_tp,
            internal_weights=False,
            shared_weights=True,
            **kwargs,
        )
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def forward(self, x1, x2):
        weights = []
        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))
        weights = torch.cat(weights)
        return self.tp(x1, x2, weights)


class SphericalBasis(nn.Module):
    def __init__(self, target_irreps, rcutoff, eps=1e-7, dtype=torch.get_default_dtype()):
        super().__init__()
        target_irreps = Irreps(target_irreps)

        self.sh = SphericalHarmonics(
            irreps_out=target_irreps,
            normalize=True,
            normalization="component",
        )

        max_order = max(map(lambda x: x[1].l, target_irreps))
        max_freq = max(map(lambda x: x[0], target_irreps))

        basis = bessel_basis(max_order + 1, max_freq)
        lambdify_torch = {"sin": torch.sin, "cos": torch.cos}
        x = sym.symbols("x")
        funcs = []
        for mul, ir in target_irreps:
            for freq in range(mul):
                funcs.append(sym.lambdify([x], basis[ir.l][freq], [lambdify_torch]))

        self.bessel_funcs = funcs
        self.multiplier = e3ElementWise(target_irreps)
        self.dtype = dtype
        self.cutoff = PolynomialCutoff(rcutoff, p=6)
        self.register_buffer("rcutoff", torch.Tensor([rcutoff]))
        self.irreps_out = target_irreps
        self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, length, direction):
        sh = self.sh(direction).type(self.dtype)
        sbf = torch.stack(
            [f((length + self.eps) / self.rcutoff) for f in self.bessel_funcs], dim=-1
        )
        return self.multiplier(sh, sbf) * self.cutoff(length)[:, None]


# ---------------------------------------------------------------------------
# from deephe3/model.py
# ---------------------------------------------------------------------------

epsilon = 1e-8


def get_gate_nonlin(
    irreps_in1,
    irreps_in2,
    irreps_out,
    act={1: torch.nn.functional.silu, -1: torch.tanh},
    act_gates={1: torch.sigmoid, -1: torch.tanh},
):
    irreps_scalars = Irreps(
        [
            (mul, ir)
            for mul, ir in irreps_out
            if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
        ]
    ).simplify()
    irreps_gated = Irreps(
        [
            (mul, ir)
            for mul, ir in irreps_out
            if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
        ]
    ).simplify()
    if irreps_gated.dim > 0:
        if tp_path_exists(irreps_in1, irreps_in2, "0e"):
            ir = "0e"
        elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
            ir = "0o"
        else:
            raise ValueError(
                f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} is unable to produce gates needed for irreps_gated={irreps_gated}"
            )
    else:
        ir = None
    irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

    gate_nonlin = Gate(
        irreps_scalars,
        [act[ir.p] for _, ir in irreps_scalars],
        irreps_gates,
        [act_gates[ir.p] for _, ir in irreps_gates],
        irreps_gated,
    )

    return gate_nonlin


class EquiConv(nn.Module):
    def __init__(
        self,
        fc_len_in,
        irreps_in1,
        irreps_in2,
        irreps_out,
        norm="",
        nonlin=True,
        act={1: torch.nn.functional.silu, -1: torch.tanh},
        act_gates={1: torch.sigmoid, -1: torch.tanh},
    ):
        super(EquiConv, self).__init__()

        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, act, act_gates)
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps(
                [(mul, ir) for mul, ir in irreps_out if tp_path_exists(irreps_in1, irreps_in2, ir)]
            )

        self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_tp_out)

        if nonlin:
            self.cfconv = e3ElementWise(self.nonlin.irreps_out)
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.cfconv = e3ElementWise(irreps_tp_out)
            self.irreps_out = irreps_tp_out

        linear_act = nn.SiLU()
        self.fc = nn.Sequential(
            nn.Linear(fc_len_in, 64),
            linear_act,
            nn.Linear(64, 64),
            linear_act,
            nn.Linear(64, self.cfconv.len_weight),
        )

        self.norm = None
        if norm:
            if norm == "e3LayerNorm":
                self.norm = e3LayerNorm(self.cfconv.irreps_in)
            else:
                raise ValueError(f"unknown norm: {norm}")

    def forward(self, fea_in1, fea_in2, fea_weight, batch_edge):
        z = self.tp(fea_in1, fea_in2)

        if self.nonlin is not None:
            z = self.nonlin(z)

        weight = self.fc(fea_weight)
        z = self.cfconv(z, weight)

        if self.norm is not None:
            z = self.norm(z, batch_edge)

        return z


class NodeUpdateBlock(nn.Module):
    def __init__(
        self,
        num_species,
        fc_len_in,
        irreps_sh,
        irreps_in_node,
        irreps_out_node,
        irreps_in_edge,
        act,
        act_gates,
        use_selftp=False,
        use_sc=True,
        concat=True,
        only_ij=False,
        nonlin=False,
        norm="e3LayerNorm",
        if_sort_irreps=False,
    ):
        super(NodeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_sh = Irreps(irreps_sh)
        irreps_out_node = Irreps(irreps_out_node)
        irreps_in_edge = Irreps(irreps_in_edge)

        if concat:
            irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
            if if_sort_irreps:
                self.sort = sort_irreps(irreps_in1)
                irreps_in1 = self.sort.irreps_out
        else:
            irreps_in1 = irreps_in_node
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_node, irreps_out=irreps_in_node, biases=True)

        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_node, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_node
            conv_nonlin = True

        self.conv = EquiConv(
            fc_len_in,
            irreps_in1,
            irreps_in2,
            irreps_conv_out,
            nonlin=conv_nonlin,
            act=act,
            act_gates=act_gates,
        )
        self.lin_post = Linear(
            irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True
        )

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.sc = None
        if use_sc:
            self.sc = FullyConnectedTensorProduct(
                irreps_in_node, f"{num_species}x0e", self.conv.irreps_out
            )

        self.norm = None
        if norm:
            if norm == "e3LayerNorm":
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f"unknown norm: {norm}")

        self.skip_connect = SkipConnection(irreps_in_node, self.irreps_out)

        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)

        self.irreps_in_node = irreps_in_node
        self.use_sc = use_sc
        self.concat = concat
        self.only_ij = only_ij
        self.if_sort_irreps = if_sort_irreps

    def forward(
        self,
        node_fea,
        node_one_hot,
        edge_sh,
        edge_fea,
        edge_length_embedded,
        edge_index,
        batch,
        selfloop_edge,
        edge_length,
    ):
        node_fea_old = node_fea

        if self.use_sc:
            node_self_connection = self.sc(node_fea, node_one_hot)

        node_fea = self.lin_pre(node_fea)

        index_i = edge_index[0]
        index_j = edge_index[1]
        if self.concat:
            fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
            if self.if_sort_irreps:
                fea_in = self.sort(fea_in)
            edge_update = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        else:
            edge_update = self.conv(
                node_fea[index_j], edge_sh, edge_length_embedded, batch[edge_index[0]]
            )

        node_fea = scatter(edge_update, index_i, dim=0, dim_size=node_fea.shape[0], reduce="add")
        if self.only_ij:
            node_fea = node_fea + scatter(
                edge_update[~selfloop_edge],
                index_j[~selfloop_edge],
                dim=0,
                dim_size=node_fea.shape[0],
                reduce="add",
            )

        node_fea = self.lin_post(node_fea)

        if self.use_sc:
            node_fea = node_fea + node_self_connection

        if self.nonlin is not None:
            node_fea = self.nonlin(node_fea)

        if self.norm is not None:
            node_fea = self.norm(node_fea, batch)

        node_fea = self.skip_connect(node_fea_old, node_fea)

        if self.self_tp is not None:
            node_fea = self.self_tp(node_fea)

        return node_fea


class EdgeUpdateBlock(nn.Module):
    def __init__(
        self,
        num_species,
        fc_len_in,
        irreps_sh,
        irreps_in_node,
        irreps_in_edge,
        irreps_out_edge,
        act,
        act_gates,
        use_selftp=False,
        use_sc=True,
        init_edge=False,
        nonlin=False,
        norm="e3LayerNorm",
        if_sort_irreps=False,
    ):
        super(EdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        if if_sort_irreps:
            self.sort = sort_irreps(irreps_in1)
            irreps_in1 = self.sort.irreps_out
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge, biases=True)

        self.nonlin = None
        self.lin_post = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_nonlin = True

        self.conv = EquiConv(
            fc_len_in,
            irreps_in1,
            irreps_in2,
            irreps_conv_out,
            nonlin=conv_nonlin,
            act=act,
            act_gates=act_gates,
        )
        self.lin_post = Linear(
            irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True
        )

        if use_sc:
            self.sc = FullyConnectedTensorProduct(
                irreps_in_edge, f"{num_species**2}x0e", self.conv.irreps_out
            )

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.norm = None
        if norm:
            if norm == "e3LayerNorm":
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f"unknown norm: {norm}")

        self.skip_connect = SkipConnection(irreps_in_edge, self.irreps_out)

        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)

        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge

    def forward(
        self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch
    ):
        if not self.init_edge:
            edge_fea_old = edge_fea
            if self.use_sc:
                edge_self_connection = self.sc(edge_fea, edge_one_hot)
            edge_fea = self.lin_pre(edge_fea)

        index_i = edge_index[0]
        index_j = edge_index[1]
        fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
        if self.if_sort_irreps:
            fea_in = self.sort(fea_in)
        edge_fea = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])

        edge_fea = self.lin_post(edge_fea)

        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection

        if self.nonlin is not None:
            edge_fea = self.nonlin(edge_fea)

        if self.norm is not None:
            edge_fea = self.norm(edge_fea, batch[edge_index[0]])

        if not self.init_edge:
            edge_fea = self.skip_connect(edge_fea_old, edge_fea)

        if self.self_tp is not None:
            edge_fea = self.self_tp(edge_fea)

        return edge_fea


class Net(nn.Module):
    """DeepH-E3's real E(3)-equivariant Hamiltonian-prediction graph network."""

    def __init__(
        self,
        num_species,
        irreps_embed_node,
        irreps_edge_init,
        irreps_sh,
        irreps_mid_node,
        irreps_post_node,
        irreps_out_node,
        irreps_mid_edge,
        irreps_post_edge,
        irreps_out_edge,
        num_block,
        r_max,
        use_sc=True,
        no_parity=False,
        use_sbf=True,
        selftp=False,
        edge_upd=True,
        only_ij=False,
        num_basis=128,
        act={1: torch.nn.functional.silu, -1: torch.tanh},
        act_gates={1: torch.sigmoid, -1: torch.tanh},
        if_sort_irreps=False,
    ):
        if no_parity:
            for irreps in (
                irreps_embed_node,
                irreps_edge_init,
                irreps_sh,
                irreps_mid_node,
                irreps_post_node,
                irreps_out_node,
                irreps_mid_edge,
                irreps_post_edge,
                irreps_out_edge,
            ):
                for _, ir in Irreps(irreps):
                    assert ir.p == 1, (
                        "Ignoring parity but requiring representations with odd parity in net"
                    )

        super(Net, self).__init__()
        self.num_species = num_species
        self.only_ij = only_ij

        irreps_embed_node = Irreps(irreps_embed_node)
        assert irreps_embed_node == Irreps(f"{irreps_embed_node.dim}x0e")
        self.embedding = Linear(irreps_in=f"{num_species}x0e", irreps_out=irreps_embed_node)

        self.basis = GaussianBasis(start=0.0, stop=r_max, n_gaussians=num_basis, trainable=False)

        irreps_edge_init = Irreps(irreps_edge_init)
        assert irreps_edge_init == Irreps(f"{irreps_edge_init.dim}x0e")
        self.distance_expansion = GaussianBasis(
            start=0.0, stop=6.0, n_gaussians=irreps_edge_init.dim, trainable=False
        )

        if use_sbf:
            self.sh = SphericalBasis(irreps_sh, r_max)
        else:
            self.sh = SphericalHarmonics(
                irreps_out=irreps_sh,
                normalize=True,
                normalization="component",
            )
        self.use_sbf = use_sbf
        if no_parity:
            irreps_sh = Irreps([(mul, (ir.l, 1)) for mul, ir in Irreps(irreps_sh)])
        self.irreps_sh = irreps_sh

        irreps_node_prev = self.embedding.irreps_out
        irreps_edge_prev = irreps_edge_init

        self.node_update_blocks = nn.ModuleList([])
        self.edge_update_blocks = nn.ModuleList([])
        for index_block in range(num_block):
            if index_block == num_block - 1:
                node_update_block = NodeUpdateBlock(
                    num_species,
                    num_basis,
                    irreps_sh,
                    irreps_node_prev,
                    irreps_post_node,
                    irreps_edge_prev,
                    act,
                    act_gates,
                    use_selftp=selftp,
                    use_sc=use_sc,
                    only_ij=only_ij,
                    if_sort_irreps=if_sort_irreps,
                )
                edge_update_block = EdgeUpdateBlock(
                    num_species,
                    num_basis,
                    irreps_sh,
                    node_update_block.irreps_out,
                    irreps_edge_prev,
                    irreps_post_edge,
                    act,
                    act_gates,
                    use_selftp=selftp,
                    use_sc=use_sc,
                    if_sort_irreps=if_sort_irreps,
                )
            else:
                node_update_block = NodeUpdateBlock(
                    num_species,
                    num_basis,
                    irreps_sh,
                    irreps_node_prev,
                    irreps_mid_node,
                    irreps_edge_prev,
                    act,
                    act_gates,
                    use_selftp=False,
                    use_sc=use_sc,
                    only_ij=only_ij,
                    if_sort_irreps=if_sort_irreps,
                )
                edge_update_block = None
                if edge_upd:
                    edge_update_block = EdgeUpdateBlock(
                        num_species,
                        num_basis,
                        irreps_sh,
                        node_update_block.irreps_out,
                        irreps_edge_prev,
                        irreps_mid_edge,
                        act,
                        act_gates,
                        use_selftp=False,
                        use_sc=use_sc,
                        if_sort_irreps=if_sort_irreps,
                    )
            irreps_node_prev = node_update_block.irreps_out
            if edge_update_block is not None:
                irreps_edge_prev = edge_update_block.irreps_out
            self.node_update_blocks.append(node_update_block)
            self.edge_update_blocks.append(edge_update_block)

        irreps_out_edge = Irreps(irreps_out_edge)
        for _, ir in irreps_out_edge:
            assert ir in irreps_edge_prev, (
                f"required ir {ir} in irreps_out_edge cannot be produced by convolution in the last edge update block ({edge_update_block.irreps_in_edge} -> {edge_update_block.irreps_out})"
            )

        self.irreps_out_node = irreps_out_node
        self.irreps_out_edge = irreps_out_edge
        self.lin_node = Linear(irreps_in=irreps_node_prev, irreps_out=irreps_out_node, biases=True)
        self.lin_edge = Linear(irreps_in=irreps_edge_prev, irreps_out=irreps_out_edge, biases=True)

    def forward(self, data):
        node_one_hot = F.one_hot(data.x, num_classes=self.num_species).type(
            torch.get_default_dtype()
        )
        edge_one_hot = F.one_hot(
            self.num_species * data.x[data.edge_index[0]] + data.x[data.edge_index[1]],
            num_classes=self.num_species**2,
        ).type(torch.get_default_dtype())

        node_fea = self.embedding(node_one_hot)

        edge_length = data["edge_attr"][:, 0]
        edge_vec = data["edge_attr"][:, [2, 3, 1]]  # (y, z, x) order

        if self.use_sbf:
            edge_sh = self.sh(edge_length, edge_vec)
        else:
            edge_sh = self.sh(edge_vec).type(torch.get_default_dtype())
        edge_length_embedded = self.basis(edge_length)

        selfloop_edge = None
        if self.only_ij:
            selfloop_edge = torch.abs(data["edge_attr"][:, 0]) < 1e-7

        edge_fea = self.distance_expansion(edge_length).type(torch.get_default_dtype())
        for node_update_block, edge_update_block in zip(
            self.node_update_blocks, self.edge_update_blocks
        ):
            node_fea = node_update_block(
                node_fea,
                node_one_hot,
                edge_sh,
                edge_fea,
                edge_length_embedded,
                data["edge_index"],
                data.batch,
                selfloop_edge,
                edge_length,
            )
            if edge_update_block is not None:
                edge_fea = edge_update_block(
                    node_fea,
                    edge_one_hot,
                    edge_sh,
                    edge_fea,
                    edge_length_embedded,
                    data["edge_index"],
                    data.batch,
                )

        node_fea = self.lin_node(node_fea)
        edge_fea = self.lin_edge(edge_fea)
        return node_fea, edge_fea


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


class _NetWrapper(nn.Module):
    """Thin positional-args wrapper: Net.forward takes a single torch_geometric Data object,
    so build_deephe3()/example_input_deephe3() supply it that way directly (a Data object is
    already a single positional argument to torchlens' trace())."""

    def __init__(self, net: Net):
        super().__init__()
        self.net = net

    def forward(self, data):
        return self.net(data)


def build_deephe3():
    """Tiny DeepH-E3 Net (default_configs/train_default.ini-style irreps, scaled down)."""
    net = Net(
        num_species=3,
        irreps_embed_node="8x0e",
        irreps_edge_init="8x0e",
        irreps_sh="1x0e+1x1o",
        irreps_mid_node="4x0e+2x1o",
        irreps_post_node="4x0e+2x1o",
        irreps_out_node="4x0e",
        irreps_mid_edge="4x0e+2x1o",
        irreps_post_edge="4x0e+2x1o",
        irreps_out_edge="1x0e+1x1o",
        num_block=2,
        r_max=6.0,
        use_sc=True,
        no_parity=False,
        use_sbf=False,
        selftp=False,
        edge_upd=True,
        only_ij=False,
        num_basis=8,
    )
    return _NetWrapper(net)


def example_input_deephe3():
    g = torch.Generator().manual_seed(0)
    n_atoms, n_edges, num_species = 6, 14, 3
    x = torch.randint(0, num_species, (n_atoms,), generator=g)
    src = torch.randint(0, n_atoms, (n_edges,), generator=g)
    dst = torch.randint(0, n_atoms, (n_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    pos1 = torch.rand(n_edges, 3, generator=g)
    pos2 = torch.rand(n_edges, 3, generator=g)
    dist = torch.norm(pos1 - pos2, dim=-1, keepdim=True) + 0.5
    edge_attr = torch.cat([dist, pos1, pos2], dim=-1)
    batch = torch.zeros(n_atoms, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = batch
    return (data,)


MENAGERIE_ENTRIES = [
    ("DeepH-E3", build_deephe3, example_input_deephe3, 2023, MENAGERIE_ZOO),
]
