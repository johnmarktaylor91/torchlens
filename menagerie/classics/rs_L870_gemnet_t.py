# ruff: noqa: E741  (verbatim upstream uses single-letter `l`/`m`/`n` for spherical-harmonic
# degree/order and bessel-function order -- standard physics notation, kept as-is)
# SOURCE: vendored from TUM-DAML/gemnet_pytorch @ a0164f74217155232d39c35f0bb2c016bd3f44da (master)
#
# GemNet-T (triplets-only variant of GemNet): a directional message-passing GNN
# interatomic potential (Gasteiger, Becker & Gunnemann 2021, "GemNet: Universal
# Directional Graph Neural Networks for Molecules").
#
# Files combined (each class copied verbatim from the real repo, imports/paths fixed
# minimally so the module is self-contained; @numba.njit dropped from the two index-
# building helpers copied from training/data_container.py -- numba JIT is a pure speed
# optimization there and the plain-Python bodies are unchanged; np.bool -> bool for
# numpy>=1.24 compatibility):
#   - gemnet/model/gemnet.py                      -> GemNet (triplets_only=True path)
#   - gemnet/model/initializers.py                 -> he_orthogonal_init
#   - gemnet/model/utils.py                        -> read_json/write_json/update_json/
#                                                      read_value_json (scale-factor I/O)
#   - gemnet/model/layers/base_layers.py           -> Dense, ScaledSiLU, ResidualLayer
#   - gemnet/model/layers/embedding_block.py       -> AtomEmbedding, EdgeEmbedding
#   - gemnet/model/layers/envelope.py              -> Envelope
#   - gemnet/model/layers/basis_utils.py           -> bessel_basis, real_sph_harm (+ helpers)
#   - gemnet/model/layers/basis_layers.py          -> BesselBasisLayer, SphericalBasisLayer
#   - gemnet/model/layers/efficient.py             -> EfficientInteractionDownProjection,
#                                                      EfficientInteractionBilinear
#   - gemnet/model/layers/atom_update_block.py     -> AtomUpdateBlock, OutputBlock
#   - gemnet/model/layers/scaling.py               -> AutomaticFit, AutoScaleFit, ScalingFactor
#   - gemnet/model/layers/interaction_block.py     -> InteractionBlockTripletsOnly,
#                                                      TripletInteraction
#   - gemnet/training/data_container.py            -> DataContainer._bmat_fast/get_triplets/
#                                                      repeat_blocks/ragged_range (the graph ->
#                                                      triplet-index preprocessing GemNet needs;
#                                                      extracted as plain functions since the
#                                                      real DataContainer class loads its data
#                                                      from an on-disk .npz dataset file)
#
# GemNet-Q/-dQ/-dT quadruplet machinery (InteractionBlock, QuadrupletInteraction,
# TensorBasisLayer, get_quadruplets) is intentionally omitted: this module builds the
# GemNet-T variant (triplets_only=True), matching the queue entry "GemNet-T".
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import scipy.sparse as sp
import sympy as sym
import torch
from scipy import special as scipy_special
from scipy.optimize import brentq
from torch_scatter import scatter

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# gemnet/model/utils.py  (verbatim)
# ------------------------------------------------------------------
def read_json(path):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "r") as f:
        content = json.load(f)
    return content


def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_value_json(path, key):
    """ """
    content = read_json(path)

    if key in content.keys():
        return content[key]
    else:
        return None


# ------------------------------------------------------------------
# gemnet/model/initializers.py  (verbatim)
# ------------------------------------------------------------------
def _standardize(kernel):
    """
    Makes sure that Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


# ------------------------------------------------------------------
# gemnet/model/layers/scaling.py  (verbatim)
# ------------------------------------------------------------------
class AutomaticFit:
    """
    All added variables are processed in the order of creation.
    """

    activeVar = None
    queue = None
    fitting_mode = False

    def __init__(self, variable, scale_file, name):
        self.variable = variable  # variable to find value for
        self.scale_file = scale_file
        self._name = name

        self._fitted = False
        self.load_maybe()

        # first instance created
        if AutomaticFit.fitting_mode and not self._fitted:
            # if first layer set to active
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []  # initialize
            # else add to queue
            else:
                self._add2queue()

    def reset():
        AutomaticFit.activeVar = None
        AutomaticFit.all_processed = False

    def fitting_completed():
        return AutomaticFit.queue is None

    def set2fitmode():
        AutomaticFit.reset()
        AutomaticFit.fitting_mode = True

    def _add2queue(self):
        # check that same variable is not added twice
        for var in AutomaticFit.queue:
            if self._name == var._name:
                raise ValueError(
                    f"Variable with the same name ({self._name}) was already added to queue!"
                )
        AutomaticFit.queue += [self]

    def set_next_active(self):
        """
        Set the next variable in the queue that should be fitted.
        """
        queue = AutomaticFit.queue
        if len(queue) == 0:
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None
            return
        AutomaticFit.activeVar = queue.pop(0)

    def load_maybe(self):
        """
        Load variable from file or set to initial value of the variable.
        """
        value = read_value_json(self.scale_file, self._name)
        if value is None:
            pass
        else:
            self._fitted = True
            with torch.no_grad():
                self.variable.copy_(torch.tensor(value))


class AutoScaleFit(AutomaticFit):
    """
    Class to automatically fit the scaling factors depending on the observed variances.

    Parameters
    ----------
        variable: tf.Variable
            Variable to fit.
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, variable, scale_file, name):
        super().__init__(variable, scale_file, name)

        if not self._fitted:
            self._init_stats()

    def _init_stats(self):
        self.variance_in = 0
        self.variance_out = 0
        self.nSamples = 0

    def observe(self, x, y):
        """
        Observe variances for inut x and output y.
        The scaling factor alpha is calculated s.t. Var(alpha * y) ~ Var(x)
        """
        if self._fitted:
            return

        # only track stats for current variable
        if AutomaticFit.activeVar == self:
            nSamples = y.shape[0]
            self.variance_in += torch.mean(torch.var(x, dim=0)) * nSamples
            self.variance_out += torch.mean(torch.var(y, dim=0)) * nSamples
            self.nSamples += nSamples

    def fit(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if AutomaticFit.activeVar == self:
            if self.variance_in == 0:
                raise ValueError(
                    f"Did not track the variable {self._name}. Add observe calls to track the variance before and after."
                )

            # calculate variance preserving scaling factor
            self.variance_in = self.variance_in / self.nSamples
            self.variance_out = self.variance_out / self.nSamples

            ratio = self.variance_out / self.variance_in
            value = np.sqrt(1 / ratio, dtype="float32")

            # set variable to calculated value
            with torch.no_grad():
                self.variable.copy_(self.variable * value)
            update_json(self.scale_file, {self._name: float(self.variable.numpy())})
            self.set_next_active()  # set next variable in queue to active


class ScalingFactor(torch.nn.Module):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference input x_ref is preserved.

    Parameters
    ----------
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
        name: str
            Name of the scaling factor
    """

    def __init__(self, scale_file, name, device=None):
        super().__init__()

        self.scale_factor = torch.nn.Parameter(
            torch.tensor(1.0, device=device), requires_grad=False
        )
        self.autofit = AutoScaleFit(self.scale_factor, scale_file, name)

    def forward(self, x_ref, y):
        y = y * self.scale_factor
        self.autofit.observe(x_ref, y)

        return y


# ------------------------------------------------------------------
# gemnet/model/layers/base_layers.py  (verbatim)
# ------------------------------------------------------------------
class Dense(torch.nn.Module):
    """
    Combines dense layer and scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(self, in_features, out_features, bias=False, activation=None, name=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError("Activation function not implemented for GemNet (yet).")

    def reset_parameters(self):
        he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        activation: str
            Name of the activation function to use.
    """

    def __init__(self, units: int, nLayers: int = 2, activation=None, name=None):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[Dense(units, units, activation=activation, bias=False) for i in range(nLayers)]
        )
        self.inv_sqrt_2 = 1 / (2.0**0.5)

    def forward(self, inputs):
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x


# ------------------------------------------------------------------
# gemnet/model/layers/embedding_block.py  (verbatim)
# ------------------------------------------------------------------
class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, name=None):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Pu (94). Use 93 dimensions because of 0-based indexing
        self.embeddings = torch.nn.Embedding(93, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        atom_features: int
            Embedding size of the atom embeddings.
        edge_features: int
            Embedding size of the edge embeddings.
        out_features: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(self, atom_features, edge_features, out_features, activation=None, name=None):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(
        self,
        h,
        m_rbf,
        idnb_a,
        idnb_c,
    ):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        # m_rbf: shape (nEdges, nFeatures)
        # in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_ca

        h_a = h[idnb_a]  # shape=(nEdges, emb_size)
        h_c = h[idnb_c]  # shape=(nEdges, emb_size)

        m_ca = torch.cat([h_a, h_c, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_ca = self.dense(m_ca)  # (nEdges, emb_size)
        return m_ca


# ------------------------------------------------------------------
# gemnet/model/layers/envelope.py  (verbatim)
# ------------------------------------------------------------------
class Envelope(torch.nn.Module):
    """
    Envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        p: int
            Exponent of the envelope function.
    """

    def __init__(self, p, name="envelope"):
        super().__init__()
        assert p > 0
        self.p = p
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


# ------------------------------------------------------------------
# gemnet/model/layers/basis_utils.py  (verbatim)
# ------------------------------------------------------------------
def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return scipy_special.spherical_jn(n, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
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
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols("x")
    # j_i = (-x)^i * (1/x * d/dx)^i * sin(x)/x
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).

    Returns:
        bess_basis: list
            Bessel basis formulas taking in a single argument x.
            Has length n where each element has length k. -> In total n*k many.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = (
            1 / np.array(normalizer_tmp) ** 0.5
        )  # sqrt(2/(j_l+1)**2) , sqrt(1/c**3) not taken into account yet
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


def sph_harm_prefactor(l, m):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.

    Parameters
    ----------
        l: int
            Degree of the spherical harmonic. l >= 0
        m: int
            Order of the spherical harmonic. -l <= m <= l

    Returns
    -------
        factor: float

    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return (
        (2 * l + 1) / (4 * np.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m))
    ) ** 0.5


def associated_legendre_polynomials(L, zero_m_only=True, pos_m_only=True):
    """Computes string formulas of the associated legendre polynomials up to degree L (excluded).

    Parameters
    ----------
        L: int
            Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.

    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sym.symbols("z")
    P_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L > 0:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l in range(2, L):
                P_l_m[l][0] = sym.simplify(
                    ((2 * l - 1) * z * P_l_m[l - 1][0] - (l - 1) * P_l_m[l - 2][0]) / l
                )
            return P_l_m
        else:
            # for m >= 0
            for l in range(1, L):
                P_l_m[l][l] = sym.simplify(
                    (1 - 2 * l) * (1 - z**2) ** 0.5 * P_l_m[l - 1][l - 1]
                )  # P_00, P_11, P_22, P_33

            for m in range(0, L - 1):
                P_l_m[m + 1][m] = sym.simplify(
                    (2 * m + 1) * z * P_l_m[m][m]
                )  # P_10, P_21, P_32, P_43

            for l in range(2, L):
                for m in range(l - 1):  # P_20, P_30, P_31
                    P_l_m[l][m] = sym.simplify(
                        ((2 * l - 1) * z * P_l_m[l - 1][m] - (l + m - 1) * P_l_m[l - 2][m])
                        / (l - m)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l in range(1, L):
                    for m in range(1, l + 1):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l][-m] = sym.simplify(
                            (-1) ** m * math.factorial(l - m) / math.factorial(l + m) * P_l_m[l][m]
                        )

            return P_l_m


def real_sph_harm(L, spherical_coordinates, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.

    Parameters
    ----------
        L: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        spherical_coordinates: bool
            - True: Expects the input of the formula strings to be phi and theta.
            - False: Expects the input of the formula strings to be x, y and z.
        zero_m_only: bool
            If True only calculate the harmonics where m=0.

    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l in range(L)]
    else:
        Y_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if spherical_coordinates:
        # replace z by cos(theta)
        theta = sym.symbols("theta")
        for l in range(L):
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sym.cos(theta))

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l in range(L):
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l in range(1, L):
            # m > 0
            for m in range(1, l + 1):
                Y_l_m[l][m] = sym.simplify(
                    2**0.5 * (-1) ** m * sph_harm_prefactor(l, m) * P_l_m[l][m] * sym.cos(m * phi)
                )
            # m < 0
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sym.simplify(
                    2**0.5 * (-1) ** m * sph_harm_prefactor(l, -m) * P_l_m[l][m] * sym.sin(m * phi)
                )

        # convert expressions to cartesian coordinates
        if not spherical_coordinates:
            # replace phi by atan2(y,x)
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l in range(L):
                for m in range(len(Y_l_m[l])):
                    Y_l_m[l][m] = sym.simplify(Y_l_m[l][m].subs(phi, sym.atan2(y, x)))
    return Y_l_m


# ------------------------------------------------------------------
# gemnet/model/layers/basis_layers.py  (BesselBasisLayer, SphericalBasisLayer verbatim;
# TensorBasisLayer omitted -- only used by the quadruplet path)
# ------------------------------------------------------------------
class BesselBasisLayer(torch.nn.Module):
    """
    1D Bessel Basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        name="bessel_basis",
    ):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5

        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.Tensor(np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32)),
            requires_grad=True,
        )

    def forward(self, d):
        d = d[:, None]  # (nEdges,1)
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env * self.norm_const * torch.sin(self.frequencies * d_scaled) / d


class SphericalBasisLayer(torch.nn.Module):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        efficient: bool = False,
        name: str = "spherical_basis",
    ):
        super().__init__()

        assert num_radial <= 64
        self.efficient = efficient
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.envelope = Envelope(envelope_exponent)
        self.inv_cutoff = 1 / cutoff

        # retrieve formulas
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(num_spherical, spherical_coordinates=True, zero_m_only=True)
        self.sph_funcs = []  # (num_spherical,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)
        self.norm_const = self.inv_cutoff**1.5
        self.register_buffer(
            "device_buffer", torch.zeros(0), persistent=False
        )  # dummy buffer to get device of layer

        # convert to torch functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        m = 0  # only single angle
        for l in range(len(Y_lm)):  # num_spherical
            if l == 0:
                # Y_00 is only a constant -> function returns value and not tensor
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(lambda theta: torch.zeros_like(theta) + first_sph(theta))
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], bessel_formulas[l][n], modules))

    def forward(self, D_ca, Angle_cab, id3_reduce_ca, Kidx):
        d_scaled = D_ca * self.inv_cutoff  # (nEdges,)
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # s: 0 0 0 0 1 1 1 1 ...
        # r: 0 1 2 3 0 1 2 3 ...
        rbf = torch.stack(rbf, dim=1)  # (nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf  # (nEdges, num_spherical * num_radial)

        sph = [f(Angle_cab) for f in self.sph_funcs]
        sph = torch.stack(sph, dim=1)  # (nTriplets, num_spherical)

        if not self.efficient:
            rbf_env = rbf_env[id3_reduce_ca]  # (nTriplets, num_spherical * num_radial)
            rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            # e.g. num_spherical = 3, num_radial = 2
            # z_ln: l: 0 0  1 1  2 2
            #       n: 0 1  0 1  0 1
            sph = sph.view(-1, self.num_spherical, 1)  # (nTriplets, num_spherical, 1)
            # e.g. num_spherical = 3, num_radial = 2
            # Y_lm: l: 0 0  1 1  2 2
            #       m: 0 0  0 0  0 0
            out = (rbf_env * sph).view(-1, self.num_spherical * self.num_radial)
            return out  # (nTriplets, num_spherical * num_radial)
        else:
            rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            rbf_env = torch.transpose(rbf_env, 0, 1)  # (num_spherical, nEdges, num_radial)

            # Zero padded dense matrix
            # maximum number of neighbors, catch empty id_reduce_ji with maximum
            Kmax = 0 if sph.shape[0] == 0 else torch.max(torch.max(Kidx + 1), torch.tensor(0))
            nEdges = d_scaled.shape[0]

            sph2 = torch.zeros(
                nEdges, Kmax, self.num_spherical, device=self.device_buffer.device, dtype=sph.dtype
            )
            sph2[id3_reduce_ca, Kidx] = sph

            # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical)
            return rbf_env, sph2


# ------------------------------------------------------------------
# gemnet/model/layers/efficient.py  (EfficientInteractionDownProjection,
# EfficientInteractionBilinear verbatim; EfficientInteractionHadamard omitted --
# only used by the quadruplet GemNet-Q interaction block)
# ------------------------------------------------------------------
class EfficientInteractionDownProjection(torch.nn.Module):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        num_spherical: int
            Same as the setting in the basis layers.
        num_radial: int
            Same as the setting in the basis layers.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        emb_size_interm: int,
        name="EfficientDownProj",
    ):
        super().__init__()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(
            torch.empty((self.num_spherical, self.num_radial, self.emb_size_interm)),
            requires_grad=True,
        )
        he_orthogonal_init(self.weight)

    def forward(self, tbf):
        """
        Returns
        -------
            (rbf_W1, sph): tuple
            - rbf_W1: Tensor, shape=(nEdges, emb_size_interm, num_spherical)
            - sph: Tensor, shape=(nEdges, Kmax, num_spherical)
        """
        rbf_env, sph = tbf
        # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical) ;  Kmax = maximum number of neighbors of the edges

        # MatMul: mul + sum over num_radial
        rbf_W1 = torch.matmul(rbf_env, self.weight)  # (num_spherical, nEdges , emb_size_interm)
        rbf_W1 = rbf_W1.permute(1, 2, 0)  # (nEdges, emb_size_interm, num_spherical)

        sph = torch.transpose(sph, 1, 2)  # (nEdges, num_spherical, Kmax)
        return rbf_W1, sph


class EfficientInteractionBilinear(torch.nn.Module):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Parameters
    ----------
        emb_size: int
            Edge embedding size.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        units_out: int
            Embedding output size of the bilinear layer.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        emb_size: int,
        emb_size_interm: int,
        units_out: int,
        name="EfficientBilinear",
    ):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.emb_size, self.emb_size_interm, self.units_out),
                requires_grad=True,
            )
        )
        he_orthogonal_init(self.weight)

    def forward(self, basis, m, id_reduce, Kidx):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, units_out)
                Edge embeddings.
        """
        # quadruplets: m = m_db , triplets: m = m_ba
        # num_spherical is actually num_spherical**2 for quadruplets
        rbf_W1, sph = (
            basis  # (nEdges, emb_size_interm, num_spherical) ,  (nEdges, num_spherical, Kmax)
        )
        nEdges = rbf_W1.shape[0]

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        # maximum number of neighbors, catch empty id_reduce_ji with maximum
        Kmax = 0 if sph.shape[2] == 0 else torch.max(torch.max(Kidx + 1), torch.tensor(0))
        m2 = torch.zeros(nEdges, Kmax, self.emb_size, device=self.weight.device, dtype=m.dtype)
        m2[id_reduce, Kidx] = m  # (nQuadruplets or nTriplets, emb_size) -> (nEdges, Kmax, emb_size)

        sum_k = torch.matmul(sph, m2)  # (nEdges, num_spherical, emb_size)

        # MatMul: mul + sum over num_spherical
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)  # (nEdges, emb_size_interm, emb_size)

        # Bilinear: Sum over emb_size_interm and emb_size
        m_ca = torch.matmul(
            rbf_W1_sum_k.permute(2, 0, 1), self.weight
        )  # (emb_size, nEdges, units_out)
        m_ca = torch.sum(m_ca, dim=0)  # (nEdges, units_out)
        return m_ca


# ------------------------------------------------------------------
# gemnet/model/layers/atom_update_block.py  (verbatim)
# ------------------------------------------------------------------
class AtomUpdateBlock(torch.nn.Module):
    """
    Aggregate the message embeddings of the atoms

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edge embeddings.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "atom_update",
    ):
        super().__init__()
        self.name = name
        self.emb_size_edge = emb_size_edge

        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_sum = ScalingFactor(scale_file=scale_file, name=name + "_sum")

        self.layers = self.get_mlp(emb_size_atom, nHidden, activation)

    def get_mlp(self, units, nHidden, activation):
        dense1 = Dense(self.emb_size_edge, units, activation=activation, bias=False)
        res = [ResidualLayer(units, nLayers=2, activation=activation) for i in range(nHidden)]
        mlp = [dense1] + res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]

        mlp_rbf = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * mlp_rbf

        x2 = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="add")
        x = self.scale_sum(m, x2)  # (nAtoms, emb_size_edge)

        for i, layer in enumerate(self.layers):
            x = layer(x)  # (nAtoms, emb_size_atom)
        return x


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edge embeddings.
        nHidden: int
            Number of residual blocks.
        num_targets: int
            Number of targets.
        activation: str
            Activation function to use in the dense layers (except for the final dense layer).
        direct_forces: bool
            If true directly predict forces without taking the gradient of the energy potential.
        output_init: str
            Kernel initializer of the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        num_targets: int,
        activation=None,
        direct_forces=True,
        output_init="HeOrthogonal",
        scale_file=None,
        name: str = "output",
        **kwargs,
    ):
        super().__init__(
            name=name,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
            scale_file=scale_file,
            **kwargs,
        )

        assert isinstance(output_init, str)
        self.output_init = output_init
        self.direct_forces = direct_forces
        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)

        self.seq_energy = self.layers  # inherited from parent class
        # do not add bias to final layer to enforce that prediction for an atom
        # without any edge embeddings is zero
        self.out_energy = Dense(emb_size_atom, num_targets, bias=False, activation=None)

        if self.direct_forces:
            self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had")
            self.seq_forces = self.get_mlp(emb_size_edge, nHidden, activation)
            # no bias in final layer to ensure continuity
            self.out_forces = Dense(emb_size_edge, num_targets, bias=False, activation=None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.output_init.lower() == "heorthogonal":
            he_orthogonal_init(self.out_energy.weight)
            if self.direct_forces:
                he_orthogonal_init(self.out_forces.weight)
        elif self.output_init.lower() == "zeros":
            torch.nn.init.zeros_(self.out_energy.weight)
            if self.direct_forces:
                torch.nn.init.zeros_(self.out_forces.weight)
        else:
            raise UserWarning(f"Unknown output_init: {self.output_init}")

    def forward(self, h, m, rbf, id_j):
        """
        Returns
        -------
            (E, F): tuple
            - E: Tensor, shape=(nAtoms, num_targets)
            - F: Tensor, shape=(nEdges, num_targets)
            Energy and force prediction
        """
        nAtoms = h.shape[0]

        rbf_mlp = self.dense_rbf(rbf)  # (nEdges, emb_size_edge)
        x = m * rbf_mlp

        # -------------------------------------- Energy Prediction -------------------------------------- #
        x_E = scatter(x, id_j, dim=0, dim_size=nAtoms, reduce="add")  # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(m, x_E)

        for i, layer in enumerate(self.seq_energy):
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        x_E = self.out_energy(x_E)  # (nAtoms, num_targets)

        # --------------------------------------- Force Prediction -------------------------------------- #
        if self.direct_forces:
            x_F = self.scale_rbf(m, x)

            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            x_F = self.out_forces(x_F)  # (nEdges, num_targets)
        else:
            x_F = 0
        # ----------------------------------------------------------------------------------------------- #

        return x_E, x_F


# ------------------------------------------------------------------
# gemnet/model/layers/interaction_block.py  (InteractionBlockTripletsOnly,
# TripletInteraction verbatim; InteractionBlock/QuadrupletInteraction omitted --
# GemNet-Q/-dQ-only)
# ------------------------------------------------------------------
class InteractionBlockTripletsOnly(torch.nn.Module):
    """
    Interaction block for GemNet-T/dT.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom,
        emb_size_edge,
        emb_size_trip,
        emb_size_quad,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_bil_trip,
        num_before_skip,
        num_after_skip,
        num_concat,
        num_atom,
        activation=None,
        scale_file=None,
        name="Interaction",
        **kwargs,
    ):
        super().__init__()
        self.name = name

        block_nr = name.split("_")[-1]

        ## -------------------------------------------- Message Passing ------------------------------------------- ##
        # Dense transformation of skip connection
        self.dense_ca = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_ca",
        )

        # Triplet Interaction
        self.trip_interaction = TripletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_bilinear=emb_size_bil_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            activation=activation,
            scale_file=scale_file,
            name=f"TripInteraction_{block_nr}",
        )

        ## ---------------------------------------- Update Edge Embeddings ---------------------------------------- ##
        # Residual layers before skip connection
        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(emb_size_edge, activation=activation, name=f"res_bef_skip_{i}")
                for i in range(num_before_skip)
            ]
        )

        # Residual layers after skip connection
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(emb_size_edge, activation=activation, name=f"res_aft_skip_{i}")
                for i in range(num_after_skip)
            ]
        )

        ## ---------------------------------------- Update Atom Embeddings ---------------------------------------- ##
        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
            scale_file=scale_file,
            name=f"AtomUpdate_{block_nr}",
        )

        ## ------------------------------ Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        self.concat_layer = EdgeEmbedding(
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            name="concat",
        )
        self.residual_m = torch.nn.ModuleList(
            [
                ResidualLayer(emb_size_edge, activation=activation, name=f"res_m_{i}")
                for i in range(num_concat)
            ]
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)

    def forward(
        self,
        h,
        m,
        rbf3,
        cbf3,
        Kidx3,
        id_swap,
        id3_expand_ba,
        id3_reduce_ca,
        rbf_h,
        id_c,
        id_a,
        **kwargs,
    ):
        """
        Returns
        -------
            h: Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        # Initial transformation
        x_ca_skip = self.dense_ca(m)  # (nEdges, emb_size_edge)

        x3 = self.trip_interaction(m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca)

        ## ----------------------------- Merge Embeddings after Triplet Interaction ------------------------------ ##
        x = x_ca_skip + x3  # (nEdges, emb_size_edge)
        x = x * self.inv_sqrt_2

        ## ---------------------------------------- Update Edge Embeddings --------------------------------------- ##
        # Transformations before skip connection
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + x  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2

        # Transformations after skip connection
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)  # (nEdges, emb_size_edge)

        ## ---------------------------------------- Update Atom Embeddings --------------------------------------- ##
        h2 = self.atom_update(h, m, rbf_h, id_a)  # (nAtoms, emb_size_atom)

        # Skip connection
        h = h + h2  # (nAtoms, emb_size_atom)
        h = h * self.inv_sqrt_2

        ## ----------------------------- Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        m2 = self.concat_layer(h, m, id_c, id_a)  # (nEdges, emb_size_edge)

        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + m2  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2
        return h, m


class TripletInteraction(torch.nn.Module):
    """
    Triplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_edge,
        emb_size_trip,
        emb_size_bilinear,
        emb_size_rbf,
        emb_size_cbf,
        activation=None,
        scale_file=None,
        name="TripletInteraction",
        **kwargs,
    ):
        super().__init__()
        self.name = name

        # Dense transformation
        self.dense_ba = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_ba",
        )

        # Down projections of basis representations, bilinear layer and scaling factors
        self.mlp_rbf = Dense(
            emb_size_rbf, emb_size_edge, activation=None, name="MLP_rbf3_2", bias=False
        )
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had_rbf")

        self.mlp_cbf = EfficientInteractionBilinear(
            emb_size_trip, emb_size_cbf, emb_size_bilinear, name="MLP_cbf3_2"
        )
        self.scale_cbf_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum_cbf"
        )  # combines scaling for bilinear layer and summation

        # Down and up projections
        self.down_projection = Dense(
            emb_size_edge,
            emb_size_trip,
            activation=activation,
            bias=False,
            name="dense_down",
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_up_ca",
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_up_ac",
        )

        self.inv_sqrt_2 = 1 / (2.0) ** 0.5

    def forward(self, m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca):
        """
        Returns
        -------
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        # Dense transformation
        x_ba = self.dense_ba(m)  # (nEdges, emb_size_edge)

        # Transform via radial bessel basis
        mlp_rbf = self.mlp_rbf(rbf3)  # (nEdges, emb_size_edge)
        x_ba2 = x_ba * mlp_rbf
        x_ba = self.scale_rbf(x_ba, x_ba2)

        x_ba = self.down_projection(x_ba)  # (nEdges, emb_size_trip)

        # Transform via circular spherical basis
        x_ba = x_ba[id3_expand_ba]  # (nTriplets, emb_size_trip)

        # Efficient bilinear layer
        x = self.mlp_cbf(cbf3, x_ba, id3_reduce_ca, Kidx3)  # (nEdges, emb_size_bilinear)
        x = self.scale_cbf_sum(x_ba, x)

        # Basis representation:
        # rbf(d_ba)
        # cbf(d_ca, angle_cab)

        # Up project embeddings
        x_ca = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
        x_ac = self.up_projection_ac(x)  # (nEdges, emb_size_edge)

        # Merge interaction of c->a and a->c
        x_ac = x_ac[id_swap]  # swap to add to edge a->c and not c->a
        x3 = x_ca + x_ac
        x3 = x3 * self.inv_sqrt_2
        return x3


# ------------------------------------------------------------------
# gemnet/model/gemnet.py  (verbatim, GemNet class; forward() triplets_only=True
# path; TF-checkpoint loader load_tfmodel() dropped as irrelevant plumbing)
# ------------------------------------------------------------------
class GemNet(torch.nn.Module):
    """
    Parameters
    ----------
        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_quad: int
            (Down-projected) Embedding size in the quadruplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        emb_size_bil_quad: int
            Embedding size of the edge embeddings in the quadruplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.
        triplets_only: bool
            If True use GemNet-T or GemNet-dT.No quadruplet based message passing.
        num_targets: int
            Number of prediction targets.
        cutoff: float
            Embedding cutoff for interatomic directions in Angstrom.
        int_cutoff: float
            Interaction cutoff for interatomic directions in Angstrom. No effect for GemNet-(d)T
        envelope_exponent: int
            Exponent of the envelope function. Determines the shape of the smooth cutoff.
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
        forces_coupled: bool
            No effect if direct_forces is False. If True enforce that |F_ac| = |F_ca|
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_quad: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        emb_size_bil_quad: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        triplets_only: bool,
        num_targets: int = 1,
        direct_forces: bool = False,
        cutoff: float = 5.0,
        int_cutoff: float = 10.0,  # no effect for GemNet-(d)T
        envelope_exponent: int = 5,
        extensive=True,
        forces_coupled: bool = False,
        output_init="HeOrthogonal",
        activation: str = "swish",
        scale_file=None,
        name="gemnet",
        **kwargs,
    ):
        super().__init__()
        assert num_blocks > 0
        self.num_targets = num_targets
        self.num_blocks = num_blocks
        self.extensive = extensive

        self.forces_coupled = forces_coupled

        AutomaticFit.reset()  # make sure that queue is empty (avoid potential error)

        # GemNet variants
        self.direct_forces = direct_forces
        self.triplets_only = triplets_only
        assert self.triplets_only, (
            "This staging module only builds the GemNet-T (triplets_only=True) variant."
        )

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.rbf_basis = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent
        )

        self.cbf_basis3 = SphericalBasisLayer(
            num_spherical,
            num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            efficient=True,
        )
        ### ------------------------------------------------------------------------------------- ###

        ### ------------------------------- Share Down Projections ------------------------------ ###
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            name="MLP_rbf3_shared",
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf, name="MLP_cbf3_shared"
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            name="MLP_rbfh_shared",
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            name="MLP_rbfout_shared",
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        out_blocks = []
        int_blocks = []

        # Interaction Blocks (GemNet-T/dT only)
        interaction_block = InteractionBlockTripletsOnly
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_quad=emb_size_quad,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    emb_size_bil_quad=emb_size_bil_quad,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i + 1}",
                )
            )

        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=direct_forces,
                    scale_file=scale_file,
                    name=f"OutBlock_{i}",
                )
            )

        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)

    @staticmethod
    def calculate_interatomic_vectors(R, id_s, id_t):
        """
        Parameters
        ----------
            R: Tensor, shape = (nAtoms,3)
                Atom positions.
            id_s: Tensor, shape = (nEdges,)
                Indices of the source atom of the edges.
            id_t: Tensor, shape = (nEdges,)
                Indices of the target atom of the edges.

        Returns
        -------
            (D_st, V_st): tuple
                D_st: Tensor, shape = (nEdges,)
                    Distance from atom t to s.
                V_st: Tensor, shape = (nEdges,)
                    Unit direction from atom t to s.
        """
        Rt = R[id_t]
        Rs = R[id_s]
        V_st = Rt - Rs  # s -> t
        D_st = torch.sqrt(torch.sum(V_st**2, dim=1))
        V_st = V_st / D_st[..., None]
        return D_st, V_st

    @staticmethod
    def calculate_neighbor_angles(R_ac, R_ab):
        """Calculate angles between atoms c <- a -> b.

        Parameters
        ----------
            R_ac: Tensor, shape = (N,3)
                Vector from atom a to c.
            R_ab: Tensor, shape = (N,3)
                Vector from atom a to b.

        Returns
        -------
            angle_cab: Tensor, shape = (N,)
                Angle between atoms c <- a -> b.
        """
        # cos(alpha) = (u * v) / (|u|*|v|)
        x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
        # sin(alpha) = |u x v| / (|u|*|v|)
        y = torch.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
        # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
        y = torch.max(y, torch.tensor(1e-9))
        angle = torch.atan2(y, x)
        return angle

    @staticmethod
    def calculate_angles3(R, id_c, id_a, id3_reduce_ca, id3_expand_ba):
        """Calculate angles for triplet-based message passing.

        Parameters
        ----------
            R: Tensor, shape = (nAtoms,3)
                Atom positions.
            id_c: Tensor, shape = (nEdges,)
                Indices of atom c (source atom of edge).
            id_a: Tensor, shape = (nEdges,)
                Indices of atom a (target atom of edge).
            id3_reduce_ca: Tensor, shape = (nTriplets,)
                Edge indices of edge c -> a of the triplets.
            id3_expand_ba: Tensor, shape = (nTriplets,)
                Edge indices of edge b -> a of the triplets.

        Returns
        -------
            angle_cab: Tensor, shape = (nTriplets,)
                Angle between atoms c <- a -> b.
        """
        Rc = R[id_c[id3_reduce_ca]]
        Ra = R[id_a[id3_reduce_ca]]
        Rb = R[id_c[id3_expand_ba]]

        # difference vectors
        R_ac = Rc - Ra  # shape = (nTriplets,3)
        R_ab = Rb - Ra  # shape = (nTriplets,3)

        # angle in triplets
        return GemNet.calculate_neighbor_angles(R_ac, R_ab)  # (nTriplets,)

    def forward(self, inputs):
        Z, R = inputs["Z"], inputs["R"]
        id_a, id_c, id_undir, id_swap = (
            inputs["id_a"],
            inputs["id_c"],
            inputs["id_undir"],
            inputs["id_swap"],
        )
        id3_expand_ba, id3_reduce_ca = inputs["id3_expand_ba"], inputs["id3_reduce_ca"]
        batch_seg, Kidx4, Kidx3 = inputs["batch_seg"], None, inputs["Kidx3"]

        if not self.direct_forces:
            inputs["R"].requires_grad = True

        # Calculate distances
        D_ca, V_ca = self.calculate_interatomic_vectors(R, id_c, id_a)

        rbf = self.rbf_basis(D_ca)
        # Triplet Interaction
        Angles3_cab = self.calculate_angles3(R, id_c, id_a, id3_reduce_ca, id3_expand_ba)
        cbf3 = self.cbf_basis3(D_ca, Angles3_cab, id3_reduce_ca, Kidx3)

        # Embedding block
        h = self.atom_emb(Z)  # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, id_c, id_a)  # (nEdges, emb_size_edge)

        # Shared Down Projections
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(cbf3)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_a, F_ca = self.out_blocks[0](h, m, rbf_out, id_a)
        # (nAtoms, num_targets), (nEdges, num_targets)

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf4=None,
                cbf4=None,
                sbf4=None,
                Kidx4=Kidx4,
                rbf3=rbf3,
                cbf3=cbf3,
                Kidx3=Kidx3,
                id_swap=id_swap,
                id3_expand_ba=id3_expand_ba,
                id3_reduce_ca=id3_reduce_ca,
                id4_reduce_ca=None,
                id4_expand_intm_db=None,
                id4_expand_abd=None,
                rbf_h=rbf_h,
                id_c=id_c,
                id_a=id_a,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F = self.out_blocks[i + 1](h, m, rbf_out, id_a)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_ca += F
            E_a += E

        nMolecules = torch.max(batch_seg) + 1
        if self.extensive:
            E_a = scatter(E_a, batch_seg, dim=0, dim_size=nMolecules, reduce="add")
            # (nMolecules, num_targets)
        else:
            E_a = scatter(E_a, batch_seg, dim=0, dim_size=nMolecules, reduce="mean")
            # (nMolecules, num_targets)

        if self.direct_forces:
            nAtoms = Z.shape[0]
            if self.forces_coupled:  # enforce F_abs_ji = F_ca
                nEdges = id_c.shape[0]
                F_ca = scatter(F_ca, id_undir, dim=0, dim_size=int(nEdges / 2), reduce="mean")
                # (nEdges/2, num_targets)
                F_ca = F_ca[id_undir]  # (nEdges, num_targets)

            # map forces in edge directions
            F_ji = F_ca[:, :, None] * V_ca[:, None, :]  # (nEdges, num_targets, 3)
            F_j = scatter(F_ji, id_a, dim=0, dim_size=nAtoms, reduce="add")
            # (nAtoms, num_targets, 3)
        else:
            if self.num_targets > 1:
                forces = []
                for i in range(self.num_targets):
                    # maybe this can be solved differently
                    forces += [
                        -torch.autograd.grad(E_a[:, i].sum(), inputs["R"], create_graph=True)[0]
                    ]
                F_j = torch.stack(forces, dim=1)
            else:
                F_j = -torch.autograd.grad(E_a.sum(), inputs["R"], create_graph=True)[0]

            inputs["R"].requires_grad = False

        return E_a, F_j  # (nMolecules, num_targets),  (nEdges, num_targets)

    def predict(self, inputs):
        E, F = self(inputs)
        E = E.detach().cpu()
        F = F.detach().cpu()
        return E, F

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------
# gemnet/training/data_container.py  (graph -> triplet-index preprocessing,
# adapted from DataContainer._bmat_fast / get_triplets / repeat_blocks /
# ragged_range as plain functions operating directly on a single molecule's
# adjacency, since the real DataContainer class loads its data from an on-disk
# .npz dataset file. @numba.njit dropped -- pure speed optimization, logic
# unchanged. np.bool -> bool for numpy>=1.24.)
# ------------------------------------------------------------------
def _bmat_fast(mats):
    """Combines multiple adjacency matrices into single sparse block matrix."""
    assert len(mats) > 0
    new_data = np.concatenate([mat.data for mat in mats])

    ind_offset = np.zeros(1 + len(mats), dtype="int32")
    ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
    new_indices = np.concatenate([mats[i].indices + ind_offset[i] for i in range(len(mats))])

    indptr_offset = np.zeros(1 + len(mats))
    indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
    new_indptr = np.concatenate(
        [mats[i].indptr[i >= 1 :] + indptr_offset[i] for i in range(len(mats))]
    )

    shape = (ind_offset[-1], ind_offset[-1])

    if len(new_data) == 0:
        return sp.csr_matrix(shape)

    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=shape)


def _get_triplets(idx_s, idx_t, edge_ids):
    """
    Get triplets c -> a <- b
    """
    id3_expand_ba = edge_ids[idx_s].data.astype("int32").flatten()
    id3_reduce_ca = edge_ids[idx_s].tocoo().row.astype("int32").flatten()

    id3_i = idx_t[id3_reduce_ca]
    id3_k = idx_s[id3_expand_ba]
    mask = id3_i != id3_k
    id3_expand_ba = id3_expand_ba[mask]
    id3_reduce_ca = id3_reduce_ca[mask]

    return id3_expand_ba, id3_reduce_ca


def _ragged_range(sizes):
    """
    Example
    -------
        sizes = [1,3,2] ;
        Return: [0  0 1 2  0 1]
    """
    a = np.arange(sizes.max())
    indices = np.empty(sizes.sum(), dtype=np.int32)
    start = 0
    for size in sizes:
        end = start + size
        indices[start:end] = a[:size]
        start = end
    return indices


def build_gemnet_t_inputs(Z: np.ndarray, R: np.ndarray, cutoff: float):
    """Build the GemNet-T `inputs` dict (id_a/id_c/id_undir/id_swap/id3_expand_ba/
    id3_reduce_ca/Kidx3/batch_seg/Z/R) for a single molecule, following
    DataContainer.__getitem__'s triplets_only=True path.
    """
    n = Z.shape[0]
    D_ij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    adj_mat = sp.csr_matrix(D_ij <= cutoff)
    adj_mat -= sp.eye(n, dtype=bool)

    adj_matrix = _bmat_fast([adj_mat])
    idx_t, idx_s = adj_matrix.nonzero()  # target and source nodes

    idx_data = {}
    if len(idx_t) == 0:
        raise ValueError("Cutoff too small: synthetic molecule has no edges.")

    edges = np.stack([idx_t, idx_s], axis=0)
    mask = edges[0] < edges[1]
    edges = edges[:, mask]
    edges = np.concatenate([edges, edges[::-1]], axis=-1).astype("int32")
    idx_t, idx_s = edges[0], edges[1]
    indices = np.arange(len(mask) / 2, dtype="int32")
    idx_data["id_undir"] = np.concatenate(2 * [indices], axis=-1).astype("int32")

    idx_data["id_c"] = idx_s  # node c is source
    idx_data["id_a"] = idx_t  # node a is target

    N_undir_edges = int(len(idx_s) / 2)
    ind = np.arange(N_undir_edges, dtype="int32")
    id_swap = np.concatenate([ind + N_undir_edges, ind])
    idx_data["id_swap"] = id_swap

    edge_ids = sp.csr_matrix(
        (np.arange(len(idx_s)), (idx_t, idx_s)),
        shape=adj_matrix.shape,
        dtype="int32",
    )

    id3_expand_ba, id3_reduce_ca = _get_triplets(idx_s, idx_t, edge_ids)
    id3_reduce_ca = id_swap[id3_reduce_ca]

    if len(id3_reduce_ca) > 0:
        idx_sorted = np.argsort(id3_reduce_ca)
        id3_reduce_ca = id3_reduce_ca[idx_sorted]
        id3_expand_ba = id3_expand_ba[idx_sorted]
        _, K = np.unique(id3_reduce_ca, return_counts=True)
        idx_data["Kidx3"] = _ragged_range(K)
    else:
        idx_data["Kidx3"] = np.array([], dtype="int32")

    idx_data["id3_expand_ba"] = id3_expand_ba
    idx_data["id3_reduce_ca"] = id3_reduce_ca

    inputs = {
        "Z": torch.tensor(Z, dtype=torch.int64),
        "R": torch.tensor(R, dtype=torch.float32),
        "batch_seg": torch.zeros(n, dtype=torch.int64),
        "id_a": torch.tensor(idx_data["id_a"], dtype=torch.int64),
        "id_c": torch.tensor(idx_data["id_c"], dtype=torch.int64),
        "id_undir": torch.tensor(idx_data["id_undir"], dtype=torch.int64),
        "id_swap": torch.tensor(idx_data["id_swap"], dtype=torch.int64),
        "id3_expand_ba": torch.tensor(idx_data["id3_expand_ba"], dtype=torch.int64),
        "id3_reduce_ca": torch.tensor(idx_data["id3_reduce_ca"], dtype=torch.int64),
        "Kidx3": torch.tensor(idx_data["Kidx3"], dtype=torch.int64),
    }
    return inputs


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
_SCALE_FILE = os.path.join(tempfile.gettempdir(), "menagerie_gemnet_t_scales.json")
if not os.path.exists(_SCALE_FILE):
    with open(_SCALE_FILE, "w") as f:
        json.dump({}, f)


def build_gemnet_t():
    torch.manual_seed(0)
    return GemNet(
        num_spherical=3,
        num_radial=4,
        num_blocks=2,
        emb_size_atom=16,
        emb_size_edge=16,
        emb_size_trip=8,
        emb_size_quad=8,
        emb_size_rbf=8,
        emb_size_cbf=8,
        emb_size_sbf=8,
        emb_size_bil_quad=4,
        emb_size_bil_trip=4,
        num_before_skip=1,
        num_after_skip=1,
        num_concat=1,
        num_atom=1,
        triplets_only=True,
        num_targets=1,
        direct_forces=True,
        cutoff=6.0,
        envelope_exponent=5,
        scale_file=_SCALE_FILE,
    )


def example_input_gemnet_t():
    rng = np.random.default_rng(0)
    # A tiny 6-atom synthetic molecule, positions spread out so a 6.0-Angstrom
    # cutoff produces a fully connected (and thus triplet-rich) local graph.
    Z = np.array([6, 1, 1, 1, 1, 8], dtype=np.int64)
    R = rng.normal(scale=1.5, size=(6, 3)).astype(np.float32)
    inputs = build_gemnet_t_inputs(Z, R, cutoff=6.0)
    return (inputs,)


MENAGERIE_ENTRIES = [
    ("gemnet_t", build_gemnet_t, example_input_gemnet_t, 2021, MENAGERIE_ZOO),
]
