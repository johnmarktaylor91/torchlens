# SOURCE: vendored from https://github.com/lsky96/gcnwmmse @ main
#   Vendored files (real model-definition code only; training/eval CLI
#   (`main.py`), the reference `deepmimo_channel_generation.py` dataset dumper,
#   `datawriter.py`, and the `reference_wmmse_unrolls`/`architectures`/`network`
#   modules -- a factory + baseline UWMMSE/IAIDNN/PGD unrolls + a separate plain-GCN
#   architecture that `gcnwmmse.py` imports but never actually calls -- are dropped):
#     - comm/mathutil.py -> `randcn`, `randc`, `logdet`, `cmat_square`,
#       `clean_hermitian`, `mmchain`, `btrace`, `bf_mat_pow`, `complex_relu`,
#       `complex_mod_relu`, `rationalfct_solve_0d2_cls`/`rationalfct_solve_0d2`/
#       `rationalfct_solve_0d2_iteration`, `matpoly_simple_norm` (verbatim; shared
#       complex-linear-algebra helpers used throughout the unrolled WMMSE layers)
#     - comm/algorithm.py -> only `downlink_mrc` (verbatim; the MRC beamformer
#       initializer `GCNWMMSE.forward` calls to seed its first unrolled iteration --
#       the file's other ~30 WMMSE/ZF baseline solvers are training/eval-only and
#       not on the traced forward path)
#     - comm/channel.py -> only `mimoifc_randcn` (verbatim; a synthetic MIMO
#       interfering-broadcast-channel scenario generator -- a data-generation
#       helper, not architecture -- used here only to build a tiny example
#       `scenario` dict for tracing; the file's other scenario generators, `.mat`/
#       pickle dataset loaders, and rate-computation utilities are dropped)
#     - comm/gcnwmmse.py -> `GCNWMMSE`, `GCNWMMSELayer`, `GCNWMMSE_SISOAdhoc`,
#       `GCNWMMSELayer_SISOAdhoc` (verbatim; `GCNWMMSE_SISOAdhoc` and its layer are
#       kept as real file content even though only `GCNWMMSE` is traced below)
#
#   Import fixes only (no architectural changes): cross-file imports
#   (`import comm.mathutil as util`, `import comm.algorithm as algo`,
#   `import comm.network as network`, `import comm.architectures as models`) are
#   flattened into this single file; `network`/`models` were unused dead imports in
#   the original `gcnwmmse.py` (grep confirms no `network.`/`models.` call sites) and
#   are dropped outright. Thin `types.SimpleNamespace` shims named `util` and `algo`
#   are added so `gcnwmmse.py`'s many `util.randcn(...)`, `util.cmat_square(...)`,
#   `algo.downlink_mrc(...)` attribute-style call sites keep working unmodified.
#
# GCN-WMMSE (lsky96/gcnwmmse; unrolled-optimization deep learning for multicell
# MU-MIMO downlink precoding) turns the classical iterative WMMSE (Weighted Minimum
# Mean-Square-Error) sum-rate-maximization algorithm into a deep network by
# "unrolling" its U/W/V alternating-update steps into stacked learnable layers: each
# `GCNWMMSELayer` computes the closed-form MMSE receive-filter (U) and weight (W)
# updates exactly as classical WMMSE, then replaces the V (transmit beamformer)
# update with a learned complex-linear combination of autoregressive (AR), moving-
# average (MA), diagonal-loading, and bypass terms through a complex nonlinearity,
# followed by power-constraint renormalization per base station. `GCNWMMSE` stacks
# `num_layers` of these (optionally weight-shared) unrolled iterations, seeded from
# a classical MRC (maximum-ratio combining) initial beamformer. Architecture is
# reproduced verbatim.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import types
from typing import Optional

MENAGERIE_ZOO = "vendored-pytorch"

# ---- comm/mathutil.py (verbatim, unused matplotlib import dropped) ----


def randcn(*args, **kwargs):
    """
    Use similarly to torch.randn, but returns complex normal and normalized by 1/sqrt(2).
    """
    re = torch.randn(*args, **kwargs) / torch.tensor(2).sqrt()
    im = torch.randn(*args, **kwargs) / torch.tensor(2).sqrt()
    return torch.complex(re, im)


def randc(*args, **kwargs):
    """
    Use similarly to torch.rand, but returns complex numbers from within [-1,1]x[-1j,1j].
    """
    re = 2 * torch.rand(*args, **kwargs) - 1
    im = 2 * torch.rand(*args, **kwargs) - 1
    return torch.complex(re, im)


def logdet(input):
    """
    Returns the real (due to numeric problems) part of the logdet of the input.
    :param input: invertible matrix
    :return: log-determinant with autograd
    """
    _, logdetval = torch.linalg.slogdet(input)
    # return LogDet.apply(input).real
    return logdetval


def cmat_square(input):
    """
    Calculates input * input^H and cleans up diagonal
    :param input: complex matrix
    :return: matrix square
    """
    square = torch.matmul(input, input.conj().transpose(-2, -1))
    return clean_hermitian(square)


def clean_hermitian(input):
    """
    Cleans up the diagonal of a hermitian matrix from numeric errors.
    :param input: matrix or stack of matrices
    :return: cleaned up matrices
    """
    """
    output = input.clone()
    output.diagonal(dim1=-2, dim2=-1).imag = torch.zeros(1)
    """
    tempcopy = input.clone()
    output = (tempcopy + tempcopy.conj().transpose(-2, -1)) / 2
    output.diagonal(dim1=-2, dim2=-1).imag = torch.zeros(1)
    return output


def mmchain(*args, **kwargs):
    """
    Calculates the chain of matrix products of two or more matrix batches
    :param args: two or more matrix batches
    :return: matrix product
    """

    if len(args) >= 2:
        return torch.matmul(args[0], mmchain(*args[1:]), **kwargs)
    else:
        return args[0]


def btrace(input):
    """
    Calculates the trace of a matrix batch
    :param input: batch of quadratic matrices
    :return: batch of the traces of matrices
    """
    return input.diagonal(dim1=-2, dim2=-1).sum(-1)


def bf_mat_pow(bf_mat):
    """
    Calulates the power of a beamformer matrices.
    :param bf_mat: beamformer matrix or batch of beamformer matrices
    :return: power as (*batch_size)
    """
    power = torch.square(bf_mat.real) + torch.square(bf_mat.imag)
    return power.sum(dim=-2).sum(dim=-1)


def complex_relu(input):
    """
    Performs ReLU operation on complex tensor by rectifying real and imag part individually.
    :param input: tensor
    :return: tensor
    """
    return torch.complex(F.relu(input.real), F.relu(input.imag))


def complex_mod_relu(input, bias):
    """
    Performs the modReLU operation according to Trabelsi et al. 'Deep Complex Networks'.
    :param input: (*batch_size, N) array
    :param bias: (N) array of real numbers
    :return: (*batch_size, N) array
    """
    mag = torch.abs(input) + bias
    zeropos = mag < 0
    output = mag * torch.sgn(input)
    output[zeropos] = 0
    return output


class rationalfct_solve_0d2_cls(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nom, denom, num_iter, x_init):
        batch_size = list(nom.size())[1:]

        x_root = torch.full([1, *batch_size], x_init, dtype=nom.dtype, device=nom.device)
        for iter in range(num_iter):
            x_root = rationalfct_solve_0d2_iteration(x_root, nom, denom)

        ctx.save_for_backward(nom, denom, x_root)

        x_root = x_root.squeeze(0)
        return x_root

    @staticmethod
    def backward(ctx, grad_xout):
        nom, denom, x_root = ctx.saved_tensors
        grad_xout = grad_xout.unsqueeze(0)
        grad_nom = grad_denom = grad_num_iter = grad_x_init = None
        nz_pos = x_root != 0
        denom_sum = torch.clamp(
            denom + x_root, min=1e-32
        )  # safety step to avoid div by 0 at all costs
        ratfun_grad_mu = -2 * torch.nansum(nom / denom_sum**3, dim=0, keepdim=True)

        if ctx.needs_input_grad[0]:
            grad_nom = -1 / ratfun_grad_mu / denom_sum**2
            grad_nom = grad_nom * grad_xout
            grad_nom = grad_nom * nz_pos
        if ctx.needs_input_grad[1]:
            grad_denom = 2 / ratfun_grad_mu * nom / denom_sum**3
            grad_denom = grad_denom * grad_xout
            grad_denom = grad_denom * nz_pos

        return grad_nom, grad_denom, grad_num_iter, grad_x_init


def rationalfct_solve_0d2(nom, denom, num_iter=4, x_init=1e-12):
    return rationalfct_solve_0d2_cls.apply(nom, denom, num_iter, x_init)


def rationalfct_solve_0d2_iteration(xin, nom, denom):
    denom_sum = torch.clamp(denom + xin, min=1e-32)  # safety step to avoid div by 0 at all costs
    fracsum_cube = torch.nansum(nom / denom_sum / denom_sum / denom_sum, dim=0, keepdim=True)
    fracsum_square = torch.nansum(nom / denom_sum / denom_sum, dim=0, keepdim=True)

    big_nom = fracsum_square.pow(1.5) - fracsum_square
    big_frac = big_nom / fracsum_cube

    xtemp = xin + big_frac
    xout = torch.clamp(xtemp, min=0)

    return xout


def matpoly_simple_norm(mat, poly_coeff, norm_above=1):
    """
    Calculates the matrix polynom with optionally multiple features
    :param mat: tensor of quadratic matrices (*batch_size, M, M)
    :param poly_coeff: tensor of coefficients belonging to poly_exp (len(polyexp))
    :return (*batch_size, M, M)
    """
    n = len(poly_coeff)  # degree + 1

    batch_size = list(mat.size())[:-2]
    expand_dim = [1] * len(batch_size)
    mat_size = list(mat.size())[-2:]
    device = mat.device
    dtype = mat.dtype
    norm = btrace(mat) / mat_size[0]

    degrees = []
    eye = torch.eye(mat_size[0], dtype=dtype, device=device).expand(
        *batch_size, *mat_size
    )  # (*bs, M, M)
    if norm_above == -1:
        # degrees.append(eye * norm.unsqueeze(-1).unsqueeze(-1) / 10)  # ADDED DIVIDE BY 10 FOR TESTING REASONS
        degrees.append(eye * norm.unsqueeze(-1).unsqueeze(-1))
        norm_above = 1
    else:
        degrees.append(eye)
    running = eye
    for i in range(1, n):
        if i > norm_above:
            running = torch.matmul(running, mat / norm.unsqueeze(-1).unsqueeze(-1))
        else:
            running = torch.matmul(running, mat)
        degrees.append(running)
    degrees = torch.stack(degrees, dim=0)

    poly_coeff_size = list(poly_coeff.size())
    result = torch.sum(
        poly_coeff.view(*poly_coeff_size, *expand_dim, 1, 1) * degrees, dim=0
    )  # sum polys

    return result


# ---- comm/algorithm.py: only `downlink_mrc` (verbatim; needs `mathutil.bf_mat_pow`,
#      already flattened above) ----


def downlink_mrc(scenario):
    """
    Calculates the equally weighted MRC beamformers for the scenario which fulfill the power constraint
    :param scenario: mimo_8Tx ifc scenario
    :return: iterable of K beamformers for each user k
    """
    num_users = scenario["num_users"]
    users_assign = scenario["users_assign"]
    channels = scenario["channels"]
    bss_maxpow = scenario["bss_pow"]

    dl_beamformers = []
    bss_pow = torch.zeros_like(bss_maxpow)  # sums up power of BS for later correction

    for i_user in range(num_users):
        i_bs = users_assign[i_user]
        assigned_beamformer = channels[i_bs][i_user].conj().transpose(-2, -1)
        dl_beamformers.append(assigned_beamformer)
        bss_pow[..., i_bs] += bf_mat_pow(assigned_beamformer)  # needs to be casted

    for i_user in range(num_users):
        i_bs = users_assign[i_user]
        dl_beamformers[i_user] = dl_beamformers[i_user] * torch.sqrt(
            bss_maxpow[..., i_bs] / bss_pow[..., i_bs]
        ).unsqueeze(-1).unsqueeze(-1)

    return dl_beamformers


# ---- comm/channel.py: only `mimoifc_randcn` (verbatim; needs `mathutil.randcn`,
#      already flattened above) ----


def mimoifc_randcn(
    bss_dim,
    users_dim,
    assignment,
    batch_size=[],
    snr=0,
    weights=1,
    interference_ratio=1,
    channel_gain=1,
    user_noise_pow=1,
    device=torch.device("cpu"),
    precision="double",
):
    """
    Generates batches of random channel vectors of a MIMO IFC and returns dict per convention.
    :param bss_dim: iterable, contains array sizes of K BSs (equal for all batches)
    :param users_dim: iterable of iterables, which contain array sizes of users (equal for all batches)
    :param assignment: iterable of size len(rx) containing indices assigning each user to a BS (equal for all batches)
    :param snr: iterable or scalar of snr in dB per base station, results in return max powers of 10^(SNR/10), see WMMSE paper, broadcastable to (batch_size, num_bss)
    :param weights: iterable of weights for rates per user or scalar, technically not a channel property, broadcastable to (batch_size, num_users)
    :param interference_ratio: average linear channel gain ratio between direct and interference channels (broadcastable to (batch_size))
    :param device: device to assign tensors to
    :param precision: double or single for float point values

    :return: dictionary, containing
        channels: iterable of K iterables, each containing all channel matrix tensors from the corresponding BS to each user
        bss_assign: iterable of K tensors containing the indices of the assigned users
        users_assign: tensor of size I containing the integers of the assigned BS index for the corresponding user
        tx_pow: size K tensor containing the power budgets for the base stations
        users_noise_pow: tensor of size I containing the noise power of the corresponding user
        rweights: size I tensor containing the rate weights for the individual users
        num_bss: number of basestations
        num_users: number of users
        bss_dim: tensor containing the BS array sizes
        users_dim: tensor containing the user array sizes
        batch_size: list containing batch dimensions
        device: device on which tensors are stored
    """

    if precision == "double":
        dtype = torch.double
    else:
        dtype = torch.single

    num_bss = len(bss_dim)
    num_users = len(users_dim)
    assert len(assignment) == len(users_dim)

    # SNR
    bss_pow = torch.pow(
        torch.tensor(10, device=device, dtype=dtype),
        torch.tensor(snr, device=device, dtype=dtype) / 10,
    )
    bss_pow = bss_pow.expand([*batch_size, num_bss])

    # Rate weights
    rweights = torch.tensor(weights, device=device, dtype=dtype)
    rweights = rweights.expand([*batch_size, num_users])

    # Noise
    users_noise_pow = torch.full(
        [*batch_size, num_users], user_noise_pow, device=device, dtype=dtype
    )

    # Channel ratio
    if_channel_element_ratio = torch.tensor(interference_ratio, device=device, dtype=dtype).sqrt()
    if_channel_element_ratio = (
        if_channel_element_ratio.expand([*batch_size]).unsqueeze(-1).unsqueeze(-1)
    )

    channels = []
    bss_assign = []
    users_assign = torch.tensor(assignment, device=device)
    for i_bs in range(num_bss):
        assigned_users = users_assign == i_bs
        channels.append([])
        for i_user in range(num_users):
            channels[i_bs].append(
                (channel_gain ** (1 / 2))
                * randcn(*batch_size, users_dim[i_user], bss_dim[i_bs], device=device, dtype=dtype)
                * (assigned_users[i_user] + (~assigned_users[i_user]) * if_channel_element_ratio)
            )
        bss_assign.append(
            (assigned_users * torch.arange(num_users, device=device))[assigned_users]
        )  # gives the indices of the users assigned to the BS

    bss_dim = torch.tensor(bss_dim, device=device)
    users_dim = torch.tensor(users_dim, device=device)

    scenario = {
        "channels": channels,
        "bss_assign": bss_assign,
        "users_assign": users_assign,
        "bss_pow": bss_pow,
        "users_noise_pow": users_noise_pow,
        "rate_weights": rweights,
        "num_bss": num_bss,
        "num_users": num_users,
        "bss_dim": bss_dim,
        "users_dim": users_dim,
        "batch_size": batch_size,
        "device": device,
    }

    return scenario


# ---- namespace shims so gcnwmmse.py's `util.X` / `algo.X` attribute-style
#      references keep working after flattening every vendored file into this
#      single module namespace ----
util = types.SimpleNamespace(
    randcn=randcn,
    randc=randc,
    logdet=logdet,
    cmat_square=cmat_square,
    clean_hermitian=clean_hermitian,
    mmchain=mmchain,
    btrace=btrace,
    bf_mat_pow=bf_mat_pow,
    complex_relu=complex_relu,
    complex_mod_relu=complex_mod_relu,
    rationalfct_solve_0d2=rationalfct_solve_0d2,
    matpoly_simple_norm=matpoly_simple_norm,
)
algo = types.SimpleNamespace(downlink_mrc=downlink_mrc)


# ---- comm/gcnwmmse.py (verbatim, cross-file imports removed; unused network/models imports dropped) ----


MIN_LAGRANGIAN = 0
REMOVE_BAD_SAMPLES = True
USE_PSEUDOINV = True
PSEUDOINV_SVAL_THRESH = 1e-12


class GCNWMMSE(nn.Module):
    def __init__(
        self,
        num_layers=5,
        diaglnorm=False,
        biasnorm=False,
        w_poly=None,  # turned off if None
        v_num_channels=4,
        v_active_components=None,
        v_bypass_param=None,  # None or {"position": "before_nonlin"/"after_nonlin"}
        lagrangian_max_iter=8,
        shared_parameters=True,
        device=torch.device("cpu"),
    ):
        super(GCNWMMSE, self).__init__()
        if v_active_components is None:
            v_active_components = {
                "ar": True,
                "ma": True,
                "diagload": True,
                "nonlin": True,
                "bias": False,
            }

        # register parameters
        self.device = device
        self.num_layers = num_layers
        if shared_parameters:
            self.layers = nn.ModuleList(
                [
                    GCNWMMSELayer(
                        diaglnorm=diaglnorm,
                        biasnorm=biasnorm,
                        w_poly=w_poly,
                        v_num_channels=v_num_channels,
                        v_active_components=v_active_components,
                        v_bypass_param=v_bypass_param,
                        lagrangian_max_iter=lagrangian_max_iter,
                        device=device,
                    )
                ]
                * num_layers
            )
        else:
            layers = []
            for i_layer in range(num_layers):
                layers.append(
                    GCNWMMSELayer(
                        diaglnorm=diaglnorm,
                        biasnorm=biasnorm,
                        w_poly=w_poly,
                        v_num_channels=v_num_channels,
                        v_active_components=v_active_components,
                        v_bypass_param=v_bypass_param,
                        lagrangian_max_iter=lagrangian_max_iter,
                        layer_id=i_layer,
                        device=device,
                    )
                )
            self.layers = nn.ModuleList(layers)

    def forward(self, scenario, layers="all", debug=False):
        if layers == "all":
            num_forward_layers = self.num_layers
        elif isinstance(layers, int):
            assert self.num_layers >= layers
            num_forward_layers = layers
        else:
            raise ValueError

        # Modified WMMSE
        v = algo.downlink_mrc(scenario)
        v_layers = [v]
        u_layers = []
        w_layers = []
        v_bypass = []
        error_occured = torch.tensor(False)
        for i_layer in range(num_forward_layers):
            if debug:
                print("Layer", i_layer + 1)
            v, u, w, v_bypass, error_occured_layer = self.layers[i_layer](scenario, v, v_bypass)
            v_layers.append(v)
            u_layers.append(u)
            w_layers.append(w)
            error_occured = torch.logical_or(error_occured_layer, error_occured)

        # Restructure to list of users
        v_out = []
        u_out = []
        w_out = []
        for v_user in list(map(list, zip(*v_layers))):  # transposes layer and user dim
            v_out.append(torch.stack(v_user, dim=0))  # stacks layers
        for u_user in list(map(list, zip(*u_layers))):  # transposes layer and user dim
            u_out.append(torch.stack(u_user, dim=0))
        for w_user in list(map(list, zip(*w_layers))):  # transposes layer and user dim
            w_out.append(torch.stack(w_user, dim=0))

        """ADDED FOR TESTING"""
        if self.training:  # to remove scenarios with numerical problems in training
            no_error = ~error_occured.unsqueeze(-1).unsqueeze(-1)
            for i_user in range(len(v_out)):
                v_out[i_user] = v_out[i_user] * no_error
        """"""

        # power
        num_users = scenario["num_users"]
        assigned_bs = scenario["users_assign"]
        with torch.no_grad():
            bss_pow = torch.zeros(
                num_forward_layers + 1, *list(scenario["bss_pow"].size()), device=self.device
            )
            # print(bss_pow.size())
            for i_user in range(num_users):
                i_bs = assigned_bs[i_user]
                # print(util.bf_mat_pow(v_out[i_user]).size())
                bss_pow[..., i_bs] = bss_pow[..., i_bs] + util.bf_mat_pow(v_out[i_user])

        return v_out, u_out, w_out, bss_pow


class GCNWMMSELayer(nn.Module):
    def __init__(
        self,
        diaglnorm=False,
        biasnorm=False,
        w_poly=None,
        v_num_channels=4,
        v_active_components=None,
        v_bypass_param=None,
        lagrangian_max_iter=8,
        layer_id=None,
        device=torch.device("cpu"),
    ):
        super(GCNWMMSELayer, self).__init__()
        if v_active_components is None:
            v_active_components = {
                "ar": True,
                "ma": True,
                "diagload": True,
                "nonlin": True,
                "bias": False,
            }

        # register parameters
        self.layer_id = layer_id
        self.device = device
        self.diaglnorm = diaglnorm
        self.biasnorm = biasnorm
        self.w_poly = w_poly
        self.lagrangian_max_iter = lagrangian_max_iter
        self.v_num_channels = v_num_channels
        self.v_active_components = v_active_components
        self.v_bypass_param = v_bypass_param

        dtype = torch.float64
        ctype = torch.complex128

        filter_param = {}
        if self.w_poly is not None:
            # dict contains: degree, norm_above
            filter_param["w_param_poly"] = nn.Parameter(
                torch.ones(self.w_poly["degree"] + 1, dtype=dtype, device=device)
                / (self.w_poly["degree"] + 1)
            )  # F2=channels, F1=1

        if v_active_components["ar"]:
            filter_param["v_param_ar"] = nn.Parameter(
                util.randcn(1, v_num_channels, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(v_num_channels + 1, device=device))
            )
        if "ar2" in v_active_components and v_active_components["ar2"]:
            filter_param["v_param_ar2"] = nn.Parameter(
                util.randcn(1, v_num_channels, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(v_num_channels + 1, device=device))
            )
        if v_active_components["ma"]:
            filter_param["v_param_ma"] = nn.Parameter(
                util.randcn(1, v_num_channels, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(v_num_channels + 1, device=device))
            )
        if v_active_components["diagload"]:
            filter_param["v_param_diagload"] = nn.Parameter(
                util.randcn(1, v_num_channels, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(v_num_channels + 1, device=device))
            )

        if v_active_components["bias"]:
            filter_param["v_param_bias"] = nn.Parameter(
                torch.zeros(1, v_num_channels, device=device, dtype=ctype)
            )

        if v_active_components["nonlin"]:
            filter_param["v_param_recomb"] = nn.Parameter(
                util.randcn(v_num_channels, 1, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(v_num_channels + 1, device=device))
            )

        self.v_bypass_mat = nn.Parameter(
            util.randcn(v_num_channels, v_num_channels, device=device, dtype=dtype)
            / torch.sqrt(torch.tensor(2 * v_num_channels, device=device))
        )  # sqrt(#in + #out)

        # buffer for diaglnorm and biasnorm
        if self.biasnorm or self.diaglnorm:
            self.running_mean_alpha = 0.99
            self.running_mean_bias_scale: Optional[Tensor]
            self.running_mean_diagl_scale: Optional[Tensor]
            self.running_mean_num_tracked_batches: Optional[Tensor]
            self.register_buffer(
                "running_mean_bias_scale", torch.tensor(1, dtype=dtype, device=device)
            )
            self.register_buffer(
                "running_mean_diagl_scale", torch.tensor(1, dtype=dtype, device=device)
            )
            self.register_buffer("running_mean_num_tracked_batches", torch.tensor(0, device=device))

        # we do not want a situation a * b where both a and b are parameters without some function inbetween
        self.filter_param = nn.ParameterDict(filter_param)
        # print(self.filter_param["v_param_ar"].device)

    def forward(self, scenario, v_in, v_bypass_in):
        """Modified WMMSE layer"""
        debug = False
        # Internal configs
        # lagrangian_max_iter = self.lagrangian_max_iter
        min_norm = 1e-12  # noqa: F841

        device = scenario["device"]
        num_users = scenario["num_users"]
        num_bss = scenario["num_bss"]
        assigned_bs = scenario["users_assign"]
        rweights = scenario["rate_weights"]

        bss_dim = scenario["bss_dim"]
        users_dim = scenario["users_dim"]
        bss_maxpow = scenario["bss_pow"]

        batch_size = list(v_in[0].size())[0:-2]
        batch_ndim = len(batch_size)  # num dimension before matrix dimensions
        expand_dim = [1] * batch_ndim
        ctype = scenario["channels"][0][0].dtype
        rtype = scenario["bss_pow"].dtype

        num_streams_assigned = torch.zeros(num_bss, dtype=rtype)
        for i_user in range(num_users):  # every user assigned to i_bs
            i_bs = assigned_bs[i_user]
            num_streams_assigned[i_bs] += users_dim[i_user]

        if not self.v_active_components["nonlin"]:
            recomb = torch.ones(self.v_num_channels, 1, device=device, dtype=ctype) / torch.sqrt(
                torch.tensor(self.v_num_channels, device=device)
            )
            recomb = recomb.type(torch.complex128)

        # prepare and expand channels
        channels = []
        for i_bs in range(num_bss):
            channels.append([])
            for i_user in range(num_users):
                channels[i_bs].append(
                    scenario["channels"][i_bs][i_user].expand(
                        [*batch_size, *list(scenario["channels"][i_bs][i_user].size())[-2:]]
                    )
                )  # extends channels into batch dim

        # Output lists
        v_out = []
        u_out = []
        w_out = []

        """ MMSE: U + W for every user"""
        # U * W * U^H is equal for every BS, it will be calculated concurrently as preparation for calculation of V
        uwu = []
        v_tilde = []  # preps for calc of V
        for i_user in range(num_users):
            # SUM(H * V * V^H * H) + sigma * I
            user_dim = scenario["users_dim"][i_user]
            eye = torch.eye(user_dim, device=device).view(*expand_dim, user_dim, user_dim)
            dl_covariance_mat = (
                scenario["users_noise_pow"][..., i_user].unsqueeze(-1).unsqueeze(-1) * eye
            )

            # SUM efficiently calculated
            for i_bs in range(num_bss):
                bs_assigned_users = scenario["bss_assign"][i_bs]
                v_stack = torch.stack([v_in[lj] for lj in bs_assigned_users], dim=0)
                bs_channel2user = channels[i_bs][i_user]
                # bs_channel2user = bs_channel2user.expand(1, *batch_size, *list(bs_channel2user.size())[-2:])  # unsqueezes into same dimensions as v_stack
                partial_covariance_mats = torch.matmul(bs_channel2user, v_stack)
                partial_covariance_mats = util.cmat_square(partial_covariance_mats)
                dl_covariance_mat = dl_covariance_mat + partial_covariance_mats.sum(dim=0)

            # U = INV * H * V
            inv_cov_mat = util.clean_hermitian(
                torch.linalg.inv(util.clean_hermitian(dl_covariance_mat))
            )
            user_channel = channels[assigned_bs[i_user]][i_user]
            u_temp = torch.matmul(inv_cov_mat, user_channel)  # takes channel mat of correct BS
            u_temp = torch.matmul(u_temp, v_in[i_user])
            u_out.append(u_temp)
            # print("u", u_temp[0])
            # W = INV(I - U^H * H * V)
            w_inverse = torch.matmul(u_temp.conj().transpose(-2, -1), user_channel)
            w_inverse = util.clean_hermitian(torch.matmul(w_inverse, v_in[i_user]))
            w_inverse = eye - w_inverse
            w_temp = util.clean_hermitian(torch.linalg.inv(w_inverse))
            # print("wtemp", w_temp[0])
            if self.w_poly is not None:
                w_filtered = util.matpoly_simple_norm(
                    w_temp, torch.abs(self.filter_param["w_param_poly"]), self.w_poly["norm_above"]
                )
            else:
                w_filtered = w_temp

            w_out.append(w_filtered)
            # U * W * U^H prepares for calc of V
            uwu_temp = torch.matmul(u_temp, w_filtered)
            uwu_temp = util.clean_hermitian(
                torch.matmul(uwu_temp, u_temp.conj().transpose(-2, -1))
            )  # matrix must be hermitian
            uwu.append(uwu_temp)

            # Preparation V_tilde
            v_tilde_temp = util.mmchain(user_channel.conj().transpose(-2, -1), u_temp, w_filtered)
            v_tilde_temp = v_tilde_temp * rweights[..., i_user].unsqueeze(-1).unsqueeze(-1)
            v_tilde.append(v_tilde_temp)

        "CALCULATION OF V"
        augmented_ul_cov_mats = []  # size num_bs list of covariance matrices
        inv_augmented_ul_cov_mats = []
        inv2_augmented_ul_cov_mats = []
        inv_augmented_ul_cov_mats = []

        for i_bs in range(num_bss):
            partial_ul_cov_mats = []
            # Calculation of UL Covariance SUM
            for i_user in range(num_users):
                bs_channel2user = channels[i_bs][i_user]
                # alpha * H^H * U * W * U^H * H
                partial_ul_cov_mat_temp = torch.matmul(
                    bs_channel2user.conj().transpose(-2, -1), uwu[i_user]
                )
                partial_ul_cov_mat_temp = torch.matmul(partial_ul_cov_mat_temp, bs_channel2user)

                partial_ul_cov_mat_temp = util.clean_hermitian(partial_ul_cov_mat_temp) * rweights[
                    ..., i_user
                ].unsqueeze(-1).unsqueeze(-1)
                partial_ul_cov_mats.append(partial_ul_cov_mat_temp)

            ul_cov_mat_temp = torch.stack(partial_ul_cov_mats, dim=0).sum(dim=0)

            """Calculation of Lagrangian multiplier"""
            # assigned usersauxiliary matrix SUM(HUWWUH)
            aux_assigned_user_mat = []
            for i_user in scenario["bss_assign"][i_bs]:  # every user assigned to i_bs
                v_tilde_temp = v_tilde[i_user]
                aux_assigned_user_mat_temp = util.cmat_square(v_tilde_temp)
                aux_assigned_user_mat.append(aux_assigned_user_mat_temp)
            aux_assigned_user_mat = torch.stack(aux_assigned_user_mat, dim=0).sum(
                dim=0, keepdim=True
            )  # keep dimension to simplify following operation

            # Decomposition of SUM for Lagrangian
            eigenval_temp, eigenvec_temp = torch.linalg.eigh(ul_cov_mat_temp)
            eigenval_temp = torch.clamp(eigenval_temp, min=0)

            eigenval_temp2 = eigenval_temp
            eigenvec_temp2 = eigenvec_temp

            eigenval_temp = eigenval_temp.movedim(
                -1, 0
            )  # now dim 0 contains M eigenval, so (M, *batch_size)
            eigenvec_temp = eigenvec_temp.unsqueeze(
                0
            ).transpose(
                0, -1
            )  # the matrix dim now only hold single eigenvecs, while dim 0 now iterates over different eigenvecs, so (M, *batchdim, M, 1)

            nominator_coeff = (
                util.mmchain(
                    eigenvec_temp.conj().transpose(-2, -1), aux_assigned_user_mat, eigenvec_temp
                )
                .real.squeeze(-1)
                .squeeze(-1)
            )  # dim 0 of size M holds coefficients, so (M, *batch_size)
            nominator_coeff = torch.clamp(nominator_coeff, min=0)  # for numerical reasons
            nominator_coeff = nominator_coeff / bss_maxpow[..., i_bs].expand(1, *batch_size)

            # Rational function rooting (per basestation due to different Tx dim)
            mu_root = util.rationalfct_solve_0d2(
                nominator_coeff, eigenval_temp, num_iter=self.lagrangian_max_iter
            )

            """ADDED FOR TESTING PURPOSES"""
            mu_root = torch.clamp(mu_root, min=MIN_LAGRANGIAN)
            """"""

            # Compute inverse mat
            eye = torch.eye(bss_dim[i_bs], device=device).view(
                *expand_dim, bss_dim[i_bs], bss_dim[i_bs]
            )
            augmented_ul_cov_mat_temp = ul_cov_mat_temp + eye * mu_root.unsqueeze(-1).unsqueeze(-1)
            if torch.any(torch.isnan(mu_root)) and debug:
                print("mu_root", mu_root)
                exit()

            augmented_ul_cov_mats.append(augmented_ul_cov_mat_temp)
            if USE_PSEUDOINV:
                eigenval_loaded = eigenval_temp2 + mu_root.unsqueeze(-1)
                eigenval_loaded[eigenval_loaded < PSEUDOINV_SVAL_THRESH] = (
                    1e12  # effectively setting the inverse to 0
                )
                eigenval_loaded_inv = 1 / eigenval_loaded
                # print(eigenval_loaded.shape, mu_root.shape, eigenvec_temp.shape)
                inv_temp = util.clean_hermitian(
                    (eigenvec_temp2 * eigenval_loaded_inv.unsqueeze(-2))
                    @ eigenvec_temp2.swapaxes(-2, -1).conj()
                )
                if debug and torch.any(torch.isnan(inv_temp)):
                    print("Rinv is NaN")
                inv_augmented_ul_cov_mats.append(inv_temp)
            else:
                inv_augmented_ul_cov_mats.append(
                    util.clean_hermitian(torch.linalg.inv(augmented_ul_cov_mat_temp))
                )
            if "ar2" in self.v_active_components.keys() and self.v_active_components["ar2"]:
                mat_temp = inv_augmented_ul_cov_mats[i_bs]
                inv2_augmented_ul_cov_mats.append(torch.matmul(mat_temp, mat_temp))
            # mu.append(mu_root)  # indexing return tensor

        # Running_Stats
        diagl_scale_batch = []
        bias_scale_batch = []

        # Calculation of V
        """ADDED FOR TESTING"""
        v_out_vanilla = []
        """"""
        v_bypass_out = []
        # bss_current_pow = torch.zeros(*batch_size, num_bss, device=device)
        for i_user in range(num_users):
            i_bs = assigned_bs[i_user]
            v_tilde_temp = v_tilde[i_user].unsqueeze(0).transpose(0, -1)  # columns into batch

            """ADDED FOR TESTING"""
            if self.training:
                v_out_vanilla.append(inv_augmented_ul_cov_mats[i_bs] @ v_tilde[i_user])
            """"""

            v_temp = 0
            # ARMA
            if self.v_active_components["ar"]:
                # print(inv_augmented_ul_cov_mats_scaled[i_bs].size(), v_tilde_scaled_temp.size())
                v_temp = v_temp + util.mmchain(
                    inv_augmented_ul_cov_mats[i_bs], v_tilde_temp, self.filter_param["v_param_ar"]
                )

            if "ar2" in self.v_active_components.keys() and self.v_active_components["ar2"]:
                v_temp = v_temp + util.mmchain(
                    inv2_augmented_ul_cov_mats[i_bs],
                    v_tilde_temp,
                    self.filter_param["v_param_ar2"] * self.filter_param["v_param_ar"],
                )  # coupled
            # MA
            if self.v_active_components["ma"]:
                v_temp = v_temp + util.mmchain(
                    augmented_ul_cov_mats[i_bs], v_tilde_temp, self.filter_param["v_param_ma"]
                )
            # Diagload
            if self.v_active_components["diagload"]:
                if self.diaglnorm == "trace":
                    diagln = (
                        util.btrace(inv_augmented_ul_cov_mats[i_bs])
                        .real.unsqueeze(-1)
                        .unsqueeze(-1)
                        / bss_dim[i_bs]
                    )
                    diagl_scale_batch.append(diagln.detach().view(-1))
                else:
                    diagln = 1
                if self.diaglnorm:
                    diagl_scale_term = (
                        (1 - self.running_mean_alpha**self.running_mean_num_tracked_batches)
                        * diagln
                        / self.running_mean_diagl_scale
                    )  # scale term biascorrected
                else:
                    diagl_scale_term = 1
                # print("diag", self.running_mean_diagl_scale)
                # print("bias", self.running_mean_bias_scale / (1 - self.running_mean_alpha**self.running_mean_num_tracked_batches))
                v_temp = v_temp + util.mmchain(
                    diagl_scale_term * v_tilde_temp, self.filter_param["v_param_diagload"]
                )

            # Bias
            if self.v_active_components["bias"]:
                if self.biasnorm:
                    if self.biasnorm == "streamwise":
                        bias_scale = torch.sqrt(
                            bss_maxpow[..., i_bs].unsqueeze(-1).unsqueeze(-1)
                            / num_streams_assigned[i_bs]
                        )  # potential improvement
                    else:  # old one
                        bias_scale = torch.sqrt(
                            bss_maxpow[..., i_bs].unsqueeze(-1).unsqueeze(-1)
                            / len(scenario["bss_assign"][i_bs])
                        )  # bss_pow/num_users
                    """print(num_streams_assigned)"""
                    bias_scale_batch.append(bias_scale.detach())
                    bias_scale_term = (
                        (1 - self.running_mean_alpha**self.running_mean_num_tracked_batches)
                        * bias_scale
                        / self.running_mean_bias_scale
                    )  # scale term biascorrected
                else:
                    bias_scale_term = 1

                if not self.v_active_components["nonlin"] == "modrelu":
                    v_temp = v_temp + bias_scale_term * self.filter_param["v_param_bias"]

            if self.v_bypass_param and len(v_bypass_in) > 0:
                v_temp = v_temp + torch.matmul(v_bypass_in[i_user], self.v_bypass_mat)

            if self.v_bypass_param and self.v_bypass_param["position"] == "before_nonlin":
                # print("outbef")
                v_bypass_out.append(v_temp)

            # Nonlin
            if self.v_active_components["nonlin"]:
                if self.v_active_components["nonlin"] == "modrelu":
                    if self.biasnorm:
                        scaled_filter_param = (
                            self.filter_param["v_param_bias"].real * bias_scale_term
                        )
                    else:
                        scaled_filter_param = self.filter_param["v_param_bias"].real
                    v_temp = util.complex_mod_relu(v_temp, scaled_filter_param)
                else:
                    v_temp = util.complex_relu(v_temp)
                if self.v_bypass_param and self.v_bypass_param["position"] == "after_nonlin":
                    v_bypass_out.append(v_temp)
                    # print("outaft")

                v_temp = torch.matmul(v_temp, self.filter_param["v_param_recomb"])
            else:
                v_temp = torch.matmul(v_temp, recomb)  # swap columns back

            v_temp = v_temp.transpose(0, -1).squeeze(0)
            v_out.append(v_temp)

            # print(bss_maxpow[..., i_bs].size())
            # print(util.bf_mat_pow(v_temp).size())
            # print(torch.maximum(bss_maxpow[..., i_bs], util.bf_mat_pow(v_temp)))
            # bss_current_pow[..., i_bs] += util.bf_mat_pow(v_temp)

        # Gradient friendly power summation
        bss_current_pow = []

        for i_bs in range(num_bss):
            pow = 0
            for i_user in scenario["bss_assign"][i_bs]:
                pow = pow + util.bf_mat_pow(v_out[i_user])
            bss_current_pow.append(pow)
        bss_current_pow = torch.stack(bss_current_pow, dim=-1)

        """ADDED FOR TESTING"""
        if self.training and REMOVE_BAD_SAMPLES:
            bss_current_pow_vanilla = []
            for i_bs in range(num_bss):
                pow = 0
                for i_user in scenario["bss_assign"][i_bs]:
                    pow = pow + util.bf_mat_pow(v_out_vanilla[i_user])
                bss_current_pow_vanilla.append(pow)
            bss_current_pow_vanilla = torch.stack(bss_current_pow_vanilla, dim=-1)
            error_occured = torch.any(bss_current_pow_vanilla > bss_maxpow * 1.1, dim=-1)
        else:
            error_occured = torch.tensor(False)
        """"""

        # scale track
        if self.training:
            if len(bias_scale_batch) > 0 or len(diagl_scale_batch) > 0:
                self.running_mean_num_tracked_batches += 1
            if len(bias_scale_batch) > 0:
                self.running_mean_bias_scale = (
                    self.running_mean_alpha * self.running_mean_bias_scale
                    + (1 - self.running_mean_alpha) * torch.cat(bias_scale_batch).mean()
                )
                # print(self.running_mean_bias_scale / (1 - self.running_mean_alpha**self.running_mean_num_tracked_batches))
            if len(diagl_scale_batch) > 0:
                self.running_mean_diagl_scale = (
                    self.running_mean_alpha * self.running_mean_diagl_scale
                    + (1 - self.running_mean_alpha) * torch.cat(diagl_scale_batch).mean()
                )

        # Power clamping
        bss_current_pow = torch.maximum(
            bss_maxpow, bss_current_pow
        )  # pre clamping, to avoid any division underflows
        correction_factor = torch.clamp(torch.sqrt(bss_maxpow / bss_current_pow), max=1)
        # print(correction_factor)
        # print(bss_current_pow)
        for i_user in range(num_users):
            i_bs = assigned_bs[i_user]
            """v_temp = util.passing(v_out[i_user], torch.tensor(self.layer_id + 10))
            v_temp = v_temp * correction_factor[..., i_bs].unsqueeze(-1).unsqueeze(-1)
            v_temp = util.passing(v_temp, torch.tensor(self.layer_id + 20))
            v_out[i_user] = v_temp"""
            v_out[i_user] = v_out[i_user] * correction_factor[..., i_bs].unsqueeze(-1).unsqueeze(-1)

        if not self.v_bypass_param:
            v_bypass_out = v_bypass_in

        return v_out, u_out, w_out, v_bypass_out, error_occured


class GCNWMMSE_SISOAdhoc(nn.Module):
    def __init__(
        self,
        num_layers=5,
        biasnorm=False,
        w_poly=None,
        v_num_channels=4,
        v_active_components=None,
        v_bypass_param=None,
        shared_parameters=True,
        device=torch.device("cpu"),
    ):
        super(GCNWMMSE_SISOAdhoc, self).__init__()
        if v_active_components is None:
            v_active_components = {
                "ar": True,
                "ma": True,
                "diagload": True,
                "nonlin": True,
                "bias": False,
            }

        # register parameters
        self.device = device
        self.num_layers = num_layers
        if shared_parameters:
            self.layers = nn.ModuleList(
                [
                    GCNWMMSELayer_SISOAdhoc(
                        biasnorm=biasnorm,
                        w_poly=w_poly,
                        v_num_channels=v_num_channels,
                        v_active_components=v_active_components,
                        v_bypass_param=v_bypass_param,
                        device=device,
                    )
                ]
                * num_layers
            )
        else:
            layers = []
            for i_layer in range(num_layers):
                layers.append(
                    GCNWMMSELayer_SISOAdhoc(
                        biasnorm=biasnorm,
                        w_poly=w_poly,
                        v_num_channels=v_num_channels,
                        v_active_components=v_active_components,
                        v_bypass_param=v_bypass_param,
                        layer_id=i_layer,
                        device=device,
                    )
                )
            self.layers = nn.ModuleList(layers)

    def forward(self, scenario, layers="all", debug=False):
        if layers == "all":
            num_forward_layers = self.num_layers
        elif isinstance(layers, int):
            assert self.num_layers >= layers
            num_forward_layers = layers
        else:
            raise ValueError

        # Modified WMMSE
        v = algo.downlink_mrc(scenario)

        v_layers = [v]
        u_layers = []
        w_layers = []
        v_bypass = []

        channel_mat, user_noise_pow, bss_pow = self.scenario_extraction(scenario)
        v = self.vectorize(v)

        for i_layer in range(num_forward_layers):
            if debug:
                print("Layer", i_layer + 1)
            v, u, w, v_bypass = self.layers[i_layer](
                channel_mat, user_noise_pow, bss_pow, v, v_bypass
            )
            v_layers.append(self.devectorize(v))
            u_layers.append(self.devectorize(u))
            w_layers.append(self.devectorize(w))

        # Restructure to list of users
        v_out = []
        u_out = []
        w_out = []
        for v_user in list(map(list, zip(*v_layers))):  # transposes layer and user dim
            v_out.append(torch.stack(v_user, dim=0))  # stacks layers
        for u_user in list(map(list, zip(*u_layers))):  # transposes layer and user dim
            u_out.append(torch.stack(u_user, dim=0))
        for w_user in list(map(list, zip(*w_layers))):  # transposes layer and user dim
            w_out.append(torch.stack(w_user, dim=0))

        # power
        num_users = scenario["num_users"]
        assigned_bs = scenario["users_assign"]
        with torch.no_grad():
            bss_pow = torch.zeros(
                num_forward_layers + 1, *list(scenario["bss_pow"].size()), device=self.device
            )
            for i_user in range(num_users):
                i_bs = assigned_bs[i_user]
                bss_pow[..., i_bs] = bss_pow[..., i_bs] + util.bf_mat_pow(v_out[i_user])

        return v_out, u_out, w_out, bss_pow

    @staticmethod
    def scenario_extraction(scenario):
        num_users = scenario["num_users"]
        num_bss = scenario["num_bss"]
        device = scenario["device"]
        assert num_users == num_bss
        user_noise_pow = scenario["users_noise_pow"]
        bss_pow = scenario["bss_pow"]

        channels = scenario["channels"]
        dtype = channels[0][0].dtype
        batch_size = list(channels[0][0].size())[:-2]

        channel_mat = torch.zeros(*batch_size, num_users, num_bss, dtype=dtype, device=device)
        for i_user in range(num_users):
            for i_bs in range(num_bss):
                channel_mat[..., i_user, i_bs] = channels[i_bs][i_user].squeeze(-1).squeeze(-1)
        return channel_mat, user_noise_pow, bss_pow

    @staticmethod
    def vectorize(v_list):
        v_vec = torch.cat(v_list, dim=-2)
        return v_vec

    @staticmethod
    def devectorize(v_vec):
        v_list = list(torch.split(v_vec, 1, dim=-2))
        return v_list


class GCNWMMSELayer_SISOAdhoc(nn.Module):
    def __init__(
        self,
        biasnorm=False,
        w_poly=None,
        v_num_channels=4,
        v_active_components=None,
        v_bypass_param=None,  # {"position": "before_nonlin"/"after_nonlin"}
        layer_id=None,
        device=torch.device("cpu"),
    ):
        super(GCNWMMSELayer_SISOAdhoc, self).__init__()
        if v_active_components is None:
            v_active_components = {
                "ar": True,
                "ma": True,
                "diagload": True,
                "nonlin": True,
                "bias": False,
            }

        # register parameters
        self.layer_id = layer_id
        self.device = device
        self.biasnorm = biasnorm
        self.w_poly = w_poly
        self.v_num_channels = v_num_channels
        self.v_active_components = v_active_components
        self.v_bypass_param = v_bypass_param

        dtype = torch.float64
        ctype = torch.complex128

        filter_param = {}
        if self.w_poly:
            filter_param["w_param_poly"] = nn.Parameter(
                torch.ones(self.w_poly["degree"] + 1, dtype=dtype, device=device)
                / (self.w_poly["degree"] + 1)
            )  # F2=channels, F1=1

        if v_active_components["ar"]:
            filter_param["v_param_ar"] = nn.Parameter(
                util.randcn(1, v_num_channels, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(v_num_channels + 1, device=device))
            )

        if v_active_components["diagload"]:
            filter_param["v_param_diagload"] = nn.Parameter(
                util.randcn(1, v_num_channels, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(v_num_channels + 1, device=device))
            )

        if v_active_components["bias"]:
            filter_param["v_param_bias"] = nn.Parameter(
                torch.zeros(1, v_num_channels, device=device, dtype=ctype)
            )

        if v_active_components["nonlin"]:
            filter_param["v_param_recomb"] = nn.Parameter(
                util.randcn(v_num_channels, 1, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(v_num_channels + 1, device=device))
            )

        if v_bypass_param:
            self.v_bypass_mat = nn.Parameter(
                util.randcn(v_num_channels, v_num_channels, device=device, dtype=dtype)
                / torch.sqrt(torch.tensor(2 * v_num_channels, device=device))
            )  # sqrt(#in + #out)

        # buffer for diaglnorm and biasnorm
        if self.biasnorm or self.diaglnorm:
            self.running_mean_alpha = 0.99
            self.running_mean_bias_scale: Optional[Tensor]
            self.running_mean_num_tracked_batches: Optional[Tensor]
            self.register_buffer(
                "running_mean_bias_scale", torch.tensor(1, dtype=dtype, device=device)
            )
            self.register_buffer("running_mean_num_tracked_batches", torch.tensor(0, device=device))

        # we do not want a situation a * b where both a and b are parameters without some function inbetween
        self.filter_param = nn.ParameterDict(filter_param)

    def forward(self, channel_mat, user_noise_pow, bss_maxpow, v_in, v_bypass_in):
        def u_step(channel_mat, user_noise_pow, v):
            u_tilde = torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1) * v
            cov = torch.matmul(channel_mat.abs().square(), v.abs().square()).sum(
                dim=-1, keepdim=True
            ) + user_noise_pow.unsqueeze(-1)
            u = u_tilde / cov
            return u

        def w_step(channel_mat, v, u):
            error = 1 - u.conj() * torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1) * v
            w_temp = 1 / error
            w_temp = w_temp.unsqueeze(-1)  # get into matrix formm

            if self.w_poly:
                w_filtered = util.matpoly_simple_norm(
                    w_temp,
                    torch.abs(self.filter_param["w_param_matrix"]),
                    self.w_poly["norm_above"],
                )
            else:
                w_filtered = w_temp

            return w_filtered.squeeze(-1)

        def v_step(channel_mat, bss_maxpow, u, w, v_byp_in):
            v_byp_out = 0
            v_tilde = torch.diagonal(channel_mat, dim1=-2, dim2=-1).unsqueeze(-1).conj() * u * w
            ul_cov = u.abs().square() * w.abs()
            ul_cov = (
                torch.matmul(channel_mat.abs().square().transpose(-2, -1), ul_cov)
                .abs()
                .sum(dim=-1, keepdim=True)
            )

            # SISO Adhoc Lagrangian
            mu = torch.abs(v_tilde) / torch.sqrt(bss_maxpow).unsqueeze(-1) - ul_cov
            mu = torch.clamp(mu, min=0)

            ul_cov_aug = ul_cov + mu

            v_temp = 0

            # ARMA
            if self.v_active_components["ar"]:
                v_temp = v_temp + util.mmchain(
                    (v_tilde / ul_cov_aug).unsqueeze(-1), self.filter_param["v_param_ar"]
                )

            # Diagload
            if self.v_active_components["diagload"]:
                v_temp = v_temp + util.mmchain(
                    v_tilde.unsqueeze(-1), self.filter_param["v_param_diagload"]
                )

            # Bias
            if self.v_active_components["bias"]:
                if self.biasnorm:
                    bias_scale = torch.sqrt(
                        bss_maxpow.unsqueeze(-1).unsqueeze(-1)
                    )  # bss_pow/num_users
                    bias_scale_term = (
                        (1 - self.running_mean_alpha**self.running_mean_num_tracked_batches)
                        * bias_scale
                        / self.running_mean_bias_scale
                    )  # scale term biascorrected
                else:
                    bias_scale_term = 1

                if not self.v_active_components["nonlin"] == "modrelu":
                    v_temp = (
                        v_temp + bias_scale_term * self.filter_param["v_param_bias"]
                    )  # unsqueezes additionally into link dim

            if self.v_bypass_param and len(v_bypass_in) > 0:
                v_temp = v_temp + torch.matmul(v_byp_in, self.v_bypass_mat)

            if self.v_bypass_param and self.v_bypass_param["position"] == "before_nonlin":
                v_byp_out = v_temp

            # Nonlin
            if self.v_active_components["nonlin"]:
                if self.v_active_components["nonlin"] == "modrelu":
                    if self.biasnorm:
                        scaled_filter_param = (
                            self.filter_param["v_param_bias"].real * bias_scale_term
                        )
                    else:
                        scaled_filter_param = self.filter_param["v_param_bias"].real
                    v_temp = util.complex_mod_relu(v_temp, scaled_filter_param)
                else:
                    v_temp = util.complex_relu(v_temp)

                if self.v_bypass_param and self.v_bypass_param["position"] == "after_nonlin":
                    v_byp_out = v_temp

                v_temp = torch.matmul(v_temp, self.filter_param["v_param_recomb"])
            else:
                v_temp = torch.matmul(v_temp, recomb)  # swap columns back
            v_temp = v_temp.squeeze(-1)  # back to LINKx1 vectors
            v = v_temp

            return v, v_byp_out, bias_scale

        device = channel_mat.device
        ctype = channel_mat.dtype
        rtype = bss_maxpow.dtype  # noqa: F841

        if not self.v_active_components["nonlin"]:
            recomb = torch.ones(self.v_num_channels, 1, device=device, dtype=ctype) / torch.sqrt(
                torch.tensor(self.v_num_channels, device=device)
            )
            recomb = recomb.type(torch.complex128)

        u_out = u_step(channel_mat, user_noise_pow, v_in)
        w_out = w_step(channel_mat, v_in, u_out)
        v_out, v_bypass_out, bias_scale_batch = v_step(
            channel_mat, bss_maxpow, u_out, w_out, v_bypass_in
        )

        # scale track
        if self.training:
            if len(bias_scale_batch) > 0:
                self.running_mean_num_tracked_batches += 1
                self.running_mean_bias_scale = (
                    self.running_mean_alpha * self.running_mean_bias_scale
                    + (1 - self.running_mean_alpha) * bias_scale_batch.mean()
                )

        # Power clamping
        bss_current_pow = v_out.abs().square()
        bss_current_pow = torch.maximum(
            bss_maxpow.unsqueeze(-1), bss_current_pow
        )  # pre clamping, to avoid any division underflows
        correction_factor = torch.clamp(
            torch.sqrt(bss_maxpow.unsqueeze(-1) / bss_current_pow), max=1
        )
        v_out = v_out * correction_factor

        """if torch.any(torch.isnan(v_out)):
            guilty = torch.isnan(v_out)
            print("v", v_out[guilty])
            print("u", u_out[guilty])
            print("w", w_out[guilty])"""

        return v_out, u_out, w_out, v_bypass_out


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_gcnwmmse():
    """GCNWMMSE at tiny size for tracing (1 unrolled layer, few v-filter channels,
    few Lagrangian iterations). Architecture is unmodified from the real repo."""
    torch.manual_seed(0)
    model = GCNWMMSE(
        num_layers=1,
        v_num_channels=2,
        lagrangian_max_iter=2,
        device=torch.device("cpu"),
    )
    model.eval()
    return model


def example_input_gcnwmmse():
    """Matches GCNWMMSE.forward(scenario): a MIMO interfering-broadcast-channel
    scenario dict, built with the repo's own `mimoifc_randcn` generator (2 base
    stations, 1 single-antenna user each)."""
    torch.manual_seed(0)
    scenario = mimoifc_randcn(
        bss_dim=[2, 2],
        users_dim=[1, 1],
        assignment=[0, 1],
        batch_size=[1],
        snr=0,
        precision="double",
    )
    return (scenario,)


MENAGERIE_ENTRIES = [
    ("GCNWMMSE", "build_gcnwmmse", "example_input_gcnwmmse", 2023, MENAGERIE_ZOO),
]
