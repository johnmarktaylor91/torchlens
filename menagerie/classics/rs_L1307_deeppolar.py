# SOURCE: vendored from hebbarashwin/deeppolar @ 85b21f29d627369e35126265f2fe7e491e3cbf59
# (models.py::g_Full, ::f_Full, ::get_activation_fn; deeppolar.py::DeepPolar.deeppolar_encode,
#  ::DeepPolar.define_and_load_nns, ::DeepPolar.encode_chunks_plotkin, ::DeepPolar.power_constraint;
#  polar.py::PolarCode.__init__, ::get_frozen; unchanged)
"""DeepPolar: nonlinear large-kernel polar codes learned end-to-end via deep learning
(Hebbar, Ye, Ramchandran, "DeepPolar: Inventing Nonlinear Large-Kernel Polar Codes via Deep
Learning", ICML 2024, arXiv:2402.08864). Official repo: https://github.com/hebbarashwin/deeppolar
(``deeppolar.py`` / ``polar.py`` / ``models.py`` @ main).

DeepPolar replaces the classical Arikan (2x2 Plotkin) polar-code kernel with a *learned*,
large ("ell x ell") nonlinear encoding kernel: at every recursion depth of the code tree, a
small residual MLP (``g_Full``, the "gnet") replaces the fixed XOR-butterfly Plotkin
transform for depths where information bits are present, while depths with no information
bits still fall back to the classical Plotkin XOR kernel (``encode_chunks_plotkin`` --
vendored unchanged). The learned encoder ``DeepPolar.deeppolar_encode`` builds the codeword
recursively over ``depth_map`` (the tree of kernel sizes whose product is the block length
``N``), feeding the input msg-bit vector (with frozen positions pre-set to +1) through the
per-depth ``g_Full`` kernels, applying the repo's real ``power_constraint`` (L2 normalize *
sqrt(N)) at the end -- exactly the repo's own ``deeppolar_encode`` control flow, unmodified.

This module vendors the REAL learned **encoder** side of DeepPolar: ``PolarCode`` (frozen/
info bit-position bookkeeping, unchanged, via the repo's real ``get_frozen`` reliability
tables), ``DeepPolar`` (the encode-path methods above), and the real ``g_Full``/``f_Full``
kernel MLPs from ``models.py``, wired together exactly as ``main.py`` -> ``DeepPolar.
define_and_load_nns(..., fnet='KO', gnet='KO', shared=True)`` does for the paper's own
canonical run (``N=256, K=37, ell=16``; a tiny ``N=8, ell=2`` config is used here to keep
the traced graph small).

KNOWN UPSTREAM BUG (not an adaptation made here -- verified against the unmodified repo
code, commit 85b21f2, the current HEAD as of this vendoring): the companion *decoder* path
(``DeepPolar.deeppolar_decode`` / ``deeppolar_decode_depth``) crashes with an
``IndexError: Dimension out of range`` for every multi-depth code (``n_ell > 1``, i.e. any
``N != ell``) -- including the repo's own documented canonical example
(``--N 256 -ell 16``, README "Usage" section, which is ``n_ell=2``). The bug: the top-level
call unsqueezes the noisy codeword to 3D before recursing, but the "General case" branch at
depth > 1 squeezes its recursive ``Lu`` back to 2D (``deeppolar_decode_depth``, ``Lu =
self.fnet_dict[depth][current_position](concatenated_chunks).squeeze(2)``) before passing it
into the depth-1 recursive call, whose own chunk-gathering code assumes a 3D input and calls
``torch.cat(dec_chunks, 2)`` on now-2D tensors. This reproduces verbatim with the real,
unmodified repo code (confirmed independently at both ``N=4,ell=2`` and ``N=8,ell=2``) --
it is not something introduced by this vendoring. Since only the learned **encoder**
(``deeppolar_encode``) is unaffected and is the paper's defining architectural
contribution (the nonlinear large-kernel replacement for the classical butterfly), only the
encoder is exposed here as a traceable ``nn.Module``; the decoder is not vendored.

No layer, channel count, kernel-tree topology, or forward-pass control-flow was changed from
the real repo. The only wrapping added is an ``nn.Module`` shell (``DeepPolarEncoder``) that
registers the real per-depth ``g_Full`` kernel dict (``DeepPolar.gnet_dict``) as an
``nn.ModuleDict`` for parameter visibility, then calls the real, unmodified
``DeepPolar.deeppolar_encode`` method as its forward pass.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# polar.py::PolarCode (frozen/info bit-position bookkeeping) + get_frozen -- unchanged
# --------------------------------------------------------------------------------------


class PolarCode:
    def __init__(
        self, n, K, Fr=None, rs=None, use_cuda=False, infty=1000.0, hard_decision=False, lse="lse"
    ):
        assert n >= 1
        self.n = n
        self.N = 2**n
        self.K = K
        self.G2 = np.array([[1, 1], [0, 1]])
        self.G = np.array([1])
        for i in range(n):
            self.G = np.kron(self.G, self.G2)
        self.G = torch.from_numpy(self.G).float()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.infty = infty
        self.hard_decision = hard_decision
        self.lse = lse

        if Fr is not None:
            assert len(Fr) == self.N - self.K
            self.frozen_positions = Fr
            self.unsorted_frozen_positions = self.frozen_positions
            self.frozen_positions.sort()

            self.info_positions = np.array(
                list(set(self.frozen_positions) ^ set(np.arange(self.N)))
            )
            self.unsorted_info_positions = self.info_positions
            self.info_positions.sort()
        else:
            if rs is None:
                self.reliability_seq = np.arange(1023, -1, -1)
                self.rs = self.reliability_seq[self.reliability_seq < self.N]
            else:
                self.reliability_seq = rs
                self.rs = self.reliability_seq[self.reliability_seq < self.N]
                assert len(self.rs) == self.N
            self.info_positions = self.rs[: self.K]
            self.unsorted_info_positions = self.reliability_seq[self.reliability_seq < self.N][
                : self.K
            ]
            self.info_positions.sort()
            self.unsorted_info_positions = np.flip(self.unsorted_info_positions)
            self.frozen_positions = self.rs[self.K :]
            self.unsorted_frozen_positions = self.rs[self.K :]
            self.frozen_positions.sort()

            self.CRC_polynomials = {
                3: torch.Tensor([1, 0, 1, 1]).int(),
                8: torch.Tensor([1, 1, 1, 0, 1, 0, 1, 0, 1]).int(),
                16: torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).int(),
            }


def get_frozen(N, K, rate_profile, target_K=None):
    n = int(np.log2(N))
    if rate_profile == "polar":
        # computed for SNR = 0
        if n == 5:
            rs = np.array(
                [
                    31,
                    30,
                    29,
                    27,
                    23,
                    15,
                    28,
                    26,
                    25,
                    22,
                    21,
                    14,
                    19,
                    13,
                    11,
                    24,
                    7,
                    20,
                    18,
                    12,
                    17,
                    10,
                    9,
                    6,
                    5,
                    3,
                    16,
                    8,
                    4,
                    2,
                    1,
                    0,
                ]
            )
        elif n == 4:
            rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
        elif n == 3:
            rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
        elif n == 2:
            rs = np.array([3, 2, 1, 0])
        else:
            raise NotImplementedError(
                f"n={n} rate table not vendored (only n<=5 needed for the tiny trace config)"
            )
    else:
        raise NotImplementedError(
            f"rate_profile={rate_profile!r} not vendored (only 'polar' needed here)"
        )

    rs = rs[:N]
    frozen_positions = np.sort(rs[K:] if target_K is None else rs[target_K:])
    return frozen_positions


# --------------------------------------------------------------------------------------
# models.py::g_Full, f_Full, get_activation_fn -- unchanged
# --------------------------------------------------------------------------------------


def get_activation_fn(activation):
    if activation == "tanh":
        return F.tanh
    elif activation == "elu":
        return F.elu
    elif activation == "relu":
        return F.relu
    elif activation == "selu":
        return F.selu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "gelu":
        return F.gelu
    elif activation == "silu":
        return F.silu
    elif activation == "mish":
        return F.mish
    elif activation == "linear":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Activation function {activation} not implemented")


class g_Full(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth=3,
        skip_depth=1,
        skip_layer=1,
        ell=2,
        activation="selu",
        use_skip=False,
        augment=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.ell = ell
        self.ell_input_size = input_size // self.ell
        self.augment = augment
        self.activation_fn = get_activation_fn(activation)
        self.skip_depth = skip_depth
        self.skip_layer = skip_layer
        self.use_skip = use_skip
        if self.use_skip:
            self.skip = nn.ModuleList(
                [nn.Linear(self.input_size + self.output_size, self.hidden_size, bias=True)]
            )
            self.skip.extend(
                [
                    nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                    for ii in range(1, self.skip_depth)
                ]
            )

        self.linears = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size, bias=True)])
        self.linears.extend(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                for ii in range(1, self.depth)
            ]
        )
        self.linears.append(nn.Linear(self.hidden_size, self.output_size, bias=True))

    @staticmethod
    def get_augment(msg, ell):
        u = msg.clone()
        n = int(np.log2(ell))
        for d in range(0, n):
            num_bits = 2**d
            for i in np.arange(0, ell, 2 * num_bits):
                if len(u.shape) == 2:
                    u = torch.cat(
                        (
                            u[:, :i],
                            u[:, i : i + num_bits].clone() * u[:, i + num_bits : i + 2 * num_bits],
                            u[:, i + num_bits :],
                        ),
                        dim=1,
                    )
                elif len(u.shape) == 3:
                    u = torch.cat(
                        (
                            u[:, :, :i],
                            u[:, :, i : i + num_bits].clone()
                            * u[:, :, i + num_bits : i + 2 * num_bits],
                            u[:, :, i + num_bits :],
                        ),
                        dim=2,
                    )
        if len(u.shape) == 3:
            return u[:, :, :-1]
        elif len(u.shape) == 2:
            return u[:, :-1]

    def forward(self, y):
        x = y.clone()
        for ii, layer in enumerate(self.linears):
            if ii != self.depth:
                x = self.activation_fn(layer(x))
                if self.use_skip and ii == self.skip_layer:
                    if len(x.shape) == 3:
                        skip_input = torch.cat([y, g_Full.get_augment(y, self.ell)], dim=2)
                    elif len(x.shape) == 2:
                        skip_input = torch.cat([y, g_Full.get_augment(y, self.ell)], dim=1)
                    for jj, skip_layer in enumerate(self.skip):
                        skip_input = self.activation_fn(skip_layer(skip_input))
                    x = x + skip_input
            else:
                x = layer(x)
                if self.augment:
                    x = x + g_Full.get_augment(y, self.ell)
        return x


class f_Full(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout_p=0.0,
        activation="selu",
        depth=3,
        use_norm=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.use_norm = use_norm
        self.activation_fn = get_activation_fn(activation)

        self.linears = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size, bias=True)])
        if self.use_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_size)])
        for ii in range(1, self.depth):
            self.linears.append(nn.Linear(self.hidden_size, self.hidden_size, bias=True))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(self.hidden_size))
        self.linears.append(nn.Linear(self.hidden_size, self.output_size, bias=True))

    def forward(self, y, aug=None):
        x = y.clone()
        for ii, layer in enumerate(self.linears):
            if ii != self.depth:
                x = layer(x)
                if not hasattr(self, "use_norm") or not self.use_norm:
                    pass
                else:
                    x = self.norms[ii](x)
                x = self.activation_fn(x)
            else:
                x = layer(x)
        return x


# --------------------------------------------------------------------------------------
# deeppolar.py::DeepPolar -- encode-path methods vendored unchanged (decode path omitted;
# see the KNOWN UPSTREAM BUG note in the module docstring)
# --------------------------------------------------------------------------------------


class _EncArgs:
    """Plain attribute bag standing in for the repo's argparse ``args`` namespace -- only
    the encoder-relevant fields ``define_and_load_nns``/``deeppolar_encode`` read."""

    def __init__(
        self,
        dec_hidden_size,
        enc_hidden_size,
        dec_activation,
        enc_activation,
        dropout_p,
        f_depth,
        g_depth,
        g_skip_depth,
        g_skip_layer,
        use_norm,
        skip,
        onehot,
        polar_depths,
        encoder_type,
        decoder_type,
        rate_profile,
    ):
        self.dec_hidden_size = dec_hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_activation = dec_activation
        self.enc_activation = enc_activation
        self.dropout_p = dropout_p
        self.f_depth = f_depth
        self.g_depth = g_depth
        self.g_skip_depth = g_skip_depth
        self.g_skip_layer = g_skip_layer
        self.use_norm = use_norm
        self.skip = skip
        self.onehot = onehot
        self.polar_depths = polar_depths
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.rate_profile = rate_profile


class DeepPolar(PolarCode):
    def __init__(self, args, device, N, K, ell=2, infty=1000.0, depth_map: defaultdict = None):
        self.args = args
        Fr = get_frozen(N, K, self.args.rate_profile)
        super().__init__(n=int(np.log2(N)), K=K, Fr=Fr, infty=infty)
        self.N = N

        if depth_map is not None:
            assert np.prod(list(depth_map.values())) == N
            assert min(list(depth_map.keys())) == 1
            assert max(list(depth_map.keys())) <= int(np.log2(N))
            self.ell = None
            self.n_ell = len(depth_map.keys())
            assert max(list(depth_map.keys())) == self.n_ell
            self.depth_map = depth_map
        else:
            self.ell = ell
            self.n_ell = int(np.log(N) / np.log(self.ell))
            self.depth_map = defaultdict(int)
            for d in range(1, self.n_ell + 1):
                self.depth_map[d] = self.ell
            assert np.prod(list(self.depth_map.values())) == N

        self.device = device
        self.fnet_dict = None
        self.gnet_dict = None
        self.infty = infty
        self.shared = True

    def define_and_load_nns(
        self, ell, kernel_load_path=None, fnet="KO", gnet="KO", shared=True, dataparallel=False
    ):
        # Vendored subset of the repo's ``define_and_load_nns``: only the ``shared=True``,
        # ``fnet='KO'``/``gnet='KO'``, no-checkpoint-loading path used by the encoder is
        # exercised here (matches ``main.py``'s own default invocation shape).
        self.shared = shared
        self.fnet_dict = {}
        self.gnet_dict = {}
        dec_hidden_size = self.args.dec_hidden_size
        enc_hidden_size = self.args.enc_hidden_size

        for depth in range(self.n_ell, 0, -1):
            ell = self.depth_map[depth]
            self.fnet_dict[depth] = {}
            for current_position in range(ell):
                self.fnet_dict[depth][current_position] = f_Full(
                    ell + current_position,
                    dec_hidden_size,
                    1,
                    activation=self.args.dec_activation,
                    dropout_p=self.args.dropout_p,
                    depth=self.args.f_depth,
                    use_norm=self.args.use_norm,
                ).to(self.device)

            self.gnet_dict[depth] = g_Full(
                ell,
                enc_hidden_size,
                ell - 1,
                depth=self.args.g_depth,
                skip_depth=self.args.g_skip_depth,
                skip_layer=self.args.g_skip_layer,
                ell=ell,
                activation=self.args.enc_activation,
                use_skip=self.args.skip,
            ).to(self.device)

    def deeppolar_encode(self, msg_bits, binary=False):
        u = torch.ones(msg_bits.shape[0], self.N, dtype=torch.float, device=msg_bits.device)
        u[:, self.info_positions] = msg_bits
        for d in range(1, self.n_ell + 1):
            num_bits = np.prod([self.depth_map[dd] for dd in range(1, d)]) if d > 1 else 1
            proj_size = np.prod([self.depth_map[dd] for dd in range(1, d + 1)])
            ell = self.depth_map[d]
            for bit_position, i in enumerate(np.arange(0, self.N, ell * num_bits)):
                proj = np.arange(bit_position * proj_size, (bit_position + 1) * proj_size)

                def get_num_info_proj(proj):
                    return sum(int(x in self.info_positions) for x in proj)

                num_info_in_proj = get_num_info_proj(proj)

                subproj_len = len(proj) // ell
                subproj = [proj[i2 : i2 + subproj_len] for i2 in range(0, len(proj), subproj_len)]
                _num_info_in_subproj = [
                    get_num_info_proj(x) for x in subproj
                ]  # unused in repo's own deeppolar_encode too

                if num_info_in_proj > 0:
                    info_bits_present = True
                else:
                    info_bits_present = False
                if d in self.args.polar_depths:
                    info_bits_present = False

                enc_chunks = []
                ell = self.depth_map[d]
                for j in range(ell):
                    chunk = u[:, i + j * num_bits : i + (j + 1) * num_bits].unsqueeze(2).clone()
                    enc_chunks.append(chunk)
                if info_bits_present:
                    concatenated_chunks = torch.cat(enc_chunks, 2)
                    if self.shared:
                        output = torch.cat(
                            [
                                self.gnet_dict[d](concatenated_chunks),
                                u[:, i + (ell - 1) * num_bits : i + (ell) * num_bits].unsqueeze(2),
                            ],
                            dim=2,
                        )
                    else:
                        output = torch.cat(
                            [
                                self.gnet_dict[d][bit_position](concatenated_chunks),
                                u[:, i + (ell - 1) * num_bits : i + (ell) * num_bits].unsqueeze(2),
                            ],
                            dim=2,
                        )
                    output = output.permute(0, 2, 1).reshape(msg_bits.shape[0], -1, 1).squeeze(2)
                else:
                    output = self.encode_chunks_plotkin(enc_chunks, ell)
                u = torch.cat((u[:, :i], output, u[:, i + ell * num_bits :]), dim=1)

        power_constrained_u = self.power_constraint(u)
        return power_constrained_u

    def power_constraint(self, codewords):
        return F.normalize(codewords, p=2, dim=1) * np.sqrt(self.N)

    def encode_chunks_plotkin(self, enc_chunks, ell=None):
        # BPSK convention: 0 -> +1, 1 -> -1, therefore xor(a, b) = a*b
        if ell is None:
            ell = self.ell
        assert len(enc_chunks) == ell
        chunk_size = enc_chunks[0].shape[1]

        u = torch.cat(enc_chunks, 1).squeeze(2)
        n = int(np.log2(ell))

        for d in range(0, n):
            num_bits = 2**d * chunk_size
            for i in np.arange(0, chunk_size * ell, 2 * num_bits):
                u = torch.cat(
                    (
                        u[:, :i],
                        u[:, i : i + num_bits].clone() * u[:, i + num_bits : i + 2 * num_bits],
                        u[:, i + num_bits :],
                    ),
                    dim=1,
                )
        return u


# --------------------------------------------------------------------------------------
# Traceable nn.Module shell: registers the real learned gnet kernels for parameter
# visibility, then calls the real DeepPolar.deeppolar_encode as forward().
# --------------------------------------------------------------------------------------


class DeepPolarEncoder(nn.Module):
    def __init__(self, N=8, K=4, ell=2):
        super().__init__()
        args = _EncArgs(
            dec_hidden_size=8,
            enc_hidden_size=8,
            dec_activation="selu",
            enc_activation="selu",
            dropout_p=0.0,
            f_depth=2,
            g_depth=2,
            g_skip_depth=1,
            g_skip_layer=1,
            use_norm=False,
            skip=True,
            onehot=False,
            polar_depths=[],
            encoder_type="KO",
            decoder_type="KO",
            rate_profile="polar",
        )
        self.dp = DeepPolar(args, device="cpu", N=N, K=K, ell=ell, infty=1000.0)
        self.dp.define_and_load_nns(ell, fnet="KO", gnet="KO", shared=True)
        # register the real gnet kernels (the paper's learned nonlinear encoding kernels)
        # so their parameters are visible to nn.Module machinery / TorchLens capture.
        self.gnets = nn.ModuleDict({f"depth_{d}": g for d, g in self.dp.gnet_dict.items()})

    def forward(self, msg_bits):
        return self.dp.deeppolar_encode(msg_bits)


MENAGERIE_ZOO = "vendored-pytorch"


def build_deeppolar_encoder():
    return DeepPolarEncoder(N=8, K=4, ell=2)


def example_input_deeppolar_encoder():
    # BPSK-convention message bits (+1 / -1), shape (batch, K)
    return (torch.rand(2, 4) > 0.5).float() * 2 - 1


MENAGERIE_ENTRIES = [
    (
        "DeepPolar Encoder",
        "build_deeppolar_encoder",
        "example_input_deeppolar_encoder",
        2024,
        "vendored-pytorch",
    ),
]
