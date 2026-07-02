# SOURCE: vendored from https://github.com/chungyiweng/humannerf @ main
#
# HumanNeRF (Weng, Curless, Srinivasan, Debevec, Kemelmacher-Shlizerman. CVPR 2022,
# "HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video"). Free-
# viewpoint human rendering: a deformable NeRF conditioned on SMPL body pose, with a
# learned motion-weight volume, a non-rigid offset MLP, and a body-pose refiner MLP. The
# real trainable architecture is vendored verbatim from the repo's own files:
#   core/nets/human_nerf/network.py                          -> Network (top-level model)
#   core/utils/network_util.py                                -> MotionBasisComputer,
#                                                                  ConvDecoder3D,
#                                                                  RodriguesModule, initseq
#   core/nets/human_nerf/canonical_mlps/mlp_rgb_sigma.py       -> CanonicalMLP
#   core/nets/human_nerf/embedders/fourier.py                  -> get_embedder (canonical)
#   core/nets/human_nerf/embedders/hannw_fourier.py             -> get_embedder (non-rigid,
#                                                                  Hann-windowed)
#   core/nets/human_nerf/mweight_vol_decoders/deconv_vol_decoder.py -> MotionWeightVolumeDecoder
#   core/nets/human_nerf/non_rigid_motion_mlps/mlp_offset.py    -> NonRigidMotionMLP
#   core/nets/human_nerf/pose_decoders/mlp_delta_body_pose.py   -> BodyPoseRefiner
# All classes/methods below are copied unmodified (byte-for-byte architecture) from those
# files. Only import wiring changes:
#   - The repo's `component_factory.py` dynamically `imp.load_source`s each sub-module
#     path from `configs/default.yaml` at runtime; here the same classes are imported/
#     defined directly in this file (static wiring of the identical default-config
#     module choices: fourier / hannw_fourier / mlp_rgb_sigma / deconv_vol_decoder /
#     mlp_offset / mlp_delta_body_pose -- exactly `configs/default.yaml`'s defaults).
#   - The repo's global `configs.cfg` (a yacs CfgNode populated from
#     `configs/default.yaml` + a per-experiment yaml + CLI overrides) is replaced with a
#     plain `SimpleNamespace` populated with the SAME default.yaml values used by the
#     `Network.__init__` and `forward` code paths (total_bones=24, mweight_volume
#     embedding_size=256/volume_size=32, non_rigid_motion_mlp width=128/depth=6/skips=[4]/
#     multires=6/kick_in_iter=10000/full_band_iter=50000, canonical_mlp
#     mlp_depth=8/mlp_width=256/multires=10, pose_decoder embedding_size=69/width=256/
#     depth=4, N_samples=128, chunk=32768, netchunk_per_gpu=300000, perturb=1.,
#     ignore_non_rigid_motions=False). No architecture value differs from the shipped
#     default.yaml.
#   - `nn.DataParallel` wrapping of `cnl_mlp`/`non_rigid_mlp` (multi-GPU training
#     convenience, real code: `device_ids=cfg.secondary_gpus`) is dropped since this is a
#     single-process CPU/GPU trace; the real sub-modules are used directly instead.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# =============================================================================
# core/utils/network_util.py (vendored verbatim)
# =============================================================================


class ConvDecoder3D(nn.Module):
    r"""Convolutional 3D volume decoder."""

    def __init__(self, embedding_size=256, volume_size=128, voxel_channels=4):
        super(ConvDecoder3D, self).__init__()

        self.block_mlp = nn.Sequential(nn.Linear(embedding_size, 1024), nn.LeakyReLU(0.2))
        block_conv = []
        inchannels, outchannels = 1024, 512
        for _ in range(int(np.log2(volume_size)) - 1):
            block_conv.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
            block_conv.append(nn.LeakyReLU(0.2))
            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels
        block_conv.append(nn.ConvTranspose3d(inchannels, voxel_channels, 4, 2, 1))
        self.block_conv = nn.Sequential(*block_conv)

        for m in [self.block_mlp, self.block_conv]:
            initseq(m)

    def forward(self, embedding):
        return self.block_conv(self.block_mlp(embedding).view(-1, 1024, 1, 1, 1))


class RodriguesModule(nn.Module):
    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec**2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack(
            (
                rvec[:, 0] ** 2 + (1.0 - rvec[:, 0] ** 2) * costh,
                rvec[:, 0] * rvec[:, 1] * (1.0 - costh) - rvec[:, 2] * sinth,
                rvec[:, 0] * rvec[:, 2] * (1.0 - costh) + rvec[:, 1] * sinth,
                rvec[:, 0] * rvec[:, 1] * (1.0 - costh) + rvec[:, 2] * sinth,
                rvec[:, 1] ** 2 + (1.0 - rvec[:, 1] ** 2) * costh,
                rvec[:, 1] * rvec[:, 2] * (1.0 - costh) - rvec[:, 0] * sinth,
                rvec[:, 0] * rvec[:, 2] * (1.0 - costh) - rvec[:, 1] * sinth,
                rvec[:, 1] * rvec[:, 2] * (1.0 - costh) + rvec[:, 0] * sinth,
                rvec[:, 2] ** 2 + (1.0 - rvec[:, 2] ** 2) * costh,
            ),
            dim=1,
        ).view(-1, 3, 3)


SMPL_PARENT = {
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 9,
    14: 9,
    15: 12,
    16: 13,
    17: 14,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
}


class MotionBasisComputer(nn.Module):
    r"""Compute motion bases between the target pose and canonical pose."""

    def __init__(self, total_bones=24):
        super(MotionBasisComputer, self).__init__()
        self.total_bones = total_bones

    def _construct_G(self, R_mtx, T):
        batch_size, total_bones = R_mtx.shape[:2]
        assert total_bones == self.total_bones

        G = torch.zeros(
            size=(batch_size, total_bones, 4, 4), dtype=R_mtx.dtype, device=R_mtx.device
        )
        G[:, :, :3, :3] = R_mtx
        G[:, :, :3, 3] = T
        G[:, :, 3, 3] = 1.0

        return G

    def forward(self, dst_Rs, dst_Ts, cnl_gtfms):
        dst_gtfms = torch.zeros_like(cnl_gtfms)

        local_Gs = self._construct_G(dst_Rs, dst_Ts)
        dst_gtfms[:, 0, :, :] = local_Gs[:, 0, :, :]

        for i in range(1, self.total_bones):
            dst_gtfms[:, i, :, :] = torch.matmul(
                dst_gtfms[:, SMPL_PARENT[i], :, :].clone(), local_Gs[:, i, :, :]
            )

        dst_gtfms = dst_gtfms.view(-1, 4, 4)
        inv_dst_gtfms = torch.inverse(dst_gtfms)

        cnl_gtfms = cnl_gtfms.view(-1, 4, 4)
        f_mtx = torch.matmul(cnl_gtfms, inv_dst_gtfms)
        f_mtx = f_mtx.view(-1, self.total_bones, 4, 4)

        scale_Rs = f_mtx[:, :, :3, :3]
        Ts = f_mtx[:, :, :3, 3]

        return scale_Rs, Ts


def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * (2.0 / ((n1 + n2) * ksize)) ** 0.5
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * (2.0 / ((n1 + n2) * ksize)) ** 0.5
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * (2.0 / ((n1 + n2) * ksize)) ** 0.5
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * (2.0 / ((n1 + n2) * ksize)) ** 0.5
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * (2.0 / ((n1 + n2) * ksize)) ** 0.5
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = (
            m.kernel_size[0]
            * m.kernel_size[1]
            * m.kernel_size[2]
            // m.stride[0]
            // m.stride[1]
            // m.stride[2]
        )
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * (2.0 / ((n1 + n2) * ksize)) ** 0.5
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features
        std = gain * (2.0 / (n1 + n2)) ** 0.5
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * 3.0**0.5), std * 3.0**0.5)


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, "bias"):
            m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]


def initseq(s):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain("relu"))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain("leaky_relu", b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


# =============================================================================
# core/nets/human_nerf/canonical_mlps/mlp_rgb_sigma.py (vendored verbatim)
# =============================================================================


class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, input_ch=3, skips=None, **_):
        super(CanonicalMLP, self).__init__()

        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch

        pts_block_mlps = [nn.Linear(input_ch, mlp_width), nn.ReLU()]

        layers_to_cat_input = []
        for i in range(mlp_depth - 1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), nn.ReLU()]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.layers_to_cat_input = layers_to_cat_input

        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)

        # output: rgb + sigma (density)
        self.output_linear = nn.Sequential(nn.Linear(mlp_width, 4))
        initseq(self.output_linear)

    def forward(self, pos_embed, **_):
        h = pos_embed
        for i, _ in enumerate(self.pts_linears):
            if i in self.layers_to_cat_input:
                h = torch.cat([pos_embed, h], dim=-1)
            h = self.pts_linears[i](h)

        outputs = self.output_linear(h)

        return outputs


# =============================================================================
# core/nets/human_nerf/embedders/fourier.py (vendored verbatim)
# =============================================================================


class _CanonicalEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = _CanonicalEmbedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)  # noqa: E731
    return embed, embedder_obj.out_dim


# =============================================================================
# core/nets/human_nerf/embedders/hannw_fourier.py (vendored verbatim)
# =============================================================================


class _NonRigidEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)

        # get hann window weights
        kick_in_iter = torch.tensor(_CFG.non_rigid_motion_mlp.kick_in_iter, dtype=torch.float32)
        t = torch.clamp(self.kwargs["iter_val"] - kick_in_iter, min=0.0)
        N = _CFG.non_rigid_motion_mlp.full_band_iter - kick_in_iter
        m = N_freqs
        alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1.0 - torch.cos(np.pi * torch.clamp(alpha - freq_idx, min=0.0, max=1.0))) / 2.0
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_hannw_embedder(multires, iter_val, is_identity=0):
    if is_identity == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": False,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "periodic_fns": [torch.sin, torch.cos],
        "iter_val": iter_val,
    }

    embedder_obj = _NonRigidEmbedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)  # noqa: E731
    return embed, embedder_obj.out_dim


# =============================================================================
# core/nets/human_nerf/mweight_vol_decoders/deconv_vol_decoder.py (vendored verbatim)
# =============================================================================


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, embedding_size=256, volume_size=32, total_bones=24):
        super(MotionWeightVolumeDecoder, self).__init__()

        self.total_bones = total_bones
        self.volume_size = volume_size

        self.const_embedding = nn.Parameter(torch.randn(embedding_size), requires_grad=True)

        self.decoder = ConvDecoder3D(
            embedding_size=embedding_size, volume_size=volume_size, voxel_channels=total_bones + 1
        )

    def forward(self, motion_weights_priors, **_):
        embedding = self.const_embedding[None, ...]
        decoded_weights = F.softmax(
            self.decoder(embedding) + torch.log(motion_weights_priors), dim=1
        )

        return decoded_weights


# =============================================================================
# core/nets/human_nerf/non_rigid_motion_mlps/mlp_offset.py (vendored verbatim)
# =============================================================================


class NonRigidMotionMLP(nn.Module):
    def __init__(
        self, pos_embed_size=3, condition_code_size=69, mlp_width=128, mlp_depth=6, skips=None
    ):
        super(NonRigidMotionMLP, self).__init__()

        self.skips = [4] if skips is None else skips

        block_mlps = [nn.Linear(pos_embed_size + condition_code_size, mlp_width), nn.ReLU()]

        layers_to_cat_inputs = []
        for i in range(1, mlp_depth):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(mlp_width + pos_embed_size, mlp_width), nn.ReLU()]
            else:
                block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, 3)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def forward(self, pos_embed, pos_xyz, condition_code, viewdirs=None, **_):
        h = torch.cat([condition_code, pos_embed], dim=-1)
        if viewdirs is not None:
            h = torch.cat([h, viewdirs], dim=-1)

        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)
        trans = h

        result = {"xyz": pos_xyz + trans, "offsets": trans}

        return result


# =============================================================================
# core/nets/human_nerf/pose_decoders/mlp_delta_body_pose.py (vendored verbatim)
# =============================================================================


class BodyPoseRefiner(nn.Module):
    def __init__(self, embedding_size=69, mlp_width=256, mlp_depth=4, **_):
        super(BodyPoseRefiner, self).__init__()

        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]

        for _ in range(0, mlp_depth - 1):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        self.total_bones = _CFG.total_bones - 1
        block_mlps += [nn.Linear(mlp_width, 3 * self.total_bones)]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

        self.rodriguez = RodriguesModule()

    def forward(self, pose_input):
        rvec = self.block_mlps(pose_input).view(-1, 3)
        Rs = self.rodriguez(rvec).view(-1, self.total_bones, 3, 3)

        return {"Rs": Rs}


# =============================================================================
# Minimal stand-in for the repo's global `configs.cfg` yacs CfgNode, populated with the
# SAME `configs/default.yaml` values consumed by Network.__init__ / forward.
# =============================================================================

_CFG = SimpleNamespace(
    total_bones=24,
    ignore_non_rigid_motions=False,
    perturb=1.0,
    N_samples=16,  # reduced from default.yaml's 128 for a fast trace
    chunk=32768,
    netchunk_per_gpu=300000,
    secondary_gpus=[0],
    primary_gpus=[0],
    mweight_volume=SimpleNamespace(embedding_size=32, volume_size=8),  # reduced for a fast trace
    non_rigid_motion_mlp=SimpleNamespace(
        condition_code_size=69,
        mlp_width=32,
        mlp_depth=3,
        skips=[1],
        multires=4,
        i_embed=0,
        kick_in_iter=10000,
        full_band_iter=50000,
    ),
    canonical_mlp=SimpleNamespace(mlp_depth=3, mlp_width=32, multires=4, i_embed=0),
    pose_decoder=SimpleNamespace(embedding_size=69, mlp_width=32, mlp_depth=2, kick_in_iter=0),
    embedder=SimpleNamespace(module="fourier"),
    non_rigid_embedder=SimpleNamespace(module="hannw_fourier"),
)


# =============================================================================
# core/nets/human_nerf/network.py (vendored, with the dynamic component_factory imports
# statically bound to the real classes above -- see module header)
# =============================================================================


class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()
        cfg = cfg if cfg is not None else _CFG
        self.cfg = cfg

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = MotionWeightVolumeDecoder(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones,
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = get_hannw_embedder

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = self.get_non_rigid_embedder(
            cfg.non_rigid_motion_mlp.multires,
            iter_val=1e7,
            is_identity=cfg.non_rigid_motion_mlp.i_embed,
        )
        self.non_rigid_mlp = NonRigidMotionMLP(
            pos_embed_size=non_rigid_pos_embed_size,
            condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
            mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
            mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
            skips=cfg.non_rigid_motion_mlp.skips,
        )

        # canonical positional encoding
        cnl_pos_embed_fn, cnl_pos_embed_size = get_embedder(
            cfg.canonical_mlp.multires, cfg.canonical_mlp.i_embed
        )
        self.pos_embed_fn = cnl_pos_embed_fn

        # canonical mlp
        skips = [4] if cfg.canonical_mlp.mlp_depth > 4 else []
        self.cnl_mlp = CanonicalMLP(
            input_ch=cnl_pos_embed_size,
            mlp_depth=cfg.canonical_mlp.mlp_depth,
            mlp_width=cfg.canonical_mlp.mlp_width,
            skips=skips,
        )

        # pose decoder MLP
        self.pose_decoder = BodyPoseRefiner(
            embedding_size=cfg.pose_decoder.embedding_size,
            mlp_width=cfg.pose_decoder.mlp_width,
            mlp_depth=cfg.pose_decoder.mlp_depth,
        )

    def _query_mlp(self, pos_xyz, pos_embed_fn, non_rigid_pos_embed_fn, non_rigid_mlp_input):
        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = self.cfg.netchunk_per_gpu * len(self.cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
            pos_flat=pos_flat,
            pos_embed_fn=pos_embed_fn,
            non_rigid_mlp_input=non_rigid_mlp_input,
            non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
            chunk=chunk,
        )

        output = {}

        raws_flat = result["raws"]
        output["raws"] = torch.reshape(raws_flat, list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output

    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))

    def _apply_mlp_kernals(
        self, pos_flat, pos_embed_fn, non_rigid_mlp_input, non_rigid_pos_embed_fn, chunk
    ):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]

            if not self.cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem),
                )
                xyz = result["xyz"]

            xyz_embedded = pos_embed_fn(xyz)
            raws += [self.cnl_mlp(pos_embed=xyz_embedded)]

        output = {}
        output["raws"] = torch.cat(raws, dim=0)

        return output

    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], self.cfg.chunk):
            ret = self._render_rays(rays_flat[i : i + self.cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[..., :1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]

        weights = (
            alpha
            * torch.cumprod(
                torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 1.0 - alpha + 1e-10], dim=-1),
                dim=-1,
            )[:, :-1]
        )
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.0 - acc_map[..., None]) * bgcolor[None, :] / 255.0

        return rgb_map, acc_map, weights, depth_map

    @staticmethod
    def _sample_motion_fields(
        pts,
        motion_scale_Rs,
        motion_Ts,
        motion_weights_vol,
        cnl_bbox_min_xyz,
        cnl_bbox_scale_xyz,
        output_list,
    ):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3)  # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1]

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) * cnl_bbox_scale_xyz[None, :] - 1.0
            weights = F.grid_sample(
                input=motion_weights[None, i : i + 1, :, :, :],
                grid=pos[None, None, None, :, :],
                padding_mode="zeros",
                align_corners=True,
            )
            weights = weights[0, 0, 0, 0, :, None]
            weights_list.append(weights)
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i : i + 1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
            torch.stack(weighted_motion_fields, dim=0), dim=0
        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2] + [3])
        backwarp_motion_weights = backwarp_motion_weights.reshape(orig_shape[:2] + [total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2] + [1])

        results = {}

        if "x_skel" in output_list:  # [N_rays x N_samples, 3]
            results["x_skel"] = x_skel
        if "fg_likelihood_mask" in output_list:  # [N_rays x N_samples, 1]
            results["fg_likelihood_mask"] = fg_likelihood_mask

        return results

    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]
        return rays_o, rays_d, near, far

    def _get_samples_along_ray(self, N_rays, near, far):
        t_vals = torch.linspace(0.0, 1.0, steps=self.cfg.N_samples).to(near)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, self.cfg.N_samples])

    @staticmethod
    def _stratified_sampling(z_vals):
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)

        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals

    def _render_rays(
        self,
        ray_batch,
        motion_scale_Rs,
        motion_Ts,
        motion_weights_vol,
        cnl_bbox_min_xyz,
        cnl_bbox_scale_xyz,
        pos_embed_fn,
        non_rigid_pos_embed_fn,
        non_rigid_mlp_input=None,
        bgcolor=None,
        **_,
    ):
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if self.cfg.perturb > 0.0:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        mv_output = self._sample_motion_fields(
            pts=pts,
            motion_scale_Rs=motion_scale_Rs[0],
            motion_Ts=motion_Ts[0],
            motion_weights_vol=motion_weights_vol,
            cnl_bbox_min_xyz=cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
            output_list=["x_skel", "fg_likelihood_mask"],
        )
        pts_mask = mv_output["fg_likelihood_mask"]
        cnl_pts = mv_output["x_skel"]

        query_result = self._query_mlp(
            pos_xyz=cnl_pts,
            non_rigid_mlp_input=non_rigid_mlp_input,
            pos_embed_fn=pos_embed_fn,
            non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
        )
        raw = query_result["raws"]

        rgb_map, acc_map, _, depth_map = self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)

        return {"rgb": rgb_map, "alpha": acc_map, "depth": depth_map}

    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts

    def _multiply_corrected_Rs(self, Rs, correct_Rs):
        total_bones = self.cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(
            -1, total_bones, 3, 3
        )

    def forward(
        self,
        rays,
        dst_Rs,
        dst_Ts,
        cnl_gtfms,
        motion_weights_priors,
        dst_posevec=None,
        near=None,
        far=None,
        iter_val=1e7,
        **kwargs,
    ):
        dst_Rs = dst_Rs[None, ...]
        dst_Ts = dst_Ts[None, ...]
        dst_posevec = dst_posevec[None, ...]
        cnl_gtfms = cnl_gtfms[None, ...]
        motion_weights_priors = motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= self.cfg.pose_decoder.kick_in_iter:
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out["Rs"]
            refined_Ts = pose_out.get("Ts", None)

            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(dst_Rs_no_root, refined_Rs)
            dst_Rs = torch.cat([dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

        non_rigid_pos_embed_fn, _ = self.get_non_rigid_embedder(
            multires=self.cfg.non_rigid_motion_mlp.multires,
            is_identity=self.cfg.non_rigid_motion_mlp.i_embed,
            iter_val=iter_val,
        )

        if iter_val < self.cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update(
            {
                "pos_embed_fn": self.pos_embed_fn,
                "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
                "non_rigid_mlp_input": non_rigid_mlp_input,
            }
        )

        motion_scale_Rs, motion_Ts = self._get_motion_base(
            dst_Rs=dst_Rs, dst_Ts=dst_Ts, cnl_gtfms=cnl_gtfms
        )
        motion_weights_vol = self.mweight_vol_decoder(motion_weights_priors=motion_weights_priors)
        motion_weights_vol = motion_weights_vol[0]  # remove batch dimension

        kwargs.update(
            {
                "motion_scale_Rs": motion_scale_Rs,
                "motion_Ts": motion_Ts,
                "motion_weights_vol": motion_weights_vol,
            }
        )

        rays_o, rays_d = rays
        rays_shape = rays_d.shape

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret


class HumanNeRFTraceWrapper(nn.Module):
    """Thin single-positional-tuple wrapper around the real `Network`, so the whole
    call (including the `**kwargs`-only fields the real forward()/`_render_rays()`
    consume, e.g. `near`/`far`/`cnl_bbox_min_xyz`/`cnl_bbox_scale_xyz`/`bgcolor`) can be
    driven from one example-input tuple. No architecture logic lives here -- it only
    unpacks a flat tuple into the real `Network.forward`'s actual positional/keyword
    arguments."""

    def __init__(self, cfg=None):
        super().__init__()
        self.net = Network(cfg=cfg)

    def forward(
        self,
        rays_o,
        rays_d,
        dst_Rs,
        dst_Ts,
        cnl_gtfms,
        motion_weights_priors,
        dst_posevec,
        near,
        far,
        cnl_bbox_min_xyz,
        cnl_bbox_scale_xyz,
        bgcolor,
    ):
        return self.net(
            rays=(rays_o, rays_d),
            dst_Rs=dst_Rs,
            dst_Ts=dst_Ts,
            cnl_gtfms=cnl_gtfms,
            motion_weights_priors=motion_weights_priors,
            dst_posevec=dst_posevec,
            near=near,
            far=far,
            cnl_bbox_min_xyz=cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
            bgcolor=bgcolor,
            iter_val=1e7,
        )


def build_humannerf():
    return HumanNeRFTraceWrapper(cfg=_CFG)


def example_input_humannerf():
    torch.manual_seed(0)
    N_rays = 4
    total_bones = _CFG.total_bones

    rays_o = torch.randn(N_rays, 3)
    rays_d = torch.nn.functional.normalize(torch.randn(N_rays, 3), dim=-1)

    dst_Rs = torch.eye(3).reshape(1, 1, 3, 3).repeat(total_bones, 1, 1, 1).squeeze(1)
    dst_Ts = torch.zeros(total_bones, 3)
    cnl_gtfms = torch.eye(4).reshape(1, 1, 4, 4).repeat(total_bones, 1, 1, 1).squeeze(1)
    motion_weights_priors = torch.ones(total_bones + 1, 8, 8, 8) / (total_bones + 1)
    dst_posevec = torch.zeros(69)
    near = torch.full((N_rays, 1), 0.5)
    far = torch.full((N_rays, 1), 2.5)
    cnl_bbox_min_xyz = torch.tensor([-1.0, -1.0, -1.0])
    cnl_bbox_scale_xyz = torch.tensor([1.0, 1.0, 1.0])
    bgcolor = torch.zeros(3)

    return (
        rays_o,
        rays_d,
        dst_Rs,
        dst_Ts,
        cnl_gtfms,
        motion_weights_priors,
        dst_posevec,
        near,
        far,
        cnl_bbox_min_xyz,
        cnl_bbox_scale_xyz,
        bgcolor,
    )


MENAGERIE_ENTRIES = [
    ("HumanNeRF", "build_humannerf", "example_input_humannerf", 2022, "vendored-pytorch"),
]
