# SOURCE: vendored from zju3dv/animatable_nerf @ master
#   (https://github.com/zju3dv/animatable_nerf)
#
# Vendors the REAL "Animatable NeRF" / "Animatable Neural Fields" per-frame
# T-pose human network:
#   - lib/networks/embedder.py (positional encoding, verbatim)
#   - lib/utils/blend_utils.py (pose<->tpose blend-weight transforms, verbatim,
#     trimmed to the functions actually exercised by Network.forward)
#   - lib/networks/bw_deform/tpose_nerf_network.py::Network / TPoseHuman /
#     BackwardBlendWeight (verbatim architecture: T-pose canonical NeRF MLP +
#     per-frame neural blend-weight field MLPs, both 8-layer Conv1d-as-Linear
#     stacks with a skip connection, following the original NeRF MLP topology)
#
# The only non-architectural change from the original source is replacing the
# module-level `lib.config.cfg` (a yacs CfgNode populated by argparse + a YAML
# file at import time, which crashes under library import / any non-original
# CLI) with a minimal plain-object shim exposing exactly the scalar fields the
# vendored classes read (`xyz_res`, `view_res`, `num_train_frame`,
# `num_eval_frame`, `norm_th`, `train_th`, `aninerf_animation`,
# `test_novel_pose`). Values are the paper's own published defaults from
# configs/default.yaml (xyz_res=10, view_res=4, train_th=0., norm_th=0.05,
# num_train_frame=260, num_eval_frame=133). No layer, dimension, or forward-pass
# computation was altered.
#
# MENAGERIE_ZOO="vendored-pytorch"

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# --------------------------------------------------------------------------
# Minimal cfg shim (replaces lib.config.cfg; values = configs/default.yaml)
# --------------------------------------------------------------------------
cfg = types.SimpleNamespace(
    xyz_res=10,
    view_res=4,
    num_train_frame=260,
    num_eval_frame=133,
    norm_th=0.05,
    train_th=0.0,
    aninerf_animation=False,
    test_novel_pose=False,
)


# --------------------------------------------------------------------------
# lib/networks/embedder.py (verbatim, using the cfg shim above)
# --------------------------------------------------------------------------
class Embedder:
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

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)  # noqa: E731
    return embed, embedder_obj.out_dim


xyz_embedder, xyz_dim = get_embedder(cfg.xyz_res)
view_embedder, view_dim = get_embedder(cfg.view_res)


class _EmbedderNS:
    xyz_embedder = staticmethod(xyz_embedder)
    view_embedder = staticmethod(view_embedder)


embedder = _EmbedderNS()


# --------------------------------------------------------------------------
# lib/utils/blend_utils.py (verbatim; only the functions Network.forward uses)
# --------------------------------------------------------------------------
def world_points_to_pose_points(wpts, Rh, Th):
    """
    wpts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(wpts - Th, Rh)
    return pts


def pose_points_to_tpose_points(ppts, bw, A):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ppts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    pts = ppts - A[..., :3, 3]
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * pts[:, :, None], dim=3)
    return pts


def pts_sample_blend_weights(pts, bw, bounds):
    """sample blend weights for points
    pts: n_batch, n_points, 3
    bw: n_batch, d, h, w, 25
    bounds: n_batch, 2, 3
    """
    pts = pts.clone()

    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]
    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]

    # the blend weight is indexed by xyz
    bw = bw.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw, grid_coords, padding_mode="border", align_corners=True)
    bw = bw[:, :, 0, 0]

    return bw


# --------------------------------------------------------------------------
# lib/networks/bw_deform/tpose_nerf_network.py (verbatim architecture)
# --------------------------------------------------------------------------
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.bw_latent = nn.Embedding(cfg.num_train_frame + 1, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList(
            [nn.Conv1d(input_ch, W, 1)]
            + [
                nn.Conv1d(W, W, 1) if i not in self.skips else nn.Conv1d(W + input_ch, W, 1)
                for i in range(D - 1)
            ]
        )
        self.bw_fc = nn.Conv1d(W, 24, 1)

        if cfg.aninerf_animation:
            self.novel_pose_bw = BackwardBlendWeight()

    def get_bw_feature(self, pts, ind):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = self.bw_latent(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_neural_blend_weights(self, pose_pts, smpl_bw, latent_index):
        features = self.get_bw_feature(pose_pts, latent_index)
        net = features
        for i, l in enumerate(self.bw_linears):  # noqa: E741
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw

    def pose_points_to_tpose_points(self, pose_pts, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        init_pbw = pts_sample_blend_weights(pose_pts, batch["pbw"], batch["pbounds"])
        init_pbw = init_pbw[:, :24]

        # neural blend weights of points at i
        if cfg.test_novel_pose:
            pbw = self.novel_pose_bw(pose_pts, init_pbw, batch["bw_latent_index"])
        else:
            pbw = self.calculate_neural_blend_weights(pose_pts, init_pbw, batch["latent_index"] + 1)

        # transform points from i to i_0
        tpose = pose_points_to_tpose_points(pose_pts, pbw, batch["A"])

        return tpose, pbw

    def forward(self, wpts, viewdir, dists, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch["R"], batch["Th"])

        with torch.no_grad():
            init_pbw = pts_sample_blend_weights(pose_pts, batch["pbw"], batch["pbounds"])
            pnorm = init_pbw[:, -1]
            norm_th = cfg.norm_th
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]
            viewdir = viewdir[pind[0]]
            dists = dists[pind[0]]

        # transform points from the pose space to the tpose space
        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)

        # calculate neural blend weights of points at the tpose space
        init_tbw = pts_sample_blend_weights(tpose, batch["tbw"], batch["tbounds"])
        init_tbw = init_tbw[:, :24]
        ind = torch.zeros_like(batch["latent_index"])
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)

        viewdir = viewdir[None]
        ind = batch["latent_index"]
        alpha, rgb = self.tpose_human.calculate_alpha_rgb(tpose, viewdir, ind)

        inside = tpose > batch["tbounds"][:, :1]
        inside = inside * (tpose < batch["tbounds"][:, 1:])
        outside = torch.sum(inside, dim=2) != 3
        alpha = alpha[:, 0]
        alpha[outside] = 0

        alpha_ind = alpha.detach() > cfg.train_th
        max_ind = torch.argmax(alpha, dim=1)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
        pbw = pbw.transpose(1, 2)[alpha_ind][None]
        tbw = tbw.transpose(1, 2)[alpha_ind][None]

        raw2alpha = lambda raw, dists, act_fn=F.relu: (
            1.0
            - torch.exp(
                -act_fn(  # noqa: E731
                    raw
                )
                * dists
            )
        )

        rgb = torch.sigmoid(rgb[0])
        alpha = raw2alpha(alpha[0], dists)

        raw = torch.cat((rgb, alpha[None]), dim=0)
        raw = raw.transpose(0, 1)

        n_batch, n_point = wpts.shape[:2]
        raw_full = torch.zeros([n_batch, n_point, 4], dtype=wpts.dtype, device=wpts.device)
        raw_full[pind] = raw

        ret = {"pbw": pbw, "tbw": tbw, "raw": raw_full}

        return ret


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.nf_latent = nn.Embedding(cfg.num_train_frame, 128)

        self.actvn = nn.ReLU()

        input_ch = 63
        D = 8
        W = 256
        self.skips = [4]
        self.pts_linears = nn.ModuleList(
            [nn.Conv1d(input_ch, W, 1)]
            + [
                nn.Conv1d(W, W, 1) if i not in self.skips else nn.Conv1d(W + input_ch, W, 1)
                for i in range(D - 1)
            ]
        )
        self.alpha_fc = nn.Conv1d(W, 1, 1)

        self.feature_fc = nn.Conv1d(W, W, 1)
        self.latent_fc = nn.Conv1d(384, W, 1)
        self.view_fc = nn.Conv1d(283, W // 2, 1)
        self.rgb_fc = nn.Conv1d(W // 2, 3, 1)

    def calculate_alpha(self, nf_pts):
        nf_pts = embedder.xyz_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts
        for i, l in enumerate(self.pts_linears):  # noqa: E741
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net), dim=1)
        alpha = self.alpha_fc(net)
        return alpha

    def calculate_alpha_rgb(self, nf_pts, viewdir, ind):
        nf_pts = embedder.xyz_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts
        for i, l in enumerate(self.pts_linears):  # noqa: E741
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net), dim=1)
        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        latent = self.nf_latent(ind)
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir.transpose(1, 2)
        features = torch.cat((features, viewdir), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        return alpha, rgb


class BackwardBlendWeight(nn.Module):
    def __init__(self):
        super(BackwardBlendWeight, self).__init__()

        self.bw_latent = nn.Embedding(cfg.num_eval_frame, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList(
            [nn.Conv1d(input_ch, W, 1)]
            + [
                nn.Conv1d(W, W, 1) if i not in self.skips else nn.Conv1d(W + input_ch, W, 1)
                for i in range(D - 1)
            ]
        )
        self.bw_fc = nn.Conv1d(W, 24, 1)

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def forward(self, ppts, smpl_bw, latent_index):
        latents = self.bw_latent
        features = self.get_point_feature(ppts, latent_index, latents)
        net = features
        for i, l in enumerate(self.bw_linears):  # noqa: E741
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw


# --------------------------------------------------------------------------
# Staging harness: real forward-pass call needs a `batch` dict of SMPL
# blend-weight-field tensors (normally produced by the dataset pipeline from
# a fitted SMPL mesh). Random synthetic tensors of the documented shapes
# (see the docstrings above, all copied from the original source) exercise
# the exact same architecture and control flow as real data would.
# --------------------------------------------------------------------------
def build_animatable_nerf():
    torch.manual_seed(0)
    return Network()


def example_input_animatable_nerf():
    torch.manual_seed(0)
    n_point = 64
    grid = 8  # small blend-weight voxel grid (real runs use larger grids)

    wpts = torch.rand(n_point, 3)
    viewdir = F.normalize(torch.randn(n_point, 3), dim=-1)
    dists = torch.rand(n_point) * 0.01 + 1e-3

    batch = {
        "R": torch.eye(3).unsqueeze(0),
        "Th": torch.zeros(1, 1, 3),
        "pbw": torch.rand(1, grid, grid, grid, 25),
        "pbounds": torch.stack([torch.full((3,), -1.0), torch.full((3,), 1.0)], dim=0).unsqueeze(0),
        "tbw": torch.rand(1, grid, grid, grid, 25),
        "tbounds": torch.stack([torch.full((3,), -1.0), torch.full((3,), 1.0)], dim=0).unsqueeze(0),
        "A": torch.eye(4).view(1, 1, 4, 4).expand(1, 24, 4, 4).contiguous(),
        "latent_index": torch.zeros(1, dtype=torch.long),
    }

    return (wpts, viewdir, dists, batch)


MENAGERIE_ENTRIES = [
    (
        "Animatable NeRF (T-pose deform network)",
        "build_animatable_nerf",
        "example_input_animatable_nerf",
        2021,
        MENAGERIE_ZOO,
    ),
]
