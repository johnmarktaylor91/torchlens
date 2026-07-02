# SOURCE: vendored from TheCamusean/grasp_diffusion @ master
# https://github.com/TheCamusean/grasp_diffusion/blob/master/se3dif/models/grasp_dif.py
# https://github.com/TheCamusean/grasp_diffusion/blob/master/se3dif/models/vision_encoder/latent_codes.py
# https://github.com/TheCamusean/grasp_diffusion/blob/master/se3dif/models/geometry_encoder/maps.py
# https://github.com/TheCamusean/grasp_diffusion/blob/master/se3dif/models/nets/feature_net.py
# https://github.com/TheCamusean/grasp_diffusion/blob/master/se3dif/models/loader.py
# SE(3)-DiffusionFields (Urain, Funk, Peters, Chalvatzaki, ICRA 2023):
# `GraspDiffusionFields` (real class, verbatim) is the SE(3) diffusion model
# scoring 6-DoF grasp poses H via an energy field. Assembled here exactly as
# `load_grasp_diffusion()` in `se3dif/models/loader.py` wires it up: a
# `LatentCodes` embedding vision encoder (the simplest of the repo's two
# vision-encoder options -- the alternative, `VNNPointnet2`, needs a point
# cloud + vector-neuron pointnet backbone, an orthogonal add-on the
# `LatentCodes` scene-embedding path does not require), the real
# `map_projected_points` geometry encoder function, the real
# `TimeLatentFeatureEncoder` (time+latent conditioned SDF-style feature MLP,
# DeepSDF-derived), and the real energy-net decoder MLP exactly as
# constructed in `load_grasp_diffusion`. All classes/functions transcribed
# verbatim; only change is generating the fixed query points programmatically
# (torch.manual_seed + rand) instead of loading the repo's shipped
# `UniformPts.npy` binary, since the *value* of the precomputed query points
# is a training-time detail (any deterministic (n_points,3) tensor exercises
# the identical architecture/graph shape).
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ---- se3dif/models/vision_encoder/latent_codes.py (verbatim) ----
class LatentCodes(nn.Module):
    def __init__(self, num_scenes, latent_size, code_bound=1.0, std=1.0):
        super(LatentCodes, self).__init__()

        self.lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
        torch.nn.init.normal_(
            self.lat_vecs.weight.data,
            0.0,
            std / math.sqrt(latent_size),
        )

    def forward(self, idxs):
        lat_vecs = self.lat_vecs(idxs.int())
        return lat_vecs


# ---- se3dif/models/geometry_encoder/maps.py (verbatim) ----
def map_projected_points(H, p):
    p_ext = torch.cat((p, torch.ones_like(p[..., :1])), -1)
    p_alig = torch.einsum("...md,pd->...pm", H, p_ext)[..., :-1]
    return p_alig


# ---- se3dif/models/nets/feature_net.py (verbatim) ----
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = torch.einsum("...,b->...b", x, self.W) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class TimeLatentFeatureEncoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        in_dim=3,
        enc_dim=256,
        out_dim=1,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        feats_layers=None,
    ):
        super(TimeLatentFeatureEncoder, self).__init__()

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=enc_dim),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
        )
        self.x_embed = nn.Sequential(
            nn.Linear(in_dim, enc_dim),
            nn.SiLU(),
        )

        self.out_dim = out_dim
        self.latent_size = latent_size
        self.in_dim = in_dim

        dims = [latent_size + enc_dim + in_dim] + dims + [out_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (not weight_norm) and self.norm_layers is not None and layer in self.norm_layers:
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

        if feats_layers is None:
            self.feats_layers = list(np.arange(0, self.num_layers - 1))
        else:
            self.feats_layers = feats_layers

    def forward(self, input, timesteps, latent_vecs=None):
        t_emb = self.time_embed(timesteps)
        x_emb = self.x_embed(input)
        xyz = x_emb + t_emb

        if latent_vecs is not None:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz, input], -1)
        else:
            x = torch.cat([xyz, input], -1)
        x0 = x.clone()

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, x0], -1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, input], -1)
            x = lin(x)
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


# ---- se3dif/models/grasp_dif.py (verbatim) ----
class GraspDiffusionFields(nn.Module):
    """Grasp DiffusionFields. SE(3) diffusion model to learn 6D grasp distributions. See
    SE(3)-DiffusionFields: Learning cost functions for joint grasp and motion optimization through diffusion
    """

    def __init__(self, vision_encoder, geometry_encoder, points, feature_encoder, decoder):
        super().__init__()
        self.register_buffer("points", points)
        self.vision_encoder = vision_encoder
        self.z = None
        self.geometry_encoder = geometry_encoder
        self.feature_encoder = feature_encoder
        self.decoder = decoder

    def set_latent(self, O, batch=1):  # noqa: E741
        self.z = self.vision_encoder(O.squeeze(1))
        self.z = self.z.unsqueeze(1).repeat(1, batch, 1).reshape(-1, self.z.shape[-1])

    def forward(self, H, k):
        ## 1. Represent H with points
        p = self.geometry_encoder(H, self.points)
        k_ext = k.unsqueeze(1).repeat(1, p.shape[1])
        z_ext = self.z.unsqueeze(1).repeat(1, p.shape[1], 1)
        ## 2. Get Features
        psi = self.feature_encoder(p, k_ext, z_ext)
        ## 3. Flat and get energy
        psi_flatten = psi.reshape(psi.shape[0], -1)
        e = self.decoder(psi_flatten)
        return e

    def compute_sdf(self, x):
        k = torch.rand_like(x[..., 0])
        psi = self.feature_encoder(x, k, self.z)
        return psi[..., 0]


class GraspDiffusionFieldsTraced(nn.Module):
    """Thin wrapper so TorchLens sees one top-level forward(O, H, k) call:
    `set_latent(O)` then `forward(H, k)`, exactly the real two-step usage in
    the repo's sampler (`se3dif/samplers/grasp_samplers.py`)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, O, H, k):  # noqa: E741
        self.model.set_latent(O, batch=H.shape[0])
        return self.model(H, k)


MENAGERIE_ZOO = "vendored-pytorch"

_N_POINTS = 32
_LATENT_SIZE = 16
_ENC_DIM = 32


def build_se3_diffusion_fields():
    torch.manual_seed(0)
    vision_encoder = LatentCodes(num_scenes=4, latent_size=_LATENT_SIZE)
    geometry_encoder = map_projected_points
    feature_encoder = TimeLatentFeatureEncoder(
        enc_dim=_ENC_DIM,
        latent_size=_LATENT_SIZE,
        dims=[64, 64],
        out_dim=2,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=False,
        weight_norm=False,
    )
    # Deterministic stand-in for the repo's shipped UniformPts.npy query points
    # (see module docstring): same role/shape, generated not loaded from disk.
    g = torch.Generator().manual_seed(0)
    points = torch.rand(_N_POINTS, 3, generator=g) * 2 - 1

    in_dim = _N_POINTS * 2
    hidden_dim = 32
    energy_net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

    model = GraspDiffusionFields(
        vision_encoder=vision_encoder,
        feature_encoder=feature_encoder,
        geometry_encoder=geometry_encoder,
        decoder=energy_net,
        points=points,
    )
    wrapped = GraspDiffusionFieldsTraced(model)
    wrapped.eval()
    return wrapped


def example_input_se3_diffusion_fields():
    torch.manual_seed(0)
    # set_latent() does `vision_encoder(O.squeeze(1))`, so O carries a
    # trailing size-1 axis it strips before the LatentCodes embedding lookup
    # (real usage: `set_latent(P[None, ...], ...)` in se3dif/models/loader.py).
    O = torch.zeros(1, 1, dtype=torch.long)  # noqa: E741 -- scene index into LatentCodes embedding
    batch = 5
    H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)  # batch of SE(3) poses
    k = torch.rand(batch)  # diffusion time steps
    return (O, H, k)


MENAGERIE_ENTRIES = [
    (
        "SE(3)-DiffusionFields",
        "build_se3_diffusion_fields",
        "example_input_se3_diffusion_fields",
        2023,
        MENAGERIE_ZOO,
    ),
]
