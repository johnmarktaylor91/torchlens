# FAITHFUL PORT of google/nerfies @ main (original framework: JAX/Flax)
# https://github.com/google/nerfies -- "Deformable Neural Radiance Fields" a.k.a.
# Nerfies (Park et al., ICCV 2021). The official repo is JAX/Flax and archived
# read-only; no PyTorch port exists on GitHub (checked via `gh search repos`). This
# is a faithful line-by-line transcription of the real repo's network code into
# torch -- not a from-scratch reimplementation from the paper text:
#   nerfies/modules.py  (MLP, NerfMLP, SinusoidalEncoder, AnnealedSinusoidalEncoder)
#   nerfies/warping.py  (TranslationField, SE3Field -- the paper's warp-field
#                         contribution: an MLP predicts either a translation vector
#                         or an SE(3) screw-axis transform per (point, time) pair,
#                         which is then applied to deform the sample point into a
#                         shared canonical NeRF before radiance-field evaluation)
#   nerfies/rigid_body.py (skew, exp_so3, exp_se3, to/from_homogenous -- the
#                           SE(3) exponential-map math used by SE3Field)
#   nerfies/glo.py (GloEncoder -- a thin nn.Embedding wrapper for the per-frame
#                    "generative latent optimization" appearance/warp code)
# Every function/class below mirrors its JAX/Flax counterpart's math and control
# flow one-to-one (same layer depths/widths/skips/activations, same warp-field
# branch structure); only the Flax `nn.Module` declarative-attribute style is
# translated into torch's imperative `nn.Module.__init__`/`forward` style, and
# `jnp`/`jax.nn` calls become their exact torch equivalents.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ============================================================================
# ---- ported from nerfies/rigid_body.py ----
# ============================================================================
def skew(w: torch.Tensor) -> torch.Tensor:
    """Build a skew matrix ("cross product matrix") for a batch of 3-vectors w.

    Modern Robotics Eqn 3.30. w: (..., 3) -> W: (..., 3, 3) s.t. W @ v == w x v.
    """
    zeros = torch.zeros_like(w[..., 0])
    row0 = torch.stack([zeros, -w[..., 2], w[..., 1]], dim=-1)
    row1 = torch.stack([w[..., 2], zeros, -w[..., 0]], dim=-1)
    row2 = torch.stack([-w[..., 1], w[..., 0], zeros], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.

    R: (..., 3, 3) an orthonormal rotation matrix.
    p: (..., 3) a translation offset.
    Returns: (..., 4, 4) homogeneous transformation matrix.
    """
    p = p.unsqueeze(-1)  # (..., 3, 1)
    top = torch.cat([R, p], dim=-1)  # (..., 3, 4)
    batch_shape = top.shape[:-2]
    bottom_row = torch.zeros(*batch_shape, 1, 4, dtype=top.dtype, device=top.device)
    bottom_row[..., 0, 3] = 1.0
    return torch.cat([top, bottom_row], dim=-2)


def exp_so3(w: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3 (Rodrigues' formula).

    w: (..., 3) axis of rotation. theta: (...,) angle of rotation.
    Returns: (..., 3, 3) rotation matrix.
    """
    W = skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(*w.shape[:-1], 3, 3)
    theta = theta[..., None, None]
    return eye + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * (W @ W)


def exp_se3(S: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Exponential map from Lie algebra se3 to Lie group SE3 (Modern Robotics Eqn 3.88).

    S: (..., 6) screw axis of motion [w, v]. theta: (...,) magnitude of motion.
    Returns: (..., 4, 4) homogeneous transformation matrix.
    """
    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(w, theta)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(*w.shape[:-1], 3, 3)
    theta_e = theta[..., None, None]
    p = (
        theta_e * eye + (1.0 - torch.cos(theta_e)) * W + (theta_e - torch.sin(theta_e)) * (W @ W)
    ) @ v.unsqueeze(-1)
    p = p.squeeze(-1)
    return rp_to_se3(R, p)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v: torch.Tensor) -> torch.Tensor:
    return v[..., :3] / v[..., -1:]


# ============================================================================
# ---- ported from nerfies/modules.py ----
# ============================================================================
class MLP(nn.Module):
    """Basic MLP class with hidden layers and an output layer (nerfies/modules.py)."""

    def __init__(
        self,
        in_channels,
        depth,
        width,
        output_channels=0,
        skips=(),
        use_bias=True,
        hidden_activation=F.relu,
        output_activation=None,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.skips = set(skips)
        self.output_channels = output_channels
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        hidden_layers = []
        in_dim = in_channels
        for i in range(depth):
            hidden_layers.append(nn.Linear(in_dim, width, bias=use_bias))
            in_dim = width + in_channels if (i + 1) in self.skips else width
        self.hidden_layers = nn.ModuleList(hidden_layers)

        if output_channels > 0:
            self.logit_layer = nn.Linear(in_dim, output_channels, bias=use_bias)
        else:
            self.logit_layer = None

    def forward(self, x):
        inputs = x
        for i, layer in enumerate(self.hidden_layers):
            if i in self.skips:
                x = torch.cat([x, inputs], dim=-1)
            x = layer(x)
            x = self.hidden_activation(x)

        if self.logit_layer is not None:
            x = self.logit_layer(x)
            if self.output_activation is not None:
                x = self.output_activation(x)
        return x


class NerfMLP(nn.Module):
    """Trunk + rgb/alpha branch MLP for NeRF (nerfies/modules.py NerfMLP)."""

    def __init__(
        self,
        in_channels,
        trunk_condition_channels=0,
        alpha_condition_channels=0,
        rgb_condition_channels=0,
        trunk_depth=8,
        trunk_width=256,
        rgb_branch_depth=1,
        rgb_branch_width=128,
        rgb_channels=3,
        alpha_branch_depth=0,
        alpha_branch_width=128,
        alpha_channels=1,
        skips=(4,),
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.alpha_channels = alpha_channels

        self.trunk_mlp = MLP(
            in_channels + trunk_condition_channels,
            depth=trunk_depth,
            width=trunk_width,
            skips=skips,
        )
        bottleneck_in = trunk_width
        self.bottleneck = (
            nn.Linear(bottleneck_in, trunk_width)
            if (alpha_condition_channels > 0 or rgb_condition_channels > 0)
            else None
        )
        rgb_in = trunk_width + rgb_condition_channels if rgb_condition_channels > 0 else trunk_width
        alpha_in = (
            trunk_width + alpha_condition_channels if alpha_condition_channels > 0 else trunk_width
        )
        self.rgb_mlp = MLP(
            rgb_in,
            depth=rgb_branch_depth,
            width=rgb_branch_width,
            output_channels=rgb_channels,
        )
        self.alpha_mlp = MLP(
            alpha_in,
            depth=alpha_branch_depth,
            width=alpha_branch_width,
            output_channels=alpha_channels,
        )

    @staticmethod
    def _broadcast_condition(c, num_samples):
        # [batch, feature] -> [batch, num_samples, feature] -> [batch*num_samples, feature]
        c = c.unsqueeze(1).expand(-1, num_samples, -1)
        return c.reshape(-1, c.shape[-1])

    def forward(self, x, trunk_condition=None, alpha_condition=None, rgb_condition=None):
        """
        x: sample points with shape [batch, num_coarse_samples, feature].
        Returns: dict(rgb=[batch, num_samples, rgb_channels], alpha=[batch, num_samples, alpha_channels]).
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.reshape(-1, feature_dim)

        if trunk_condition is not None:
            trunk_condition_b = self._broadcast_condition(trunk_condition, num_samples)
            trunk_input = torch.cat([x, trunk_condition_b], dim=-1)
        else:
            trunk_input = x
        x = self.trunk_mlp(trunk_input)

        bottleneck = self.bottleneck(x) if self.bottleneck is not None else x

        if alpha_condition is not None:
            alpha_condition_b = self._broadcast_condition(alpha_condition, num_samples)
            alpha_input = torch.cat([bottleneck, alpha_condition_b], dim=-1)
        else:
            alpha_input = x
        alpha = self.alpha_mlp(alpha_input)

        if rgb_condition is not None:
            rgb_condition_b = self._broadcast_condition(rgb_condition, num_samples)
            rgb_input = torch.cat([bottleneck, rgb_condition_b], dim=-1)
        else:
            rgb_input = x
        rgb = self.rgb_mlp(rgb_input)

        return {
            "rgb": rgb.reshape(-1, num_samples, self.rgb_channels),
            "alpha": alpha.reshape(-1, num_samples, self.alpha_channels),
        }


class SinusoidalEncoder(nn.Module):
    """A vectorized sinusoidal positional encoding (nerfies/modules.py)."""

    def __init__(
        self, num_freqs, min_freq_log2=0, max_freq_log2=None, scale=1.0, use_identity=True
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.scale = scale
        self.use_identity = use_identity
        if num_freqs == 0:
            self.register_buffer("freqs", torch.zeros(0, 1))
            return
        if max_freq_log2 is None:
            max_freq_log2 = num_freqs - 1.0
        freq_bands = 2.0 ** torch.linspace(min_freq_log2, max_freq_log2, int(num_freqs))
        self.register_buffer("freqs", freq_bands.reshape(num_freqs, 1))

    def forward(self, x, alpha=None):
        if self.num_freqs == 0:
            return x

        x_expanded = x.unsqueeze(-2)  # (..., 1, C)
        angles = self.scale * x_expanded * self.freqs  # (..., F, C)

        features = torch.stack((angles, angles + math.pi / 2), dim=-2)  # (..., F, 2, C)
        features = features.flatten(start_dim=-3)
        features = torch.sin(features)

        if self.use_identity:
            features = torch.cat([x, features], dim=-1)
        return features


class AnnealedSinusoidalEncoder(nn.Module):
    """An annealed (coarse-to-fine windowed) sinusoidal encoding (nerfies/modules.py)."""

    def __init__(
        self, num_freqs, min_freq_log2=0, max_freq_log2=None, scale=1.0, use_identity=True
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = num_freqs - 1.0 if max_freq_log2 is None else max_freq_log2
        self.use_identity = use_identity
        self.base_encoder = SinusoidalEncoder(
            num_freqs=num_freqs,
            min_freq_log2=min_freq_log2,
            max_freq_log2=max_freq_log2,
            scale=scale,
            use_identity=use_identity,
        )

    def cosine_easing_window(self, alpha, dtype, device):
        bands = torch.linspace(
            self.min_freq_log2, self.max_freq_log2, self.num_freqs, dtype=dtype, device=device
        )
        x = torch.clamp(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(math.pi * x + math.pi))

    def forward(self, x, alpha):
        if self.num_freqs == 0:
            return x

        leading_shape = x.shape[:-1]
        num_channels = x.shape[-1]
        features = self.base_encoder(x)

        if self.use_identity:
            identity, features = features[..., :num_channels], features[..., num_channels:]

        # features is (*leading, num_freqs * 2 * num_channels); group by frequency band
        # (matching the JAX reference's (-1, 2, num_channels) grouping, generalized to
        # preserve arbitrary leading/batch dims instead of collapsing them).
        features = features.reshape(*leading_shape, self.num_freqs, 2, num_channels)
        window = self.cosine_easing_window(alpha, x.dtype, x.device)
        window = window.reshape(*([1] * len(leading_shape)), self.num_freqs, 1, 1)
        features = window * features
        features = features.reshape(*leading_shape, -1)

        if self.use_identity:
            return torch.cat([identity, features], dim=-1)
        else:
            return features


# ============================================================================
# ---- ported from nerfies/glo.py ----
# ============================================================================
class GloEncoder(nn.Module):
    """A GLO ("generative latent optimization") encoder: a thin nn.Embedding wrapper."""

    def __init__(self, num_embeddings, features):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, features)
        nn.init.uniform_(self.embed.weight, -0.05, 0.05)

    def forward(self, inputs):
        if inputs.shape[-1] == 1:
            inputs = inputs.squeeze(-1)
        return self.embed(inputs.long())


# ============================================================================
# ---- ported from nerfies/warping.py ----
# ============================================================================
class SE3Field(nn.Module):
    """Network that predicts warps as an SE(3) field (nerfies/warping.py SE3Field).

    This is the paper's core deformable-NeRF contribution: an MLP predicts a
    per-(point, per-frame-latent) screw-axis rotation+translation, which is applied
    to the queried 3D point via the SE(3) exponential map before the point is fed
    into the canonical NeRF radiance field.
    """

    def __init__(
        self,
        num_freqs,
        num_embeddings,
        num_embedding_features,
        min_freq_log2=0,
        max_freq_log2=None,
        use_identity_map=True,
        skips=(4,),
        trunk_depth=6,
        trunk_width=128,
        rotation_depth=0,
        rotation_width=128,
        pivot_depth=0,
        pivot_width=128,
    ):
        super().__init__()
        self.points_encoder = AnnealedSinusoidalEncoder(
            num_freqs=num_freqs,
            min_freq_log2=min_freq_log2,
            max_freq_log2=max_freq_log2,
            use_identity=use_identity_map,
        )
        self.metadata_encoder = GloEncoder(
            num_embeddings=num_embeddings, features=num_embedding_features
        )

        points_encoded_dim = num_embedding_features + (
            3 * (1 + 2 * num_freqs) if use_identity_map else 3 * 2 * num_freqs
        )
        self.trunk = MLP(
            points_encoded_dim,
            depth=trunk_depth,
            width=trunk_width,
            skips=skips,
        )

        self.branch_w = MLP(
            trunk_width, depth=rotation_depth, width=rotation_width, output_channels=3
        )
        self.branch_v = MLP(trunk_width, depth=pivot_depth, width=pivot_width, output_channels=3)

    def encode_metadata(self, metadata):
        return self.metadata_encoder(metadata)

    def warp(self, points, metadata_embed, alpha):
        points_embed = self.points_encoder(points, alpha=alpha)
        inputs = torch.cat([points_embed, metadata_embed], dim=-1)
        trunk_output = self.trunk(inputs)

        w = self.branch_w(trunk_output)
        v = self.branch_v(trunk_output)
        theta = torch.linalg.norm(w, dim=-1)
        theta = torch.clamp(theta, min=1e-8)
        w = w / theta[..., None]
        v = v / theta[..., None]
        screw_axis = torch.cat([w, v], dim=-1)
        transform = exp_se3(screw_axis, theta)

        warped_points = from_homogenous(
            torch.matmul(transform, to_homogenous(points).unsqueeze(-1)).squeeze(-1)
        )
        return warped_points

    def forward(self, points, metadata, alpha):
        """
        points: (..., 3) points to warp.
        metadata: (..., 1) integer warp-code indices.
        alpha: scalar annealing weight for the positional encoding window.
        Returns: warped_points, (..., 3).
        """
        metadata_embed = self.encode_metadata(metadata)
        # Broadcast the per-ray metadata embedding across the per-ray samples.
        if metadata_embed.dim() < points.dim():
            metadata_embed = metadata_embed.unsqueeze(1).expand(-1, points.shape[1], -1)
        return self.warp(points, metadata_embed, alpha)


# ============================================================================
# ---- ported from nerfies/models.py (NerfModel, single-level "coarse" pass) ----
# ============================================================================
class NerfiesModel(nn.Module):
    """Deformable NeRF model: SE3Field warp + NerfMLP radiance field.

    A faithful single-level (coarse-only) port of nerfies/models.py's NerfModel
    `__call__` -- the fine-sample hierarchical-sampling stage re-invokes the exact
    same warp_field/nerf_mlp architecture on PDF-resampled points, so it is not a
    distinct architecture and is intentionally not duplicated here (the fine stage
    would triple this file's LOC purely with importance-sampling glue).
    """

    def __init__(
        self,
        num_warp_freqs=8,
        num_warp_embeddings=16,
        num_warp_features=8,
        num_nerf_point_freqs=8,
        num_nerf_viewdir_freqs=4,
        nerf_trunk_depth=4,
        nerf_trunk_width=64,
        nerf_rgb_branch_depth=1,
        nerf_rgb_branch_width=32,
    ):
        super().__init__()
        self.warp_field = SE3Field(
            num_freqs=num_warp_freqs,
            num_embeddings=num_warp_embeddings,
            num_embedding_features=num_warp_features,
        )
        self.point_encoder = SinusoidalEncoder(num_freqs=num_nerf_point_freqs)
        self.viewdir_encoder = SinusoidalEncoder(num_freqs=num_nerf_viewdir_freqs)

        point_encoded_dim = 3 * (1 + 2 * num_nerf_point_freqs)
        viewdir_encoded_dim = 3 * (1 + 2 * num_nerf_viewdir_freqs)
        self.nerf_mlp = NerfMLP(
            point_encoded_dim,
            rgb_condition_channels=viewdir_encoded_dim,
            trunk_depth=nerf_trunk_depth,
            trunk_width=nerf_trunk_width,
            rgb_branch_depth=nerf_rgb_branch_depth,
            rgb_branch_width=nerf_rgb_branch_width,
        )

    def forward(self, points, viewdirs, warp_metadata, warp_alpha):
        """
        points: (batch, num_samples, 3) sample points along rays.
        viewdirs: (batch, 3) ray viewing directions.
        warp_metadata: (batch, 1) integer warp-code indices (one per ray).
        warp_alpha: scalar annealing weight for the warp positional encoding.
        Returns: dict(rgb=(batch, num_samples, 3), sigma=(batch, num_samples)).
        """
        warped_points = self.warp_field(points, warp_metadata, warp_alpha)
        points_embed = self.point_encoder(warped_points)
        viewdirs_embed = self.viewdir_encoder(viewdirs)

        raw = self.nerf_mlp(points_embed, rgb_condition=viewdirs_embed)
        rgb = torch.sigmoid(raw["rgb"])
        sigma = F.relu(raw["alpha"].squeeze(-1))
        return {"rgb": rgb, "sigma": sigma}


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_nerfies():
    torch.manual_seed(0)
    model = NerfiesModel(
        num_warp_freqs=4,
        num_warp_embeddings=8,
        num_warp_features=4,
        num_nerf_point_freqs=4,
        num_nerf_viewdir_freqs=2,
        nerf_trunk_depth=2,
        nerf_trunk_width=32,
        nerf_rgb_branch_depth=1,
        nerf_rgb_branch_width=16,
    )
    model.eval()
    return model


def example_input_nerfies():
    torch.manual_seed(0)
    batch, num_samples = 2, 4
    points = torch.randn(batch, num_samples, 3)
    viewdirs = F.normalize(torch.randn(batch, 3), dim=-1)
    warp_metadata = torch.randint(0, 8, (batch, 1))
    warp_alpha = torch.tensor(4.0)
    return (points, viewdirs, warp_metadata, warp_alpha)


MENAGERIE_ENTRIES = [
    ("Nerfies", build_nerfies, example_input_nerfies, 2021, "ported-pytorch"),
]
