# SOURCE: vendored from sxyu/plenoctree @ master (octree/nerf/models.py,
# octree/nerf/model_utils.py). https://github.com/sxyu/plenoctree -- "PlenOctrees for
# Real-time Rendering of Neural Radiance Fields" (Yu, Li, Tancik, Li, Ng, Kanazawa,
# ICCV 2021). This is the repo's PyTorch NeRF-SH evaluation model used for octree
# extraction (distinct from the JAX/Flax `nerf_sh/` training path in the same repo):
# a coarse+fine positional-encoding MLP pair (`model_utils.MLP`) whose final layer
# emits spherical-harmonics RGB coefficients instead of raw RGB, wrapped by
# `NerfModel.eval_points_raw`. `dense_layer`/`MLP`/`posenc`/`NerfModel` are transcribed
# verbatim; only the `get_model`/`construct_nerf`/checkpoint-restore helpers (which
# depend on a training-time `args` FLAGS object) were dropped since they are I/O
# plumbing, not part of the model architecture.
import math
from typing import Any, Callable

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from octree/nerf/model_utils.py ----
def dense_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    # The initialization matters!
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self,
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        net_activation: Callable[..., Any] = nn.ReLU(),  # The activation function.
        skip_layer: int = 4,  # The layer to add skip layers to.
        num_rgb_channels: int = 3,  # The number of RGB channels.
        num_sigma_channels: int = 1,  # The number of sigma channels.
        input_dim: int = 63,  # The number of input tensor channels.
        condition_dim: int = 27,  # The number of conditional tensor channels.
    ):
        super(MLP, self).__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.net_depth_condition = net_depth_condition
        self.net_width_condition = net_width_condition
        self.net_activation = net_activation
        self.skip_layer = skip_layer
        self.num_rgb_channels = num_rgb_channels
        self.num_sigma_channels = num_sigma_channels
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        self.input_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.input_layers.append(dense_layer(in_features, self.net_width))
            if i % self.skip_layer == 0 and i > 0:
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        self.sigma_layer = dense_layer(in_features, self.num_sigma_channels)

        if self.condition_dim > 0:
            self.bottleneck_layer = dense_layer(in_features, self.net_width)
            self.condition_layers = nn.ModuleList()
            in_features = self.net_width + self.condition_dim
            for i in range(self.net_depth_condition):
                self.condition_layers.append(dense_layer(in_features, self.net_width_condition))
                in_features = self.net_width_condition
        self.rgb_layer = dense_layer(in_features, self.num_rgb_channels)

    def forward(self, x, condition=None, cross_broadcast=False):
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.view([-1, feature_dim])
        inputs = x
        for i in range(self.net_depth):
            x = self.input_layers[i](x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_sigma = self.sigma_layer(x).view([-1, num_samples, self.num_sigma_channels])

        if condition is not None:
            bottleneck = self.bottleneck_layer(x)
            if len(condition.shape) == 2 and (not cross_broadcast):
                condition = condition[:, None, :].repeat(1, num_samples, 1)
            if cross_broadcast:
                condition = condition.view([batch_size, -1, condition.shape[-1]])
                num_rays = condition.shape[1]
                condition = condition[:, None, :, :].repeat(1, num_samples, 1, 1)
                bottleneck = bottleneck.view([batch_size, -1, bottleneck.shape[-1]])
                bottleneck = bottleneck[:, :, None, :].repeat(1, 1, num_rays, 1)
            x = torch.cat(
                [
                    bottleneck.view([-1, bottleneck.shape[-1]]),
                    condition.view([-1, condition.shape[-1]]),
                ],
                dim=-1,
            )
            for i in range(self.net_depth_condition):
                x = self.condition_layers[i](x)
                x = self.net_activation(x)
        raw_rgb = self.rgb_layer(x).view(
            [batch_size, num_samples, self.num_rgb_channels]
            if not cross_broadcast
            else [batch_size, num_samples, num_rays, self.num_rgb_channels]
        )
        return raw_rgb, raw_sigma


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1]."""
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device)
    if legacy_posenc_order:
        xb = x[..., None, :] * scales[:, None]
        four_feat = torch.reshape(
            torch.sin(torch.stack([xb, xb + 0.5 * math.pi], -2)), list(x.shape[:-1]) + [-1]
        )
    else:
        xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


# ---- verbatim from octree/nerf/models.py ----
class NerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(
        self,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        use_viewdirs: bool = True,
        sh_deg: int = -1,
        sg_dim: int = -1,
        near: float = 2.0,
        far: float = 6.0,
        noise_std: float = 0.0,
        net_depth: int = 8,
        net_width: int = 256,
        net_depth_condition: int = 1,
        net_width_condition: int = 128,
        net_activation: Callable[..., Any] = nn.ReLU(),
        skip_layer: int = 4,
        num_rgb_channels: int = 3,
        num_sigma_channels: int = 1,
        white_bkgd: bool = True,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        lindisp: bool = False,
        rgb_activation: Callable[..., Any] = nn.Sigmoid(),
        sigma_activation: Callable[..., Any] = nn.ReLU(),
        legacy_posenc_order: bool = False,
    ):
        super(NerfModel, self).__init__()
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.use_viewdirs = use_viewdirs
        self.sh_deg = sh_deg
        self.sg_dim = sg_dim
        self.near = near
        self.far = far
        self.noise_std = noise_std
        self.net_depth = net_depth
        self.net_width = net_width
        self.net_depth_condition = net_depth_condition
        self.net_width_condition = net_width_condition
        self.net_activation = net_activation
        self.skip_layer = skip_layer
        self.num_rgb_channels = num_rgb_channels
        self.num_sigma_channels = num_sigma_channels
        self.white_bkgd = white_bkgd
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.deg_view = deg_view
        self.lindisp = lindisp
        self.rgb_activation = rgb_activation
        self.sigma_activation = sigma_activation
        self.legacy_posenc_order = legacy_posenc_order
        # Construct the "coarse" MLP. Weird name is for
        # compatibility with 'compact' version
        self.MLP_0 = MLP(
            net_depth=self.net_depth,
            net_width=self.net_width,
            net_depth_condition=self.net_depth_condition,
            net_width_condition=self.net_width_condition,
            net_activation=self.net_activation,
            skip_layer=self.skip_layer,
            num_rgb_channels=self.num_rgb_channels,
            num_sigma_channels=self.num_sigma_channels,
            input_dim=3 * (1 + 2 * (self.max_deg_point - self.min_deg_point)),
            condition_dim=3 * (1 + 2 * self.deg_view) if self.use_viewdirs else 0,
        )
        # Construct the "fine" MLP.
        self.MLP_1 = MLP(
            net_depth=self.net_depth,
            net_width=self.net_width,
            net_depth_condition=self.net_depth_condition,
            net_width_condition=self.net_width_condition,
            net_activation=self.net_activation,
            skip_layer=self.skip_layer,
            num_rgb_channels=self.num_rgb_channels,
            num_sigma_channels=self.num_sigma_channels,
            input_dim=3 * (1 + 2 * (self.max_deg_point - self.min_deg_point)),
            condition_dim=3 * (1 + 2 * self.deg_view) if self.use_viewdirs else 0,
        )

        if self.sg_dim > 0:
            self.register_parameter("sg_lambda", nn.Parameter(torch.ones([self.sg_dim])))
            self.register_parameter(
                "sg_mu_spher",
                nn.Parameter(
                    torch.stack(
                        [
                            torch.rand([self.sg_dim]) * math.pi,
                            torch.rand([self.sg_dim]) * math.pi * 2,
                        ],
                        dim=-1,
                    )
                ),
            )

    def eval_points_raw(self, points, viewdirs=None, coarse=False, cross_broadcast=False):
        """
        Evaluate at points, returning rgb and sigma.

        Args:
          points: torch.tensor [B, 3]
          viewdirs: torch.tensor [B, 3]. if cross_broadcast = True, it can be [M, 3].
          coarse: if true, uses coarse MLP.
          cross_broadcast: if true, cross broadcast between points and viewdirs.
        """
        points = points[None]
        points_enc = posenc(
            points,
            self.min_deg_point,
            self.max_deg_point,
            self.legacy_posenc_order,
        )
        if self.num_fine_samples > 0 and not coarse:
            mlp = self.MLP_1
        else:
            mlp = self.MLP_0
        if self.use_viewdirs:
            assert viewdirs is not None
            viewdirs = viewdirs[None]
            viewdirs_enc = posenc(
                viewdirs,
                0,
                self.deg_view,
                self.legacy_posenc_order,
            )
            raw_rgb, raw_sigma = mlp(points_enc, viewdirs_enc, cross_broadcast=cross_broadcast)
        else:
            raw_rgb, raw_sigma = mlp(points_enc)
        return raw_rgb[0], raw_sigma[0]

    def forward(self, points, viewdirs):
        """Menagerie forward shim: exposes eval_points_raw (the model's real
        callable path -- the original repo has no bare `forward`/`__call__`
        usage) as a plain forward pass for a standard TorchLens capture."""
        return self.eval_points_raw(points, viewdirs, coarse=False, cross_broadcast=False)


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_plenoctree_nerfsh():
    torch.manual_seed(0)
    model = NerfModel(
        net_depth=2,
        net_width=16,
        net_depth_condition=1,
        net_width_condition=8,
        min_deg_point=0,
        max_deg_point=4,
        deg_view=2,
        sh_deg=2,
        use_viewdirs=True,
        num_fine_samples=1,  # forces forward() to use the fine MLP (MLP_1)
    )
    model.eval()
    return model


def example_input_plenoctree_nerfsh():
    torch.manual_seed(0)
    n_points = 8
    points = torch.randn(n_points, 3)
    viewdirs = torch.randn(n_points, 3)
    return (points, viewdirs)


MENAGERIE_ENTRIES = [
    (
        "PlenOctree-NeRF-SH",
        build_plenoctree_nerfsh,
        example_input_plenoctree_nerfsh,
        2021,
        "vendored-pytorch",
    ),
]
