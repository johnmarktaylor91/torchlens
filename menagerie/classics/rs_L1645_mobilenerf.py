# FAITHFUL PORT of google-research/jax3d @ 5ee7fe79c1f4 (original framework: JAX/Flax)
# Ported from jax3d/projects/mobilenerf/stage1.py
# https://github.com/google-research/jax3d/blob/main/jax3d/projects/mobilenerf/stage1.py
"""MobileNeRF (Chen et al., CVPR 2023): "Exploiting the Polygon Rasterization
Pipeline for Efficient Neural Field Rendering on Mobile Architectures".

The official code is a monolithic JAX/Flax training script (jax3d is JAX-only
and is not one of this environment's installed base libraries, so it cannot be
run directly here); the actual network architecture it defines -- a skip-
connected sinusoidally-encoded MLP (`RadianceField`, reused for both the scalar
density head and the num_bottleneck_features appearance-feature head) plus a
tiny 2-hidden-layer color head (`MLP`) that consumes the interpolated per-vertex
feature + view-independent bottleneck feature -- is faithfully transcribed here
into base-env torch, layer for layer and hyperparameter for hyperparameter
(trunk_width=384, trunk_depth=8, trunk_skip_length=4, position_encoding_max_
frequency_power=10, num_bottleneck_features=8, color MLP features=[16,16,3]),
matching stage1.py lines ~1220-1267 exactly. Flax's `nn.compact` skip-connection
loop (`if i % trunk_skip_length == 0 and i > 0: net = concat([net, inputs])`) is
reproduced as-is; Flax `nn.Dense` -> `torch.nn.Linear`.
"""

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def sinusoidal_encoding(
    position: torch.Tensor,
    minimum_frequency_power: int,
    maximum_frequency_power: int,
    include_identity: bool = False,
) -> torch.Tensor:
    """Port of stage1.py's `sinusoidal_encoding` (lines ~337-348): sin(x*2^k) and
    sin(x*2^k + pi/2) (i.e. cos via phase shift) stacked and flattened, matching the
    original jnp implementation elementwise."""
    device = position.device
    dtype = position.dtype
    frequency = 2.0 ** torch.arange(
        minimum_frequency_power, maximum_frequency_power, device=device, dtype=dtype
    )
    angle = position[..., None, :] * frequency[:, None]
    encoding = torch.sin(torch.stack([angle, angle + 0.5 * math.pi], dim=-2))
    encoding = encoding.reshape(*position.shape[:-1], -1)
    if include_identity:
        encoding = torch.cat([position, encoding], dim=-1)
    return encoding


class RadianceField(nn.Module):
    """Port of stage1.py's `RadianceField` (lines ~1230-1251): sinusoidally-encoded
    coordinate MLP with a skip connection every `trunk_skip_length` layers, used for
    both the density_model (out_dim=1) and feature_model (out_dim=num_bottleneck_
    features=8) in the original script."""

    def __init__(
        self,
        out_dim: int,
        trunk_width: int = 384,
        trunk_depth: int = 8,
        trunk_skip_length: int = 4,
        position_encoding_max_frequency_power: int = 10,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.trunk_width = trunk_width
        self.trunk_depth = trunk_depth
        self.trunk_skip_length = trunk_skip_length
        self.position_encoding_max_frequency_power = position_encoding_max_frequency_power

        in_dim = (
            3 * 2 * position_encoding_max_frequency_power
        )  # sinusoidal_encoding(pos, 0, K) width
        layers = []
        cur_in = in_dim
        for i in range(trunk_depth):
            layers.append(nn.Linear(cur_in, trunk_width))
            if i % trunk_skip_length == 0 and i > 0:
                cur_in = trunk_width + in_dim
            else:
                cur_in = trunk_width
        self.trunk = nn.ModuleList(layers)
        self.out = nn.Linear(cur_in, out_dim)
        self.activation = nn.functional.relu

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        inputs = sinusoidal_encoding(positions, 0, self.position_encoding_max_frequency_power)
        net = inputs
        for i, layer in enumerate(self.trunk):
            net = layer(net)
            net = self.activation(net)
            if i % self.trunk_skip_length == 0 and i > 0:
                net = torch.cat([net, inputs], dim=-1)
        net = self.out(net)
        return net


class MLP(nn.Module):
    """Port of stage1.py's `MLP` (lines ~1254-1262): the tiny color head, instantiated
    upstream as `MLP([16, 16, 3])`."""

    def __init__(self, features):
        super().__init__()
        self.features = list(features)
        self.linears = nn.ModuleList()

    def build(self, in_dim: int):
        # Flax `nn.compact` infers `nn.Dense` input width lazily from the first call;
        # this staging port takes it explicitly at construction time instead (see
        # build_mobilenerf below), which is a mechanical torch-vs-flax lazy-init
        # difference only -- the layer widths/activations are identical to upstream.
        cur = in_dim
        for feat in self.features[:-1]:
            self.linears.append(nn.Linear(cur, feat))
            cur = feat
        self.linears.append(nn.Linear(cur, self.features[-1]))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.linears[:-1]:
            x = torch.relu(linear(x))
        x = self.linears[-1](x)
        return x


class MobileNeRFRadianceModel(nn.Module):
    """Wraps the three networks stage1.py instantiates together
    (density_model / feature_model / color_model, lines ~1265-1267) into a single
    forward-traceable module: evaluate density + bottleneck feature at a batch of 3D
    positions, then evaluate view-dependent color from the feature concatenated with
    a view direction, exactly mirroring how stage1.py's rendering loop calls each
    sub-network on the same positions."""

    def __init__(self, num_bottleneck_features: int = 8):
        super().__init__()
        self.density_model = RadianceField(1)
        self.feature_model = RadianceField(num_bottleneck_features)
        self.color_model = MLP([16, 16, 3]).build(3 + num_bottleneck_features)

    def forward(self, positions: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
        density = self.density_model(positions)
        feature = self.feature_model(positions)
        color_input = torch.cat([view_dirs, feature], dim=-1)
        color = self.color_model(color_input)
        return torch.cat([color, density], dim=-1)


def build_mobilenerf():
    torch.manual_seed(0)
    return MobileNeRFRadianceModel(num_bottleneck_features=8)


def example_input_mobilenerf():
    torch.manual_seed(0)
    positions = torch.randn(16, 3)
    view_dirs = torch.randn(16, 3)
    return (positions, view_dirs)


MENAGERIE_ENTRIES = [
    ("MobileNeRF", build_mobilenerf, example_input_mobilenerf, 2023, "ported-pytorch"),
]
