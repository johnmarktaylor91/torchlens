# SOURCE: vendored from computational-imaging/ACORN @ main
# (https://github.com/computational-imaging/ACORN/blob/main/modules.py)
#
# Martel, Lindell, Lin, Chan, Monteiro, Tancik, Wetzstein 2021 (SIGGRAPH) "ACORN:
# Adaptive Coordinate Networks for Neural Scene Representation". `ImplicitAdaptivePatchNet`
# (the multiscale-coordinate-network the paper uses for 2D images -- selected by
# `experiment_scripts/train_img.py` when `--model_type multiscale`, the default) is
# vendored verbatim: a coordinate MLP (`coord2features_net`, itself the real `FCBlock`)
# maps a (possibly positionally-encoded) per-block coordinate to a small local feature
# grid, `torch.nn.functional.grid_sample` bilinearly interpolates that grid at each
# block's fine local coordinates, and a second small MLP (`features2sample_net`) maps
# each sampled feature vector to the output pixel value. No architectural changes.
#
# The real training script (train_img.py) constructs this with
# `in_features=3` (a [scale/level, y, x]-style multiscale coordinate, not just 2D
# pixel position -- ACORN's "adaptive" quadtree indexes patches at different octree/
# quadtree depths) and `patch_size=opt.patch_size[1:]` as a 2-tuple; `example_input_acorn`
# below only needs a correctly-shaped random `coords`/`fine_rel_coords` pair (per the
# TorchLens capture convention of random-init + random input for structural tracing,
# not a trained/semantically-meaningful scene), not real image content.
import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class Sine(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor w0
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


class FCBlock(nn.Module):
    """A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=False,
        nonlinearity="relu",
        weight_init=None,
        w0=30,
    ):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            "sine": (Sine(w0=w0), lambda m: sine_init(m, w0=w0), first_layer_sine_init),
            "relu": (nn.ReLU(inplace=True), init_weights_normal, None),
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nl))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if (
            first_layer_init is not None
        ):  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords):
        output = self.net(coords)
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        num_encoding_functions=6,
        include_input=True,
        log_sampling=True,
        normalize=False,
        input_dim=3,
        gaussian_pe=False,
        gaussian_variance=38,
    ):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(
                gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                requires_grad=False,
            )
        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0, self.num_encoding_functions - 1, self.num_encoding_functions
                )
            else:
                self.frequency_bands = torch.linspace(
                    2.0**0.0, 2.0 ** (self.num_encoding_functions - 1), self.num_encoding_functions
                )

            if normalize:
                self.normalization = torch.tensor(1 / self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx] * func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


class ImplicitAdaptivePatchNet(nn.Module):
    def __init__(
        self,
        in_features=3,
        out_features=1,
        feature_grid_size=(8, 8, 8),
        hidden_features=256,
        num_hidden_layers=3,
        patch_size=8,
        code_dim=8,
        use_pe=True,
        num_encoding_functions=6,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feature_grid_size = feature_grid_size
        self.patch_size = patch_size
        self.use_pe = use_pe

        if self.use_pe:
            self.positional_encoding = PositionalEncoding(
                num_encoding_functions=num_encoding_functions
            )
            in_features = 2 * in_features * num_encoding_functions + in_features

        self.coord2features_net = FCBlock(
            in_features=in_features,
            out_features=np.prod(feature_grid_size),
            num_hidden_layers=num_hidden_layers,
            hidden_features=hidden_features,
            outermost_linear=True,
            nonlinearity="relu",
        )

        self.features2sample_net = FCBlock(
            in_features=self.feature_grid_size[0],
            out_features=out_features,
            num_hidden_layers=1,
            hidden_features=64,
            outermost_linear=True,
            nonlinearity="relu",
        )

    def forward(self, model_input):
        # Enables us to compute gradients w.r.t. coordinates
        coords = model_input["coords"].clone().detach().requires_grad_(True)
        fine_coords = model_input["fine_rel_coords"].clone().detach().requires_grad_(True)

        if self.use_pe:
            coords = self.positional_encoding(coords)

        features = self.coord2features_net(coords)

        # features is size (Batch Size, Blocks, prod(feature_grid_size))
        # but currently interpolate bilinear only supports one batch dimension,
        # therefore, for now assume that Batch Size == 1
        assert features.shape[0] == 1, "Code currently only supports Batch Size == 1"

        n_channels, dx, dy = self.feature_grid_size
        features = features.squeeze(0)
        b_size = features.shape[0]

        features_in = features.squeeze().reshape(b_size, n_channels, dx, dy)
        sample_coords_out = fine_coords[0, ...].reshape(1, -1, 2)
        sample_coords = sample_coords_out.reshape(b_size, self.patch_size[0], self.patch_size[1], 2)

        y = sample_coords[..., :1]
        x = sample_coords[..., 1:]
        sample_coords = torch.cat([y, x], dim=-1)

        features_out = torch.nn.functional.grid_sample(
            features_in, sample_coords, mode="bilinear", padding_mode="border", align_corners=True
        ).reshape(b_size, n_channels, np.prod(self.patch_size))

        # permute from (Blocks, feature_grid_size[0], patch_size**2)->(Blocks, patch_size**2, feature_grid_size[0])
        # so the network maps features to function output
        features_out = features_out.permute(0, 2, 1)

        # for all spatial feature vectors, extract function value
        patch_out = self.features2sample_net(features_out)

        # squeeze out last dimension and restore batch dimension
        patch_out = patch_out.unsqueeze(0)

        return {
            "model_in": {"sample_coords_out": sample_coords_out, "model_in_coarse": coords},
            "model_out": {"output": patch_out, "codes": None},
        }


def build_acorn():
    return ImplicitAdaptivePatchNet(
        in_features=3,
        out_features=1,
        feature_grid_size=(4, 4, 4),
        hidden_features=16,
        num_hidden_layers=1,
        patch_size=(4, 4),
        use_pe=True,
        num_encoding_functions=2,
    )


def example_input_acorn():
    # 2 quadtree blocks; each block's local feature grid is bilinearly sampled at a
    # 4x4 patch of fine (per-pixel) relative coordinates.
    num_blocks = 2
    patch_h, patch_w = 4, 4
    coords = torch.rand(1, num_blocks, 3) * 2 - 1
    fine_rel_coords = torch.rand(1, num_blocks * patch_h * patch_w, 2) * 2 - 1
    return ({"coords": coords, "fine_rel_coords": fine_rel_coords},)


MENAGERIE_ENTRIES = [
    (
        "ACORN (Adaptive Coordinate Networks)",
        "build_acorn",
        "example_input_acorn",
        2021,
        MENAGERIE_ZOO,
    ),
]
