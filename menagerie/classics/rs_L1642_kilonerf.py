# SOURCE: vendored from creiser/kilonerf @ master (multi_modules.py)
# https://github.com/creiser/kilonerf -- "KiloNeRF: Speeding up Neural Radiance Fields
# with Thousands of Tiny MLPs" (Reiser, Peng, Liao, Geiger; ICCV 2021). The official
# repo's fast path routes through a custom CUDA extension (`kilonerf_cuda`) for the
# grouped multi-matmul kernel, but `MultiNetworkLinear` also ships a pure-PyTorch
# `implementation='bmm'` path (batched `torch.bmm`) that is architecturally identical --
# thousands of independent tiny MLPs sharing one batched weight tensor, one dedicated
# to each spatial partition of the scene, queried in parallel via batched matmul. The
# CUDA-only helpers (`AddMultiMatMul`, `naive_multimatmul*`, `query_multi_network`, the
# occupancy-grid routing driver) are the standalone fast-path/rendering machinery, not
# part of the `MultiNetwork` module graph, and are dropped here; only the top-level
# `import kilonerf_cuda` and the `from utils import *` (which pulled in `lpips` +
# `run_nerf_helpers`, both unrelated to the tiny-MLP architecture) are removed. Every
# class body below (`MultiNetworkLinear`, `SharedLinear`, `Sine`,
# `MultiNetworkFourierEmbedding`, `MultiNetwork`) is transcribed verbatim.
import math

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from multi_modules.py (init helpers; only this fn differs from
# torch.nn.init's private helper, to account for the extra leading network dim) ----
def _calculate_fan_in_and_fan_out(tensor):
    fan_in = tensor.size(-1)
    fan_out = tensor.size(-2)
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def calculate_gain(nonlinearity, param=None):
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def xavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-a, a)


def xavier_normal_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0.0, std)


class MultiNetworkFourierEmbedding(nn.Module):
    def __init__(self, num_networks, num_input_channels, num_frequencies):
        super(MultiNetworkFourierEmbedding, self).__init__()

        max_frequency = num_frequencies - 1
        self.frequency_bands = 2.0 ** torch.linspace(0.0, max_frequency, steps=num_frequencies)
        self.num_frequencies = num_frequencies
        self.num_output_channels = (2 * num_frequencies + 1) * num_input_channels
        self.num_networks = num_networks

    def forward(self, x, implementation="pytorch", num_blocks=46, num_threads=512):
        # x: num_networks x batch_size x num_input_channels
        batch_size, num_input_channels = x.size(1), x.size(2)
        x = (
            x.unsqueeze(3)
            .expand(self.num_networks, batch_size, num_input_channels, 2 * self.num_frequencies + 1)
            .contiguous()
        )
        x[:, :, :, 1 : 1 + self.num_frequencies] = x[:, :, :, 0].unsqueeze(
            3
        ) * self.frequency_bands.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(x)
        x[:, :, :, 1 + self.num_frequencies :] = x[:, :, :, 1 : 1 + self.num_frequencies]
        x[:, :, :, 1 : 1 + self.num_frequencies] = torch.cos(
            x[:, :, :, 1 : 1 + self.num_frequencies]
        )
        x[:, :, :, 1 + self.num_frequencies :] = torch.sin(x[:, :, :, 1 + self.num_frequencies :])
        return x.view(self.num_networks, batch_size, -1)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# For hard parameter sharing
class SharedLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        nonlinearity="leaky_relu",
        weight_initialization_method="kaiming_uniform",
        bias_initialization_method="standard",
    ):
        super(SharedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.nonlinearity = nonlinearity
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initialization_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
        elif self.weight_initialization_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
        elif self.weight_initialization_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(self.nonlinearity))
        elif self.weight_initialization_method == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=nn.init.calculate_gain(self.nonlinearity))
        if self.bias is not None:
            if self.bias_initialization_method == "standard":
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            elif self.bias_initialization_method == "zeros":
                nn.init.zeros_(self.bias)

    def forward(self, input, batch_size_per_network=None):
        has_network_dim = len(list(input.size())) == 3
        if has_network_dim:
            num_networks = input.size(0)
            input = input.view(-1, self.in_features)
        out = F.linear(input, self.weight, self.bias)
        if has_network_dim:
            out = out.view(num_networks, -1, self.out_features)
        return out


class MultiNetworkLinear(nn.Module):
    rng_state = None

    def __init__(
        self,
        num_networks,
        in_features,
        out_features,
        nonlinearity="leaky_relu",
        bias=True,
        implementation="bmm",
        nonlinearity_params=None,
        use_same_initialization_for_all_networks=False,
        network_rng_seed=None,
        weight_initialization_method="kaiming_uniform",
        bias_initialization_method="standard",
    ):
        super(MultiNetworkLinear, self).__init__()
        self.num_networks = num_networks
        self.in_features = in_features
        self.out_features = out_features
        self.implementation = implementation
        self.use_same_initialization_for_all_networks = use_same_initialization_for_all_networks
        self.network_rng_seed = network_rng_seed
        # weight is created in reset_parameters(); the 'multimatmul*' implementations
        # (CUDA-only fast path) are intentionally not vendored -- the default 'bmm'
        # path below never touches kilonerf_cuda.
        self.nonlinearity = nonlinearity
        self.nonlinearity_params = nonlinearity_params
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_networks, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(
            torch.Tensor(self.num_networks, self.out_features, self.in_features)
        )

        if self.network_rng_seed is not None:
            previous_rng_state = torch.random.get_rng_state()
            if MultiNetworkLinear.rng_state is None:
                torch.random.manual_seed(self.network_rng_seed)
            else:
                torch.random.set_rng_state(MultiNetworkLinear.rng_state)

        if self.nonlinearity != "sine":
            if self.weight_initialization_method == "kaiming_uniform":
                kaiming_uniform_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
            elif self.weight_initialization_method == "kaiming_normal":
                kaiming_normal_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
            elif self.weight_initialization_method == "xavier_uniform":
                xavier_uniform_(self.weight, gain=calculate_gain(self.nonlinearity))
            elif self.weight_initialization_method == "xavier_normal":
                xavier_normal_(self.weight, gain=calculate_gain(self.nonlinearity))
            if self.bias is not None:
                if self.bias_initialization_method == "standard":
                    fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(self.bias, -bound, bound)
                elif self.bias_initialization_method == "zeros":
                    nn.init.zeros_(self.bias)
        else:  # For SIREN
            c, w0, is_first = (
                self.nonlinearity_params["c"],
                self.nonlinearity_params["w0"],
                self.nonlinearity_params["is_first"],
            )
            w_std = (1 / self.in_features) if is_first else (math.sqrt(c / self.in_features) / w0)
            nn.init.uniform_(self.weight, -w_std, w_std)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -w_std, w_std)

        if self.network_rng_seed is not None:
            MultiNetworkLinear.rng_state = torch.random.get_rng_state()
            torch.random.set_rng_state(previous_rng_state)

        if self.use_same_initialization_for_all_networks:
            with torch.no_grad():
                self.weight[1:] = self.weight[0]
                self.bias[1:] = self.bias[0]

        if "multimatmul" in self.implementation:
            self.weight.data = self.weight.data.view(
                self.num_networks, self.in_features, self.out_features
            ).contiguous()

    def forward(self, x, batch_size_per_network=None, bias=None, weight=None):
        # For testing purposes override weight and bias
        if bias is not None:
            self.bias = bias
        if weight is not None:
            self.weight = weight
        # x = num_networks x batch_size x in_features
        batch_size = x.size(1)
        if self.num_networks > 1:
            if self.implementation == "bmm":
                weight_transposed = self.weight.permute(
                    0, 2, 1
                )  # num_networks x in_features x out_features
                # num_networks x batch_size x in_features @ num_networks x in_features x out_features
                product = torch.bmm(x, weight_transposed)
                bias_view = self.bias.unsqueeze(1)
            elif self.implementation == "matmul":
                input_view = x.unsqueeze(3)  # num_networks x batch_size x in_features x 1
                weight_view = self.weight.unsqueeze(
                    1
                )  # num_networks x 1 x out_features x in_features
                product = torch.matmul(weight_view, input_view).squeeze(
                    3
                )  # num_networks x batch_size x out_features
                bias_view = self.bias.unsqueeze(1)
            result = product + bias_view
        else:
            input_view = x.squeeze(0)
            weight_view = self.weight.squeeze(0)
            bias_view = self.bias.squeeze(0)
            result = F.linear(input_view, weight_view, bias_view)
        return result.view(self.num_networks, batch_size, self.out_features)


class MultiNetwork(nn.Module):
    def __init__(
        self,
        num_networks,
        num_position_channels,
        num_direction_channels,
        num_output_channels,
        hidden_layer_size,
        num_hidden_layers,
        refeed_position_index=None,
        late_feed_direction=False,
        direction_layer_size=None,
        nonlinearity="relu",
        nonlinearity_initalization="pass_leaky_relu",
        use_single_net=False,
        linear_implementation="bmm",
        use_same_initialization_for_all_networks=False,
        network_rng_seed=None,
        weight_initialization_method="kaiming_uniform",
        bias_initialization_method="standard",
        alpha_rgb_initalization="updated_yenchenlin",
        use_hard_parameter_sharing_for_color=False,
        view_dependent_dropout_probability=-1,
        use_view_independent_color=False,
    ):
        super(MultiNetwork, self).__init__()

        self.num_networks = num_networks
        self.num_position_channels = num_position_channels
        self.num_direction_channels = num_direction_channels
        self.num_output_channels = num_output_channels
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.refeed_position_index = refeed_position_index
        self.late_feed_direction = late_feed_direction
        self.direction_layer_size = direction_layer_size
        self.nonlinearity = nonlinearity
        self.nonlinearity_initalization = nonlinearity_initalization
        self.use_single_net = use_single_net
        self.linear_implementation = linear_implementation
        self.use_same_initialization_for_all_networks = use_same_initialization_for_all_networks
        self.network_rng_seed = network_rng_seed
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        self.alpha_rgb_initalization = alpha_rgb_initalization
        self.use_hard_parameter_sharing_for_color = use_hard_parameter_sharing_for_color
        self.view_dependent_dropout_probability = view_dependent_dropout_probability
        self.use_view_independent_color = use_view_independent_color

        nonlinearity_params = {}
        if nonlinearity == "sigmoid":
            self.activation = nn.Sigmoid()
        if nonlinearity == "tanh":
            self.activation = nn.Tanh()
        if nonlinearity == "relu":
            self.activation = nn.ReLU()
        if nonlinearity == "leaky_relu":
            self.activation = nn.LeakyReLU()
        if nonlinearity == "sine":
            nonlinearity_params = {"w0": 30.0, "c": 6.0, "is_first": True}
            self.activation = Sine(nonlinearity_params["w0"])

        def linear_layer(
            in_features, out_features, actual_nonlinearity, use_hard_parameter_sharing=False
        ):
            if self.nonlinearity_initalization == "pass_actual_nonlinearity":
                passed_nonlinearity = actual_nonlinearity
            elif self.nonlinearity_initalization == "pass_leaky_relu":
                passed_nonlinearity = "leaky_relu"
            if not use_hard_parameter_sharing:
                return MultiNetworkLinear(
                    self.num_networks,
                    in_features,
                    out_features,
                    nonlinearity=passed_nonlinearity,
                    nonlinearity_params=nonlinearity_params,
                    implementation=linear_implementation,
                    use_same_initialization_for_all_networks=use_same_initialization_for_all_networks,
                    network_rng_seed=network_rng_seed,
                )
            else:
                return SharedLinear(
                    in_features, out_features, bias=True, nonlinearity=passed_nonlinearity
                )

        if self.late_feed_direction:
            self.pts_linears = [
                linear_layer(self.num_position_channels, self.hidden_layer_size, self.nonlinearity)
            ]
            nonlinearity_params = nonlinearity_params.copy()
            nonlinearity_params.update({"is_first": False})
            for i in range(self.num_hidden_layers - 1):
                if i == self.refeed_position_index:
                    new_layer = linear_layer(
                        self.hidden_layer_size + self.num_position_channels,
                        self.hidden_layer_size,
                        self.nonlinearity,
                    )
                else:
                    new_layer = linear_layer(
                        self.hidden_layer_size, self.hidden_layer_size, self.nonlinearity
                    )
                self.pts_linears.append(new_layer)
            self.pts_linears = nn.ModuleList(self.pts_linears)
            self.direction_layer = linear_layer(
                self.num_direction_channels + self.hidden_layer_size,
                self.direction_layer_size,
                self.nonlinearity,
                self.use_hard_parameter_sharing_for_color,
            )

            if self.use_view_independent_color:
                feature_output_size = self.hidden_layer_size + 4  # + RGBA
            else:
                feature_output_size = self.hidden_layer_size
            self.feature_linear = linear_layer(
                self.hidden_layer_size, feature_output_size, "linear"
            )
            if not self.use_view_independent_color:
                self.alpha_linear = linear_layer(
                    self.hidden_layer_size,
                    1,
                    "linear" if self.alpha_rgb_initalization == "updated_yenchenlin" else "relu",
                )
            self.rgb_linear = linear_layer(
                self.direction_layer_size,
                3,
                "linear" if self.alpha_rgb_initalization == "updated_yenchenlin" else "sigmoid",
                self.use_hard_parameter_sharing_for_color,
            )

            self.view_dependent_parameters = list(self.direction_layer.parameters()) + list(
                self.rgb_linear.parameters()
            )

            if self.view_dependent_dropout_probability > 0:
                self.dropout_after_feature = nn.Dropout(self.view_dependent_dropout_probability)
                self.dropout_after_direction_layer = nn.Dropout(
                    self.view_dependent_dropout_probability
                )

        else:
            layers = [
                linear_layer(
                    self.num_position_channels + self.num_direction_channels,
                    self.hidden_layer_size,
                    self.nonlinearity,
                ),
                self.activation,
            ]
            nonlinearity_params = nonlinearity_params.copy()
            nonlinearity_params.update({"is_first": False})
            for _ in range(self.num_hidden_layers):
                layers += [
                    linear_layer(self.hidden_layer_size, self.hidden_layer_size, self.nonlinearity),
                    self.activation,
                ]
            layers += [linear_layer(self.hidden_layer_size, self.num_output_channels, "linear")]
            self.layers = nn.Sequential(*layers)

    # random_directions will be used for regularizing the view-independent color
    def forward(self, x, batch_size_per_network=None, random_directions=None):
        if self.late_feed_direction:
            if isinstance(x, list):
                positions, directions = x
            else:
                positions, directions = torch.split(
                    x, [self.num_position_channels, self.num_direction_channels], dim=-1
                )
            h = positions
            for i, l in enumerate(self.pts_linears):  # noqa: E741 (verbatim from source)
                h = self.pts_linears[i](h, batch_size_per_network)
                h = self.activation(h)
                if i == self.refeed_position_index:
                    h = torch.cat([positions, h], -1)
            if not self.use_view_independent_color:
                alpha = self.alpha_linear(h, batch_size_per_network)
            feature = self.feature_linear(h, batch_size_per_network)
            if self.view_dependent_dropout_probability > 0:
                feature = self.dropout_after_feature(feature)
            if self.use_view_independent_color:
                rgb_view_independent, alpha, feature = torch.split(
                    feature, [3, 1, self.hidden_layer_size], dim=-1
                )

            if random_directions is not None:
                assert self.use_view_independent_color, (
                    "this regularization only makes sense if we output a view-independent color"
                )
                num_random_directions = random_directions.size(0)
                batch_size = feature.size(0)
                feature_size = feature.size(1)
                feature = feature.repeat(1, num_random_directions + 1).view(-1, feature_size)
                random_directions = random_directions.repeat(batch_size, 1).view(
                    batch_size, num_random_directions, -1
                )
                directions = torch.cat([directions.unsqueeze(1), random_directions], dim=1).view(
                    batch_size * (num_random_directions + 1), -1
                )
                batch_size_per_network = (num_random_directions + 1) * batch_size_per_network

            # View-dependent part of the network:
            h = torch.cat([feature, directions], -1)
            h = self.direction_layer(h, batch_size_per_network)
            h = self.activation(h)
            if self.view_dependent_dropout_probability > 0:
                h = self.dropout_after_direction_layer(h)
            rgb = self.rgb_linear(h, batch_size_per_network)

            if self.use_view_independent_color:
                if random_directions is None:
                    rgb = rgb + rgb_view_independent
                else:
                    mean_rgb = rgb.view(batch_size, num_random_directions + 1, 3)
                    mean_rgb = mean_rgb + rgb_view_independent.unsqueeze(1)
                    rgb = mean_rgb[:, 0]
                    mean_rgb = mean_rgb.mean(dim=1)
                    mean_regularization_term = torch.abs(mean_rgb - rgb_view_independent).mean()

            result = torch.cat([rgb, alpha], -1)

            if random_directions is not None:
                return result, mean_regularization_term
            else:
                return result
        else:
            return self.layers(x)


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_kilonerf_multinetwork():
    """Small grid of tiny per-cell MLPs (the KiloNeRF architecture), late-feed-direction
    mode matching the paper's default (position -> hidden trunk -> alpha/feature ->
    direction-conditioned RGB head), pure-PyTorch 'bmm' linear implementation so no
    CUDA extension is required.
    """
    torch.manual_seed(0)
    num_networks = 8  # e.g. an 2x2x2 occupancy-grid partition of tiny MLPs
    return MultiNetwork(
        num_networks=num_networks,
        num_position_channels=3,
        num_direction_channels=3,
        num_output_channels=4,
        hidden_layer_size=16,
        num_hidden_layers=3,
        refeed_position_index=1,
        late_feed_direction=True,
        direction_layer_size=8,
        nonlinearity="relu",
        linear_implementation="bmm",
    )


def example_input_kilonerf_multinetwork():
    torch.manual_seed(0)
    num_networks = 8
    batch_size_per_network = 4
    positions = torch.randn(num_networks, batch_size_per_network, 3)
    directions = torch.randn(num_networks, batch_size_per_network, 3)
    return ([positions, directions],)


MENAGERIE_ENTRIES = [
    (
        "KiloNeRF",
        build_kilonerf_multinetwork,
        example_input_kilonerf_multinetwork,
        2021,
        "vendored-pytorch",
    ),
]
