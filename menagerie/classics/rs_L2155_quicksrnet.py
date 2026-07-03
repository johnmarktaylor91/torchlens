# SOURCE: vendored from quic/aimet-model-zoo @ develop
# https://raw.githubusercontent.com/quic/aimet-model-zoo/develop/aimet_zoo_torch/quicksrnet/model/models.py
# https://raw.githubusercontent.com/quic/aimet-model-zoo/develop/aimet_zoo_torch/quicksrnet/model/blocks.py
#
# QuickSRNet (Qualcomm AI Research, arXiv:2303.04336) is exposed in the qai_hub_models
# package (quic/ai-hub-models -> qai_hub_models/models/quicksrnet{small,medium,large})
# but that wrapper only re-imports the real architecture from
# aimet_zoo_torch.quicksrnet.model.models.QuickSRNetBase (quic/aimet-model-zoo), which is
# pure-torch (no aimet/quantization dependency needed to construct/run the plain fp32
# module -- `AnchorOp`/`AddOp` from blocks.py are plain nn.Module helpers). Copied
# verbatim: `QuickSRNetBase`, `QuickSRNetSmall`, `QuickSRNetMedium`, `QuickSRNetLarge`
# (models.py) and their `AddOp`/`AnchorOp` dependencies (blocks.py). The
# quantization-collapse helpers (`to_dcr`, `convert_conv_*`) and the aimet-zoo checkpoint
# downloader (`_load_quicksrnet_source_model`, requires the aimet_zoo_torch package + repo
# checkout) are not vendored -- only the architecture-defining classes, constructed here
# with random init instead of the released pretrained weights.
"""QuickSRNet: anchor-based single-image super-resolution CNN for mobile/edge (Qualcomm)."""

from collections import OrderedDict

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from blocks.py ---
class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2


class AnchorOp(nn.Module):
    """
    Repeat interleaves the input scaling_factor**2 number of times along the channel axis.
    """

    def __init__(
        self,
        scaling_factor,
        in_channels=3,
        init_weights=True,
        freeze_weights=True,
        kernel_size=1,
        **kwargs,
    ):
        """
        Args:
            scaling_factor: Scaling factor
            init_weights:   Initializes weights to perform nearest upsampling (Default for Anchor)
            freeze_weights:         Whether to freeze weights (if initialised as nearest upsampling weights)
        """
        super().__init__()

        self.net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(in_channels * scaling_factor**2),
            kernel_size=kernel_size,
            **kwargs,
        )

        if init_weights:
            num_channels_per_group = in_channels // self.net.groups
            weight = torch.zeros(
                in_channels * scaling_factor**2, num_channels_per_group, kernel_size, kernel_size
            )

            bias = torch.zeros(weight.shape[0])
            for ii in range(in_channels):
                weight[
                    ii * scaling_factor**2 : (ii + 1) * scaling_factor**2,
                    ii % num_channels_per_group,
                    kernel_size // 2,
                    kernel_size // 2,
                ] = 1.0

            new_state_dict = OrderedDict({"weight": weight, "bias": bias})
            self.net.load_state_dict(new_state_dict)

            if freeze_weights:
                for param in self.net.parameters():
                    param.requires_grad = False

    def forward(self, input):
        return self.net(input)


# --- vendored from models.py ---
class QuickSRNetBase(nn.Module):
    """
    Base class for all QuickSRNet variants.

    Note on supported scaling factors: this class supports integer scaling factors. 1.5x upscaling is
    the only non-integer scaling factor supported.
    """

    def __init__(
        self,
        scaling_factor,
        num_channels,
        num_intermediate_layers,
        use_ito_connection,
        in_channels=3,
        out_channels=3,
    ):
        """
        :param scaling_factor:           scaling factor for LR-to-HR upscaling (2x, 3x, 4x... or 1.5x)
        :param num_channels:             number of feature channels for convolutional layers
        :param num_intermediate_layers:  number of intermediate conv layers
        :param use_ito_connection:       whether to use an input-to-output residual connection or not
                                         (using one facilitates quantization)
        :param in_channels:              number of channels for LR input (default 3 for RGB frames)
        :param out_channels:             number of channels for HR output (default 3 for RGB frames)
        """

        super().__init__()
        self.out_channels = out_channels
        self._use_ito_connection = use_ito_connection
        self._has_integer_scaling_factor = float(scaling_factor).is_integer()

        if self._has_integer_scaling_factor:
            self.scaling_factor = int(scaling_factor)

        elif scaling_factor == 1.5:
            self.scaling_factor = scaling_factor

        else:
            raise NotImplementedError(
                f"1.5 is the only supported non-integer scaling factor. Received {scaling_factor}."
            )

        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=num_channels,
                        out_channels=num_channels,
                        kernel_size=(3, 3),
                        padding=1,
                    ),
                    nn.Hardtanh(min_val=0.0, max_val=1.0),
                ]
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1
            ),
            nn.Hardtanh(min_val=0.0, max_val=1.0),
            *intermediate_layers,
        )

        if scaling_factor == 1.5:
            cl_in_channels = num_channels * (2**2)
            cl_out_channels = out_channels * (3**2)
            cl_kernel_size = (1, 1)
            cl_padding = 0
        else:
            cl_in_channels = num_channels
            cl_out_channels = out_channels * (self.scaling_factor**2)
            cl_kernel_size = (3, 3)
            cl_padding = 1

        self.conv_last = nn.Conv2d(
            in_channels=cl_in_channels,
            out_channels=cl_out_channels,
            kernel_size=cl_kernel_size,
            padding=cl_padding,
        )

        if use_ito_connection:
            self.add_op = AddOp()

            if scaling_factor == 1.5:
                self.anchor = AnchorOp(
                    scaling_factor=3, kernel_size=3, stride=2, padding=1, freeze_weights=False
                )
            else:
                self.anchor = AnchorOp(scaling_factor=self.scaling_factor, freeze_weights=False)

        if scaling_factor == 1.5:
            self.space_to_depth = nn.PixelUnshuffle(2)
            self.depth_to_space = nn.PixelShuffle(3)
        else:
            self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

        self.clip_output = nn.Hardtanh(min_val=0.0, max_val=1.0)

        self.initialize()

        self._is_dcr = False

    def forward(self, input):
        x = self.cnn(input)

        if not self._has_integer_scaling_factor:
            x = self.space_to_depth(x)

        if self._use_ito_connection:
            residual = self.conv_last(x)
            input_convolved = self.anchor(input)
            x = self.add_op(input_convolved, residual)
        else:
            x = self.conv_last(x)

        x = self.clip_output(x)

        return self.depth_to_space(x)

    def initialize(self):
        for conv_layer in self.cnn:
            # Initialise each conv layer so that it behaves similarly to:
            # y = conv(x) + x after initialization
            if isinstance(conv_layer, nn.Conv2d):
                middle = conv_layer.kernel_size[0] // 2
                num_residual_channels = min(conv_layer.in_channels, conv_layer.out_channels)
                with torch.no_grad():
                    for idx in range(num_residual_channels):
                        conv_layer.weight[idx, idx, middle, middle] += 1.0

        if not self._use_ito_connection:
            # This will initialize the weights of the last conv so that it behaves like:
            # y = conv(x) + repeat_interleave(x, scaling_factor ** 2) after initialization
            middle = self.conv_last.kernel_size[0] // 2
            out_channels = self.conv_last.out_channels
            scaling_factor_squarred = out_channels // self.out_channels
            with torch.no_grad():
                for idx_out in range(out_channels):
                    idx_in = (idx_out % out_channels) // scaling_factor_squarred
                    self.conv_last.weight[idx_out, idx_in, middle, middle] += 1.0


class QuickSRNetSmall(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=2,
            use_ito_connection=False,
            **kwargs,
        )


class QuickSRNetMedium(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=5,
            use_ito_connection=False,
            **kwargs,
        )


class QuickSRNetLarge(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=64,
            num_intermediate_layers=11,
            use_ito_connection=True,
            **kwargs,
        )


def build_quicksrnet_small():
    torch.manual_seed(0)
    return QuickSRNetSmall(scaling_factor=2)


def example_input_quicksrnet_small():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 64, 64),)


def build_quicksrnet_medium():
    torch.manual_seed(0)
    return QuickSRNetMedium(scaling_factor=2)


def example_input_quicksrnet_medium():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 64, 64),)


def build_quicksrnet_large():
    torch.manual_seed(0)
    # Large is the only variant with the input-to-output anchor/residual connection.
    return QuickSRNetLarge(scaling_factor=2)


def example_input_quicksrnet_large():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 64, 64),)


MENAGERIE_ENTRIES = [
    (
        "QuickSRNet-Small",
        "build_quicksrnet_small",
        "example_input_quicksrnet_small",
        2023,
        "vendored",
    ),
    (
        "QuickSRNet-Medium",
        "build_quicksrnet_medium",
        "example_input_quicksrnet_medium",
        2023,
        "vendored",
    ),
    (
        "QuickSRNet-Large",
        "build_quicksrnet_large",
        "example_input_quicksrnet_large",
        2023,
        "vendored",
    ),
]
