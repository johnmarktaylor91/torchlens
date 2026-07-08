# SOURCE: vendored from HaoyeYang/akita_pytorch_replica @ c90221a3324b00c85b288108fb42ad1a80a08947
# https://raw.githubusercontent.com/HaoyeYang/akita_pytorch_replica/c90221a3324b00c85b288108fb42ad1a80a08947/src/model/model.py
# https://raw.githubusercontent.com/HaoyeYang/akita_pytorch_replica/c90221a3324b00c85b288108fb42ad1a80a08947/src/model/modules.py
# https://raw.githubusercontent.com/HaoyeYang/akita_pytorch_replica/c90221a3324b00c85b288108fb42ad1a80a08947/configs/model_params.json
#
# Fudenberg et al. 2020 (Nat Methods) "Predicting 3D genome folding from DNA
# sequence with Akita" -- the original Akita is a Basenji-style 1D dilated-conv
# trunk feeding a 2D dilated-conv head that predicts Hi-C-like chromatin
# contact maps from a ~1Mb one-hot DNA window; original code is TF1.x
# (`calico/basenji`). This is a faithful PyTorch reimplementation/replica of
# that architecture by a third party (`HaoyeYang/akita_pytorch_replica`),
# config-driven off `configs/model_params.json` (the actual published Akita
# hyperparameters: conv_block -> conv_tower(x10) -> dilated_residual(x8) 1D
# trunk, then one_to_two -> concat_dist_2d -> conv_block_2d -> symmetrize_2d
# -> dilated_residual_2d(x6) -> cropping_2d -> upper_tri -> final Linear 2D
# head). `AkitaNet` (in `model.py`) and its building blocks
# (`StochasticReverseComplement`, `StochasticShift`, `OneToTwo`,
# `ConcatDist2D`, `Symmetrize2D`, `Cropping2D`, `UpperTri`,
# `SwitchReverseTriu`, `ConvBlock1D`, `DilatedResidual1D`, `ConvBlock2D`,
# `DilatedResidual2D`, in `modules.py`) are the real, unmodified classes from
# the two files above -- layer composition, channel arithmetic, and the
# trunk/head block-dispatch loop driven by `model_params["trunk"]` /
# `model_params["head_hic"]` are byte-for-byte the original. Only mechanical
# import-isolation edits: merged the two-file `from .modules import (...)`
# package-relative import into a single flat file (no package layout needed
# for a standalone staging module).
#
# `model_params.json`'s published config uses `seq_length: 1048576` (~1Mb DNA
# window) with 10 conv_tower repeats + 8/6 dilated-residual repeats -- far too
# large for a menagerie trace. `build_akitanet()` below passes the SAME
# `model_params` dict shape (same block types, same block ORDER, same
# architectural knobs: trunk block list, head_hic block list, activation,
# bn_momentum, diagonal_offset) but with drastically smaller
# `seq_length`/`filters`/`repeat` values so the identical code path runs at
# a size that traces quickly; no block type, order, or arithmetic is changed.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# --- Custom Layers (from modules.py) ---
class StochasticReverseComplement(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc_map = torch.tensor([3, 2, 1, 0], dtype=torch.long)

    def forward(self, x_one_hot, augment_rc_flag=True):
        if augment_rc_flag and self.training and random.random() < 0.5:
            x_indices = torch.argmax(x_one_hot, dim=1)
            x_rc_indices = self.rc_map.to(x_indices.device)[x_indices]
            x_rc_indices_flipped = torch.flip(x_rc_indices, dims=[1])
            x_rc_one_hot = F.one_hot(x_rc_indices_flipped, num_classes=4).permute(0, 2, 1).float()
            return x_rc_one_hot, torch.tensor(True, device=x_one_hot.device)
        else:
            return x_one_hot, torch.tensor(False, device=x_one_hot.device)


class StochasticShift(nn.Module):
    def __init__(self, shift_max=0, pad_value=0.25):
        super().__init__()
        self.shift_max = shift_max
        self.pad_value = pad_value

    def forward(self, x):
        if self.shift_max > 0 and self.training:
            shift_val = random.randint(-self.shift_max, self.shift_max)
            if shift_val == 0:
                return x

            if shift_val > 0:  # Shift right, pad left
                padding = (shift_val, 0)
                x_padded = F.pad(x, padding, mode="constant", value=self.pad_value)
                x_shifted = x_padded[..., : x.size(-1)]
            else:  # Shift left, pad right
                padding = (0, -shift_val)
                x_padded = F.pad(x, padding, mode="constant", value=self.pad_value)
                x_shifted = x_padded[..., -x.size(-1) :]
            return x_shifted
        return x


class OneToTwo(nn.Module):
    def __init__(self, operation="mean"):
        super().__init__()
        self.operation = operation

    def forward(self, x):
        # x: (B, C, L_repr)
        x_col = x.unsqueeze(3).expand(-1, -1, -1, x.size(2))  # (B, C, L_repr, L_repr)
        x_row = x.unsqueeze(2).expand(-1, -1, x.size(2), -1)  # (B, C, L_repr, L_repr)

        if self.operation == "mean":
            out = (x_col + x_row) / 2.0
        elif self.operation == "concat":
            out = torch.cat([x_col, x_row], dim=1)
        else:
            raise ValueError(f"Unknown OneToTwo operation: {self.operation}")
        return out


class ConcatDist2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, "Height and Width must be equal for distance calculation"
        L = H
        coords = torch.arange(L, device=x.device, dtype=torch.float32)
        dist_matrix = torch.abs(coords.unsqueeze(0) - coords.unsqueeze(1))

        # Normalize distance to [0,1] roughly
        dist_feature = dist_matrix / (L - 1) if L > 1 else torch.zeros_like(dist_matrix)

        dist_feature = dist_feature.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)
        return torch.cat([x, dist_feature], dim=1)


class Symmetrize2D(nn.Module):
    def forward(self, x):
        return (x + x.transpose(-2, -1)) / 2.0


class UpperTri(nn.Module):
    def __init__(self, diagonal_offset=0):
        super().__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, "Input for UpperTri must be square (H==W)"
        row_idx, col_idx = torch.triu_indices(H, W, offset=self.diagonal_offset, device=x.device)
        result = x[:, :, row_idx, col_idx]
        result = result.permute(0, 2, 1)  # (B, N_elements, C)
        return result


class SwitchReverseTriu(nn.Module):
    def __init__(self, diagonal_offset=0):
        super().__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, inputs_tuple):
        preds, rc_status = (
            inputs_tuple  # preds: (B, N, C), rc_status: scalar tensor for batch or (B,)
        )

        # This is a simplified implementation
        # For proper implementation, we would need to reconstruct the matrix,
        # flip it, and re-extract the upper triangular elements
        if torch.any(rc_status):
            pass  # Placeholder for RC logic
        return preds


class Cropping2D(nn.Module):
    def __init__(self, cropping_val):
        super().__init__()
        self.cropping_val = cropping_val

    def forward(self, x):
        # x: (B, C, H, W)
        crop = self.cropping_val
        return x[..., crop:-crop, crop:-crop]


# --- Building Block Modules (from modules.py) ---
class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size,
        activation_fn,
        bn_momentum,
        pool_size=0,
        dropout_rate=0.0,
    ):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv1d(
                in_channels,
                filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm1d(filters, momentum=bn_momentum))
        layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        if pool_size > 0:
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DilatedResidual1D(nn.Module):
    def __init__(
        self,
        channels,
        bottleneck_filters,
        kernel_size,
        activation_fn,
        bn_momentum,
        dilation_rate,
        dropout_rate,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, bottleneck_filters, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_filters, momentum=bn_momentum)
        self.act1 = activation_fn()

        # Dilated convolution
        self.conv2 = nn.Conv1d(
            bottleneck_filters,
            bottleneck_filters,
            kernel_size=kernel_size,
            padding=((kernel_size - 1) * dilation_rate) // 2,
            dilation=dilation_rate,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(bottleneck_filters, momentum=bn_momentum)
        self.act2 = activation_fn()

        self.conv3 = nn.Conv1d(bottleneck_filters, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels, momentum=bn_momentum)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.dropout(out)
        out += residual
        return out


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, activation_fn, bn_momentum, pool_size=0):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(filters, momentum=bn_momentum))
        layers.append(activation_fn())
        if pool_size > 0:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DilatedResidual2D(nn.Module):
    def __init__(
        self,
        channels,
        bottleneck_filters,
        kernel_size,
        activation_fn,
        bn_momentum,
        dilation_rate,
        dropout_rate,
    ):
        super().__init__()
        # Pre-activation style is common in residual blocks
        self.act_input = activation_fn()
        self.conv1 = nn.Conv2d(channels, bottleneck_filters, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_filters, momentum=bn_momentum)
        self.act1 = activation_fn()

        self.conv2 = nn.Conv2d(
            bottleneck_filters,
            bottleneck_filters,
            kernel_size=kernel_size,
            padding=((kernel_size - 1) * dilation_rate) // 2,
            dilation=dilation_rate,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_filters, momentum=bn_momentum)
        self.act2 = activation_fn()

        self.conv3 = nn.Conv2d(bottleneck_filters, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels, momentum=bn_momentum)

        self.dropout = nn.Dropout(dropout_rate)
        self.symmetrize_after_add = Symmetrize2D()

    def forward(self, x):
        residual = x

        out = self.act_input(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.dropout(out)
        out += residual
        out = self.symmetrize_after_add(out)
        return out


# --- AkitaNet (from model.py) ---
class AkitaNet(nn.Module):
    def __init__(self, model_params):
        super().__init__()

        self.seq_length = model_params["seq_length"]
        self.augment_rc_flag = model_params["augment_rc"]
        self.diagonal_offset = model_params["diagonal_offset"]
        self.bn_momentum_pytorch = 1.0 - model_params["bn_momentum"]

        activation_name = model_params.get("activation", "relu").lower()
        if activation_name == "relu":
            self.activation_fn = nn.ReLU
        elif activation_name == "gelu":
            self.activation_fn = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        self.stochastic_rc = StochasticReverseComplement()
        self.stochastic_shift = StochasticShift(shift_max=model_params["augment_shift"])

        # Build Trunk
        self.trunk_layers = nn.ModuleList()
        current_channels = 4  # Initial channels for DNA one-hot
        current_seq_len = self.seq_length

        for block_params in model_params["trunk"]:
            name = block_params["name"]
            if name == "conv_block":
                # Apply global activation before conv_block, typical for Basenji
                self.trunk_layers.append(self.activation_fn())
                layer = ConvBlock1D(
                    in_channels=current_channels,
                    filters=block_params["filters"],
                    kernel_size=block_params["kernel_size"],
                    activation_fn=self.activation_fn,
                    bn_momentum=self.bn_momentum_pytorch,
                    pool_size=block_params.get("pool_size", 0),
                )
                self.trunk_layers.append(layer)
                current_channels = block_params["filters"]
                if block_params.get("pool_size", 0) > 0:
                    current_seq_len //= block_params["pool_size"]

            elif name == "conv_tower":
                filters_init = block_params["filters_init"]
                assert current_channels == filters_init, (
                    f"ConvTower input mismatch: {current_channels} vs {filters_init}"
                )

                for i in range(block_params["repeat"]):
                    # Apply global activation before each conv_block in tower
                    self.trunk_layers.append(self.activation_fn())

                    tower_block_filters = int(
                        filters_init * (block_params.get("filters_mult", 1.0) ** i)
                    )

                    layer = ConvBlock1D(
                        in_channels=current_channels,
                        filters=tower_block_filters,
                        kernel_size=block_params["kernel_size"],
                        activation_fn=self.activation_fn,
                        bn_momentum=self.bn_momentum_pytorch,
                        pool_size=block_params["pool_size"],
                    )
                    self.trunk_layers.append(layer)
                    current_channels = tower_block_filters
                    if block_params["pool_size"] > 0:
                        current_seq_len //= block_params["pool_size"]

            elif name == "dilated_residual":
                bottleneck_f = block_params["filters"]
                dropout_p = block_params["dropout"]
                base_dilation = 1
                for i in range(block_params["repeat"]):
                    dilation = int(round(base_dilation * (block_params.get("rate_mult", 1.0) ** i)))
                    self.trunk_layers.append(self.activation_fn())
                    layer = DilatedResidual1D(
                        channels=current_channels,
                        bottleneck_filters=bottleneck_f,
                        kernel_size=3,  # Assuming kernel_size=3 for internal conv
                        activation_fn=self.activation_fn,
                        bn_momentum=self.bn_momentum_pytorch,
                        dilation_rate=dilation,
                        dropout_rate=dropout_p,
                    )
                    self.trunk_layers.append(layer)
            else:
                raise ValueError(f"Unknown trunk block name: {name}")

        self.trunk_output_channels = current_channels
        self.trunk_output_len = current_seq_len

        # Build Head (head_hic)
        self.head_layers = nn.ModuleList()
        # Input to head is output of trunk (current_channels, current_seq_len)
        # current_channels for 2D part starts from output of 1D trunk

        for block_params in model_params["head_hic"]:
            name = block_params["name"]
            if name == "one_to_two":
                layer = OneToTwo(operation=block_params.get("operation", "mean"))
                self.head_layers.append(layer)
            elif name == "concat_dist_2d":
                layer = ConcatDist2D()
                self.head_layers.append(layer)
                current_channels += 1  # Added one distance channel
            elif name == "conv_block_2d":
                # Apply global activation before conv_block_2d
                self.head_layers.append(self.activation_fn())
                layer = ConvBlock2D(
                    in_channels=current_channels,
                    filters=block_params["filters"],
                    kernel_size=block_params["kernel_size"],
                    activation_fn=self.activation_fn,
                    bn_momentum=self.bn_momentum_pytorch,
                )
                self.head_layers.append(layer)
                current_channels = block_params["filters"]
            elif name == "symmetrize_2d":
                layer = Symmetrize2D()
                self.head_layers.append(layer)
            elif name == "dilated_residual_2d":
                bottleneck_f = block_params["filters"]
                dropout_p = block_params["dropout"]
                kernel_s = block_params["kernel_size"]
                base_dilation = 1
                for i in range(block_params["repeat"]):
                    dilation = int(round(base_dilation * (block_params.get("rate_mult", 1.0) ** i)))
                    layer = DilatedResidual2D(
                        channels=current_channels,
                        bottleneck_filters=bottleneck_f,
                        kernel_size=kernel_s,
                        activation_fn=self.activation_fn,
                        bn_momentum=self.bn_momentum_pytorch,
                        dilation_rate=dilation,
                        dropout_rate=dropout_p,
                    )
                    self.head_layers.append(layer)
            elif name == "cropping_2d":
                layer = Cropping2D(cropping_val=block_params["cropping"])
                self.head_layers.append(layer)
            elif name == "upper_tri":
                layer = UpperTri(
                    diagonal_offset=block_params.get("diagonal_offset", self.diagonal_offset)
                )
                self.head_layers.append(layer)
            elif name == "final":
                # Input to dense is current_channels from UpperTri's output features
                self.final_dense = nn.Linear(current_channels, block_params["units"])
                # SwitchReverseTriu is applied after Dense
                self.switch_reverse_triu = SwitchReverseTriu(diagonal_offset=self.diagonal_offset)
            else:
                raise ValueError(f"Unknown head block name: {name}")

    def forward(self, x):
        # Augmentations
        x, rc_status = self.stochastic_rc(x, augment_rc_flag=self.augment_rc_flag)
        x = self.stochastic_shift(x)

        # Trunk
        for layer in self.trunk_layers:
            x = layer(x)

        # Head
        for layer in self.head_layers:
            x = layer(x)

        # Apply final dense and switch_reverse_triu
        x = self.final_dense(x)
        x = self.switch_reverse_triu((x, rc_status))

        return x


# Tiny config: same block TYPES, same ORDER, same architectural knobs as the
# real published configs/model_params.json ("model" sub-dict) -- only
# seq_length/filters/repeat/kernel_size magnitudes are shrunk so the trunk's
# pooling arithmetic (seq_length // pool_size per stage) and the head's
# cropping/upper-tri arithmetic stay self-consistent at a traceable size.
_TINY_MODEL_PARAMS = {
    "seq_length": 4096,
    "target_length": 512,
    "target_crop": 2,
    "diagonal_offset": 2,
    "augment_rc": True,
    "augment_shift": 3,
    "activation": "relu",
    "norm_type": "batch",
    "bn_momentum": 0.9265,
    "trunk": [
        {"name": "conv_block", "filters": 8, "kernel_size": 11, "pool_size": 2},
        {
            "name": "conv_tower",
            "filters_init": 8,
            "filters_mult": 1.0,
            "kernel_size": 5,
            "pool_size": 2,
            "repeat": 3,
        },
        {
            "name": "dilated_residual",
            "filters": 4,
            "kernel_size": 3,
            "rate_mult": 1.0,
            "repeat": 2,
            "dropout": 0.0,
        },
        {"name": "conv_block", "filters": 8, "kernel_size": 5},
    ],
    "head_hic": [
        {"name": "one_to_two", "operation": "mean"},
        {"name": "concat_dist_2d"},
        {"name": "conv_block_2d", "filters": 6, "kernel_size": 3},
        {"name": "symmetrize_2d"},
        {
            "name": "dilated_residual_2d",
            "filters": 3,
            "kernel_size": 3,
            "rate_mult": 1.0,
            "repeat": 2,
            "dropout": 0.0,
        },
        {"name": "cropping_2d", "cropping": 2},
        {"name": "upper_tri", "diagonal_offset": 2},
        {"name": "final", "units": 5, "activation": "linear"},
    ],
}


def build_akitanet():
    model = AkitaNet(_TINY_MODEL_PARAMS)
    model.eval()  # stochastic RC/shift augmentations are training-only branches
    return model


def example_input_akitanet():
    # One-hot DNA sequence: (batch, 4 bases, seq_length)
    seq_length = _TINY_MODEL_PARAMS["seq_length"]
    indices = torch.randint(0, 4, (1, seq_length))
    return F.one_hot(indices, num_classes=4).permute(0, 2, 1).float()


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Akita", "build_akitanet", "example_input_akitanet", 2020, "vendored"),
]
