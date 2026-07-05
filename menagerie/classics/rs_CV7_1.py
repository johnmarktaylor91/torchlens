# SOURCE: vendored from IDKiro/CBDNet-pytorch @ 09a2e55, ZitongYu/CDCN @ fd8370e, apourchot/CEM-RL @ 1a45822, MingLunHan/CIF-PyTorch @ 48c581e
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class CbdSingleConv(nn.Module):
    """Vendored CBDNet single convolution block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize the single convolution block."""
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block forward pass."""
        return self.conv(x)


class CbdUp(nn.Module):
    """Vendored CBDNet upsampling block."""

    def __init__(self, in_ch: int) -> None:
        """Initialize the transposed-convolution upsampler."""
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample and align two feature maps."""
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        return x2 + x1


class CbdOutConv(nn.Module):
    """Vendored CBDNet output convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize the output projection."""
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project feature maps to output channels."""
        return self.conv(x)


class CbdFcn(nn.Module):
    """Vendored CBDNet noise-estimation FCN."""

    def __init__(self) -> None:
        """Initialize the CBDNet FCN."""
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate per-pixel noise level."""
        return self.fcn(x)


class CbdUNet(nn.Module):
    """Vendored CBDNet non-blind denoising U-Net."""

    def __init__(self) -> None:
        """Initialize the CBDNet U-Net."""
        super().__init__()
        self.inc = nn.Sequential(CbdSingleConv(6, 64), CbdSingleConv(64, 64))
        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            CbdSingleConv(64, 128), CbdSingleConv(128, 128), CbdSingleConv(128, 128)
        )
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            CbdSingleConv(128, 256),
            CbdSingleConv(256, 256),
            CbdSingleConv(256, 256),
            CbdSingleConv(256, 256),
            CbdSingleConv(256, 256),
            CbdSingleConv(256, 256),
        )
        self.up1 = CbdUp(256)
        self.conv3 = nn.Sequential(
            CbdSingleConv(128, 128), CbdSingleConv(128, 128), CbdSingleConv(128, 128)
        )
        self.up2 = CbdUp(128)
        self.conv4 = nn.Sequential(CbdSingleConv(64, 64), CbdSingleConv(64, 64))
        self.outc = CbdOutConv(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise an image concatenated with its estimated noise level."""
        inx = self.inc(x)
        conv1 = self.conv1(self.down1(inx))
        conv2 = self.conv2(self.down2(conv1))
        conv3 = self.conv3(self.up1(conv2, conv1))
        conv4 = self.conv4(self.up2(conv3, inx))
        return self.outc(conv4)


class CbdNetwork(nn.Module):
    """Vendored CBDNet noise-estimation and denoising network."""

    def __init__(self) -> None:
        """Initialize CBDNet."""
        super().__init__()
        self.fcn = CbdFcn()
        self.unet = CbdUNet()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CBDNet and return estimated noise and denoised image."""
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        return noise_level, self.unet(concat_img) + x


class Conv2dCd(nn.Module):
    """Vendored CDCN central-difference convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        theta: float = 0.7,
    ) -> None:
        """Initialize central-difference convolution."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run normal convolution minus the central-difference response."""
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out_normal
        kernel_diff = self.conv.weight.sum(2).sum(2)[:, :, None, None]
        out_diff = F.conv2d(
            input=x,
            weight=kernel_diff,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
            groups=self.conv.groups,
        )
        return out_normal - self.theta * out_diff


class CDCN(nn.Module):
    """Vendored Central Difference Convolutional Network."""

    def __init__(self, theta: float = 0.7) -> None:
        """Initialize CDCN."""
        super().__init__()
        basic_conv = Conv2dCd
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.lastconv1 = nn.Sequential(
            basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode="bilinear")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run CDCN and return depth map plus intermediate features."""
        x_input = x
        x = self.conv1(x)
        x_block1 = self.block1(x)
        x_block1_32x32 = self.downsample32x32(x_block1)
        x_block2 = self.block2(x_block1)
        x_block2_32x32 = self.downsample32x32(x_block2)
        x_block3 = self.block3(x_block2)
        x_block3_32x32 = self.downsample32x32(x_block3)
        x_concat = torch.cat((x_block1_32x32, x_block2_32x32, x_block3_32x32), dim=1)
        x = self.lastconv1(x_concat)
        x = self.lastconv2(x)
        x = self.lastconv3(x)
        map_x = x.squeeze(1)
        return map_x, x_concat, x_block1, x_block2, x_block3, x_input


class CemActor(nn.Module):
    """Vendored CEM-RL actor network."""

    def __init__(
        self, state_dim: int, action_dim: int, max_action: float, layer_norm: bool = False
    ) -> None:
        """Initialize the CEM-RL actor."""
        super().__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the actor forward pass."""
        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
        return self.max_action * torch.tanh(self.l3(x))


class CifMiddleware(nn.Module):
    """Vendored Continuous Integrate-and-Fire middleware."""

    def __init__(self, cfg: SimpleNamespace) -> None:
        """Initialize CIF middleware."""
        super().__init__()
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.conv_cif_width = cfg.conv_cif_width
        self.conv_cif_dropout = cfg.conv_cif_dropout
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling
        self.tail_handling_firing_threshold = cfg.tail_handling_firing_threshold
        if self.produce_weight_type == "dense":
            self.dense_proj = cif_linear(self.encoder_embed_dim, self.encoder_embed_dim)
            self.weight_proj = cif_linear(self.encoder_embed_dim, 1)
        elif self.produce_weight_type == "conv":
            self.conv = nn.Conv1d(
                self.encoder_embed_dim,
                self.encoder_embed_dim,
                self.conv_cif_width,
                stride=1,
                padding=int(self.conv_cif_width / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            )
            self.conv_dropout = nn.Dropout(p=self.conv_cif_dropout)
            self.weight_proj = cif_linear(self.encoder_embed_dim, 1)
        else:
            self.weight_proj = cif_linear(self.encoder_embed_dim, 1)
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = cif_linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False
            )

    def forward(
        self, encoder_outputs: dict[str, torch.Tensor], target_lengths: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Run continuous integrate-and-fire over encoder outputs."""
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]
        encoder_padding_mask = encoder_outputs["encoder_padding_mask"]
        device = encoder_raw_outputs.device
        if self.produce_weight_type == "dense":
            sig_input = self.weight_proj(torch.relu(self.dense_proj(encoder_raw_outputs)))
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            proj_input = self.conv(conv_input).permute(0, 2, 1)
            sig_input = self.weight_proj(self.conv_dropout(proj_input))
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
        weight = torch.sigmoid(sig_input)
        not_padding_mask = ~encoder_padding_mask
        weight = torch.squeeze(weight, dim=-1) * not_padding_mask.int()
        org_weight = weight
        if self.training and self.apply_scaling and target_lengths is not None:
            weight_sum = weight.sum(-1)
            normalize_scalar = torch.unsqueeze(target_lengths / weight_sum, -1)
            weight = weight * normalize_scalar
        batch_size = encoder_raw_outputs.size(0)
        max_length = encoder_raw_outputs.size(1)
        encoder_embed_dim = encoder_raw_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)
        accumulated_weights = torch.zeros(batch_size, 0, device=device)
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim, device=device)
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim, device=device)
        for i in range(max_length):
            prev_accumulated_weight = (
                torch.zeros([batch_size], device=device)
                if i == 0
                else accumulated_weights[:, i - 1]
            )
            prev_accumulated_state = (
                torch.zeros([batch_size, encoder_embed_dim], device=device)
                if i == 0
                else accumulated_states[:, i - 1, :]
            )
            cur_is_fired = (
                (prev_accumulated_weight + weight[:, i]) >= self.cif_threshold
            ).unsqueeze(dim=-1)
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
            remained_weight = (
                torch.ones_like(prev_accumulated_weight, device=device) - prev_accumulated_weight
            )
            cur_accumulated_weight = torch.where(
                cur_is_fired, cur_weight - remained_weight, cur_weight + prev_accumulated_weight
            )
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_outputs[:, i, :],
            )
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim], device=device),
            )
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i, device=device)
                > padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim], device=device),
                cur_fired_state,
            )
            accumulated_weights = torch.cat((accumulated_weights, cur_accumulated_weight), 1)
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1
            )
            fired_states = torch.cat((fired_states, torch.unsqueeze(cur_fired_state, 1)), 1)
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()
        fired_utt_length = fired_marks.sum(-1)
        fired_max_length = int(fired_utt_length.max().item())
        cif_outputs = torch.zeros([0, fired_max_length, encoder_embed_dim], device=device)
        for j in range(batch_size):
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = cif_dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = cur_utt_outputs[1]
            cur_utt_length = cur_utt_output.size(0)
            pad_length = fired_max_length - cur_utt_length
            cur_utt_output = torch.cat(
                (cur_utt_output, torch.full([pad_length, encoder_embed_dim], 0.0, device=device)),
                dim=0,
            )
            cif_outputs = torch.cat([cif_outputs, torch.unsqueeze(cur_utt_output, 0)], 0)
        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        quantity_out = org_weight.sum(-1) if self.training else weight.sum(-1)
        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)
        return {
            "cif_out": cif_outputs,
            "cif_out_padding_mask": cif_out_padding_mask,
            "quantity_out": quantity_out,
        }


class CifTraceWrapper(nn.Module):
    """Single-input wrapper for the vendored CIF middleware."""

    def __init__(self) -> None:
        """Initialize the wrapper with a tiny CIF configuration."""
        super().__init__()
        cfg = SimpleNamespace(
            cif_threshold=1.0,
            cif_embedding_dim=8,
            encoder_embed_dim=8,
            produce_weight_type="dense",
            conv_cif_width=3,
            conv_cif_dropout=0.0,
            apply_scaling=True,
            apply_tail_handling=False,
            tail_handling_firing_threshold=0.5,
        )
        self.cif = CifMiddleware(cfg)

    def forward(self, encoder_raw_out: torch.Tensor) -> torch.Tensor:
        """Run CIF with an all-valid padding mask and target lengths."""
        batch_size = encoder_raw_out.shape[0]
        encoder_outputs = {
            "encoder_raw_out": encoder_raw_out,
            "encoder_padding_mask": torch.zeros(
                batch_size,
                encoder_raw_out.shape[1],
                dtype=torch.bool,
                device=encoder_raw_out.device,
            ),
        }
        target_lengths = torch.full((batch_size,), 2.0, device=encoder_raw_out.device)
        return self.cif(encoder_outputs, target_lengths)["cif_out"]


def cif_linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    """Create a CIF linear layer using the source initialization."""
    layer = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.constant_(layer.bias, 0.0)
    return layer


def cif_dynamic_partition(
    data: torch.Tensor, partitions: torch.Tensor, num_partitions: int | None = None
) -> list[torch.Tensor]:
    """Partition a tensor according to integer labels as in the CIF source."""
    if num_partitions is None:
        num_partitions = int(max(torch.unique(partitions)).item())
    return [data[partitions == index] for index in range(num_partitions)]


def build_cbdnet() -> nn.Module:
    """Build the vendored CBDNet model."""
    return CbdNetwork()


def example_input_cbdnet() -> torch.Tensor:
    """Return an example CBDNet image."""
    return torch.randn(1, 3, 32, 32)


def build_cdcn() -> nn.Module:
    """Build the vendored CDCN model."""
    return CDCN()


def example_input_cdcn() -> torch.Tensor:
    """Return an example CDCN image."""
    return torch.randn(1, 3, 64, 64)


def build_cem_rl() -> nn.Module:
    """Build the vendored CEM-RL actor."""
    return CemActor(state_dim=8, action_dim=3, max_action=1.0)


def example_input_cem_rl() -> torch.Tensor:
    """Return an example CEM-RL state vector."""
    return torch.randn(2, 8)


def build_cif() -> nn.Module:
    """Build the vendored CIF middleware wrapper."""
    return CifTraceWrapper()


def example_input_cif() -> torch.Tensor:
    """Return an example CIF encoder output."""
    return torch.randn(1, 4, 8)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("CBDNet", "build_cbdnet", "example_input_cbdnet", 2018, "CV7-192"),
    ("CDCN", "build_cdcn", "example_input_cdcn", 2020, "CV7-194"),
    ("CEM-RL", "build_cem_rl", "example_input_cem_rl", 2019, "CV7-200"),
    ("CIF (Continuous Integrate-and-Fire)", "build_cif", "example_input_cif", 2020, "CV7-220"),
]
