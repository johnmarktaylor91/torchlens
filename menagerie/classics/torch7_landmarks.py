"""Torch7/Lua landmark architectures, 2015-2016.

Paper: Torch7-era landmark implementations: Stacked Hourglass, SoundNet, Grid LSTM, char-rnn.

This module collects compact random-initialized reimplementations of notable
Torch7/Lua-era architectures that were originally distributed as Lua/Torch
repositories and are less often available as small modern PyTorch constructors.
The implementations preserve the forward topology while reducing widths,
sequence lengths, and image/audio sizes for TorchLens tracing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """Bottleneck residual block used by the stacked hourglass pose model."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the bottleneck and skip projection.

        Parameters
        ----------
        in_channels:
            Number of input feature channels.
        out_channels:
            Number of output feature channels.
        """
        super().__init__()
        mid_channels = max(out_channels // 2, 8)
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
        )
        if in_channels == out_channels:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual bottleneck transformation.

        Parameters
        ----------
        x:
            Input feature map ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Output feature map after residual addition.
        """
        return self.main(x) + self.skip(x)


class Hourglass(nn.Module):
    """Recursive bottom-up/top-down hourglass module."""

    def __init__(self, depth: int, channels: int) -> None:
        """Initialize a compact recursive hourglass.

        Parameters
        ----------
        depth:
            Number of downsample/upsample recursions.
        channels:
            Feature channels inside the hourglass.
        """
        super().__init__()
        self.depth = depth
        self.up = ResidualBlock(channels, channels)
        self.low1 = ResidualBlock(channels, channels)
        if depth > 1:
            self.low2: nn.Module = Hourglass(depth - 1, channels)
        else:
            self.low2 = ResidualBlock(channels, channels)
        self.low3 = ResidualBlock(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Run upper skip branch plus pooled recursive lower branch.

        Parameters
        ----------
        x:
            Feature map ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Same-resolution feature map with multi-scale context.
        """
        up = self.up(x)
        low = F.max_pool2d(x, kernel_size=2, stride=2)
        low = self.low1(low)
        low = self.low2(low)
        low = self.low3(low)
        low = F.interpolate(low, size=up.shape[-2:], mode="nearest")
        return up + low


class StackedHourglassPose(nn.Module):
    """Two-stack Newell-style hourglass network for pose heatmaps."""

    def __init__(self, channels: int = 48, n_joints: int = 16, depth: int = 3) -> None:
        """Initialize stem, two hourglass stacks, and intermediate supervision.

        Parameters
        ----------
        channels:
            Compact internal channel count.
        n_joints:
            Number of predicted joint heatmaps.
        depth:
            Recursive hourglass depth.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=False),
            ResidualBlock(channels // 2, channels // 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(channels // 2, channels // 2),
            ResidualBlock(channels // 2, channels),
        )
        self.hg1 = Hourglass(depth, channels)
        self.lin1 = nn.Sequential(
            ResidualBlock(channels, channels), nn.Conv2d(channels, channels, 1)
        )
        self.out1 = nn.Conv2d(channels, n_joints, kernel_size=1)
        self.remap1 = nn.Conv2d(n_joints, channels, kernel_size=1)
        self.merge1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.hg2 = Hourglass(depth, channels)
        self.lin2 = nn.Sequential(
            ResidualBlock(channels, channels), nn.Conv2d(channels, channels, 1)
        )
        self.out2 = nn.Conv2d(channels, n_joints, kernel_size=1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict final pose heatmaps after intermediate heatmap reinjection.

        Parameters
        ----------
        image:
            RGB image tensor ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Final joint heatmaps ``(B, n_joints, H / 4, W / 4)``.
        """
        base = self.stem(image)
        feat1 = self.lin1(self.hg1(base))
        heat1 = self.out1(feat1)
        fused = base + self.merge1(feat1) + self.remap1(heat1)
        feat2 = self.lin2(self.hg2(fused))
        return self.out2(feat2)


class SoundNet(nn.Module):
    """Compact SoundNet one-dimensional waveform convolutional network."""

    def __init__(self) -> None:
        """Initialize SoundNet's temporal conv/pool hierarchy and two teacher heads."""
        super().__init__()
        specs = [
            (1, 8, 32, 2, 16, 4),
            (8, 16, 16, 2, 8, 4),
            (16, 32, 8, 2, 4, 2),
            (32, 48, 4, 2, 2, 0),
            (48, 64, 4, 2, 2, 0),
            (64, 96, 4, 2, 2, 0),
            (96, 128, 4, 2, 2, 0),
        ]
        layers: list[nn.Module] = []
        for in_channels, out_channels, kernel, stride, padding, pool in specs:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=False),
                ]
            )
            if pool:
                layers.append(nn.MaxPool1d(kernel_size=pool, stride=pool))
        self.features = nn.Sequential(*layers)
        self.imagenet_head = nn.Conv1d(128, 1000, kernel_size=1)
        self.places_head = nn.Conv1d(128, 401, kernel_size=1)

    def forward(self, waveform: Tensor) -> Tensor:
        """Map raw waveform to ImageNet and Places teacher logits.

        Parameters
        ----------
        waveform:
            Mono waveform tensor ``(B, 1, T)``.

        Returns
        -------
        Tensor
            Concatenated time-averaged teacher logits ``(B, 1401)``.
        """
        features = self.features(waveform)
        image_logits = self.imagenet_head(features).mean(dim=-1)
        place_logits = self.places_head(features).mean(dim=-1)
        return torch.cat([image_logits, place_logits], dim=1)


class GridLSTMCell(nn.Module):
    """Two-dimensional Grid LSTM cell with temporal and depth memories."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initialize temporal and depth LSTM transforms.

        Parameters
        ----------
        input_size:
            Feature width entering the depth dimension.
        hidden_size:
            Hidden and memory size for both dimensions.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.temporal = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.depth = nn.Linear(2 * hidden_size, 4 * hidden_size)

    def _lstm_update(self, gates: Tensor, cell: Tensor) -> tuple[Tensor, Tensor]:
        """Apply standard LSTM gate equations.

        Parameters
        ----------
        gates:
            Pre-activation gate tensor ``(B, 4H)``.
        cell:
            Previous cell state ``(B, H)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated hidden and cell tensors.
        """
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=-1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)
        next_cell = f_gate * cell + i_gate * g_gate
        next_hidden = o_gate * torch.tanh(next_cell)
        return next_hidden, next_cell

    def forward(
        self,
        x_depth: Tensor,
        h_time: Tensor,
        c_time: Tensor,
        c_depth: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance a Grid LSTM cell along time and depth dimensions.

        Parameters
        ----------
        x_depth:
            Input from the previous depth layer.
        h_time:
            Hidden state from the previous time step at the same depth.
        c_time:
            Temporal cell state from the previous time step.
        c_depth:
            Depth cell state from the previous layer at the same time step.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Depth hidden output, next temporal cell, and next depth cell.
        """
        h_time_next, c_time_next = self._lstm_update(
            self.temporal(torch.cat([x_depth, h_time], dim=-1)), c_time
        )
        h_depth_next, c_depth_next = self._lstm_update(
            self.depth(torch.cat([x_depth, h_time_next], dim=-1)), c_depth
        )
        return h_depth_next, c_time_next, c_depth_next


class GridLSTM(nn.Module):
    """Compact 2D Grid LSTM character model."""

    def __init__(
        self,
        vocab_size: int = 48,
        embed_size: int = 24,
        hidden_size: int = 32,
        depth: int = 3,
    ) -> None:
        """Initialize token embedding, depth-tied grid cells, and decoder.

        Parameters
        ----------
        vocab_size:
            Number of character ids.
        embed_size:
            Token embedding width.
        hidden_size:
            Grid LSTM hidden width.
        depth:
            Number of depth layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.input_proj = nn.Linear(embed_size, hidden_size)
        self.cells = nn.ModuleList([GridLSTMCell(hidden_size, hidden_size) for _ in range(depth)])
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run Grid LSTM over a token sequence.

        Parameters
        ----------
        tokens:
            Integer token ids ``(B, T)``.

        Returns
        -------
        Tensor
            Per-token logits ``(B, T, vocab_size)``.
        """
        batch, steps = tokens.shape
        h_time = [
            tokens.new_zeros(batch, self.hidden_size, dtype=torch.float32)
            for _ in range(self.depth)
        ]
        c_time = [
            tokens.new_zeros(batch, self.hidden_size, dtype=torch.float32)
            for _ in range(self.depth)
        ]
        outputs = []
        embedded = self.input_proj(self.embedding(tokens))
        for step in range(steps):
            x_depth = embedded[:, step]
            c_depth = tokens.new_zeros(batch, self.hidden_size, dtype=torch.float32)
            for layer, cell in enumerate(self.cells):
                x_depth, c_time[layer], c_depth = cell(
                    x_depth, h_time[layer], c_time[layer], c_depth
                )
                h_time[layer] = x_depth
            outputs.append(self.decoder(x_depth))
        return torch.stack(outputs, dim=1)


class CharRNNLanguageModel(nn.Module):
    """Karpathy char-rnn style multi-layer character LSTM language model."""

    def __init__(
        self,
        vocab_size: int = 64,
        embed_size: int = 32,
        hidden_size: int = 48,
        num_layers: int = 2,
    ) -> None:
        """Initialize embedding, stacked LSTM, and character decoder.

        Parameters
        ----------
        vocab_size:
            Number of character ids.
        embed_size:
            Learned character embedding width.
        hidden_size:
            LSTM hidden width.
        num_layers:
            Number of recurrent layers.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Predict next-character logits for each sequence position.

        Parameters
        ----------
        tokens:
            Integer character ids ``(B, T)``.

        Returns
        -------
        Tensor
            Log-probabilities over characters ``(B, T, vocab_size)``.
        """
        output, _ = self.rnn(self.embedding(tokens))
        return F.log_softmax(self.decoder(output), dim=-1)


def build_stacked_hourglass_pose() -> nn.Module:
    """Build a compact Stacked Hourglass pose network.

    Returns
    -------
    nn.Module
        Random-initialized Stacked Hourglass model.
    """
    return StackedHourglassPose()


def example_input_stacked_hourglass_pose() -> Tensor:
    """Return a small RGB image for Stacked Hourglass tracing.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


def build_soundnet() -> nn.Module:
    """Build a compact SoundNet model.

    Returns
    -------
    nn.Module
        Random-initialized SoundNet model.
    """
    return SoundNet()


def example_input_soundnet() -> Tensor:
    """Return a mono waveform for compact SoundNet tracing.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 8192)``.
    """
    return torch.randn(1, 1, 8192)


def build_grid_lstm() -> nn.Module:
    """Build a compact Grid LSTM character model.

    Returns
    -------
    nn.Module
        Random-initialized Grid LSTM model.
    """
    return GridLSTM()


def example_input_grid_lstm() -> Tensor:
    """Return token ids for Grid LSTM tracing.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 8)``.
    """
    return torch.randint(0, 48, (1, 8), dtype=torch.long)


def build_char_rnn() -> nn.Module:
    """Build a compact char-rnn style language model.

    Returns
    -------
    nn.Module
        Random-initialized character-level LSTM model.
    """
    return CharRNNLanguageModel()


def example_input_char_rnn() -> Tensor:
    """Return character ids for char-rnn tracing.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 12)``.
    """
    return torch.randint(0, 64, (1, 12), dtype=torch.long)


MENAGERIE_ENTRIES = [
    (
        "Stacked Hourglass Networks for Human Pose Estimation",
        "build_stacked_hourglass_pose",
        "example_input_stacked_hourglass_pose",
        "2016",
        "E5",
    ),
    ("SoundNet 8-layer audio ConvNet", "build_soundnet", "example_input_soundnet", "2016", "E5"),
    (
        "Grid LSTM character language model",
        "build_grid_lstm",
        "example_input_grid_lstm",
        "2015",
        "E5",
    ),
    ("char-rnn Torch7 character LSTM", "build_char_rnn", "example_input_char_rnn", "2015", "E5"),
]
