# SOURCE: vendored from https://github.com/ndming/NSRT @ main
#
# NSRT community implementation of "Neural Supersampling for Real-time
# Rendering" (Xiao et al., SIGGRAPH 2020) -- no official author release
# exists; ndming/NSRT is a from-the-paper PyTorch reimplementation with an
# active repo (train.py/model/*.py). Vendored verbatim from model/nsrt.py
# (the actual nn.Module graph: current-frame convolution, a gated-convolution
# "reliable warping" context encoder over the temporal-history frames +
# per-frame motion masks, a frame-recurrent ConvLSTM feature fusion, and a
# U-Net with residual channel-attention (RCAB) blocks that reconstructs the
# super-resolved frame via pixel-shuffle upsampling). Only the unrelated
# training-utility imports (adabelief_pytorch/OpenEXR/data loading in
# utils.py/data.py/step.py) are dropped -- the model file itself imports
# only torch/torch.nn/torch.nn.functional.
import torch
import torch.nn as nn
import torch.nn.functional as F


class NSRT(nn.Module):
    def __init__(self, frame_channels, upscale_factor, context_length, conv_features):
        super().__init__()
        self.frame_channels = frame_channels
        self.context_length = context_length

        self.curr_convolution = nn.Conv2d(
            frame_channels, conv_features, kernel_size=3, padding="same"
        )
        self.reliable_warping = ReliableWarping(frame_channels, context_length, conv_features)
        self.reconstruction = FrameRecurrentReconstruction(
            context_length, upscale_factor, conv_features, out_channels=6
        )

    def forward(self, x, motion_masks, warped_logits, lstm_state):
        # x: current and all previous frame inputs in the context  (B, frame_channels * context_length, H, W)
        # motion_masks: motion masks of all previous frames        (B, context_length - 1,              H, W)
        # warped_logits: warped logits from the previous frame     (B, 6, H * upscale_factor, W * upscale_factor)

        x_current = self.curr_convolution(
            x[:, : self.frame_channels, ...]
        )  # (B, n_features,                        H, W)
        x_context = self.reliable_warping(
            x, motion_masks
        )  # (B, n_features * (context_length - 1), H, W)
        logits, state = self.reconstruction(x_current, x_context, warped_logits, lstm_state)
        return logits, state


class ReliableWarping(nn.Module):
    def __init__(self, frame_channels, context_length, n_features):
        super().__init__()
        self.frame_channels = frame_channels
        self.context_length = context_length

        # Previous frames are concatenated with their motion mask, hence frame_channels + 1
        self.gated_conv = GatedConvolution(frame_channels + 1, n_features, kernel_size=3)

    def forward(self, x, motion_masks):
        # The parent model will pass parameters as is, so we need to extract the relevant tensors
        # x: current and all previous frame inputs in the context  (B, frame_channels * context_length, H, W)
        # motion_masks: motion masks of all previous frames        (B, context_length - 1,              H, W)

        x_context = []
        for i in range(1, self.context_length):  # skip the current frame
            x_prev = x[:, i * self.frame_channels : (i + 1) * self.frame_channels, ...]
            x_mask = motion_masks[:, i - 1, ...].unsqueeze(1)
            x_gate = torch.cat([x_prev, x_mask], dim=1)
            x_gate = self.gated_conv(x_gate)
            x_context.append(x_gate)
        return torch.cat(x_context, dim=1)


class GatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # Initialize of the convolutional and gating layers with He initialization
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.kaiming_normal_(self.gate.weight)

    def forward(self, x):
        x = self.activation(self.conv(x)) * F.sigmoid(self.gate(x))
        x = self.batch_norm(x)
        return x


class FrameRecurrentReconstruction(nn.Module):
    def __init__(self, context_length, upscale_factor, n_features, out_channels):
        super().__init__()

        self.frame_recurrent = nn.Sequential(
            nn.PixelUnshuffle(upscale_factor),
            nn.Conv2d(
                out_channels * upscale_factor * upscale_factor,
                n_features,
                kernel_size=3,
                padding="same",
            ),
        )
        self.conv_lstm = ConvLSTM(n_features * context_length + n_features, 64, kernel_size=3)
        self.unet_rcab = UNetRCAB(64, out_channels, upscale_factor, n_features)

    def forward(self, x_current, x_context, warped_logits, lstm_state):
        # x_current: high-level features of the current frame                   (B, n_features,                        H, W)
        # x_context: gated features of all previous frames                      (B, n_features * (context_length - 1), H, W)
        # warped_logits: wraped previously reconstructed diffuse and speecular  (B, 6, H * upscale_factor, W * upscale_factor)

        x = self.frame_recurrent(
            warped_logits
        )  # (B, n_features,                               H, W)
        x = torch.cat(
            [x_current, x_context, x], dim=1
        )  # (B, n_features * context_length + n_features, H, W)
        x_lstm = self.conv_lstm(x, lstm_state)  # (hidden, cell)
        logits = self.unet_rcab(x_lstm[0])  # (B, 6, H * upscale_factor, W * upscale_factor)
        return logits, x_lstm


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Conv2d(
            input_size + hidden_size, 4 * hidden_size, kernel_size, padding="same"
        )

    def forward(self, x, prev_state):
        batch_size, _, height, width = x.shape
        hidden, cell = (
            self.init_state(batch_size, height, width) if prev_state is None else prev_state
        )

        stack = torch.cat([x, hidden], dim=1)
        gates = self.gates(stack)

        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        return hidden, cell

    def init_state(self, batch_size, height, width):
        return (
            torch.zeros(batch_size, self.hidden_size, height, width).to(self.gates.weight.device),
            torch.zeros(batch_size, self.hidden_size, height, width).to(self.gates.weight.device),
        )


class UNetRCAB(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, n_features):
        super().__init__()

        self.encoder_0 = nn.Sequential(
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
        )
        self.encoder_1 = nn.Sequential(
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
        )
        self.center = nn.Sequential(
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
        )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding="same"),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
        )
        self.decoder_0 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding="same"),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
            ResidualChannelAttention(in_channels, kernel_size=3, reduction=16),
        )

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.bilinear_upsampling = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.upsampling = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding="same"),
            nn.Conv2d(
                n_features,
                out_channels * upscale_factor * upscale_factor,
                kernel_size=3,
                padding="same",
            ),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        x_0 = self.encoder_0(x)
        x_1 = self.encoder_1(self.pooling(x_0))
        x = self.center(self.pooling(x_1))
        x = UNetRCAB._align_tensor(x_1, self.bilinear_upsampling(x))
        x = self.decoder_1(torch.cat([x, x_1], dim=1))
        x = UNetRCAB._align_tensor(x_0, self.bilinear_upsampling(x))
        x = self.decoder_0(torch.cat([x, x_0], dim=1))
        x = self.upsampling(x)
        return x

    @staticmethod
    def _align_tensor(actual, target):
        diffY = int(actual.size()[2] - target.size()[2])
        diffX = int(actual.size()[3] - target.size()[3])
        x = F.pad(target, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x


class ResidualChannelAttention(nn.Module):
    def __init__(self, in_channels, kernel_size, reduction):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(in_channels),
            ChannelAttention(in_channels, reduction),
        )

    def forward(self, x):
        x = x + self.residual(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)  # global average pooling
        self.conv = nn.Sequential(  # feature transformation
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


# --- staging harness: build + example input ---------------------------------


class NSRTTraceWrapper(nn.Module):
    """Flattens the real NSRT.forward's `lstm_state` tuple param into two
    plain tensor args so a flat-tensor example_input can drive tl.trace();
    the real (unmodified) NSRT module is called with the tuple re-assembled,
    exercising the identical forward path.
    """

    def __init__(self, nsrt):
        super().__init__()
        self.nsrt = nsrt

    def forward(self, x, motion_masks, warped_logits, lstm_hidden, lstm_cell):
        logits, (hidden, cell) = self.nsrt(x, motion_masks, warped_logits, (lstm_hidden, lstm_cell))
        return logits, hidden, cell


def build_nsrt():
    # Shrunk from typical real usage (upscale_factor=4, conv_features>=32,
    # context_length up to 5) to a tiny architecturally faithful config.
    nsrt = NSRT(frame_channels=3, upscale_factor=2, context_length=3, conv_features=8)
    return NSRTTraceWrapper(nsrt).eval()


def example_input_nsrt():
    batch, frame_channels, upscale_factor, context_length = 1, 3, 2, 3
    h, w = 16, 16
    x = torch.rand(batch, frame_channels * context_length, h, w)
    motion_masks = torch.rand(batch, context_length - 1, h, w)
    warped_logits = torch.rand(batch, 6, h * upscale_factor, w * upscale_factor)
    # ConvLSTM.init_state's own zero-init (used internally whenever
    # prev_state is None); materialized here as concrete tensors so the
    # traced example exercises the real forward path.
    lstm_hidden_size = 64
    lstm_hidden = torch.zeros(batch, lstm_hidden_size, h, w)
    lstm_cell = torch.zeros(batch, lstm_hidden_size, h, w)
    return (x, motion_masks, warped_logits, lstm_hidden, lstm_cell)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("NSRT-NeuralSupersampling", build_nsrt, example_input_nsrt, 2020, MENAGERIE_ZOO),
]
