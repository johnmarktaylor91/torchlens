# FAITHFUL PORT of multicom-toolbox/deepdist @ master (original framework: Keras 1.x / TF1)
# https://github.com/multicom-toolbox/deepdist/blob/master/lib/Model_construct.py
# (DeepDistRes_with_paras_2D, method="mul_class_C" head, the default method in
# run_deepdist.py's --method argument)
#
# DeepDist predicts inter-residue real-valued/categorical distances from 2D
# co-evolutionary contact features. The real Keras graph
# (DeepDistRes_with_paras_2D in Model_construct.py) is:
#   1) stem: InstanceNorm -> 1x1 conv (128) -> "Maxout" block: 64 parallel
#      1x1(filters=4) conv+elu branches, each max-reduced over channels and
#      concatenated -> 64 channels
#   2) trunk: one stage of 20 repeated "dilated_bottleneck_rc" residual
#      blocks with a per-block dilation cycle [1,2,4,8,1]*4. Each block:
#      (rcin+relu) -> 1x1 conv -> (rcin+relu) -> 3x3 conv -> parallel 7x1 and
#      1x7 convs on the 3x3 output, concatenated -> (rcin+relu) -> 1x1 conv
#      -> squeeze-excite -> shortcut-add with the block input (1x1 projected
#      if channel counts differ). "rcin" = concat of instance-norm,
#      row-norm (mean/var over the row axis) and column-norm (mean/var over
#      the column axis) of the same tensor, then ReLU.
#   3) each trunk stage output has Dropout(0.2) applied, then a final
#      rcin+relu.
#   4) head (mul_class_C, the run_deepdist.py default `--method mul_class_C`
#      path): 3x3 conv -> InstanceNorm -> Dense(10, softmax) per-pixel
#      (10-way distance-bin classifier).
#
# Keras's TF1 channels-last conv layout is mapped to torch's channels-first
# layout. Keras's custom InstanceNormalization/RowNormalization/
# ColumnNormalization Layer subclasses (affine gamma/beta, no running stats)
# map to torch modules computing the identical per-sample mean/var reduction
# (over H,W for instance-norm; over H only for row-norm; over W only for
# column-norm) with learnable affine parameters. Training-only machinery
# (Keras Model.compile, the weighted/categorical losses, the L2 kernel
# regularizers) is dropped since this staging module is forward-pass-only;
# every layer in the real graph (Maxout stem, the row/col/instance "rcin"
# fusion norm, the dilated multi-kernel bottleneck block, squeeze-excite,
# the residual shortcut, and the mul_class_C classification head) is kept.

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNorm2D(nn.Module):
    """InstanceNormalization Layer (Keras): per-sample mean/var over (H,W),
    learnable affine gamma/beta, no running stats."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class RowNorm2D(nn.Module):
    """RowNormalization Layer (Keras): mean/var over axis=1 (the row/H axis
    in the original channels-last tensor)."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class ColumnNorm2D(nn.Module):
    """ColumNormalization Layer (Keras): mean/var over axis=2 (the column/W
    axis in the original channels-last tensor)."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(dim=3, keepdim=True)
        var = x.var(dim=3, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class RCInReLU(nn.Module):
    """_rcin_relu_K: concat(instance_norm(x), row_norm(x), col_norm(x)) then
    ReLU. Triples the channel count."""

    def __init__(self, channels):
        super().__init__()
        self.in_norm = InstanceNorm2D(channels)
        self.row_norm = RowNorm2D(channels)
        self.col_norm = ColumnNorm2D(channels)

    def forward(self, x):
        norm = torch.cat([self.in_norm(x), self.row_norm(x), self.col_norm(x)], dim=1)
        return F.relu(norm)


class SqueezeExcite2D(nn.Module):
    """squeeze_excite_block: global-avg-pool -> FC(reduce) -> ReLU ->
    FC(restore) -> sigmoid -> channel-wise rescale."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        reduced = max(channels // ratio, 1)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)

    def forward(self, x):
        se = x.mean(dim=(2, 3))  # global average pool -> [N, C]
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se.unsqueeze(-1).unsqueeze(-1)


class DilatedBottleneckRC(nn.Module):
    """dilated_bottleneck_rc: one residual block of the DeepDist trunk.

    is_first_block: skip the leading rcin+relu (mirrors
    is_first_block_of_first_layer in the real code, which plugs the stem
    output straight into a 1x1 conv without a preceding norm).
    """

    def __init__(self, in_channels, filters, is_first_block=False, use_se=True):
        super().__init__()
        self.is_first_block = is_first_block
        rc_channels = in_channels

        if is_first_block:
            self.conv_1_1 = nn.Conv2d(in_channels, filters, kernel_size=1)
        else:
            self.rc0 = RCInReLU(rc_channels)
            self.conv_1_1 = nn.Conv2d(rc_channels * 3, filters, kernel_size=1)

        self.rc1 = RCInReLU(filters)
        self.conv_3_3 = nn.Conv2d(filters * 3, filters, kernel_size=3, padding=1)

        self.conv_7_1 = nn.Conv2d(filters, filters, kernel_size=(7, 1), padding=(3, 0))
        self.conv_1_7 = nn.Conv2d(filters, filters, kernel_size=(1, 7), padding=(0, 3))

        self.rc2 = RCInReLU(filters * 3)
        self.conv_residual = nn.Conv2d(filters * 3 * 3, filters, kernel_size=1)

        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcite2D(filters)

        self.equal_channels = in_channels == filters
        if not self.equal_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, filters, kernel_size=1)

    def forward(self, x):
        residual_input = x

        if self.is_first_block:
            conv_1_1 = self.conv_1_1(x)
        else:
            conv_1_1 = self.rc0(x)
            conv_1_1 = self.conv_1_1(conv_1_1)

        conv_3_3 = self.rc1(conv_1_1)
        conv_3_3 = self.conv_3_3(conv_3_3)

        conv_7_1 = self.conv_7_1(conv_3_3)
        conv_1_7 = self.conv_1_7(conv_3_3)
        conv_3_3 = torch.cat([conv_3_3, conv_7_1, conv_1_7], dim=1)

        residual = self.rc2(conv_3_3)
        residual = self.conv_residual(residual)

        if self.use_se:
            residual = self.se(residual)

        shortcut = residual_input if self.equal_channels else self.shortcut_conv(residual_input)
        return shortcut + residual


class MaxoutAct2D(nn.Module):
    """MaxoutAct: `output_dim` parallel (conv -> activation) branches, each
    channel-max-reduced to a single channel, then concatenated."""

    def __init__(self, in_channels, filters, output_dim, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=padding)
                for _ in range(output_dim)
            ]
        )

    def forward(self, x):
        outs = []
        for conv in self.convs:
            act = F.elu(conv(x))
            outs.append(act.max(dim=1, keepdim=True).values)
        return torch.cat(outs, dim=1)


class DeepDist(nn.Module):
    """Faithful port of DeepDistRes_with_paras_2D (method='mul_class_C').

    Real defaults: stem projects to 128 then Maxout(filters=4, output_dim=64)
    -> 64 channels; trunk filters=64, repetitions=20 dilated bottleneck
    blocks (dilation cycle [1,2,4,8,1] repeated 4x); head projects to
    `filters` via 3x3 conv, instance-norms, then a per-pixel Dense(10,
    softmax) 10-way distance-bin classifier.
    """

    def __init__(self, in_channels, filters=64, repetitions=20, num_bins=10):
        super().__init__()
        self.stem_norm = InstanceNorm2D(in_channels)
        self.stem_conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.stem_maxout = MaxoutAct2D(128, filters=4, output_dim=64, kernel_size=1)

        blocks = []
        for i in range(repetitions):
            blocks.append(
                DilatedBottleneckRC(
                    in_channels=64 if i == 0 else filters,
                    filters=filters,
                    is_first_block=(i == 0),
                    use_se=True,
                )
            )
        self.trunk = nn.ModuleList(blocks)
        self.trunk_dropout = nn.Dropout(p=0.2)

        self.final_rc = RCInReLU(filters)

        self.head_conv = nn.Conv2d(filters * 3, filters, kernel_size=3, padding=1)
        self.head_norm = InstanceNorm2D(filters)
        self.head_dense = nn.Linear(filters, num_bins)

    def forward(self, x):
        # x: [N, C, H, W] pairwise 2D contact/co-evolution feature map.
        h = self.stem_norm(x)
        h = self.stem_conv(h)
        h = self.stem_maxout(h)

        for block in self.trunk:
            h = block(h)
        h = self.trunk_dropout(h)

        h = self.final_rc(h)

        h = self.head_conv(h)
        h = self.head_norm(h)
        h = h.permute(0, 2, 3, 1)  # NCHW -> NHWC for the per-pixel Dense head
        logits = self.head_dense(h)
        probs = torch.softmax(logits, dim=-1)
        return probs


# ---------------------------------------------------------------------------
# Menagerie staging entries
# ---------------------------------------------------------------------------


def build_deepdist():
    torch.manual_seed(0)
    model = DeepDist(in_channels=57, filters=8, repetitions=2, num_bins=10)
    model.eval()
    return model


def example_input_deepdist():
    torch.manual_seed(0)
    return (torch.randn(1, 57, 16, 16),)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    ("DeepDist", "build_deepdist", "example_input_deepdist", 2021, "ported-pytorch"),
]
