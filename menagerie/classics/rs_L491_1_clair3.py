# SOURCE: vendored from HKU-BAL/Clair3 @ main (clair3/model.py)
#
# Clair3 is HKU-BAL's official small-variant caller: a dual-model design that jointly
# scores a pileup summary (Bi-LSTM, `Clair3_P`) and a full-alignment tensor (residual CNN
# + pyramid pooling, `Clair3_F`). Moved from TensorFlow to native PyTorch upstream; the
# classes below are copied verbatim from clair3/model.py aside from replacing the repo's
# `clair3.task.main` / `shared.param_f` / `shared.param_p` imports with the same
# label-count / channel-size constants those modules define (pure integers derived from
# clair3/task/gt21.py, clair3/task/variant_length.py, shared/param_f.py, shared/param_p.py
# -- no architecture logic lives in those modules).
#
# `Clair3_FB` (a third class present verbatim in the same upstream file: a lightweight
# full-alignment "Bloom filter" residual-CNN + self-attention model) is also transcribed
# below for completeness but is NOT registered as a menagerie entry: as committed upstream
# it has a real shape bug (concat of the 160-channel residual features with the 96-channel
# attention map yields a 512-wide pooled vector, but `dense1 = nn.Linear(96 * 2, 128)`
# expects 192) that makes forward() crash on any input. This is a genuine defect in
# HKU-BAL's actual repo code, not a transcription error here -- left disabled rather than
# "fixed" so the staged module doesn't silently diverge from the real source.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# ---------------------------------------------------------------------------
# Constants inlined from clair3/task/main.py + clair3/task/variant_length.py
# (GT21.output_label_count=21, GENOTYPE.output_label_count=3, VariantLength
# .output_label_count = variant_length_index_offset*2+1 = 16*2+1 = 33).
# ---------------------------------------------------------------------------
GT21_OUTPUT_LABEL_COUNT = 21
GENOTYPE_OUTPUT_LABEL_COUNT = 3
VARIANT_LENGTH_OUTPUT_LABEL_COUNT = 33

# Constants inlined from shared/param_p.py (pileup input: channel_size=18,
# flankingBaseNum=16 -> no_of_positions=33) and shared/param_f.py
# (full-alignment input: channel_size=8, flankingBaseNum=16 -> no_of_positions=33,
# matrix_depth_dict['hifi']=55, NORMALIZE_NUM=100).
PILEUP_CHANNEL_SIZE = 18
PILEUP_NO_OF_POSITIONS = 33
FULL_ALIGNMENT_CHANNEL_SIZE = 8
FULL_ALIGNMENT_NO_OF_POSITIONS = 33
FULL_ALIGNMENT_HIFI_DEPTH = 55
NORMALIZE_NUM = 100

params = dict(
    L3_dropout_rate=0.2,
    L4_num_units=256,
    L4_pileup_num_units=128,
    L4_dropout_rate=0.5,
    L5_1_num_units=128,
    L5_1_dropout_rate=0.2,
    L5_2_num_units=128,
    L5_2_dropout_rate=0.2,
    L5_3_num_units=128,
    L5_3_dropout_rate=0.2,
    L5_4_num_units=128,
    L5_4_dropout_rate=0.2,
    LSTM1_num_units=128,
    LSTM2_num_units=160,
    LSTM1_dropout_rate=0,
    LSTM2_dropout_rate=0.5,
)


def _infer_channels(default_channels, override):
    return override if override is not None else default_channels


class Clair3_P(nn.Module):
    """Bi-LSTM model for Clair3 pileup input."""

    def __init__(self, add_indel_length=False, predict=False, input_channels=None):
        super().__init__()

        self.output_gt21_shape = GT21_OUTPUT_LABEL_COUNT
        self.output_genotype_shape = GENOTYPE_OUTPUT_LABEL_COUNT
        self.output_indel_length_shape_1 = VARIANT_LENGTH_OUTPUT_LABEL_COUNT
        self.output_indel_length_shape_2 = VARIANT_LENGTH_OUTPUT_LABEL_COUNT

        self.L3_dropout_rate = params["L3_dropout_rate"]
        self.L4_pileup_num_units = params["L4_pileup_num_units"]
        self.L4_dropout_rate = params["L4_dropout_rate"]
        self.L5_1_num_units = params["L5_1_num_units"]
        self.L5_1_dropout_rate = params["L5_1_dropout_rate"]
        self.L5_2_num_units = params["L5_2_num_units"]
        self.L5_2_dropout_rate = params["L5_2_dropout_rate"]
        self.L5_3_num_units = params["L5_3_num_units"]
        self.L5_3_dropout_rate = params["L5_3_dropout_rate"]
        self.L5_4_num_units = params["L5_4_num_units"]
        self.L5_4_dropout_rate = params["L5_4_dropout_rate"]
        self.LSTM1_num_units = params["LSTM1_num_units"]
        self.LSTM2_num_units = params["LSTM2_num_units"]

        self.output_label_split = [
            self.output_gt21_shape,
            self.output_genotype_shape,
            self.output_indel_length_shape_1,
            self.output_indel_length_shape_2,
        ]

        self.add_indel_length = add_indel_length
        self.predict = predict

        no_of_positions = PILEUP_NO_OF_POSITIONS
        input_channels = _infer_channels(PILEUP_CHANNEL_SIZE, input_channels)

        self.LSTM1 = nn.LSTM(
            input_size=input_channels,
            hidden_size=self.LSTM1_num_units,
            batch_first=True,
            bidirectional=True,
        )
        self.LSTM2 = nn.LSTM(
            input_size=self.LSTM1_num_units * 2,
            hidden_size=self.LSTM2_num_units,
            batch_first=True,
            bidirectional=True,
        )

        self.L3_dropout = nn.Dropout(p=self.L3_dropout_rate)
        self.L4 = nn.Linear(self.LSTM2_num_units * 2 * no_of_positions, self.L4_pileup_num_units)
        self.L4_dropout = nn.Dropout(p=self.L4_dropout_rate)
        self.L5_1 = nn.Linear(self.L4_pileup_num_units, self.L5_1_num_units)
        self.L5_1_dropout = nn.Dropout(p=self.L5_1_dropout_rate)
        self.L5_2 = nn.Linear(self.L4_pileup_num_units, self.L5_2_num_units)
        self.L5_2_dropout = nn.Dropout(p=self.L5_2_dropout_rate)
        self.Y_gt21_logits = nn.Linear(self.L5_1_num_units, self.output_gt21_shape)
        self.Y_genotype_logits = nn.Linear(self.L5_2_num_units, self.output_genotype_shape)

        if self.add_indel_length:
            self.L5_3 = nn.Linear(self.L4_pileup_num_units, self.L5_3_num_units)
            self.L5_3_dropout = nn.Dropout(p=self.L5_3_dropout_rate)
            self.L5_4 = nn.Linear(self.L4_pileup_num_units, self.L5_4_num_units)
            self.L5_4_dropout = nn.Dropout(p=self.L5_4_dropout_rate)
            self.Y_indel_length_logits_1 = nn.Linear(
                self.L5_3_num_units, self.output_indel_length_shape_1
            )
            self.Y_indel_length_logits_2 = nn.Linear(
                self.L5_4_num_units, self.output_indel_length_shape_2
            )

        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.SELU()

    def forward(self, x):
        x = x.float()
        x, _ = self.LSTM1(x)
        x, _ = self.LSTM2(x)
        x = self.L3_dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.L4(x))
        x = self.L4_dropout(x)

        l5_1_dropout = self.L5_1_dropout(self.activation(self.L5_1(x)))
        l5_2_dropout = self.L5_2_dropout(self.activation(self.L5_2(x)))

        y_gt21_logits = self.softmax(self.activation(self.Y_gt21_logits(l5_1_dropout)))
        y_genotype_logits = self.softmax(self.activation(self.Y_genotype_logits(l5_2_dropout)))

        if self.add_indel_length:
            l5_3_dropout = self.L5_3_dropout(self.activation(self.L5_3(x)))
            l5_4_dropout = self.L5_4_dropout(self.activation(self.L5_4(x)))

            y_indel_length_logits_1 = self.softmax(
                self.activation(self.Y_indel_length_logits_1(l5_3_dropout))
            )
            y_indel_length_logits_2 = self.softmax(
                self.activation(self.Y_indel_length_logits_2(l5_4_dropout))
            )

            if self.predict:
                return torch.cat(
                    [
                        y_gt21_logits,
                        y_genotype_logits,
                        y_indel_length_logits_1,
                        y_indel_length_logits_2,
                    ],
                    dim=1,
                )
            return [
                y_gt21_logits,
                y_genotype_logits,
                y_indel_length_logits_1,
                y_indel_length_logits_2,
            ]

        if self.predict:
            return torch.cat([y_gt21_logits, y_genotype_logits], dim=1)

        return [y_gt21_logits, y_genotype_logits]


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class BasicConv2D(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, padding, separable=False):
        super().__init__()
        if separable:
            conv = SeparableConv2d(in_channels, filters, kernel_size, strides, padding)
        else:
            conv = nn.Conv2d(
                in_channels,
                filters,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                bias=True,
            )
        self.conv = conv
        self.bn = nn.BatchNorm2d(filters, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.bn(output)
        return self.relu(output)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, filter_num, stride=1, separable=False):
        super().__init__()
        if separable:
            conv1 = SeparableConv2d(
                in_channels, filter_num, kernel_size=3, stride=stride, padding=1
            )
            conv2 = SeparableConv2d(filter_num, filter_num, kernel_size=3, stride=1, padding=1)
        else:
            conv1 = nn.Conv2d(
                in_channels, filter_num, kernel_size=3, stride=stride, padding=1, bias=True
            )
            conv2 = nn.Conv2d(filter_num, filter_num, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1 = conv1
        self.bn1 = nn.BatchNorm2d(filter_num, eps=1e-3)
        self.conv2 = conv2
        self.bn2 = nn.BatchNorm2d(filter_num, eps=1e-3)

        if stride != 1 or in_channels != filter_num:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, filter_num, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(filter_num, eps=1e-3),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        output = self.relu(residual + x)
        return output


def make_basic_block_layer(in_channels, filter_num, blocks, stride=1, separable=False):
    layers = [BasicBlock(in_channels, filter_num, stride=stride, separable=separable)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(filter_num, filter_num, stride=1, separable=separable))
    return nn.Sequential(*layers)


class PyramidPolling(nn.Module):
    def __init__(self, spatial_pool_size=(3, 2, 1)):
        super().__init__()
        self.spatial_pool_size = spatial_pool_size

    def forward(self, x):
        pooled = []
        height = x.shape[-2]
        width = x.shape[-1]
        for pool_size in self.spatial_pool_size:
            window_h = int(np.ceil(height / pool_size))
            window_w = int(np.ceil(width / pool_size))
            stride_h = window_h
            stride_w = window_w
            out_h = int(np.ceil(height / stride_h))
            out_w = int(np.ceil(width / stride_w))
            pad_h = max((out_h - 1) * stride_h + window_h - height, 0)
            pad_w = max((out_w - 1) * stride_w + window_w - width, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            if pad_h or pad_w:
                x_padded = F.pad(
                    x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
                )
            else:
                x_padded = x
            max_pool = F.max_pool2d(
                x_padded,
                kernel_size=(window_h, window_w),
                stride=(stride_h, stride_w),
            )
            # Match TF NHWC flatten ordering for the dense layer weights.
            max_pool = max_pool.permute(0, 2, 3, 1)
            pooled.append(torch.flatten(max_pool, start_dim=1))
        return torch.cat(pooled, dim=1)


class Clair3_F(nn.Module):
    """Residual CNN model for Clair3 full-alignment input."""

    def __init__(self, add_indel_length=False, predict=False, input_channels=None):
        super().__init__()
        self.output_gt21_shape = GT21_OUTPUT_LABEL_COUNT
        self.output_genotype_shape = GENOTYPE_OUTPUT_LABEL_COUNT
        self.output_indel_length_shape_1 = VARIANT_LENGTH_OUTPUT_LABEL_COUNT
        self.output_indel_length_shape_2 = VARIANT_LENGTH_OUTPUT_LABEL_COUNT

        self.L3_dropout_rate = params["L3_dropout_rate"]
        self.L4_num_units = params["L4_num_units"]
        self.L4_dropout_rate = params["L4_dropout_rate"]
        self.L5_1_num_units = params["L5_1_num_units"]
        self.L5_1_dropout_rate = params["L5_1_dropout_rate"]
        self.L5_2_num_units = params["L5_2_num_units"]
        self.L5_2_dropout_rate = params["L5_2_dropout_rate"]
        self.L5_3_num_units = params["L5_3_num_units"]
        self.L5_3_dropout_rate = params["L5_3_dropout_rate"]
        self.L5_4_num_units = params["L5_4_num_units"]
        self.L5_4_dropout_rate = params["L5_4_dropout_rate"]

        self.output_label_split = [
            self.output_gt21_shape,
            self.output_genotype_shape,
            self.output_indel_length_shape_1,
            self.output_indel_length_shape_2,
        ]

        self.add_indel_length = add_indel_length
        self.predict = predict

        input_channels = _infer_channels(FULL_ALIGNMENT_CHANNEL_SIZE, input_channels)
        self.input_channels = input_channels

        self.conv1 = BasicConv2D(
            in_channels=input_channels,
            filters=64,
            kernel_size=(3, 3),
            strides=2,
            padding=1,
        )
        self.res_block1 = make_basic_block_layer(64, filter_num=64, blocks=1, stride=1)

        self.conv3 = BasicConv2D(
            in_channels=64,
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding=1,
        )
        self.res_block2 = make_basic_block_layer(128, filter_num=128, blocks=1, stride=1)

        self.conv5 = BasicConv2D(
            in_channels=128,
            filters=256,
            kernel_size=(3, 3),
            strides=2,
            padding=1,
        )
        self.res_block3 = make_basic_block_layer(256, filter_num=256, blocks=1, stride=1)

        self.pyramidpolling = PyramidPolling()
        self.L3_dropout = nn.Dropout(p=self.L3_dropout_rate)

        self.L4 = nn.Linear(self._infer_fc_input_dim(), self.L4_num_units)
        self.L4_dropout = nn.Dropout(p=self.L4_dropout_rate)

        self.L5_1 = nn.Linear(self.L4_num_units, self.L5_1_num_units)
        self.L5_1_dropout = nn.Dropout(p=self.L5_1_dropout_rate)

        self.L5_2 = nn.Linear(self.L4_num_units, self.L5_2_num_units)
        self.L5_2_dropout = nn.Dropout(p=self.L5_2_dropout_rate)

        self.Y_gt21_logits = nn.Linear(self.L5_1_num_units, self.output_gt21_shape)
        self.Y_genotype_logits = nn.Linear(self.L5_2_num_units, self.output_genotype_shape)

        if self.add_indel_length:
            self.L5_3 = nn.Linear(self.L4_num_units, self.L5_3_num_units)
            self.L5_3_dropout = nn.Dropout(p=self.L5_3_dropout_rate)
            self.L5_4 = nn.Linear(self.L4_num_units, self.L5_4_num_units)
            self.L5_4_dropout = nn.Dropout(p=self.L5_4_dropout_rate)
            self.Y_indel_length_logits_1 = nn.Linear(
                self.L5_3_num_units, self.output_indel_length_shape_1
            )
            self.Y_indel_length_logits_2 = nn.Linear(
                self.L5_4_num_units, self.output_indel_length_shape_2
            )

        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.SELU()

    def _infer_fc_input_dim(self):
        pooled_sizes = np.array([3, 2, 1], dtype=int)
        pooled = (pooled_sizes**2).sum()
        return pooled * 256

    def forward(self, inputs):
        x = inputs.float() / NORMALIZE_NUM
        if x.ndim == 4 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.conv5(x)
        x = self.res_block3(x)
        x = self.pyramidpolling(x)
        x = self.L3_dropout(x)

        x = self.activation(self.L4(x))
        x = self.L4_dropout(x)

        l5_1_dropout = self.L5_1_dropout(self.activation(self.L5_1(x)))
        l5_2_dropout = self.L5_2_dropout(self.activation(self.L5_2(x)))

        y_gt21_logits = self.softmax(self.activation(self.Y_gt21_logits(l5_1_dropout)))
        y_genotype_logits = self.softmax(self.activation(self.Y_genotype_logits(l5_2_dropout)))

        if self.add_indel_length:
            l5_3_dropout = self.L5_3_dropout(self.activation(self.L5_3(x)))
            l5_4_dropout = self.L5_4_dropout(self.activation(self.L5_4(x)))

            y_indel_length_logits_1 = self.softmax(
                self.activation(self.Y_indel_length_logits_1(l5_3_dropout))
            )
            y_indel_length_logits_2 = self.softmax(
                self.activation(self.Y_indel_length_logits_2(l5_4_dropout))
            )

            if self.predict:
                return torch.cat(
                    [
                        y_gt21_logits,
                        y_genotype_logits,
                        y_indel_length_logits_1,
                        y_indel_length_logits_2,
                    ],
                    dim=1,
                )
            return [
                y_gt21_logits,
                y_genotype_logits,
                y_indel_length_logits_1,
                y_indel_length_logits_2,
            ]

        if self.predict:
            return torch.cat([y_gt21_logits, y_genotype_logits], dim=1)

        return [y_gt21_logits, y_genotype_logits]


class Clair3_FB(nn.Module):
    """Lightweight full-alignment Bloom Filter style model."""

    def __init__(self, predict=False, input_channels=None):
        super().__init__()
        self.predict = predict

        input_channels = _infer_channels(FULL_ALIGNMENT_CHANNEL_SIZE, input_channels)
        self.input_channels = input_channels

        self.conv1 = BasicConv2D(
            in_channels=input_channels,
            filters=48,
            kernel_size=(3, 3),
            strides=2,
            padding=1,
            separable=True,
        )
        self.res1 = BasicBlock(48, filter_num=48, stride=1, separable=True)

        self.conv2 = BasicConv2D(
            in_channels=48,
            filters=96,
            kernel_size=(3, 3),
            strides=2,
            padding=1,
            separable=True,
        )
        self.res2 = BasicBlock(96, filter_num=96, stride=1, separable=True)

        self.conv3 = BasicConv2D(
            in_channels=96,
            filters=160,
            kernel_size=(3, 3),
            strides=2,
            padding=1,
            separable=True,
        )
        self.res3 = BasicBlock(160, filter_num=160, stride=1, separable=True)

        self.attn_proj = nn.Conv2d(160, 96, kernel_size=1, padding=0, bias=True)
        self.attn_dropout = nn.Dropout(p=0.2)
        self.attn_mha = nn.MultiheadAttention(
            embed_dim=96, num_heads=4, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(96, eps=1e-5)

        self.dropout1 = nn.Dropout(p=0.35)
        self.dense1 = nn.Linear(96 * 2, 128)
        self.dropout2 = nn.Dropout(p=0.25)
        self.dense2 = nn.Linear(128, 96)

        self.logit = nn.Linear(96, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.SELU()

    def forward(self, inputs):
        x = inputs.float() / NORMALIZE_NUM
        if x.ndim == 4 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)

        attn_input = self.activation(self.attn_proj(x))
        batch_size, channels, height, width = attn_input.shape
        seq = attn_input.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        attn_out, _ = self.attn_mha(seq, seq, seq)
        attn_out = self.attn_dropout(attn_out)
        seq = self.attn_norm(seq + attn_out)
        attn_map = seq.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        x = torch.cat([x, attn_map], dim=1)

        pooled_avg = x.mean(dim=(2, 3))
        pooled_max = x.amax(dim=(2, 3))
        pooled = torch.cat([pooled_avg, pooled_max], dim=1)

        x = self.dropout1(pooled)
        x = self.activation(self.dense1(x))
        x = self.dropout2(x)
        x = self.activation(self.dense2(x))

        prob = self.sigmoid(self.logit(x))

        if self.predict:
            return prob
        return [prob]


# ---------------------------------------------------------------------------
# Menagerie staging entrypoints (tiny random-init builds + example inputs).
# ---------------------------------------------------------------------------
def build_clair3_p():
    return Clair3_P(add_indel_length=True, predict=True)


def example_input_clair3_p():
    # (batch, no_of_positions, channel_size) pileup tensor.
    return torch.randn(1, PILEUP_NO_OF_POSITIONS, PILEUP_CHANNEL_SIZE)


def build_clair3_f():
    return Clair3_F(add_indel_length=True, predict=True)


def example_input_clair3_f():
    # (batch, depth, no_of_positions, channel_size) full-alignment tensor, NHWC as in the
    # original TF-derived layout; forward() permutes to NCHW internally.
    return torch.randn(
        1, FULL_ALIGNMENT_HIFI_DEPTH, FULL_ALIGNMENT_NO_OF_POSITIONS, FULL_ALIGNMENT_CHANNEL_SIZE
    )


# NOTE: Clair3_FB is transcribed above verbatim from the real repo for completeness, but
# is intentionally NOT wired into MENAGERIE_ENTRIES -- see header note (upstream shape bug
# makes its forward() crash unconditionally; not something to paper over here).


MENAGERIE_ENTRIES = [
    ("Clair3-Pileup", build_clair3_p, example_input_clair3_p, 2021, "vendored-pytorch"),
    ("Clair3-FullAlignment", build_clair3_f, example_input_clair3_f, 2021, "vendored-pytorch"),
]
