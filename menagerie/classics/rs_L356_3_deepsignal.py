# FAITHFUL PORT of bioinfomaticsCSU/deepsignal @ master (original framework: TensorFlow 1.x)
#   deepsignal/model.py (class Model) + deepsignal/layers.py
#     (rnn_layers/Event_model, inception_layer/incept_net, Fully_connected/Joint_model)
# The real repo is TF1.x graph-mode (tf.placeholder, tf.contrib.rnn LSTMCell,
# tf.variable_scope, tf.contrib.layers.batch_norm) and cannot run in the base torch env
# (tf.contrib was removed after TF 1.x; incompatible with the installed TF/torch stack).
# This ports the architecture faithfully:
#   - Event_model: 3-layer bidirectional LSTM ("brnn") over concatenated base-embedding +
#     mean/std/count-of-signal features per base position; output = concat of the last
#     forward hidden state and first backward hidden state.
#   - incept_net: a GoogLeNet/Inception-style 1D-signal CNN (conv stem -> 11 inception
#     branches with maxpool/1x1/1x3/1x5/residual branches -> avgpool -> flatten), applied
#     over the raw nanopore signal trace.
#   - Joint_model: concatenates event+signal branch outputs, 2 fully-connected layers with
#     dropout, producing class logits (methylation call, binary).
# Ported 1:1 layer-for-layer (channel counts, branch structure, kernel sizes, layer
# counts) from the real TF1.x code above; only the framework changes (TF -> torch).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class InceptionBranch(nn.Module):
    """Ports `inception_layer()` from deepsignal/layers.py: 5 parallel branches
    (maxpool+1x1, 1x1, 1x1->1x3, 1x1->1x5, residual 1x1(stem)+1x1->1x3->1x1) each
    Conv2d(bias=False) + BatchNorm2d + ReLU, concatenated on the channel axis."""

    def __init__(self, in_channels: int, times: int = 16):
        super().__init__()
        t = times
        # branch1: maxpool(1x3) -> conv 1x1 (times*3)
        self.branch1_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_conv = nn.Conv2d(in_channels, t * 3, kernel_size=(1, 1), bias=False)
        self.branch1_bn = nn.BatchNorm2d(t * 3)

        # branch2: conv 1x1 (times*3)
        self.branch2_conv = nn.Conv2d(in_channels, t * 3, kernel_size=(1, 1), bias=False)
        self.branch2_bn = nn.BatchNorm2d(t * 3)

        # branch3: conv 1x1 (times*2) -> conv 1x3 (times*3)
        self.branch3_conv0 = nn.Conv2d(in_channels, t * 2, kernel_size=(1, 1), bias=False)
        self.branch3_bn0 = nn.BatchNorm2d(t * 2)
        self.branch3_conv1 = nn.Conv2d(t * 2, t * 3, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.branch3_bn1 = nn.BatchNorm2d(t * 3)

        # branch4: conv 1x1 (times*2) -> conv 1x5 (times*3)
        self.branch4_conv0 = nn.Conv2d(in_channels, t * 2, kernel_size=(1, 1), bias=False)
        self.branch4_bn0 = nn.BatchNorm2d(t * 2)
        self.branch4_conv1 = nn.Conv2d(t * 2, t * 3, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.branch4_bn1 = nn.BatchNorm2d(t * 3)

        # branch5: residual stem (conv 1x1, times*3) + (conv1x1->conv1x3->conv1x1)
        self.branch5_stem = nn.Conv2d(in_channels, t * 3, kernel_size=(1, 1), bias=False)
        self.branch5_stem_bn = nn.BatchNorm2d(t * 3)
        self.branch5_conv0 = nn.Conv2d(in_channels, t * 2, kernel_size=(1, 1), bias=False)
        self.branch5_bn0 = nn.BatchNorm2d(t * 2)
        self.branch5_conv1 = nn.Conv2d(t * 2, t * 4, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.branch5_bn1 = nn.BatchNorm2d(t * 4)
        self.branch5_conv2 = nn.Conv2d(t * 4, t * 3, kernel_size=(1, 1), bias=False)
        self.branch5_bn2 = nn.BatchNorm2d(t * 3)

    def forward(self, x):
        # branch1: maxpool -> conv1a -> bn -> relu
        b1 = self.branch1_pool(x)
        b1 = F.relu(self.branch1_bn(self.branch1_conv(b1)))

        # branch2: conv0b -> bn -> relu
        b2 = F.relu(self.branch2_bn(self.branch2_conv(x)))

        # branch3: conv0c -> bn1 -> relu -> conv1c -> bn2 -> relu
        b3 = F.relu(self.branch3_bn0(self.branch3_conv0(x)))
        b3 = F.relu(self.branch3_bn1(self.branch3_conv1(b3)))

        # branch4: conv0d -> bn1 -> relu -> conv1d -> bn2 -> relu
        b4 = F.relu(self.branch4_bn0(self.branch4_conv0(x)))
        b4 = F.relu(self.branch4_bn1(self.branch4_conv1(b4)))

        # branch5: stem + (conv0e -> bn1 -> relu -> conv1e -> bn2 -> relu -> conv2e -> bn3)
        stem = self.branch5_stem_bn(self.branch5_stem(x))
        b5 = F.relu(self.branch5_bn0(self.branch5_conv0(x)))
        b5 = F.relu(self.branch5_bn1(self.branch5_conv1(b5)))
        b5 = self.branch5_bn2(self.branch5_conv2(b5))
        b5 = F.relu(stem + b5)

        return torch.cat([b1, b2, b3, b4, b5], dim=1)


class InceptNet(nn.Module):
    """Ports `incept_net` from deepsignal/layers.py: conv stem (3 conv+bn+relu, with a
    maxpool after the first) -> 11 InceptionBranch layers interleaved with 2 maxpools ->
    average pool -> flatten."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        # channel counts through the inception stack: 5 branches, each emits t*3 channels
        # (branch1/2/3/4 end in a t*3 conv; branch5's residual add is also t*3-wide),
        # concatenated on the channel axis -> 5 * t*3 total.
        incept_out = 16 * 3 * 5  # 240 channels with times=16
        self.incp1 = InceptionBranch(256, times=16)
        self.incp2 = InceptionBranch(incept_out, times=16)
        self.incp3 = InceptionBranch(incept_out, times=16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.incp4 = InceptionBranch(incept_out, times=16)
        self.incp5 = InceptionBranch(incept_out, times=16)
        self.incp6 = InceptionBranch(incept_out, times=16)
        self.incp7 = InceptionBranch(incept_out, times=16)
        self.incp8 = InceptionBranch(incept_out, times=16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.incp9 = InceptionBranch(incept_out, times=16)
        self.incp10 = InceptionBranch(incept_out, times=16)
        self.incp11 = InceptionBranch(incept_out, times=16)
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 7), stride=1, padding=(0, 3))

    def forward(self, signals):
        x = F.relu(self.bn1(self.conv1(signals)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.incp1(x)
        x = self.incp2(x)
        x = self.incp3(x)
        x = self.pool2(x)
        x = self.incp4(x)
        x = self.incp5(x)
        x = self.incp6(x)
        x = self.incp7(x)
        x = self.incp8(x)
        x = self.pool3(x)
        x = self.incp9(x)
        x = self.incp10(x)
        x = self.incp11(x)
        x = self.avgpool(x)
        # flatten channel * width, matching TF `tf.reshape(x, [-1, C*W])` (NHWC layout);
        # torch is NCHW, so flatten (C, W) here for the equivalent per-sample feature vec
        b = x.shape[0]
        return x.reshape(b, -1)


class EventModel(nn.Module):
    """Ports `Event_model`/`rnn_layers` from deepsignal/layers.py: 3-layer bidirectional
    LSTM ("brnn"), output = concat(last forward hidden state, first backward hidden
    state)."""

    def __init__(
        self, input_size: int, hidden_num: int = 256, layer_num: int = 3, dropout: float = 0.2
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_num,
            num_layers=layer_num,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layer_num > 1 else 0.0,
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        rnn_out, _ = self.rnn(x)
        hidden_num = self.rnn.hidden_size
        fw_out = rnn_out[:, :, :hidden_num]
        bw_out = rnn_out[:, :, hidden_num:]
        # extract_rnn_out = concat(fw_out[:, -1, :], bw_out[:, 0, :])
        extract_rnn_out = torch.cat([fw_out[:, -1, :], bw_out[:, 0, :]], dim=1)
        return extract_rnn_out


class JointModel(nn.Module):
    """Ports `Joint_model` from deepsignal/layers.py: concat event+signal branch
    outputs -> FC(same width, no bias) -> dropout -> FC(out_hidden, no bias) -> dropout."""

    def __init__(self, joint_input_dim: int, output_hidden: int, dropout_p: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(joint_input_dim, joint_input_dim, bias=False)
        self.fc2 = nn.Linear(joint_input_dim, output_hidden, bias=False)
        self.dropout_p = dropout_p

    def forward(self, event_model_output, signal_model_output):
        if signal_model_output is not None:
            if event_model_output is not None:
                joint_input = torch.cat([event_model_output, signal_model_output], dim=1)
            else:
                joint_input = signal_model_output
        else:
            joint_input = event_model_output

        fc1 = self.fc1(joint_input)
        drop1 = F.dropout(fc1, p=self.dropout_p, training=self.training)
        fc2 = self.fc2(drop1)
        drop2 = F.dropout(fc2, p=self.dropout_p, training=self.training)
        return drop2


class DeepSignalModel(nn.Module):
    """Faithful port of bioinfomaticsCSU/deepsignal's `Model` (deepsignal/model.py):
    base-embedding + Event_model (bi-LSTM) fused with incept_net (Inception CNN over
    raw nanopore signal) via Joint_model -> binary methylation-call logits."""

    def __init__(
        self,
        base_num: int = 17,
        signal_num: int = 120,
        class_num: int = 2,
        vocab_size: int = 1024,
        embedding_size: int = 128,
        hidden_num: int = 256,
        rnn_layer_num: int = 3,
    ):
        super().__init__()
        self.base_num = base_num
        self.signal_num = signal_num
        self.class_num = class_num

        self.base_embedding = nn.Embedding(vocab_size, embedding_size)
        # fusion_vector1 = concat(embedded_base, means, stds, sanums) along feature dim
        event_input_dim = embedding_size + 3
        self.event_model = EventModel(
            input_size=event_input_dim, hidden_num=hidden_num, layer_num=rnn_layer_num
        )
        self.signal_model = InceptNet(in_channels=1)
        # Joint input width: event branch outputs 2*hidden_num; signal branch width
        # depends on signal_num after 3 stride-2 downsamples -- computed lazily via a
        # probe forward at build time in `build_deepsignal()` for a concrete signal_num.
        self.join_model = None  # constructed lazily once signal branch width is known
        self._class_num = class_num

    def _ensure_join_model(self, event_dim, signal_dim, device):
        if self.join_model is None:
            self.join_model = JointModel(event_dim + signal_dim, self._class_num).to(device)

    def forward(self, base_int, means, stds, sanums, signals):
        # base_int: (batch, base_num) long
        # means/stds/sanums: (batch, base_num) float
        # signals: (batch, signal_num) float (middle-base raw signal trace)
        embedded_base = self.base_embedding(base_int)  # (batch, base_num, embed)
        fusion_vector1 = torch.cat(
            [
                embedded_base,
                means.unsqueeze(-1),
                stds.unsqueeze(-1),
                sanums.unsqueeze(-1),
            ],
            dim=2,
        )
        event_model_output = self.event_model(fusion_vector1)

        signals_reshaped = signals.reshape(signals.shape[0], 1, 1, signals.shape[1])
        signal_model_output = self.signal_model(signals_reshaped)

        self._ensure_join_model(
            event_model_output.shape[1], signal_model_output.shape[1], signals.device
        )
        logits = self.join_model(event_model_output, signal_model_output)
        activation_logits = torch.sigmoid(logits)
        return activation_logits


def build_deepsignal():
    """Tiny random-init DeepSignalModel, faithfully ported from the real TF1.x
    architecture in bioinfomaticsCSU/deepsignal (Event_model bi-LSTM + incept_net
    Inception CNN + Joint_model fusion head)."""
    torch.manual_seed(0)
    model = DeepSignalModel(
        base_num=9,
        signal_num=32,
        class_num=2,
        vocab_size=64,
        embedding_size=16,
        hidden_num=8,
        rnn_layer_num=2,
    )
    # run one forward to materialize the lazily-built join_model before returning, so
    # the module has all its parameters registered prior to being handed to the caller.
    model.eval()
    with torch.no_grad():
        model(*example_input_deepsignal())
    return model


def example_input_deepsignal():
    torch.manual_seed(0)
    batch_size, base_num, signal_num, vocab_size = 2, 9, 32, 64
    base_int = torch.randint(0, vocab_size, (batch_size, base_num))
    means = torch.randn(batch_size, base_num)
    stds = torch.rand(batch_size, base_num)
    sanums = torch.rand(batch_size, base_num)
    signals = torch.randn(batch_size, signal_num)
    return (base_int, means, stds, sanums, signals)


MENAGERIE_ENTRIES = [
    (
        "DeepSignal",
        build_deepsignal,
        example_input_deepsignal,
        2019,
        "REIMPLEMENT",
    ),
]
