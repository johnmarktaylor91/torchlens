# FAITHFUL PORT of smousavi05/CRED @ master (cred_utils.py: block_CNN, block_BiLSTM,
# model_cred; cred_original.py: model_cred((151, 41, 3), filters=[8, 16, 32, 64, 128, 256]))
# (original framework: TF1-era standalone Keras + tensorflow-gpu, cudatoolkit==9.0 -- not
# reasonably installable alongside the installed torch/torch-ecosystem env)
#
# CRED (Mousavi et al., 2019, "Earthquake Transformer"-predecessor "CRED: A Deep Residual
# Network of Convolutional and Recurrent Units for Earthquake Signal Detection") is a
# seismic-waveform CNN+RNN residual detector. A spectrogram-like input tensor
# (time, freq, channels) passes through two strided Conv2D-residual-CNN stages (each a
# strided Conv2D projection plus a BatchNorm-ReLU-Conv2D-BatchNorm-ReLU-Conv2D residual
# block, summed with the projection), is reshaped to a (time, features) sequence, passed
# through a 2-layer residual bidirectional-LSTM stack, a unidirectional LSTM, and a
# per-timestep (TimeDistributed) Dense-Dense classification head producing a sigmoid
# P-pick probability at every output timestep. Ported faithfully layer-for-layer from the
# real Keras functional-API graph in `model_cred`/`block_CNN`/`block_BiLSTM`; the only
# non-architectural changes are Keras -> torch API translation (`Conv2D(... 'same')` ->
# explicit `nn.Conv2d` with computed same-padding, `TimeDistributed(Dense)` -> `nn.Linear`
# applied over the last dim which is the timestep-shared behavior TimeDistributed provides,
# `l1` weight regularizers dropped since they only affect the training loss not the forward
# architecture, and Keras "channels_last" (N,H,W,C) tensors mapped to torch's Conv2d
# "channels_first" (N,C,H,W) with an explicit permute at the model boundary).
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def _same_pad(kernel_size: int) -> int:
    # Keras Conv2D padding="same" with stride 1 or 2 and odd kernel size is equivalent to
    # torch's symmetric padding = (kernel_size - 1) // 2 for these odd kernel sizes.
    return (kernel_size - 1) // 2


class BlockCNN(nn.Module):
    """Port of cred_utils.block_CNN: BN-ReLU-Conv-BN-ReLU-Conv residual branch."""

    def __init__(self, in_channels: int, filters: int, ker: int):
        super().__init__()
        k = ker - 2
        pad = _same_pad(k)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=k, padding=pad)
        self.bn2 = nn.BatchNorm2d(filters)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=k, padding=pad)

    def forward(self, x):
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x


class ResidualCNNStage(nn.Module):
    """Port of the `conv2D_N` / `res_conv_N` pattern in model_cred: a strided Conv2D
    projection, summed with a block_CNN residual branch computed on top of the projection.
    """

    def __init__(self, in_channels: int, filters: int, ker: int, stride: int):
        super().__init__()
        pad = _same_pad(ker)
        self.proj = nn.Conv2d(in_channels, filters, kernel_size=ker, stride=stride, padding=pad)
        self.proj_act = nn.ReLU()
        self.res_block = BlockCNN(filters, filters, ker)

    def forward(self, x):
        proj = self.proj(x)
        proj = self.proj_act(proj)
        res = self.res_block(proj)
        return proj + res


class BlockBiLSTM(nn.Module):
    """Port of cred_utils.block_BiLSTM: rnn_depth stacked bidirectional LSTMs, with a
    residual add on every layer after the first (matching `if i > 0: x = add([x, x_rnn])`).
    """

    def __init__(self, input_size: int, filters: int, rnn_depth: int):
        super().__init__()
        self.rnn_depth = rnn_depth
        self.lstms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(rnn_depth):
            in_size = input_size if i == 0 else 2 * filters
            self.lstms.append(nn.LSTM(in_size, filters, batch_first=True, bidirectional=True))
            self.dropouts.append(nn.Dropout(0.7))

    def forward(self, x):
        for i in range(self.rnn_depth):
            x_rnn, _ = self.lstms[i](x)
            x_rnn = self.dropouts[i](x_rnn)
            if i > 0:
                x = x + x_rnn
            else:
                x = x_rnn
        return x


class TimeDistributedDense(nn.Module):
    """Port of Keras TimeDistributed(Dense(...)): applies the same Linear across the time
    dimension. `nn.Linear` already broadcasts over leading dims, which is exactly this
    behavior for a (batch, time, features) tensor.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class CRED(nn.Module):
    """Port of cred_utils.model_cred (Keras functional-API graph)."""

    def __init__(self, in_channels: int = 3, filters=(8, 16, 32, 64, 128, 256)):
        super().__init__()
        f = filters
        self.stage2 = ResidualCNNStage(in_channels, f[0], ker=9, stride=2)
        self.stage3 = ResidualCNNStage(f[0], f[1], ker=5, stride=2)

        # reshaped = Reshape((time, freq * channels)) after stage3 -- feature dim depends
        # on the spatial (freq) size after the two stride-2 convs, resolved lazily below.
        self._bilstm_filters = f[3]
        self.bilstm = None  # constructed on first forward once the reshaped size is known
        self._bilstm_in_channels = f[1]

        self.uni_lstm = nn.LSTM(2 * f[3], f[3], batch_first=True)
        self.uni_dropout = nn.Dropout(0.8)
        self.uni_bn = nn.BatchNorm1d(f[3])

        self.dense2 = TimeDistributedDense(f[3], f[3])
        self.dense2_bn = nn.BatchNorm1d(f[3])
        self.dense2_dropout = nn.Dropout(0.8)
        self.dense2_act = nn.ReLU()

        self.dense3 = TimeDistributedDense(f[3], 1)
        self.dense3_act = nn.Sigmoid()

    def _build_bilstm(self, feature_dim: int, device):
        self.bilstm = BlockBiLSTM(feature_dim, self._bilstm_filters, rnn_depth=2).to(device)

    def forward(self, x):
        # x: (batch, time, freq, channels) Keras "channels_last" -> torch "channels_first"
        x = x.permute(0, 3, 1, 2)

        x = self.stage2(x)
        x = self.stage3(x)

        # x: (batch, filters[1], time', freq') -> reshape to (batch, time', freq' * filters[1])
        b, c, t, freq = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, t, freq * c)

        if self.bilstm is None:
            self._build_bilstm(freq * c, x.device)

        x = self.bilstm(x)

        x, _ = self.uni_lstm(x)
        x = self.uni_dropout(x)
        # BatchNorm1d normalizes over the channel dim -> transpose (batch, time, feat) to
        # (batch, feat, time) around the norm, matching Keras BatchNormalization's default
        # feature-axis normalization on a (batch, time, feat) tensor.
        x = self.uni_bn(x.transpose(1, 2)).transpose(1, 2)

        x = self.dense2(x)
        x = self.dense2_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.dense2_dropout(x)
        x = self.dense2_act(x)

        x = self.dense3(x)
        x = self.dense3_act(x)
        return x


def build_cred():
    torch.manual_seed(0)
    model = CRED(in_channels=3, filters=(4, 8, 16, 16, 16, 16))
    # Materialize the lazily-built BlockBiLSTM sub-module with one warmup forward pass so
    # the returned module's parameters are fully instantiated before tracing.
    model.eval()
    with torch.no_grad():
        model(example_input_cred()[0])
    return model


def example_input_cred():
    # Real usage feeds a (151, 41, 3) time x frequency x channel spectrogram-like tensor
    # (cred_original.py: `model_cred((151, 41, 3), filters=...)`); shrunk here to keep the
    # trace small while preserving the same rank/semantics.
    torch.manual_seed(0)
    return (torch.randn(2, 38, 11, 3),)


MENAGERIE_ENTRIES = [
    ("CRED_seismic", "build_cred", "example_input_cred", 2019, "ported-pytorch"),
]
