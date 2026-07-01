# SOURCE: vendored from volpato30/DeepNovoV2 @ DeepNovoV2 branch
# (model.py, the PointNovo codebase referenced by the queue row "PointNovo")
#
# Original file:
#   https://raw.githubusercontent.com/volpato30/DeepNovoV2/DeepNovoV2/model.py
#
# The real repo's `model.py` imports a module-level `deepnovo_config` that runs
# `argparse.ArgumentParser().parse_args()` at import time (it is a CLI training
# script's config, not an importable constants module) -- that would consume/crash
# on this harness's own argv. To vendor the *model* faithfully without dragging in
# a script that parses sys.argv, the small set of scalar constants model.py actually
# reads from deepnovo_config (vocab_size, num_ion, num_units, embedding_size,
# num_lstm_layers, lstm_hidden_units, dropout_rate, distance_scale_factor, use_lstm)
# are inlined below with the exact values the real deepnovo_config.py assigns
# (verified against https://raw.githubusercontent.com/volpato30/DeepNovoV2/DeepNovoV2/deepnovo_config.py).
# The nn.Module classes (TNet, DeepNovoPointNet, InitNet, DeepNovoPointNetWithLSTM)
# are transcribed verbatim; only `deepnovo_config.X` references were rewritten to
# the inlined constants of the same name, and the unused Direction/InferenceModelWrapper
# inference-only helper classes were dropped since they are not part of the traced
# forward architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

activation_func = F.relu

# --- inlined deepnovo_config constants (values copied verbatim from the real repo) ---
vocab_size = 29  # len(_START_VOCAB) + len(vocab_reverse) == 3 + 26
num_ion = 12
num_units = 64
embedding_size = 512
num_lstm_layers = 1
lstm_hidden_units = 512
dropout_rate = 0.25
distance_scale_factor = 100.0
use_lstm = True


class TNet(nn.Module):
    """
    the T-net structure in the Point Net paper
    """

    def __init__(self, with_lstm=False):
        super(TNet, self).__init__()
        self.with_lstm = with_lstm
        self.conv1 = nn.Conv1d(vocab_size * num_ion + 1, num_units, 1)
        self.conv2 = nn.Conv1d(num_units, 2 * num_units, 1)
        self.conv3 = nn.Conv1d(2 * num_units, 4 * num_units, 1)
        self.fc1 = nn.Linear(4 * num_units, 2 * num_units)
        self.fc2 = nn.Linear(2 * num_units, num_units)
        if not with_lstm:
            self.output_layer = nn.Linear(num_units, vocab_size)
        self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(vocab_size * num_ion + 1)

        self.bn1 = nn.BatchNorm1d(num_units)
        self.bn2 = nn.BatchNorm1d(2 * num_units)
        self.bn3 = nn.BatchNorm1d(4 * num_units)
        self.bn4 = nn.BatchNorm1d(2 * num_units)
        self.bn5 = nn.BatchNorm1d(num_units)

    def forward(self, x):
        """

        :param x: [batch * T, 26*8+1, N]
        :return:
            logit: [batch * T, 26]
        """
        x = self.input_batch_norm(x)
        x = activation_func(self.bn1(self.conv1(x)))
        x = activation_func(self.bn2(self.conv2(x)))
        x = activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4 * num_units

        x = activation_func(self.bn4(self.fc1(x)))
        x = activation_func(self.bn5(self.fc2(x)))
        if not self.with_lstm:
            x = self.output_layer(x)  # [batch * T, 26]
        return x


class DeepNovoPointNet(nn.Module):
    def __init__(self):
        super(DeepNovoPointNet, self).__init__()
        self.t_net = TNet(with_lstm=False)
        self.distance_scale_factor = distance_scale_factor

    def forward(self, location_index, peaks_location, peaks_intensity):
        """

        :param location_index: [batch, T, 26, 8] long
        :param peaks_location: [batch, N] N stands for MAX_NUM_PEAK, long
        :param peaks_intensity: [batch, N], float32
        :return:
            logits: [batch, T, 26]
        """

        N = peaks_location.size(1)
        assert N == peaks_intensity.size(1)
        batch_size, T, vocab_size_, num_ion_ = location_index.size()

        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)
        peaks_location = peaks_location.expand(-1, T, -1, -1)  # [batch, T, N, 1]
        peaks_location_mask = (peaks_location > 1e-5).float()
        peaks_intensity = peaks_intensity.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        location_index = location_index.view(batch_size, T, 1, vocab_size_ * num_ion_)
        location_index_mask = (location_index > 1e-5).float()

        location_exp_minus_abs_diff = torch.exp(
            -torch.abs((peaks_location - location_index) * self.distance_scale_factor)
        )
        # [batch, T, N, 26*8]

        location_exp_minus_abs_diff = (
            location_exp_minus_abs_diff * peaks_location_mask * location_index_mask
        )

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)
        input_feature = input_feature.view(batch_size * T, N, vocab_size_ * num_ion_ + 1)
        input_feature = input_feature.transpose(1, 2)

        result = self.t_net(input_feature).view(batch_size, T, vocab_size_)
        return result


class InitNet(nn.Module):
    def __init__(self):
        super(InitNet, self).__init__()
        self.init_state_layer = nn.Linear(embedding_size, 2 * lstm_hidden_units)

    def forward(self, spectrum_representation):
        """

        :param spectrum_representation: [N, embedding_size]
        :return:
            [num_lstm_layers, batch_size, lstm_units], [num_lstm_layers, batch_size, lstm_units],
        """
        x = torch.tanh(self.init_state_layer(spectrum_representation))
        h_0, c_0 = torch.split(x, lstm_hidden_units, dim=1)
        h_0 = torch.unsqueeze(h_0, dim=0)
        h_0 = h_0.repeat(num_lstm_layers, 1, 1)
        c_0 = torch.unsqueeze(c_0, dim=0)
        c_0 = c_0.repeat(num_lstm_layers, 1, 1)
        return h_0, c_0


class DeepNovoPointNetWithLSTM(nn.Module):
    def __init__(self):
        super(DeepNovoPointNetWithLSTM, self).__init__()
        self.t_net = TNet(with_lstm=True)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, lstm_hidden_units, num_layers=num_lstm_layers, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(num_units + lstm_hidden_units, vocab_size)

    def forward(
        self, location_index, peaks_location, peaks_intensity, aa_input=None, state_tuple=None
    ):
        """

        :param location_index: [batch, T, 26, 8] long
        :param peaks_location: [batch, N] N stands for MAX_NUM_PEAK, long
        :param peaks_intensity: [batch, N], float32
        :param aa_input:[batch, T]
        :param state_tuple: (h0, c0), where each is [num_lstm_layer, batch_size, num_units] tensor
        :return:
            logits: [batch, T, 26]
        """
        assert aa_input is not None
        N = peaks_location.size(1)
        assert N == peaks_intensity.size(1)
        batch_size, T, vocab_size_, num_ion_ = location_index.size()

        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)
        peaks_location = peaks_location.expand(-1, T, -1, -1)  # [batch, T, N, 1]
        peaks_location_mask = (peaks_location > 1e-5).float()
        peaks_intensity = peaks_intensity.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        location_index = location_index.view(batch_size, T, 1, vocab_size_ * num_ion_)
        location_index_mask = (location_index > 1e-5).float()

        location_exp_minus_abs_diff = torch.exp(
            -torch.abs((peaks_location - location_index) * distance_scale_factor)
        )
        # [batch, T, N, 26*8]

        location_exp_minus_abs_diff = (
            location_exp_minus_abs_diff * peaks_location_mask * location_index_mask
        )

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)
        input_feature = input_feature.view(batch_size * T, N, vocab_size_ * num_ion_ + 1)
        input_feature = input_feature.transpose(1, 2)

        ion_feature = self.t_net(input_feature).view(batch_size, T, num_units)  # attention on peaks

        # embedding
        aa_embedded = self.embedding(aa_input)
        lstm_input = aa_embedded  # [batch, T, embedding_size]
        # lstm_input = self.dropout(lstm_input)
        output_feature, new_state_tuple = self.lstm(lstm_input, state_tuple)
        output_feature = torch.cat((ion_feature, activation_func(output_feature)), dim=2)
        output_feature = self.dropout(output_feature)
        logit = self.output_layer(output_feature)
        return logit, new_state_tuple


DeepNovoModel = DeepNovoPointNetWithLSTM if use_lstm else DeepNovoPointNet


# --- staging entry points ----------------------------------------------------


def build_pointnovo():
    return DeepNovoPointNetWithLSTM()


def example_input_pointnovo():
    batch_size, T, N = 2, 4, 10
    location_index = torch.rand(batch_size, T, vocab_size, num_ion)
    peaks_location = torch.rand(batch_size, N)
    peaks_intensity = torch.rand(batch_size, N)
    aa_input = torch.randint(0, vocab_size, (batch_size, T))
    return (location_index, peaks_location, peaks_intensity, aa_input)


MENAGERIE_ENTRIES = [
    (
        "PointNovo",
        build_pointnovo,
        example_input_pointnovo,
        2019,
        "SOURCE_AVAILABLE",
    ),
]
