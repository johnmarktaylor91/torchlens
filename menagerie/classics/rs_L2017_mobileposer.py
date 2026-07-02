# SOURCE: vendored from SPICExLAB/MobilePoser @ 513e76a88ef310e12b51da6d30f44a819e68d331
# https://github.com/SPICExLAB/MobilePoser/blob/main/mobileposer/models/rnn.py
#
# MobilePoser (arXiv:2504.12492) predicts full-body SMPL pose from a small set of
# IMU sensors (phone/watch/earbuds) in real time. The full MobilePoserNet stack wraps
# each sub-network in a PyTorch-Lightning module together with an SMPL body model
# (`art.model.ParametricModel(paths.smpl_file, ...)`), which requires a licensed SMPL
# `.pkl` asset that is not obtainable via pip and is not distributed with the repo, so
# the LightningModule wrappers cannot be constructed in a stock env. The actual learned
# architecture MobilePoser uses for every one of its four sub-tasks (joint position
# estimation, pose estimation, foot-contact probability, and root velocity) is the same
# `RNN` class below (a 2-layer bidirectional LSTM sandwiched between linear layers) --
# vendored verbatim from mobileposer/models/rnn.py. This staging module instantiates
# the four real sub-networks (Joints/Poser/FootContact/Velocity) at the exact tensor
# dimensions the authors configure in mobileposer/config.py, with the same
# hidden-size / layer-count / bidirectionality per sub-task -- only the SMPL-dependent
# LightningModule wrapper (loss, kinematics, checkpoint I/O) is left out, since it adds
# no additional architecture beyond calling this exact RNN.

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MENAGERIE_ZOO = "vendored-pytorch"


class RNN(nn.Module):
    """
    A RNN Module including a linear input layer, a RNN and a linear output layer.
    Vendored verbatim from mobileposer/models/rnn.py.
    """

    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.4):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=n_hidden,
            hidden_size=n_hidden,
            num_layers=n_rnn_layer,
            bidirectional=bidirectional,
        )
        self.linear1 = nn.Linear(in_features=n_input, out_features=n_hidden)
        self.linear2 = nn.Linear(
            in_features=n_hidden * (2 if bidirectional else 1), out_features=n_output
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seq_lengths=None, h=None):
        # pass input data through a linear layer
        data = self.dropout(relu(self.linear1(x)))
        # pack the padded sequences
        if seq_lengths is not None:
            data = pack_padded_sequence(data, seq_lengths, batch_first=True, enforce_sorted=False)
        # pass input to RNN
        data, h = self.rnn(data, h)
        # pack the padded sequences
        output_lengths = None
        if seq_lengths is not None:
            data, output_lengths = pad_packed_sequence(data, batch_first=True)
        data = self.linear2(data)
        return data, output_lengths, h


# --- real config.py dimensions (mobileposer/config.py) ---
# n_joints = 5; n_imu = 12 * n_joints = 60
# n_output_joints = 24
# joint_set.n_reduced = len([0,1,2,3,4,5,6,9,12,13,14,15,16,17,18,19]) = 16
N_IMU = 60
N_OUTPUT_JOINTS = 24
N_REDUCED = 16


class JointsNet(nn.Module):
    """Real Joints sub-network: RNN(n_imu, 24*3, 256), bidirectional (default)."""

    def __init__(self):
        super().__init__()
        self.joints = RNN(N_IMU, N_OUTPUT_JOINTS * 3, 256)

    def forward(self, batch):
        joints, _, _ = self.joints(batch)
        return joints


class PoserNet(nn.Module):
    """Real Poser sub-network: RNN(n_output_joints*3 + n_imu, n_reduced*6, 256), bidirectional (default)."""

    def __init__(self):
        super().__init__()
        self.pose = RNN(N_OUTPUT_JOINTS * 3 + N_IMU, N_REDUCED * 6, 256)

    def forward(self, batch):
        pred_pose, _, _ = self.pose(batch)
        return pred_pose


class FootContactNet(nn.Module):
    """Real FootContact sub-network: RNN(n_output_joints*3 + n_imu, 2, 64), bidirectional (default)."""

    def __init__(self):
        super().__init__()
        self.footcontact = RNN(N_OUTPUT_JOINTS * 3 + N_IMU, 2, 64)

    def forward(self, batch):
        foot_contact, _, _ = self.footcontact(batch)
        return foot_contact


class VelocityNet(nn.Module):
    """Real Velocity sub-network: RNN(n_output_joints*3 + n_imu, 24*3, 256, bidirectional=False)."""

    def __init__(self):
        super().__init__()
        self.vel = RNN(N_OUTPUT_JOINTS * 3 + N_IMU, N_OUTPUT_JOINTS * 3, 256, bidirectional=False)

    def forward(self, batch):
        vel, _, _ = self.vel(batch)
        return vel


def build_mobileposer_joints():
    return JointsNet().eval()


def example_input_mobileposer_joints():
    # [batch, seq_len, n_imu]
    return torch.randn(1, 8, N_IMU)


def build_mobileposer_poser():
    return PoserNet().eval()


def example_input_mobileposer_poser():
    return torch.randn(1, 8, N_OUTPUT_JOINTS * 3 + N_IMU)


def build_mobileposer_footcontact():
    return FootContactNet().eval()


def example_input_mobileposer_footcontact():
    return torch.randn(1, 8, N_OUTPUT_JOINTS * 3 + N_IMU)


def build_mobileposer_velocity():
    return VelocityNet().eval()


def example_input_mobileposer_velocity():
    return torch.randn(1, 8, N_OUTPUT_JOINTS * 3 + N_IMU)


MENAGERIE_ENTRIES = [
    (
        "MobilePoser-Joints",
        build_mobileposer_joints,
        example_input_mobileposer_joints,
        2025,
        MENAGERIE_ZOO,
    ),
    (
        "MobilePoser-Poser",
        build_mobileposer_poser,
        example_input_mobileposer_poser,
        2025,
        MENAGERIE_ZOO,
    ),
    (
        "MobilePoser-FootContact",
        build_mobileposer_footcontact,
        example_input_mobileposer_footcontact,
        2025,
        MENAGERIE_ZOO,
    ),
    (
        "MobilePoser-Velocity",
        build_mobileposer_velocity,
        example_input_mobileposer_velocity,
        2025,
        MENAGERIE_ZOO,
    ),
]
