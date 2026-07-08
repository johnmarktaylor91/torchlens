# SOURCE: vendored from geonchoi/Split-KalmanNet @ main
# https://raw.githubusercontent.com/geonchoi/Split-KalmanNet/main/GSSFiltering/dnn.py
#
# Split-KalmanNet (Choi et al., IEEE Trans. Vehicular Technology 2023 / arXiv:2210.09636,
# "Split-KalmanNet: A Robust Model-Based Deep Learning Approach for State Estimation")
# -- official reference PyTorch implementation of the `DNN_SKalmanNet_GSS` module, the
# real trainable network behind "Split-KalmanNet". Unlike the baseline KalmanNet (single
# GRU regressing the Kalman gain directly), Split-KalmanNet SPLITS gain estimation into
# two independently-trained GRU branches: one regresses the process-noise-driven
# state-covariance term (Pk, from state-innovation/diff/linearization-error/Jacobian
# features), the other regresses the observation-covariance term (Sk, from
# observation-innovation/diff/linearization-error/Jacobian features); Pk and Sk are the
# two building blocks the outer (non-learned) Kalman-gain computation combines, hence
# "split".
#
# Transcribed verbatim from GSSFiltering/dnn.py's `DNN_SKalmanNet_GSS` class: every
# nn.Linear/nn.GRU layer, its widths (derived from x_dim/y_dim and the config-driven
# H1/H2/gru_hidden_dim scale factors), and the forward() logic (concatenating the four
# innovation/diff/error/Jacobian feature blocks, running each through its own
# Linear->ReLU->GRU->Linear->ReLU->Linear head) is unchanged. The only change is
# resolving the two module-level `configparser.read('./config.ini')`-derived constants
# (`nGRU`, `gru_scale_s`) to their literal values from the repo's own checked-in
# `config.ini` `[DNN.size]` section (`nGRU = 2`, `gru_scale_s = 2` -- the "general"
# NCLT/SyntheticNL setting, not the commented-out time-varying alternative) instead of
# reading a config file at import time, since this module is used standalone; the
# `gru_scale_k` constant belongs to the separate (non-split) `DNN_KalmanNet_GSS` class
# and is not needed here.

import torch
import torch.nn as nn

# From the repo's config.ini [DNN.size] section (general NCLT / SyntheticNL setting):
#   nGRU = 2
#   gru_scale_s = 2
_NGRU = 2
_GRU_SCALE_S = 2


class DNN_SKalmanNet_GSS(torch.nn.Module):
    def __init__(self, x_dim: int = 2, y_dim: int = 2):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # For NCLT, SyntheticNL (general)
        H1 = (x_dim + y_dim) * (10) * 8
        H2 = (x_dim * y_dim) * 1 * (4)

        self.input_dim_1 = (self.x_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)
        self.input_dim_2 = (self.y_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)

        self.output_dim_1 = self.x_dim * self.x_dim
        self.output_dim_2 = self.y_dim * self.y_dim

        # input layer {x_k - x_{k-1}}
        self.l1 = nn.Sequential(nn.Linear(self.input_dim_1, H1), nn.ReLU())

        # GRU
        self.gru_input_dim = H1
        self.gru_hidden_dim = round(
            _GRU_SCALE_S * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim))
        )
        self.gru_n_layer = _NGRU
        self.batch_size = 1
        self.seq_len_input = 1

        self.hn1 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn1_init = self.hn1.detach().clone()
        self.GRU1 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> Pk
        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_1),
        )

        # input layer {residual}
        self.l3 = nn.Sequential(nn.Linear(self.input_dim_2, H1), nn.ReLU())

        # GRU
        self.hn2 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn2_init = self.hn2.detach().clone()
        self.GRU2 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> Sk
        self.l4 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_2),
        )

    def initialize_hidden(self):
        self.hn1 = self.hn1_init.detach().clone()
        self.hn2 = self.hn2_init.detach().clone()

    def forward(
        self, state_inno, observation_inno, diff_state, diff_obs, linearization_error, Jacobian
    ):
        input1 = torch.cat((state_inno, diff_state, linearization_error, Jacobian), axis=0).reshape(
            -1
        )
        input2 = torch.cat(
            (observation_inno, diff_obs, linearization_error, Jacobian), axis=0
        ).reshape(-1)

        l1_out = self.l1(input1)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0, 0, :] = l1_out
        GRU_out, self.hn1 = self.GRU1(GRU_in, self.hn1)
        l2_out = self.l2(GRU_out)
        Pk = l2_out.reshape((self.x_dim, self.x_dim))

        l3_out = self.l3(input2)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0, 0, :] = l3_out
        GRU_out, self.hn2 = self.GRU2(GRU_in, self.hn2)
        l4_out = self.l4(GRU_out)
        Sk = l4_out.reshape((self.y_dim, self.y_dim))

        return (Pk, Sk)


def build_split_kalmannet():
    torch.manual_seed(0)
    # Matches the repo's default state-space dims for the NCLT / SyntheticNL
    # experiments (x_dim=2, y_dim=2, e.g. the SyntheticNL 2-D nonlinear tracking task).
    model = DNN_SKalmanNet_GSS(x_dim=2, y_dim=2)
    model.eval()
    return model


def example_input_split_kalmannet():
    torch.manual_seed(0)
    # One filtering-step's worth of KalmanNet feature vectors, matching the real
    # forward() signature: state-innovation, observation-innovation, state-diff,
    # observation-diff (all length x_dim or y_dim column vectors) plus the
    # linearization error (y_dim) and flattened Jacobian (x_dim*y_dim).
    x_dim, y_dim = 2, 2
    state_inno = torch.randn(x_dim, 1)
    observation_inno = torch.randn(y_dim, 1)
    diff_state = torch.randn(x_dim, 1)
    diff_obs = torch.randn(y_dim, 1)
    linearization_error = torch.randn(y_dim, 1)
    Jacobian = torch.randn(x_dim * y_dim, 1)
    return (state_inno, observation_inno, diff_state, diff_obs, linearization_error, Jacobian)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Split-KalmanNet",
        "build_split_kalmannet",
        "example_input_split_kalmannet",
        2023,
        MENAGERIE_ZOO,
    ),
]
