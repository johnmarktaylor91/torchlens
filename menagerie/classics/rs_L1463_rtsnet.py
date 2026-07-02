# SOURCE: vendored from KalmanNet/RTSNet_TSP @ e226a9200cb3
# https://raw.githubusercontent.com/KalmanNet/RTSNet_TSP/e226a9200cb3/RTSNet/KalmanNet_nn.py
# https://raw.githubusercontent.com/KalmanNet/RTSNet_TSP/e226a9200cb3/RTSNet/RTSNet_nn.py
#
# RTSNet (Ni, Milstein, Buchnik, Sloin, Klein, Shlezinger, Eldar, van Sloun, "RTSNet:
# Deep Learning Aided Rauch-Tung-Striebel Smoother for State Estimation", IEEE
# Transactions on Signal Processing 2024) hybridizes a classical linear-Gaussian
# Kalman-filter / RTS-smoother recursion with learned GRU-based gain networks. It
# subclasses `KalmanNetNN` (the forward-pass Kalman-gain network, from the companion
# KalmanNet paper) and adds a backward-pass smoother-gain network (`InitRTSGainNet` /
# `RTSGain_step` / `RTSNet_step`) that estimates the RTS smoother gain from per-step
# innovation/evolution/update feature differences via three GRUs (GRU_Q_bw, GRU_Sigma_bw)
# and four small FC blocks. Vendored verbatim from `RTSNet/KalmanNet_nn.py` and
# `RTSNet/RTSNet_nn.py` -- every Linear/GRU/matmul, hidden-state bookkeeping, and the
# forward-pass/backward-pass step dispatch in `forward()` is unchanged; only the module
# docstring headers are dropped.
#
# The real network is driven by an explicit linear state-space model (F, H matrices,
# `SystemModel.f`/`SystemModel.h` -- here the identity-observation 2-state
# "Linear_canonical" example from `Simulations/Linear_canonical/parameters.py` /
# `main_linear_canonical.py`) and a small `args` namespace of network-size multipliers
# (`Simulations/config.py::general_settings`, default CLI values transcribed directly
# instead of via argparse so the module constructs without CLI args). `NNBuild` +
# `InitSequence` + `init_hidden` are called exactly as the real training/test pipelines
# call them (see `Pipelines/Pipeline_ERTS.py`) before the first forward step.
#
# TorchLens capture note: RTSNetNN.forward() dispatches on whether `yt is None`
# (forward Kalman-gain step vs backward smoother-gain step); `example_input_rtsnet`
# below drives ONE forward (`KNet_step`) step to populate `s_m1x_nexttime` /
# `filter_x_prior` via `InitBackward`, then ONE backward (`RTSNet_step`) step, matching
# how `Pipeline_ERTS.py` alternates forward and backward passes per time step.
#
# Only base-lib deps used: torch, torch.nn, torch.nn.functional.

import torch
import torch.nn as nn
import torch.nn.functional as func


class KalmanNetNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args):
        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        self.seq_len_input = 1
        self.batch_size = 1

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S

        # GRU to track Q
        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m**2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m**2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma)

        # GRU to track S
        self.d_input_S = self.n**2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n**2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_S)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n**2
        self.FC1 = nn.Sequential(nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
        )

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m**2
        self.FC3 = nn.Sequential(nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU())

        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU())

        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU())

    def InitSystemDynamics(self, f, h, m, n):
        self.f = f
        self.m = m
        self.h = h
        self.n = n

    def InitSequence(self, M1_0, T):
        self.T = T
        self.m1x_posterior = torch.squeeze(M1_0)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    def step_prior(self):
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior))
        self.m1y = torch.squeeze(self.h(self.m1x_prior))

    def step_KGain_est(self, y):
        obs_diff = y - torch.squeeze(self.y_previous)
        obs_innov_diff = y - torch.squeeze(self.m1y)
        fw_evol_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(
            self.m1x_posterior_previous
        )
        fw_update_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_prior_previous)

        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)

        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG, (self.m, self.n))

    def KNet_step(self, y):
        self.step_prior()
        self.step_KGain_est(y)

        dy = y - self.m1y
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y

        return torch.squeeze(self.m1x_posterior)

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        # Forward Flow
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        # Backward Flow
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        self.h_Sigma = out_FC4

        return out_FC2

    def forward(self, y):
        y = torch.squeeze(y)
        return self.KNet_step(y)

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()


class RTSNetNN(KalmanNetNN):
    def __init__(self):
        super().__init__()

    def NNBuild(self, ssModel, args):
        self.InitSystemDynamics(ssModel.f, ssModel.h, ssModel.m, ssModel.n)
        self.InitKGainNet(ssModel.prior_Q, ssModel.prior_Sigma, ssModel.prior_S, args)
        self.InitRTSGainNet(ssModel.prior_Q, ssModel.prior_Sigma, args)

    def InitRTSGainNet(self, prior_Q, prior_Sigma, args):
        self.seq_len_input = 1
        self.batch_size = 1

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma

        # BW GRU to track Q
        self.d_input_Q_bw = self.m * args.in_mult_RTSNet
        self.d_hidden_Q_bw = self.m**2
        self.GRU_Q_bw = nn.GRU(self.d_input_Q_bw, self.d_hidden_Q_bw)
        self.h_Q_bw = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q_bw)

        # BW GRU to track Sigma
        self.d_input_Sigma_bw = self.d_hidden_Q_bw + 2 * self.m * args.in_mult_RTSNet
        self.d_hidden_Sigma_bw = self.m**2
        self.GRU_Sigma_bw = nn.GRU(self.d_input_Sigma_bw, self.d_hidden_Sigma_bw)
        self.h_Sigma_bw = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma_bw)

        # BW Fully connected 1
        self.d_input_FC1_bw = self.d_hidden_Sigma_bw
        self.d_output_FC1_bw = self.m * self.m
        self.d_hidden_FC1_bw = self.d_input_FC1_bw * args.out_mult_RTSNet
        self.FC1_bw = nn.Sequential(
            nn.Linear(self.d_input_FC1_bw, self.d_hidden_FC1_bw),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC1_bw, self.d_output_FC1_bw),
        )

        # BW Fully connected 2
        self.d_input_FC2_bw = self.d_hidden_Sigma_bw + self.d_output_FC1_bw
        self.d_output_FC2_bw = self.d_hidden_Sigma_bw
        self.FC2_bw = nn.Sequential(nn.Linear(self.d_input_FC2_bw, self.d_output_FC2_bw), nn.ReLU())

        # BW Fully connected 3
        self.d_input_FC3_bw = self.m
        self.d_output_FC3_bw = self.m * args.in_mult_RTSNet
        self.FC3_bw = nn.Sequential(nn.Linear(self.d_input_FC3_bw, self.d_output_FC3_bw), nn.ReLU())

        # BW Fully connected 4
        self.d_input_FC4_bw = 2 * self.m
        self.d_output_FC4_bw = 2 * self.m * args.in_mult_RTSNet
        self.FC4_bw = nn.Sequential(nn.Linear(self.d_input_FC4_bw, self.d_output_FC4_bw), nn.ReLU())

    def InitBackward(self, filter_x):
        self.s_m1x_nexttime = torch.squeeze(filter_x)

    def S_Innovation(self, filter_x):
        self.filter_x_prior = self.f(filter_x)
        self.dx = self.s_m1x_nexttime - self.filter_x_prior

    def step_RTSGain_est(self, filter_x_nexttime, smoother_x_tplus2):
        dm1x_tilde = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_tilde_reshape = torch.squeeze(dm1x_tilde)
        bw_innov_diff = func.normalize(dm1x_tilde_reshape, p=2, dim=0, eps=1e-12, out=None)

        if smoother_x_tplus2 is None:
            dm1x_input2 = filter_x_nexttime - self.filter_x_prior
            dm1x_input2_reshape = torch.squeeze(dm1x_input2)
            bw_evol_diff = func.normalize(dm1x_input2_reshape, p=2, dim=0, eps=1e-12, out=None)
        else:
            dm1x_input2 = smoother_x_tplus2 - self.s_m1x_nexttime
            dm1x_input2_reshape = torch.squeeze(dm1x_input2)
            bw_evol_diff = func.normalize(dm1x_input2_reshape, p=2, dim=0, eps=1e-12, out=None)

        dm1x_f7 = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_f7_reshape = torch.squeeze(dm1x_f7)
        bw_update_diff = func.normalize(dm1x_f7_reshape, p=2, dim=0, eps=1e-12, out=None)

        SG = self.RTSGain_step(bw_innov_diff, bw_evol_diff, bw_update_diff)
        self.SGain = torch.reshape(SG, (self.m, self.m))

    def RTSNet_step(self, filter_x, filter_x_nexttime, smoother_x_tplus2):
        self.S_Innovation(filter_x)
        self.step_RTSGain_est(filter_x_nexttime, smoother_x_tplus2)

        INOV = torch.matmul(self.SGain, self.dx)
        self.s_m1x_nexttime = filter_x + INOV

        return torch.squeeze(self.s_m1x_nexttime)

    def RTSGain_step(self, bw_innov_diff, bw_evol_diff, bw_update_diff):
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        bw_innov_diff = expand_dim(bw_innov_diff)
        bw_evol_diff = expand_dim(bw_evol_diff)
        bw_update_diff = expand_dim(bw_update_diff)

        # Forward Flow
        in_FC3 = bw_update_diff
        out_FC3 = self.FC3_bw(in_FC3)

        in_Q = out_FC3
        out_Q, self.h_Q_bw = self.GRU_Q_bw(in_Q, self.h_Q_bw)

        in_FC4 = torch.cat((bw_innov_diff, bw_evol_diff), 2)
        out_FC4 = self.FC4_bw(in_FC4)

        in_Sigma = torch.cat((out_Q, out_FC4), 2)
        out_Sigma, self.h_Sigma_bw = self.GRU_Sigma_bw(in_Sigma, self.h_Sigma_bw)

        in_FC1 = out_Sigma
        out_FC1 = self.FC1_bw(in_FC1)

        # Backward Flow
        in_FC2 = torch.cat((out_Sigma, out_FC1), 2)
        out_FC2 = self.FC2_bw(in_FC2)

        self.h_Sigma_bw = out_FC2

        return out_FC1

    def forward(self, yt, filter_x, filter_x_nexttime, smoother_x_tplus2):
        if yt is None:
            return self.RTSNet_step(filter_x, filter_x_nexttime, smoother_x_tplus2)
        else:
            return self.KNet_step(yt)

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()

        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()

        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()

        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q_bw).zero_()
        self.h_Q_bw = hidden.data
        self.h_Q_bw[0, 0, :] = self.prior_Q.flatten()

        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma_bw).zero_()
        self.h_Sigma_bw = hidden.data
        self.h_Sigma_bw[0, 0, :] = self.prior_Sigma.flatten()


class _Args:
    """Transcribed CLI defaults from Simulations/config.py::general_settings, used
    directly (bypassing argparse/sys.argv) so the module constructs standalone."""

    in_mult_KNet = 5
    out_mult_KNet = 40
    in_mult_RTSNet = 5
    out_mult_RTSNet = 40


class _LinearCanonicalSysModel:
    """Transcribed from Simulations/Linear_canonical/parameters.py (m=n=2 case) plus
    Simulations/Linear_sysmdl.py::SystemModel.f/.h (plain `self.F.matmul(x)` /
    `self.H.matmul(x)`, no reshape -- the real pipeline always passes 1-D (m,)/(n,)
    state slices, see Pipelines/Pipeline_ERTS.py) and prior_Q/prior_Sigma/prior_S
    defaults (identity / zeros)."""

    def __init__(self):
        self.m = 2
        self.n = 2
        F = torch.eye(self.m)
        F[0] = torch.ones(1, self.m)
        H = torch.eye(2)
        self.F = F
        self.H = H
        self.f = lambda x: torch.matmul(F, x)
        self.h = lambda x: torch.matmul(H, x)
        self.prior_Q = torch.eye(self.m)
        self.prior_Sigma = torch.zeros((self.m, self.m))
        self.prior_S = torch.eye(self.n)


def build_rtsnet():
    torch.manual_seed(0)
    sys_model = _LinearCanonicalSysModel()
    args = _Args()

    model = RTSNetNN()
    model.NNBuild(sys_model, args)

    m1_0 = torch.zeros(sys_model.m, 1)
    model.InitSequence(m1_0, T=4)
    model.init_hidden()
    model.eval()
    return model


def example_input_rtsnet():
    torch.manual_seed(0)
    # One forward (Kalman-gain) observation followed by the two filtered states the
    # backward RTS step needs (filter_x, filter_x_nexttime), matching
    # Pipelines/Pipeline_ERTS.py's shape convention: `y_training[:, t]` /
    # `x_out_training_forward[:, t]` are 1-D (m,)/(n,) slices of a (m, T) trajectory
    # tensor, NOT (m, 1) column vectors.
    y0 = torch.randn(2)
    filter_x = torch.randn(2)
    filter_x_nexttime = torch.randn(2)
    return (y0, filter_x, filter_x_nexttime)


class _RTSNetTraceWrapper(nn.Module):
    """Thin wrapper driving one forward KNet step then one backward RTSNet_step, so a
    single example_input_/build_ pair captures both branches of RTSNetNN.forward()."""

    def __init__(self, rtsnet: RTSNetNN):
        super().__init__()
        self.rtsnet = rtsnet

    def forward(self, y0, filter_x, filter_x_nexttime):
        # Forward pass: populate m1x_prior / KGain state via one Kalman step.
        self.rtsnet(y0, None, None, None)
        # Prime the backward recursion at t=T-1 (smoother_x_tplus2=None case).
        self.rtsnet.InitBackward(filter_x)
        smoothed = self.rtsnet(None, filter_x, filter_x_nexttime, None)
        return smoothed


def build_rtsnet_traced():
    return _RTSNetTraceWrapper(build_rtsnet())


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RTSNet", "build_rtsnet_traced", "example_input_rtsnet", 2024, MENAGERIE_ZOO),
]
