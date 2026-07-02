# SOURCE: vendored from KalmanNet/KalmanNet_TSP @ main
# https://github.com/KalmanNet/KalmanNet_TSP
#
# Files vendored:
#   KNet/KalmanNet_nn.py       (KalmanNetNN -- the KalmanNet architecture itself)
#   Simulations/config.py      (SystemModel -- linear state-space wrapper providing f/h)
#
# KalmanNet (Revach, Shlezinger, Ni, Escoriza, van Sloun & Eldar, IEEE TSP 2022,
# "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics") learns
# the Kalman gain of an Extended-Kalman-Filter-style recursive state estimator with a
# small GRU-based network (3 GRUs tracking process/prior/innovation covariance proxies +
# 7 FC layers), replacing the explicit covariance recursion while keeping the classic
# predict/innovate/update state-space recursion structure.
#
# Only mechanical fixes applied (no architectural change):
#  1. `KalmanNetNN.NNBuild(SysModel, args)` expects an argparse.Namespace-like `args` with
#     `.use_cuda`, `.n_batch`, `.in_mult_KNet`, `.out_mult_KNet` (built by
#     Simulations/config.py's `general_settings()` CLI parser upstream); replaced with a
#     plain object carrying the same 4 attributes (no argparse, no CLI dependency).
#  2. `SystemModel.f`/`.h` reference `.to(x.device)` already (device-parametric in the
#     original code -- no change needed).
#  3. `from KNet.KalmanNet_nn import KalmanNetNN` / package-relative imports collapsed
#     into this single file.

import torch
import torch.nn as nn
import torch.nn.functional as func

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# Simulations/config.py (SystemModel) -- linear state-space wrapper. KalmanNet needs
# `SysModel.f`, `.h`, `.m`, `.n`, `.prior_Q`, `.prior_Sigma`, `.prior_S` to build.
# ------------------------------------------------------------------
class SystemModel:
    def __init__(self, F, Q, H, R, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):
        self.F = F
        self.m = self.F.size()[0]
        self.Q = Q

        self.H = H
        self.n = self.H.size()[0]
        self.R = R

        self.T = T
        self.T_test = T_test

        self.prior_Q = torch.eye(self.m) if prior_Q is None else prior_Q
        self.prior_Sigma = torch.zeros((self.m, self.m)) if prior_Sigma is None else prior_Sigma
        self.prior_S = torch.eye(self.n) if prior_S is None else prior_S

    def f(self, x):
        batched_F = (
            self.F.to(x.device).view(1, self.F.shape[0], self.F.shape[1]).expand(x.shape[0], -1, -1)
        )
        return torch.bmm(batched_F, x)

    def h(self, x):
        batched_H = (
            self.H.to(x.device).view(1, self.H.shape[0], self.H.shape[1]).expand(x.shape[0], -1, -1)
        )
        return torch.bmm(batched_H, x)

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0


# ------------------------------------------------------------------
# KNet/KalmanNet_nn.py
# ------------------------------------------------------------------
class KalmanNetNN(torch.nn.Module):
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args):
        # Device
        if args.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        self.seq_len_input = 1  # KNet calculates time-step by time-step
        self.batch_size = args.n_batch  # Batch size

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        # GRU to track Q
        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m**2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m**2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        # GRU to track S
        self.d_input_S = self.n**2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n**2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n**2
        self.FC1 = nn.Sequential(nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU()).to(
            self.device
        )

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
        ).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m**2
        self.FC3 = nn.Sequential(nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU()).to(
            self.device
        )

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU()).to(
            self.device
        )

        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU()).to(
            self.device
        )

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU()).to(
            self.device
        )

        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU()).to(
            self.device
        )

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        """
        input M1_0 (torch.tensor): 1st moment of x at time 0 [batch_size, m, 1]
        """
        self.T = T

        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        # both in size [batch_size, n]
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(
            self.m1x_posterior_previous, 2
        )
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(
            self.m1x_prior_previous, 2
        )

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        dy = y - self.m1y  # [batch_size, n, 1]

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        return self.m1x_posterior

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################

        # FC 5
        in_FC5 = fw_update_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        # FC 6
        in_FC6 = fw_evol_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2

    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        y = y.to(self.device)
        return self.KNet_step(y)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = (
            self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        )
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = (
            self.prior_Sigma.flatten()
            .reshape(1, 1, -1)
            .repeat(self.seq_len_input, self.batch_size, 1)
        )
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = (
            self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        )


class _Args:
    """Plain stand-in for the argparse.Namespace built by Simulations/config.py's
    `general_settings()`; carries only the 4 attributes KalmanNetNN.NNBuild/InitKGainNet
    actually reads."""

    def __init__(self, use_cuda=False, n_batch=4, in_mult_KNet=5, out_mult_KNet=40):
        self.use_cuda = use_cuda
        self.n_batch = n_batch
        self.in_mult_KNet = in_mult_KNet
        self.out_mult_KNet = out_mult_KNet


class KalmanNetStepper(nn.Module):
    """Menagerie wrapper: initializes a KalmanNetNN over a small linear canonical
    2-state / 2-observation system model (matching Simulations/Linear_canonical/parameters.py)
    and drives 3 recursive `forward(y_t)` steps -- the real usage pattern from
    Pipelines/Pipeline_EKF.py's NNTest loop (`x_out[..., t] = model(y[..., t])` per
    timestep) -- to exercise the full GRU-state recursion in a single traced call."""

    def __init__(self, m=2, n=2, batch_size=4, n_steps=3):
        super().__init__()
        F = torch.eye(m)
        F[0] = torch.ones(1, m)
        H = torch.eye(n) if m == n else torch.eye(m, n)
        Q = 0.1 * torch.eye(m)
        R = 0.1 * torch.eye(n)
        sys_model = SystemModel(F, Q, H, R, T=n_steps, T_test=n_steps)
        m1_0 = torch.zeros(m, 1)
        m2_0 = torch.zeros(m, m)
        sys_model.InitSequence(m1_0, m2_0)

        args = _Args(use_cuda=False, n_batch=batch_size, in_mult_KNet=5, out_mult_KNet=40)
        self.knet = KalmanNetNN()
        self.knet.NNBuild(sys_model, args)
        self.batch_size = batch_size
        self.m = m
        self.n_steps = n_steps

    def forward(self, y_seq):
        """y_seq: [batch, n, n_steps] observation sequence."""
        self.knet.init_hidden_KNet()
        m1x_0_batch = torch.zeros(self.batch_size, self.m, 1)
        self.knet.InitSequence(m1x_0_batch, self.n_steps)

        outs = []
        for t in range(self.n_steps):
            y_t = torch.unsqueeze(y_seq[:, :, t], 2)
            outs.append(self.knet(y_t))
        return torch.cat(outs, dim=2)


# ------------------------------------------------------------------
# Menagerie staging entry points
# ------------------------------------------------------------------
def build_kalmannet():
    return KalmanNetStepper(m=2, n=2, batch_size=4, n_steps=3)


def example_input_kalmannet():
    return torch.randn(4, 2, 3)


MENAGERIE_ENTRIES = [
    ("KalmanNet", "build_kalmannet", "example_input_kalmannet", "2022", "vendored-pytorch"),
]
