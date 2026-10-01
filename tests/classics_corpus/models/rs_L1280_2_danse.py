# SOURCE: vendored from https://github.com/saikatchatt/danse-jrnl @ main
#   Vendored files: src/danse.py (DANSE class, dropping training-only methods not part
#   of the nn.Module forward graph -- push_model/save_model/train_danse/
#   compute_predictions/compute_logprob_batch) and src/rnn.py (RNN_model class). The
#   create_diag helper from utils/utils.py, which DANSE.forward calls, is included
#   unmodified.
#
# "DANSE: Data-driven Non-linear State Estimation of Model-free Process in
# Unsupervised Learning based Setup" (Ghosh, Honore, Chatterjee; IEEE TSP 2024).
# DANSE combines a recurrent network (RNN_model: RNN/LSTM/GRU + two linear heads
# producing per-timestep Gaussian mean/variance of the latent state) with a
# closed-form linear-Gaussian measurement update (prior/marginal/posterior mean-
# covariance recursion via the fixed observation matrix H) to perform unsupervised,
# model-free nonlinear state estimation directly from an observation sequence. We
# trace `DANSE.forward`, which is the full nn.Module used for both training and
# posterior-mean-var computation: it runs Yi_batch through the internal RNN, computes
# prior/marginal Gaussian parameters, and returns the average sequence log-likelihood
# (all tensor ops; no python-side control flow branches on input values). We
# instantiate with rnn_type="gru" and a small n_states/n_obs/hidden size for a tiny
# random-init trace.

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- from src/rnn.py (RNN_model), unmodified ----
class RNN_model(nn.Module):
    """This super class defines the specific model to be used i.e. LSTM or GRU or RNN"""

    def __init__(
        self,
        input_size,
        output_size,
        n_hidden,
        n_layers,
        model_type,
        lr,
        num_epochs,
        n_hidden_dense=32,
        num_directions=1,
        batch_first=True,
        min_delta=1e-2,
        device="cpu",
    ):
        super(RNN_model, self).__init__()
        self.hidden_dim = n_hidden
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size

        self.model_type = model_type
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

        self.num_directions = num_directions
        self.batch_first = batch_first

        if model_type.lower() == "rnn":
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=self.batch_first,
            )
        elif model_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=self.batch_first,
            )
        elif model_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=self.batch_first,
            )
        else:
            raise ValueError("Model type unknown: {}".format(model_type.lower()))

        self.fc = nn.Linear(self.hidden_dim * self.num_directions, n_hidden_dense).to(self.device)
        self.fc_mean = nn.Linear(n_hidden_dense, self.output_size).to(self.device)
        self.fc_vars = nn.Linear(n_hidden_dense, self.output_size).to(self.device)

    def init_h0(self, batch_size):
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return h0

    def forward(self, x):
        batch_size = x.shape[0]

        r_out, _ = self.rnn(x)

        r_out_all_steps = r_out.contiguous().view(
            batch_size, -1, self.num_directions * self.hidden_dim
        )

        y = F.relu(self.fc(r_out_all_steps))

        mu_2T_1 = self.fc_mean(y)
        vars_2T_1 = F.softplus(self.fc_vars(y))

        mu_1 = self.fc_mean(F.relu(self.fc(self.init_h0(batch_size)[-1, :, :]))).view(
            batch_size, 1, -1
        )
        var_1 = F.softplus(
            self.fc_vars(F.relu(self.fc(self.init_h0(batch_size)[-1, :, :]))).view(
                batch_size, 1, -1
            )
        )

        mu = torch.cat((mu_1, mu_2T_1[:, :-1, :]), dim=1)
        vars = torch.cat((var_1, vars_2T_1[:, :-1, :]), dim=1)

        return mu, vars


# ---- from utils/utils.py, unmodified ----
def create_diag(x):
    return torch.diag_embed(x)


# ---- from src/danse.py (DANSE), unmodified except drop of training-only methods ----
class DANSE(nn.Module):
    def __init__(
        self,
        n_states,
        n_obs,
        mu_w,
        C_w,
        H,
        mu_x0,
        C_x0,
        batch_size,
        rnn_type,
        rnn_params_dict,
        device="cpu",
    ):
        super(DANSE, self).__init__()

        self.device = device

        self.n_states = n_states
        self.n_obs = n_obs

        self.mu_x0 = self.push_to_device(mu_x0)
        self.C_x0 = self.push_to_device(C_x0)

        self.mu_w = self.push_to_device(mu_w)
        self.C_w = self.push_to_device(C_w)

        self.H = self.push_to_device(H)

        self.batch_size = batch_size

        self.rnn_type = rnn_type

        self.rnn = RNN_model(**rnn_params_dict[self.rnn_type]).to(self.device)

        self.mu_xt_yt_current = None
        self.L_xt_yt_current = None

        self.mu_yt_current = None
        self.L_yt_current = None

        self.mu_xt_yt_prev = None
        self.L_xt_yt_prev = None

    def push_to_device(self, x):
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def compute_prior_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev):
        self.mu_xt_yt_prev = mu_xt_yt_prev
        self.L_xt_yt_prev = create_diag(L_xt_yt_prev)
        return self.mu_xt_yt_prev, self.L_xt_yt_prev

    def compute_marginal_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev):
        self.mu_yt_current = torch.einsum("ij,ntj->nti", self.H, mu_xt_yt_prev) + self.mu_w.squeeze(
            -1
        )
        self.L_yt_current = self.H @ L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w

    def compute_posterior_mean_vars(self, Yi_batch):
        Re_t_inv = torch.inverse(
            self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w
        )
        self.K_t = self.L_xt_yt_prev @ (self.H.T @ Re_t_inv)
        self.mu_xt_yt_current = self.mu_xt_yt_prev + torch.einsum(
            "ntij,ntj->nti",
            self.K_t,
            (Yi_batch - torch.einsum("ij,ntj->nti", self.H, self.mu_xt_yt_prev)),
        )
        self.L_xt_yt_current = self.L_xt_yt_prev - (
            torch.einsum(
                "ntij,ntjk->ntik",
                self.K_t,
                self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w,
            )
            @ torch.transpose(self.K_t, 2, 3)
        )
        return self.mu_xt_yt_current, self.L_xt_yt_current

    def compute_logpdf_Gaussian(self, Y):
        _, T, _ = Y.shape
        logprob = (
            0.5 * self.n_obs * T * math.log(math.pi * 2)
            - 0.5 * torch.logdet(self.L_yt_current).sum(1)
            - 0.5
            * torch.einsum(
                "nti,nti->nt",
                (Y - self.mu_yt_current),
                torch.einsum(
                    "ntij,ntj->nti", torch.inverse(self.L_yt_current), (Y - self.mu_yt_current)
                ),
            ).sum(1)
        )

        return logprob

    def forward(self, Yi_batch):
        mu_batch, vars_batch = self.rnn.forward(x=Yi_batch)
        mu_xt_yt_prev, L_xt_yt_prev = self.compute_prior_mean_vars(
            mu_xt_yt_prev=mu_batch, L_xt_yt_prev=vars_batch
        )
        self.compute_marginal_mean_vars(mu_xt_yt_prev=mu_xt_yt_prev, L_xt_yt_prev=L_xt_yt_prev)
        logprob_batch = self.compute_logpdf_Gaussian(Y=Yi_batch) / (
            Yi_batch.shape[1] * Yi_batch.shape[2]
        )
        log_pYT_batch_avg = logprob_batch.mean(0)

        return log_pYT_batch_avg


_N_STATES = 3
_N_OBS = 3
_SEQ_LEN = 6
_BATCH = 2


def build_danse():
    rnn_params_dict = {
        "gru": dict(
            input_size=_N_OBS,
            output_size=_N_STATES,
            n_hidden=8,
            n_layers=1,
            model_type="gru",
            lr=1e-3,
            num_epochs=1,
            n_hidden_dense=8,
            device="cpu",
        )
    }

    rng = np.random.default_rng(0)
    mu_w = np.zeros((_N_OBS, 1), dtype=np.float32)
    C_w = (0.1 * np.eye(_N_OBS)).astype(np.float32)
    H = rng.standard_normal((_N_OBS, _N_STATES)).astype(np.float32)
    mu_x0 = np.zeros((_N_STATES, 1), dtype=np.float32)
    C_x0 = np.eye(_N_STATES, dtype=np.float32)

    model = DANSE(
        n_states=_N_STATES,
        n_obs=_N_OBS,
        mu_w=mu_w,
        C_w=C_w,
        H=H,
        mu_x0=mu_x0,
        C_x0=C_x0,
        batch_size=_BATCH,
        rnn_type="gru",
        rnn_params_dict=rnn_params_dict,
        device="cpu",
    )
    model.eval()
    return model


def example_input_danse():
    return torch.randn(_BATCH, _SEQ_LEN, _N_OBS)


MENAGERIE_ENTRIES = [
    ("DANSE", "build_danse", "example_input_danse", 2024, MENAGERIE_ZOO),
]
