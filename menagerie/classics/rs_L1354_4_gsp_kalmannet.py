# FAITHFUL PORT of https://github.com/NimrodLeinwand/GSP-KalmanNet @ main
#   (original framework: PyTorch, but the model file itself cannot be imported
#   standalone -- see below -- so this is a transcription rather than a vendor)
#
#   Ported source: `GSP-KNet/GSP-KalmanNet.py` (class `GSPKalmanNetNN`, verbatim
#   method bodies) plus `Simulations/GSP_Linear_Sys_model.py` (class
#   `SystemModel_KF`, used here only as the reference for the `V`/`V_t`/`F`/`H`
#   argument contract `Build()` expects -- not itself ported, since we call
#   `Build(F, H, V, V_t)` directly instead of the repo's
#   `Build(SystemModel_KF_instance)`).
#
#   Why this is a PORT and not a RUNG-2 VENDOR: `GSP-KalmanNet.py` does
#   `from Main_Pypower import dev` at module scope. `Main_Pypower_gaussian_noise.py`
#   (the only module named `Main_Pypower` in the repo) requires `pypower`
#   (not a base lib here), does `torch.set_default_tensor_type(...)` as a
#   process-wide side effect, and loads `.pt` checkpoint/dataset files from
#   `./drive/MyDrive/...` at import time -- so `GSP-KalmanNet.py` cannot be
#   imported standalone in any base-lib environment. Every method body below is
#   transcribed verbatim from the real class; only the module-scope `dev` import
#   is replaced with a local `dev = torch.device("cpu")`, `Build()` is changed to
#   take `(F, H, V, V_t)` directly instead of a `SystemModel_KF` object (its body
#   -- `self.V = ssModel.V; self.V_t = ssModel.V_t; self.InitSystemDynamics(...)`
#   -- is unchanged, just reading the 4 fields as plain arguments instead of
#   attribute lookups on an un-portable helper object), and `batch_size` is
#   exposed as an explicit `Build(..., batch_size=1)` argument instead of being
#   hardcoded to `1` inside `InitKGainNet` -- the real repo's own
#   `GSP-KNet/BatchedPipeline_EKF.py` driver script does exactly this same
#   override (`self.model.batch_size = self.N_T`, 3 call sites) before running
#   multi-trajectory batches through what is otherwise a `batch_size=1`-only
#   `InitKGainNet` allocation, so this is the repo's own documented usage
#   pattern, not a new architectural choice.
#
# GSP-KalmanNet (Graph Signal Processing KalmanNet; IEEE TSP 2024) extends
# KalmanNet -- a hybrid model-based/data-driven Kalman filter that replaces the
# classical Kalman gain computation with a learned recurrent network (Linear ->
# ReLU -> GRU -> Linear -> ReLU -> Linear) fed normalized state/observation
# innovations -- to graph-structured state spaces: the linear/observation
# matrices F/H and the running state/observation are first transformed into the
# graph Fourier (spectral) domain via `V`/`V_t` (the eigenbasis of the graph
# Laplacian), the Kalman-gain-network recursion runs entirely in that spectral
# domain, and the final posterior state estimate is transformed back with the
# inverse GFT before being returned. Architecture (GFT/IGFT wrapping around a
# GRU-based learned Kalman gain estimator) is reproduced faithfully from the
# real class body.

import torch
import torch.nn as nn
import torch.nn.functional as func

MENAGERIE_ZOO = "ported-pytorch"

dev = torch.device("cpu")


class GSPKalmanNetNN(torch.nn.Module):
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    #############
    ### Build ###
    #############
    def Build(self, F, H, V, V_t, batch_size=1):
        self.V = V
        self.V_t = V_t
        self.InitSystemDynamics(F, H)
        # Number of neurons in the 1st hidden layer
        H1_KNet = (self.m + self.n) * 10 * 8  # Here we can reduce the latent space

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (self.m * self.n) * 1 * (4)  # Here we can reduce the latent space

        self.batch_size = batch_size
        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        D_out = self.n  # Diagonal Kalman Gain ## GSP update 8.3

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 10
        # Number of Layers
        self.n_layers = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(
            self.device, non_blocking=True
        )

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H):
        # Set State Evolution Matrix
        self.F = self.GFT_matrix(F.to(self.device, non_blocking=True))
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = self.GFT_matrix(H.to(self.device, non_blocking=True))
        self.H_T = torch.transpose(H, 0, 1)
        self.n = self.H.size()[0]

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        M1_0 = self.GFT(M1_0.squeeze())

        self.m1x_posterior = torch.squeeze(M1_0).to(dev)
        self.m1x_posterior_previous = 0  # for t=0

        self.T = T
        self.x_out = torch.empty(self.m, T)

        self.m1x_prior = M1_0.to(self.device, non_blocking=True)
        self.state_process_posterior_0 = torch.squeeze(M1_0).to(dev)
        self.m1x_prior_previous = self.m1x_posterior

        # KGain saving
        self.i = 0
        self.KGain_array = self.KG_array = torch.zeros((self.T * 10, self.m, self.n)).to(dev)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Compute the 1-st moment of x based on model knowledge and without process noise
        bmm_mul = torch.bmm(
            self.F.expand(self.state_process_posterior_0.size()[0], -1, -1)
            .type(torch.DoubleTensor)
            .to(dev),
            self.state_process_posterior_0.unsqueeze(-1).type(torch.DoubleTensor).to(dev),
        ).squeeze(-1)
        self.state_process_prior_0 = bmm_mul

        # Compute the 1-st moment of y based on model knowledge and without noise
        bmm_mul = torch.bmm(
            self.H.expand(self.state_process_prior_0.size()[0], -1, -1)
            .type(torch.DoubleTensor)
            .to(dev),
            self.state_process_prior_0.unsqueeze(-1).type(torch.DoubleTensor).to(dev),
        ).squeeze(-1)
        self.obs_process_0 = bmm_mul

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior.squeeze()
        bmm_mul = torch.bmm(
            self.F.expand(self.m1x_posterior.size()[0], -1, -1).type(torch.DoubleTensor).to(dev),
            self.m1x_posterior.unsqueeze(-1).type(torch.DoubleTensor).to(dev),
        ).squeeze(-1)
        self.m1x_prior = bmm_mul

        # Predict the 1-st moment of y
        bmm_mul = torch.bmm(
            self.H.expand(self.m1x_prior.size()[0], -1, -1).type(torch.DoubleTensor).to(dev),
            self.m1x_prior.unsqueeze(-1).type(torch.DoubleTensor).to(dev),
        ).squeeze(-1)
        self.m1y = bmm_mul

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        # Reshape and Normalize the difference in X prior
        # Featture 4: x_t|t - x_t|t-1
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 2: yt - y_t+1|t
        dm1y = y.squeeze() - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=1)
        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)
        # Reshape Kalman Gain to a Matrix
        KG = torch.diag_embed(KG)
        self.KGain = torch.reshape(KG, (-1, self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = torch.squeeze(y)
        dy = y_obs - self.m1y
        INOV = torch.matmul(self.KGain.float(), dy.unsqueeze(-1).float()).squeeze()
        self.m1x_posterior = self.m1x_prior + INOV

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):
        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in.type(torch.FloatTensor).to(dev))
        La1_out = self.KG_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(
            self.device, non_blocking=True
        )
        GRU_in[0, :, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = self.GFT(yt.squeeze())
        yt = yt.to(self.device, non_blocking=True)
        return self.IGFT(self.KNet_step(yt))

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data

    def GFT(self, input):
        return self.BMM_multipy(self.V_t, input.to(dev)).type(torch.FloatTensor)

    def IGFT(self, input):
        return self.BMM_multipy(self.V, input.to(dev)).type(torch.FloatTensor)

    def GFT_matrix(self, input):
        return torch.matmul(torch.matmul(self.V_t, input.to(dev)), self.V).type(torch.FloatTensor)

    def IGFT_matrix(self, input):
        return torch.matmul(torch.matmul(self.V, input.to(dev)), self.V).type(torch.FloatTensor)

    def BMM_multipy(self, a, b):
        return torch.bmm(
            a.expand(b.size()[0], -1, -1).type(torch.DoubleTensor).to(dev),
            b.unsqueeze(-1).type(torch.DoubleTensor).to(dev),
        ).squeeze(-1)


# ---- tiny build/example (architecture unmodified from the real repo) ----


def _graph_gft_basis(num_nodes):
    """Builds a ring-graph Laplacian eigenbasis (V, V_t) -- the real repo builds
    equivalent (F, H, L, V, V_t) tuples per-scenario in its `Simulations/*.py`
    parameter files (e.g. `Random_Walk_parameters.py`); a small ring graph is
    used here as a self-contained stand-in graph topology."""
    A = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        A[i, (i + 1) % num_nodes] = 1
        A[(i + 1) % num_nodes, i] = 1
    D = torch.diag(A.sum(dim=1))
    L = D - A
    _, eigvecs = torch.linalg.eigh(L)
    V = eigvecs
    V_t = eigvecs.t()
    return V, V_t


def build_gsp_kalmannet():
    """GSPKalmanNetNN at tiny size for tracing (state/obs dim m=n=4, batch=2).
    Architecture is unmodified from the real repo."""
    torch.manual_seed(0)
    m = 4
    n = 4
    V, V_t = _graph_gft_basis(m)
    F = torch.eye(m) * 0.9  # state-transition matrix (identity-like random-walk dynamics)
    H = torch.eye(n)  # observation matrix (direct observation of all graph nodes)

    model = GSPKalmanNetNN()
    model.Build(F, H, V, V_t, batch_size=2)
    model.init_hidden()

    B = 2
    T = 5
    M1_0 = torch.zeros(B, m, 1)
    model.InitSequence(M1_0, T)
    model.eval()
    return model


def example_input_gsp_kalmannet():
    """Matches GSPKalmanNetNN.forward(yt): a single timestep's observation batch,
    shape (BatchSize, n, 1), the exact call convention used by the repo's own
    `GSP-KNet/BatchedPipeline_EKF.py` driver
    (`self.model(torch.unsqueeze(test_input[:, :, t], 2))`)."""
    torch.manual_seed(0)
    B = 2
    n = 4
    y = torch.randn(B, n, 1)
    return (y,)


MENAGERIE_ENTRIES = [
    ("GSPKalmanNet", "build_gsp_kalmannet", "example_input_gsp_kalmannet", 2024, MENAGERIE_ZOO),
]
