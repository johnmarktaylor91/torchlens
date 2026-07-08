# SOURCE: vendored from rsinghlab/scNODE @ main
# https://raw.githubusercontent.com/rsinghlab/scNODE/main/model/dynamic_model.py
# https://raw.githubusercontent.com/rsinghlab/scNODE/main/model/layer.py
# https://raw.githubusercontent.com/rsinghlab/scNODE/main/model/diff_solver.py
#
# scNODE (Zhang & Singh, 2024, NeurIPS) -- VAE encoder + latent neural-ODE dynamics
# decoder + VAE decoder for temporal single-cell RNA-seq trajectory prediction. The
# real model class `scNODE(nn.Module)` (dynamic_model.py), its `LinearNet` /
# `LinearVAENet` sub-networks (layer.py), and the `ODE` neural-ODE wrapper around
# `torchdiffeq.odeint` (diff_solver.py) are copied verbatim, unchanged from the
# official repo -- only the `from model.layer import ...` relative import was
# adjusted to a same-file reference since all three files are combined here.
#
# Upstream license: rsinghlab/scNODE (MIT).

import torch
import torch.nn as nn
import torch.distributions as dist
import torchdiffeq

MENAGERIE_ZOO = "vendored-pytorch"

# ===========================================
# model/layer.py (verbatim)
# ===========================================

ACT_FUNC_MAP = {
    "none": nn.Identity(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(),
}


class LinearNet(nn.Module):
    """
    Fully-connected neural network.
    """

    def __init__(self, input_dim, latent_size_list, output_dim, act_name):
        super(LinearNet, self).__init__()
        layer_list = []
        if act_name not in ACT_FUNC_MAP:
            raise ValueError(
                "The activation function should be one of {}.".format(ACT_FUNC_MAP.keys())
            )
        act_func = ACT_FUNC_MAP[act_name]
        if latent_size_list is not None:
            layer_list.extend([nn.Linear(input_dim, latent_size_list[0]), act_func])
            for i in range(len(latent_size_list) - 1):
                layer_list.extend(
                    [nn.Linear(latent_size_list[i], latent_size_list[i + 1]), act_func]
                )
            layer_list.extend([nn.Linear(latent_size_list[-1], output_dim), act_func])
        else:
            layer_list.extend([nn.Linear(input_dim, output_dim)])
        self.net = nn.Sequential(*layer_list)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, data):
        out = self.net(data)
        return out

    def forwardWithTime(self, t, data):
        out = self.net(data)
        return out


class LinearVAENet(nn.Module):
    """
    Fully-connected neural network used for variational autoencoder (VAE) encoder.
    """

    def __init__(self, input_dim, latent_size_list, output_dim, act_name):
        super(LinearVAENet, self).__init__()
        layer_list = []
        if act_name not in ACT_FUNC_MAP:
            raise ValueError(
                "The activation function should be one of {}.".format(ACT_FUNC_MAP.keys())
            )
        act_func = ACT_FUNC_MAP[act_name]
        if latent_size_list is not None:
            layer_list.extend([nn.Linear(input_dim, latent_size_list[0]), act_func])
            for i in range(len(latent_size_list) - 1):
                layer_list.extend(
                    [nn.Linear(latent_size_list[i], latent_size_list[i + 1]), act_func]
                )
            layer_list.extend([nn.Linear(latent_size_list[-1], output_dim), act_func])
        else:
            layer_list.extend([nn.Linear(input_dim, output_dim)])
        self.net = nn.Sequential(*layer_list)
        self.mu_layer = nn.Linear(output_dim, output_dim)
        self.var_layer = nn.Linear(output_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, data):
        out = self.net(data)
        mu = self.mu_layer(out)
        std = torch.abs(self.var_layer(out))  # avoid incompatible std values
        return mu, std


# ===========================================
# model/diff_solver.py (verbatim)
# ===========================================


class ODE(nn.Module):
    """
    Ordinary differential equation (ODE) solver.
    """

    def __init__(self, input_dim, drift_net, ode_method):
        super(ODE, self).__init__()
        self.input_dim = input_dim
        self.net = drift_net
        self.ode_method = ode_method
        self.rtol = 1e-5  # upper bound on relative error
        self.atol = 1e-5  # upper bound on absolute error

    def forward(self, first_data, tp_to_pred):
        pred_data = torchdiffeq.odeint(
            self.net.forwardWithTime,
            first_data,
            tp_to_pred,
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol,
        )
        pred_data = torch.moveaxis(
            pred_data, [0, 1, 2], [1, 0, 2]
        )  # (# cells, # tps, # genes) after flipping axes
        return pred_data


# ===========================================
# model/dynamic_model.py (verbatim)
# ===========================================


class scNODE(nn.Module):
    """
    scNODE model.
    """

    def __init__(
        self, input_dim, latent_dim, output_dim, latent_encoder, diffeq_decoder, obs_decoder
    ):
        super(scNODE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        # -----
        assert isinstance(latent_encoder, LinearVAENet)
        self.latent_encoder = latent_encoder
        self.diffeq_decoder = diffeq_decoder
        self.obs_decoder = obs_decoder

    def forward(self, data, tps, batch_size=None):
        first_tp_data = data[0]
        if batch_size is not None:
            import numpy as np

            cell_idx = np.random.choice(
                np.arange(first_tp_data.shape[0]),
                size=batch_size,
                replace=(first_tp_data.shape[0] < batch_size),
            )
            first_tp_data = first_tp_data[cell_idx, :]
        # Map data at the first timepoint to the VAE latent space
        first_latent_mu, first_latent_std = self.latent_encoder(first_tp_data)
        first_latent_dist = dist.Normal(first_latent_mu, first_latent_std)
        first_latent_sample = self._sampleGaussian(first_latent_mu, first_latent_std)
        # Predict forward with ODE solver in the latent space
        latent_seq = self.diffeq_decoder(first_latent_sample, tps)
        # Convert latent variables (at all timepoints) back to the gene space
        recon_obs = self.obs_decoder(latent_seq)  # (batch size, # tps, # genes)
        return recon_obs, first_latent_dist, first_tp_data, latent_seq

    def _sampleGaussian(self, mean, std):
        d = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        r = d.sample(mean.size()).squeeze(-1)
        x = r * std.float() + mean.float()
        return x


def build_scnode():
    input_dim = 20
    latent_dim = 4
    hidden = [16]
    latent_encoder = LinearVAENet(
        input_dim=input_dim, latent_size_list=hidden, output_dim=latent_dim, act_name="relu"
    )
    drift_net = LinearNet(
        input_dim=latent_dim, latent_size_list=hidden, output_dim=latent_dim, act_name="relu"
    )
    diffeq_decoder = ODE(input_dim=latent_dim, drift_net=drift_net, ode_method="euler")
    obs_decoder = LinearNet(
        input_dim=latent_dim, latent_size_list=hidden, output_dim=input_dim, act_name="relu"
    )
    model = scNODE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=input_dim,
        latent_encoder=latent_encoder,
        diffeq_decoder=diffeq_decoder,
        obs_decoder=obs_decoder,
    )
    model.eval()
    return model


def example_input_scnode():
    first_tp_data = torch.randn(6, 20)
    tps = torch.tensor([0.0, 1.0, 2.0])
    return (
        [first_tp_data],
        tps,
    )


MENAGERIE_ENTRIES = [
    ("scNODE", "build_scnode", "example_input_scnode", 2024, "vendored-pytorch"),
]
