# SOURCE: vendored from https://github.com/yz-cnsdqz/GAMMA-release @ main
#
# GAMMA (CVPR 2022): "The Wanderings of Odysseus in 3D Scenes" -- generative
# motion-primitive VAE for human motion generation. Vendored verbatim (only
# import paths trimmed to drop unused smplx/torchgeometry/tensorboardX
# module-level imports, which are not needed to construct/forward the VAE)
# from models/baseops.py (MLP, VAE base, ResNetBlock) and
# models/models_GAMMA_primitive.py (GAMMAPrimitiveVAE, MoshRegressor).
"""
===============================================================================
basic network modules (models/baseops.py, vendored)
===============================================================================
"""

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, h_dims=[128, 128], activation="tanh"):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_dim_, h_dim))
            in_dim_ = h_dim

    def forward(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def _sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, *args, **kwargs):
        pass

    def decode(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass

    def sample_prior(self, *args, **kwargs):  # [t, b, d]
        pass


class ResNetBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_blocks, actfun="relu"):
        super(ResNetBlock, self).__init__()

        self.in_fc = nn.Linear(in_dim, h_dim)
        self.layers = nn.ModuleList(
            [MLP(h_dim, h_dims=(h_dim, h_dim), activation=actfun) for _ in range(n_blocks)]
        )  # two fc layers in each MLP
        self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h = self.in_fc(x)
        for layer in self.layers:
            h = layer(h) + h
        y = self.out_fc(h)
        return y


"""
===============================================================================
motion-primitive VAE + body regressor (models/models_GAMMA_primitive.py, vendored)
===============================================================================
"""


class GAMMAPrimitiveVAE(VAE):
    """the marker predictor in GAMMA.

    GAMMA contains two basic modules, the marker predictor and the body regressor.
    This marker predictor takes a motion seed as input, and produces future frames in a motion primitive.
    In addition, it can take a goal as an extra condition, so as to produce goal-driven motions.
    """

    def __init__(self, configs):
        super(GAMMAPrimitiveVAE, self).__init__()
        self.body_repr = body_repr = configs["body_repr"]
        if body_repr == "ssm2_67":
            self.in_dim = in_dim = 67 * 3  # the marker dim
            self.c_dim = c_dim = in_dim
        elif body_repr in ["ssm2_67_condi_marker2tarloc"]:  # when close to the target
            self.in_dim = in_dim = 67 * 3  # the marker dim
            self.c_dim = c_dim = 67 * 3 * 2  # the marker dim, the vec_to_target dim

        self.h_dim = h_dim = configs["h_dim"]  # 256
        self.z_dim = z_dim = configs["z_dim"]
        self.use_drnn_mlp = configs["use_drnn_mlp"]
        self.hdims_mlp = hdims_mlp = configs["hdims_mlp"]  # [512, 256]
        self.residual = configs["residual"]

        # encode
        self.x_enc = nn.GRU(c_dim, h_dim)
        self.e_rnn = nn.GRU(in_dim, h_dim)
        self.e_mlp = MLP(2 * h_dim, hdims_mlp, activation="tanh")
        self.e_mu = nn.Linear(self.e_mlp.out_dim, z_dim)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, z_dim)

        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(h_dim, hdims_mlp + [h_dim], activation="tanh")
        self.d_rnn = nn.GRUCell(in_dim + z_dim + h_dim, h_dim)
        self.d_mlp = MLP(h_dim, hdims_mlp, activation="tanh")
        self.d_out = nn.Linear(self.d_mlp.out_dim, in_dim)

    def encode(self, x, y):
        _, hx = self.x_enc(x)
        _, hy = self.e_rnn(y)
        h = torch.cat((hx[0], hy[0]), dim=-1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def decode(self, x, z, t_pred):
        _, hx = self.x_enc(x)
        hx = hx[0]  # [b, d]
        if self.use_drnn_mlp:
            h_rnn = self.drnn_mlp(hx)
        else:
            h_rnn = hx
        y = []
        for i in range(t_pred):
            y_p = x[-1][:, : self.in_dim] if i == 0 else y_i  # noqa: F821 (bound by prior iteration; verbatim from source)
            rnn_in = torch.cat([hx, z, y_p], dim=-1)
            h_rnn = self.d_rnn(rnn_in, h_rnn)
            hfc = self.d_mlp(h_rnn)
            y_i = self.d_out(hfc)
            if self.residual:
                y_i = y_i + y_p
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(self, x, y):
        t_pred = y.shape[0]
        mu, logvar = self.encode(x, y)
        z = self._sample(mu, logvar)
        y_pred = self.decode(x, z, t_pred)

        return y_pred, mu, logvar


# --- staging harness: build + example input ---------------------------------


def build_gamma_primitive_vae():
    configs = {
        "body_repr": "ssm2_67",
        "h_dim": 16,
        "z_dim": 8,
        "use_drnn_mlp": True,
        "hdims_mlp": [16, 16],
        "residual": True,
    }
    return GAMMAPrimitiveVAE(configs)


def example_input_gamma_primitive_vae():
    # x: motion seed [t_his, b, c_dim], y: future frames [t_pred, b, in_dim]
    t_his, t_pred, b = 2, 3, 2
    in_dim = 67 * 3
    x = torch.randn(t_his, b, in_dim)
    y = torch.randn(t_pred, b, in_dim)
    return (x, y)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "GAMMA-MPVAE",
        build_gamma_primitive_vae,
        example_input_gamma_primitive_vae,
        2022,
        MENAGERIE_ZOO,
    ),
]
