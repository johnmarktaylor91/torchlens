# FAITHFUL PORT of https://github.com/nyu-wireless/mmwchanmod @ master
# (mmwchanmod/learn/models.py :: CondVAE, original framework: TensorFlow/Keras)
#
# nyu-wireless/mmwchanmod is the real, runnable code behind "GAN-based massive
# MIMO channel model trained on measured data" (the queue candidate's primary
# repo, github.com/Jeija/GAN-Wireless-Channel-Model, ships ONLY Jupyter
# notebooks that train a TensorFlow/Keras model inline -- no importable
# PyTorch or TensorFlow .py model-definition file exists in that repo at all,
# so it cannot be vendored). The notes column names nyu-wireless/mmwchanmod
# as the closest alternative with real source; that repo's
# `mmwchanmod/learn/models.py` module is entirely `tensorflow.keras`
# (`tfkl.Dense`, `tfkm.Model`) with no PyTorch path either, so this is a
# faithful PORT (rung 3), not a vendor (rung 2).
#
# `CondVAE` is the actual deep generative model used by the package's
# `ChanMod.path_mod` to synthesize per-link multipath channel parameters
# (path loss values / angles) conditioned on link geometry -- i.e. it is the
# "GAN-based channel model"'s real neural architecture: a conditional VAE
# with
#   encoder:  [x, cond] --(Dense+sigmoid)*--> [z_mu, z_log_var]
#   reparam:  z = z_mu + eps * exp(0.5 * z_log_var),  eps ~ N(0, I)
#   decoder:  [z, cond] --(Dense+sigmoid)*--> [x_mu, x_log_var]
# with the decoder additionally descending-sorting the first `nsort` output
# means (used upstream for path-loss values, which are stored sorted) and
# clamping the output log-variance from below by log(out_var_min). This port
# transcribes that exact structure layer-for-layer from the original
# TensorFlow `CondVAE.build_encoder`/`build_decoder`/`reparm` methods into
# self-contained torch (`nn.Linear` + `nn.Sigmoid` in place of
# `tfkl.Dense(activation='sigmoid')`, `torch.sort` in place of `tf.sort`).
# The VAE-loss/`build_vae` training-only machinery and the outer
# `ChanMod`/`link_mod` classifier are not part of the generative forward
# path and are not ported.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class CondVAEEncoder(nn.Module):
    """Port of CondVAE.build_encoder: maps [x, cond] -> [z_mu, z_log_var]."""

    def __init__(self, ndat: int, ncond: int, nlatent: int, nunits_enc: tuple):
        super().__init__()
        layers = []
        in_dim = ndat + ncond
        for nh in nunits_enc:
            layers.append(nn.Linear(in_dim, nh))
            layers.append(nn.Sigmoid())
            in_dim = nh
        self.hidden = nn.Sequential(*layers)
        # z_mu / z_log_var: bias_initializer=None in the original Keras code
        # -> no bias term.
        self.z_mu = nn.Linear(in_dim, nlatent, bias=False)
        self.z_log_var = nn.Linear(in_dim, nlatent, bias=False)

    def forward(self, x, cond):
        dat_cond = torch.cat([x, cond], dim=-1)
        h = self.hidden(dat_cond)
        z_mu = self.z_mu(h)
        z_log_var = self.z_log_var(h)
        return z_mu, z_log_var


class CondVAEDecoder(nn.Module):
    """Port of CondVAE.build_decoder: maps [z, cond] -> [x_mu, x_log_var]."""

    def __init__(
        self, ndat: int, ncond: int, nlatent: int, nunits_enc: tuple, nsort: int, out_var_min: float
    ):
        super().__init__()
        self.ndat = ndat
        self.nsort = nsort
        self.out_var_min = out_var_min

        layers = []
        in_dim = nlatent + ncond
        for nh in nunits_enc:
            # Original decoder reuses `self.nunits_enc` (not `nunits_dec`) for
            # its hidden sizes -- see CondVAE.build_decoder in the source.
            layers.append(nn.Linear(in_dim, nh, bias=False))
            layers.append(nn.Sigmoid())
            in_dim = nh
        self.hidden = nn.Sequential(*layers)

        self.x_mu = nn.Linear(in_dim, ndat, bias=False)
        self.x_log_var = nn.Linear(in_dim, ndat)

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=-1)
        h = self.hidden(z_cond)

        x_mu = self.x_mu(h)
        if self.nsort > 0:
            x_mu_pl = x_mu[:, : self.nsort]
            x_mu_pl, _ = torch.sort(x_mu_pl, dim=-1, descending=True)
            x_mu_ang = x_mu[:, self.nsort :]
            x_mu = torch.cat([x_mu_pl, x_mu_ang], dim=-1)

        x_log_var = self.x_log_var(h)
        x_log_var = torch.clamp(x_log_var, min=float(torch.log(torch.tensor(self.out_var_min))))

        return x_mu, x_log_var


class CondVAE(nn.Module):
    """
    Port of mmwchanmod.learn.models.CondVAE (Conditional VAE class).

    Ports the actual generative forward path used by `ChanMod.path_mod` to
    sample synthetic multipath channel parameters conditioned on link
    geometry: encoder -> reparameterization -> decoder -> sampler.

    Parameters
    ----------
    nlatent : int
        number of latent states
    ndat : int
        number of features in the data to be modeled
    ncond : int
        number of conditional variables
    nunits_enc : tuple of int
        hidden units per encoder/decoder layer (the original code re-uses
        this same tuple for the decoder hidden layers)
    out_var_min : float
        minimum output variance (for improved conditioning)
    nsort : int
        sort the first `nsort` values of the decoder mean output (used for
        path-loss values, which are stored in sorted order)
    """

    def __init__(
        self,
        nlatent: int,
        ndat: int,
        ncond: int,
        nunits_enc: tuple = (25, 10),
        out_var_min: float = 1e-4,
        nsort: int = 0,
    ):
        super().__init__()
        self.nlatent = nlatent
        self.ndat = ndat
        self.ncond = ncond
        self.nunits_enc = nunits_enc
        self.out_var_min = out_var_min
        self.nsort = nsort

        self.encoder = CondVAEEncoder(ndat, ncond, nlatent, nunits_enc)
        self.decoder = CondVAEDecoder(ndat, ncond, nlatent, nunits_enc, nsort, out_var_min)

    def reparm(self, z_mu, z_log_var):
        """Port of CondVAE.reparm: z = z_mu + eps * exp(0.5 * z_log_var)."""
        eps = torch.randn_like(z_mu)
        z = eps * torch.exp(z_log_var * 0.5) + z_mu
        return z

    def forward(self, x, cond):
        """
        Port of the `self.vae` model built in CondVAE.build_vae: takes an
        input sample x (conditioned on `cond`) and outputs the decoder's
        reconstruction mean and log-variance [xhat, x_log_var].
        """
        z_mu, z_log_var = self.encoder(x, cond)
        z_samp = self.reparm(z_mu, z_log_var)
        xhat, x_log_var = self.decoder(z_samp, cond)
        return xhat, x_log_var


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing. The repo's own
# `ChanMod.__init__` defaults are nlatent=10, nunits_enc=(200,80),
# nunits_dec=(80,200) (the decoder ignores nunits_dec and reuses nunits_enc,
# per the source), out_var_min=1e-4; `ndat`/`ncond`/`nsort` are derived at
# runtime from the dataset's npaths_max/rx_types (e.g. nsort=npaths_max for
# the path-loss VAE). We keep nlatent/out_var_min at the real defaults and
# use small ndat/ncond/nunits_enc/nsort for a fast CPU trace.
# ---------------------------------------------------------------------------
_NDAT = 6
_NCOND = 4
_NLATENT = 10
_NSORT = 3


def build_condvae():
    torch.manual_seed(0)
    model = CondVAE(
        nlatent=_NLATENT,
        ndat=_NDAT,
        ncond=_NCOND,
        nunits_enc=(8, 6),
        out_var_min=1e-4,
        nsort=_NSORT,
    )
    model.eval()
    return model


def example_input_condvae():
    torch.manual_seed(0)
    x = torch.randn(5, _NDAT)
    cond = torch.randn(5, _NCOND)
    return (x, cond)


MENAGERIE_ENTRIES = [
    ("CondVAE-ChanMod", "build_condvae", "example_input_condvae", 2020, MENAGERIE_ZOO),
]
