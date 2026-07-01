# SOURCE: vendored from cmzuo11/scMVAE @ master
# https://github.com/cmzuo11/scMVAE/blob/master/scMVAE/MVAE_model.py
# https://github.com/cmzuo11/scMVAE/blob/master/scMVAE/layers.py
# https://github.com/cmzuo11/scMVAE/blob/master/scMVAE/loss_function.py
#
# scMVAE-Concat: a single-cell multi-omics (scRNA + scATAC) mixture-of-
# Gaussians VAE. `Encoder`/`Decoder_ZINB`/`Decoder`/`build_multi_layers`
# (layers.py), `scMVAE_Concat` (MVAE_model.py), and the loss helpers
# `log_zinb_positive`/`binary_cross_entropy`/`mse_loss`/`GMM_loss`
# (loss_function.py) are copied verbatim. loss_function.py's parent module
# also imports matplotlib (unused by the traced forward/loss path), so only
# the needed functions are vendored here to avoid that extra import.
# sklearn-based GMM initialization (`init_gmm_params`) and I/O helpers are
# training-time utilities outside the traced forward path and are omitted.

import collections
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


MENAGERIE_ZOO = "vendored-pytorch"


# --- from loss_function.py (verbatim) ---------------------------------


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    x = x.float()

    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    softplus_pi = F.softplus(-pi)

    log_theta_eps = torch.log(theta + eps)

    log_theta_mu_eps = torch.log(theta + mu + eps)

    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return -torch.sum(res, dim=1)


def binary_cross_entropy(recon_x, x):
    return -torch.sum(
        x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=1
    )


def mse_loss(y_true, y_pred):
    mask = torch.sign(y_true)
    y_pred = y_pred * mask
    ret = torch.pow((y_true - y_pred), 2)
    return torch.sum(ret, dim=1)


def GMM_loss(gamma, c_params, z_params):
    """
    L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
              = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
    """
    mu_c, var_c, pi = c_params
    n_centroids = pi.size(1)
    mu, logvar = z_params
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)

    # log p(z|c)
    logpzc = -0.5 * torch.sum(
        gamma
        * torch.sum(
            math.log(2 * math.pi)
            + torch.log(var_c)
            + torch.exp(logvar_expand) / var_c
            + (mu_expand - mu_c) ** 2 / var_c,
            dim=1,
        ),
        dim=1,
    )
    # log p(c)
    logpc = torch.sum(gamma * torch.log(pi), 1)

    # log q(z|x) or q entropy
    qentropy = -0.5 * torch.sum(1 + logvar + math.log(2 * math.pi), 1)

    # log q(c|x)
    logqcx = torch.sum(gamma * torch.log(gamma), 1)

    kld = -logpzc - logpc + qentropy + logqcx

    return kld


# --- from layers.py (verbatim) -----------------------------------------


def build_multi_layers(layers, use_batch_norm=True, dropout_rate=0.1):
    """Build multilayer linear perceptron"""
    if dropout_rate > 0:
        fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))
                ]
            )
        )

    else:
        fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))
                ]
            )
        )

    return fc_layers


class Encoder(nn.Module):
    # for one modulity
    def __init__(self, layer, hidden, Z_DIMS, dropout_rate=0.1):
        super(Encoder, self).__init__()

        if len(layer) > 1:
            self.fc1 = build_multi_layers(layers=layer, dropout_rate=dropout_rate)

        self.layer = layer
        self.fc_means = nn.Linear(hidden, Z_DIMS)
        self.fc_logvar = nn.Linear(hidden, Z_DIMS)

    def reparametrize(self, means, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(means)
        else:
            return means

    def forward(self, x):
        if len(self.layer) > 1:
            h = self.fc1(x)
        else:
            h = x
        mean_x = self.fc_means(h)
        logvar_x = self.fc_logvar(h)
        latent = self.reparametrize(mean_x, logvar_x)

        return mean_x, logvar_x, latent


class Decoder_ZINB(nn.Module):
    # for scRNA-seq

    def __init__(self, layer, hidden, input_size, dropout_rate=0.1):
        super(Decoder_ZINB, self).__init__()

        if len(layer) > 1:
            self.decoder = build_multi_layers(layer, dropout_rate=dropout_rate)

        self.decoder_scale = nn.Linear(hidden, input_size)
        self.decoder_r = nn.Linear(hidden, input_size)
        self.dropout = nn.Linear(hidden, input_size)

        self.layer = layer

    def forward(self, z, library):
        if len(self.layer) > 1:
            latent = self.decoder(z)
        else:
            latent = z

        normalized_x = F.softmax(self.decoder_scale(latent), dim=1)  # mean gamma

        recon_final = torch.exp(library) * normalized_x  # mu
        disper_x = self.decoder_r(latent)  # theta
        disper_x = torch.exp(disper_x)
        dropout_rate = self.dropout(latent)

        return dict(
            normalized=normalized_x,
            disperation=disper_x,
            imputation=recon_final,
            dropoutrate=dropout_rate,
        )


class Decoder(nn.Module):
    # for scATAC-seq
    def __init__(self, layer, hidden, input_size, Type="Bernoulli", dropout_rate=0.1):
        super(Decoder, self).__init__()

        if len(layer) > 1:
            self.decoder = build_multi_layers(layer, dropout_rate=dropout_rate)

        self.decoder_x = nn.Linear(hidden, input_size)
        self.Type = Type
        self.layer = layer

    def forward(self, z):
        if len(self.layer) > 1:
            latent = self.decoder(z)
        else:
            latent = z

        recon_x = self.decoder_x(latent)

        if self.Type == "Bernoulli":
            Final_x = torch.sigmoid(recon_x)

        elif self.Type == "Gaussian":
            Final_x = F.softmax(recon_x, dim=1)

        elif self.Type == "Gaussian1":
            Final_x = torch.sigmoid(recon_x)

        else:
            Final_x = F.relu(recon_x)

        return Final_x


# --- from MVAE_model.py (verbatim) --------------------------------------


class scMVAE_Concat(nn.Module):
    def __init__(
        self,
        layer_e,
        hidden1,
        Zdim,
        layer_l,
        hidden3,
        layer_d,
        hidden4,
        logchange=True,
        Type="ZINB",
        n_centroids=4,
        penality="GMM",
    ):
        super(scMVAE_Concat, self).__init__()
        # function definition
        self.encoder_x = Encoder(layer_e, hidden1, Zdim)
        self.encoder_l = Encoder(layer_l, hidden3, 1)

        if Type == "ZINB":
            self.decoder_x = Decoder_ZINB(layer_d, hidden4, layer_e[0])

        else:
            self.decoder_x = Decoder(layer_d, hidden4, layer_e[0], Type)

        # parameters definition
        self.logchange = logchange
        self.Type = Type
        self.penality = penality
        self.n_centroids = n_centroids

        self.pi = nn.Parameter(torch.ones(n_centroids) / n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(Zdim, n_centroids))  # mu
        self.var_c = nn.Parameter(torch.ones(Zdim, n_centroids))  # sigma^2

    def reparametrize(self, means, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(means)
        else:
            return means

    def inference(self, X):
        X_ = X
        if self.logchange:
            X_ = torch.log(X_ + 1)

        # latent encoder for x
        mean_z, logvar_z, latent_z = self.encoder_x(X_)
        mean_l, logvar_l, library = self.encoder_l(X_)  # library scale factor

        # decoder for x latent
        if self.Type == "ZINB":
            output = self.decoder_x(latent_z, library)
            normalized_x = output["normalized"]
            disper_x = output["disperation"]
            dropout_rate = output["dropoutrate"]
            recon_final = output["imputation"]
        else:
            recons_x = self.decoder_x(latent_z)
            recon_final = recons_x
            normalized_x = None
            disper_x = None
            dropout_rate = None

        return dict(
            latent_z_mu=mean_z,
            latent_z_logvar=logvar_z,
            latent_z=latent_z,
            latent_l_mu=mean_l,
            latent_l_logvar=logvar_l,
            normalized=normalized_x,
            disperation=disper_x,
            imputation=recon_final,
            dropoutrate=dropout_rate,
        )

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        if self.Type == "ZINB":
            loss = log_zinb_positive(x, px_rate, px_r, px_dropout)

        elif self.Type == "Bernoulli":
            loss = binary_cross_entropy(x, px_rate)

        else:
            loss = mse_loss(x, px_rate)

        return loss

    def forward(self, X, local_l_mean=None, local_l_var=None):
        result = self.inference(X)

        latent_z_mu = result["latent_z_mu"]
        latent_z_logvar = result["latent_z_logvar"]
        latent_z = result["latent_z"]

        latent_l_mu = result["latent_l_mu"]
        latent_l_logvar = result["latent_l_logvar"]

        imputation = result["imputation"]
        disperation = result["disperation"]
        dropoutrate = result["dropoutrate"]

        # KL Divergence for library factor
        if local_l_mean is not None:
            kl_divergence_l = torch.distributions.kl_divergence(
                torch.distributions.Normal(latent_l_mu, latent_l_logvar),
                torch.distributions.Normal(local_l_mean, torch.sqrt(local_l_var)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0)

        # KL Divergence for latent code
        if self.penality == "GMM":
            gamma, mu_c, var_c, pi = self.get_gamma(latent_z)
            kl_divergence_z = GMM_loss(gamma, (mu_c, var_c, pi), (latent_z_mu, latent_z_logvar))

        else:
            mean = torch.zeros_like(latent_z_mu)
            scale = torch.ones_like(latent_z_logvar)
            kl_divergence_z = torch.distributions.kl_divergence(
                torch.distributions.Normal(latent_z_mu, latent_z_logvar),
                torch.distributions.Normal(mean, scale),
            ).sum(dim=1)

        reconst_loss = self.get_reconstruction_loss(X, imputation, disperation, dropoutrate)

        return reconst_loss, kl_divergence_l, kl_divergence_z

    def get_gamma(self, z):
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N, 1)  # NxK
        mu_c = self.mu_c.repeat(N, 1, 1)  # NxDxK
        var_c = self.var_c.repeat(N, 1, 1)  # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = (
            torch.exp(
                torch.log(pi)
                - torch.sum(
                    0.5 * torch.log(2 * math.pi * var_c) + (z - mu_c) ** 2 / (2 * var_c), dim=1
                )
            )
            + 1e-10
        )
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi


def build_scmvae():
    n_genes = 64
    return scMVAE_Concat(
        layer_e=[n_genes, 64],
        hidden1=64,
        Zdim=10,
        layer_l=[n_genes, 64],
        hidden3=64,
        layer_d=[10, 64],
        hidden4=64,
        Type="ZINB",
        n_centroids=4,
        penality="GMM",
    )


def example_input_scmvae():
    batch = 8
    n_genes = 64
    return (torch.rand(batch, n_genes) * 5.0,)


MENAGERIE_ENTRIES = [
    ("scMVAE-Concat", "build_scmvae", "example_input_scmvae", 2021, "vendored-pytorch"),
]
