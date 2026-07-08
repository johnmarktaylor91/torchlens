# SOURCE: vendored from bm2-lab/scMVP @ master
# https://github.com/bm2-lab/scMVP/blob/master/scMVP/models/multi_vae_attention.py
# https://github.com/bm2-lab/scMVP/blob/master/scMVP/models/modules.py
# https://github.com/bm2-lab/scMVP/blob/master/scMVP/models/log_likelihood.py
# https://github.com/bm2-lab/scMVP/blob/master/scMVP/models/utils.py
#
# scMVP Multi_VAE_Attention (mode="mm-vae"): a multi-omics (scRNA + scATAC)
# variational autoencoder with per-modality self-attention encoders, a
# shared joint encoder/decoder, and a GMM-structured latent prior with a
# joint-consistency loss across RNA/ATAC/joint reconstructions. All classes
# below (modules.py in full, the log_likelihood.py functions actually
# imported by multi_vae_attention.py, utils.py's one_hot, and the
# Multi_VAE_Attention class itself) are copied verbatim; only the
# `scMVP.models.*` package-relative imports are collapsed since everything
# now lives in one file. sklearn-based GMM initialization
# (`init_gmm_params*`) and posterior/dataset utilities are training-time
# helpers outside the traced forward path and are omitted; other subclasses
# in these files (LDVAE, Classifier, TOTALVI-related decoders/encoders) are
# not required by Multi_VAE_Attention("mm-vae") and are omitted.

import collections
import math
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.distributions import Normal, kl_divergence as kl
from torch.nn import ModuleList


MENAGERIE_ZOO = "vendored-pytorch"


# --- from utils.py (verbatim: one_hot) ----------------------------------


def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


# --- from log_likelihood.py (verbatim, functions used by Multi_VAE_Attention) ---


def binary_cross_entropy(x, recon_x, eps=1e-8):
    recon_x = torch.sigmoid(recon_x)
    res = x * torch.log(recon_x + eps) + (1 - x) * torch.log(1 - recon_x + eps)
    #    print(torch.mean(recon_x))
    return res


def mean_square_error_positive(x, recon_x):
    # res = (x - recon_x + 1)*(x - recon_x + 1)
    # res[x==0] = 0
    res = torch.abs((x - recon_x))
    res[x == 0] = 0  # test this property
    return res


def log_zip_positive(x, mu, pi, eps=1e-8):
    # the likelihood of zero probability p(x=0) = -softplus(-pi)+softplus(-pi-mu)
    softplus_pi = F.softplus(-pi)
    softplus_mu_pi = F.softplus(-pi - mu)
    case_zero = -softplus_pi + softplus_mu_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    # the likelihood of p(x>0) = -softplus(-pi) - pi - mu +x*ln(mu) - ln(x!)
    case_non_zero = -softplus_pi - pi - mu + x * torch.log(mu + eps) - torch.lgamma(x + 1)
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
    Note: All inputs are torch Tensors
    log likelihood (scalar) of a minibatch according to a zinb model.
    Notes:
    We parametrize the bernoulli using the logits, hence the softplus functions appearing

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant
    """

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

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

    return res


def log_nb_positive(x, mu, theta, eps=1e-8):
    """
    Note: All inputs should be torch Tensors
    log likelihood (scalar) of a minibatch according to a nb model.

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res


# --- from modules.py (verbatim) ------------------------------------------


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_cat_list: A list containing, for each category of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: Whether to have `BatchNorm` layers or not
    :param use_relu: Whether to have `ReLU` layers or not
    :param bias: Whether to learn bias in linear layers or not

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        # use_batch_norm: bool = False,
        use_relu: bool = True,
        # use_relu: bool = False,
        bias: bool = True,
        RNA_mode=True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out, bias=bias),
                            # Below, 0.01 and 0.001 are the default values for `momentum` and `eps` from
                            # the tensorflow implementation of batch norm; we're using those settings
                            # here too so that the results match our old tensorflow code. The default
                            # setting from pytorch would probably be fine too but we haven't tested that.
                            nn.LayerNorm(n_out, eps=0.0001),
                            nn.LeakyReLU() if RNA_mode else None,
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.0001)
                            if use_batch_norm
                            else None,
                            nn.LeakyReLU() if use_relu else nn.ReLU(),
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :param instance_id: Use a specific conditional instance normalization (batchnorm)
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(cat_list), (
            "nb. categorical args provided doesn't match init. params."
        )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (n_cat and cat is None), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x


# Classifer
class Classifer(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
    ):
        super().__init__()
        self.classifer = nn.Sequential(nn.Linear(n_input, n_output), nn.Softmax(dim=-1))

    def forward(
        self,
        z: torch.Tensor,
        *cat_list: int,
    ):
        predict_z = self.classifer(z)
        return predict_z


# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Encoder_nb
class Encoder_nb(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Encoder_nb_layers
class Encoder_nb_layers(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.prelayers = nn.Sequential(
            nn.Linear(n_input, 10 * n_hidden),
            nn.Linear(10 * n_hidden, 5 * n_hidden),
            nn.Linear(5 * n_hidden, n_hidden),
            nn.LeakyReLU(),
        )
        self.encoder = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        pre_x = self.prelayers(x)
        q = self.encoder(pre_x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Encoder_peak_layers
class Encoder_peak_layers(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.prelayers = nn.Sequential(
            nn.Linear(n_input, 50 * n_hidden),
            nn.Linear(50 * n_hidden, 10 * n_hidden),
            nn.Linear(10 * n_hidden, n_hidden),
            nn.LeakyReLU(),
        )
        self.encoder = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        pre_x = self.prelayers(x)
        q = self.encoder(pre_x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Encoder_nb_attention
class Encoder_nb_attention(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list) * self.px_decoder_aux(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Encoder_nb_selfattention_layer
class Encoder_nb_selfattention_layer(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.encoder1 = FCLayers(
            n_in=n_input,
            n_out=8 * n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=8 * n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.w_q1 = nn.Linear(8 * n_hidden, 8 * n_hidden)
        self.w_k1 = nn.Linear(8 * n_hidden, 8 * n_hidden)
        self.w_v1 = nn.Linear(8 * n_hidden, 8 * n_hidden)
        self.encoder3 = FCLayers(
            n_in=8 * n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.w_q3 = nn.Linear(n_hidden, n_hidden)
        self.w_k3 = nn.Linear(n_hidden, n_hidden)
        self.w_v3 = nn.Linear(n_hidden, n_hidden)

        self.do = nn.Dropout(dropout_rate)

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        x: torch.Tensor,
        *cat_list: int,
    ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        q = self.encoder1(x, *cat_list)
        assert q.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q1(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        K = self.w_k1(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        V = self.w_v1(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q = torch.matmul(attention, V).view(q.shape[0], q.shape[1])
        q = self.encoder3(q, *cat_list)
        Q = self.w_q3(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        K = self.w_k3(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        V = self.w_v3(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q = torch.matmul(attention, V).view(q.shape[0], q.shape[1])

        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Encoder_nb_selfattention
class Encoder_nb_selfattention(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.px_encoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)

        self.do = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(n_hidden, eps=0.0001)

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        x: torch.Tensor,
        *cat_list: int,
    ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        assert q.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        K = self.w_k(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        V = self.w_v(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q_a = torch.matmul(attention, V).view(q.shape[0], q.shape[1])

        q_m = self.mean_encoder(q_a)
        q_v = torch.exp(self.var_encoder(q_a)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# encoder_libary
class Encoder_l(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            RNA_mode=False,
            use_relu=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# encoder_mse
class Encoder_mse(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Linear(n_input, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Multi-Encoder-nb
class Multi_Encoder_nb(nn.Module):
    def __init__(
        self,
        RNA_input: int,
        ATAC_input,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.scRNA_encoder = FCLayers(
            n_in=RNA_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.scATAC_encoder = FCLayers(
            n_in=ATAC_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.concat1 = nn.Linear(2 * n_hidden, n_hidden)
        self.concat2 = nn.Linear(n_hidden, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: list, *cat_list: int):
        # Parameters for latent distribution
        if x.__len__() != 2:
            raise ValueError(
                "Input training data should be 2 data types(RNA and ATAC),"
                "but input was only {}.format(x.__len__())"
            )
        if not torch.is_tensor(x[0]):
            raise ValueError("training data should be a tensor!")

        q1 = self.scRNA_encoder(x[0], *cat_list)
        q2 = self.scATAC_encoder(x[1], *cat_list)
        q = self.concat2(self.concat1(torch.cat((q1, q2), 1)))
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Multi-Encoder-nb-attention
class Multi_Encoder_nb_attention(nn.Module):
    def __init__(
        self,
        RNA_input: int,
        ATAC_input,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.scRNA_encoder = FCLayers(
            n_in=RNA_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.scATAC_encoder = FCLayers(
            n_in=ATAC_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.RNA_encoder_aux = nn.Sequential(
            nn.Linear(RNA_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.ATAC_encoder_aux = nn.Sequential(
            nn.Linear(ATAC_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.concat = nn.Linear(2 * n_hidden, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: list, *cat_list: int):
        # Parameters for latent distribution
        if x.__len__() != 2:
            raise ValueError(
                "Input training data should be 2 data types(RNA and ATAC),"
                "but input was only {}.format(x.__len__())"
            )
        if not torch.is_tensor(x[0]):
            raise ValueError("training data should be a tensor!")

        q1 = self.scRNA_encoder(x[0], *cat_list) * self.RNA_encoder_aux(x[0])
        q2 = self.scATAC_encoder(x[1], *cat_list) * self.ATAC_encoder_aux(x[1])
        q = self.concat(torch.cat((q1, q2), 1))
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Multi-Encoder-self-attention
class Multi_Encoder_nb_SelfAttention(nn.Module):
    def __init__(
        self,
        RNA_input: int,
        ATAC_input,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.scRNA_encoder = FCLayers(
            n_in=RNA_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.scATAC_encoder = FCLayers(
            n_in=ATAC_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.RNA_encoder_aux = nn.Sequential(
            nn.Linear(RNA_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)

        self.do = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(n_hidden, eps=0.0001)

        self.concat = nn.Linear(2 * n_hidden, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: list, *cat_list: int):
        # Parameters for latent distribution
        if x.__len__() != 2:
            raise ValueError(
                "Input training data should be 2 data types(RNA and ATAC),"
                "but input was only {}.format(x.__len__())"
            )
        if not torch.is_tensor(x[0]):
            raise ValueError("training data should be a tensor!")

        q1 = self.scRNA_encoder(x[0], *cat_list) * self.RNA_encoder_aux(x[0])
        q2 = self.scATAC_encoder(x[1], *cat_list)
        assert q2.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(q2).view(q2.shape[0], self.n_heads, q2.shape[1] // self.n_heads, -1)
        K = self.w_k(q2).view(q2.shape[0], self.n_heads, q2.shape[1] // self.n_heads, -1)
        V = self.w_v(q2).view(q2.shape[0], self.n_heads, q2.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q2 = torch.matmul(attention, V).view(q2.shape[0], q2.shape[1])

        q = self.concat(torch.cat((q1, q2), 1))
        q = self.layernorm(q)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Multi-Encoder
class Multi_Encoder(nn.Module):
    def __init__(
        self,
        RNA_input: int,
        ATAC_input,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.scRNA_encoder = FCLayers(
            n_in=RNA_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.scATAC_encoder = FCLayers(
            n_in=ATAC_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.concat1 = nn.Linear(2 * n_hidden, n_hidden)
        self.concat2 = nn.Linear(n_hidden, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: list, *cat_list: int):
        # Parameters for latent distribution
        if x.__len__() != 2:
            raise ValueError(
                "Input training data should be 2 data types(RNA and ATAC),"
                "but input was only {}.format(x.__len__())"
            )
        if not torch.is_tensor(x[0]):
            raise ValueError("training data should be a tensor!")

        q1 = self.scRNA_encoder(x[0], *cat_list)
        q2 = self.scATAC_encoder(x[1], *cat_list)
        q = self.concat2(self.concat1(torch.cat((q1, q2), 1)))
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Multi-Decoder-nb-log-peak
class Multi_Decoder_nb_log(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None,
    ):
        super().__init__()

        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean gamma
        if is_cluster:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2 * n_hidden),
                nn.Linear(2 * n_hidden, RNA_output),
                nn.Softmax(dim=-1),
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)
        # auxiliary decoder
        self.px_rna_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, RNA_output), nn.Sigmoid()
        )
        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        self.atac_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output)
        )

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Sigmoid()
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))
        self.libaray_atac_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))

    def forward(
        self,
        z: torch.Tensor,
        z_c: torch.Tensor,
        *cat_list: int,
        libary_scale=None,
        gamma=None,
        libary_atac=None,
    ):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = self.cluster_decoder(gamma, *cat_list)
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)

        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        if libary_scale is not None:
            p_rna_rate = (
                torch.exp(libary_scale) * p_rna_scale * self.px_rna_decoder_aux(z)
            )  # libary_scale
        else:
            p_rna_rate = (
                torch.exp(libaray_gene) * p_rna_scale * self.px_rna_decoder_aux(z)
            )  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12)  # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        p_atac = self.scATAC_decoder(z, *cat_list)
        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        else:
            p_atac_scale = self.atac_scale_decoder(
                torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1)
            )

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_mean = torch.softmax(p_atac_scale, dim=-1) * self.px_atac_decoder_aux(
            z
        )  # for zinp and zip loss
        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_mean

        return (
            p_rna_scale,
            p_rna_r,
            p_rna_rate,
            p_rna_dropout,
            p_atac_scale,
            p_atac_r,
            p_atac_mean,
            p_atac_dropout,
        )


# Multi-Dncoder-nb-log RNA count
class Multi_Decoder_nb_log_peak(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None,
    ):
        super().__init__()

        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean gamma
        if is_cluster:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2 * n_hidden),
                nn.Linear(2 * n_hidden, RNA_output),
                nn.Softmax(dim=-1),
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)
        # auxiliary decoder
        self.px_rna_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, RNA_output), nn.Sigmoid()
        )
        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        self.atac_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output), nn.Sigmoid()
        )

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Softmax(dim=-1)
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))
        self.libaray_atac_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))

    def forward(
        self,
        z: torch.Tensor,
        z_c: torch.Tensor,
        *cat_list: int,
        libary_scale=None,
        gamma=None,
        libary_atac=None,
    ):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = self.cluster_decoder(gamma, *cat_list)
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)

        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        if libary_scale is not None:
            p_rna_rate = (
                torch.exp(libary_scale) * p_rna_scale * self.px_rna_decoder_aux(z)
            )  # libary_scale
        else:
            p_rna_rate = (
                torch.exp(libaray_gene) * p_rna_scale * self.px_rna_decoder_aux(z)
            )  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12)  # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        p_atac = self.scATAC_decoder(z, *cat_list)
        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        else:
            p_atac_scale = self.atac_scale_decoder(
                torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1)
            )

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_scale = p_atac_scale * self.px_atac_decoder_aux(z)  # for zinp and zip loss
        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_scale

        return (
            p_rna_scale,
            p_rna_r,
            p_rna_rate,
            p_rna_dropout,
            p_atac_scale,
            p_atac_r,
            p_atac_mean,
            p_atac_dropout,
        )


# Multi-Dncoder-nb-selfattention
class Multi_Decoder_nb_SelfAttention(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean gamma
        if is_cluster:
            # release version 210228
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2 * n_hidden),
                nn.Linear(2 * n_hidden, RNA_output),
                nn.Softmax(dim=-1),
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )

        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)
        # auxiliary decoder
        self.px_rna_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, RNA_output), nn.Sigmoid()
        )

        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )

        self.atac_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output), nn.Sigmoid()
        )

        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.do = nn.Dropout(0.01)

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Softmax(dim=-1)
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))
        self.libaray_atac_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))

    def forward(
        self,
        z: torch.Tensor,
        z_c: torch.Tensor,
        *cat_list: int,
        libary_scale=None,
        gamma=None,
        libary_atac=None,
    ):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = self.cluster_decoder(gamma, *cat_list)
            # test version 210302
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)

        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        if libary_scale is not None:
            p_rna_rate = (
                torch.exp(libary_scale) * p_rna_scale * self.px_rna_decoder_aux(z)
            )  # libary_scale
        else:
            p_rna_rate = (
                torch.exp(libaray_gene) * p_rna_scale * self.px_rna_decoder_aux(z)
            )  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12)  # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        # The decoder returns values for the parameters of the ZIP distribution of scATAC-seq
        p_atac = self.scATAC_decoder(z, *cat_list)
        assert p_atac.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(p_atac).view(
            p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1
        )
        K = self.w_k(p_atac).view(
            p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1
        )
        V = self.w_v(p_atac).view(
            p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1
        )
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        p_atac = torch.matmul(attention, V).view(p_atac.shape[0], p_atac.shape[1])

        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))
        else:
            p_atac_scale = self.atac_scale_decoder(
                torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1)
            )

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_scale = p_atac_scale * self.px_atac_decoder_aux(z)  # for zinp and zip loss

        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_scale
        else:
            p_atac_mean = torch.exp(libaray_atac) * p_atac_scale

        return (
            p_rna_scale,
            p_rna_r,
            p_rna_rate,
            p_rna_dropout,
            p_atac_scale,
            p_atac_r,
            p_atac_mean,
            p_atac_dropout,
        )


# Multi-Dncoder-nb
class Multi_Decoder_nb(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None,
    ):
        super().__init__()

        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean gamma
        if is_cluster:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2 * n_hidden),
                nn.Linear(2 * n_hidden, RNA_output),
                nn.Softmax(dim=-1),
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)
        # auxiliary decoder
        self.px_rna_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, RNA_output), nn.Sigmoid()
        )
        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        self.atac_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output)
        )

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Sigmoid()
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))
        self.libaray_atac_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))

    def forward(
        self,
        z: torch.Tensor,
        z_c: torch.Tensor,
        *cat_list: int,
        libary_scale=None,
        gamma=None,
        libary_atac=None,
    ):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        # print(gamma)
        if gamma is not None:
            cluster_temp = self.cluster_decoder(gamma, *cat_list)
            # test version 210302
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
            # release version 210228
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)

        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        if libary_scale is not None:
            p_rna_rate = (libary_scale) * p_rna_scale * self.px_rna_decoder_aux(z)  # libary_scale
        else:
            p_rna_rate = (
                (libaray_gene) * p_rna_scale * self.px_rna_decoder_aux(z)
            )  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12)  # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        # The decoder returns values for the parameters of the ZIP distribution of scATAC-seq
        p_atac = self.scATAC_decoder(z, *cat_list)
        if gamma is not None:
            # test version 210302
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        else:
            p_atac_scale = self.atac_scale_decoder(
                torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1)
            )

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_mean = torch.softmax(p_atac_scale, dim=-1) * self.px_atac_decoder_aux(
            z
        )  # for zinp and zip loss
        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_mean

        return (
            p_rna_scale,
            p_rna_r,
            p_rna_rate,
            p_rna_dropout,
            p_atac_scale,
            p_atac_r,
            p_atac_mean,
            p_atac_dropout,
        )


# Multi-Dncoder
class Multi_Decoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None,
    ):
        super().__init__()

        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        # mean gamma
        if is_cluster:
            # release version 210228
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2 * n_hidden),
                nn.Linear(2 * n_hidden, RNA_output),
                nn.Softmax(dim=-1),
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)

        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        self.atac_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output)
        )

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Sigmoid()
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))
        self.libaray_atac_scale_decoder = nn.Sequential(nn.Linear(n_hidden, 1))

    def forward(
        self,
        z: torch.Tensor,
        z_c: torch.Tensor,
        *cat_list: int,
        libary_scale=None,
        gamma=None,
        libary_atac=None,
    ):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = self.cluster_decoder(gamma, *cat_list)
            # test version 210302
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)

        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        if libary_scale is not None:
            p_rna_rate = torch.exp(libary_scale) * p_rna_scale  # libary_scale
        else:
            p_rna_rate = torch.exp(libaray_gene) * p_rna_scale  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12)  # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        # The decoder returns values for the parameters of the ZIP distribution of scATAC-seq
        p_atac = self.scATAC_decoder(z, *cat_list)
        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        else:
            p_atac_scale = self.atac_scale_decoder(
                torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1)
            )

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_mean = torch.softmax(p_atac_scale, dim=-1) * self.px_atac_decoder_aux(
            z
        )  # for zinp and zip loss
        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_mean

        return (
            p_rna_scale,
            p_rna_r,
            p_rna_rate,
            p_rna_dropout,
            p_atac_scale,
            p_atac_r,
            p_atac_mean,
            p_atac_dropout,
        )


# Decoder
class DecoderSCVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class DecoderSCVI_nb_rna(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_output), nn.Sigmoid()
        )
        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = (library) * px_scale  # torch.clamp( , max=12) for scaled RNA data
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class DecoderSCVI_nb(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_output), nn.Sigmoid()
        )
        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


# Decoder_nb_selfattention
class DecoderSCVI_nb_Selfattention(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_heads: int = 8,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.n_heads = n_heads
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_output), nn.Sigmoid()
        )
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.do = nn.Dropout(0.01)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        assert px.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        K = self.w_k(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        V = self.w_v(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q_a = torch.matmul(attention, V).view(px.shape[0], px.shape[1])
        px_scale = self.px_scale_decoder(q_a) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(q_a)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(q_a) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class DecoderSCVI_Peak(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )
        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


# Decoder_peak_selfattention
class DecoderSCVI_Peak_Selfattention(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_heads: int = 8,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.n_heads = n_heads
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.do = nn.Dropout(0.01)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        assert px.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        K = self.w_k(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        V = self.w_v(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))

        q_a = torch.matmul(attention, V).view(px.shape[0], px.shape[1])
        px_scale = self.px_scale_decoder(q_a) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(q_a)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(q_a) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


# Decoder_peak_selfattention_layer
class DecoderSCVI_Peak_Selfattention_Layer(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        self.px_decoder1 = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.w_q1 = nn.Linear(n_hidden, n_hidden)
        self.w_k1 = nn.Linear(n_hidden, n_hidden)
        self.w_v1 = nn.Linear(n_hidden, n_hidden)

        self.px_decoder2 = FCLayers(
            n_in=n_hidden,
            n_out=8 * n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=8 * n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.w_q2 = nn.Linear(8 * n_hidden, 8 * n_hidden)
        self.w_k2 = nn.Linear(8 * n_hidden, 8 * n_hidden)
        self.w_v2 = nn.Linear(8 * n_hidden, 8 * n_hidden)

        self.do = nn.Dropout(0.01)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(8 * n_hidden, n_output), nn.Sigmoid())

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(8 * n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(8 * n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder1(z, *cat_list)
        assert px.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q1(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        K = self.w_k1(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        V = self.w_v1(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        px = torch.matmul(attention, V).view(px.shape[0], px.shape[1])

        px = self.px_decoder2(px, *cat_list)
        Q = self.w_q2(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        K = self.w_k2(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        V = self.w_v2(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q_a = torch.matmul(attention, V).view(px.shape[0], px.shape[1])

        px_scale = self.px_scale_decoder(q_a) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(q_a)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(q_a) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class DecoderSCVI_mse(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = nn.Linear(n_input, n_hidden)

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, 2 * n_hidden), nn.Linear(2 * n_hidden, n_output)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class LinearDecoderSCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super(LinearDecoderSCVI, self).__init__()

        # mean gamma
        self.n_batches = n_cat_list[0]  # Just try a simple case for now
        if self.n_batches > 1:
            self.batch_regressor = nn.Linear(self.n_batches - 1, n_output, bias=False)
        else:
            self.batch_regressor = None

        self.factor_regressor = nn.Linear(n_input, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_input, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        # The decoder returns values for the parameters of the ZINB distribution
        p1_ = self.factor_regressor(z)
        if self.n_batches > 1:
            one_hot_cat = one_hot(cat_list[0], self.n_batches)[:, :-1]
            p2_ = self.batch_regressor(one_hot_cat)
            raw_px_scale = p1_ + p2_
        else:
            raw_px_scale = p1_

        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z)
        px_rate = torch.exp(library) * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        :param x: tensor with shape ``(n_input,)``
        :param cat_list: list of category membership(s) for this sample
        :return: Mean and variance tensors of shape ``(n_output,)``
        :rtype: 2-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m, p_v


class MultiEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_input_list: List[int],
        n_output: int,
        n_hidden: int = 128,
        n_layers_individual: int = 1,
        n_layers_shared: int = 2,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, head_id: int, *cat_list: int):
        q = self.encoders[head_id](x, *cat_list)
        q = self.encoder_shared(q, *cat_list)

        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent


class MultiDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_conditioned: int = 32,
        n_hidden_shared: int = 128,
        n_layers_conditioned: int = 1,
        n_layers_shared: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        n_out = n_hidden_conditioned if n_layers_shared else n_hidden_shared
        if n_layers_conditioned:
            self.px_decoder_conditioned = FCLayers(
                n_in=n_input,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_layers=n_layers_conditioned,
                n_hidden=n_hidden_conditioned,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_out
        else:
            self.px_decoder_conditioned = None
            n_in = n_input

        if n_layers_shared:
            self.px_decoder_final = FCLayers(
                n_in=n_in,
                n_out=n_hidden_shared,
                n_cat_list=[],
                n_layers=n_layers_shared,
                n_hidden=n_hidden_shared,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_hidden_shared
        else:
            self.px_decoder_final = None

        self.px_scale_decoder = nn.Sequential(nn.Linear(n_in, n_output), nn.Softmax(dim=-1))
        self.px_r_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: int,
        library: torch.Tensor,
        dispersion: str,
        *cat_list: int,
    ):
        px = z
        if self.px_decoder_conditioned:
            px = self.px_decoder_conditioned(px, *cat_list, instance_id=dataset_id)
        if self.px_decoder_final:
            px = self.px_decoder_final(px, *cat_list)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_r, px_rate, px_dropout


class DecoderTOTALVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a linear decoder

    :param n_input: The dimensionality of the input (latent space)
    :param n_output_genes: The dimensionality of the output (gene space)
    :param n_output_proteins: The dimensionality of the output (protein space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    """

    def __init__(
        self,
        n_input: int,
        n_output_genes: int,
        n_output_proteins: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.n_output_genes = n_output_genes
        self.n_output_proteins = n_output_proteins

        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden + n_input, n_output_genes), nn.Softmax(dim=-1)
        )

        # background mean first decoder
        self.py_back_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # background mean parameters second decoder
        self.py_back_mean_log_alpha = nn.Linear(n_hidden + n_input, n_output_proteins)
        self.py_back_mean_log_beta = nn.Linear(n_hidden + n_input, n_output_proteins)

        # foreground increment decoder step 1
        self.py_fore_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # foreground increment decoder step 2
        self.py_fore_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden + n_input, n_output_proteins), nn.ReLU()
        )

        # dropout (mixture component for proteins, ZI probability for genes)
        self.sigmoid_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.px_dropout_decoder_gene = nn.Linear(n_hidden + n_input, n_output_genes)

        self.py_background_decoder = nn.Linear(n_hidden + n_input, n_output_proteins)

    def forward(self, z: torch.Tensor, library_gene: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns local parameters for the ZINB distribution for genes
         #. Returns local parameters for the Mixture NB distribution for proteins

         We use the dictionary `px_` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary `py_` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. `scale` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         `back_alpha` and `back_beta` are the posterior parameters for `rate_back`.  `fore_scale` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

        :param z: tensor with shape ``(n_input,)``
        :param library_gene: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 3-tuple (first 2-tuple :py:class:`dict`, last :py:class:`torch.Tensor`)
        """
        px_ = {}
        py_ = {}

        px = self.px_decoder(z, *cat_list)
        px_cat_z = torch.cat([px, z], dim=-1)
        px_["scale"] = self.px_scale_decoder(px_cat_z)
        px_["rate"] = library_gene * px_["scale"]

        py_back = self.py_back_decoder(z, *cat_list)
        py_back_cat_z = torch.cat([py_back, z], dim=-1)

        py_["back_alpha"] = self.py_back_mean_log_alpha(py_back_cat_z)
        py_["back_beta"] = torch.exp(self.py_back_mean_log_beta(py_back_cat_z))
        log_pro_back_mean = Normal(py_["back_alpha"], py_["back_beta"]).rsample()
        py_["rate_back"] = torch.exp(log_pro_back_mean)

        py_fore = self.py_fore_decoder(z, *cat_list)
        py_fore_cat_z = torch.cat([py_fore, z], dim=-1)
        py_["fore_scale"] = self.py_fore_scale_decoder(py_fore_cat_z) + 1
        py_["rate_fore"] = py_["rate_back"] * py_["fore_scale"]

        p_mixing = self.sigmoid_decoder(z, *cat_list)
        p_mixing_cat_z = torch.cat([p_mixing, z], dim=-1)
        px_["dropout"] = self.px_dropout_decoder_gene(p_mixing_cat_z)
        py_["mixing"] = self.py_background_decoder(p_mixing_cat_z)

        return (px_, py_, log_pro_back_mean)


# Encoder
class EncoderTOTALVI(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    :distribution: Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.z_encoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.z_mean_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_encoder = nn.Linear(n_hidden, n_output)

        self.l_gene_encoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)

        self.distribution = distribution

        def identity(x):
            return x

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

        self.l_transformation = torch.exp

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary `latent` contains the samples of the latent variables, while `untran_latent`
        contains the untransformed versions of these latent variables. For example, the library size is log normally distributed,
        so `untran_latent["l"]` gives the normal sample that was later exponentiated to become `latent["l"]`.
        The logistic normal distribution is equivalent to applying softmax to a normal sample.

        :param data: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 6-tuple. First 4 of :py:class:`torch.Tensor`, next 2 are `dict` of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(data, *cat_list)
        qz = self.z_encoder(q)
        qz_m = self.z_mean_encoder(qz)
        qz_v = torch.exp(self.z_var_encoder(qz)) + 1e-4
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)

        ql_gene = self.l_gene_encoder(q)
        ql_m = self.l_gene_mean_encoder(ql_gene)
        ql_v = torch.exp(self.l_gene_var_encoder(ql_gene)) + 1e-4
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_m, ql_v), max=15)
        library_gene = self.l_transformation(log_library_gene)

        latent = {}
        untran_latent = {}
        latent["z"] = z
        latent["l"] = library_gene
        untran_latent["z"] = untran_z
        untran_latent["l"] = log_library_gene

        return qz_m, qz_v, ql_m, ql_v, latent, untran_latent


# --- from multi_vae_attention.py (verbatim) ------------------------------
class Multi_VAE_Attention(nn.Module):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param mode: One of the following:
        * ``'vae'`` -single channel auto-encoder decoder neural framework for scRNA-seq data
        * ``'mm-vae'`` -multi-channels auto-encoder decoder neural framework for scRNA and scATAC data
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

    def __init__(
        self,
        RNA_input: int,
        ATAC_input: int = 0,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_centroids: int = 20,
        n_alfa: float = 1.0,
        dropout_rate: float = 0.1,
        mode="vae",
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        isLibrary: bool = True,
        is_cluster: bool = True,
        classifer_num: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_input_atac = ATAC_input
        self.n_input_RNA = RNA_input
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_centroids = n_centroids
        self.alfa = n_alfa
        self.isLibrary = isLibrary
        self.is_cluster = is_cluster
        self.classifer_num = classifer_num

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(RNA_input))
            self.p_atac_r = torch.nn.Parameter(torch.randn(ATAC_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(RNA_input, n_batch))
            self.p_atac_r = torch.nn.Parameter(torch.randn(ATAC_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(RNA_input, n_labels))
            self.p_atac_r = torch.nn.Parameter(torch.randn(ATAC_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        if self.mode == "vae":
            # z encoder goes from the n_input-dimensional data to an n_latent-d
            # latent space representation
            self.z_encoder = Encoder(
                RNA_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            # l encoder goes from n_input-dimensional data to 1-d library size
            self.l_encoder = Encoder(
                RNA_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
            )
            # decoder goes from n_latent-dimensional space to n_input-d data
            self.decoder = DecoderSCVI(
                n_latent,
                RNA_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
        elif self.mode == "mm-vae":
            if ATAC_input <= 0:
                raise ValueError(
                    "Input size of ATAC channel should be positive value,"
                    "but input was {}.format(self.ATAC_input)"
                )

            # init c_params
            self.pi = nn.Parameter(torch.ones(n_centroids) / n_centroids, requires_grad=True)  # pc
            self.mu_c = nn.Parameter(torch.zeros(n_latent, n_centroids), requires_grad=True)  # mu
            self.var_c = nn.Parameter(
                torch.ones(n_latent, n_centroids), requires_grad=True
            )  # sigma^2
            self.counter = nn.Parameter(torch.zeros(2), requires_grad=False)  # sigma^2

            if self.classifer_num > 0:
                self.classifer = Classifer(
                    n_latent,
                    self.classifer_num,
                )

            self.RNA_encoder = Encoder_nb_attention(
                RNA_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            self.ATAC_encoder = Encoder_nb_selfattention(
                ATAC_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            self.concatenter = nn.Linear(2 * self.n_latent, self.n_latent)
            if self.isLibrary == True:
                # l encoder goes from n_input-dimensional data to 1-d library size
                self.l_encoder = Encoder_l(
                    RNA_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
                )
            self.RNA_ATAC_encoder = Multi_Encoder_nb_SelfAttention(
                RNA_input,
                ATAC_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            self.RNA_ATAC_decoder = Multi_Decoder_nb_SelfAttention(
                n_latent,
                RNA_input,
                ATAC_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
                is_cluster=is_cluster,
                n_cluster=n_centroids,
            )
        else:
            raise ValueError(
                "mode must be one of ['vae', 'mm-vae' ], but input was {}.format(self.mode)"
            )

    def get_params(self):
        params = [self.pi, self.mu_c, self.var_c]
        return params

    def get_latents(self, x_rna, y=None, x_atac=None):
        r"""returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z([x_rna, x_atac], y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=True):
        r"""samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x[0] = torch.log(1 + x[0])
            x[1] = torch.log(1 + x[1])

        qz_rna_m, qz_rna_v, rna_z = self.RNA_encoder(x[0], None)
        qz_atac_m, qz_atac_v, atac_z = self.ATAC_encoder(x[1], None)
        qz_m, qz_v, z = self.RNA_ATAC_encoder(x, None)
        if give_mean:
            z = (qz_m,)
            rna_z = (qz_rna_m,)
            atac_z = qz_atac_m
        return [z, rna_z, atac_z]

    def sample_from_posterior_l(self, x):
        r"""samples the tensor of library sizes from the posterior
        #doesn't really sample, returns the tensor of the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: tensor of shape ``(batch_size, 1)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        outputs = self.inference(x=x, batch_index=batch_index, y=y, n_samples=n_samples)
        return outputs["p_rna_scale"], outputs["p_atac_scale"]

    def get_sample_rate(
        self, x, batch_index=None, y=None, n_samples=1, local_l_mean=None, local_l_var=None
    ):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        outputs = self.inference(
            x=x,
            batch_index=batch_index,
            y=y,
            n_samples=n_samples,
            local_l_mean=local_l_mean,
            local_l_var=local_l_var,
        )
        return outputs["p_rna_rate"], outputs["p_atac_mean"]

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, **kwargs):
        # Reconstruction Loss
        if self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(
                dim=-1
            ) + 0.5 * mean_square_error_positive(x, px_rate).sum(dim=-1)
        elif self.reconstruction_loss == "zinb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(
                dim=-1
            ) + 0.5 * mean_square_error_positive(x, px_rate).sum(dim=-1)

        return reconst_loss

    def get_reconstruction_atac_loss(self, x, mu, dispersion, dropout, type="zip", **kwargs):
        if type == "zinb":
            reconst_loss = -log_zinb_positive(x, mu, dispersion, dropout).sum(dim=-1)
        elif type == "zip":
            reconst_loss = 0.5 * mean_square_error_positive(x, mu).sum(dim=-1) - log_zip_positive(
                x, mu, dropout
            ).sum(dim=-1)
            mu[x > 0] = 0
            reconst_loss = reconst_loss + 0.05 * mu.sum(dim=-1)
        elif type == "zip_bu":
            reconst_loss = -log_zip_positive(x, mu, dropout).sum(dim=-1) - binary_cross_entropy(
                x, mu
            ).sum(dim=-1)
        elif type == "bu":
            reconst_loss = -binary_cross_entropy(x, mu).sum(dim=-1)
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch[0] = torch.log(1 + sample_batch[0])
            sample_batch[1] = torch.log(1 + sample_batch[1])
        qz_rna_m, qz_rna_v, rna_z = self.RNA_encoder(sample_batch[0])
        qz_atac_m, qz_atac_v, atac_z = self.ATAC_encoder(sample_batch[1])
        qz_m, qz_v, z = self.RNA_ATAC_encoder(sample_batch)

        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4.0 * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

    def init_gmm_params(self, z):
        """
        Init SCALE model with GMM model parameters
        """
        if z is None:
            raise ("Input data is empty!")

        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type="diag")
        gmm.fit(z)
        # gmm.weights_
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
        clust_index = gmm.predict(z)

        return clust_index

    def init_gmm_params_with_louvain(self, z, label):
        """
        Init SCALE model with GMM model parameters
        """
        if z is None or label is None:
            raise ("Input data is empty!")

        mu = np.zeros((z.shape[1], len(np.unique(label))))
        var = np.zeros((z.shape[1], len(np.unique(label))))
        pi = np.zeros(len(np.unique(label)))
        for i in range(len(np.unique(label))):
            mu[:, i] = np.mean(z[label == i, :], axis=0)
            var[:, i] = np.var(z[label == i, :], axis=0)
            pi[i] = np.sum(label == i) / len(label)

        self.mu_c.data.copy_(torch.from_numpy(mu.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(var.astype(np.float32)))
        self.pi.data.copy_(torch.from_numpy(pi.astype(np.float32)))

        return True

    def get_gamma(self, z, update=False):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z_org = z
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = torch.abs(self.pi.repeat(N, 1))  # NxK
        mu_c = self.mu_c.repeat(N, 1, 1)  # NxDxK
        var_c = torch.abs(self.var_c.repeat(N, 1, 1))  # NxDxK

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

    def inference(
        self,
        x,
        batch_index=None,
        y=None,
        local_l_mean=None,
        local_l_var=None,
        update=False,
        n_samples=1,
    ):
        x_ = x
        if len(x_) != 2:
            raise ValueError(
                "Input training data should be 2 data types(RNA and ATAC),"
                "but input was only {}.format(len(x_))"
            )
        x_rna = x_[0]
        x_atac = x_[1]
        libary_atac = torch.log(x_[1].sum(dim=-1)).reshape(-1, 1)
        libary_rna = torch.log(x_[0].sum(dim=-1)).reshape(-1, 1)
        if self.log_variational:
            x_rna = torch.log(1 + x_rna)
            x_atac = torch.log(1 + x_atac)

        # Sampling
        if self.isLibrary:
            ql_m, ql_v, l_z = self.l_encoder(x_rna, batch_index)
        qz_rna_m, qz_rna_v, rna_z = self.RNA_encoder(x_rna, batch_index)
        qz_atac_m, qz_atac_v, atac_z = self.ATAC_encoder(x_atac, batch_index)
        qz_m, qz_v, z = self.RNA_ATAC_encoder([x_rna, x_atac], batch_index)

        qz_joint_mu = self.concatenter(torch.cat((qz_rna_m, qz_atac_m), 1))
        qz_joint_v = self.concatenter(torch.cat((torch.log(qz_rna_v), torch.log(qz_atac_v)), 1))
        qz_joint_v = torch.exp(qz_joint_v)
        qz_joint_z = Normal(qz_joint_mu, qz_joint_v.sqrt()).rsample()
        gamma_joint, _, _, _ = self.get_gamma(qz_joint_z)

        gamma, mu_c, var_c, pi = self.get_gamma(z, update)  # , self.n_centroids, c_params)
        index = torch.argmax(gamma, dim=1)

        index1 = [i for i in range(len(index))]
        mu_c_max = mu_c[index1, :, index]
        var_c_max = var_c[index1, :, index]
        z_c_max = reparameterize_gaussian(mu_c_max, var_c_max)

        libary_scale = reparameterize_gaussian(local_l_mean, local_l_var)
        if self.isLibrary:
            libary_scale = libary_rna
        # decoder
        (
            p_rna_scale,
            p_rna_r,
            p_rna_rate,
            p_rna_dropout,
            p_atac_scale,
            p_atac_r,
            p_atac_mean,
            p_atac_dropout,
        ) = self.RNA_ATAC_decoder(
            z, z_c_max, batch_index, libary_scale=libary_scale, gamma=gamma, libary_atac=libary_atac
        )
        # classifer
        if self.classifer_num > 0 and y is not None:
            classifer_pred = self.classifer(z)
            classifer_loss = -100 * (
                one_hot(y, self.classifer_num) * torch.log(classifer_pred + 1.0e-10)
            ).sum(dim=-1)

        if self.log_variational:
            p_rna_rate_norm = torch.log(1 + p_rna_rate)
            p_atac_mean_norm = torch.log(1 + p_atac_mean)
        rec_rna_mu, rec_rna_v, rec_rna_z = self.RNA_encoder(p_rna_rate_norm, batch_index)
        gamma_rna_rec, _, _, _ = self.get_gamma(rec_rna_z)
        rec_atac_mu, rec_atac_v, rec_atac_z = self.ATAC_encoder(p_atac_mean_norm, batch_index)
        gamma_atac_rec, _, _, _ = self.get_gamma(rec_atac_z)
        rec_joint_mu = self.concatenter(torch.cat((rec_rna_mu, rec_atac_mu), 1))
        rec_joint_v = self.concatenter(torch.cat((torch.log(rec_rna_v), torch.log(rec_atac_v)), 1))
        rec_joint_v = torch.exp(rec_joint_v)
        rec_joint_z = Normal(rec_joint_mu, rec_joint_v.sqrt()).rsample()
        gamma_joint_rec, _, _, _ = self.get_gamma(rec_joint_z)

        if self.dispersion == "gene-label":
            p_rna_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
            p_atac_r = F.linear(one_hot(y, self.n_labels), self.p_atac_r)
        elif self.dispersion == "gene-batch":
            p_rna_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
            p_atac_r = F.linear(one_hot(batch_index, self.n_batch), self.p_atac_r)
        elif self.dispersion == "gene":
            p_rna_r = self.px_r
            p_atac_r = self.p_atac_r

        p_rna_r = torch.exp(p_rna_r)
        p_atac_r = torch.exp(p_atac_r)

        return dict(
            p_rna_scale=p_rna_scale,
            p_rna_r=p_rna_r,
            p_rna_rate=p_rna_rate,
            p_rna_dropout=p_rna_dropout,
            p_atac_scale=p_atac_scale,
            p_atac_r=p_atac_r,
            p_atac_mean=p_atac_mean,
            p_atac_dropout=p_atac_dropout,
            qz_rna_m=qz_rna_m,
            qz_rna_v=qz_rna_v,
            rna_z=rna_z,
            qz_atac_m=qz_atac_m,
            qz_atac_v=qz_atac_v,
            atac_z=atac_z,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            mu_c=mu_c,
            var_c=var_c,
            gamma=gamma,
            pi=pi,
            mu_c_max=mu_c_max,
            var_c_max=var_c_max,
            z_c_max=z_c_max,
            gamma_rna_rec=gamma_rna_rec,
            gamma_atac_rec=gamma_atac_rec,
            rec_atac_mu=rec_atac_mu,
            rec_atac_v=rec_atac_v,
            rec_rna_mu=rec_rna_mu,
            rec_rna_v=rec_rna_v,
            ql_m=ql_m,
            ql_v=ql_v,
            l_z=l_z,
            rec_joint_mu=rec_joint_mu,
            rec_joint_v=rec_joint_v,
            rec_joint_z=rec_joint_z,
            gamma_joint_rec=gamma_joint_rec,
            qz_joint_mu=qz_joint_mu,
            qz_joint_v=qz_joint_v,
            qz_joint_z=qz_joint_z,
            gamma_joint=gamma_joint,
            classifer_loss=classifer_loss if self.classifer_num > 0 else 0,
        )

    def forward(self, x_rna, x_atac, local_l_mean, local_l_var, batch_index=None, y=None):
        r"""Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution
        x = [x_rna, x_atac]
        outputs = self.inference(x, batch_index, y, local_l_mean, local_l_var, update=False)
        qz_rna_m = outputs["qz_rna_m"]
        qz_rna_v = outputs["qz_rna_v"]
        qz_atac_m = outputs["qz_atac_m"]
        qz_atac_v = outputs["qz_atac_v"]
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        p_rna_rate = outputs["p_rna_rate"]
        p_rna_r = outputs["p_rna_r"]
        p_rna_dropout = outputs["p_rna_dropout"]
        p_atac_r = outputs["p_atac_r"]
        p_atac_mean = outputs["p_atac_mean"]
        p_atac_dropout = outputs["p_atac_dropout"]
        mu_c = outputs["mu_c"]
        var_c = outputs["var_c"]
        gamma = outputs["gamma"]
        pi = outputs["pi"]
        gamma_rna_rec = outputs["gamma_rna_rec"]
        gamma_atac_rec = outputs["gamma_atac_rec"]
        rec_atac_mu = outputs["rec_atac_mu"]
        rec_atac_v = outputs["rec_atac_v"]
        rec_rna_mu = outputs["rec_rna_mu"]
        rec_rna_v = outputs["rec_rna_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        l_z = outputs["l_z"]
        rec_joint_mu = outputs["rec_joint_mu"]
        rec_joint_v = outputs["rec_joint_v"]
        rec_joint_z = outputs["rec_joint_z"]
        gamma_joint_rec = outputs["gamma_joint_rec"]
        qz_joint_mu = outputs["qz_joint_mu"]
        qz_joint_v = outputs["qz_joint_v"]
        qz_joint_z = outputs["qz_joint_z"]
        gamma_joint = outputs["gamma_joint"]
        classifer_loss = outputs["classifer_loss"]

        n_centroids = pi.size(1)
        mu_expand = qz_m.unsqueeze(2).expand(qz_m.size(0), qz_m.size(1), n_centroids)
        logvar_expand = qz_v.unsqueeze(2).expand(qz_v.size(0), qz_v.size(1), n_centroids)
        # zl

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
        qentropy = -0.5 * torch.sum(1 + qz_v + math.log(2 * math.pi), 1)

        # log q(c|x)
        logqcx = torch.sum(gamma * torch.log(gamma), 1)

        # kl(qz||pz)
        kld_qz_pz = -logpzc - logpc + qentropy + logqcx
        print("logpzc:{}, logqcx:{}".format(torch.mean(logpzc), torch.mean(logqcx)))
        # print("gamma={},var_c={}".format(gamma,var_c))
        # kl(qz||qz_rna)
        kld_qz_rna = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(qz_rna_m, torch.sqrt(qz_rna_v))).sum(
            dim=1
        )

        # kl(qz||qz_atac)
        kld_qz_atac = kl(
            Normal(qz_m, torch.sqrt(qz_v)), Normal(qz_atac_m, torch.sqrt(qz_atac_v))
        ).sum(
            # check the postive qz_v
            dim=1
        )

        # kl(qz||qz_joint)
        kld_qz_joint = kl(
            Normal(qz_m, torch.sqrt(qz_v)), Normal(qz_joint_mu, torch.sqrt(qz_joint_v))
        ).sum(
            # check the postive qz_v
            dim=1
        )

        # KL Divergence
        kl_divergence = kld_qz_pz + 0.1 * (kld_qz_joint)
        if self.isLibrary:
            consistent_loss_rna = -(
                torch.softmax(gamma, dim=-1)
                * torch.log(torch.softmax(gamma_rna_rec, dim=-1) + 1.0e-6)
                + (1 - torch.softmax(gamma, dim=-1))
                * torch.log(1 - torch.softmax(gamma_rna_rec, dim=-1) + 1.0e-6)
            ).sum(dim=-1)
            consistent_loss_atac = -(
                torch.softmax(gamma, dim=-1)
                * torch.log(torch.softmax(gamma_atac_rec, dim=-1) + 1.0e-6)
                + (1 - torch.softmax(gamma, dim=-1))
                * torch.log(1 - torch.softmax(gamma_atac_rec, dim=-1) + 1.0e-6)
            ).sum(dim=-1)
            consistent_loss_joint = -(
                torch.softmax(gamma, dim=-1)
                * torch.log(torch.softmax(gamma_joint_rec, dim=-1) + 1.0e-6)
                + (1 - torch.softmax(gamma, dim=-1))
                * torch.log(1 - torch.softmax(gamma_joint_rec, dim=-1) + 1.0e-6)
            ).sum(dim=-1)
            rec_rna_kl = kl(
                Normal(qz_m, torch.sqrt(qz_v)), Normal(rec_rna_mu, torch.sqrt(rec_rna_v))
            ).sum(dim=1)
            rec_atac_kl = kl(
                Normal(qz_m, torch.sqrt(qz_v)), Normal(rec_atac_mu, torch.sqrt(rec_atac_v))
            ).sum(dim=1)
            rec_joint_kl = kl(
                Normal(qz_joint_mu, torch.sqrt(qz_joint_v)),
                Normal(rec_joint_mu, torch.sqrt(rec_joint_v)),
            ).sum(dim=1)

        # likelihood
        reconst_loss_rna = 3.0 * self.get_reconstruction_loss(
            x[0], p_rna_rate, p_rna_r, p_rna_dropout
        )
        reconst_loss_atac = 0.1 * self.get_reconstruction_atac_loss(
            x[1], p_atac_mean, p_atac_r, p_atac_dropout
        )  # implement this function
        reconst_loss = reconst_loss_rna + reconst_loss_atac + classifer_loss
        if self.isLibrary:
            reconst_loss = reconst_loss + 0.5 * (
                consistent_loss_joint
                - 50 * torch.sum(gamma * gamma, dim=-1)
                - 50
                * torch.sum(
                    (torch.sum(gamma, dim=0) / gamma.shape[0])
                    * (torch.log(torch.sum(gamma, dim=0) / gamma.shape[0] + 1.0e-10))
                )
            )
            kl_divergence = kl_divergence + 0.1 * (rec_joint_kl)

        # init the gmm model, training pc
        print(
            "kld_qz_pz = %f,kld_qz_rna = %f,kld_qz_atac = %f,kl_divergence = %f,reconst_loss_rna = %f,\
        reconst_loss_atac = %f, mu=%f, sigma=%f"
            % (
                torch.mean(kld_qz_pz),
                torch.mean(kld_qz_rna),
                torch.mean(kld_qz_atac),
                torch.mean(kl_divergence),
                torch.mean(reconst_loss_rna),
                torch.mean(reconst_loss_atac),
                torch.mean(self.mu_c),
                torch.mean(self.var_c),
            )
        )
        return reconst_loss, kl_divergence, 0.0


def build_scmvp():
    return Multi_VAE_Attention(
        RNA_input=48,
        ATAC_input=64,
        n_batch=0,
        n_labels=0,
        n_hidden=32,
        n_latent=8,
        n_layers=1,
        n_centroids=4,
        dropout_rate=0.1,
        mode="mm-vae",
        dispersion="gene",
        log_variational=True,
        reconstruction_loss="zinb",
        isLibrary=True,
        is_cluster=True,
        classifer_num=0,
    )


def example_input_scmvp():
    batch = 8
    n_rna = 48
    n_atac = 64
    x_rna = torch.rand(batch, n_rna) * 5.0
    x_atac = torch.rand(batch, n_atac) * 3.0
    local_l_mean = torch.zeros(batch, 1)
    local_l_var = torch.ones(batch, 1)
    return (x_rna, x_atac, local_l_mean, local_l_var)


MENAGERIE_ENTRIES = [
    ("scMVP", "build_scmvp", "example_input_scmvp", 2021, "vendored-pytorch"),
]
