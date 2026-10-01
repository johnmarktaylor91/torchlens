# FAITHFUL PORT of debbiemarkslab/DeepSequence @ master (original framework: Theano)
#   DeepSequence/model.py (class VariationalAutoencoder)
# The real repo is written in Theano (`theano.tensor`, `theano.shared`) and cannot run
# in the base torch env. This ports the "Doubly Variational Autoencoder" (Riesselman,
# Ingraham & Marks 2018 / sparsity priors from Ingraham & Marks 2016) faithfully: an MLP
# encoder producing a diagonal-Gaussian posterior over latent z, and a Bayesian SVI
# decoder whose EVERY weight/bias is itself a variational (mu, log_sigma) pair that gets
# reparameterization-sampled on every forward pass, with a tiled group-sparsity ("logit")
# scale, a convolutional pattern dictionary over the decoder output, and a softplus PWM
# inverse-temperature scale -- matching the default config used in examples/run_svi.py
# (sparsity="logit", convolve_patterns=True, output_bias=True, final_pwm_scale=True).
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class DeepSequenceVAE(nn.Module):
    """Faithful port of debbiemarkslab/DeepSequence's VariationalAutoencoder
    (SVI/Bayesian decoder variant, the class actually used by examples/run_svi.py).
    """

    def __init__(
        self,
        seq_len: int = 20,
        alphabet_size: int = 20,
        encoder_architecture=(64, 64),
        decoder_architecture=(32, 32),
        n_latent: int = 8,
        n_patterns: int = 4,
        encode_nonlinearity_type: str = "relu",
        decode_nonlinearity_type: str = "relu",
        final_decode_nonlinearity: str = "sigmoid",
        sparsity: str = "logit",
        convolve_patterns: bool = True,
        conv_decoder_size: int = 6,
        output_bias: bool = True,
        final_pwm_scale: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.alphabet_size = alphabet_size
        self.encoder_architecture = list(encoder_architecture)
        # decoder's last layer width gets multiplied by n_patterns, exactly as the
        # Theano __init__ does to self.decoder_architecture[-1]
        decoder_architecture = list(decoder_architecture)
        decoder_architecture[-1] = decoder_architecture[-1] * n_patterns
        self.decoder_architecture = decoder_architecture
        self.n_latent = n_latent
        self.n_patterns = n_patterns
        self.encode_nonlinearity_type = encode_nonlinearity_type
        self.decode_nonlinearity_type = decode_nonlinearity_type
        self.final_decode_nonlinearity = final_decode_nonlinearity
        self.sparsity = sparsity
        self.convolve_patterns = convolve_patterns
        self.conv_decoder_size = conv_decoder_size
        self.output_bias = output_bias
        self.final_pwm_scale = final_pwm_scale

        sigma_init = 0.01
        logsig_init = -5.0

        def weight(d_in, d_out):
            std = math.sqrt(2.0 / (d_in + d_out))
            return nn.Parameter(torch.randn(d_in, d_out) * std)

        def weight_logsig(d_in, d_out):
            return nn.Parameter(torch.full((d_in, d_out), logsig_init))

        def bias(d_out):
            return nn.Parameter(torch.full((d_out,), 0.1))

        def bias_logsig(d_out):
            return nn.Parameter(torch.full((d_out,), logsig_init))

        del sigma_init  # kept for parity with upstream naming; unused deterministically

        ## Encoder (plain MLP, deterministic weights -- matches upstream: only the
        ## decoder side is variational/SVI)
        self.encoder_weights = nn.ParameterList()
        self.encoder_biases = nn.ParameterList()
        prev = seq_len * alphabet_size
        for hidden_units in self.encoder_architecture:
            self.encoder_weights.append(weight(prev, hidden_units))
            self.encoder_biases.append(bias(hidden_units))
            prev = hidden_units

        self.W_hmu = weight(self.encoder_architecture[-1], n_latent)
        self.b_hmu = bias(n_latent)
        self.W_hsigma = weight(self.encoder_architecture[-1], n_latent)
        self.b_hsigma = bias_logsig(n_latent)

        ## Decoder (SVI: every weight/bias is a (mu, log_sigma) variational pair)
        self.decoder_w_mu = nn.ParameterList()
        self.decoder_w_logsig = nn.ParameterList()
        self.decoder_b_mu = nn.ParameterList()
        self.decoder_b_logsig = nn.ParameterList()
        prev = n_latent
        for hidden_units in self.decoder_architecture:
            self.decoder_w_mu.append(weight(prev, hidden_units))
            self.decoder_w_logsig.append(weight_logsig(prev, hidden_units))
            self.decoder_b_mu.append(bias(hidden_units))
            self.decoder_b_logsig.append(bias_logsig(hidden_units))
            prev = hidden_units

        self.final_output_size = (
            self.decoder_architecture[-1] if self.decoder_architecture else n_latent
        )

        if self.convolve_patterns:
            self.W_conv_mu = weight(conv_decoder_size, alphabet_size)
            self.W_conv_logsig = weight_logsig(conv_decoder_size, alphabet_size)
            self.W_out_mu = weight(self.final_output_size, seq_len * conv_decoder_size)
            self.W_out_logsig = weight_logsig(self.final_output_size, seq_len * conv_decoder_size)
        else:
            self.W_out_mu = weight(self.final_output_size, seq_len * alphabet_size)
            self.W_out_logsig = weight_logsig(self.final_output_size, seq_len * alphabet_size)

        if self.output_bias:
            self.b_out_mu = bias(seq_len * alphabet_size)
            self.b_out_logsig = bias_logsig(seq_len * alphabet_size)

        if self.sparsity:
            self.W_out_scale_mu = weight(self.final_output_size // n_patterns, seq_len)
            self.W_out_scale_logsig = weight_logsig(self.final_output_size // n_patterns, seq_len)

        if self.final_pwm_scale:
            self.final_pwm_scale_mu = nn.Parameter(torch.ones(1))
            self.final_pwm_scale_logsig = nn.Parameter(torch.full((1,), -5.0))

    # ------------------------------------------------------------------
    def _encode_nonlinearity(self, x):
        if self.encode_nonlinearity_type == "relu":
            return F.relu(x)
        elif self.encode_nonlinearity_type == "tanh":
            return torch.tanh(x)
        elif self.encode_nonlinearity_type == "sigmoid":
            return torch.sigmoid(x)
        elif self.encode_nonlinearity_type == "elu":
            return F.elu(x)
        raise ValueError(self.encode_nonlinearity_type)

    def _decode_nonlinearity(self, x):
        if self.decode_nonlinearity_type == "relu":
            return F.relu(x)
        elif self.decode_nonlinearity_type == "tanh":
            return torch.tanh(x)
        elif self.decode_nonlinearity_type == "sigmoid":
            return torch.sigmoid(x)
        elif self.decode_nonlinearity_type == "elu":
            return F.elu(x)
        raise ValueError(self.decode_nonlinearity_type)

    @staticmethod
    def _sampler(mu, log_sigma):
        """Reparameterized sample from a diagonal Gaussian, N(mu, exp(log_sigma)^2)."""
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma) * eps

    def encoder(self, x):
        """x: (batch, seq_len, alphabet_size) -> (mu, log_sigma) each (batch, n_latent)"""
        batch_size = x.shape[0]
        layer_up_val = x.reshape(batch_size, self.seq_len * self.alphabet_size)
        for w, b in zip(self.encoder_weights, self.encoder_biases):
            layer_up_val = self._encode_nonlinearity(layer_up_val @ w + b)

        mu = layer_up_val @ self.W_hmu + self.b_hmu
        log_sigma = layer_up_val @ self.W_hsigma + self.b_hsigma
        return mu, log_sigma

    def decoder_sparse(self, x, z):
        """z: (batch, n_latent) -> reconstructed_x: (batch, seq_len, alphabet_size)"""
        layer_up_val = z
        n_layers = len(self.decoder_architecture)
        for layer_num in range(n_layers):
            W = self._sampler(self.decoder_w_mu[layer_num], self.decoder_w_logsig[layer_num])
            b = self._sampler(self.decoder_b_mu[layer_num], self.decoder_b_logsig[layer_num])
            if layer_num + 1 == n_layers:
                if self.final_decode_nonlinearity == "sigmoid":
                    layer_up_val = torch.sigmoid(layer_up_val @ W + b)
                else:
                    layer_up_val = self._decode_nonlinearity(layer_up_val @ W + b)
            else:
                layer_up_val = self._decode_nonlinearity(layer_up_val @ W + b)

        W_out = self._sampler(self.W_out_mu, self.W_out_logsig)

        if self.sparsity:
            W_scale = self._sampler(self.W_out_scale_mu, self.W_out_scale_logsig)
            W_scale = W_scale.tile((self.n_patterns, 1))  # (final_output_size, seq_len)
            if self.sparsity == "logit":
                W_scale = torch.sigmoid(W_scale.unsqueeze(-1))
            else:
                W_scale = torch.exp(W_scale.unsqueeze(-1))

        if self.convolve_patterns:
            W_conv = self._sampler(self.W_conv_mu, self.W_conv_logsig)
            W_out = (
                W_out.reshape(self.final_output_size * self.seq_len, self.conv_decoder_size)
                @ W_conv
            )
            if self.sparsity:
                W_out = (
                    W_out.reshape(self.final_output_size, self.seq_len, self.alphabet_size)
                    * W_scale
                )
            W_out = W_out.reshape(self.final_output_size, self.seq_len * self.alphabet_size)
        elif self.sparsity:
            W_out = (
                W_out.reshape(self.final_output_size, self.seq_len, self.alphabet_size) * W_scale
            )
            W_out = W_out.reshape(self.final_output_size, self.seq_len * self.alphabet_size)

        if self.output_bias:
            b_out = self._sampler(self.b_out_mu, self.b_out_logsig)
            reconstructed_x_flat = layer_up_val @ W_out + b_out
        else:
            reconstructed_x_flat = layer_up_val @ W_out

        if self.final_pwm_scale:
            pwm_scale = self._sampler(self.final_pwm_scale_mu, self.final_pwm_scale_logsig)[0]
            reconstructed_x_flat = reconstructed_x_flat * F.softplus(pwm_scale)

        reconstructed_x_unnorm = reconstructed_x_flat.reshape(
            layer_up_val.shape[0], self.seq_len, self.alphabet_size
        )

        reconstructed_x = F.softmax(reconstructed_x_unnorm, dim=2)

        log_softmax = F.log_softmax(reconstructed_x_unnorm, dim=2)
        logpxz = (x * log_softmax).sum(dim=-1).sum(dim=-1)

        return reconstructed_x, logpxz, layer_up_val

    def forward(self, x):
        """x: one-hot encoded sequence batch, (batch, seq_len, alphabet_size)."""
        mu, log_sigma = self.encoder(x)
        z = self._sampler(mu, log_sigma)
        reconstructed_x, logpxz, _pattern_activations = self.decoder_sparse(x, z)
        return reconstructed_x, mu, log_sigma, logpxz


def build_deepsequence():
    """Tiny random-init DeepSequenceVAE, faithfully ported from the real Theano
    VariationalAutoencoder (SVI/sparsity-prior decoder) in debbiemarkslab/DeepSequence."""
    torch.manual_seed(0)
    model = DeepSequenceVAE(
        seq_len=12,
        alphabet_size=20,  # amino acid alphabet, matches real usage
        encoder_architecture=(32, 32),
        decoder_architecture=(16, 16),
        n_latent=6,
        n_patterns=2,
        conv_decoder_size=4,
    )
    model.eval()
    return model


def example_input_deepsequence():
    torch.manual_seed(0)
    batch_size, seq_len, alphabet_size = 2, 12, 20
    idx = torch.randint(0, alphabet_size, (batch_size, seq_len))
    x = F.one_hot(idx, num_classes=alphabet_size).float()
    return x


MENAGERIE_ENTRIES = [
    (
        "DeepSequence",
        build_deepsequence,
        example_input_deepsequence,
        2018,
        "REIMPLEMENT",
    ),
]
