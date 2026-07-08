# SOURCE: vendored from OATML-Markslab/EVE @ 460d70ef
# https://raw.githubusercontent.com/OATML-Markslab/EVE/master/EVE/VAE_encoder.py
# https://raw.githubusercontent.com/OATML-Markslab/EVE/master/EVE/VAE_decoder.py
# https://raw.githubusercontent.com/OATML-Markslab/EVE/master/EVE/VAE_model.py
#
# Frazer, Notin, Dias, Gomez, Min, Brock, Gal, Marks 2021 (Nature Genetics) "Disease variant
# prediction with deep generative models of evolutionary data" (EVE). VAE_MLP_encoder and
# VAE_Bayesian_MLP_decoder below are copied verbatim from the real repo's EVE/VAE_encoder.py
# and EVE/VAE_decoder.py (the Bayesian decoder -- weight/bias sampled via the reparameterization
# trick each forward call -- is the model actually used for EVE scoring, per VAE_model.py
# wiring `self.decoder = VAE_decoder.VAE_Bayesian_MLP_decoder(params=decoder_parameters)` when
# `decoder_parameters['bayesian_decoder']` is True, which is EVE's shipped default config).
# EveModel below reproduces VAE_model's encoder(x) -> sample_latent(mu, log_var) -> decoder(z)
# forward path from EVE/VAE_model.py (training-loop / evol-index / data-loading code -- which
# needs a real MSA alignment -- is intentionally omitted; only the inference forward graph is
# needed for capture). Built here at tiny size (seq_len=12 positions, alphabet_size=20 amino
# acids, z_dim=8) matching the real MSA one-hot-encoding input format (batch, seq_len, alphabet).
# Portability fix (device placement, not architecture): the real repo's
# `self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` auto-detect is
# replaced with a fixed CPU device so this tiny random-init capture harness behaves identically
# regardless of host GPU presence (matches the precedent in menagerie/classics/rs_L1453_rim.py).
"""EVE: Bayesian MLP-VAE on multiple-sequence-alignment one-hot encodings for variant effect prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from EVE/VAE_encoder.py ---
class VAE_MLP_encoder(nn.Module):
    """
    MLP encoder class for the VAE model.
    """

    def __init__(self, params):
        super().__init__()
        self.device = torch.device("cpu")  # portability fix: fixed device, see header
        self.seq_len = params["seq_len"]
        self.alphabet_size = params["alphabet_size"]
        self.hidden_layers_sizes = params["hidden_layers_sizes"]
        self.z_dim = params["z_dim"]
        self.convolve_input = params["convolve_input"]
        self.convolution_depth = params["convolution_input_depth"]
        self.dropout_proba = params["dropout_proba"]

        self.mu_bias_init = 0.1
        self.log_var_bias_init = -10.0

        if self.convolve_input:
            self.input_convolution = nn.Conv1d(
                in_channels=self.alphabet_size,
                out_channels=self.convolution_depth,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            self.channel_size = self.convolution_depth
        else:
            self.channel_size = self.alphabet_size

        self.hidden_layers = torch.nn.ModuleDict()
        for layer_index in range(len(self.hidden_layers_sizes)):
            if layer_index == 0:
                self.hidden_layers[str(layer_index)] = nn.Linear(
                    (self.channel_size * self.seq_len), self.hidden_layers_sizes[layer_index]
                )
                nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.mu_bias_init)
            else:
                self.hidden_layers[str(layer_index)] = nn.Linear(
                    self.hidden_layers_sizes[layer_index - 1], self.hidden_layers_sizes[layer_index]
                )
                nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.mu_bias_init)

        self.fc_mean = nn.Linear(self.hidden_layers_sizes[-1], self.z_dim)
        nn.init.constant_(self.fc_mean.bias, self.mu_bias_init)
        self.fc_log_var = nn.Linear(self.hidden_layers_sizes[-1], self.z_dim)
        nn.init.constant_(self.fc_log_var.bias, self.log_var_bias_init)

        if params["nonlinear_activation"] == "relu":
            self.nonlinear_activation = nn.ReLU()
        elif params["nonlinear_activation"] == "tanh":
            self.nonlinear_activation = nn.Tanh()
        elif params["nonlinear_activation"] == "sigmoid":
            self.nonlinear_activation = nn.Sigmoid()
        elif params["nonlinear_activation"] == "elu":
            self.nonlinear_activation = nn.ELU()
        elif params["nonlinear_activation"] == "linear":
            self.nonlinear_activation = nn.Identity()

        if self.dropout_proba > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_proba)

    def forward(self, x):
        if self.dropout_proba > 0.0:
            x = self.dropout_layer(x)

        if self.convolve_input:
            x = x.permute(0, 2, 1)
            x = self.input_convolution(x)
            x = x.view(-1, self.seq_len * self.channel_size)
        else:
            x = x.view(-1, self.seq_len * self.channel_size)

        for layer_index in range(len(self.hidden_layers_sizes)):
            x = self.nonlinear_activation(self.hidden_layers[str(layer_index)](x))
            if self.dropout_proba > 0.0:
                x = self.dropout_layer(x)

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        return z_mean, z_log_var


# --- vendored from EVE/VAE_decoder.py (Bayesian decoder only; this is the shipped-default
# decoder per VAE_model.py's decoder_parameters['bayesian_decoder']=True wiring) ---
class VAE_Bayesian_MLP_decoder(nn.Module):
    """
    Bayesian MLP decoder class for the VAE model.
    """

    def __init__(self, params):
        super().__init__()
        self.device = torch.device("cpu")  # portability fix: fixed device, see header
        self.seq_len = params["seq_len"]
        self.alphabet_size = params["alphabet_size"]
        self.hidden_layers_sizes = params["hidden_layers_sizes"]
        self.z_dim = params["z_dim"]
        self.bayesian_decoder = True
        self.dropout_proba = params["dropout_proba"]
        self.convolve_output = params["convolve_output"]
        self.convolution_depth = params["convolution_output_depth"]
        self.include_temperature_scaler = params["include_temperature_scaler"]
        self.include_sparsity = params["include_sparsity"]
        self.num_tiles_sparsity = params["num_tiles_sparsity"]

        self.mu_bias_init = 0.1
        self.logvar_init = -10.0
        self.logit_scale_p = 0.001

        self.hidden_layers_mean = nn.ModuleDict()
        self.hidden_layers_log_var = nn.ModuleDict()
        for layer_index in range(len(self.hidden_layers_sizes)):
            if layer_index == 0:
                self.hidden_layers_mean[str(layer_index)] = nn.Linear(
                    self.z_dim, self.hidden_layers_sizes[layer_index]
                )
                self.hidden_layers_log_var[str(layer_index)] = nn.Linear(
                    self.z_dim, self.hidden_layers_sizes[layer_index]
                )
                nn.init.constant_(self.hidden_layers_mean[str(layer_index)].bias, self.mu_bias_init)
                nn.init.constant_(
                    self.hidden_layers_log_var[str(layer_index)].weight, self.logvar_init
                )
                nn.init.constant_(
                    self.hidden_layers_log_var[str(layer_index)].bias, self.logvar_init
                )
            else:
                self.hidden_layers_mean[str(layer_index)] = nn.Linear(
                    self.hidden_layers_sizes[layer_index - 1], self.hidden_layers_sizes[layer_index]
                )
                self.hidden_layers_log_var[str(layer_index)] = nn.Linear(
                    self.hidden_layers_sizes[layer_index - 1], self.hidden_layers_sizes[layer_index]
                )
                nn.init.constant_(self.hidden_layers_mean[str(layer_index)].bias, self.mu_bias_init)
                nn.init.constant_(
                    self.hidden_layers_log_var[str(layer_index)].weight, self.logvar_init
                )
                nn.init.constant_(
                    self.hidden_layers_log_var[str(layer_index)].bias, self.logvar_init
                )

        if params["first_hidden_nonlinearity"] == "relu":
            self.first_hidden_nonlinearity = nn.ReLU()
        elif params["first_hidden_nonlinearity"] == "tanh":
            self.first_hidden_nonlinearity = nn.Tanh()
        elif params["first_hidden_nonlinearity"] == "sigmoid":
            self.first_hidden_nonlinearity = nn.Sigmoid()
        elif params["first_hidden_nonlinearity"] == "elu":
            self.first_hidden_nonlinearity = nn.ELU()
        elif params["first_hidden_nonlinearity"] == "linear":
            self.first_hidden_nonlinearity = nn.Identity()

        if params["last_hidden_nonlinearity"] == "relu":
            self.last_hidden_nonlinearity = nn.ReLU()
        elif params["last_hidden_nonlinearity"] == "tanh":
            self.last_hidden_nonlinearity = nn.Tanh()
        elif params["last_hidden_nonlinearity"] == "sigmoid":
            self.last_hidden_nonlinearity = nn.Sigmoid()
        elif params["last_hidden_nonlinearity"] == "elu":
            self.last_hidden_nonlinearity = nn.ELU()
        elif params["last_hidden_nonlinearity"] == "linear":
            self.last_hidden_nonlinearity = nn.Identity()

        if self.dropout_proba > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_proba)

        if self.convolve_output:
            self.output_convolution_mean = nn.Conv1d(
                in_channels=self.convolution_depth,
                out_channels=self.alphabet_size,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            self.output_convolution_log_var = nn.Conv1d(
                in_channels=self.convolution_depth,
                out_channels=self.alphabet_size,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            nn.init.constant_(self.output_convolution_log_var.weight, self.logvar_init)
            self.channel_size = self.convolution_depth
        else:
            self.channel_size = self.alphabet_size

        if self.include_sparsity:
            self.sparsity_weight_mean = nn.Parameter(
                torch.zeros(
                    int(self.hidden_layers_sizes[-1] / self.num_tiles_sparsity), self.seq_len
                )
            )
            self.sparsity_weight_log_var = nn.Parameter(
                torch.ones(
                    int(self.hidden_layers_sizes[-1] / self.num_tiles_sparsity), self.seq_len
                )
            )
            nn.init.constant_(self.sparsity_weight_log_var, self.logvar_init)

        self.last_hidden_layer_weight_mean = nn.Parameter(
            torch.zeros(self.channel_size * self.seq_len, self.hidden_layers_sizes[-1])
        )
        self.last_hidden_layer_weight_log_var = nn.Parameter(
            torch.zeros(self.channel_size * self.seq_len, self.hidden_layers_sizes[-1])
        )
        nn.init.xavier_normal_(self.last_hidden_layer_weight_mean)  # Glorot initialization
        nn.init.constant_(self.last_hidden_layer_weight_log_var, self.logvar_init)

        self.last_hidden_layer_bias_mean = nn.Parameter(
            torch.zeros(self.alphabet_size * self.seq_len)
        )
        self.last_hidden_layer_bias_log_var = nn.Parameter(
            torch.zeros(self.alphabet_size * self.seq_len)
        )
        nn.init.constant_(self.last_hidden_layer_bias_mean, self.mu_bias_init)
        nn.init.constant_(self.last_hidden_layer_bias_log_var, self.logvar_init)

        if self.include_temperature_scaler:
            self.temperature_scaler_mean = nn.Parameter(torch.ones(1))
            self.temperature_scaler_log_var = nn.Parameter(torch.ones(1) * self.logvar_init)

    def sampler(self, mean, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mean).to(self.device)
        z = torch.exp(0.5 * log_var) * eps + mean
        return z

    def forward(self, z):
        batch_size = z.shape[0]
        if self.dropout_proba > 0.0:
            x = self.dropout_layer(z)
        else:
            x = z

        for layer_index in range(len(self.hidden_layers_sizes) - 1):
            layer_i_weight = self.sampler(
                self.hidden_layers_mean[str(layer_index)].weight,
                self.hidden_layers_log_var[str(layer_index)].weight,
            )
            layer_i_bias = self.sampler(
                self.hidden_layers_mean[str(layer_index)].bias,
                self.hidden_layers_log_var[str(layer_index)].bias,
            )
            x = self.first_hidden_nonlinearity(
                F.linear(x, weight=layer_i_weight, bias=layer_i_bias)
            )
            if self.dropout_proba > 0.0:
                x = self.dropout_layer(x)

        last_index = len(self.hidden_layers_sizes) - 1
        last_layer_weight = self.sampler(
            self.hidden_layers_mean[str(last_index)].weight,
            self.hidden_layers_log_var[str(last_index)].weight,
        )
        last_layer_bias = self.sampler(
            self.hidden_layers_mean[str(last_index)].bias,
            self.hidden_layers_log_var[str(last_index)].bias,
        )
        x = self.last_hidden_nonlinearity(
            F.linear(x, weight=last_layer_weight, bias=last_layer_bias)
        )
        if self.dropout_proba > 0.0:
            x = self.dropout_layer(x)

        W_out = self.sampler(
            self.last_hidden_layer_weight_mean, self.last_hidden_layer_weight_log_var
        )
        b_out = self.sampler(self.last_hidden_layer_bias_mean, self.last_hidden_layer_bias_log_var)

        if self.convolve_output:
            output_convolution_weight = self.sampler(
                self.output_convolution_mean.weight, self.output_convolution_log_var.weight
            )
            W_out = torch.mm(
                W_out.view(self.seq_len * self.hidden_layers_sizes[-1], self.channel_size),
                output_convolution_weight.view(self.channel_size, self.alphabet_size),
            )

        if self.include_sparsity:
            sparsity_weights = self.sampler(self.sparsity_weight_mean, self.sparsity_weight_log_var)
            sparsity_tiled = sparsity_weights.repeat(self.num_tiles_sparsity, 1)
            sparsity_tiled = nn.Sigmoid()(sparsity_tiled).unsqueeze(2)

            W_out = (
                W_out.view(self.hidden_layers_sizes[-1], self.seq_len, self.alphabet_size)
                * sparsity_tiled
            )

        W_out = W_out.view(self.seq_len * self.alphabet_size, self.hidden_layers_sizes[-1])

        x = F.linear(x, weight=W_out, bias=b_out)

        if self.include_temperature_scaler:
            temperature_scaler = self.sampler(
                self.temperature_scaler_mean, self.temperature_scaler_log_var
            )
            x = torch.log(1.0 + torch.exp(temperature_scaler)) * x

        x = x.view(batch_size, self.seq_len, self.alphabet_size)
        x_recon_log = F.log_softmax(x, dim=-1)  # of shape (batch_size, seq_len, alphabet)

        return x_recon_log


# --- reproduces the inference forward path from EVE/VAE_model.py's VAE_model
# (encoder -> sample_latent -> decoder); training/data-loading/evol-index code omitted
# since it requires a real MSA alignment and is not part of the network's forward graph ---
class EveModel(nn.Module):
    """
    EVE VAE forward path: MLP encoder -> reparameterized latent sample -> Bayesian MLP decoder.
    """

    def __init__(self, encoder_parameters, decoder_parameters):
        super().__init__()
        self.device = torch.device("cpu")  # portability fix: fixed device, see header
        self.encoder = VAE_MLP_encoder(params=encoder_parameters)
        self.decoder = VAE_Bayesian_MLP_decoder(params=decoder_parameters)

    def sample_latent(self, mu, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu).to(self.device)
        z = torch.exp(0.5 * log_var) * eps + mu
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample_latent(mu, log_var)
        recon_x_log = self.decoder(z)
        return recon_x_log


def build_eve():
    torch.manual_seed(0)
    seq_len = 12
    alphabet_size = 20
    z_dim = 8

    encoder_parameters = {
        "hidden_layers_sizes": [32, 16],
        "z_dim": z_dim,
        "convolve_input": False,
        "convolution_input_depth": 40,
        "nonlinear_activation": "relu",
        "dropout_proba": 0.0,
        "seq_len": seq_len,
        "alphabet_size": alphabet_size,
    }
    decoder_parameters = {
        "hidden_layers_sizes": [16, 32],
        "z_dim": z_dim,
        "bayesian_decoder": True,
        "first_hidden_nonlinearity": "relu",
        "last_hidden_nonlinearity": "relu",
        "dropout_proba": 0.0,
        "convolve_output": True,
        "convolution_output_depth": 40,
        "include_temperature_scaler": True,
        "include_sparsity": False,
        "num_tiles_sparsity": 4,
        "logit_sparsity_p": 0.15,
        "seq_len": seq_len,
        "alphabet_size": alphabet_size,
    }

    return EveModel(encoder_parameters, decoder_parameters)


def example_input_eve():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 12
    alphabet_size = 20
    # one-hot MSA encoding, as consumed by VAE_model.train_model / all_likelihood_components
    idx = torch.randint(0, alphabet_size, (batch_size, seq_len))
    x = F.one_hot(idx, num_classes=alphabet_size).float()
    return (x,)


MENAGERIE_ENTRIES = [
    ("EVE", "build_eve", "example_input_eve", 2021, "vendored"),
]
