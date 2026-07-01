# SOURCE: vendored from OATML-Markslab/EVEscape @ main (EVEscape depends on the
# separate OATML-Markslab/EVE @ master repo for its fitness-scoring VAE)
# https://raw.githubusercontent.com/OATML-Markslab/EVE/master/EVE/VAE_encoder.py
# https://raw.githubusercontent.com/OATML-Markslab/EVE/master/EVE/VAE_decoder.py
#
# Frazer, Notin, Dias, Gomez, Min, Brock, Gal, Marks 2021 (Nature) "Disease
# variant prediction with deep generative models of evolution" (EVE) + Thadani,
# Notin, Groves, ... Marks 2023 (Nature) "Learning from pre-pandemic data to
# forecast viral escape" (EVEscape). EVEscape's "Fitness" component is computed
# directly from EVE: an unsupervised Bayesian VAE trained on a multiple sequence
# alignment of the viral protein family, whose per-mutation evolutionary-index
# score (Fitness component 1 of 3 in EVEscape, alongside structure-based
# Accessibility and residue-chemistry Dissimilarity) drives escape prediction.
# EVE's own `EVE/VAE_model.py::VAE_model` class is a training/scoring
# orchestration wrapper (dataset preprocessing, ELBO loss, evol-index sampling
# loops) rather than a clean tensor-in/tensor-out module, but its `encoder`
# (`VAE_MLP_encoder`) and `decoder` (`VAE_Bayesian_MLP_decoder`) submodules ARE
# real, self-contained nn.Modules with proper forward() methods, and
# `VAE_model.all_likelihood_components` shows the exact real forward path used
# at inference: `mu, log_var = encoder(x); z = sample_latent(mu, log_var);
# recon_x_log = decoder(z)`.
#
# `VAE_MLP_encoder` and `VAE_Bayesian_MLP_decoder` are copied verbatim from the
# real source files (only the module-level imports were merged into this file --
# no architectural change). `EVE_VAE` below is a thin wrapper module that
# reproduces the real `VAE_model.all_likelihood_components` forward computation
# (encode -> reparameterize -> decode) using these two real submodules; the
# reparameterization trick (`sample_latent`) is copied verbatim from
# `EVE/VAE_model.py`.

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_MLP_encoder(nn.Module):
    """
    MLP encoder class for the VAE model.
    """

    def __init__(self, params):
        """
        Required input parameters:
        - seq_len: (Int) Sequence length of sequence alignment
        - alphabet_size: (Int) Alphabet size of sequence alignment (will be driven by the data helper object)
        - hidden_layers_sizes: (List) List of sizes of DNN linear layers
        - z_dim: (Int) Size of latent space
        - convolve_input: (Bool) Whether to perform 1d convolution on input (kernel size 1, stide 1)
        - convolution_depth: (Int) Size of the 1D-convolution on input
        - nonlinear_activation: (Str) Type of non-linear activation to apply on each hidden layer
        - dropout_proba: (Float) Dropout probability applied on all hidden layers. If 0.0 then no dropout applied
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = params["seq_len"]
        self.alphabet_size = params["alphabet_size"]
        self.hidden_layers_sizes = params["hidden_layers_sizes"]
        self.z_dim = params["z_dim"]
        self.convolve_input = params["convolve_input"]
        self.convolution_depth = params["convolution_input_depth"]
        self.dropout_proba = params["dropout_proba"]

        self.mu_bias_init = 0.1
        self.log_var_bias_init = -10.0

        # Convolving input with kernels of size 1 to capture potential similarities across amino acids when encoding sequences
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

        # set up non-linearity
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


class VAE_Bayesian_MLP_decoder(nn.Module):
    """
    Bayesian MLP decoder class for the VAE model.
    """

    def __init__(self, params):
        """
        Required input parameters:
        - seq_len: (Int) Sequence length of sequence alignment
        - alphabet_size: (Int) Alphabet size of sequence alignment (will be driven by the data helper object)
        - hidden_layers_sizes: (List) List of the sizes of the hidden layers (all DNNs)
        - z_dim: (Int) Dimension of latent space
        - first_hidden_nonlinearity: (Str) Type of non-linear activation applied on the first (set of) hidden layer(s)
        - last_hidden_nonlinearity: (Str) Type of non-linear activation applied on the very last hidden layer (pre-sparsity)
        - dropout_proba: (Float) Dropout probability applied on all hidden layers. If 0.0 then no dropout applied
        - convolve_output: (Bool) Whether to perform 1d convolution on output (kernel size 1, stide 1)
        - convolution_depth: (Int) Size of the 1D-convolution on output
        - include_temperature_scaler: (Bool) Whether we apply the global temperature scaler
        - include_sparsity: (Bool) Whether we use the sparsity inducing scheme on the output from the last hidden layer
        - num_tiles_sparsity: (Int) Number of tiles to use in the sparsity inducing scheme (the more the tiles, the stronger the sparsity)
        - bayesian_decoder: (Bool) Whether the decoder is bayesian or not
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            )  # product of size (H * seq_len, alphabet)

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


class EVE_VAE(nn.Module):
    """
    Thin wrapper reproducing the real `EVE/VAE_model.py::VAE_model.
    all_likelihood_components` forward path (encode one-hot MSA sequences,
    reparameterize, decode) using the real `VAE_MLP_encoder` /
    `VAE_Bayesian_MLP_decoder` submodules verbatim. `sample_latent` is copied
    verbatim from `VAE_model.sample_latent`.
    """

    def __init__(self, encoder_parameters, decoder_parameters):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def build_evescape():
    seq_len = 20
    alphabet_size = 20
    encoder_parameters = dict(
        hidden_layers_sizes=[64, 32],
        z_dim=8,
        convolve_input=False,
        convolution_input_depth=20,
        nonlinear_activation="relu",
        dropout_proba=0.0,
        seq_len=seq_len,
        alphabet_size=alphabet_size,
    )
    decoder_parameters = dict(
        hidden_layers_sizes=[32, 64],
        z_dim=8,
        first_hidden_nonlinearity="relu",
        last_hidden_nonlinearity="relu",
        dropout_proba=0.0,
        convolve_output=False,
        convolution_output_depth=20,
        include_temperature_scaler=True,
        include_sparsity=False,
        num_tiles_sparsity=1,
        seq_len=seq_len,
        alphabet_size=alphabet_size,
    )
    model = EVE_VAE(encoder_parameters=encoder_parameters, decoder_parameters=decoder_parameters)
    # Real EVE usage (train_VAE.py): `model = model.to(model.device)` right after
    # construction, since VAE_model.__init__ always sets self.device to CUDA when
    # available (independent of where the freshly-constructed parameters live).
    model = model.to(model.device)
    model.eval()
    return model


def example_input_evescape():
    # One-hot encoded MSA sequences: (batch_size, seq_len, alphabet_size),
    # matching the real `msa_data.one_hot_encoding` / `mutated_sequences_one_hot`
    # tensor layout consumed by `VAE_model.encoder(x)`. Real usage
    # (`compute_evol_indices`) does `x = batch.type(self.dtype).to(self.device)`
    # before the forward pass, so we match that device placement here too.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 20
    alphabet_size = 20
    idx = torch.randint(0, alphabet_size, (batch_size, seq_len))
    x = torch.nn.functional.one_hot(idx, num_classes=alphabet_size).float().to(device)
    return x


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("EVEscape", "build_evescape", "example_input_evescape", 2023, "vendored"),
]
