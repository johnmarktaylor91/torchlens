# SOURCE: vendored from alduston/SindyTorchEncoder @ dd7f61791c90c96ad05aec5a18695d94d3a4c534
# https://raw.githubusercontent.com/alduston/SindyTorchEncoder/main/SindyTorchEnconder/src/torch_autoencoder.py
# https://raw.githubusercontent.com/alduston/SindyTorchEncoder/main/SindyTorchEnconder/src/sindy_utils.py
# (PyTorch port of kpchamp/SindyAutoencoders (Champion, Lusch, Kutz, Brunton, PNAS 2019)
#  "Data-driven discovery of coordinates and governing equations" -- the original repo is
#  TF1.x graph-mode, so per the ladder we vendor the community PyTorch port instead of
#  transcribing the TF1.x code ourselves.)
#
# SINDy-AE: a plain fully-connected autoencoder (`SindyNet.encoder`/`.decoder`, both
# `nn.Sequential` stacks of `nn.Linear` + activation) whose latent space is additionally
# regularized, at training time, to admit a sparse polynomial ("SINDy") dynamical-systems
# fit via a learned coefficient tensor `sindy_coeffs` over a fixed polynomial/sine feature
# library (`Theta`/`sindy_library_torch`). `SindyNet.__init__`, `.Encoder`, `.Decoder`,
# `.get_activation_f`, `.sindy_coefficients`, `.initializer`, and `.forward` are copied
# verbatim from the real `torch_autoencoder.py`; `get_initialized_weights` and
# `library_size` are copied verbatim from the real `sindy_utils.py` (the latter from the
# original kpchamp/SindyAutoencoders `src/sindy_utils.py`, since the PyTorch port re-exports
# it unmodified). No architectural changes. `torch.nn.init.xavier_uniform` (the real repo's
# deprecated-but-still-functional spelling, not `xavier_uniform_`) is kept as-is.
#
# `SindyNet.forward(x)` -- the traced entry point -- only exercises `self.encoder` and
# `self.decoder` (the autoencoder itself); the sparse-regression head (`Theta`/
# `sindy_predict`/`sindy_coeffs`) is constructed as real `nn.Parameter` state at `__init__`
# time (so it is present and traced as parameter data) but is only invoked from the
# loss-computation methods (`sindy_z_loss`/`sindy_x_loss`/`Loss`), which the original repo
# calls separately from `forward` during training (see `torch_training.py`: `auto_Loss`
# calls `self.forward(x)` then `self.Loss(...)` as two distinct steps) -- so this mirrors
# the real repo's own forward/loss split, not an omission.
#
# Real config field names/values (input_dim, latent_dim, model_order, poly_order,
# include_sine, coefficient_initialization, activation, widths, loss_weight_*) are taken
# verbatim from the shipped `examples/pendulum/train_pendulum.ipynb` params dict in the
# original kpchamp/SindyAutoencoders repo, scaled down (smaller widths/input_dim) only for
# tiny-trace size.
"""SINDy Autoencoder: plain FC autoencoder + sparse-regression (SINDy) latent-dynamics head.

Faithful tiny-scale assembly of the real (PyTorch-ported) SindyNet for TorchLens tracing.
Random init, no pretrained/hub downloads.
"""

from math import comb as binom

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# sindy_utils.py (verbatim, only the two helpers SindyNet.__init__/forward need)
# ---------------------------------------------------------------------------
def library_size(n, poly_order, use_sine=False, include_constant=True):
    # NOTE: real sindy_utils.py names this accumulator `l`; renamed to `total`
    # here only to satisfy lint (ambiguous-name rule) -- no logic changed.
    total = 0
    for k in range(poly_order + 1):
        total += int(binom(n + k - 1, k))
    if use_sine:
        total += n
    if not include_constant:
        total -= 1
    return total


def get_initialized_weights(shape, initializer, init_param=None, device="cpu"):
    W = torch.nn.Linear(shape[0], shape[1], device=device)
    if init_param:
        initializer(W.weight, *init_param)
    else:
        initializer(W.weight)
    return torch.transpose(W.state_dict()["weight"], 0, 1)


# ---------------------------------------------------------------------------
# torch_autoencoder.py (verbatim)
# ---------------------------------------------------------------------------
class SindyNet(nn.Module):
    def __init__(self, params, device=None):
        super().__init__()
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        params["device"] = self.device
        self.params = params
        self.activation_f = self.get_activation_f(params)

        encoder, encoder_layers = self.Encoder(self.params)
        self.encoder = encoder
        self.encoder_layers = encoder_layers

        decoder, decoder_layers = self.Decoder(self.params)
        self.decoder = decoder
        self.decoder_layers = decoder_layers

        self.iter_count = torch.tensor(0, device=device)
        self.epoch = torch.tensor(0, device=device)

        self.sindy_coeffs = torch.nn.Parameter(self.sindy_coefficients(), requires_grad=True)
        self.coefficient_mask = torch.tensor(
            params["coefficient_mask"], dtype=torch.float32, device=self.device
        )
        self.num_active_coeffs = torch.sum(self.coefficient_mask).cpu().detach().numpy()

    def Encoder(self, params):
        activation_function = self.get_activation_f(params)
        input_dim = params["input_dim"]
        latent_dim = params["latent_dim"]
        widths = params["widths"]

        layers = []
        for output_dim in widths:
            encoder = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform(encoder.weight)
            nn.init.constant_(encoder.bias.data, 0)

            input_dim = output_dim
            layers.append(encoder)
            layers.append(activation_function)

        encoder = nn.Linear(input_dim, latent_dim)
        nn.init.xavier_uniform(encoder.weight)
        nn.init.constant_(encoder.bias.data, 0)
        layers.append(encoder)
        Encoder = nn.Sequential(*layers)
        return Encoder, layers

    def Decoder(self, params):
        activation_function = self.get_activation_f(params)
        final_dim = params["input_dim"]
        input_dim = params["latent_dim"]
        widths = params["widths"]

        layers = []
        for output_dim in widths[::-1]:
            decoder = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform(decoder.weight)
            nn.init.constant_(decoder.bias.data, 0)

            input_dim = output_dim
            layers.append(decoder)
            layers.append(activation_function)

        decoder = nn.Linear(input_dim, final_dim)
        nn.init.xavier_uniform(decoder.weight)
        nn.init.constant_(decoder.bias.data, 0)
        layers.append(decoder)
        Decoder = nn.Sequential(*layers)
        return Decoder, layers

    def get_activation_f(self, params):
        activation = params["activation"]
        if activation == "relu":
            activation_function = torch.nn.ReLU()
        elif activation == "elu":
            activation_function = torch.nn.ELU()
        elif activation == "sigmoid":
            activation_function = torch.nn.Sigmoid()
        return activation_function

    def initializer(self):
        init_param = None
        init = self.params["coefficient_initialization"]
        if init == "xavier":
            intializer = torch.nn.init.xavier_uniform
        elif init == "specified":
            intializer = torch.nn.init.xavier_uniform
        elif init == "constant":
            intializer = torch.nn.init.constant_
            init_param = [1]
        elif init == "normal":
            intializer = torch.nn.init.normal_
        return intializer, init_param

    def sindy_coefficients(self):
        library_dim = self.params["library_dim"]
        latent_dim = self.params["latent_dim"]
        initializer, init_param = self.initializer()
        return get_initialized_weights(
            [library_dim, latent_dim], initializer, init_param=init_param, device=self.device
        )

    def forward(self, x):
        z = self.encoder(x)
        x_decode = self.decoder(z)
        return x_decode, z


# ---------------------------------------------------------------------------
# Tiny-scale build/example helpers (field names/values mirror the real shipped
# examples/pendulum/train_pendulum.ipynb params dict, scaled down for a fast trace)
# ---------------------------------------------------------------------------
def _tiny_params():
    params = {}
    params["input_dim"] = 12
    params["latent_dim"] = 1
    params["model_order"] = 2
    params["poly_order"] = 3
    params["include_sine"] = True
    params["library_dim"] = library_size(
        2 * params["latent_dim"], params["poly_order"], params["include_sine"], True
    )

    params["sequential_thresholding"] = True
    params["coefficient_threshold"] = 0.1
    params["threshold_frequency"] = 500
    params["coefficient_mask"] = [
        [1.0] * params["latent_dim"] for _ in range(params["library_dim"])
    ]
    params["coefficient_initialization"] = "constant"

    params["loss_weight_decoder"] = 1.0
    params["loss_weight_sindy_x"] = 5e-4
    params["loss_weight_sindy_z"] = 5e-5
    params["loss_weight_sindy_regularization"] = 1e-5

    params["activation"] = "sigmoid"
    params["widths"] = [8, 4]

    params["batch_size"] = 16
    params["learning_rate"] = 1e-4
    return params


def build_sindy_ae():
    return SindyNet(_tiny_params(), device="cpu")


def example_input_sindy_ae():
    return torch.randn(16, 12)


MENAGERIE_ENTRIES = [
    ("SINDy-AE", "build_sindy_ae", "example_input_sindy_ae", 2019, "vendored"),
]
