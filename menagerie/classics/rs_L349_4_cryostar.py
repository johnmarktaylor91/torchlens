# SOURCE: vendored from bytedance/cryostar @ main (cryostar/utils/ml_modules.py, projects/star/miscs.py)
"""CryoSTAR: AlphaFold-predicted-structure-prior + MLP VAE for cryo-EM
heterogeneous reconstruction (Nature Methods 2024). The trainable network
core (independent of the dataset-specific GMM/PDB/CTF machinery orchestrated
by the ``CryoEMTask`` Lightning module in ``projects/star/train_atom.py``) is
the ``VAE`` class in ``projects/star/miscs.py``: an MLP ``VAEEncoder``
(image -> mean, log_var) -> reparameterize -> MLP ``Decoder`` (z ->
per-atom/mode deformation field), built from the residual-MLP primitives in
``cryostar/utils/ml_modules.py``. This is exactly what
``CryoEMTask.__init__`` constructs as ``self.model = VAE(...)`` and calls
each training step via ``self.model(prepare_images(images, ...), idxes,
rots)``.

Code below is copied verbatim from the official repo's two modules (only
unused imports dropped). Architecture logic is untouched.
"""

from typing import List, Union

import torch
from torch import nn
from torch.nn import Linear

MENAGERIE_ZOO = "vendored-pytorch"


# --- cryostar/utils/ml_modules.py -------------------------------------------


class ResLinear(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.linear = Linear(in_chs, out_chs)

    def forward(self, x):
        return self.linear(x) + x


class MLP(nn.Module):
    def __init__(self, in_dims: List[int], out_dims: List[int]):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims

        layers = []
        for i, o in zip(in_dims, out_dims):
            layers.append(ResLinear(i, o) if i == o else Linear(i, o))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VAEEncoder(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: Union[int, List[int]], out_dim: int, num_hidden_layers=3
    ):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim,) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0])
            if in_dim == self.hidden_dim[0]
            else Linear(in_dim, self.hidden_dim[0]),
            nn.ReLU(inplace=True),
        )
        self.mlp = MLP(self.hidden_dim[:-1], self.hidden_dim[1:])

        self.mean_layer = Linear(self.hidden_dim[-1], out_dim)
        self.var_layer = Linear(self.hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.mlp(self.input_layer(x))
        mean = self.mean_layer(x)
        log_var = self.var_layer(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: Union[int, List[int]], out_dim: int, num_hidden_layers=3
    ):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim,) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0])
            if in_dim == self.hidden_dim[0]
            else Linear(in_dim, self.hidden_dim[0]),
            nn.ReLU(inplace=True),
        )
        self.mlp = MLP(self.hidden_dim[:-1], self.hidden_dim[1:])

        self.out_layer = Linear(self.hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.mlp(self.input_layer(x))
        return self.out_layer(x)


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# --- projects/star/miscs.py::VAE --------------------------------------------


class VAE(nn.Module):
    def __init__(
        self,
        encoder_cls: str,
        decoder_cls: str,
        in_dim: int,
        e_hidden_dim: Union[int, list, tuple],
        latent_dim: int,
        d_hidden_dim: Union[int, list, tuple],
        out_dim: int,
        e_hidden_layers: int,
        d_hidden_layers: int,
    ):
        super().__init__()
        if encoder_cls == "MLP":
            self.encoder = VAEEncoder(in_dim, e_hidden_dim, latent_dim, e_hidden_layers)
        else:
            raise Exception()

        if decoder_cls == "MLP":
            self.decoder = Decoder(latent_dim, d_hidden_dim, out_dim, d_hidden_layers)
        else:
            print(f"{decoder_cls} not in presets, you may set it manually later.")
            self.decoder: torch.nn.Module

    def encode(self, x, idx):
        mean, log_var = self.encoder(x)
        return mean, log_var

    def forward(self, x, idx, *args):
        mean, log_var = self.encode(x, idx)
        z = reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var


# --- staging harness -------------------------------------------------------
# Mirrors CryoEMTask.__init__'s real construction:
#   in_dim = 2 * down_side_shape**2  (fourier-space image, real+imag halves)
#   out_dim = num_pts * 3            (E3Deformer: per-atom xyz shift)
# at toy scale (tiny down_side_shape, tiny atom count).


def build_cryostar():
    down_side_shape = 4
    num_pts = 6
    in_dim = 2 * down_side_shape**2
    return VAE(
        encoder_cls="MLP",
        decoder_cls="MLP",
        in_dim=in_dim,
        e_hidden_dim=16,
        latent_dim=4,
        d_hidden_dim=16,
        out_dim=num_pts * 3,
        e_hidden_layers=3,
        d_hidden_layers=3,
    )


def example_input_cryostar():
    batch = 3
    down_side_shape = 4
    in_dim = 2 * down_side_shape**2
    x = torch.randn(batch, in_dim)
    idx = torch.arange(batch)
    return (x, idx)


MENAGERIE_ENTRIES = [
    ("CryoSTAR", "build_cryostar", "example_input_cryostar", 2024, "vendored"),
]
