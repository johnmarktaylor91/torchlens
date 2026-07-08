# SOURCE: vendored from FLC-QU-hep/getting_high @ master (BIBAE/BIBAE_models.py)

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class BiBAE_F_3D_LayerNorm_SmallLatent(nn.Module):
    """
    Generator component of WGAN, adapted as VAE, with direct energy conditioning.
    """

    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        """Initialize the official BIB-AE 3D layer-normalized generator."""

        super().__init__()
        self.ngf = ngf
        self.nc = nc
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        self.enconv1 = nn.Conv3d(
            in_channels=1,
            out_channels=ngf,
            kernel_size=(4, 4, 4),
            stride=(2, 2, 2),
            padding=(2, 2, 2),
            bias=False,
            padding_mode="zeros",
        )
        self.bnen1 = torch.nn.LayerNorm([16, 16, 16])
        self.enconv2 = nn.Conv3d(
            in_channels=ngf,
            out_channels=ngf * 2,
            kernel_size=(4, 4, 4),
            stride=(2, 2, 2),
            padding=(2, 2, 2),
            bias=False,
            padding_mode="zeros",
        )
        self.bnen2 = torch.nn.LayerNorm([9, 9, 9])
        self.enconv3 = nn.Conv3d(
            in_channels=ngf * 2,
            out_channels=ngf * 4,
            kernel_size=(4, 4, 4),
            stride=(2, 2, 2),
            padding=(2, 2, 2),
            bias=False,
            padding_mode="zeros",
        )
        self.bnen3 = torch.nn.LayerNorm([5, 5, 5])
        self.enconv4 = nn.Conv3d(
            in_channels=ngf * 4,
            out_channels=ngf * 8,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
            padding_mode="zeros",
        )
        self.bnen4 = torch.nn.LayerNorm([5, 5, 5])

        self.fc1 = nn.Linear(5 * 5 * 5 * ngf * 8 + 1, ngf * 500, bias=True)
        self.fc2 = nn.Linear(ngf * 500, int(self.z_full * 1.5), bias=True)

        self.fc31 = nn.Linear(int(self.z_full * 1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full * 1.5), self.z_enc, bias=True)

        self.cond1 = torch.nn.Linear(self.z_full + 1, int(self.z_full * 1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full * 1.5), ngf * 500, bias=True)
        self.cond3 = torch.nn.Linear(ngf * 500, 10 * 10 * 10 * ngf, bias=True)

        self.deconv1 = torch.nn.ConvTranspose3d(
            ngf,
            ngf,
            kernel_size=(3, 3, 3),
            stride=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bnde1 = torch.nn.LayerNorm([30, 30, 30])
        self.deconv2 = torch.nn.ConvTranspose3d(
            ngf,
            ngf * 2,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bnde2 = torch.nn.LayerNorm([60, 60, 60])

        self.conv0 = torch.nn.Conv3d(
            ngf * 2,
            ngf,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=(0, 0, 0),
            bias=False,
        )
        self.bnco0 = torch.nn.LayerNorm([30, 30, 30])
        self.conv1 = torch.nn.Conv3d(
            ngf,
            ngf * 2,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bnco1 = torch.nn.LayerNorm([30, 30, 30])
        self.conv2 = torch.nn.Conv3d(
            ngf * 2,
            ngf * 4,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bnco2 = torch.nn.LayerNorm([30, 30, 30])
        self.conv3 = torch.nn.Conv3d(
            ngf * 4,
            ngf * 2,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bnco3 = torch.nn.LayerNorm([30, 30, 30])
        self.conv4 = torch.nn.Conv3d(
            ngf * 2,
            1,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )

    def encode(self, x, E_true):
        """Encode a shower image and true energy into latent parameters."""

        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1, 1, 30, 30, 30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat((x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4)), E_true), 1)

        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat(
            (self.fc31(x), torch.zeros(x.size(0), self.z_rand, device=self.device)),
            1,
        ), torch.cat(
            (self.fc32(x), torch.zeros(x.size(0), self.z_rand, device=self.device)),
            1,
        )

    def reparameterize(self, mu, logvar):
        """Sample a latent code with the reparameterization trick."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode a latent code into a generated 3D shower image."""

        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)

        x = x.view(-1, self.ngf, 10, 10, 10)

        x = F.leaky_relu(
            self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])),
            0.2,
            inplace=True,
        )
        x = F.leaky_relu(
            self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])),
            0.2,
            inplace=True,
        )

        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x

    def forward(self, x, E_true, z=None, mode="full"):
        """Run encode, decode, or full BIB-AE forward pass."""

        if mode == "encode":
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z
        if mode == "decode":
            return self.decode(torch.cat((z, E_true), 1))
        if mode == "full":
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z, E_true), 1)), mu, logvar, z
        return None


def build_bib_ae() -> BiBAE_F_3D_LayerNorm_SmallLatent:
    """Build a small-channel BIB-AE model."""

    model = BiBAE_F_3D_LayerNorm_SmallLatent(
        args=SimpleNamespace(),
        device=torch.device("cpu"),
        ngf=1,
        z_rand=4,
        z_enc=2,
    )
    model.eval()
    return model


def example_input_bib_ae() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a 3D shower image and scalar energy input."""

    return torch.rand(1, 30, 30, 30), torch.rand(1, 1)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BIB-AE", "build_bib_ae", "example_input_bib_ae", "2020", "CV2c_133"),
]
