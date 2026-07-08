# SOURCE: vendored from ml-struct-bio/cryodrgn @ main
# https://raw.githubusercontent.com/ml-struct-bio/cryodrgn/main/cryodrgn/models.py
# https://raw.githubusercontent.com/ml-struct-bio/cryodrgn/main/cryodrgn/lattice.py
#
# cryoDRGN -- Zhong, Bepler, Berger & Davis 2021 (Nature Methods) "CryoDRGN:
# reconstruction of heterogeneous cryo-EM structures using neural networks" --
# the base heterogeneous-reconstruction VAE (the encoder maps a particle image to
# a latent conformation code `z`; the decoder is an implicit MLP that maps a 3D
# Fourier-space coordinate (+ z) directly to a Hartley-transform voxel value).
# The referenced "latent-space cryo-EM diffusion" extension (Kreis et al.,
# arXiv:2211.14169) is a diffusion prior trained on top of this VAE's frozen
# latent space; its own diffusion-model code is not public, so this module
# vendors the real, public, foundational cryoDRGN VAE itself (`HetOnlyVAE` with
# its default `encode_mode="resid"`/`domain="fourier"` configuration --
# `ResidLinearMLP` encoder + `FTPositionalDecoder` implicit decoder). `Lattice`,
# `Decoder`, `HetOnlyVAE`, `get_decoder`, `FTPositionalDecoder`, `ResidLinearMLP`,
# `MyLinear`, `ResidLinear`, and `ConvEncoder` below are copied verbatim from the
# real `cryodrgn/models.py` and `cryodrgn/lattice.py`. Two narrow trims, both
# outside the traced forward path: (1) `HetOnlyVAE.load()` (a yaml-config +
# checkpoint loader, not architecture) is omitted since this module constructs
# `HetOnlyVAE` directly; (2) the `encode_mode="tilt"` branch (`TiltEncoder`/
# `SO3reparameterize`, which pull in `cryodrgn.lie_tools`) is omitted since it is
# not exercised by the default `encode_mode="resid"` configuration used here --
# the `resid`/`mlp`/`conv` encoder branches that ARE reachable are all present
# verbatim. No architectural code was rewritten.
#
# Upstream license: GPLv3 (ml-struct-bio/cryodrgn).

from typing import Optional, Tuple, Type

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# --- verbatim from cryodrgn/lattice.py (Lattice) --------------------------------


class Lattice:
    def __init__(self, D: int, extent: float = 0.5, ignore_DC: bool = True, device=None):
        assert D % 2 == 1, "Lattice size must be odd"
        x0, x1 = np.meshgrid(
            np.linspace(-extent, extent, D, endpoint=True),
            np.linspace(-extent, extent, D, endpoint=True),
        )
        coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(D**2)], 1).astype(np.float32)
        self.coords = torch.tensor(coords, device=device)
        self.extent = extent
        self.D = D
        self.D2 = int(D / 2)
        self.center = torch.tensor([0.0, 0.0], device=device)
        self.square_masks = {}
        self.circle_masks = {}
        self.freqs2d = self.coords[:, 0:2] / extent / 2
        self.ignore_DC = ignore_DC
        self.device = device


# --- verbatim from cryodrgn/models.py --------------------------------------------


class Decoder(nn.Module):
    def eval_volume(self, coords, D, extent, norm, zval=None):
        raise NotImplementedError

    def get_voxel_decoder(self) -> Optional["Decoder"]:
        return None


class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(
        self,
        lattice: Lattice,
        qlayers: int,
        qdim: int,
        players: int,
        pdim: int,
        in_dim: int,
        zdim: int = 1,
        encode_mode: str = "resid",
        enc_mask=None,
        enc_type="linear_lowf",
        enc_dim=None,
        domain="fourier",
        activation=nn.ReLU,
        feat_sigma: Optional[float] = None,
        tilt_params={},
    ):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        if encode_mode == "conv":
            self.encoder = ConvEncoder(qdim, zdim * 2)
        elif encode_mode == "resid":
            self.encoder = ResidLinearMLP(
                in_dim,
                qlayers,
                qdim,
                zdim * 2,
                activation,  # nlayers  # hidden_dim  # out_dim
            )
        elif encode_mode == "mlp":
            self.encoder = MLP(
                in_dim,
                qlayers,
                qdim,
                zdim * 2,
                activation,  # hidden_dim  # out_dim
            )  # in_dim -> hidden_dim
        else:
            raise RuntimeError("Encoder mode {} not recognized".format(encode_mode))
        self.encode_mode = encode_mode
        self.decoder = get_decoder(
            3 + zdim,
            lattice.D,
            players,
            pdim,
            domain,
            enc_type,
            enc_dim,
            activation,
            feat_sigma,
        )

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, *img) -> Tuple[Tensor, Tensor]:
        img = (x.view(x.shape[0], -1) for x in img)
        if self.enc_mask is not None:
            img = (x[:, self.enc_mask] for x in img)
        z = self.encoder(*img)
        return z[:, : self.zdim], z[:, self.zdim :]

    def cat_z(self, coords, z) -> Tensor:
        """
        coords: Bx...x3
        z: Bxzdim
        """
        assert coords.size(0) == z.size(0), (coords.shape, z.shape)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat((coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, z=None) -> torch.Tensor:
        """
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        """
        decoder = self.decoder
        assert isinstance(decoder, nn.Module)
        retval = decoder(self.cat_z(coords, z) if z is not None else coords)
        return retval

    # Need forward func for DataParallel -- TODO: refactor
    def forward(self, *args, **kwargs):
        return self.decode(*args, **kwargs)


class MyLinear(nn.Linear):
    def forward(self, input):
        if input.dtype == torch.half:
            return F.linear(input, self.weight.half(), self.bias.half())
        else:
            return F.linear(input, self.weight, self.bias)


class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = MyLinear(nin, nout)

    def forward(self, x):
        z = self.linear(x) + x
        return z


class ResidLinearMLP(Decoder):
    def __init__(
        self,
        in_dim: int,
        nlayers: int,
        hidden_dim: int,
        out_dim: int,
        activation: Type,
    ):
        super(ResidLinearMLP, self).__init__()
        layers = [
            (
                ResidLinear(in_dim, hidden_dim)
                if in_dim == hidden_dim
                else MyLinear(in_dim, hidden_dim)
            ),
            activation(),
        ]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(
            ResidLinear(hidden_dim, out_dim)
            if out_dim == hidden_dim
            else MyLinear(hidden_dim, out_dim)
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        flat = x.view(-1, x.shape[-1])
        ret_flat = self.main(flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        return ret


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nlayers: int,
        hidden_dim: int,
        out_dim: int,
        activation: Type,
    ):
        super(MLP, self).__init__()
        layers = [MyLinear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(MyLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(MyLinear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# Adapted from soumith DCGAN
class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(ConvEncoder, self).__init__()
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_dim, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.main(x)
        return x.view(x.size(0), -1)  # flatten


def get_decoder(
    in_dim: int,
    D: int,
    layers: int,
    dim: int,
    domain: str,
    enc_type: str,
    enc_dim: Optional[int] = None,
    activation: Type = nn.ReLU,
    feat_sigma: Optional[float] = None,
) -> Decoder:
    if enc_type == "none":
        if domain == "hartley":
            model = ResidLinearMLP(in_dim, layers, dim, 1, activation)
        else:
            raise NotImplementedError("FTSliceDecoder not vendored (unused path)")
    else:
        model_t = PositionalDecoder if domain == "hartley" else FTPositionalDecoder
        model = model_t(
            in_dim,
            D,
            layers,
            dim,
            activation,
            enc_type=enc_type,
            enc_dim=enc_dim,
            feat_sigma=feat_sigma,
        )
    return model


class PositionalDecoder(Decoder):
    def __init__(
        self,
        in_dim,
        D,
        nlayers,
        hidden_dim,
        activation,
        enc_type="linear_lowf",
        enc_dim=None,
        feat_sigma: Optional[float] = None,
    ):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.enc_type = enc_type
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation)

        if enc_type == "gaussian":
            rand_freqs = torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * feat_sigma
            self.rand_freqs = Parameter(rand_freqs, requires_grad=False)
        else:
            self.rand_feats = None

    def positional_encoding_geom(self, coords):
        if self.enc_type == "gaussian":
            return self.random_fourier_encoding(coords)
        freqs = torch.arange(self.enc_dim, dtype=torch.float, device=coords.device)
        if self.enc_type == "geom_ft":
            freqs = self.DD * np.pi * (2.0 / self.DD) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == "geom_full":
            freqs = self.DD * np.pi * (1.0 / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == "geom_lowf":
            freqs = self.D2 * (1.0 / self.D2) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == "geom_nohighf":
            freqs = self.D2 * (2.0 * np.pi / self.D2) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == "linear_lowf":
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError("Encoding type {} not recognized".format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def random_fourier_encoding(self, coords):
        assert self.rand_freqs is not None
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2
        kxkykz = coords[..., None, 0:3] * freqs
        k = kxkykz.sum(-1)
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float, device=coords.device)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, coords: Tensor) -> Tensor:
        """Input should be coordinates from [-.5,.5]"""
        assert (coords[..., 0:3].abs() - 0.5 < 1e-4).all()
        return self.decoder(self.positional_encoding_geom(coords))


class FTPositionalDecoder(Decoder):
    def __init__(
        self,
        in_dim: int,
        D: int,
        nlayers: int,
        hidden_dim: int,
        activation: Type,
        enc_type: str = "linear_lowf",
        enc_dim: Optional[int] = None,
        feat_sigma: Optional[float] = None,
    ):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)

        if enc_type == "gaussian":
            rand_freqs = torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * feat_sigma
            self.rand_freqs = Parameter(rand_freqs, requires_grad=False)
        else:
            self.rand_feats = None

    def positional_encoding_geom(self, coords: Tensor) -> Tensor:
        """Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi"""
        if self.enc_type == "gaussian":
            return self.random_fourier_encoding(coords)
        freqs = torch.arange(self.enc_dim, dtype=torch.float, device=coords.device)
        if self.enc_type == "geom_ft":
            freqs = self.DD * np.pi * (2.0 / self.DD) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == "geom_full":
            freqs = self.DD * np.pi * (1.0 / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == "geom_lowf":
            freqs = self.D2 * (1.0 / self.D2) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == "geom_nohighf":
            freqs = self.D2 * (2.0 * np.pi / self.D2) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == "linear_lowf":
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError("Encoding type {} not recognized".format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def random_fourier_encoding(self, coords):
        assert self.rand_freqs is not None
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2
        kxkykz = coords[..., None, 0:3] * freqs
        k = kxkykz.sum(-1)
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords: Tensor) -> Tensor:
        """Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2"""
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float, device=coords.device)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, lattice: Tensor) -> Tensor:
        """
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        """
        c = lattice.shape[-2] // 2  # top half
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c  # include the origin
        assert abs(lattice[..., 0:3].mean()) < 1e-4, "{} != 0.0".format(lattice[..., 0:3].mean())
        image = torch.empty(lattice.shape[:-1], device=lattice.device)
        top_half = self.decode(lattice[..., 0:cc, :])
        image[..., 0:cc] = top_half[..., 0] - top_half[..., 1]
        # the bottom half of the image is the complex conjugate of the top half
        image[..., cc:] = (top_half[..., 0] + top_half[..., 1])[..., np.arange(c - 1, -1, -1)]
        return image

    def decode(self, lattice: Tensor):
        """Return FT transform"""
        assert (lattice[..., 0:3].abs() - 0.5 < 1e-4).all()
        # convention: only evalute the -z points
        w = lattice[..., 2] > 0.0
        new_lattice = lattice.clone()
        # negate lattice coordinates where z > 0
        new_lattice[..., 0:3][w] *= -1
        result = self.decoder(self.positional_encoding_geom(new_lattice))
        # replace with complex conjugate to get correct values for original lattice positions
        result[..., 1][w] *= -1
        return result


# --- menagerie staging wrapper --------------------------------------------------
# HetOnlyVAE.forward(*args, **kwargs) delegates straight to decode(coords, z);
# tracing the decode path (encoder -> reparameterize -> decoder) end to end
# requires a tiny wrapper that runs the real encode/decode methods on a
# particle-image-shaped input, matching the model's actual training-time usage
# (see cryodrgn/commands/train_vae.py: model.encode(...) -> model.decode(coords, z)).


class HetOnlyVAETraceWrapper(nn.Module):
    def __init__(self, D=9, zdim=4):
        super().__init__()
        self.lattice = Lattice(D)
        self.vae = HetOnlyVAE(
            self.lattice,
            qlayers=2,
            qdim=32,
            players=2,
            pdim=32,
            in_dim=D * D,
            zdim=zdim,
            encode_mode="resid",
            enc_mask=None,
            domain="fourier",
        )

    def forward(self, img, coords):
        mu, logvar = self.vae.encode(img)
        z = self.vae.reparameterize(mu, logvar)
        return self.vae.decode(coords, z)


def build_cryodrgn_hetonlyvae():
    model = HetOnlyVAETraceWrapper()
    model.eval()
    return model


def example_input_cryodrgn_hetonlyvae():
    D = 9
    b = 2
    img = torch.randn(b, D, D)
    # Use the real Lattice's own coordinate grid (exactly centered by
    # construction, as FTPositionalDecoder.forward requires) rather than
    # arbitrary random coords, matching real train_vae.py usage.
    lattice = Lattice(D)
    coords = lattice.coords.unsqueeze(0).expand(b, -1, -1).contiguous()
    return (img, coords)


MENAGERIE_ENTRIES = [
    (
        "cryoDRGN (HetOnlyVAE)",
        "build_cryodrgn_hetonlyvae",
        "example_input_cryodrgn_hetonlyvae",
        2021,
        "vendored-pytorch",
    ),
]
