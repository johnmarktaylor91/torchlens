# SOURCE: vendored from autonomousvision/texture_fields @ master
# (mesh2tex/texnet/models/__init__.py, mesh2tex/texnet/models/decoder.py,
# mesh2tex/texnet/models/image_encoder.py, mesh2tex/texnet/models/vae_encoder.py,
# mesh2tex/geometry/pointnet.py, mesh2tex/layers.py)
# https://github.com/autonomousvision/texture_fields -- "Texture Fields: Learning
# Texture Representations in Function Space" (Oechsle, Mescheder, Niemeyer, Strauss,
# Geiger, ICCV 2019). The core architectural contribution is `TextureNetwork`: a
# conditional-VAE generator that represents a continuous texture field over 3D mesh
# coordinates via an implicit decoder MLP (`DecoderEachLayerC`, injecting a shape
# code `c` and latent code `z` at every ResNet block, per §3.2 of the paper),
# conditioned on a geometry encoder (`SimplePointnet`, a PointNet-style encoder over
# mesh point/normal samples) and, when generating from an image, an image encoder
# (`Resnet18`) plus a VAE posterior encoder (`vae_encoder.Resnet`). `TextureNetwork`,
# `DecoderEachLayerC`/`DecoderEachLayerCLarger`, `Resnet18`, `vae_encoder.Resnet`,
# `SimplePointnet`, and the shared layer primitives in `mesh2tex/layers.py`
# (`ResnetBlockFC`, `ResnetBlockConv1D`, `ResnetBlockPointwise`, `ResnetBlockConv2d`,
# `EqualizedLR`, `pixel_norm`) are transcribed verbatim. Only `mesh2tex.common`'s
# `normalize_imagenet` helper is inlined directly into `image_encoder.py` (it was
# already duplicated there in the original repo) and `Resnet18`'s
# `models.resnet18(pretrained=True)` is switched to `pretrained=False` so the module
# builds without a network fetch (same real torchvision ResNet18 class, just
# randomly initialized instead of loading ImageNet weights). `TextureNetwork`'s
# unused `load_mesh2facecenter` static method (a `trimesh` file-loading utility, not
# part of the trainable network) was dropped.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torchvision import models

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from mesh2tex/layers.py ----
class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockConv1D(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockPointwise(nn.Module):
    def __init__(
        self, f_in, f_out=None, f_hidden=None, is_bias=True, actvn=F.relu, factor=1.0, eq_lr=False
    ):
        super().__init__()
        if f_out is None:
            f_out = f_in

        if f_hidden is None:
            f_hidden = min(f_in, f_out)

        self.f_in = f_in
        self.f_hidden = f_hidden
        self.f_out = f_out

        self.factor = factor
        self.eq_lr = eq_lr

        self.actvn = actvn

        self.conv_0 = nn.Conv1d(f_in, f_hidden, 1)
        self.conv_1 = nn.Conv1d(f_hidden, f_out, 1, bias=is_bias)

        if self.eq_lr:
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)

        if f_in == f_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv1d(f_in, f_out, 1, bias=False)
            if self.eq_lr:
                self.shortcut = EqualizedLR(self.shortcut)

        nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        net = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(net))
        x_s = self.shortcut(x)
        return x_s + self.factor * dx


class ResnetBlockConv2d(nn.Module):
    def __init__(
        self,
        f_in,
        f_out=None,
        f_hidden=None,
        is_bias=True,
        actvn=F.relu,
        factor=1.0,
        eq_lr=False,
        pixel_norm=False,
    ):
        super().__init__()
        if f_out is None:
            f_out = f_in

        if f_hidden is None:
            f_hidden = min(f_in, f_out)

        self.f_in = f_in
        self.f_hidden = f_hidden
        self.f_out = f_out
        self.factor = factor
        self.eq_lr = eq_lr
        self.use_pixel_norm = pixel_norm

        self.actvn = actvn

        self.conv_0 = nn.Conv2d(self.f_in, self.f_hidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.f_hidden, self.f_out, 3, stride=1, padding=1, bias=is_bias)

        if self.eq_lr:
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)

        if f_in == f_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv2d(f_in, f_out, 1, bias=False)
            if self.eq_lr:
                self.shortcut = EqualizedLR(self.shortcut)

        nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        x_s = self.shortcut(x)

        if self.use_pixel_norm:
            x = pixel_norm(x)
        dx = self.conv_0(self.actvn(x))

        if self.use_pixel_norm:
            dx = pixel_norm(dx)
        dx = self.conv_1(self.actvn(dx))

        out = x_s + self.factor * dx

        return out


class EqualizedLR(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._make_params()

    def _make_params(self):
        weight = self.module.weight

        height = weight.data.shape[0]
        width = weight.view(height, -1).data.shape[1]

        del self.module._parameters["weight"]
        self.module.weight = None

        self.weight = nn.Parameter(weight.data)

        self.factor = np.sqrt(2 / width)

        nn.init.normal_(self.weight)

        self.bias = self.module.bias
        self.module.bias = None

        if self.bias is not None:
            del self.module._parameters["bias"]
            nn.init.zeros_(self.bias)

    def forward(self, *args, **kwargs):
        self.module.weight = self.factor * self.weight
        if self.bias is not None:
            self.module.bias = 1.0 * self.bias
        out = self.module.forward(*args, **kwargs)
        self.module.weight = None
        self.module.bias = None
        return out


def pixel_norm(x):
    sigma = x.norm(dim=1, keepdim=True)
    out = x / (sigma + 1e-5)
    return out


# ---- verbatim from mesh2tex/geometry/pointnet.py ----
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    def __init__(self, c_dim=128, hidden_dim=128, leaky=False, eq_lr=False):
        super().__init__()
        self.c_dim = c_dim
        self.eq_lr = eq_lr

        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = maxpool

        self.conv_p = nn.Conv1d(6, 2 * hidden_dim, 1)
        self.conv_0 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.conv_1 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.conv_3 = nn.Conv1d(2 * hidden_dim, hidden_dim, 1)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        if self.eq_lr:
            self.conv_p = EqualizedLR(self.conv_p)
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)
            self.conv_2 = EqualizedLR(self.conv_2)
            self.conv_3 = EqualizedLR(self.conv_3)
            self.fc_c = EqualizedLR(self.fc_c)

    def forward(self, geometry):
        p = geometry["points"]
        n = geometry["normals"]

        pn = torch.cat([p, n], dim=1)
        net = self.conv_p(pn)

        net = self.conv_0(self.actvn(net))
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled = pooled.expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.conv_1(self.actvn(net))
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled = pooled.expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.conv_2(self.actvn(net))
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled = pooled.expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.conv_3(self.actvn(net))

        net = self.pool(net, dim=2)

        c = self.fc_c(self.actvn(net))

        geom_descr = {
            "global": c,
        }

        return geom_descr


# ---- verbatim from mesh2tex/texnet/models/decoder.py ----
class DecoderEachLayerC(nn.Module):
    def __init__(
        self,
        c_dim=128,
        z_dim=128,
        dim=3,
        hidden_size=128,
        leaky=True,
        resnet_leaky=True,
        eq_lr=False,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.eq_lr = eq_lr

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if not resnet_leaky:
            self.resnet_actvn = F.relu
        else:
            self.resnet_actvn = lambda x: F.leaky_relu(x, 0.2)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)

        self.block0 = ResnetBlockPointwise(hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block1 = ResnetBlockPointwise(hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block2 = ResnetBlockPointwise(hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block3 = ResnetBlockPointwise(hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block4 = ResnetBlockPointwise(hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)

        self.fc_cz_0 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_1 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_2 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_3 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_4 = nn.Linear(c_dim + z_dim, hidden_size)

        self.conv_out = nn.Conv1d(hidden_size, 3, 1)

        if self.eq_lr:
            self.conv_p = EqualizedLR(self.conv_p)
            self.conv_out = EqualizedLR(self.conv_out)
            self.fc_cz_0 = EqualizedLR(self.fc_cz_0)
            self.fc_cz_1 = EqualizedLR(self.fc_cz_1)
            self.fc_cz_2 = EqualizedLR(self.fc_cz_2)
            self.fc_cz_3 = EqualizedLR(self.fc_cz_3)
            self.fc_cz_4 = EqualizedLR(self.fc_cz_4)

        nn.init.zeros_(self.conv_out.weight)

    def forward(self, p, geom_descr, z, **kwargs):
        c = geom_descr["global"]
        batch_size, D, T = p.size()

        cz = torch.cat([c, z], dim=1)
        net = self.conv_p(p)
        net = net + self.fc_cz_0(cz).unsqueeze(2)
        net = self.block0(net)
        net = net + self.fc_cz_1(cz).unsqueeze(2)
        net = self.block1(net)
        net = net + self.fc_cz_2(cz).unsqueeze(2)
        net = self.block2(net)
        net = net + self.fc_cz_3(cz).unsqueeze(2)
        net = self.block3(net)
        net = net + self.fc_cz_4(cz).unsqueeze(2)
        net = self.block4(net)

        out = self.conv_out(self.actvn(net))
        out = torch.sigmoid(out)

        return out


# ---- verbatim from mesh2tex/texnet/models/image_encoder.py (normalize_imagenet
# inlined from mesh2tex/common.py, pretrained=False to avoid a network fetch) ----
def normalize_imagenet(x):
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class Resnet18(nn.Module):
    """ResNet-18 conditioning network."""

    def __init__(self, c_dim=128, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=False)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError("c_dim must be 512 if use_linear is False")

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


# ---- verbatim from mesh2tex/texnet/models/vae_encoder.py ----
class VAEEncoderResnet(nn.Module):
    def __init__(
        self,
        img_size,
        z_dim=128,
        c_dim=128,
        embed_size=256,
        nfilter=32,
        nfilter_max=1024,
        leaky=True,
        eq_lr=False,
    ):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.eq_lr = eq_lr
        self.c_dim = c_dim

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlockConv2d(nf, nf, actvn=self.actvn, eq_lr=eq_lr)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlockConv2d(nf0, nf1, actvn=self.actvn, eq_lr=eq_lr),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc_mean = nn.Linear(self.nf0 * s0 * s0, z_dim)
        self.fc_logstd = nn.Linear(self.nf0 * s0 * s0, z_dim)
        self.fc_inject_c = nn.Linear(self.c_dim, 1 * nf)
        if self.eq_lr:
            self.conv_img = EqualizedLR(self.conv_img)
            self.fc = EqualizedLR(self.fc)

    def forward(self, x, geom_descr):
        c = geom_descr["global"]
        batch_size = x.size(0)

        out = self.conv_img(x)
        add = self.fc_inject_c(c).view(out.size(0), self.nf, 1, 1)
        out = out + add
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)

        mean = self.fc_mean(self.actvn(out))
        logstd = self.fc_logstd(self.actvn(out))
        return mean, logstd


# ---- verbatim from mesh2tex/texnet/models/__init__.py ----
class TextureNetwork(nn.Module):
    def __init__(
        self, decoder, geometry_encoder, encoder=None, vae_encoder=None, p0_z=None, white_bg=True
    ):
        super().__init__()

        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder
        self.encoder = encoder
        self.geometry_encoder = geometry_encoder
        self.vae_encoder = vae_encoder
        self.p0_z = p0_z
        self.white_bg = white_bg

    def forward(self, depth, cam_K, cam_W, geometry, condition=None, z=None, sample=True):
        batch_size, _, N, M = depth.size()
        assert depth.size(1) == 1
        assert cam_K.size() == (batch_size, 3, 4)
        assert cam_W.size() == (batch_size, 3, 4)

        loc3d, mask = self.depth_map_to_3d(depth, cam_K, cam_W)
        geom_descr = self.encode_geometry(geometry)

        if self.encoder is not None:
            z = self.encode(condition)
        elif z is None:
            z = self.get_z_from_prior((batch_size,), sample=sample)

        loc3d = loc3d.view(batch_size, 3, N * M)
        x = self.decode(loc3d, geom_descr, z)
        x = x.view(batch_size, 3, N, M)

        if self.white_bg is False:
            x_bg = torch.zeros_like(x)
        else:
            x_bg = torch.ones_like(x)

        img = (mask * x).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * x_bg

        return img

    def elbo(self, image_real, depth, cam_K, cam_W, geometry):
        batch_size, _, N, M = depth.size()

        assert depth.size(1) == 1
        assert cam_K.size() == (batch_size, 3, 4)
        assert cam_W.size() == (batch_size, 3, 4)

        loc3d, mask = self.depth_map_to_3d(depth, cam_K, cam_W)
        geom_descr = self.encode_geometry(geometry)

        q_z = self.infer_z(image_real, geom_descr)
        z = q_z.rsample()

        loc3d = loc3d.view(batch_size, 3, N * M)
        x = self.decode(loc3d, geom_descr, z)
        x = x.view(batch_size, 3, N, M)

        if self.white_bg is False:
            x_bg = torch.zeros_like(x)
        else:
            x_bg = torch.ones_like(x)

        image_fake = (mask * x).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * x_bg

        recon_loss = F.mse_loss(image_fake, image_real).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = recon_loss.mean() + kl.mean() / float(N * M * 3)
        return elbo, recon_loss.mean(), kl.mean() / float(N * M * 3), image_fake

    def encode(self, cond):
        z = self.encoder(cond)
        return z

    def encode_geometry(self, geometry):
        geom_descr = self.geometry_encoder(geometry)
        return geom_descr

    def decode(self, loc3d, c, z):
        rgb = self.decoder(loc3d, c, z)
        return rgb

    def depth_map_to_3d(self, depth, cam_K, cam_W):
        assert depth.size(1) == 1
        batch_size, _, N, M = depth.size()
        device = depth.device
        depth = -depth.permute(0, 1, 3, 2)

        zero_one_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        zero_one_row = zero_one_row.expand(batch_size, 1, 4).to(device)

        cam_W = torch.cat((cam_W, zero_one_row), dim=1)

        mask = (depth.abs() != float("Inf")).float()
        depth[depth == float("Inf")] = 0
        depth[depth == -1 * float("Inf")] = 0

        d = depth.reshape(batch_size, 1, N * M)

        px, py = torch.meshgrid([torch.arange(0, N), torch.arange(0, M)], indexing="ij")
        px, py = px.to(device), py.to(device)

        p = torch.cat(
            (
                px.expand(batch_size, 1, px.size(0), px.size(1)),
                (M - py).expand(batch_size, 1, py.size(0), py.size(1)),
            ),
            dim=1,
        )
        p = p.reshape(batch_size, 2, py.size(0) * py.size(1))
        p = p.float() / M * 2

        P = cam_K[:, :2, :2].float().to(device)
        q = cam_K[:, 2:3, 2:3].float().to(device)
        b = cam_K[:, :2, 2:3].expand(batch_size, 2, d.size(2)).to(device)
        Inv_P = torch.inverse(P).to(device)

        rightside = (p.float() * q.float() - b.float()) * d.float()
        x_xy = torch.bmm(Inv_P, rightside)

        x_world = torch.cat((x_xy, d, torch.ones_like(d)), dim=1)

        Inv_W = torch.inverse(cam_W)
        loc3d = torch.bmm(Inv_W.expand(batch_size, 4, 4), x_world).reshape(batch_size, 4, N, M)

        loc3d = loc3d[:, :3].to(device)
        mask = mask.to(device)
        return loc3d, mask

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        if sample:
            z = self.p0_z.sample(size)
        else:
            z = self.p0_z.mean
            z = z.expand(*size, *z.size())

        return z

    def infer_z(self, image, c, **kwargs):
        if self.vae_encoder is not None:
            mean_z, logstd_z = self.vae_encoder(image, c, **kwargs)
        else:
            batch_size = image.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_texture_fields():
    torch.manual_seed(0)
    c_dim = 16
    z_dim = 8
    img_size = 32

    decoder = DecoderEachLayerC(c_dim=c_dim, z_dim=z_dim, dim=3, hidden_size=16)
    geometry_encoder = SimplePointnet(c_dim=c_dim, hidden_dim=16)
    # NOTE: at generation time `TextureNetwork.forward` calls `z = self.encode(condition)`
    # (image encoder path) and feeds that `z` straight into the decoder's `z_dim`-sized
    # slot -- so the image encoder must be built with c_dim==z_dim here, matching how the
    # image-conditioned (non-VAE-posterior) generation path is actually used at inference.
    encoder = Resnet18(c_dim=z_dim)
    vae_encoder = VAEEncoderResnet(img_size=img_size, z_dim=z_dim, c_dim=c_dim, nfilter=8)

    model = TextureNetwork(
        decoder, geometry_encoder, encoder, vae_encoder, p0_z=None, white_bg=True
    )
    model.eval()
    return model


def example_input_texture_fields():
    torch.manual_seed(0)
    batch_size = 2
    n_pixels = 8
    n_points = 32
    img_size = 32

    depth = torch.rand(batch_size, 1, n_pixels, n_pixels) + 0.5
    cam_K = torch.eye(3, 4).unsqueeze(0).repeat(batch_size, 1, 1)
    cam_W = torch.eye(3, 4).unsqueeze(0).repeat(batch_size, 1, 1)
    geometry = {
        "points": torch.randn(batch_size, 3, n_points),
        "normals": torch.randn(batch_size, 3, n_points),
    }
    condition = torch.rand(batch_size, 3, img_size, img_size)
    return (depth, cam_K, cam_W, geometry, condition)


MENAGERIE_ENTRIES = [
    ("TextureFields", build_texture_fields, example_input_texture_fields, 2019, "vendored-pytorch"),
]
