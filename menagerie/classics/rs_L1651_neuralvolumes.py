# SOURCE: vendored from facebookresearch/neuralvolumes @ main
#
# Neural Volumes (Lombardi, Simon, Saragih, Schwartz, Lehrmann, Sheikh. 2019, SIGGRAPH,
# "Neural Volumes: Learning Dynamic Renderable Volumes from Images"). Learns an implicit
# volumetric (RGBA voxel grid) representation of a dynamic scene from multi-view video via
# a VAE-style encoder-decoder plus a differentiable ray-marching renderer, with a learned
# warp field to concentrate representational capacity where needed.
#
# Files vendored verbatim (architecture unmodified, only import paths adjusted to be
# self-contained in this single staging module):
#   models/neurvol1.py           -> Autoencoder (top-level model: encode -> decode ->
#                                    differentiable ray-march through the warped voxel
#                                    template -> composite RGB/alpha).
#   models/encoders/mvconv1.py   -> Encoder (multi-view strided-conv VAE encoder producing
#                                    a 256-d latent mu/logstd code).
#   models/decoders/voxel1.py    -> Decoder (ConvTemplate/LinearTemplate: latent -> 3D
#                                    deconv voxel template; AffineMixWarp/ConvWarp: latent
#                                    -> learned voxel warp field; global rigid warp branch).
#   models/volsamplers/warpvoxel.py -> VolSampler (applies the global + local warp fields
#                                    via `grid_sample`, then samples the RGBA template at
#                                    the warped ray position).
#   models/colorcals/colorcal1.py -> Colorcal (per-camera learnable 1x1 grouped-conv color
#                                    calibration correction).
#   models/utils.py               -> initmod/initseq (Xavier init helpers used by every
#                                    submodule above) and Quaternion (quaternion -> 3x3
#                                    rotation matrix, used by the global + per-part warp
#                                    branches).
#
# What is kept: every architectural mechanism from the real repo -- the multi-view conv
# VAE encoder, the deconv voxel template decoder (conv-transpose-3d stack), the
# affine-mixture local warp field + global rigid (quaternion) warp, VolSampler's warped
# grid-sample lookup, per-camera color calibration, and the differentiable fixed-step
# ray-marching compositing loop from `Autoencoder.forward` (all real torch ops, no
# custom CUDA/C++ extensions anywhere in the original repo).
#
# What is adapted (plumbing, not architecture): the real `Autoencoder.__init__` takes a
# `dataset` object and pulls `get_allcameras()` / `get_krt()` / `known_background()` /
# `imagemean` / `imagestd` off it to size the per-camera background parameter dict and
# color-calibration module dict. The real `Dataset` class (`data/dryice1.py`) loads KRT
# camera-calibration files and JPEG frames from disk and cannot construct without that
# dataset directory. Here `_TinyCameraSet` is a minimal duck-typed stand-in exposing the
# same 4 accessors with synthetic camera calibration for a single tiny camera -- this is
# configuration plumbing (analogous to a tiny `dataset`/`config` object passed to any
# constructor), not a reimplementation of any network layer. `estimatebg=False` (the
# real repo's own default) is kept in this build, so the background parameter is
# unused/non-trainable in the forward path, matching the real code's own gating.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# models/utils.py (verbatim, only the pieces neurvol1/voxel1/mvconv1 use)
# ---------------------------------------------------------------------------


def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = (
            m.kernel_size[0]
            * m.kernel_size[1]
            * m.kernel_size[2]
            // m.stride[0]
            // m.stride[1]
            // m.stride[2]
        )
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features
        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None
    return std


def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]
    if any(isinstance(m, x) for x in validclasses):
        weightinitfunc(m, gain)
        if hasattr(m, "bias"):
            m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]


def initseq(s):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain("relu"))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain("leaky_relu", b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)
    initmod(s[-1])


class Quaternion(nn.Module):
    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec**2, dim=1))
        rvec = rvec / theta[:, None]
        return torch.stack(
            (
                1.0 - 2.0 * rvec[:, 1] ** 2 - 2.0 * rvec[:, 2] ** 2,
                2.0 * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]),
                2.0 * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]),
                2.0 * (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
                1.0 - 2.0 * rvec[:, 0] ** 2 - 2.0 * rvec[:, 2] ** 2,
                2.0 * (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]),
                2.0 * (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]),
                2.0 * (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
                1.0 - 2.0 * rvec[:, 0] ** 2 - 2.0 * rvec[:, 1] ** 2,
            ),
            dim=1,
        ).view(-1, 3, 3)


# ---------------------------------------------------------------------------
# models/encoders/mvconv1.py (verbatim)
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, ninputs, tied=False):
        super(Encoder, self).__init__()

        self.ninputs = ninputs
        self.tied = tied

        self.down1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 64, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(128, 128, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(128, 256, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256, 256, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256, 256, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                )
                for i in range(1 if self.tied else self.ninputs)
            ]
        )
        self.down2 = nn.Sequential(nn.Linear(256 * self.ninputs * 4 * 3, 512), nn.LeakyReLU(0.2))
        height, width = 512, 334
        ypad = ((height + 127) // 128) * 128 - height
        xpad = ((width + 127) // 128) * 128 - width
        self.pad = nn.ZeroPad2d((xpad // 2, xpad - xpad // 2, ypad // 2, ypad - ypad // 2))
        self.mu = nn.Linear(512, 256)
        self.logstd = nn.Linear(512, 256)

        for i in range(1 if self.tied else self.ninputs):
            initseq(self.down1[i])
        initseq(self.down2)
        initmod(self.mu)
        initmod(self.logstd)

    def forward(self, x, losslist=[]):
        x = self.pad(x)
        x = [
            self.down1[0 if self.tied else i](x[:, i * 3 : (i + 1) * 3, :, :]).view(-1, 256 * 3 * 4)
            for i in range(self.ninputs)
        ]
        x = torch.cat(x, dim=1)
        x = self.down2(x)

        mu, logstd = self.mu(x) * 0.1, self.logstd(x) * 0.01
        if self.training:
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        else:
            z = mu

        losses = {}
        if "kldiv" in losslist:
            losses["kldiv"] = torch.mean(
                -0.5 - logstd + 0.5 * mu**2 + 0.5 * torch.exp(2 * logstd), dim=-1
            )

        return {"encoding": z, "losses": losses}


# ---------------------------------------------------------------------------
# models/decoders/voxel1.py (verbatim)
# ---------------------------------------------------------------------------


class ConvTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=4, templateres=128):
        super(ConvTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        self.template1 = nn.Sequential(nn.Linear(self.encodingsize, 1024), nn.LeakyReLU(0.2))
        template2 = []
        inchannels, outchannels = 1024, 512
        for i in range(int(np.log2(self.templateres)) - 1):
            template2.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
            template2.append(nn.LeakyReLU(0.2))
            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels
        template2.append(nn.ConvTranspose3d(inchannels, 4, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)

        for m in [self.template1, self.template2]:
            initseq(m)

    def forward(self, encoding):
        return self.template2(self.template1(encoding).view(-1, 1024, 1, 1, 1))


class LinearTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=4, templateres=128):
        super(LinearTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        self.template1 = nn.Sequential(
            nn.Linear(self.encodingsize, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, self.templateres**3 * self.outchannels),
        )

        for m in [self.template1]:
            initseq(m)

    def forward(self, encoding):
        return self.template1(encoding).view(
            -1, self.outchannels, self.templateres, self.templateres, self.templateres
        )


def gettemplate(templatetype, **kwargs):
    if templatetype == "conv":
        return ConvTemplate(**kwargs)
    elif templatetype == "affinemix":
        return LinearTemplate(**kwargs)
    else:
        return None


class ConvWarp(nn.Module):
    def __init__(self, displacementwarp=False, **kwargs):
        super(ConvWarp, self).__init__()

        self.displacementwarp = displacementwarp

        self.warp1 = nn.Sequential(nn.Linear(256, 1024), nn.LeakyReLU(0.2))
        self.warp2 = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(512, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(256, 3, 4, 2, 1),
        )
        for m in [self.warp1, self.warp2]:
            initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32),
            indexing="ij",
        )
        self.register_buffer(
            "grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=0)[None].astype(np.float32))
        )

    def forward(self, encoding):
        finalwarp = self.warp2(self.warp1(encoding).view(-1, 1024, 1, 1, 1)) * (2.0 / 1024)
        if not self.displacementwarp:
            finalwarp = finalwarp + self.grid
        return finalwarp


class AffineMixWarp(nn.Module):
    def __init__(self, **kwargs):
        super(AffineMixWarp, self).__init__()

        self.quat = Quaternion()

        self.warps = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 3 * 16))
        self.warpr = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 4 * 16))
        self.warpt = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 3 * 16))
        self.weightbranch = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.2), nn.Linear(64, 16 * 32 * 32 * 32)
        )
        for m in [self.warps, self.warpr, self.warpt, self.weightbranch]:
            initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32),
            indexing="ij",
        )
        self.register_buffer(
            "grid",
            torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)),
        )

    def forward(self, encoding):
        warps = self.warps(encoding).view(encoding.size(0), 16, 3)
        warpr = self.warpr(encoding).view(encoding.size(0), 16, 4)
        warpt = self.warpt(encoding).view(encoding.size(0), 16, 3) * 0.1
        warprot = self.quat(warpr.view(-1, 4)).view(encoding.size(0), 16, 3, 3)

        weight = torch.exp(self.weightbranch(encoding).view(encoding.size(0), 16, 32, 32, 32))

        warpedweight = torch.cat(
            [
                F.grid_sample(
                    weight[:, i : i + 1, :, :, :],
                    torch.sum(
                        (
                            (self.grid - warpt[:, None, None, None, i, :])[:, :, :, :, None, :]
                            * warprot[:, None, None, None, i, :, :]
                        ),
                        dim=5,
                    )
                    * warps[:, None, None, None, i, :],
                    padding_mode="border",
                )
                for i in range(weight.size(1))
            ],
            dim=1,
        )

        warp = (
            torch.sum(
                torch.stack(
                    [
                        warpedweight[:, i, :, :, :, None]
                        * (
                            torch.sum(
                                (
                                    (self.grid - warpt[:, None, None, None, i, :])[
                                        :, :, :, :, None, :
                                    ]
                                    * warprot[:, None, None, None, i, :, :]
                                ),
                                dim=5,
                            )
                            * warps[:, None, None, None, i, :]
                        )
                        for i in range(weight.size(1))
                    ],
                    dim=1,
                ),
                dim=1,
            )
            / torch.sum(warpedweight, dim=1).clamp(min=0.001)[:, :, :, :, None]
        )

        return warp.permute(0, 4, 1, 2, 3)


def getwarp(warptype, **kwargs):
    if warptype == "conv":
        return ConvWarp(**kwargs)
    elif warptype == "affinemix":
        return AffineMixWarp(**kwargs)
    else:
        return None


class Decoder(nn.Module):
    def __init__(
        self,
        templatetype="conv",
        templateres=128,
        viewconditioned=False,
        globalwarp=True,
        warptype="affinemix",
        displacementwarp=False,
    ):
        super(Decoder, self).__init__()

        self.templatetype = templatetype
        self.templateres = templateres
        self.viewconditioned = viewconditioned
        self.globalwarp = globalwarp
        self.warptype = warptype
        self.displacementwarp = displacementwarp

        if self.viewconditioned:
            self.template = gettemplate(
                self.templatetype, encodingsize=256 + 3, outchannels=3, templateres=self.templateres
            )
            self.templatealpha = gettemplate(
                self.templatetype, encodingsize=256, outchannels=1, templateres=self.templateres
            )
        else:
            self.template = gettemplate(self.templatetype, templateres=self.templateres)

        self.warp = getwarp(self.warptype, displacementwarp=self.displacementwarp)

        if self.globalwarp:
            self.quat = Quaternion()

            self.gwarps = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 3))
            self.gwarpr = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 4))
            self.gwarpt = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 3))

            for m in [self.gwarps, self.gwarpr, self.gwarpt]:
                initseq(m)

    def forward(self, encoding, viewpos, losslist=[]):
        scale = torch.tensor([25.0, 25.0, 25.0, 1.0], device=encoding.device)[
            None, :, None, None, None
        ]
        bias = torch.tensor([100.0, 100.0, 100.0, 0.0], device=encoding.device)[
            None, :, None, None, None
        ]

        viewdir = viewpos / torch.sqrt(torch.sum(viewpos**2, dim=-1, keepdim=True))
        templatein = torch.cat([encoding, viewdir], dim=1) if self.viewconditioned else encoding
        template = self.template(templatein)
        if self.viewconditioned:
            template = torch.cat([template, self.templatealpha(encoding)], dim=1)
        template = F.softplus(bias + scale * template)

        warp = self.warp(encoding) if self.warp is not None else None

        if self.globalwarp:
            gwarps = 1.0 * torch.exp(0.05 * self.gwarps(encoding).view(encoding.size(0), 3))
            gwarpr = self.gwarpr(encoding).view(encoding.size(0), 4) * 0.1
            gwarpt = self.gwarpt(encoding).view(encoding.size(0), 3) * 0.025
            gwarprot = self.quat(gwarpr.view(-1, 4)).view(encoding.size(0), 3, 3)

        losses = {}

        if "tvl1" in losslist:
            logalpha = torch.log(1e-5 + template[:, -1, :, :, :])
            losses["tvl1"] = torch.mean(
                torch.sqrt(
                    1e-5
                    + (logalpha[:, :-1, :-1, 1:] - logalpha[:, :-1, :-1, :-1]) ** 2
                    + (logalpha[:, :-1, 1:, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2
                    + (logalpha[:, 1:, :-1, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2
                )
            )

        return {
            "template": template,
            "warp": warp,
            **(
                {"gwarps": gwarps, "gwarprot": gwarprot, "gwarpt": gwarpt}
                if self.globalwarp
                else {}
            ),
            "losses": losses,
        }


# ---------------------------------------------------------------------------
# models/volsamplers/warpvoxel.py (verbatim)
# ---------------------------------------------------------------------------


class VolSampler(nn.Module):
    def __init__(self, displacementwarp=False):
        super(VolSampler, self).__init__()

        self.displacementwarp = displacementwarp

    def forward(
        self,
        pos,
        template,
        warp=None,
        gwarps=None,
        gwarprot=None,
        gwarpt=None,
        viewtemplate=False,
        **kwargs,
    ):
        valid = None
        if not viewtemplate:
            if gwarps is not None:
                pos = (
                    torch.sum(
                        (pos - gwarpt[:, None, None, None, :])[:, :, :, :, None, :]
                        * gwarprot[:, None, None, None, :, :],
                        dim=-1,
                    )
                    * gwarps[:, None, None, None, :]
                )
            if warp is not None:
                if self.displacementwarp:
                    pos = pos + F.grid_sample(warp, pos).permute(0, 2, 3, 4, 1)
                else:
                    valid = torch.prod((pos > -1.0) * (pos < 1.0), dim=-1).float()
                    pos = F.grid_sample(warp, pos).permute(0, 2, 3, 4, 1)
        val = F.grid_sample(template, pos)
        if valid is not None:
            val = val * valid[:, None, :, :, :]
        return val[:, :3, :, :, :], val[:, 3:, :, :, :]


# ---------------------------------------------------------------------------
# models/colorcals/colorcal1.py (verbatim)
# ---------------------------------------------------------------------------


class Colorcal(nn.Module):
    """Apply learnable 3 channel scale and bias to an image to handle un(color)calibrated cameras."""

    def __init__(self, allcameras):
        super(Colorcal, self).__init__()

        self.allcameras = allcameras

        self.conv = nn.ModuleDict({k: nn.Conv2d(3, 3, 1, 1, 0, groups=3) for k in self.allcameras})

        for k in self.allcameras:
            self.conv[k].weight.data[:] = 1.0
            self.conv[k].bias.data.zero_()

    def forward(self, image, camindex):
        return torch.cat(
            [
                self.conv[self.allcameras[camindex[i].item()]](image[i : i + 1, :, :, :])
                for i in range(image.size(0))
            ]
        )


# ---------------------------------------------------------------------------
# models/neurvol1.py (verbatim, top-level Autoencoder)
# ---------------------------------------------------------------------------


class Autoencoder(nn.Module):
    def __init__(
        self, dataset, encoder, decoder, volsampler, colorcal, dt, stepjitter=0.01, estimatebg=False
    ):
        super(Autoencoder, self).__init__()

        self.estimatebg = estimatebg
        self.allcameras = dataset.get_allcameras()

        self.encoder = encoder
        self.decoder = decoder
        self.volsampler = volsampler
        self.bg = nn.ParameterDict(
            {
                k: nn.Parameter(torch.ones(3, v["size"][1], v["size"][0]), requires_grad=estimatebg)
                for k, v in dataset.get_krt().items()
            }
        )
        self.colorcal = colorcal
        self.dt = dt
        self.stepjitter = stepjitter

        self.imagemean = dataset.imagemean
        self.imagestd = dataset.imagestd

        if dataset.known_background():
            dataset.get_background(self.bg)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        ret = super(Autoencoder, self).state_dict(destination, prefix, keep_vars)
        if not self.estimatebg:
            for k in self.bg.keys():
                del ret[prefix + "bg." + k]
        return ret

    def forward(
        self,
        iternum,
        losslist,
        camrot,
        campos,
        focal,
        princpt,
        pixelcoords,
        validinput,
        fixedcamimage=None,
        encoding=None,
        keypoints=None,
        camindex=None,
        image=None,
        imagevalid=None,
        viewtemplate=False,
        outputlist=[],
    ):
        result = {"losses": {}}

        if encoding is None:
            encout = self.encoder(fixedcamimage, losslist)
            encoding = encout["encoding"]
            result["losses"].update(encout["losses"])

        decout = self.decoder(encoding, campos, losslist)
        result["losses"].update(decout["losses"])

        raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]
        raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
        raydir = torch.sum(camrot[:, None, None, :, :] * raydir[:, :, :, :, None], dim=-2)
        raydir = raydir / torch.sqrt(torch.sum(raydir**2, dim=-1, keepdim=True))

        with torch.no_grad():
            t1 = (-1.0 - campos[:, None, None, :]) / raydir
            t2 = (1.0 - campos[:, None, None, :]) / raydir
            tmin = torch.max(
                torch.min(t1[..., 0], t2[..., 0]),
                torch.max(torch.min(t1[..., 1], t2[..., 1]), torch.min(t1[..., 2], t2[..., 2])),
            )
            tmax = torch.min(
                torch.max(t1[..., 0], t2[..., 0]),
                torch.min(torch.max(t1[..., 1], t2[..., 1]), torch.max(t1[..., 2], t2[..., 2])),
            )

            intersections = tmin < tmax
            t = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.0)
            tmin = torch.where(intersections, tmin, torch.zeros_like(tmin))
            tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

        t = t - self.dt * torch.rand_like(t)

        raypos = campos[:, None, None, :] + raydir * t[..., None]
        rayrgb = torch.zeros_like(raypos.permute(0, 3, 1, 2))
        rayalpha = torch.zeros_like(rayrgb[:, 0:1, :, :])

        done = torch.zeros_like(t).bool()
        while not done.all():
            valid = torch.prod(torch.gt(raypos, -1.0) * torch.lt(raypos, 1.0), dim=-1).byte()
            validf = valid.float()

            sample_rgb, sample_alpha = self.volsampler(
                raypos[:, None, :, :, :], **decout, viewtemplate=viewtemplate
            )

            with torch.no_grad():
                step = self.dt * torch.exp(self.stepjitter * torch.randn_like(t))
                done = done | ((t + step) >= tmax)

            contrib = (
                (rayalpha + sample_alpha[:, :, 0, :, :] * step[:, None, :, :]).clamp(max=1.0)
                - rayalpha
            ) * validf[:, None, :, :]

            rayrgb = rayrgb + sample_rgb[:, :, 0, :, :] * contrib
            rayalpha = rayalpha + contrib

            raypos = raypos + raydir * step[:, :, :, None]
            t = t + step

        if image is not None:
            imagesize = torch.tensor(
                image.size()[3:1:-1], dtype=torch.float32, device=pixelcoords.device
            )
            samplecoords = pixelcoords * 2.0 / (imagesize[None, None, None, :] - 1.0) - 1.0

        if camindex is not None:
            rayrgb = self.colorcal(rayrgb, camindex)

            if pixelcoords.size()[1:3] != image.size()[2:4]:
                bg = F.grid_sample(
                    torch.stack(
                        [
                            self.bg[self.allcameras[camindex[i].item()]]
                            for i in range(campos.size(0))
                        ],
                        dim=0,
                    ),
                    samplecoords,
                )
            else:
                bg = torch.stack(
                    [self.bg[self.allcameras[camindex[i].item()]] for i in range(campos.size(0))],
                    dim=0,
                )

            rayrgb = rayrgb + (1.0 - rayalpha) * bg.clamp(min=0.0)

        if "irgbrec" in outputlist:
            result["irgbrec"] = rayrgb
        if "ialpharec" in outputlist:
            result["ialpharec"] = rayalpha

        if "alphapr" in losslist:
            alphaprior = torch.mean(
                torch.log(0.1 + rayalpha.view(rayalpha.size(0), -1))
                + torch.log(0.1 + 1.0 - rayalpha.view(rayalpha.size(0), -1))
                - -2.20727,
                dim=-1,
            )
            result["losses"]["alphapr"] = alphaprior

        if image is not None:
            if pixelcoords.size()[1:3] != image.size()[2:4]:
                image = F.grid_sample(image, samplecoords)

            rayrgb = (rayrgb - self.imagemean) / self.imagestd
            image = (image - self.imagemean) / self.imagestd

            if imagevalid is not None:
                weight = (
                    imagevalid[:, None, None, None].expand_as(image)
                    * validinput[:, None, None, None]
                )
            else:
                weight = torch.ones_like(image) * validinput[:, None, None, None]

            irgbsqerr = weight * (image - rayrgb) ** 2

            if "irgbsqerr" in outputlist:
                result["irgbsqerr"] = irgbsqerr

            if "irgbmse" in losslist:
                irgbmse = torch.sum(irgbsqerr.view(irgbsqerr.size(0), -1), dim=-1)
                irgbmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

                result["losses"]["irgbmse"] = (irgbmse, irgbmse_weight)

        return result


# ---------------------------------------------------------------------------
# Staging build/example helpers (plumbing only; not part of the network).
# ---------------------------------------------------------------------------


class _TinyCameraSet:
    """Minimal duck-typed stand-in for the real `data/dryice1.Dataset`.

    The real `Dataset` loads KRT camera-calibration files and JPEG frames from a
    dataset directory on disk (`experiments/dryice1/data/...`), which does not
    exist in this environment. `Autoencoder.__init__` only calls
    `get_allcameras()`, `get_krt()`, `known_background()`, and reads
    `.imagemean`/`.imagestd`. This stand-in supplies those 4 accessors with
    synthetic calibration for a single tiny camera; it introduces no network
    layers of its own.
    """

    def __init__(self):
        self._camera = "cam0"
        self.imagemean = 0.0
        self.imagestd = 1.0

    def get_allcameras(self):
        return [self._camera]

    def get_krt(self):
        return {self._camera: {"size": np.array([8, 8])}}

    def known_background(self):
        return False

    def get_background(self, bg):  # pragma: no cover - not called (known_background() is False)
        return None


def build_neuralvolumes() -> nn.Module:
    """Build a tiny Neural Volumes Autoencoder (small template res, single camera)."""

    torch.manual_seed(0)
    dataset = _TinyCameraSet()
    templateres = 8

    encoder = Encoder(ninputs=1)
    decoder = Decoder(
        templatetype="affinemix",
        templateres=templateres,
        viewconditioned=False,
        globalwarp=True,
        warptype="affinemix",
        displacementwarp=False,
    )
    volsampler = VolSampler()
    colorcal = Colorcal(dataset.get_allcameras())

    model = Autoencoder(dataset, encoder, decoder, volsampler, colorcal, dt=1.0, estimatebg=False)
    return model.eval()


def example_input_neuralvolumes():
    """Return a forward-pass input tuple (iternum, losslist, camera params, 2x2 pixel grid).

    The real `Encoder` (models/encoders/mvconv1.py) hardcodes 7 stride-2 conv layers sized
    for a 512x334 input (zero-padded to 512x384, giving a 4x3 spatial bottleneck that the
    `down2` linear layer's `256*ninputs*4*3` in_features is fixed to) -- this sizing is
    architecture, not a tunable hyperparameter, so the real 512x334 image size is kept
    verbatim rather than shrunk.
    """

    torch.manual_seed(0)
    batch = 1
    fixedcamimage = torch.randn(batch, 3, 512, 334)
    camrot = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    campos = torch.tensor([[0.0, 0.0, -3.0]])
    focal = torch.tensor([[4.0, 4.0]])
    princpt = torch.tensor([[1.0, 1.0]])
    py, px = torch.meshgrid(
        torch.arange(2, dtype=torch.float32), torch.arange(2, dtype=torch.float32), indexing="ij"
    )
    pixelcoords = torch.stack((px, py), dim=-1).unsqueeze(0)
    validinput = torch.ones(batch)

    return (
        0,
        [],
        camrot,
        campos,
        focal,
        princpt,
        pixelcoords,
        validinput,
        fixedcamimage,
    )


MENAGERIE_ENTRIES = [
    (
        "NeuralVolumes",
        "build_neuralvolumes",
        "example_input_neuralvolumes",
        2019,
        "vendored-pytorch",
    ),
]
