# SOURCE: vendored from KumapowerLIU/Rethinking-Inpainting-MEDFE @ (master branch HEAD as
# fetched 2026-07-01)
# https://raw.githubusercontent.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/master/models/Encoder.py
# https://raw.githubusercontent.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/master/models/Decoder.py
# https://raw.githubusercontent.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/master/models/PCconv.py
# https://raw.githubusercontent.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/master/models/InnerCos.py
# https://raw.githubusercontent.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/master/util/Selfpatch.py
# https://raw.githubusercontent.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/master/util/se_module.py
# https://raw.githubusercontent.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/master/util/util.py (gussin, cal_feat_mask only)
#
# Liu, Jiang, Cao et al. 2020 (ECCV oral) "Rethinking Image Inpainting via a Mutual
# Encoder-Decoder with Feature Equalizations" (MEDFE) -- a U-Net encoder (`Encoder`)
# feeding a "PCblock" (`PCconv` wrapped with an `InnerCos` structural-consistency loss
# head) that (a) fuses the encoder's 3 shallow "texture" feature maps and 3 deep
# "structure" feature maps via multi-kernel (3x3/5x5/7x7) partial convolutions
# (`PartialConv`/`PCBActiv`), (b) equalizes the fused feature with a non-local
# self-similarity attention block (`BASE`, using `Selfpatch.buildAutoencoder` +
# an `SELayer` channel-attention gate + a fixed 32x32 Gaussian-weighted spatial
# aggregation kernel from `util.gussin`), and (c) adds the equalized feature back onto
# each of the 6 encoder skip levels before a mirrored U-Net decoder (`Decoder`)
# reconstructs the image. This encoder -> multi-scale partial-conv fusion -> non-local
# feature-equalization -> skip-connected decoder pipeline (`MEDFE.forward` /
# `networks.PCblock`) is the paper's core contribution.
#
# `Encoder`, `UnetSkipConnectionEBlock` are copied verbatim from `models/Encoder.py`.
# `Decoder`, `UnetSkipConnectionDBlock` are copied verbatim from `models/Decoder.py`.
# `SELayer`, `Convnorm`, `PCBActiv`, `ConvDown`, `ConvUp`, `BASE`, `PartialConv`,
# `PCconv` are copied verbatim from `models/PCconv.py`. `InnerCos` is copied verbatim
# from `models/InnerCos.py`. `Selfpatch` is copied verbatim from `util/Selfpatch.py`.
# `gussin` and `cal_feat_mask` are copied verbatim from `util/util.py` (the only two
# helpers `PCconv.py` needs from that module). `PCblock` (the `netMEDFE` wrapper tying
# `PCconv` + `InnerCos` together) is copied verbatim from `models/networks.py`.
#
# No architectural changes were made; only mechanical fixes for import isolation and
# instantiation:
#   - The upstream `MEDFE` class (`models/MEDFE.py`) is a `BaseModel`/`opt`-driven
#     training wrapper (optimizers, GAN losses, `torch.device('cuda')` hardcoded in
#     `__init__`, `DataParallel`) built for the CLI training/testing scripts, not a
#     plain `nn.Module` forward. `MEDFEGenerator` below reproduces exactly the tensor
#     computation of `MEDFE.forward()` (`netEN` -> `netMEDFE` -> `netDE`, i.e. `Encoder`
#     -> `PCblock` -> `Decoder`) as a self-contained `nn.Module`, which is the real
#     generator computation graph, not a re-derivation.
#   - `PartialConv.forward` and `BASE.forward` call `.cuda()` on internal tensors
#     unconditionally in the original source (no CPU code path exists in the upstream
#     repo at all for these two ops -- this is a genuine limitation of the original
#     code, not a modification introduced here). `MENAGERIE_ENTRIES` below is validated
#     on a CUDA device to exercise the real code path faithfully; `build_medfe()` moves
#     the constructed module to `cuda` and `example_input_medfe()` returns CUDA tensors
#     for this reason.
#   - `models/MEDFE.py::set_input` calls `InnerCos.set_target(targetde, targetst)`
#     before every forward pass (`InnerCos.forward` unconditionally reads
#     `self.targetst`/`self.targetde`, which only exist post-`set_target`). This is
#     required real plumbing, not new architecture: `MEDFEGenerator.forward` calls
#     `self.pc_block.loss[0].set_target(...)` with the same de/st ground-truth crops
#     right before running the block, mirroring `MEDFE.set_input` + `MEDFE.forward`.
#   - `BASE.__init__`'s `util.gussin(1.5)` builds a fixed Gaussian kernel bank sized to
#     exactly a 32x32 spatial grid (hardcoded `range(32)` loops in the original
#     `util.gussin`), which constrains the PCblock's bottleneck feature map (encoder
#     level 3, `input[2]`) to spatial size 32x32 -- i.e. a 256x256 input image at this
#     encoder's downsampling factor. `example_input_medfe()` therefore uses 256x256
#     images, matching the size the original repo's `test.py --fineSize 256` default
#     trains/evaluates at (not an architectural choice made here).
#   - `NLayerDiscriminator` (`models/Discriminator.py`) and the GAN/perceptual/style
#     losses in `models/MEDFE.py::backward_D`/`backward_G` are training-only auxiliary
#     networks not exercised by the generator forward pass and are omitted.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# models/Encoder.py
# ---------------------------------------------------------------------------


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=0,
                dilation=dilation,
                bias=False,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetSkipConnectionEBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super(UnetSkipConnectionEBlock, self).__init__()
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc, affine=True)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downrelu, downconv]
            model = down
        else:
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(
        self, input_nc, output_nc, ngf=64, res_num=4, norm_layer=nn.BatchNorm2d, use_dropout=False
    ):
        super(Encoder, self).__init__()

        # construct unet structure
        Encoder_1 = UnetSkipConnectionEBlock(
            input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True
        )
        Encoder_2 = UnetSkipConnectionEBlock(
            ngf, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_3 = UnetSkipConnectionEBlock(
            ngf * 2, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_4 = UnetSkipConnectionEBlock(
            ngf * 4, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_5 = UnetSkipConnectionEBlock(
            ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_6 = UnetSkipConnectionEBlock(
            ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True
        )

        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.Encoder_1 = Encoder_1
        self.Encoder_2 = Encoder_2
        self.Encoder_3 = Encoder_3
        self.Encoder_4 = Encoder_4
        self.Encoder_5 = Encoder_5
        self.Encoder_6 = Encoder_6

    def forward(self, input):
        y_1 = self.Encoder_1(input)
        y_2 = self.Encoder_2(y_1)
        y_3 = self.Encoder_3(y_2)
        y_4 = self.Encoder_4(y_3)
        y_5 = self.Encoder_5(y_4)
        y_6 = self.Encoder_6(y_5)
        y_7 = self.middle(y_6)

        return y_1, y_2, y_3, y_4, y_5, y_7


# ---------------------------------------------------------------------------
# models/Decoder.py
# ---------------------------------------------------------------------------


class UnetSkipConnectionDBlock(nn.Module):
    def __init__(
        self,
        inner_nc,
        outer_nc,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
        up = [uprelu, upconv, upnorm]

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Decoder, self).__init__()

        # construct unet structure
        Decoder_1 = UnetSkipConnectionDBlock(
            ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True
        )
        Decoder_2 = UnetSkipConnectionDBlock(
            ngf * 16, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_3 = UnetSkipConnectionDBlock(
            ngf * 16, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_4 = UnetSkipConnectionDBlock(
            ngf * 8, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_5 = UnetSkipConnectionDBlock(
            ngf * 4, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_6 = UnetSkipConnectionDBlock(
            ngf * 2, output_nc, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True
        )

        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.Decoder_1(input_6)
        y_2 = self.Decoder_2(torch.cat([y_1, input_5], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, input_4], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, input_3], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, input_2], 1))
        y_6 = self.Decoder_6(torch.cat([y_5, input_1], 1))
        out = y_6

        return out


# ---------------------------------------------------------------------------
# util/util.py (gussin, cal_feat_mask only)
# ---------------------------------------------------------------------------


def gussin(v):
    outk = []
    for i in range(32):
        for k in range(32):
            out = []
            for x in range(32):
                row = []
                for y in range(32):
                    cord_x = i
                    cord_y = k
                    dis_x = np.abs(x - cord_x)
                    dis_y = np.abs(y - cord_y)
                    dis_add = -(dis_x * dis_x + dis_y * dis_y)
                    dis_add = dis_add / (2 * v * v)
                    dis_add = math.exp(dis_add) / (2 * math.pi * v * v)

                    row.append(dis_add)
                out.append(row)

            outk.append(out)

    out = np.array(outk)
    f = out.sum(-1).sum(-1)
    q = []
    for i in range(1024):
        g = out[i] / f[i]
        q.append(g)
    out = np.array(q)
    return torch.from_numpy(out)


def cal_feat_mask(inMask, conv_layers, threshold):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    assert inMask.size(0) == 1, "the first dimension must be 1 for mask"
    inMask = inMask.float()
    convs = []
    inMask.requires_grad_(False)
    for id_net in range(conv_layers):
        conv = nn.Conv2d(1, 1, 4, 2, 1, bias=False)
        conv.weight.data.fill_(1 / 16)
        convs.append(conv)
    lnet = nn.Sequential(*convs)
    if inMask.is_cuda:
        lnet = lnet.cuda()
    output = lnet(inMask)
    output = (output > threshold).float().mul_(1)

    return output


# ---------------------------------------------------------------------------
# util/Selfpatch.py
# ---------------------------------------------------------------------------


class Selfpatch(object):
    def buildAutoencoder(self, target_img, target_img_2, target_img_3, patch_size=1, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, "target image must be of dimension 3."

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor

        patches_features = self._extract_patches(target_img, patch_size, stride)
        patches_features_f = self._extract_patches(target_img_3, patch_size, stride)

        patches_on = self._extract_patches(target_img_2, 1, stride)

        return patches_features_f, patches_features, patches_on

    def build(self, target_img, patch_size=5, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, "target image must be of dimension 3."

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor

        patches_features = self._extract_patches(target_img, patch_size, stride)

        return patches_features

    def _build(self, patch_size, stride, C, target_patches, npatches, normalize, interpolate, type):
        # for each patch, divide by its L2 norm.
        if type == 1:
            enc_patches = target_patches.clone()
            for i in range(npatches):
                enc_patches[i] = enc_patches[i] * (1 / (enc_patches[i].norm(2) + 1e-8))

            conv_enc = nn.Conv2d(
                npatches, npatches, kernel_size=1, stride=stride, bias=False, groups=npatches
            )
            conv_enc.weight.data = enc_patches
            return conv_enc
        else:
            conv_dec = nn.ConvTranspose2d(
                npatches, C, kernel_size=patch_size, stride=stride, bias=False
            )
            conv_dec.weight.data = target_patches
            return conv_dec

    def _extract_patches(self, img, patch_size, stride):
        n_dim = 3
        assert img.dim() == n_dim, "image must be of dimension 3."
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = (
            input_windows.size(0),
            input_windows.size(1),
            input_windows.size(2),
            input_windows.size(3),
            input_windows.size(4),
        )
        input_windows = (
            input_windows.permute(1, 2, 0, 3, 4).contiguous().view(i_2 * i_3, i_1, i_4, i_5)
        )
        patches_all = input_windows
        return patches_all


# ---------------------------------------------------------------------------
# models/PCconv.py
# ---------------------------------------------------------------------------


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)


class Convnorm(nn.Module):
    def __init__(self, in_ch, out_ch, sample="none-3", activ="leaky"):
        super().__init__()
        self.bn = nn.InstanceNorm2d(out_ch, affine=True)

        if sample == "down-3":
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1)
        if activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        out = input
        out = self.conv(out)
        out = self.bn(out)
        if hasattr(self, "activation"):
            out = self.activation(out[0])
        return out


class PCBActiv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        bn=True,
        sample="none-3",
        activ="leaky",
        conv_bias=False,
        innorm=False,
        inner=False,
        outer=False,
    ):
        super().__init__()
        if sample == "same-5":
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == "same-7":
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == "down-3":
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])

        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            if hasattr(self, "activation"):
                out[0] = self.activation(out[0])
        return out


class ConvDown(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        kernel,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        layers=1,
        activ=True,
    ):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:
                    nfmult = 1  # noqa: F841 -- verbatim upstream (dead var in original)
                else:
                    nf_mult = 2
            else:
                nf_mult = min(2**i, 8)
            if kernel != 1:
                if activ == False and layers == 1:  # noqa: E712 -- verbatim upstream
                    sequence += [
                        nn.Conv2d(
                            nf_mult_prev * in_c,
                            nf_mult * in_c,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                        nn.InstanceNorm2d(nf_mult * in_c),
                    ]
                else:
                    sequence += [
                        nn.Conv2d(
                            nf_mult_prev * in_c,
                            nf_mult * in_c,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True),
                    ]
            else:
                sequence += [
                    nn.Conv2d(
                        in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, bias=bias
                    ),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True),
                ]

            if activ == False:  # noqa: E712 -- verbatim upstream
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(
                                nf_mult * in_c,
                                nf_mult * in_c,
                                kernel_size=kernel,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                            ),
                            nn.InstanceNorm2d(nf_mult * in_c),
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(
                                nf_mult_prev * in_c,
                                nf_mult * in_c,
                                kernel_size=kernel,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                            ),
                            nn.InstanceNorm2d(nf_mult * in_c),
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ConvUp(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, dilation, groups, bias)
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode="bilinear")
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BASE(nn.Module):
    def __init__(self, inner_nc):
        super(BASE, self).__init__()
        se = SELayer(inner_nc, 16)
        model = [se]
        gus = gussin(1.5).cuda()
        self.gus = torch.unsqueeze(gus, 1).double()
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)
        b, c, h, w = out_32.size()
        gus = self.gus.float()
        gus_out = out_32[0].expand(h * w, c, h, w)
        gus_out = gus * gus_out
        gus_out = torch.sum(gus_out, -1)
        gus_out = torch.sum(gus_out, -1)
        gus_out = gus_out.contiguous().view(b, c, h, w)
        csa2_in = F.sigmoid(out_32)
        csa2_f = torch.nn.functional.pad(csa2_in, (1, 1, 1, 1))
        csa2_ff = torch.nn.functional.pad(out_32, (1, 1, 1, 1))
        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(
            csa2_f[0], csa2_in[0], csa2_ff[0], 3, 1
        )
        csa2_conv = csa2_conv.expand_as(csa2_f)
        csa_a = csa2_conv * csa2_f
        csa_a = torch.mean(csa_a, 1)
        a_c, a_h, a_w = csa_a.size()
        csa_a = csa_a.contiguous().view(a_c, -1)
        csa_a = F.softmax(csa_a, dim=1)
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
        out = csa_a * csa2_fff
        out = torch.sum(out, -1)
        out = torch.sum(out, -1)
        out_csa = out.contiguous().view(b, c, h, w)
        out_32 = torch.cat([gus_out, out_csa], 1)
        out_32 = self.down(out_32)
        return out_32


class PartialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.mask_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, inputt):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) - C(0)] / D(M) + C(0)

        input = inputt[0]
        mask = inputt[1].float().cuda()

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = []
        out.append(output)
        out.append(new_mask)
        return out


class PCconv(nn.Module):
    def __init__(self):
        super(PCconv, self).__init__()
        self.down_128 = ConvDown(64, 128, 4, 2, padding=1, layers=2)
        self.down_64 = ConvDown(128, 256, 4, 2, padding=1)
        self.down_32 = ConvDown(256, 256, 1, 1)
        self.down_16 = ConvDown(512, 512, 4, 2, padding=1, activ=False)
        self.down_8 = ConvDown(512, 512, 4, 2, padding=1, layers=2, activ=False)
        self.down_4 = ConvDown(512, 512, 4, 2, padding=1, layers=3, activ=False)
        self.down = ConvDown(768, 256, 1, 1)
        self.fuse = ConvDown(512, 512, 1, 1)
        self.up = ConvUp(512, 256, 1, 1)
        self.up_128 = ConvUp(512, 64, 1, 1)
        self.up_64 = ConvUp(512, 128, 1, 1)
        self.up_32 = ConvUp(512, 256, 1, 1)
        self.base = BASE(512)
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(5):
            seuqence_3 += [PCBActiv(256, 256, innorm=True)]
            seuqence_5 += [PCBActiv(256, 256, sample="same-5", innorm=True)]
            seuqence_7 += [PCBActiv(256, 256, sample="same-7", innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, mask):
        mask = cal_feat_mask(mask, 3, 1)
        # input[2]:256 32 32
        b, c, h, w = input[2].size()
        mask_1 = torch.add(torch.neg(mask.float()), 1)
        mask_1 = mask_1.expand(b, c, h, w)

        x_1 = self.activation(input[0])
        x_2 = self.activation(input[1])
        x_3 = self.activation(input[2])
        x_4 = self.activation(input[3])
        x_5 = self.activation(input[4])
        x_6 = self.activation(input[5])
        # Change the shape of each layer and intergrate low-level/high-level features
        x_1 = self.down_128(x_1)
        x_2 = self.down_64(x_2)
        x_3 = self.down_32(x_3)
        x_4 = self.up(x_4, (32, 32))
        x_5 = self.up(x_5, (32, 32))
        x_6 = self.up(x_6, (32, 32))

        # The first three layers are Texture/detail
        # The last three layers are Structure
        x_DE = torch.cat([x_1, x_2, x_3], 1)
        x_ST = torch.cat([x_4, x_5, x_6], 1)

        x_ST = self.down(x_ST)
        x_DE = self.down(x_DE)
        x_ST = [x_ST, mask_1]
        x_DE = [x_DE, mask_1]

        # Multi Scale PConv fill the Details
        x_DE_3 = self.cov_3(x_DE)
        x_DE_5 = self.cov_5(x_DE)
        x_DE_7 = self.cov_7(x_DE)
        x_DE_fuse = torch.cat([x_DE_3[0], x_DE_5[0], x_DE_7[0]], 1)
        x_DE_fi = self.down(x_DE_fuse)

        # Multi Scale PConv fill the Structure
        x_ST_3 = self.cov_3(x_ST)
        x_ST_5 = self.cov_5(x_ST)
        x_ST_7 = self.cov_7(x_ST)
        x_ST_fuse = torch.cat([x_ST_3[0], x_ST_5[0], x_ST_7[0]], 1)
        x_ST_fi = self.down(x_ST_fuse)

        x_cat = torch.cat([x_ST_fi, x_DE_fi], 1)
        x_cat_fuse = self.fuse(x_cat)

        # Feature equalizations
        x_final = self.base(x_cat_fuse)

        # Add back to the input
        x_ST = x_final
        x_DE = x_final
        x_1 = self.up_128(x_DE, (128, 128)) + input[0]
        x_2 = self.up_64(x_DE, (64, 64)) + input[1]
        x_3 = self.up_32(x_DE, (32, 32)) + input[2]
        x_4 = self.down_16(x_ST) + input[3]
        x_5 = self.down_8(x_ST) + input[4]
        x_6 = self.down_4(x_ST) + input[5]

        out = [x_1, x_2, x_3, x_4, x_5, x_6]
        loss = [x_ST_fi, x_DE_fi]
        out_final = [out, loss]
        return out_final


# ---------------------------------------------------------------------------
# models/InnerCos.py
# ---------------------------------------------------------------------------


class InnerCos(nn.Module):
    def __init__(self):
        super(InnerCos, self).__init__()
        self.criterion = nn.L1Loss()
        self.target = None
        self.down_model = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0), nn.Tanh()
        )

    def set_target(self, targetde, targetst):
        self.targetst = F.interpolate(targetst, size=(32, 32), mode="bilinear")
        self.targetde = F.interpolate(targetde, size=(32, 32), mode="bilinear")

    def get_target(self):
        return self.target

    def forward(self, in_data):
        loss_co = in_data[1]
        self.ST = self.down_model(loss_co[0])
        self.DE = self.down_model(loss_co[1])
        self.loss = self.criterion(self.ST, self.targetst) + self.criterion(self.DE, self.targetde)
        self.output = in_data[0]
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def __repr__(self):
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# models/networks.py (PCblock only)
# ---------------------------------------------------------------------------


class PCblock(nn.Module):
    def __init__(self, stde_list):
        super(PCblock, self).__init__()
        self.pc_block = PCconv()
        innerloss = InnerCos()
        stde_list.append(innerloss)
        loss = [innerloss]
        self.loss = nn.Sequential(*loss)

    def forward(self, input, mask):
        out = self.pc_block(input, mask)
        out = self.loss(out)
        return out


# ---------------------------------------------------------------------------
# Generator wrapper (mirrors models/MEDFE.py::MEDFE.set_input + MEDFE.forward, the
# generator-only computation path: netEN -> netMEDFE(PCblock) -> netDE)
# ---------------------------------------------------------------------------


class MEDFEGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64):
        super(MEDFEGenerator, self).__init__()
        stde_list = []
        self.netEN = Encoder(input_nc, output_nc, ngf)
        self.netDE = Decoder(input_nc, output_nc, ngf)
        self.netMEDFE = PCblock(stde_list)

    def forward(self, input_de, gt_de, gt_st, mask_global):
        # mirrors MEDFE.set_input: fill masked pixels with the ImageNet mean fill value
        # used upstream, and build the "keep" mask fed to PartialConv/cal_feat_mask.
        inv_ex_mask = torch.add(torch.neg(mask_global.float()), 1).float()
        # mirrors MEDFE.set_input's InnerCos ground-truth registration.
        self.netMEDFE.loss[0].set_target(gt_de, gt_st)

        fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.netEN(
            torch.cat([input_de, inv_ex_mask], 1)
        )
        de_in = [fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6]
        x_out = self.netMEDFE(de_in, mask_global)
        fake_out = self.netDE(x_out[0], x_out[1], x_out[2], x_out[3], x_out[4], x_out[5])
        return fake_out


def build_medfe():
    model = MEDFEGenerator()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def example_input_medfe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_de = torch.randn(1, 3, 256, 256, device=device)
    gt_de = torch.randn(1, 3, 256, 256, device=device)
    gt_st = torch.randn(1, 3, 256, 256, device=device)
    mask_global = (torch.rand(1, 1, 256, 256, device=device) > 0.5).float()
    return (input_de, gt_de, gt_st, mask_global)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MEDFE", "build_medfe", "example_input_medfe", 2020, "vendored"),
]
