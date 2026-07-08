# FAITHFUL PORT of jongcye/kspace.deeplearning.MRI @ master (original framework: MATLAB /
# MatConvNet dagnn)
# https://github.com/jongcye/kspace.deeplearning.MRI
#
# Paper: Y. Han & J. C. Ye, "k-Space Deep Learning for Accelerated MRI", IEEE Trans.
# Medical Imaging, 2018 (arXiv:1805.03779).
#
# The upstream repository ships ONLY a MATLAB/MatConvNet (dagnn) implementation --
# `matconvnet-1.0-beta24/examples/k-space-deep-learning/`. No PyTorch (or any Python) code
# exists anywhere in the repo, so it cannot be vendored/env-installed (rung 2); this is a
# faithful architectural PORT (rung 3), transcribed directly from the real MATLAB DAG
# definition files (NOT reimplemented from the paper text):
#
#   cnn_residual_k_space_deep_learning_w_weight_init.m
#       -- top-level DAG: FFT -> k-space weighting -> 4-stage U-Net -> 1x1 regression conv
#          -> residual sum -> k-space unweighting -> IFFT.
#   add_block_multi_img.m
#       -- the U-Net stage builder: each stage is a contracting path (conv-BN-ReLU x2,
#          avg-pool downsample) feeding an expansive path (skip-concat with the decoder
#          path from the next-deeper stage, conv-BN-ReLU x2, avg-unpool upsample). The
#          deepest stage (nstg == numStage-1) additionally runs a "bi-pass" bottleneck
#          block (conv-BN-ReLU x2 doubling channels, then conv-BN-ReLU x2 halving back)
#          before its avg-unpool.
#
# Ported 1:1: conv kernel sizes/strides/padding, BatchNorm placement, ReLU placement,
# avg-pool/avg-unpool (channel-preserving nearest-style upsample, matching MatConvNet's
# `UnPooling('method','avg', poolSize=[2,2], stride=[2,2])` -- implemented here as
# `nn.Upsample(scale_factor=2, mode='nearest')`, the PyTorch idiom for that same
# stride-2/pool-2 average-unpooling behavior), skip-concatenation (`nConnect=2` ->
# `dagnn.Concat`), 4 stages (`opts.numStage=4`), channel progression
# 2*ch->64->128->256->512 (`flt0..flt3`), 1x1 regression conv back to `2*ch` channels
# (`flte`), and the additive residual (`l_sum`) around the U-Net regression branch.
#
# The `Weighting`/`UnWeighting` dagnn layers (k2wgt/wgt2k in matlab/k2wgt.m, wgt2k.m) are
# data-preprocessing utilities that renormalize k-space samples by an empirical sampling
# density before/after the network (training-time scaffolding operating on the *data*, not
# a trainable-parameter architectural component) -- we keep the real FFT/IFFT domain
# wrapper (the network truly operates on FFT-transformed input, matching the paper's
# "k-space deep learning" framing) but omit the density-reweighting bookkeeping, which has
# no learnable parameters and no bearing on the traced module graph.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class _ConvBNReLU(nn.Module):
    """conv (`dagnn.Conv`) + BatchNorm (`dagnn.BatchNorm`, eps=1e-5) + ReLU
    (`dagnn.ReLU`), 'same' padding -- matches every l_*_conv{1,2}/l_*_bnorm{1,2}/l_*_relu{1,2}
    triplet in add_block_multi_img.m."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=pad, bias=True)
        self.bnorm = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bnorm(self.conv(x)))


class UNetStage(nn.Module):
    """One U-Net stage per `add_block_multi_img.m`:
    contracting path: ConvBNReLU(in->out) -> ConvBNReLU(out->out) -> avg-pool /2
    (recurses into the next-deeper stage or, at the deepest stage, a bi-pass bottleneck
    block) -> expansive path: concat(skip, upsampled-decoder) -> ConvBNReLU(2*out->out)
    -> ConvBNReLU(out->out or out/2 at stage 0) -> avg-unpool x2 (except stage 0, whose
    caller applies the top-level 1x1 regression conv instead of another unpool).
    """

    def __init__(self, in_ch, out_ch, nstg, num_stages, deeper=None):
        super().__init__()
        self.nstg = nstg
        self.num_stages = num_stages
        self.is_deepest = nstg == num_stages - 1

        # CONTRACTING PATH (l_con_conv1/bnorm1/relu1, l_con_conv2/bnorm2/relu2, l_con_mp)
        self.con_block1 = _ConvBNReLU(in_ch, out_ch)
        self.con_block2 = _ConvBNReLU(out_ch, out_ch)
        self.con_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        if self.is_deepest:
            # BI-PASSING PATH bottleneck (l_bi_conv1/bnorm1/relu1, l_bi_conv2/bnorm2/relu2)
            self.bi_block1 = _ConvBNReLU(out_ch, 2 * out_ch)
            self.bi_block2 = _ConvBNReLU(2 * out_ch, out_ch)
            self.deeper = None
            up_ch = out_ch  # bi-pass block output channels feeding the unpool
        else:
            assert deeper is not None
            self.deeper = deeper
            up_ch = deeper.out_channels  # deeper stage's actual ext_block2 output width

        # EXPANSIVE PATH unpool (l_ext_convt / l_bi_convt): MatConvNet's
        # UnPooling('method','avg', poolSize=[2,2], stride=[2,2]) is a channel-preserving
        # 2x nearest-style spatial upsample.
        self.unpool = nn.Upsample(scale_factor=2, mode="nearest")

        # EXPANSIVE PATH conv block (l_ext_conv1/bnorm1/relu1 on concat(skip, up), then
        # l_ext_conv2/bnorm2/relu2 -- output channels halve at stage 0 per
        # `cfd/(2^boolean(nstg))` in add_block_multi_img.m, i.e. only stage 0 halves).
        self.out_channels = out_ch // 2 if nstg == 0 else out_ch
        self.ext_block1 = _ConvBNReLU(out_ch + up_ch, out_ch)
        self.ext_block2 = _ConvBNReLU(out_ch, self.out_channels)

    def forward(self, x):
        # contracting path
        c = self.con_block1(x)
        c = self.con_block2(c)
        pooled = self.con_pool(c)

        # deeper stage (recursion) or bottleneck bi-pass
        if self.is_deepest:
            b = self.bi_block1(pooled)
            b = self.bi_block2(b)
            up = self.unpool(b)
        else:
            deeper_out = self.deeper(pooled)
            up = self.unpool(deeper_out)

        # expansive path: skip-concat with contracting-path features at this stage
        merged = torch.cat([c, up], dim=1)
        e = self.ext_block1(merged)
        e = self.ext_block2(e)
        return e


class KSpaceUNet(nn.Module):
    """Faithful port of `cnn_residual_k_space_deep_learning_w_weight_init.m`'s top-level
    DAG: FFT -> 4-stage U-Net (channel progression 2*ch -> 64 -> 128 -> 256 -> 512) ->
    1x1 regression conv back to 2*ch -> additive residual -> IFFT.

    Input/output are real-valued 2-channel (real, imaginary) k-space-domain tensors of
    shape (B, 2*ch, H, W) -- MatConvNet's `comp2ri`/`ri2comp` convention for representing
    complex k-space data as a stacked real/imag channel pair, ch=1 for single-coil MRI.
    """

    def __init__(self, ch=1, base_width=64):
        super().__init__()
        in_ch = 2 * ch  # comp2ri: complex k-space -> stacked real/imag channels
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]  # flt0..flt3
        num_stages = len(widths)

        # Build stages innermost-out (stage 3 is deepest / bi-pass bottleneck stage).
        stage3 = UNetStage(widths[2], widths[3], nstg=3, num_stages=num_stages)
        stage2 = UNetStage(widths[1], widths[2], nstg=2, num_stages=num_stages, deeper=stage3)
        stage1 = UNetStage(widths[0], widths[1], nstg=1, num_stages=num_stages, deeper=stage2)
        stage0 = UNetStage(in_ch, widths[0], nstg=0, num_stages=num_stages, deeper=stage1)
        self.unet = stage0

        # 1x1 regression conv back to input channel count (flte = [1,1,64,2*ch]).
        self.reg_conv = nn.Conv2d(
            widths[0] // 2, in_ch, kernel_size=1, stride=1, padding=0, bias=True
        )

    def _fft2c(self, x_ri):
        """Centered 2D FFT of a stacked real/imag tensor (B, 2*ch, H, W) -> same shape.
        Matches dagnn.FFT's `nnfft2 = fftshift(fftshift(fft2(ifftshift(ifftshift(x,1),2)),1),2)`."""
        b, c2, h, w = x_ri.shape
        ch = c2 // 2
        real, imag = x_ri[:, :ch], x_ri[:, ch:]
        xc = torch.complex(real, imag)
        xc = torch.fft.ifftshift(xc, dim=(-2, -1))
        xc = torch.fft.fft2(xc, dim=(-2, -1))
        xc = torch.fft.fftshift(xc, dim=(-2, -1))
        return torch.cat([xc.real, xc.imag], dim=1)

    def _ifft2c(self, x_ri):
        """Centered 2D inverse FFT, inverse of `_fft2c`."""
        b, c2, h, w = x_ri.shape
        ch = c2 // 2
        real, imag = x_ri[:, :ch], x_ri[:, ch:]
        xc = torch.complex(real, imag)
        xc = torch.fft.fftshift(xc, dim=(-2, -1))
        xc = torch.fft.ifft2(xc, dim=(-2, -1))
        xc = torch.fft.ifftshift(xc, dim=(-2, -1))
        return torch.cat([xc.real, xc.imag], dim=1)

    def forward(self, image):
        """image: (B, ch, H, W) real-valued image-domain input (e.g. zero-filled recon of
        undersampled k-space). Returns the reconstructed image (B, ch, H, W)."""
        b, ch, h, w = image.shape
        zeros = torch.zeros_like(image)
        img_ri = torch.cat([image, zeros], dim=1)  # comp2ri: real image has zero imag part

        input_wfft = self._fft2c(img_ri)  # l_fft (weighting step's data-dependent
        # renormalization omitted -- see module header)
        reg_wfft = self.reg_conv(self.unet(input_wfft))  # U-Net + 1x1 regression conv
        regr_wfft = input_wfft + reg_wfft  # l_sum: additive residual
        regr_img = self._ifft2c(regr_wfft)  # l_ifft

        # Return magnitude-image channels only (real part), matching `ssos.m`'s
        # real-image convention for the single-coil case.
        return regr_img[:, :ch]


def build_kspace_dl_mri():
    # base_width shrunk from the paper's 64 for a fast trace; 4-stage U-Net topology and
    # channel-doubling progression unchanged.
    return KSpaceUNet(ch=1, base_width=8)


def example_input_kspace_dl_mri():
    # 64x64 must be divisible by 2**4=16 for the 4 pool/unpool stages to round-trip exactly.
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "k-Space Deep Learning for Accelerated MRI",
        "build_kspace_dl_mri",
        "example_input_kspace_dl_mri",
        "2018",
        "ported-pytorch",
    ),
]
