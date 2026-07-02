# SOURCE: vendored from santi-pdp/segan_pytorch @ 052238718b037cd72c7fa69bc1577782e5ef015d
# https://raw.githubusercontent.com/santi-pdp/segan_pytorch/052238718b037cd72c7fa69bc1577782e5ef015d/segan/models/generator.py
# https://raw.githubusercontent.com/santi-pdp/segan_pytorch/052238718b037cd72c7fa69bc1577782e5ef015d/segan/models/modules.py
# https://raw.githubusercontent.com/santi-pdp/segan_pytorch/052238718b037cd72c7fa69bc1577782e5ef015d/segan/models/core.py
#
# SEGAN (Pascual, Bonafonte, Serra, Interspeech 2017, "SEGAN: Speech Enhancement
# Generative Adversarial Network") -- the official PyTorch implementation
# (santi-pdp/segan_pytorch, the SEGAN+ follow-up repo used for
# run_segan+_train.sh). The Generator is a 1D fully-convolutional
# encoder-decoder ("no z" skip-connected variant is default here): a stack of
# strided GConv1DBlock encoder layers (reflect-padded Conv1d + PReLU) that
# progressively downsample a raw waveform, optionally concatenated with a
# latent noise code z at the bottleneck, followed by a mirrored stack of
# GDeconv1DBlock decoder layers (ConvTranspose1d + PReLU/Tanh) with alpha-gated
# skip connections (GSkip) from each encoder stage to its matching decoder stage
# -- the "SEGAN+" fully-convolutional generator architecture.
#
# Transcribed verbatim from generator.py/modules.py/core.py -- every Conv1d/
# ConvTranspose1d/PReLU layer, every GSkip alpha-gate, every reflect-pad/forward
# computation is unchanged. Only the parts that are not exercised by a Generator
# forward pass are dropped: `Saver`/checkpoint I/O plumbing (`Model.save`/
# `Model.load`/`load_pretrained`, which need a filesystem `save_path`), the
# unrelated Discriminator/Generator1D/ResBlock1D/ResARModule/SincConv/CombFilter/
# PostProcessingCombNet classes from the same files, and modules.py's
# `if __name__ == '__main__'` demo block. `train.py`'s own `--genc_fmaps
# [64,128,256,512,1024] --genc_poolings [4,4,4,4,4] --gkwidth 31 --z_dim 1024`
# defaults establish the real hyperparameter convention, shrunk here to a tiny
# 2-layer encoder/decoder for fast tracing (skip=True, no_z=False as in the
# repo's default `SEGAN(name='no_z_hidden_skip')` config).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm


# --------------------------------------------------------------------------
# from core.py
# --------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, name="BaseModel"):
        super().__init__()
        self.name = name
        self.optim = None

    def activation(self, name):
        return getattr(nn, name)()

    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn_ = 1
            for s in list(p.size()):
                nn_ = nn_ * s
            pp += nn_
        return pp


# --------------------------------------------------------------------------
# from modules.py
# --------------------------------------------------------------------------
def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == "bnorm":
        return nn.BatchNorm1d(num_feats)
    elif norm_type == "snorm":
        spectral_norm(param)
        return None
    elif norm_type is None:
        return None
    else:
        raise TypeError("Unrecognized norm type: ", norm_type)


class GConv1DBlock(nn.Module):
    def __init__(self, ninp, fmaps, kwidth, stride=1, bias=True, norm_type=None):
        super().__init__()
        self.conv = nn.Conv1d(ninp, fmaps, kwidth, stride=stride, bias=bias)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x, ret_linear=False):
        if self.stride > 1:
            P = (self.kwidth // 2 - 1, self.kwidth // 2)
        else:
            P = (self.kwidth // 2, self.kwidth // 2)
        x_p = F.pad(x, P, mode="reflect")
        a = self.conv(x_p)
        a = self.forward_norm(a, self.norm)
        h = self.act(a)
        if ret_linear:
            return h, a
        else:
            return h


class GDeconv1DBlock(nn.Module):
    def __init__(self, ninp, fmaps, kwidth, stride=4, bias=True, norm_type=None, act=None):
        super().__init__()
        pad = max(0, (stride - kwidth) // -2)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps, kwidth, stride=stride, padding=pad)
        self.norm = build_norm_layer(norm_type, self.deconv, fmaps)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        h = self.deconv(x)
        if self.kwidth % 2 != 0:
            h = h[:, :, :-1]
        h = self.forward_norm(h, self.norm)
        h = self.act(h)
        return h


# --------------------------------------------------------------------------
# from generator.py
# --------------------------------------------------------------------------
class GSkip(nn.Module):
    def __init__(
        self, skip_type, size, skip_init, skip_dropout=0, merge_mode="sum", kwidth=11, bias=True
    ):
        # skip_init only applies to alpha skips
        super().__init__()
        self.merge_mode = merge_mode
        if skip_type == "alpha" or skip_type == "constant":
            if skip_init == "zero":
                alpha_ = torch.zeros(size)
            elif skip_init == "randn":
                alpha_ = torch.randn(size)
            elif skip_init == "one":
                alpha_ = torch.ones(size)
            else:
                raise TypeError("Unrecognized alpha init scheme: ", skip_init)
            if skip_type == "alpha":
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
            else:
                # constant, not learnable
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
                self.skip_k.requires_grad = False
        elif skip_type == "conv":
            if kwidth > 1:
                pad = kwidth // 2
            else:
                pad = 0
            self.skip_k = nn.Conv1d(size, size, kwidth, stride=1, padding=pad, bias=bias)
        else:
            raise TypeError("Unrecognized GSkip scheme: ", skip_type)
        self.skip_type = skip_type
        if skip_dropout > 0:
            self.skip_dropout = nn.Dropout(skip_dropout)

    def __repr__(self):
        if self.skip_type == "alpha":
            return self._get_name() + "(Alpha(1))"
        elif self.skip_type == "constant":
            return self._get_name() + "(Constant(1))"
        else:
            return super().__repr__()

    def forward(self, hj, hi):
        if self.skip_type == "conv":
            sk_h = self.skip_k(hj)
        else:
            skip_k = self.skip_k.repeat(hj.size(0), 1, hj.size(2))
            sk_h = skip_k * hj
        if hasattr(self, "skip_dropout"):
            sk_h = self.skip_dropout(sk_h)
        if self.merge_mode == "sum":
            # merge with input hi on current layer
            return sk_h + hi
        elif self.merge_mode == "concat":
            return torch.cat((hi, sk_h), dim=1)
        else:
            raise TypeError("Unrecognized skip merge mode: ", self.merge_mode)


class Generator(Model):
    def __init__(
        self,
        ninputs,
        fmaps,
        kwidth,
        poolings,
        dec_fmaps=None,
        dec_kwidth=None,
        dec_poolings=None,
        z_dim=None,
        no_z=False,
        skip=True,
        bias=False,
        skip_init="one",
        skip_dropout=0,
        skip_type="alpha",
        norm_type=None,
        skip_merge="sum",
        skip_kwidth=11,
        name="Generator",
    ):
        super().__init__(name=name)
        self.skip = skip
        self.bias = bias
        self.no_z = no_z
        self.z_dim = z_dim
        self.enc_blocks = nn.ModuleList()
        assert isinstance(fmaps, list), type(fmaps)
        assert isinstance(poolings, list), type(poolings)
        if isinstance(kwidth, int):
            kwidth = [kwidth] * len(fmaps)
        assert isinstance(kwidth, list), type(kwidth)
        skips = {}
        ninp = ninputs
        for pi, (fmap, pool, kw) in enumerate(zip(fmaps, poolings, kwidth), start=1):
            if skip and pi < len(fmaps):
                # Make a skip connection for all but last hidden layer
                gskip = GSkip(
                    skip_type,
                    fmap,
                    skip_init,
                    skip_dropout,
                    merge_mode=skip_merge,
                    kwidth=skip_kwidth,
                    bias=bias,
                )
                l_i = pi - 1
                skips[l_i] = {"alpha": gskip}
                setattr(self, "alpha_{}".format(l_i), skips[l_i]["alpha"])
            enc_block = GConv1DBlock(ninp, fmap, kw, stride=pool, bias=bias, norm_type=norm_type)
            self.enc_blocks.append(enc_block)
            ninp = fmap

        self.skips = skips
        if not no_z and z_dim is None:
            z_dim = fmaps[-1]
        if not no_z:
            ninp += z_dim
        # Ensure we have fmaps, poolings and kwidth ready to decode
        if dec_fmaps is None:
            dec_fmaps = fmaps[::-1][1:] + [1]
        else:
            assert isinstance(dec_fmaps, list), type(dec_fmaps)
        if dec_poolings is None:
            dec_poolings = poolings[:]
        else:
            assert isinstance(dec_poolings, list), type(dec_poolings)
        self.dec_poolings = dec_poolings
        if dec_kwidth is None:
            dec_kwidth = kwidth[:]
        else:
            if isinstance(dec_kwidth, int):
                dec_kwidth = [dec_kwidth] * len(dec_fmaps)
        assert isinstance(dec_kwidth, list), type(dec_kwidth)
        # Build the decoder
        self.dec_blocks = nn.ModuleList()
        for pi, (fmap, pool, kw) in enumerate(zip(dec_fmaps, dec_poolings, dec_kwidth), start=1):
            if skip and pi > 1 and pool > 1:
                if skip_merge == "concat":
                    ninp *= 2

            if pi >= len(dec_fmaps):
                act = "Tanh"
            else:
                act = None
            if pool > 1:
                dec_block = GDeconv1DBlock(
                    ninp, fmap, kw, stride=pool, norm_type=norm_type, bias=bias, act=act
                )
            else:
                dec_block = GConv1DBlock(ninp, fmap, kw, stride=1, bias=bias, norm_type=norm_type)
            self.dec_blocks.append(dec_block)
            ninp = fmap

    def forward(self, x, z=None, ret_hid=False):
        hall = {}
        hi = x
        skips = self.skips
        for l_i, enc_layer in enumerate(self.enc_blocks):
            hi, linear_hi = enc_layer(hi, True)
            if self.skip and l_i < (len(self.enc_blocks) - 1):
                skips[l_i]["tensor"] = linear_hi
            if ret_hid:
                hall["enc_{}".format(l_i)] = hi
        if not self.no_z:
            if z is None:
                # make z
                z = torch.randn(hi.size(0), self.z_dim, *hi.size()[2:])
                if hi.is_cuda:
                    z = z.to("cuda")
            if len(z.size()) != len(hi.size()):
                raise ValueError(
                    "len(z.size) {} != len(hi.size) {}".format(len(z.size()), len(hi.size()))
                )
            if not hasattr(self, "z"):
                self.z = z
            hi = torch.cat((z, hi), dim=1)
            if ret_hid:
                hall["enc_zc"] = hi
        else:
            z = None
        enc_layer_idx = len(self.enc_blocks) - 1
        for l_i, dec_layer in enumerate(self.dec_blocks):
            if self.skip and enc_layer_idx in self.skips and self.dec_poolings[l_i] > 1:
                skip_conn = skips[enc_layer_idx]
                hi = skip_conn["alpha"](skip_conn["tensor"], hi)
            hi = dec_layer(hi)
            enc_layer_idx -= 1
            if ret_hid:
                hall["dec_{}".format(l_i)] = hi
        if ret_hid:
            return hi, hall
        else:
            return hi


def build_segan_generator():
    torch.manual_seed(0)
    # train.py's own defaults are genc_fmaps=[64,128,256,512,1024],
    # genc_poolings=[4,4,4,4,4], gkwidth=31, z_dim=1024 -- shrunk to a tiny
    # 2-layer encoder/decoder (skip=True, no_z=False) for fast tracing.
    model = Generator(
        ninputs=1,
        fmaps=[4, 8],
        kwidth=5,
        poolings=[2, 2],
        z_dim=8,
        no_z=False,
        skip=True,
        bias=False,
    )
    model.eval()
    return model


def example_input_segan_generator():
    torch.manual_seed(0)
    # Raw waveform chunk, (batch, channels=1, samples).
    return torch.randn(1, 1, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SEGAN", "build_segan_generator", "example_input_segan_generator", 2017, MENAGERIE_ZOO),
]
