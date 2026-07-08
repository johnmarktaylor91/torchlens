# SOURCE: vendored from KIMGEONUNG/BigColor @ main
# https://github.com/KIMGEONUNG/BigColor
# Files:
#   https://raw.githubusercontent.com/KIMGEONUNG/BigColor/main/models/layers.py
#   https://raw.githubusercontent.com/KIMGEONUNG/BigColor/main/models/encoders.py
#   https://raw.githubusercontent.com/KIMGEONUNG/BigColor/main/models/biggan.py
#   https://raw.githubusercontent.com/KIMGEONUNG/BigColor/main/models/common.py
#
# Kim et al. 2022 (ECCV) "BigColor: Colorization using a Generative
# Color Prior for Natural Images" -- the official repo's `Colorizer`
# (`models/common.py`) wraps a pretrained-class-conditional BigGAN
# `Generator` (`models/biggan.py`, the standard BigGAN-deep generator
# architecture: an `nn.Linear` projection into a `bottom_width x
# bottom_width` feature grid, a stack of class-conditional-batchnorm
# `GBlock` residual upsampling blocks with a self-attention block inserted
# at 64x64 resolution, and a final batchnorm-relu-conv-tanh RGB head) with a
# dedicated grayscale-image encoder (`EncoderF_Res`, `models/encoders.py`:
# a `ResConvBlock` residual-conv downsampling tower over the grayscale
# input, also with a self-attention block at 64x64) that injects
# colorization content features directly into an intermediate BigGAN
# resolution stage via `Generator.forward_from(z, y, num_layer, h)` --
# skipping the generator's earlier layers and resuming the block loop at
# `num_layer` using the encoder's feature map `h` as the incoming
# activation, so the encoder's grayscale content features and the class
# embedding `y` jointly drive the remaining upsampling/coloring blocks
# (this "inject at an intermediate BigGAN layer" mechanism, using BigGAN as
# a generative color prior rather than training a colorization network from
# scratch, is BigColor's core architectural contribution).
#
# `identity`/`SN`/`SNConv2d`/`SNLinear`/`SNEmbedding`/`Attention`/`myBN`/
# `ccbn`/`bn`/`GBlock`/`DBlock` (from `layers.py`), `ClassConditionNorm`/
# `ResConvBlock`/`EncoderF64_Res`/`EncoderF32_Res`/`EncoderF8_Res`/
# `EncoderZ_Res`/`EncoderF_Res` (from `encoders.py`), `G_arch`/`Generator`/
# `D_arch`/`Discriminator` (from `biggan.py`), and `Colorizer` (from
# `common.py`) are copied VERBATIM (architecture completely unchanged) from
# the four files above, concatenated into one module. The only edits are
# import-path fixes:
#   - each file's own intra-package relative imports (`from . import
#     layers`, `from .layers import SNConv2d, Attention`, `from .encoders
#     import (...)`, `from .biggan import Generator`) are dropped, since
#     those exact names are already defined module-level earlier in this
#     same concatenated file (all four files originate from the same repo
#     commit, so behavior is identical to a real four-file import).
#   - `layers.SNConv2d` / `layers.GBlock` / etc. qualified references inside
#     `biggan.py` are rewritten to bare `SNConv2d` / `GBlock` / etc. for the
#     same reason (the `layers` module's contents are inlined above).
#   - `common.py`'s `VGG16Perceptual` class (a fixed VGG perceptual-loss
#     helper used only for the training-time loss, not the colorization
#     architecture) is dropped: its `__init__` unpickles a
#     `path_vgg`-provided external VGG checkpoint file at construction time
#     (`pickle.load(open(path_vgg, 'rb'))`), which is not part of this base
#     environment and is irrelevant to the `Colorizer` forward path
#     exercised here.
#   - module-level `import functools`/`torch`/`torch.nn`/`torch.nn.
#     functional`/`torch.optim`/`torch.nn.Parameter as P`/`torch.nn.init`
#     are merged into one shared import block at the top (each source file
#     imported an overlapping subset of these); `torchvision.transforms`
#     (only used by the dropped `VGG16Perceptual`) is not needed and is
#     therefore omitted.
#
# `build_bigcolor()` constructs the REAL `Colorizer(config, path_ckpt_g,
# norm_type, dim_f=16)` exactly as the real `colorize.py`/`train.py`
# scripts do (`dim_f=16` -> `EncoderF_Res`, `id_mid_layer=2`, matching
# `Colorizer.__init__`'s own `dim_f == 16` branch), with `load_g=False` to
# skip loading the pretrained `G_ema_256.pth` checkpoint (real,
# already-present configuration branch in `Colorizer.__init__`, not an
# architecture change -- the exact same branch a from-scratch/randomly
# initialized training run takes) and `norm_type="batch"` (one of the
# `ResConvBlock`'s real, already-present `norm` choices; batch avoids
# needing a class-embedding-conditioned norm plumbed through the encoder,
# which `Colorizer.forward` itself does not do for the encoder branch --
# see `self.E(x_gray, self.G.shared(c))`, whose second positional arg
# `ResConvBlock.forward` only consults when `has_condition` is True, i.e.
# for the `adain`/`adabatch` norm choices). `G_ch=96` (BigGAN's real
# default generator channel width, matching the fixed `ch_unit=96` default
# hard-coded into `EncoderF_Res`, since the encoder's output channel count
# at the injection point -- `ch_unit * 8 = 768` -- must equal the
# generator's `in_channels` at `Generator.blocks[id_mid_layer]`, which for
# `resolution=256` and `G_ch=96` is `96 * 8 = 768`) and `resolution=256`
# (matching the real `G_ema_256.pth` checkpoint's resolution) are the real
# config values the repo uses for its shipped 256x256 checkpoint, not
# invented ones; `example_input_bigcolor()` drives the real `Colorizer.
# forward(x_gray, c, z)` signature with a random grayscale image, a class
# index, and a latent noise vector.
#
# NOTE: `Colorizer.train(self, mode=True)` (kept verbatim below) does not
# `return self` in either branch, unlike the base `nn.Module.train`/`eval`
# it overrides (which do); this is the real, unmodified upstream behavior
# (`models/common.py:182-186`), not a vendoring bug -- calling
# `colorizer.eval()` and then discarding its (None) return value, as
# `build_bigcolor()` does below, is required to avoid propagating `None`.

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter as P
from torch.nn import init


# ---- from layers.py ----


def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
    def forward(self, input):
        return input


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer("u%d" % i, torch.randn(1, num_outputs))
            self.register_buffer("sv%d" % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, "u%d" % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, "sv%d" % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        # Update the svs
        if self.training:
            with (
                torch.no_grad()
            ):  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
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
        num_svs=1,
        num_itrs=1,
        eps=1e-12,
    ):
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(
            x, self.W_(), self.bias, self.stride, self.padding, self.dilation, self.groups
        )


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True, num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        num_svs=1,
        num_itrs=1,
        eps=1e-12,
    ):
        nn.Embedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
        )
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def forward(self, x):
        return F.embedding(x, self.W_())


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name="attention"):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y=None, use_in=False):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3])
        )
        output = self.gamma * o + x
        return output


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift
    # return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x**2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = m2 - m**2
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super(myBN, self).__init__()
        # momentum for updating running stats
        self.momentum = momentum
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Register buffers
        self.register_buffer("stored_mean", torch.zeros(num_channels))
        self.register_buffer("stored_var", torch.ones(num_channels))
        self.register_buffer("accumulation_counter", torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
            # If accumulating standing stats, increment them
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            # If not accumulating standing stats, take running averages
            else:
                self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
                self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        which_linear,
        eps=1e-5,
        momentum=0.1,
        cross_replica=False,
        mybn=False,
        norm_style="bn",
    ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # Norm style?
        self.norm_style = norm_style

        if self.cross_replica:
            raise NotImplementedError
        elif self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ["bn", "in"]:
            self.register_buffer("stored_mean", torch.zeros(output_size))
            self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        if y is not None:
            gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
            bias = self.bias(y).view(y.size(0), -1, 1, 1)
        # If using my batchnorm
        if self.mybn:
            return self.bn(x, gain=gain, bias=bias)
        elif self.cross_replica:
            return self.bn(x) * gain + bias
        # else:
        else:
            if self.norm_style == "bn":
                out = F.batch_norm(
                    x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps
                )
            elif self.norm_style == "in":
                out = F.instance_norm(
                    x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps
                )
            elif self.norm_style == "nonorm":
                out = x
            if y is not None:
                out = out * gain + bias
            return out

    def extra_repr(self):
        s = "out: {output_size}, in: {input_size},"
        s += " cross_replica={cross_replica}"
        return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1, cross_replica=False, mybn=False):
        super(bn, self).__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn

        if self.cross_replica:
            raise NotImplementedError
        elif mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        # Register buffers if neither of the above
        else:
            self.register_buffer("stored_mean", torch.zeros(output_size))
            self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y=None):
        if self.cross_replica or self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            if self.mybn:
                return self.bn(x, gain=gain, bias=bias)
            elif self.cross_replica:
                return self.bn(x) * gain + bias
        else:
            return F.batch_norm(
                x,
                self.stored_mean,
                self.stored_var,
                self.gain,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )


# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must
# be preselected)
class GBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv=nn.Conv2d,
        which_bn=bn,
        activation=None,
        upsample=None,
    ):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

        # instance norm layers
        self.in_initialized = False
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.in1.weight.requires_grad = False
        self.in1.bias.requires_grad = False
        self.in2.weight.requires_grad = False
        self.in2.bias.requires_grad = False

    def reset_in_init(self):
        self.in_initialized = False

    def init_in(self, which_bn, which_in, x, y):
        # carefully initialize IN's weights such that the output does not change
        with torch.no_grad():
            h = which_bn(x, y)
            mean = torch.mean(h, (2, 3)).squeeze(0)
            std = torch.std(h.view(h.size(0), h.size(1), -1), 2).squeeze(0)
            which_in.weight.copy_(std)
            which_in.bias.copy_(mean)

    def forward(self, x, y, use_in):
        if use_in:
            if not self.in_initialized:
                self.init_in(self.bn1, self.in1, x, y)
            h = self.in1(x)
        else:
            h = self.bn1(x, y)
            self.in_initialized = False

        h = self.activation(h)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)

        if use_in:
            if not self.in_initialized:
                self.init_in(self.bn2, self.in2, h, y)
                self.in_initialized = True
            h = self.in2(h)
        else:
            h = self.bn2(h, y)

        h = self.activation(h)
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv=SNConv2d,
        wide=True,
        preactivation=False,
        activation=None,
        downsample=None,
    ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


# ---- from encoders.py ----


class ClassConditionNorm(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        which_linear=functools.partial(nn.Linear, bias=False),
        eps=1e-5,
        norm_style="bn",
    ):
        super().__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        self.norm_style = norm_style

        self.register_buffer("stored_mean", torch.zeros(output_size))
        self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)

        if self.norm_style == "bn":
            out = F.batch_norm(
                x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps
            )
        elif self.norm_style == "in":
            out = F.instance_norm(
                x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps
            )

        return out * gain + bias


class ResConvBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        ch_c=128,
        is_down=False,
        dropout=0.2,
        activation="relu",
        pool="avg",
        norm="batch",
        use_res=True,
        **kwargs,
    ):
        super().__init__()

        self.is_down = is_down
        self.has_condition = False
        self.use_res = use_res

        # Convolution
        if self.use_res:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

        self.conv_1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)

        # Normalization
        if norm == "batch":
            self.normalize_1 = nn.BatchNorm2d(ch_in)
            self.normalize_2 = nn.BatchNorm2d(ch_out)
        elif norm == "id":
            self.normalize_1 = nn.Identity()
            self.normalize_2 = nn.Identity()
        elif norm == "instance":
            self.normalize_1 = nn.InstanceNorm2d(ch_in)
            self.normalize_2 = nn.InstanceNorm2d(ch_out)
        elif norm == "layer":
            self.normalize_1 = nn.LayerNorm(kwargs["l_norm_shape_1"])
            self.normalize_2 = nn.LayerNorm(kwargs["l_norm_shape_2"])
        elif norm == "adain":
            self.has_condition = True
            self.normalize_1 = ClassConditionNorm(ch_in, ch_c, norm_style="in")
            self.normalize_2 = ClassConditionNorm(ch_out, ch_c, norm_style="in")
        elif norm == "adabatch":
            self.has_condition = True
            self.normalize_1 = ClassConditionNorm(ch_in, ch_c, norm_style="bn")
            self.normalize_2 = ClassConditionNorm(ch_out, ch_c, norm_style="bn")
        else:
            raise Exception("Invalid Normalization")

        # Nonlinearity
        self.activation = None
        if activation == "relu":
            self.activation = lambda x: F.relu(x, True)
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "lrelu":
            slope = kwargs["l_slope"]
            self.activation = lambda x: F.leaky_relu(x, slope, True)
        else:
            raise Exception("Invalid Nonlinearity")

        # Pooling
        self.pool = None
        if pool == "avg":
            self.pool = lambda x: F.avg_pool2d(x, kernel_size=2)
        elif pool == "max":
            self.pool = lambda x: F.max_pool2d(x, kernel_size=2)
        elif pool == "min":
            self.pool = lambda x: F.min_pool2d(x, kernel_size=2)
        else:
            raise Exception("Invalid Pooling")

        # Dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, c=None):
        # Residual Path
        x_ = x

        if self.has_condition:
            x_ = self.normalize_1(x_, c)
        else:
            x_ = self.normalize_1(x_)
        x_ = self.activation(x_)

        if self.is_down:
            x_ = self.pool(x_)

        x_ = self.conv_1(x_)

        if self.has_condition:
            x_ = self.normalize_2(x_, c)
        else:
            x_ = self.normalize_2(x_)
        x_ = self.activation(x_)
        x_ = self.conv_2(x_)

        # Main Path
        if self.use_res:
            if self.is_down:
                x = self.pool(x)
            x = self.conv(x)
        else:
            x = 0

        # Merge
        x = x + x_

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class EncoderF64_Res(nn.Module):
    def __init__(
        self,
        ch_in=1,
        ch_out=768,
        ch_unit=96,
        norm="batch",
        activation="relu",
        init="ortho",
        use_att=False,
    ):
        super().__init__()

        self.init = init
        self.use_att = use_att

        kwargs = {}
        if activation == "lrelu":
            kwargs["l_slope"] = 0.2

        if use_att:
            print("Adding attention layer in E at resolution %d" % (64))
            conv4att = functools.partial(
                SNConv2d, kernel_size=3, padding=1, num_svs=1, num_itrs=1, eps=1e-06
            )
            self.att = Attention(384, conv4att)

        # output is 96 x 256 x 256
        self.res1 = ResConvBlock(
            ch_in, ch_unit * 1, is_down=False, activation=activation, norm=norm, **kwargs
        )
        # output is 192 x 128 x 128
        self.res2 = ResConvBlock(
            ch_unit * 1, ch_unit * 2, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  384 x 64 x 64
        self.res3 = ResConvBlock(
            ch_unit * 2, ch_unit * 4, is_down=True, activation=activation, norm=norm, **kwargs
        )

        # output is  384 x 64 x 64
        self.res4 = ResConvBlock(
            ch_unit * 4,
            ch_unit * 4,
            is_down=False,
            activation=activation,
            norm=norm,
            dropout=None,
            **kwargs,
        )

        self.init_weights()

    def forward(self, x, c=None):
        x = self.res1(x, c)
        x = self.res2(x, c)
        x = self.res3(x, c)
        if self.use_att:
            x = self.att(x)
        x = self.res4(x, c)
        return x

    def forward_with_cp(self, x, cp):
        x = self.res1(x, cp[0])
        x = self.res2(x, cp[1])
        x = self.res3(x, cp[2])
        if self.use_att:
            x = self.att(x)
        x = self.res4(x, cp[3])
        return x

    def init_weights(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    pass
                    # print('Init style not recognized...')


class EncoderF32_Res(nn.Module):
    def __init__(
        self,
        ch_in=1,
        ch_out=768,
        ch_unit=96,
        norm="batch",
        activation="relu",
        init="ortho",
        use_att=False,
    ):
        super().__init__()

        self.init = init
        self.use_att = use_att

        kwargs = {}
        if activation == "lrelu":
            kwargs["l_slope"] = 0.2

        if use_att:
            print("Adding attention layer in E at resolution %d" % (64))
            conv4att = functools.partial(
                SNConv2d, kernel_size=3, padding=1, num_svs=1, num_itrs=1, eps=1e-06
            )
            self.att = Attention(384, conv4att)

        # output is 96 x 256 x 256
        self.res1 = ResConvBlock(
            ch_in, ch_unit * 1, is_down=False, activation=activation, norm=norm, **kwargs
        )
        # output is 192 x 128 x 128
        self.res2 = ResConvBlock(
            ch_unit * 1, ch_unit * 2, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  384 x 64 x 64
        self.res3 = ResConvBlock(
            ch_unit * 2, ch_unit * 4, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  768 x 32 x 32
        self.res4 = ResConvBlock(
            ch_unit * 4, ch_unit * 8, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  768 x 32 x 32
        self.res5 = ResConvBlock(
            ch_unit * 8,
            ch_unit * 8,
            is_down=False,
            activation=activation,
            norm=norm,
            dropout=None,
            **kwargs,
        )

        self.init_weights()

    def forward(self, x, c=None):
        x = self.res1(x, c)
        x = self.res2(x, c)
        x = self.res3(x, c)
        if self.use_att:
            x = self.att(x)
        x = self.res4(x, c)
        x = self.res5(x, c)
        return x

    def forward_with_cp(self, x, cp):
        x = self.res1(x, cp[0])
        x = self.res2(x, cp[1])
        x = self.res3(x, cp[2])
        if self.use_att:
            x = self.att(x)
        x = self.res4(x, cp[3])
        x = self.res5(x, cp[4])
        return x

    def init_weights(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    pass
                    # print('Init style not recognized...')


class EncoderF8_Res(nn.Module):
    def __init__(
        self,
        ch_in=1,
        ch_out=768,
        ch_unit=96,
        norm="batch",
        activation="relu",
        init="ortho",
        use_att=False,
    ):
        super().__init__()

        self.init = init
        self.use_att = use_att

        kwargs = {}
        if activation == "lrelu":
            kwargs["l_slope"] = 0.2

        if use_att:
            print("Adding attention layer in E at resolution %d" % (64))
            conv4att = functools.partial(
                SNConv2d, kernel_size=3, padding=1, num_svs=1, num_itrs=1, eps=1e-06
            )
            self.att = Attention(384, conv4att)

        # output is 96 x 256 x 256
        self.res1 = ResConvBlock(
            ch_in, ch_unit * 1, is_down=False, activation=activation, norm=norm, **kwargs
        )
        # output is 192 x 128 x 128
        self.res2 = ResConvBlock(
            ch_unit * 1, ch_unit * 2, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  384 x 64 x 64
        self.res3 = ResConvBlock(
            ch_unit * 2, ch_unit * 4, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  768 x 32 x 32
        self.res4 = ResConvBlock(
            ch_unit * 4, ch_unit * 8, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  768 x 16 x 16
        self.res5 = ResConvBlock(
            ch_unit * 8, ch_unit * 8, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  1536 x 8 x 8
        self.res6 = ResConvBlock(
            ch_unit * 8,
            ch_unit * 16,
            is_down=True,
            activation=activation,
            norm=norm,
            dropout=None,
            **kwargs,
        )
        self.init_weights()

    def forward(self, x, c=None):
        x = self.res1(x, c)
        x = self.res2(x, c)
        x = self.res3(x, c)
        x = self.res4(x, c)
        x = self.res5(x, c)
        x = self.res6(x, c)
        return x

    def init_weights(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    pass
                    # print('Init style not recognized...')


class EncoderZ_Res(nn.Module):
    def __init__(
        self,
        ch_in=1,
        ch_out=768,
        ch_unit=96,
        norm="batch",
        activation="relu",
        init="ortho",
        use_att=False,
    ):
        super().__init__()

        self.init = init
        self.use_att = use_att

        kwargs = {}
        if activation == "lrelu":
            kwargs["l_slope"] = 0.2

        if use_att:
            print("Adding attention layer in E at resolution %d" % (64))
            conv4att = functools.partial(
                SNConv2d, kernel_size=3, padding=1, num_svs=1, num_itrs=1, eps=1e-06
            )
            self.att = Attention(384, conv4att)

        # output is 96 x 256 x 256
        self.res1 = ResConvBlock(
            ch_in, ch_unit * 1, is_down=False, activation=activation, norm=norm, **kwargs
        )
        # output is 192 x 128 x 128
        self.res2 = ResConvBlock(
            ch_unit * 1, ch_unit * 2, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  384 x 64 x 64
        self.res3 = ResConvBlock(
            ch_unit * 2, ch_unit * 4, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  384 x 32 x 32
        self.res4 = ResConvBlock(
            ch_unit * 4, ch_unit * 4, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  384 x 16 x 16
        self.res5 = ResConvBlock(
            ch_unit * 4, ch_unit * 4, is_down=True, activation=activation, norm=norm, **kwargs
        )
        # output is  384 x 8 x 8
        self.res6 = ResConvBlock(
            ch_unit * 4,
            ch_unit * 4,
            is_down=True,
            activation=activation,
            norm=norm,
            dropout=None,
            **kwargs,
        )
        # output is  384 x 4 x 4
        self.res7 = ResConvBlock(
            ch_unit * 4,
            ch_unit * 4,
            is_down=True,
            activation=activation,
            norm=norm,
            dropout=None,
            **kwargs,
        )
        self.pool = nn.AvgPool2d(4)
        self.mlp = nn.Linear(384, 119)

        self.init_weights()

    def forward(self, x, c=None):
        x = self.res1(x, c)
        x = self.res2(x, c)
        x = self.res3(x, c)
        x = self.res4(x, c)
        x = self.res5(x, c)
        x = self.res6(x, c)
        x = self.res7(x, c)
        x = self.pool(x)
        x = x.squeeze()
        x = self.mlp(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    pass
                    # print('Init style not recognized...')


class EncoderF_Res(nn.Module):
    def __init__(
        self,
        ch_in=1,
        ch_out=768,
        ch_unit=96,
        norm="batch",
        activation="relu",
        init="ortho",
        use_att=False,
        use_res=True,
    ):
        super().__init__()

        self.init = init
        self.use_att = use_att

        kwargs = {}
        if activation == "lrelu":
            kwargs["l_slope"] = 0.2

        if use_att:
            print("Adding attention layer in E at resolution %d" % (64))
            conv4att = functools.partial(
                SNConv2d, kernel_size=3, padding=1, num_svs=1, num_itrs=1, eps=1e-06
            )
            self.att = Attention(384, conv4att)

        # output is 96 x 256 x 256
        self.res1 = ResConvBlock(
            ch_in,
            ch_unit * 1,
            is_down=False,
            activation=activation,
            norm=norm,
            use_res=use_res,
            **kwargs,
        )
        # output is 192 x 128 x 128
        self.res2 = ResConvBlock(
            ch_unit * 1,
            ch_unit * 2,
            is_down=True,
            activation=activation,
            norm=norm,
            use_res=use_res,
            **kwargs,
        )
        # output is  384 x 64 x 64
        self.res3 = ResConvBlock(
            ch_unit * 2,
            ch_unit * 4,
            is_down=True,
            activation=activation,
            norm=norm,
            use_res=use_res,
            **kwargs,
        )
        # output is  768 x 32 x 32
        self.res4 = ResConvBlock(
            ch_unit * 4,
            ch_unit * 8,
            is_down=True,
            activation=activation,
            norm=norm,
            use_res=use_res,
            **kwargs,
        )
        # output is  768 x 16 x 16
        self.res5 = ResConvBlock(
            ch_unit * 8,
            ch_unit * 8,
            is_down=True,
            activation=activation,
            norm=norm,
            use_res=use_res,
            dropout=None,
            **kwargs,
        )

        self.init_weights()

    def forward(self, x, c=None):
        x = self.res1(x, c)
        x = self.res2(x, c)
        x = self.res3(x, c)
        if self.use_att:
            x = self.att(x)
        x = self.res4(x, c)
        x = self.res5(x, c)
        return x

    def forward_with_cp(self, x, cp):
        x = self.res1(x, cp[0])
        x = self.res2(x, cp[1])
        x = self.res3(x, cp[2])
        if self.use_att:
            x = self.att(x)
        x = self.res4(x, cp[3])
        x = self.res5(x, cp[4])
        return x

    def init_weights(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    pass
                    # print('Init style not recognized...')


# z: ([batch, 17])
# h: ([batch, 24576])
# index 0 : ([batch, 1536, 4, 4])
# index 1 : ([batch, 1536, 8, 8])
# index 2 : ([batch, 768, 16, 16])
# index 3 : ([batch, 768, 32, 32])
# index 4 : ([batch, 384, 64, 64])
# index 5 : ([batch, 192, 128, 128])
# index 6: ([batch, 96, 256, 256])
# result: ([batch, 3 256, 256])
if __name__ == "__main__":
    model = EncoderZ_Res(use_att=True)
    model.float()
    y = model(torch.randn(4, 1, 256, 256))
    print(y.shape)


# ---- from biggan.py ----


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention="64", ksize="333333", dilation="111111"):
    arch = {}
    arch[512] = {
        "in_channels": [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
        "out_channels": [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
        "upsample": [True] * 7,
        "resolution": [8, 16, 32, 64, 128, 256, 512],
        "attention": {
            2**i: (2**i in [int(item) for item in attention.split("_")]) for i in range(3, 10)
        },
    }
    arch[256] = {
        "in_channels": [ch * item for item in [16, 16, 8, 8, 4, 2]],
        "out_channels": [ch * item for item in [16, 8, 8, 4, 2, 1]],
        "upsample": [True] * 6,
        "resolution": [8, 16, 32, 64, 128, 256],
        "attention": {
            2**i: (2**i in [int(item) for item in attention.split("_")]) for i in range(3, 9)
        },
    }
    arch[128] = {
        "in_channels": [ch * item for item in [16, 16, 8, 4, 2]],
        "out_channels": [ch * item for item in [16, 8, 4, 2, 1]],
        "upsample": [True] * 5,
        "resolution": [8, 16, 32, 64, 128],
        "attention": {
            2**i: (2**i in [int(item) for item in attention.split("_")]) for i in range(3, 8)
        },
    }
    arch[64] = {
        "in_channels": [ch * item for item in [16, 16, 8, 4]],
        "out_channels": [ch * item for item in [16, 8, 4, 2]],
        "upsample": [True] * 4,
        "resolution": [8, 16, 32, 64],
        "attention": {
            2**i: (2**i in [int(item) for item in attention.split("_")]) for i in range(3, 7)
        },
    }
    arch[32] = {
        "in_channels": [ch * item for item in [4, 4, 4]],
        "out_channels": [ch * item for item in [4, 4, 4]],
        "upsample": [True] * 3,
        "resolution": [8, 16, 32],
        "attention": {
            2**i: (2**i in [int(item) for item in attention.split("_")]) for i in range(3, 6)
        },
    }

    return arch


class Generator(nn.Module):
    def __init__(
        self,
        G_ch=64,
        dim_z=128,
        bottom_width=4,
        resolution=128,
        G_kernel_size=3,
        G_attn="64",
        n_classes=1000,
        num_G_SVs=1,
        num_G_SV_itrs=1,
        G_shared=True,
        shared_dim=0,
        hier=False,
        cross_replica=False,
        mybn=False,
        G_activation=nn.ReLU(inplace=True),
        optimizer="Adam",
        G_lr=5e-5,
        G_B1=0.0,
        G_B2=0.999,
        adam_eps=1e-8,
        BN_eps=1e-5,
        SN_eps=1e-12,
        G_mixed_precision=False,
        G_fp16=False,
        G_init="ortho",
        skip_init=False,
        no_optim=False,
        G_param="SN",
        norm_style="bn",
        **kwargs,
    ):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch["in_channels"]) + 1
            self.z_chunk_size = self.dim_z // self.num_slots
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == "SN":
            self.which_conv = functools.partial(
                SNConv2d,
                kernel_size=3,
                padding=1,
                num_svs=num_G_SVs,
                num_itrs=num_G_SV_itrs,
                eps=self.SN_eps,
            )
            self.which_linear = functools.partial(
                SNLinear, num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps
            )
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding
        bn_linear = (
            functools.partial(self.which_linear, bias=False)
            if self.G_shared
            else self.which_embedding
        )
        self.which_bn = functools.partial(
            ccbn,
            which_linear=bn_linear,
            cross_replica=self.cross_replica,
            mybn=self.mybn,
            input_size=(self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes),
            norm_style=self.norm_style,
            eps=self.BN_eps,
        )

        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = self.which_embedding(n_classes, self.shared_dim) if G_shared else identity()

        # First linear layer
        self.linear = self.which_linear(
            self.dim_z // self.num_slots, self.arch["in_channels"][0] * (self.bottom_width**2)
        )

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch["out_channels"])):
            self.blocks += [
                [
                    GBlock(
                        in_channels=self.arch["in_channels"][index],
                        out_channels=self.arch["out_channels"][index],
                        which_conv=self.which_conv,
                        which_bn=self.which_bn,
                        activation=self.activation,
                        upsample=(
                            functools.partial(F.interpolate, scale_factor=2)
                            if self.arch["upsample"][index]
                            else None
                        ),
                    )
                ]
            ]

            # If attention on this block, attach it to the end
            if self.arch["attention"][self.arch["resolution"][index]]:
                print(
                    "Adding attention layer in G at resolution %d" % self.arch["resolution"][index]
                )
                self.blocks[-1] += [Attention(self.arch["out_channels"][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(
            bn(self.arch["out_channels"][-1], cross_replica=self.cross_replica, mybn=self.mybn),
            self.activation,
            self.which_conv(self.arch["out_channels"][-1], 3),
        )

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print("Using fp16 adam in G...")
            import utils

            self.optim = utils.Adam16(
                params=self.parameters(),
                lr=self.lr,
                betas=(self.B1, self.B2),
                weight_decay=0,
                eps=self.adam_eps,
            )
        else:
            if optimizer == "Adam":
                self.optim = optim.Adam(
                    params=self.parameters(),
                    lr=self.lr,
                    betas=(self.B1, self.B2),
                    weight_decay=0,
                    eps=self.adam_eps,
                )
            elif optimizer == "SGD":
                self.optim = optim.SGD(
                    params=self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0
                )
            else:
                raise ValueError("optim has to be Adam or SGD, but got {}".format(optimizer))

        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    # print('Init style not recognized...')
                    pass
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print("Param count for Gs initialized parameters: %d" % self.param_count)

    def reset_in_init(self):
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                if isinstance(block, GBlock):
                    block.reset_in_init()

    def get_params(self, index=0, update_embed=False):
        if index == 0:
            for param in self.linear.parameters():
                yield param
            if update_embed:
                for param in self.shared.parameters():
                    yield param
        elif index < len(self.blocks) + 1:
            for param in self.blocks[index - 1].parameters():
                yield param
        elif index == len(self.blocks) + 1:
            for param in self.output_layer.parameters():
                yield param
        else:
            raise ValueError("Index out of range")

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y, use_in=False):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index], use_in)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))

    def forward_verbose(self, z, y, use_in=False):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        print("z:", z.shape)
        # First linear layer
        h = self.linear(z)
        print("h:", h.shape)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            print("index", index, ":", h.shape)
            for block in blocklist:
                h = block(h, ys[index], use_in)

        print("result:", h.shape)
        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))

    def forward_from(self, z, y, num_layer, h, use_in=False):
        if num_layer == 0:
            if len(h.shape) == 1:
                h = h[None, ...]
            return self.forward(h, y)
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        if num_layer == 0:
            # First linear layer
            h = self.linear(z)
            # Reshape
            h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            if index < num_layer:
                continue

            for block in blocklist:
                h = block(h, ys[index], use_in)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))

    def forward_from_to(self, z, y, num_layer_from, num_layer_to, h, use_in=False):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            if index < num_layer_from:
                continue

            if index == num_layer_to:
                return h

            for block in blocklist:
                h = block(h, ys[index], use_in)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))

    def forward_from_with_cp(self, z, cp, num_layer, h, use_in=False):
        zs = torch.split(z, self.z_chunk_size, 1)
        z = zs[0]
        ys = [torch.cat([c, item], 1) for c, item in zip(cp, zs[1:])]

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            if index < num_layer:
                continue

            for block in blocklist:
                h = block(h, ys[index], use_in)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))

    def forward_to(self, z, y, num_layer, use_in=False):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            if index == num_layer:
                return h

            for block in blocklist:
                h = block(h, ys[index], use_in)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))

    def forward_from_multi(self, z, y, num_layer, h, use_in=False):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        hs = [h]
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            if index < num_layer:
                continue

            for block in blocklist:
                h = block(h, ys[index], use_in)
                hs.append(h)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h)), hs

    def forward_to_multi(self, z, y, num_layer, use_in=False):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        hs = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            hs.append(h)
            if index == num_layer:
                return hs

            for block in blocklist:
                h = block(h, ys[index], use_in)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention="64", ksize="333333", dilation="111111"):
    arch = {}
    arch[256] = {
        "in_channels": [3] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
        "out_channels": [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
        "downsample": [True] * 6 + [False],
        "resolution": [128, 64, 32, 16, 8, 4, 4],
        "attention": {
            2**i: 2**i in [int(item) for item in attention.split("_")] for i in range(2, 8)
        },
    }
    arch[128] = {
        "in_channels": [3] + [ch * item for item in [1, 2, 4, 8, 16]],
        "out_channels": [item * ch for item in [1, 2, 4, 8, 16, 16]],
        "downsample": [True] * 5 + [False],
        "resolution": [64, 32, 16, 8, 4, 4],
        "attention": {
            2**i: 2**i in [int(item) for item in attention.split("_")] for i in range(2, 8)
        },
    }
    arch[64] = {
        "in_channels": [3] + [ch * item for item in [1, 2, 4, 8]],
        "out_channels": [item * ch for item in [1, 2, 4, 8, 16]],
        "downsample": [True] * 4 + [False],
        "resolution": [32, 16, 8, 4, 4],
        "attention": {
            2**i: 2**i in [int(item) for item in attention.split("_")] for i in range(2, 7)
        },
    }
    arch[32] = {
        "in_channels": [3] + [item * ch for item in [4, 4, 4]],
        "out_channels": [item * ch for item in [4, 4, 4, 4]],
        "downsample": [True, True, False, False],
        "resolution": [16, 16, 16, 16],
        "attention": {
            2**i: 2**i in [int(item) for item in attention.split("_")] for i in range(2, 6)
        },
    }
    return arch


class Discriminator(nn.Module):
    def __init__(
        self,
        D_ch=64,
        D_wide=True,
        resolution=128,
        D_kernel_size=3,
        D_attn="64",
        n_classes=1000,
        num_D_SVs=1,
        num_D_SV_itrs=1,
        D_activation=nn.ReLU(inplace=False),
        D_lr=2e-4,
        D_B1=0.0,
        D_B2=0.999,
        adam_eps=1e-8,
        SN_eps=1e-12,
        output_dim=1,
        D_mixed_precision=False,
        D_fp16=False,
        D_init="ortho",
        skip_init=False,
        D_param="SN",
        **kwargs,
    ):
        super(Discriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == "SN":
            self.which_conv = functools.partial(
                SNConv2d,
                kernel_size=3,
                padding=1,
                num_svs=num_D_SVs,
                num_itrs=num_D_SV_itrs,
                eps=self.SN_eps,
            )
            self.which_linear = functools.partial(
                SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps
            )
            self.which_embedding = functools.partial(
                SNEmbedding, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps
            )
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            self.which_embedding = nn.Embedding

        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch["out_channels"])):
            self.blocks += [
                [
                    DBlock(
                        in_channels=self.arch["in_channels"][index],
                        out_channels=self.arch["out_channels"][index],
                        which_conv=self.which_conv,
                        wide=self.D_wide,
                        activation=self.activation,
                        preactivation=(index > 0),
                        downsample=(nn.AvgPool2d(2) if self.arch["downsample"][index] else None),
                    )
                ]
            ]
            # If attention on this block, attach it to the end
            if self.arch["attention"][self.arch["resolution"][index]]:
                print(
                    "Adding attention layer in D at resolution %d" % self.arch["resolution"][index]
                )
                self.blocks[-1] += [Attention(self.arch["out_channels"][index], self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch["out_channels"][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch["out_channels"][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print("Using fp16 adam in D...")
            import utils

            self.optim = utils.Adam16(
                params=self.parameters(),
                lr=self.lr,
                betas=(self.B1, self.B2),
                weight_decay=0,
                eps=self.adam_eps,
            )
        else:
            self.optim = optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                betas=(self.B1, self.B2),
                weight_decay=0,
                eps=self.adam_eps,
            )
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    pass
                    # print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print("Param count for Ds initialized parameters: %d" % self.param_count)

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        h_list = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
                h_list.append(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        h_list.append(h)
        # Get initial class-unconditional output
        out = self.linear(h)
        if y is not None:
            # Get projection of final featureset onto class vectors and add to evidence
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)

        return out, h_list


# ---- from common.py (Colorizer only; VGG16Perceptual dropped, needs external pickle checkpoint) ----


class Colorizer(nn.Module):
    def __init__(
        self,
        config,
        path_ckpt_g,
        norm_type,
        activation="relu",
        id_mid_layer=2,
        fix_g=False,
        load_g=True,
        init_e=None,
        use_attention=False,
        use_res=True,
        dim_f=16,
    ):
        super().__init__()

        self.id_mid_layer = id_mid_layer
        self.use_attention = use_attention
        self.use_res = use_res

        if not use_res:
            print("Warning: without residual path")

        if dim_f == 64:
            self.E = EncoderF64_Res(
                norm=norm_type, activation=activation, init=init_e, use_att=use_attention
            )
            self.id_mid_layer = 4
        elif dim_f == 32:
            self.E = EncoderF32_Res(
                norm=norm_type, activation=activation, init=init_e, use_att=use_attention
            )
            self.id_mid_layer = 3
        elif dim_f == 16:
            self.E = EncoderF_Res(
                norm=norm_type,
                activation=activation,
                init=init_e,
                use_res=use_res,
                use_att=use_attention,
            )
            self.id_mid_layer = 2
        elif dim_f == 8:
            self.E = EncoderF8_Res(
                norm=norm_type, activation=activation, init=init_e, use_att=use_attention
            )
            self.id_mid_layer = 1
        elif dim_f == 1:
            self.E = EncoderZ_Res(
                norm=norm_type, activation=activation, init=init_e, use_att=use_attention
            )
            self.id_mid_layer = 0
        else:
            raise Exception("In valid dim_f")

        self.G = Generator(**config)
        if load_g:
            print("Use pretraind G")
            self.G.load_state_dict(torch.load(path_ckpt_g, map_location="cpu"), strict=False)
        self.fix_g = fix_g
        if fix_g:
            for p in self.G.parameters():
                p.requires_grad = False

    def forward(self, x_gray, c, z):
        f = self.E(x_gray, self.G.shared(c))
        output = self.G.forward_from(z, self.G.shared(c), self.id_mid_layer, f)
        return output

    def forward_with_c(self, x_gray, c_embd, z):
        f = self.E(x_gray, c_embd)
        output = self.G.forward_from(z, c_embd, self.id_mid_layer, f)
        return output

    def forward_with_c2(self, x_gray, c_embds, z):
        f = self.E(x_gray, c_embds[0])
        output = self.G.forward_from(z, c_embds[1], self.id_mid_layer, f)
        return output

    def forward_with_cp(self, x_gray, c_embds, z):
        c_embds_E = c_embds[:5]
        c_embds_G = c_embds[5:]
        f = self.E.forward_with_cp(x_gray, c_embds_E)
        output = self.G.forward_from_with_cp(z, c_embds_G, self.id_mid_layer, f)
        return output

    def train(self, mode=True):
        if self.fix_g:
            self.E.train(mode)
        else:
            super().train(mode)


def build_bigcolor():
    colorizer = Colorizer(
        config={
            "G_ch": 96,
            "resolution": 256,
            "dim_z": 128,
            "n_classes": 1000,
            "skip_init": True,
            "no_optim": True,
        },
        path_ckpt_g=None,
        norm_type="batch",
        load_g=False,
        dim_f=16,
    )
    colorizer.eval()  # NOTE: does not return self (real upstream behavior); see header
    return colorizer


def example_input_bigcolor():
    x_gray = torch.randn(1, 1, 256, 256)
    c = torch.zeros(1, dtype=torch.long)
    z = torch.randn(1, 128)
    return (x_gray, c, z)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BigColor", "build_bigcolor", "example_input_bigcolor", 2022, "vendored"),
]
