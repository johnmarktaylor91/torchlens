# SOURCE: vendored from HuiZeng/Image-Adaptive-3DLUT @ master
# https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/master/models.py
# https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/master/trilinear_cpp/src/trilinear.cpp
# https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/master/trilinear_cpp/src/trilinear.h
#
# Zeng, Cai, Li, Cao, Zhang 2020 "Learning Image-Adaptive 3D Lookup Tables for High
# Performance Photo Enhancement in Real-Time" (TPAMI 2020 / CVPR-style backbone,
# arXiv:2009.14468). CVPRW/TPAMI photo-enhancement model: a lightweight CNN
# "Classifier" (real code, unmodified) regresses per-image blend weights over a bank
# of learnable 3D color lookup tables (`Generator3DLUT_zero`, real code, unmodified);
# the LUTs are applied to the image via a custom `TrilinearInterpolation`
# `torch.autograd.Function` that calls into a small pybind11 C++ extension
# (`trilinear.cpp`/`trilinear.h`, real code, unmodified CPU forward/backward path --
# the CUDA path from `trilinear_cuda.cu` is dropped since this is a CPU random-init
# probe). `Classifier`, `Generator3DLUT_zero`, and `TrilinearInterpolation` are copied
# verbatim from the two source files above; only the `from trilinear_c._ext import
# trilinear` import is replaced with an equivalent lazy `torch.utils.cpp_extension.load`
# call (the repo's own `trilinear_cpp/setup.py` mechanism, applied inline instead of a
# separate pip install step -- both are the standard PyTorch C++-extension build path,
# no non-base package). `Generator3DLUT_identity` (loads `IdentityLUT33.txt`, a 36K-line
# data asset) is not used here; `Generator3DLUT_zero` (zero-init, no external asset) is
# used for the build function instead -- both classes are otherwise identical vendored
# real code, and a random-init probe does not need pretrained/identity LUT contents.
"""Image-Adaptive 3D LUT: a CNN classifier predicts per-image blend weights over a bank
of learnable 3D color lookup tables, applied via trilinear interpolation (Zeng et al.
2020)."""

import os

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.cpp_extension import load

MENAGERIE_ZOO = "vendored-pytorch"

_TRILINEAR_SRC_DIR = os.path.join(os.path.dirname(__file__), "_trilinear_ext")
_trilinear_module = None


def _trilinear_cpp_source():
    # --- vendored from trilinear_cpp/src/trilinear.h ---
    header = """
#ifndef TRILINEAR_H
#define TRILINEAR_H
#include<torch/extension.h>
int trilinear_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);
int trilinear_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch);
#endif
"""
    # --- vendored from trilinear_cpp/src/trilinear.cpp ---
    source = """
#include "trilinear.h"

void TriLinearForwardCpu(const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);
void TriLinearBackwardCpu(const float* image, const float* image_grad, float* lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);

int trilinear_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    float * lut_flat = lut.data_ptr<float>();
    float * image_flat = image.data_ptr<float>();
    float * output_flat = output.data_ptr<float>();
    auto image_size = image.sizes();
    int channels = image_size[1];
    if (channels != 3) { return 0; }
    TriLinearForwardCpu(lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, channels);
    return 1;
}

int trilinear_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    float * image_grad_flat = image_grad.data_ptr<float>();
    float * image_flat = image.data_ptr<float>();
    float * lut_grad_flat = lut_grad.data_ptr<float>();
    auto image_size = image.sizes();
    int channels = image_size[1];
    if (channels != 3) { return 0; }
    TriLinearBackwardCpu(image_flat, image_grad_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, channels);
    return 1;
}

void TriLinearForwardCpu(const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels)
{
    const int output_size = height * width;
    int index = 0;
    for (index = 0; index < output_size; ++index)
    {
        float r = image[index];
        float g = image[index + width * height];
        float b = image[index + width * height * 2];
        int r_id = floor(r / binsize);
        int g_id = floor(g / binsize);
        int b_id = floor(b / binsize);
        float r_d = fmod(r,binsize) / binsize;
        float g_d = fmod(g,binsize) / binsize;
        float b_d = fmod(b,binsize) / binsize;
        int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;
        output[index] = w000 * lut[id000] + w100 * lut[id100] +
                        w010 * lut[id010] + w110 * lut[id110] +
                        w001 * lut[id001] + w101 * lut[id101] +
                        w011 * lut[id011] + w111 * lut[id111];
        output[index + width * height] = w000 * lut[id000 + shift] + w100 * lut[id100 + shift] +
                                         w010 * lut[id010 + shift] + w110 * lut[id110 + shift] +
                                         w001 * lut[id001 + shift] + w101 * lut[id101 + shift] +
                                         w011 * lut[id011 + shift] + w111 * lut[id111 + shift];
        output[index + width * height * 2] = w000 * lut[id000 + shift * 2] + w100 * lut[id100 + shift * 2] +
                                             w010 * lut[id010 + shift * 2] + w110 * lut[id110 + shift * 2] +
                                             w001 * lut[id001 + shift * 2] + w101 * lut[id101 + shift * 2] +
                                             w011 * lut[id011 + shift * 2] + w111 * lut[id111 + shift * 2];
    }
}

void TriLinearBackwardCpu(const float* image, const float* image_grad, float* lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels)
{
    const int output_size = height * width;
    int index = 0;
    for (index = 0; index < output_size; ++index)
    {
        float r = image[index];
        float g = image[index + width * height];
        float b = image[index + width * height * 2];
        int r_id = floor(r / binsize);
        int g_id = floor(g / binsize);
        int b_id = floor(b / binsize);
        float r_d = fmod(r,binsize) / binsize;
        float g_d = fmod(g,binsize) / binsize;
        float b_d = fmod(b,binsize) / binsize;
        int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;
        lut_grad[id000] += w000 * image_grad[index];
        lut_grad[id100] += w100 * image_grad[index];
        lut_grad[id010] += w010 * image_grad[index];
        lut_grad[id110] += w110 * image_grad[index];
        lut_grad[id001] += w001 * image_grad[index];
        lut_grad[id101] += w101 * image_grad[index];
        lut_grad[id011] += w011 * image_grad[index];
        lut_grad[id111] += w111 * image_grad[index];
        lut_grad[id000 + shift] += w000 * image_grad[index + width * height];
        lut_grad[id100 + shift] += w100 * image_grad[index + width * height];
        lut_grad[id010 + shift] += w010 * image_grad[index + width * height];
        lut_grad[id110 + shift] += w110 * image_grad[index + width * height];
        lut_grad[id001 + shift] += w001 * image_grad[index + width * height];
        lut_grad[id101 + shift] += w101 * image_grad[index + width * height];
        lut_grad[id011 + shift] += w011 * image_grad[index + width * height];
        lut_grad[id111 + shift] += w111 * image_grad[index + width * height];
        lut_grad[id000 + shift* 2] += w000 * image_grad[index + width * height * 2];
        lut_grad[id100 + shift* 2] += w100 * image_grad[index + width * height * 2];
        lut_grad[id010 + shift* 2] += w010 * image_grad[index + width * height * 2];
        lut_grad[id110 + shift* 2] += w110 * image_grad[index + width * height * 2];
        lut_grad[id001 + shift* 2] += w001 * image_grad[index + width * height * 2];
        lut_grad[id101 + shift* 2] += w101 * image_grad[index + width * height * 2];
        lut_grad[id011 + shift* 2] += w011 * image_grad[index + width * height * 2];
        lut_grad[id111 + shift* 2] += w111 * image_grad[index + width * height * 2];
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &trilinear_forward, "Trilinear forward");
  m.def("backward", &trilinear_backward, "Trilinear backward");
}
"""
    return header, source


def _get_trilinear_module():
    """Lazily JIT-build the real trilinear.cpp CPU extension (torch.utils.cpp_extension
    is part of base torch; no non-base package is required). Cached on module import."""
    global _trilinear_module
    if _trilinear_module is not None:
        return _trilinear_module
    os.makedirs(_TRILINEAR_SRC_DIR, exist_ok=True)
    header, source = _trilinear_cpp_source()
    header_path = os.path.join(_TRILINEAR_SRC_DIR, "trilinear.h")
    source_path = os.path.join(_TRILINEAR_SRC_DIR, "trilinear.cpp")
    with open(header_path, "w") as f:
        f.write(header)
    with open(source_path, "w") as f:
        f.write(source)
    _trilinear_module = load(name="menagerie_trilinear_3dlut", sources=[source_path], verbose=False)
    return _trilinear_module


# --- vendored from models.py (discriminator_block helper, used by Classifier) ---
def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


# --- vendored from models.py ---
class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear"),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


# --- vendored from models.py (`self.TrilinearInterpolation = TrilinearInterpolation()`
# + `self.TrilinearInterpolation(self.LUT, x)` becomes `TrilinearInterpolation.apply`,
# the modern torch.autograd.Function call convention paired with the @staticmethod
# rewrite above -- same op, current-API invocation) ---
class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        self.LUT = nn.Parameter(torch.tensor(self.LUT))

    def forward(self, x):
        return TrilinearInterpolation.apply(self.LUT, x)


# --- vendored from models.py (import replaced: `from trilinear_c._ext import trilinear`
# becomes the lazy `_get_trilinear_module()` JIT build above; `.data<float>()` calls in
# the C++ source updated to `.data_ptr<float>()` for the installed torch ABI; CPU-only
# forward/backward path, matching the `else:` branch of the real `forward`/`backward`;
# the original repo predates PyTorch's static-method autograd.Function requirement --
# `forward(self, ...)` / `backward(self, ...)` instance methods raise
# "Legacy autograd function with non-static forward method is deprecated" on this torch
# version, so the same algorithm is expressed with `@staticmethod` + `ctx` instead of
# `self`, which is the minimal current-torch-API-compatible rewrite of the identical
# computation, not an architectural change) ---
class TrilinearInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, LUT, x):
        trilinear = _get_trilinear_module()

        x = x.contiguous()
        output = x.new(x.size())
        dim = LUT.size()[-1]
        shift = dim**3
        binsize = 1.0001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        ctx.x = x
        ctx.LUT = LUT
        ctx.dim = dim
        ctx.shift = shift
        ctx.binsize = binsize
        ctx.W = W
        ctx.H = H
        ctx.batch = batch

        trilinear.forward(LUT, x, output, dim, shift, binsize, W, H, batch)

        return output

    @staticmethod
    def backward(ctx, grad_x):
        trilinear = _get_trilinear_module()

        grad_LUT = torch.zeros(3, ctx.dim, ctx.dim, ctx.dim, dtype=torch.float)
        trilinear.backward(
            ctx.x, grad_x, grad_LUT, ctx.dim, ctx.shift, ctx.binsize, ctx.W, ctx.H, ctx.batch
        )

        return grad_LUT, None


# --- vendored from models.py (full Image-Adaptive-3DLUT inference pipeline: classifier
# predicts 3 blend weights, each weight scales one learnable 3D LUT, the weighted-sum
# LUT is applied to the input image via trilinear interpolation) ---
class ImageAdaptive3DLUT(nn.Module):
    def __init__(self, lut_dim=9):
        super(ImageAdaptive3DLUT, self).__init__()
        self.classifier = Classifier()
        self.LUT0 = Generator3DLUT_zero(dim=lut_dim)
        self.LUT1 = Generator3DLUT_zero(dim=lut_dim)
        self.LUT2 = Generator3DLUT_zero(dim=lut_dim)

    def forward(self, img):
        pred = self.classifier(img).squeeze()
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        gen_a0 = self.LUT0(img)
        gen_a1 = self.LUT1(img)
        gen_a2 = self.LUT2(img)
        weights_norm = torch.mean(pred**2)
        combine_a = img.new(img.size())
        for b in range(img.size(0)):
            combine_a[b, ...] = (
                pred[b, 0] * gen_a0[b, ...]
                + pred[b, 1] * gen_a1[b, ...]
                + pred[b, 2] * gen_a2[b, ...]
            )
        return combine_a, weights_norm


def build_neural_3dlut():
    # small LUT grid (dim=9 instead of the paper's dim=33) for a fast random-init trace;
    # Generator3DLUT_zero/Classifier/TrilinearInterpolation are the real vendored classes.
    return ImageAdaptive3DLUT(lut_dim=9)


def example_input_neural_3dlut():
    torch.manual_seed(0)
    # trilinear.cpp requires channel-first RGB in [0, 1); a 32x32 random-init probe keeps
    # the JIT-compiled CPU extension fast.
    return (torch.rand(1, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    ("Image-Adaptive-3DLUT", "build_neural_3dlut", "example_input_neural_3dlut", 2020, "vendored"),
]
