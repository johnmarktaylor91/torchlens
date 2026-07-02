# SOURCE: vendored from tflsguoyu/materialgan @ main
# (archived upstream successor: tflsguoyu/svbrdf-diff-renderer)
#
# MaterialGAN (Guo, Smith, Deschaintre, Yan, Kalantari, Hasan, Ramamoorthi.
# 2020, SIGGRAPH Asia, "MaterialGAN: Reflectance Capture using a
# Generative SVBRDF Model") -- the generator is a StyleGAN2 network
# (Karras et al. 2020) trained on synthetic SVBRDF material maps and
# repurposed as a differentiable material prior for gradient-based
# reflectance-capture optimization. The generator architecture itself is
# vendored VERBATIM from the official repo's PyTorch StyleGAN2 generator
# implementation, used by MaterialGAN with `image_channels=9` (2-channel
# normal xy + 3-channel diffuse + 1-channel roughness + 3-channel specular,
# packed as a 9-channel "SVBRDF map" image) instead of the 3-channel RGB
# used by the vanilla StyleGAN2 configs in the same repo -- confirmed via
# `higan/models/model_settings.py` `MODEL_POOL['svbrdf']`
# (`gan_type='stylegan2'`, `resolution=256`, `g_architecture_type='skip'`)
# and `higan/models/base_generator.py`
# (`self.image_channels = getattr(self, 'image_channels', 9)`).
#
# Real source file (copied verbatim below, only the `global_var` cross-module
# noise-buffer singleton was localized into this single-file staging module
# as `_MaterialGANGlobalNoise` / `_MATERIALGAN_GLOBAL_NOISE`, matching the
# real repo's `src/global_var.py` + `src/applications.py`
# `global_var.init_global_noise(resolution, "random")` usage pattern that
# pre-populates the per-layer noise buffers consumed when
# `randomize_noise=False` (MaterialGAN's `STYLEGAN2_RANDOMIZE_NOISE = False`
# setting in `model_settings.py`)):
#   https://raw.githubusercontent.com/tflsguoyu/materialgan/main/higan/models/stylegan2_generator_network.py
#
# MENAGERIE_ZOO = "vendored-pytorch"
#
# Original module docstring (from stylegan2_generator_network.py):
#   Contains the implementation of generator described in StyleGAN2.
#   Different from the official tensorflow version in folder
#   `stylegan2_tf_official`, this is a simple pytorch version which only
#   contains the generator part. This class is specially used for inference.
#   NOTE: Compared to that of StyleGAN, the generator in StyleGAN2 mainly
#   introduces style demodulation, adds skip connections, increases model
#   size, and disables progressive growth. This script ONLY supports config F
#   in the original paper. For more details, please check the original
#   paper: https://arxiv.org/pdf/1912.04958.pdf

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

__all__ = ["StyleGAN2GeneratorNet"]


class _MaterialGANGlobalNoise:
    """Localized stand-in for the real repo's module-level `global_var` noise
    singleton (`src/global_var.py`). The real `StyleGAN2GeneratorNet.forward`
    (via `ModulateConvBlock.forward`) reads pre-generated per-layer noise
    tensors from this shared state when `randomize_noise=False`, exactly as
    `global_var.noises[global_var.noise_idx]` does in the vendored source."""

    def __init__(self):
        self.noises = []
        self.noise_idx = 0

    def init_random(self, resolution, init_res=4):
        """Mirrors `global_var.init_global_noise(resolution, "random")`,
        sized for the requested (small) generator resolution rather than the
        full 256px MaterialGAN checkpoint resolution."""
        noises = []
        res = init_res
        # First block at init_res gets one noise buffer (`layer0`); every
        # subsequent resolution doubling gets two (`layer{2k-1}`, `layer{2k}`)
        # -- matches the real `SynthesisModule.__init__` layer construction.
        noises.append(torch.randn(1, 1, res, res, dtype=torch.float32))
        res *= 2
        while res <= resolution:
            noises.append(torch.randn(1, 1, res, res, dtype=torch.float32))
            noises.append(torch.randn(1, 1, res, res, dtype=torch.float32))
            res *= 2
        self.noises = noises
        self.noise_idx = 0


_MATERIALGAN_GLOBAL_NOISE = _MaterialGANGlobalNoise()

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Architectures allowed.
_ARCHITECTURES_ALLOWED = ["resnet", "skip", "origin"]


class StyleGAN2GeneratorNet(nn.Module):
    """Defines the generator network in StyleGAN2.

    NOTE: the generated images are with `RGB` color channels and range [-1, 1].
    """

    def __init__(
        self,
        resolution=1024,
        z_space_dim=512,
        w_space_dim=512,
        image_channels=3,
        architecture_type="skip",
        fused_modulate=True,
        truncation_psi=0.5,
        truncation_layers=18,
        randomize_noise=False,
        num_mapping_layers=8,
        fmaps_base=32 << 10,
        fmaps_max=512,
    ):
        """Initializes the generator with basic settings.

        Args:
          resolution: The resolution of the output image. (default: 1024)
          z_space_dim: The dimension of the initial latent space. (default: 512)
          w_space_dim: The dimension of the disentangled latent vectors, w.
            (default: 512)
          image_channels: Number of channels of output image. (default: 3)
          architecture_type: Defines the architecture type. (default: `resnet`)
          fused_modulate: Whether to fuse `style_modulate` and `conv2d` together.
            (default: True)
          truncation_psi: Style strength multiplier for the truncation trick.
            `None` or `1.0` indicates no truncation. (default: 0.5)
          truncation_layers: Number of layers for which to apply the truncation
            trick. `None` or `0` indicates no truncation. (default: 18)
          randomize_noise: Whether to add random noise for each convolutional layer.
            (default: False)
          num_mapping_layers: Number of fully-connected layers to map Z space to W
            space. (default: 8)
          fmaps_base: Base factor to compute number of feature maps for each layer.
            (default: 32 << 10)
          fmaps_max: Maximum number of feature maps in each layer. (default: 512)

        Raises:
          ValueError: If the input `resolution` is not supported, or
            `architecture_type` is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(
                f"Invalid resolution: {resolution}!\nResolutions allowed: {_RESOLUTIONS_ALLOWED}."
            )
        if architecture_type not in _ARCHITECTURES_ALLOWED:
            raise ValueError(
                f"Invalid fused-scale option: {architecture_type}!\n"
                f"Architectures allowed: {_ARCHITECTURES_ALLOWED}."
            )

        self.init_res = _INIT_RES
        self.resolution = resolution
        self.z_space_dim = z_space_dim
        self.w_space_dim = w_space_dim
        self.image_channels = image_channels
        self.architecture_type = architecture_type
        self.fused_modulate = fused_modulate
        self.truncation_psi = truncation_psi
        self.truncation_layers = truncation_layers
        self.randomize_noise = randomize_noise
        self.num_mapping_layers = num_mapping_layers
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2

        self.mapping = MappingModule(
            input_space_dim=self.z_space_dim,
            hidden_space_dim=self.fmaps_max,
            final_space_dim=self.w_space_dim,
            num_layers=self.num_mapping_layers,
        )
        self.truncation = TruncationModule(
            num_layers=self.num_layers,
            w_space_dim=self.w_space_dim,
            truncation_psi=self.truncation_psi,
            truncation_layers=self.truncation_layers,
        )
        self.synthesis = SynthesisModule(
            init_resolution=self.init_res,
            resolution=self.resolution,
            w_space_dim=self.w_space_dim,
            image_channels=self.image_channels,
            architecture_type=self.architecture_type,
            fused_modulate=self.fused_modulate,
            randomize_noise=self.randomize_noise,
            fmaps_base=self.fmaps_base,
            fmaps_max=self.fmaps_max,
        )

        self.pth_to_tf_var_mapping = {}
        for key, val in self.mapping.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f"mapping.{key}"] = val
        for key, val in self.truncation.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f"truncation.{key}"] = val
        for key, val in self.synthesis.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f"synthesis.{key}"] = val

    def forward(self, z):
        w = self.mapping(z)
        w = self.truncation(w)
        x = self.synthesis(w)
        return x


class MappingModule(nn.Sequential):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence.
    """

    def __init__(
        self,
        input_space_dim=512,
        hidden_space_dim=512,
        final_space_dim=512,
        num_layers=8,
        normalize_input=True,
    ):
        self.input_space_dim = input_space_dim
        sequence = OrderedDict()

        if normalize_input:
            sequence["normalize"] = PixelNormLayer(dim=1)

        self.pth_to_tf_var_mapping = {}
        for i in range(num_layers):
            in_dim = input_space_dim if i == 0 else hidden_space_dim
            out_dim = final_space_dim if i == (num_layers - 1) else hidden_space_dim
            sequence[f"dense{i}"] = DenseBlock(in_dim, out_dim)
            self.pth_to_tf_var_mapping[f"dense{i}.fc.weight"] = f"Dense{i}/weight"
            self.pth_to_tf_var_mapping[f"dense{i}.bias"] = f"Dense{i}/bias"

        super().__init__(sequence)

    def forward(self, z):
        if not (len(z.shape) == 2 and z.shape[1] == self.input_space_dim):
            raise ValueError(
                f"The input tensor should be with shape [batch_size, "
                f"input_dim], where `input_dim` equals to "
                f"{self.input_space_dim}!\n"
                f"But {z.shape} received!"
            )
        return super().forward(z)


class TruncationModule(nn.Module):
    """Implements the truncation module."""

    def __init__(self, num_layers, w_space_dim=512, truncation_psi=0.7, truncation_layers=8):
        super().__init__()

        self.num_layers = num_layers
        self.w_space_dim = w_space_dim
        if truncation_psi is not None and truncation_layers is not None:
            self.use_truncation = True
        else:
            self.use_truncation = False
            truncation_psi = 1.0
            truncation_layers = 0

        self.register_buffer("w_avg", torch.zeros(w_space_dim))
        self.pth_to_tf_var_mapping = {"w_avg": "dlatent_avg"}

        layer_idx = np.arange(self.num_layers).reshape(1, self.num_layers, 1)
        coefs = np.ones_like(layer_idx, dtype=np.float32)
        coefs[layer_idx < truncation_layers] *= truncation_psi
        self.register_buffer("truncation", torch.from_numpy(coefs))

    def forward(self, w):
        if len(w.shape) == 2:
            assert w.shape[1] == self.w_space_dim
            w = w.view(-1, 1, self.w_space_dim).repeat(1, self.num_layers, 1)
        assert (
            len(w.shape) == 3 and w.shape[1] == self.num_layers and w.shape[2] == self.w_space_dim
        )
        if self.use_truncation:
            w_avg = self.w_avg.view(1, 1, self.w_space_dim)
            w = w_avg + (w - w_avg) * self.truncation
        return w


class SynthesisModule(nn.Module):
    """Implements the image synthesis module.

    Basically, this module executes several convolutional layers in sequence.
    """

    def __init__(
        self,
        init_resolution=4,
        resolution=1024,
        w_space_dim=512,
        image_channels=3,
        architecture_type="skip",
        fused_modulate=True,
        randomize_noise=False,
        fmaps_base=32 << 10,
        fmaps_max=512,
    ):
        super().__init__()

        self.init_res = init_resolution
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.w_space_dim = w_space_dim
        self.architecture_type = architecture_type
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        # pylint: disable=line-too-long
        self.pth_to_tf_var_mapping = {}
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2**res_log2
            block_idx = res_log2 - self.init_res_log2

            # First convolution layer for each resolution.
            if res == self.init_res:
                self.add_module(
                    "early_layer",
                    InputBlock(init_resolution=self.init_res, channels=self.get_nf(res)),
                )
                self.pth_to_tf_var_mapping["early_layer.const"] = f"{res}x{res}/Const/const"
            else:
                self.add_module(
                    f"layer{2 * block_idx - 1}",
                    ModulateConvBlock(
                        resolution=res,
                        in_channels=self.get_nf(res // 2),
                        out_channels=self.get_nf(res),
                        w_space_dim=self.w_space_dim,
                        scale_factor=2,
                        fused_modulate=fused_modulate,
                        demodulate=True,
                        add_noise=True,
                        randomize_noise=randomize_noise,
                    ),
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx - 1}.weight"] = (
                    f"{res}x{res}/Conv0_up/weight"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx - 1}.bias"] = (
                    f"{res}x{res}/Conv0_up/bias"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx - 1}.style.fc.weight"] = (
                    f"{res}x{res}/Conv0_up/mod_weight"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx - 1}.style.bias"] = (
                    f"{res}x{res}/Conv0_up/mod_bias"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx - 1}.noise_strength"] = (
                    f"{res}x{res}/Conv0_up/noise_strength"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx - 1}.noise"] = (
                    f"noise{2 * block_idx - 1}"
                )

                if self.architecture_type == "resnet":
                    self.add_module(
                        f"skip_layer{block_idx - 1}",
                        ConvBlock(
                            in_channels=self.get_nf(res // 2),
                            out_channels=self.get_nf(res),
                            kernel_size=1,
                            scale_factor=2,
                            add_bias=False,
                            activation_type="linear",
                        ),
                    )
                    self.pth_to_tf_var_mapping[f"skip_layer{block_idx - 1}.weight"] = (
                        f"{res}x{res}/Skip/weight"
                    )

            # Second convolution layer for each resolution.
            self.add_module(
                f"layer{2 * block_idx}",
                ModulateConvBlock(
                    resolution=res,
                    in_channels=self.get_nf(res),
                    out_channels=self.get_nf(res),
                    w_space_dim=self.w_space_dim,
                    fused_modulate=fused_modulate,
                    demodulate=True,
                    add_noise=True,
                    randomize_noise=randomize_noise,
                ),
            )
            if res == self.init_res:
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.weight"] = (
                    f"{res}x{res}/Conv/weight"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.bias"] = f"{res}x{res}/Conv/bias"
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.style.fc.weight"] = (
                    f"{res}x{res}/Conv/mod_weight"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.style.bias"] = (
                    f"{res}x{res}/Conv/mod_bias"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.noise_strength"] = (
                    f"{res}x{res}/Conv/noise_strength"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.noise"] = f"noise{2 * block_idx}"
            else:
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.weight"] = (
                    f"{res}x{res}/Conv1/weight"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.bias"] = f"{res}x{res}/Conv1/bias"
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.style.fc.weight"] = (
                    f"{res}x{res}/Conv1/mod_weight"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.style.bias"] = (
                    f"{res}x{res}/Conv1/mod_bias"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.noise_strength"] = (
                    f"{res}x{res}/Conv1/noise_strength"
                )
                self.pth_to_tf_var_mapping[f"layer{2 * block_idx}.noise"] = f"noise{2 * block_idx}"

            # Output convolution layer for each resolution (if needed).
            if res_log2 == self.final_res_log2 or self.architecture_type == "skip":
                self.add_module(
                    f"output{block_idx}",
                    ModulateConvBlock(
                        resolution=res,
                        in_channels=self.get_nf(res),
                        out_channels=image_channels,
                        w_space_dim=self.w_space_dim,
                        kernel_size=1,
                        fused_modulate=fused_modulate,
                        demodulate=False,
                        add_noise=False,
                        activation_type="linear",
                    ),
                )
                self.pth_to_tf_var_mapping[f"output{block_idx}.weight"] = (
                    f"{res}x{res}/ToRGB/weight"
                )
                self.pth_to_tf_var_mapping[f"output{block_idx}.bias"] = f"{res}x{res}/ToRGB/bias"
                self.pth_to_tf_var_mapping[f"output{block_idx}.style.fc.weight"] = (
                    f"{res}x{res}/ToRGB/mod_weight"
                )
                self.pth_to_tf_var_mapping[f"output{block_idx}.style.bias"] = (
                    f"{res}x{res}/ToRGB/mod_bias"
                )
        if self.architecture_type == "skip":
            self.upsample = UpsamplingLayer()
        # pylint: enable=line-too-long

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, w):
        if not (
            len(w.shape) == 3 and w.shape[1] == self.num_layers and w.shape[2] == self.w_space_dim
        ):
            raise ValueError(
                f"The input tensor should be with shape [batch_size, "
                f"num_layers, w_space_dim], where "
                f"`num_layers` equals to {self.num_layers}, and "
                f"`w_space_dim` equals to {self.w_space_dim}!\n"
                f"But {w.shape} received!"
            )

        x = self.early_layer(w)

        if self.architecture_type == "origin":
            for layer_idx in range(self.num_layers - 1):
                x = self.__getattr__(f"layer{layer_idx}")(x, w[:, layer_idx])
            image = self.__getattr__(f"output{layer_idx // 2}")(x, w[:, layer_idx + 1])

        elif self.architecture_type == "skip":
            for layer_idx in range(self.num_layers - 1):
                x = self.__getattr__(f"layer{layer_idx}")(x, w[:, layer_idx])
                if layer_idx % 2 == 0:
                    if layer_idx == 0:
                        image = self.__getattr__(f"output{layer_idx // 2}")(x, w[:, layer_idx + 1])
                    else:
                        image = self.__getattr__(f"output{layer_idx // 2}")(
                            x, w[:, layer_idx + 1]
                        ) + self.upsample(image)

        elif self.architecture_type == "resnet":
            x = self.layer0(x)
            for layer_idx in range(1, self.num_layers - 1, 2):
                residual = self.__getattr__(f"skip_layer{layer_idx // 2}")(x)
                x = self.__getattr__(f"layer{layer_idx}")(x, w[:, layer_idx])
                x = self.__getattr__(f"layer{layer_idx + 1}")(x, w[:, layer_idx + 1])
                x = (x + residual) / np.sqrt(2.0)
            image = self.__getattr__(f"output{layer_idx // 2 + 1}")(x, w[:, layer_idx + 2])

        return image


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, dim, epsilon=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = epsilon

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)
        return x / norm


class UpsamplingLayer(nn.Module):
    """Implements the upsampling layer.

    This layer can also be used as filtering layer by setting `scale_factor` as 1.
    """

    def __init__(self, scale_factor=2, kernel=(1, 3, 3, 1), extra_padding=0, kernel_gain=None):
        super().__init__()
        assert scale_factor >= 1
        self.scale_factor = scale_factor

        if extra_padding != 0:
            assert scale_factor == 1

        if kernel is None:
            kernel = np.ones((scale_factor), dtype=np.float32)
        else:
            kernel = np.array(kernel, dtype=np.float32)
        assert kernel.ndim == 1
        kernel = np.outer(kernel, kernel)
        kernel = kernel / np.sum(kernel)
        if kernel_gain is None:
            kernel = kernel * (scale_factor**2)
        else:
            assert kernel_gain > 0
            kernel = kernel * (kernel_gain**2)
        assert kernel.ndim == 2
        assert kernel.shape[0] == kernel.shape[1]
        kernel = kernel[:, :, np.newaxis, np.newaxis]
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer("kernel", torch.from_numpy(kernel))
        self.kernel = self.kernel.flip(0, 1)

        self.upsample_padding = (
            0,
            scale_factor - 1,  # Width padding.
            0,
            0,  # Width.
            0,
            scale_factor - 1,  # Height padding.
            0,
            0,  # Height.
            0,
            0,  # Channel.
            0,
            0,
        )  # Batch size.

        padding = kernel.shape[2] - scale_factor + extra_padding
        self.padding = (
            (padding + 1) // 2 + scale_factor - 1,
            padding // 2,
            (padding + 1) // 2 + scale_factor - 1,
            padding // 2,
        )

    def forward(self, x):
        assert len(x.shape) == 4
        channels = x.shape[1]
        if self.scale_factor > 1:
            x = x.view(-1, channels, x.shape[2], 1, x.shape[3], 1)
            x = F.pad(x, self.upsample_padding, mode="constant", value=0)
            x = x.view(-1, channels, x.shape[2] * self.scale_factor, x.shape[4] * self.scale_factor)
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        x = F.pad(x, self.padding, mode="constant", value=0)
        x = F.conv2d(x, self.kernel, stride=1)
        x = x.view(-1, channels, x.shape[2], x.shape[3])
        return x


class InputBlock(nn.Module):
    """Implements the input block.

    Basically, this block starts from a const input, which is with shape
    `(channels, init_resolution, init_resolution)`.
    """

    def __init__(self, init_resolution, channels):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channels, init_resolution, init_resolution))

    def forward(self, w):
        x = self.const.repeat(w.shape[0], 1, 1, 1)
        return x


class ConvBlock(nn.Module):
    """Implements the convolutional block (no style modulation)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        scale_factor=1,
        filtering_kernel=(1, 3, 3, 1),
        weight_gain=1.0,
        lr_multiplier=1.0,
        add_bias=True,
        activation_type="lrelu",
    ):
        """Initializes the class with block settings.

        NOTE: Wscale is used as default.

        Args:
          in_channels: Number of channels of the input tensor.
          out_channels: Number of channels (kernels) of the output tensor.
          kernel_size: Size of the convolutional kernel.
          scale_factor: Scale factor for upsampling. `1` means skip upsampling.
          filtering_kernel: Kernel used for filtering after upsampling.
          weight_gain: Gain factor for weight parameter in convolutional layer.
          lr_multiplier: Learning rate multiplier.
          add_bias: Whether to add bias after convolution.
          activation_type: Type of activation function. Support `linear`, `relu`
            and `lrelu`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()
        assert scale_factor >= 1
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size, in_channels, out_channels))
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_scale = weight_gain / np.sqrt(fan_in)
        self.lr_multiplier = lr_multiplier

        if scale_factor > 1:
            self.filter = UpsamplingLayer(
                scale_factor=1,
                kernel=filtering_kernel,
                extra_padding=scale_factor - kernel_size,
                kernel_gain=scale_factor,
            )
        else:
            assert kernel_size % 2 == 1
            self.conv_padding = kernel_size // 2

        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        if activation_type == "linear":
            self.activate = nn.Identity()
            self.activate_scale = 1.0
        elif activation_type == "relu":
            self.activate = nn.ReLU(inplace=True)
            self.activate_scale = np.sqrt(2.0)
        elif activation_type == "lrelu":
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.activate_scale = np.sqrt(2.0)
        else:
            raise NotImplementedError(f"Not implemented activation function: {activation_type}!")

    def forward(self, x):
        weight = self.weight * self.weight_scale * self.lr_multiplier
        if self.scale_factor > 1:
            weight = weight.flip(0, 1).permute(2, 3, 0, 1)
            x = F.conv_transpose2d(x, weight, stride=self.scale_factor, padding=0)
            x = self.filter(x)
        else:
            weight = weight.permute(3, 2, 0, 1)
            x = F.conv2d(x, weight, stride=1, padding=self.conv_padding)

        if self.add_bias:
            bias = self.bias * self.lr_multiplier
            x = x + bias.view(1, -1, 1, 1)
        x = self.activate(x) * self.activate_scale
        return x


class ModulateConvBlock(nn.Module):
    """Implements the convolutional block with style modulation."""

    def __init__(
        self,
        resolution,
        in_channels,
        out_channels,
        kernel_size=3,
        scale_factor=1,
        filtering_kernel=(1, 3, 3, 1),
        w_space_dim=512,
        fused_modulate=True,
        demodulate=True,
        weight_gain=1.0,
        lr_multiplier=1.0,
        add_bias=True,
        activation_type="lrelu",
        add_noise=True,
        randomize_noise=True,
        epsilon=1e-8,
    ):
        """Initializes the class with block settings.

        NOTE: Wscale is used as default.

        Args:
          resolution: Spatial resolution of current convolution block.
          in_channels: Number of channels of the input tensor.
          out_channels: Number of channels (kernels) of the output tensor.
          kernel_size: Size of the convolutional kernel.
          scale_factor: Scale factor for upsampling. `1` means skip upsampling.
          filtering_kernel: Kernel used for filtering after upsampling.
          w_space_dim: Dimension of disentangled latent space. This is used for
            style modulation.
          fused_modulate: Whether to fuse `style_modulate` and `conv2d` together.
          demodulate: Whether to perform style demodulation.
          weight_gain: Gain factor for weight parameter in convolutional layer.
          lr_multiplier: Learning rate multiplier.
          add_bias: Whether to add bias after convolution.
          activation_type: Type of activation function. Support `linear`, `relu`
            and `lrelu`.
          add_noise: Whether to add noise to spatial feature map.
          randomize_noise: Whether to randomize new noises at runtime.
          epsilon: Small number to avoid `divide by zero`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()

        self.res = resolution
        self.in_c = in_channels
        self.out_c = out_channels
        self.ksize = kernel_size
        self.eps = epsilon

        self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size, in_channels, out_channels))
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_scale = weight_gain / np.sqrt(fan_in)
        self.lr_multiplier = lr_multiplier

        self.scale_factor = scale_factor
        if scale_factor > 1:
            self.filter = UpsamplingLayer(
                scale_factor=1,
                kernel=filtering_kernel,
                extra_padding=scale_factor - kernel_size,
                kernel_gain=scale_factor,
            )
        else:
            assert kernel_size % 2 == 1
            self.conv_padding = kernel_size // 2

        self.w_space_dim = w_space_dim
        self.style = DenseBlock(
            in_channels=w_space_dim,
            out_channels=in_channels,
            lr_multiplier=1.0,
            init_bias=1.0,
            activation_type="linear",
        )

        self.fused_modulate = fused_modulate
        self.demodulate = demodulate

        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        if activation_type == "linear":
            self.activate = nn.Identity()
            self.activate_scale = 1.0
        elif activation_type == "relu":
            self.activate = nn.ReLU(inplace=True)
            self.activate_scale = np.sqrt(2.0)
        elif activation_type == "lrelu":
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.activate_scale = np.sqrt(2.0)
        else:
            raise NotImplementedError(f"Not implemented activation function: {activation_type}!")

        self.add_noise = add_noise
        self.randomize_noise = randomize_noise
        if add_noise:
            self.register_buffer("noise", torch.randn(1, 1, self.res, self.res))
            self.noise_strength = nn.Parameter(torch.zeros(()))

    def forward(self, x, w):
        assert (
            len(x.shape) == 4
            and len(w.shape) == 2
            and w.shape[0] == x.shape[0]
            and w.shape[1] == self.w_space_dim
        )
        batch = x.shape[0]

        weight = self.weight * self.weight_scale * self.lr_multiplier

        # Style modulation.
        style = self.style(w)
        _weight = weight.view(1, self.ksize, self.ksize, self.in_c, self.out_c)
        _weight = _weight * style.view(batch, 1, 1, self.in_c, 1)

        # Style demodulation.
        if self.demodulate:
            _weight_norm = torch.sqrt(torch.sum(_weight**2, dim=[1, 2, 3]) + self.eps)
            _weight = _weight / _weight_norm.view(batch, 1, 1, 1, self.out_c)

        if self.fused_modulate:
            x = x.view(1, batch * self.in_c, x.shape[2], x.shape[3])
            weight = _weight.permute(1, 2, 3, 0, 4).reshape(
                self.ksize, self.ksize, self.in_c, batch * self.out_c
            )
        else:
            x = x * style.view(batch, self.in_c, 1, 1)

        if self.scale_factor > 1:
            weight = weight.flip(0, 1)
            if self.fused_modulate:
                weight = weight.view(self.ksize, self.ksize, self.in_c, batch, self.out_c)
                weight = weight.permute(0, 1, 4, 3, 2)
                weight = weight.reshape(self.ksize, self.ksize, self.out_c, batch * self.in_c)
                weight = weight.permute(3, 2, 0, 1)
            else:
                weight = weight.permute(2, 3, 0, 1)
            x = F.conv_transpose2d(
                x,
                weight,
                stride=self.scale_factor,
                padding=0,
                groups=(batch if self.fused_modulate else 1),
            )
            x = self.filter(x)
        else:
            weight = weight.permute(3, 2, 0, 1)
            x = F.conv2d(
                x,
                weight,
                stride=1,
                padding=self.conv_padding,
                groups=(batch if self.fused_modulate else 1),
            )

        if self.fused_modulate:
            x = x.view(batch, self.out_c, self.res, self.res)
        elif self.demodulate:
            x = x / _weight_norm.view(batch, self.out_c, 1, 1)

        if self.add_noise:
            if self.randomize_noise:
                noise = torch.randn(x.shape[0], 1, self.res, self.res).to(x)
            else:
                noise = _MATERIALGAN_GLOBAL_NOISE.noises[_MATERIALGAN_GLOBAL_NOISE.noise_idx]
                if _MATERIALGAN_GLOBAL_NOISE.noise_idx == len(_MATERIALGAN_GLOBAL_NOISE.noises) - 1:
                    _MATERIALGAN_GLOBAL_NOISE.noise_idx = 0
                else:
                    _MATERIALGAN_GLOBAL_NOISE.noise_idx += 1
                # noise = self.noise
            x = x + noise * self.noise_strength.view(1, 1, 1, 1)

        if self.add_bias:
            bias = self.bias * self.lr_multiplier
            x = x + bias.view(1, -1, 1, 1)
        x = self.activate(x) * self.activate_scale

        return x


class DenseBlock(nn.Module):
    """Implements the dense block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        weight_gain=1.0,
        lr_multiplier=0.01,
        add_bias=True,
        init_bias=0,
        activation_type="lrelu",
    ):
        """Initializes the class with block settings.

        NOTE: Wscale is used as default.

        Args:
          in_channels: Number of channels of the input tensor.
          out_channels: Number of channels of the output tensor.
          weight_gain: Gain factor for weight parameter in dense layer.
          lr_multiplier: Learning rate multiplier.
          add_bias: Whether to add bias after fully-connected operation.
          init_bias: Initialized bias.
          activation_type: Type of activation function. Support `linear`, `relu`
            and `lrelu`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()

        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)

        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        self.init_bias = init_bias

        self.weight_scale = weight_gain / np.sqrt(in_channels)
        self.lr_multiplier = lr_multiplier

        if activation_type == "linear":
            self.activate = nn.Identity()
            self.activate_scale = 1.0
        elif activation_type == "relu":
            self.activate = nn.ReLU(inplace=True)
            self.activate_scale = np.sqrt(2.0)
        elif activation_type == "lrelu":
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.activate_scale = np.sqrt(2.0)
        else:
            raise NotImplementedError(f"Not implemented activation function: {activation_type}!")

    def forward(self, x):
        if len(x.shape) != 2:
            x = x.view(x.shape[0], -1)
        x = self.fc(x) * self.weight_scale * self.lr_multiplier
        if self.add_bias:
            x = x + self.bias.view(1, -1) * self.lr_multiplier + self.init_bias
        x = self.activate(x) * self.activate_scale
        return x


def build_materialgan():
    """Builds the real MaterialGAN generator (StyleGAN2GeneratorNet) at the
    smallest supported resolution (8px, `resolution` must be a power of two
    in {8,16,...,1024} per `_RESOLUTIONS_ALLOWED`) with the real MaterialGAN
    config: `image_channels=9` (packed SVBRDF map), `architecture_type='skip'`,
    `fused_modulate=True` -- matching `MODEL_POOL['svbrdf']` /
    `STYLEGAN2Generator.build()` in the real repo, just with tiny dims
    (z/w space 8, fmaps_max 8) for a fast trace instead of the real
    z_space_dim=512/w_space_dim=512/fmaps_max=512 checkpoint sizing."""
    resolution = 8
    _MATERIALGAN_GLOBAL_NOISE.init_random(resolution, init_res=4)
    return StyleGAN2GeneratorNet(
        resolution=resolution,
        z_space_dim=8,
        w_space_dim=8,
        image_channels=9,
        architecture_type="skip",
        fused_modulate=True,
        truncation_psi=1.0,
        truncation_layers=0,
        randomize_noise=False,
        num_mapping_layers=2,
        fmaps_base=8 << 4,
        fmaps_max=8,
    )


def example_input_materialgan():
    import torch

    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [
    ("MaterialGAN", "build_materialgan", "example_input_materialgan", 2020, "vendored-pytorch"),
]
