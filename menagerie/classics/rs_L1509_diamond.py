# SOURCE: vendored from eloialonso/diamond @ main
# https://github.com/eloialonso/diamond
# Files vendored (near-verbatim, only import paths adjusted to be self-contained):
#   src/models/blocks.py            -> block primitives (GroupNorm, AdaGroupNorm, SelfAttention2d,
#                                       FourierFeatures, Downsample/Upsample, ResBlock(s), UNet)
#   src/models/diffusion/inner_model.py -> InnerModel (the diffusion denoiser network; this IS the
#                                       DIAMOND architectural contribution: an action/noise-conditioned
#                                       UNet operating in pixel space for world-model rollout)
#
# The full DIAMOND agent also trains a separate ActorCritic (imagination rollout) and a
# RewEndModel (reward/termination head); those require env/coroutine machinery (envs.py,
# coroutines/env_loop.py) that is training-infrastructure, not architecture. InnerModel is the
# single self-contained network that captures the paper's core contribution and traces cleanly
# on its own.
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# ---------------------------------------------------------------------------
# src/models/blocks.py (verbatim)
# ---------------------------------------------------------------------------

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)


class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [t for t in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 1
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SmallResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.f = nn.Sequential(
            GroupNorm(in_channels), nn.SiLU(inplace=True), Conv3x3(in_channels, out_channels)
        )
        self.skip_projection = (
            nn.Identity() if in_channels == out_channels else Conv1x1(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.skip_projection(x) + self.f(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        x = self.attn(x)
        return x


class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, cond_channels, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(self, x: Tensor, cond: Tensor, to_cat: Optional[List[Tensor]] = None) -> Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs


class UNet(nn.Module):
    def __init__(
        self, cond_channels: int, depths: List[int], channels: List[int], attn_depths: List[int]
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )

        downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        upsamples = [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        *_, h, w = x.size()
        n = self._num_down
        padding_h = math.ceil(h / 2**n) * 2**n - h
        padding_w = math.ceil(w / 2**n) * 2**n - w
        x = F.pad(x, (0, padding_w, 0, padding_h))

        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, cond)

        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, skip[::-1])
            u_outputs.append((x_up, *block_outputs))

        x = x[..., :h, :w]
        return x, d_outputs, u_outputs


# ---------------------------------------------------------------------------
# src/models/diffusion/inner_model.py (verbatim)
# ---------------------------------------------------------------------------


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.act_emb = nn.Sequential(
            nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
            nn.Flatten(),  # b t e -> b (t e)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act))
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------

_IMG_CH = 3
_NUM_STEPS_COND = 2
_COND_CH = 16
_NUM_ACTIONS = 4


class DiamondInnerModelWrapper(nn.Module):
    """Thin wrapper fixing the noise/action conditioning tensors so the network
    exposes a single-tensor forward (obs) suitable for a menagerie trace."""

    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.inner_model = InnerModel(cfg)
        self.num_steps_conditioning = cfg.num_steps_conditioning
        self.img_channels = cfg.img_channels
        self.register_buffer("act", torch.zeros(1, cfg.num_steps_conditioning, dtype=torch.long))
        self.register_buffer("c_noise", torch.zeros(1))

    def forward(self, obs_and_noisy: Tensor) -> Tensor:
        n = self.num_steps_conditioning
        c = self.img_channels
        obs = obs_and_noisy[:, : n * c]
        noisy_next_obs = obs_and_noisy[:, n * c : (n + 1) * c]
        return self.inner_model(noisy_next_obs, self.c_noise, obs, self.act)


def build_diamond_denoiser() -> nn.Module:
    cfg = InnerModelConfig(
        img_channels=_IMG_CH,
        num_steps_conditioning=_NUM_STEPS_COND,
        cond_channels=_COND_CH,
        depths=[1, 1],
        channels=[8, 16],
        attn_depths=[False, False],
        num_actions=_NUM_ACTIONS,
    )
    return DiamondInnerModelWrapper(cfg)


def example_input_diamond_denoiser() -> Tensor:
    # channel-concatenated (num_steps_conditioning + 1) frames at 16x16, batch=1
    return torch.randn(1, (_NUM_STEPS_COND + 1) * _IMG_CH, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "DIAMOND_diffusion_world_model",
        build_diamond_denoiser,
        example_input_diamond_denoiser,
        2024,
        "vendored-pytorch",
    ),
]
