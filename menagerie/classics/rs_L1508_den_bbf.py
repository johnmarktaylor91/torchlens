# SOURCE: vendored from https://github.com/NoSavedDATA/PyTorch-BBF-Bigger-Better-Faster-Atari-100k @ main
# (network backbone is imported from the author's companion package NoSavedDATA/NoSavedDATA;
# `nosaveddata/builders/resnet.py` :: IMPALA_Resnet + DQN_Conv, plus the small helper modules
# it depends on: `nosaveddata/builders/weight_init.py` :: init_relu, and the frame-inspection
# `nsd_Module` hyperparameter-saving base class from `nosaveddata/nsd_utils/save_hypers.py`.
# These are vendored verbatim (only the pieces actually reached by IMPALA_Resnet/DQN_Conv);
# unrelated builder classes (FiLM, ConvNeXt, Dreamer blocks, etc.) are intentionally omitted.)
"""BBF (Bigger, Better, Faster; Schwarzer et al. 2023) is a sample-efficient Atari-100k
agent. Its network backbone is a 4x-scaled IMPALA-style ResNet CNN encoder (this module)
feeding a distributional (C51) dueling Q-head; this staging module captures the real
convolutional trunk (`IMPALA_Resnet`, built from `DQN_Conv` + residual blocks) exactly as
used by the author's BBF training scripts (`bbf_taco.py` etc., which import it from the
sibling `nosaveddata` package rather than redefining it)."""

import torch
import torch.nn as nn
import inspect


# --- from nosaveddata/nsd_utils/save_hypers.py (verbatim) ---
class Hypers:
    """Sorcery: automatically saves all arguments of the inherited class __init__."""

    def __init__(self, max_depth=3, **kwargs):
        super().__init__(**kwargs)
        self.save_hypers(max_depth)

    def save_hypers(self, max_depth, ignore=[]):
        """Save function arguments into class attributes."""
        seen_init = False
        frame = inspect.currentframe()
        for d in range(max_depth):
            frame = frame.f_back
            if frame.f_back and frame.f_back.f_code.co_name == "__init__":
                seen_init = True
            if seen_init and frame.f_back.f_code.co_name != "__init__":
                break

        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        for k, v in self.hparams.items():
            setattr(self, k, v)


class nsd_Module(Hypers, nn.Module):
    def __init__(self):
        super().__init__(max_depth=3)


# --- from nosaveddata/builders/weight_init.py (verbatim, only what's used) ---
def init_relu(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.orthogonal_(module.weight, gain=1.41421)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# --- from nosaveddata/builders/resnet.py (verbatim) ---
class DQN_Conv(nn.Module):
    def __init__(
        self,
        in_hiddens,
        hiddens,
        ks,
        stride,
        padding=0,
        max_pool=False,
        norm=True,
        init=init_relu,
        act=nn.SiLU(),
        bias=True,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_hiddens, hiddens, ks, stride, padding, bias=bias),
            nn.MaxPool2d(3, 2, padding=1) if max_pool else nn.Identity(),
            (
                nn.GroupNorm(32, hiddens, eps=1e-6)
                if hiddens % 32 == 0
                else nn.BatchNorm2d(hiddens, eps=1e-6)
            )
            if norm
            else nn.Identity(),
            act,
        )
        self.conv.apply(init)

    def forward(self, X):
        return self.conv(X)


class Residual_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        act=nn.SiLU(),
        out_act=nn.SiLU(),
        norm=True,
        init=None,
        bias=True,
    ):
        super().__init__()
        if init is None:
            init = init_xavier

        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=stride, bias=bias),
            (
                nn.GroupNorm(32, channels, eps=1e-6)
                if channels % 32 == 0
                else nn.BatchNorm2d(channels, eps=1e-6)
            )
            if norm
            else nn.Identity(),
            act,
        )
        conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
            (
                nn.GroupNorm(32, channels, eps=1e-6)
                if channels % 32 == 0
                else nn.BatchNorm2d(channels, eps=1e-6)
            )
            if norm
            else nn.Identity(),
            out_act,
        )

        conv1.apply(init)
        conv2.apply(init if out_act != nn.Identity() else init_xavier)

        self.conv = nn.Sequential(conv1, conv2)

        self.proj = nn.Identity()
        if stride > 1 or in_channels != channels:
            self.proj = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=stride)

        self.proj.apply(init_proj2d)
        self.out_act = out_act

    def forward(self, X):
        Y = self.conv(X)
        return Y + self.proj(X)


def init_xavier(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_proj2d(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        torch.nn.init.dirac_(module.weight, groups=1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class IMPALA_Resnet(nsd_Module):
    """4x-scaled (with scale_width=4) IMPALA-style ResNet CNN encoder -- BBF's backbone."""

    def __init__(
        self, first_channels=12, scale_width=1, norm=True, init=init_relu, act=nn.SiLU(), bias=True
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            self.get_block(first_channels, 16 * scale_width),
            self.get_block(16 * scale_width, 32 * scale_width),
            self.get_block(32 * scale_width, 32 * scale_width, last_relu=True),
        )

    def get_block(self, in_hiddens, out_hiddens, last_relu=False):
        blocks = nn.Sequential(
            DQN_Conv(
                in_hiddens,
                out_hiddens,
                3,
                1,
                1,
                max_pool=True,
                bias=self.bias,
                act=self.act,
                norm=self.norm,
                init=self.init,
            ),
            Residual_Block(
                out_hiddens,
                out_hiddens,
                bias=self.bias,
                norm=self.norm,
                act=self.act,
                init=self.init,
            ),
            Residual_Block(
                out_hiddens,
                out_hiddens,
                bias=self.bias,
                norm=self.norm,
                act=self.act,
                init=self.init,
                out_act=self.act if last_relu else nn.Identity(),
            ),
        )
        return blocks

    def forward(self, X):
        return self.cnn(X)


def build_den_bbf():
    # BBF uses 4 stacked grayscale Atari frames (first_channels=4) with scale_width=4
    # ("Bigger" = 4x width). Shrunk here to scale_width=1 for a tiny param count while
    # keeping the real architecture (3 IMPALA blocks, each: DQN_Conv + 2 Residual_Blocks).
    return IMPALA_Resnet(first_channels=4, scale_width=1)


def example_input_den_bbf():
    # BBF trains on 84x84 Atari frames stacked 4-deep; kept at native 84x84 (spatial size
    # doesn't affect param count, only activation memory).
    return torch.randn(2, 4, 84, 84)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DEN (Deep Exploration Network) / BBF IMPALA-ResNet backbone",
        build_den_bbf,
        example_input_den_bbf,
        2023,
        MENAGERIE_ZOO,
    ),
]
