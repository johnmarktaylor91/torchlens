"""SuSIE subgoal image-editing diffusion U-Net.

Paper: Black et al. 2023, "Zero-Shot Robotic Manipulation with Pretrained
Image-Editing Diffusion Models." SuSIE uses an InstructPix2Pix-style
language-conditioned image-editing diffusion model to synthesize visual
subgoals for a low-level goal-reaching policy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeTextBlock(nn.Module):
    """Residual convolution block with timestep and text conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        """Initialize conditioned residual block.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        cond_dim:
            Conditioning vector width.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply conditioned residual update.

        Parameters
        ----------
        x:
            Feature map.
        cond:
            Conditioning vector.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        scale, shift = self.cond(cond).chunk(2, dim=-1)
        y = F.silu(self.conv1(x))
        y = y * (1 + scale[..., None, None]) + shift[..., None, None]
        y = self.conv2(F.silu(y))
        return y + self.skip(x)


class SuSIESubgoalUNet(nn.Module):
    """Compact InstructPix2Pix-style subgoal denoising U-Net."""

    def __init__(self, vocab: int = 128, cond_dim: int = 48) -> None:
        """Initialize image, timestep, and language-conditioned U-Net.

        Parameters
        ----------
        vocab:
            Text vocabulary size.
        cond_dim:
            Conditioning width.
        """

        super().__init__()
        self.text = nn.Embedding(vocab, cond_dim)
        self.time = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.down1 = TimeTextBlock(6, 24, cond_dim)
        self.down2 = TimeTextBlock(24, 48, cond_dim)
        self.mid_attn = nn.MultiheadAttention(48, 4, batch_first=True)
        self.up1 = TimeTextBlock(72, 24, cond_dim)
        self.out = nn.Conv2d(24, 3, 3, padding=1)

    def forward(
        self, noisy: torch.Tensor, image: torch.Tensor, text: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict denoised subgoal-image residual.

        Parameters
        ----------
        noisy:
            Noisy target/subgoal image.
        image:
            Current observation image.
        text:
            Language command token ids.
        t:
            Diffusion timestep or noise level.

        Returns
        -------
        torch.Tensor
            Predicted RGB residual.
        """

        cond = self.text(text).mean(dim=1) + self.time(t.view(t.shape[0], 1))
        h1 = self.down1(torch.cat([noisy, image], dim=1), cond)
        h2 = self.down2(F.avg_pool2d(h1, 2), cond)
        tokens = h2.flatten(2).transpose(1, 2)
        tokens, _ = self.mid_attn(tokens, tokens, tokens)
        h2 = tokens.transpose(1, 2).reshape_as(h2)
        up = F.interpolate(h2, size=h1.shape[-2:], mode="nearest")
        return self.out(self.up1(torch.cat([up, h1], dim=1), cond))


def build() -> nn.Module:
    """Build the compact SuSIE subgoal U-Net.

    Returns
    -------
    nn.Module
        SuSIE denoiser.
    """

    return SuSIESubgoalUNet()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create example noisy image, observation, language, and timestep inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Model inputs.
    """

    return (
        torch.randn(1, 3, 24, 24),
        torch.randn(1, 3, 24, 24),
        torch.randint(0, 128, (1, 8)),
        torch.ones(1),
    )


MENAGERIE_ENTRIES = [("SUSIE_subgoal_unet", "build", "example_input", "2023", "E7")]
