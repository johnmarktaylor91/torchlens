"""DRAW, 2015, Karol Gregor et al.

Paper: DRAW: A Recurrent Neural Network For Image Generation.
The model repeatedly reads and writes differentiable Gaussian-filterbank
glimpses with encoder/decoder LSTMs and a latent code at each step.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DRAW(nn.Module):
    """Small DRAW generator-reconstructor with Gaussian attention."""

    def __init__(
        self,
        image_size: int = 28,
        glimpse_size: int = 5,
        hidden_size: int = 32,
        latent_size: int = 8,
        steps: int = 3,
    ) -> None:
        """Initialize DRAW recurrent modules.

        Parameters
        ----------
        image_size:
            Height and width of square grayscale images.
        glimpse_size:
            Side length of the attended patch.
        hidden_size:
            LSTM hidden size.
        latent_size:
            Per-step latent size.
        steps:
            Number of recurrent read/write steps.
        """
        super().__init__()
        self.image_size = image_size
        self.glimpse_size = glimpse_size
        self.steps = steps
        read_size = 2 * glimpse_size * glimpse_size
        self.encoder = nn.LSTMCell(read_size + hidden_size, hidden_size)
        self.decoder = nn.LSTMCell(latent_size, hidden_size)
        self.to_mu = nn.Linear(hidden_size, latent_size)
        self.to_logvar = nn.Linear(hidden_size, latent_size)
        self.read_params = nn.Linear(hidden_size, 5)
        self.write_params = nn.Linear(hidden_size, 5)
        self.write_patch = nn.Linear(hidden_size, glimpse_size * glimpse_size)

    def _filterbank(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Build DRAW's separable Gaussian attention filters.

        Parameters
        ----------
        params:
            Attention parameter tensor ``(B, 5)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Row filters, column filters, and gamma scale.
        """
        batch = params.shape[0]
        gx_, gy_, log_sigma, log_delta, log_gamma = params.chunk(5, dim=1)
        scale = float(self.image_size + 1) / 2.0
        center_x = scale * (gx_.tanh() + 1.0)
        center_y = scale * (gy_.tanh() + 1.0)
        delta = (float(self.image_size - 1) / float(self.glimpse_size - 1)) * log_delta.exp()
        sigma2 = log_sigma.exp().square() + 1.0e-4
        gamma = log_gamma.exp().view(batch, 1, 1)
        offsets = torch.arange(self.glimpse_size, device=params.device, dtype=params.dtype)
        offsets = offsets - float(self.glimpse_size - 1) / 2.0
        mu_x = center_x + offsets.view(1, -1) * delta
        mu_y = center_y + offsets.view(1, -1) * delta
        grid = torch.arange(self.image_size, device=params.device, dtype=params.dtype)
        fx = torch.exp(
            -((grid.view(1, 1, -1) - mu_x.view(batch, -1, 1)) ** 2)
            / (2.0 * sigma2.view(batch, 1, 1))
        )
        fy = torch.exp(
            -((grid.view(1, 1, -1) - mu_y.view(batch, -1, 1)) ** 2)
            / (2.0 * sigma2.view(batch, 1, 1))
        )
        fx = fx / (fx.sum(dim=2, keepdim=True) + 1.0e-8)
        fy = fy / (fy.sum(dim=2, keepdim=True) + 1.0e-8)
        return fy, fx, gamma

    def _read(self, x: Tensor, error: Tensor, h_dec: Tensor) -> Tensor:
        """Read original and residual images through the same filterbank.

        Parameters
        ----------
        x:
            Input image tensor.
        error:
            Reconstruction residual tensor.
        h_dec:
            Previous decoder hidden state.

        Returns
        -------
        Tensor
            Concatenated attention glimpses.
        """
        fy, fx, gamma = self._filterbank(self.read_params(h_dec))
        image = x[:, 0]
        residual = error[:, 0]
        read_x = torch.bmm(torch.bmm(fy, image), fx.transpose(1, 2))
        read_err = torch.bmm(torch.bmm(fy, residual), fx.transpose(1, 2))
        return torch.cat([read_x.flatten(1), read_err.flatten(1)], dim=1) * gamma.flatten(1)

    def _write(self, h_dec: Tensor) -> Tensor:
        """Write a decoded patch back to image coordinates.

        Parameters
        ----------
        h_dec:
            Decoder hidden state.

        Returns
        -------
        Tensor
            Canvas update with shape ``(B, 1, H, W)``.
        """
        batch = h_dec.shape[0]
        fy, fx, gamma = self._filterbank(self.write_params(h_dec))
        patch = self.write_patch(h_dec).view(batch, self.glimpse_size, self.glimpse_size)
        canvas = torch.bmm(torch.bmm(fy.transpose(1, 2), patch), fx)
        return (canvas / gamma).unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        """Run DRAW's unrolled read-encode-sample-decode-write loop.

        Parameters
        ----------
        x:
            Grayscale image tensor with shape ``(B, 1, H, W)``.

        Returns
        -------
        Tensor
            Reconstructed image probabilities.
        """
        batch = x.shape[0]
        h_enc = x.new_zeros(batch, self.encoder.hidden_size)
        c_enc = x.new_zeros(batch, self.encoder.hidden_size)
        h_dec = x.new_zeros(batch, self.decoder.hidden_size)
        c_dec = x.new_zeros(batch, self.decoder.hidden_size)
        canvas = x.new_zeros(batch, 1, self.image_size, self.image_size)
        for _ in range(self.steps):
            residual = x - torch.sigmoid(canvas)
            read_vec = self._read(x, residual, h_dec)
            h_enc, c_enc = self.encoder(torch.cat([read_vec, h_dec], dim=1), (h_enc, c_enc))
            mu = self.to_mu(h_enc)
            logvar = self.to_logvar(h_enc)
            z = mu + torch.zeros_like(mu) * torch.exp(0.5 * logvar)
            h_dec, c_dec = self.decoder(z, (h_dec, c_dec))
            canvas = canvas + self._write(h_dec)
        return torch.sigmoid(canvas)


def build() -> nn.Module:
    """Build a small DRAW module.

    Returns
    -------
    nn.Module
        Random-initialized DRAW model.
    """
    return DRAW()


def example_input() -> Tensor:
    """Return a traceable image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 28, 28)``.
    """
    return torch.randn(1, 1, 28, 28)
