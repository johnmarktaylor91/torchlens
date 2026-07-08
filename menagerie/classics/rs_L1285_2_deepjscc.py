# SOURCE: vendored from chunbaobao/Deep-JSCC-PyTorch @ main
# (model.py class DeepJSCC/_Encoder/_Decoder/_ConvWithPReLU/_TransConvWithPReLU;
# channel.py class Channel; unchanged)
"""DeepJSCC: Deep Joint Source-Channel Coding for Wireless Image Transmission (Bourtsoulatze,
Burth Kurka, Gunduz, IEEE TCCN 2019). Official PyTorch implementation of that paper:
https://github.com/chunbaobao/Deep-JSCC-PyTorch (``model.py`` + ``channel.py`` @ main; the
``queue.tsv`` candidate list points at ``tinyxuyan/AE-Com-Roadmap`` as a DeepJSCC pointer repo,
but that repo is only a README link collection with no model code -- this vendors the actual
canonical PyTorch DeepJSCC reimplementation of the Bourtsoulatze et al. architecture instead).

``DeepJSCC`` is a CNN autoencoder for end-to-end wireless image transmission: a 5-layer
strided/same Conv2d+PReLU encoder compresses an RGB image into a complex-valued channel
codeword and power-normalizes it (``_Encoder._normlizationLayer``), a simulated wireless
``Channel`` (AWGN or Rayleigh fading, additive per-symbol noise at a target SNR) corrupts the
codeword, and a mirrored 5-layer ConvTranspose2d+PReLU/Sigmoid decoder reconstructs the image.
No layer, channel count, normalization math, or forward-pass control-flow was changed from the
source files.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# channel.py
# ---------------------------------------------------------------------------
class Channel(nn.Module):
    def __init__(self, channel_type="AWGN", snr=20):
        if channel_type not in ["AWGN", "Rayleigh"]:
            raise Exception("Unknown type of channel")
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr

    def forward(self, z_hat):
        if z_hat.dim() not in {3, 4}:
            raise ValueError("Input tensor must be 3D or 4D")

        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)

        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2)
        if self.channel_type == "Rayleigh":
            hc = torch.randn(2, device=z_hat.device)

            # clone for in-place operation
            z_hat = z_hat.clone()
            z_hat[:, : z_hat.size(1) // 2] = hc[0] * z_hat[:, : z_hat.size(1) // 2]
            z_hat[:, z_hat.size(1) // 2 :] = hc[1] * z_hat[:, z_hat.size(1) // 2 :]

        return z_hat + noise

    def get_channel(self):
        return self.channel_type, self.snr


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------
class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    # NOTE: default arg is a single shared nn.PReLU() instance, and the source's own
    # ``if activate == nn.PReLU():`` init-branch check always evaluates False (nn.Module
    # equality is identity-based, so a freshly constructed comparison PReLU never equals
    # the passed-in one) -- both are genuine latent quirks in the upstream repo, preserved
    # here verbatim rather than "fixed", per the faithful-vendoring contract.
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activate=nn.PReLU(),
        padding=0,
        output_padding=0,
    ):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.activate = activate
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(
                self.transconv.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        self.conv1 = _ConvWithPReLU(
            in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = _ConvWithPReLU(
            in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2
        )
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2 * c, kernel_size=5, padding=2)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception("Unknown size of input")
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor

        return _inner

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.norm(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2 * c, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.tconv4 = _TransConvWithPReLU(
            in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16,
            out_channels=3,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            activate=nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type="AWGN", snr=None):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        if snr is not None:
            self.channel = Channel(channel_type, snr)
        self.decoder = _Decoder(c=c)

    def forward(self, x):
        z = self.encoder(x)
        if hasattr(self, "channel") and self.channel is not None:
            z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type="AWGN", snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, "channel") and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction="mean")
        loss = criterion(prd, gt)
        return loss


# ---------------------------------------------------------------------------
# Menagerie staging harness
# ---------------------------------------------------------------------------
def build_deepjscc():
    """AWGN-channel DeepJSCC at SNR=10dB, c=4 codeword channels (repo's ``c`` sweeps
    include 4/8/16 for CIFAR10 at 32x32 input; kept small here for a tiny image size)."""
    return DeepJSCC(c=4, channel_type="AWGN", snr=10)


def example_input_deepjscc():
    torch.manual_seed(0)
    return torch.rand(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "DeepJSCC Wireless Image Transmission",
        "build_deepjscc",
        "example_input_deepjscc",
        2019,
        "vendored-pytorch",
    ),
]
