# SOURCE: vendored from EmilienDupont/coinpp @ master (coinpp/models.py)
# SOURCE: vendored from richzhang/colorization @ master (colorizers/eccv16.py, base_color.py)
# SOURCE: vendored from erichson/koopmanAE @ master (model.py)
# SOURCE: vendored from ycq091044/ContraWR @ main (src/model.py)
from __future__ import annotations

from math import sqrt

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class Sine(nn.Module):
    """Sine activation with scaling from COIN++."""

    def __init__(self, w0: float = 1.0) -> None:
        """Initialize the scaled sine activation."""
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the scaled sine activation."""
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Single SIREN layer vendored from COIN++."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 30.0,
        c: float = 6.0,
        is_first: bool = False,
        is_last: bool = False,
        use_bias: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        """Initialize the SIREN layer."""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)
        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SIREN layer."""
        out = self.linear(x)
        if self.is_last:
            out += 0.5
        else:
            out = self.activation(out)
        return out


class LatentToModulation(nn.Module):
    """Map a COIN++ latent vector to modulation values."""

    def __init__(
        self,
        latent_dim: int,
        num_modulations: int,
        dim_hidden: int,
        num_layers: int,
    ) -> None:
        """Initialize the latent-to-modulation network."""
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers: list[nn.Module] = [nn.Linear(latent_dim, dim_hidden), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return modulations for a latent vector."""
        return self.net(latent)


class Bias(nn.Module):
    """Learned COIN++ bias modulation."""

    def __init__(self, size: int) -> None:
        """Initialize the learned bias."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size), requires_grad=True)
        self.latent_dim = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the learned bias to the input."""
        return x + self.bias


class ModulatedSiren(nn.Module):
    """Modulated SIREN model vendored from COIN++."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 30.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        modulate_scale: bool = False,
        modulate_shift: bool = True,
        use_latent: bool = False,
        latent_dim: int = 64,
        modulation_net_dim_hidden: int = 64,
        modulation_net_num_layers: int = 1,
    ) -> None:
        """Initialize the modulated SIREN."""
        super().__init__()
        if not (modulate_scale or modulate_shift):
            raise ValueError("COIN++ must modulate scale or shift.")
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layers.append(
                SirenLayer(
                    dim_in=dim_in if is_first else dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0_initial if is_first else w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )
        self.net = nn.Sequential(*layers)
        self.last_layer = SirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            is_last=True,
        )
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            num_modulations *= 2
        if use_latent:
            self.modulation_net: nn.Module = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)
        if not use_latent:
            bias_module = self.modulation_net
            if isinstance(bias_module, Bias):
                if self.modulate_shift and self.modulate_scale:
                    bias_module.bias.data = torch.cat(
                        (
                            torch.ones(num_modulations // 2),
                            torch.zeros(num_modulations // 2),
                        ),
                        dim=0,
                    )
                elif self.modulate_scale:
                    bias_module.bias.data = torch.ones(num_modulations)
                else:
                    bias_module.bias.data = torch.zeros(num_modulations)
        self.num_modulations = num_modulations

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Run the COIN++ modulated forward pass."""
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        modulations = self.modulation_net(latent)
        mid_idx = self.num_modulations // 2 if self.modulate_scale and self.modulate_shift else 0
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                scale = modulations[:, idx : idx + self.dim_hidden].unsqueeze(1) + 1.0
            else:
                scale = 1.0
            if self.modulate_shift:
                shift = modulations[:, mid_idx + idx : mid_idx + idx + self.dim_hidden].unsqueeze(1)
            else:
                shift = 0.0
            x = module.linear(x)
            x = scale * x + shift
            x = module.activation(x)
            idx += self.dim_hidden
        out = self.last_layer(x)
        return out.view(*x_shape, out.shape[-1])


class BaseColor(nn.Module):
    """Base normalization module vendored from richzhang/colorization."""

    def __init__(self) -> None:
        """Initialize LAB normalization constants."""
        super().__init__()
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

    def normalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        """Normalize L-channel input."""
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        """Unnormalize AB-channel output."""
        return in_ab * self.ab_norm


class ECCVGenerator(BaseColor):
    """ECCV 2016 colorization generator vendored from richzhang/colorization."""

    def __init__(self, norm_layer: type[nn.Module] = nn.BatchNorm2d) -> None:
        """Initialize the colorization generator."""
        super().__init__()
        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]
        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]
        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]
        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        model8 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        ]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(
            313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False
        )
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, input_l: torch.Tensor) -> torch.Tensor:
        """Run the colorization generator."""
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))
        return self.unnormalize_ab(self.upsample4(out_reg))


def gaussian_init_(n_units: int, std: float = 1.0) -> torch.Tensor:
    """Return the KoopmanAE Gaussian initialization matrix."""
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    return sampler.sample((n_units, n_units))[..., 0]


class EncoderNet(nn.Module):
    """KoopmanAE encoder vendored from erichson/koopmanAE."""

    def __init__(self, m: int, n: int, b: int, alpha: int = 1) -> None:
        """Initialize the KoopmanAE encoder."""
        super().__init__()
        self.n_features = m * n
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(self.n_features, 16 * alpha)
        self.fc2 = nn.Linear(16 * alpha, 16 * alpha)
        self.fc3 = nn.Linear(16 * alpha, b)
        self._init_linear()

    def _init_linear(self) -> None:
        """Initialize linear layers as in the source."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a spatial state into Koopman latent space."""
        x = x.view(-1, 1, self.n_features)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3(x)


class DecoderNet(nn.Module):
    """KoopmanAE decoder vendored from erichson/koopmanAE."""

    def __init__(self, m: int, n: int, b: int, alpha: int = 1) -> None:
        """Initialize the KoopmanAE decoder."""
        super().__init__()
        self.m = m
        self.n = n
        self.b = b
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(b, 16 * alpha)
        self.fc2 = nn.Linear(16 * alpha, 16 * alpha)
        self.fc3 = nn.Linear(16 * alpha, m * n)
        self._init_linear()

    def _init_linear(self) -> None:
        """Initialize linear layers as in the source."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a Koopman latent state."""
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x.view(-1, 1, self.m, self.n)


class Dynamics(nn.Module):
    """Forward Koopman dynamics layer."""

    def __init__(self, b: int, init_scale: float) -> None:
        """Initialize the forward dynamics matrix."""
        super().__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)
        u, _, v = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(u, v.t()) * init_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Koopman dynamics step."""
        return self.dynamics(x)


class DynamicsBack(nn.Module):
    """Backward Koopman dynamics layer."""

    def __init__(self, b: int, omega: Dynamics) -> None:
        """Initialize inverse dynamics from the forward dynamics."""
        super().__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one backward dynamics step."""
        return self.dynamics(x)


class KoopmanAE(nn.Module):
    """Consistent Koopman Autoencoder vendored from erichson/koopmanAE."""

    def __init__(
        self,
        m: int,
        n: int,
        b: int,
        steps: int,
        steps_back: int,
        alpha: int = 1,
        init_scale: float = 1.0,
    ) -> None:
        """Initialize the Koopman autoencoder."""
        super().__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.encoder = EncoderNet(m, n, b, alpha=alpha)
        self.dynamics = Dynamics(b, init_scale)
        self.backdynamics = DynamicsBack(b, self.dynamics)
        self.decoder = DecoderNet(m, n, b, alpha=alpha)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the source forward-mode KoopmanAE path."""
        out: list[torch.Tensor] = []
        out_back: list[torch.Tensor] = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()
        for _ in range(self.steps):
            q = self.dynamics(q)
            out.append(self.decoder(q))
        out.append(self.decoder(z.contiguous()))
        return out, out_back


class ResBlock(nn.Module):
    """Residual block vendored from ContraWR."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
        pooling: bool = False,
    ) -> None:
        """Initialize the ContraWR residual block."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample_or_not = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ContraWR residual block."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(x) if self.downsample_or_not else x
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        return self.dropout(out)


class CNNEncoder2DSleep(nn.Module):
    """ContraWR Sleep-EDF encoder vendored from ycq091044/ContraWR."""

    def __init__(self, n_dim: int) -> None:
        """Initialize the ContraWR encoder."""
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        self.n_dim = n_dim
        self.fc = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )
        self.sup = nn.Sequential(
            nn.Linear(128, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 5, bias=True),
        )
        self.byol_mapping = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, x_train: torch.Tensor) -> torch.Tensor:
        """Compute the ContraWR spectrogram frontend."""
        signal = []
        for s in range(x_train.shape[1]):
            spectral = torch.stft(
                x_train[:, s, :],
                n_fft=256,
                hop_length=256 * 1 // 4,
                center=False,
                onesided=True,
                return_complex=False,
            )
            signal.append(spectral)
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)
        return torch.cat(
            [torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the default ContraWR mid-feature path."""
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.reshape(x.shape[0], -1)


def build_coinpp() -> ModulatedSiren:
    """Build a trace-sized COIN++ ModulatedSiren."""
    return ModulatedSiren(
        dim_in=2,
        dim_hidden=8,
        dim_out=3,
        num_layers=3,
        modulate_scale=True,
        modulate_shift=True,
        use_latent=True,
        latent_dim=5,
        modulation_net_dim_hidden=8,
    )


def example_input_coinpp() -> tuple[torch.Tensor, torch.Tensor]:
    """Return example COIN++ coordinates and latent vector."""
    return torch.randn(1, 4, 2), torch.randn(1, 5)


def build_colorful_image_colorization() -> ECCVGenerator:
    """Build the ECCV colorization generator."""
    return ECCVGenerator()


def example_input_colorful_image_colorization() -> torch.Tensor:
    """Return an example L-channel image."""
    return torch.randn(1, 1, 32, 32)


def build_consistent_koopman_autoencoder() -> KoopmanAE:
    """Build a trace-sized KoopmanAE."""
    return KoopmanAE(m=4, n=4, b=3, steps=1, steps_back=1)


def example_input_consistent_koopman_autoencoder() -> torch.Tensor:
    """Return a small KoopmanAE input image."""
    return torch.randn(1, 1, 4, 4)


def build_contrawr() -> CNNEncoder2DSleep:
    """Build a trace-sized ContraWR encoder."""
    return CNNEncoder2DSleep(n_dim=8)


def example_input_contrawr() -> torch.Tensor:
    """Return a short four-channel sleep EEG waveform."""
    return torch.randn(1, 2, 3000)


MENAGERIE_ENTRIES = [
    ("COIN++", "build_coinpp", "example_input_coinpp", 2021, "CV9"),
    (
        "Colorful Image Colorization",
        "build_colorful_image_colorization",
        "example_input_colorful_image_colorization",
        2016,
        "CV9",
    ),
    (
        "Consistent Koopman Autoencoder",
        "build_consistent_koopman_autoencoder",
        "example_input_consistent_koopman_autoencoder",
        2020,
        "CV9",
    ),
    ("ContraWR", "build_contrawr", "example_input_contrawr", 2023, "CV9"),
]
