# SOURCE: vendored from Singingkettle/ChangShuoRadioRecognition @ 088320d and silviutroscot/CodeSLAM @ 8570f67
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class CLDNNL(nn.Module):
    """Vendored CLDNNL radio-modulation backbone."""

    def __init__(self, frame_length: int = 128, num_classes: int = -1) -> None:
        """Initialize the CLDNNL convolutional, LSTM, and optional classifier blocks."""
        super().__init__()
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=(2, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 80, kernel_size=(1, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(80, 80, kernel_size=(1, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.lstm = nn.LSTM(input_size=self.frame_length - 8, hidden_size=50, batch_first=True)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Run the CLDNNL forward pass."""
        x = self.cnn(x)
        x = torch.reshape(x, [-1, 80, self.frame_length - 8])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)


class CLDNNW(nn.Module):
    """Vendored CLDNNW radio-modulation backbone."""

    def __init__(self, frame_length: int = 128, num_classes: int = -1) -> None:
        """Initialize the CLDNNW convolutional, LSTM, and optional classifier blocks."""
        super().__init__()
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.lstm = nn.LSTM(
            input_size=(self.frame_length * 2 - 7 * 4) * 2,
            hidden_size=50,
            batch_first=True,
        )

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(50, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Run the CLDNNW forward pass."""
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        x = torch.concatenate((x1, x2), dim=3)
        x = torch.reshape(x, [-1, 50, (self.frame_length * 2 - 7 * 4) * 2])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)


class ModelConfig:
    """Vendored CodeSLAM model configuration."""

    def __init__(self) -> None:
        """Initialize the compact trace-gate configuration."""
        self.input_height = 64
        self.input_width = 64
        self.code_dim = 16
        self.base_channels = 4
        self.pyramid_levels = 4
        self.latent_hidden_dim = 64
        self.linear_decoder = True
        self.min_depth = 0.1
        self.max_depth = 40.0
        self.proximity_average_depth = 4.0
        self.min_uncertainty = 1e-3

    @property
    def prediction_height(self) -> int:
        """Return the prediction height."""
        return self.input_height // 2

    @property
    def prediction_width(self) -> int:
        """Return the prediction width."""
        return self.input_width // 2

    @property
    def proximity_transition(self) -> float:
        """Return the depth/proximity transition parameter."""
        return self.proximity_average_depth


class DepthPrediction:
    """Vendored CodeSLAM prediction container."""

    def __init__(
        self,
        code: torch.Tensor,
        proximity_pyramid: list[torch.Tensor],
        depth_pyramid: list[torch.Tensor],
        scale_pyramid: list[torch.Tensor],
        posterior_mean: torch.Tensor | None = None,
        posterior_logvar: torch.Tensor | None = None,
    ) -> None:
        """Initialize the CodeSLAM prediction container."""
        self.code = code
        self.proximity_pyramid = proximity_pyramid
        self.depth_pyramid = depth_pyramid
        self.scale_pyramid = scale_pyramid
        self.posterior_mean = posterior_mean
        self.posterior_logvar = posterior_logvar


def positive_scale(log_scale: torch.Tensor, minimum: float) -> torch.Tensor:
    """Map a raw log-scale tensor to a positive uncertainty scale."""
    return F.softplus(log_scale) + minimum


def depth_to_proximity(depth: torch.Tensor, transition: float) -> torch.Tensor:
    """Map depth to CodeSLAM's hybrid proximity parametrization."""
    transition_tensor = torch.as_tensor(transition, device=depth.device, dtype=depth.dtype)
    return transition_tensor / (depth + transition_tensor).clamp_min(1e-6)


def proximity_to_depth(proximity: torch.Tensor, transition: float) -> torch.Tensor:
    """Invert CodeSLAM's hybrid proximity parametrization."""
    transition_tensor = torch.as_tensor(transition, device=proximity.device, dtype=proximity.dtype)
    return transition_tensor * (1.0 - proximity) / proximity.clamp_min(1e-6)


def _group_count(channels: int) -> int:
    """Return the largest preferred GroupNorm group count that divides channels."""
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Vendored CodeSLAM convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        linear: bool = False,
        use_norm: bool = True,
    ) -> None:
        """Initialize a convolution, optional GroupNorm, and optional ReLU."""
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=not use_norm,
            ),
        ]
        if use_norm:
            layers.append(nn.GroupNorm(_group_count(out_channels), out_channels))
        if not linear:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional block."""
        return self.block(x)


class EncoderStage(nn.Module):
    """Vendored CodeSLAM encoder stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize downsampling and refinement blocks."""
        super().__init__()
        self.down = ConvBlock(in_channels, out_channels, stride=2, linear=False, use_norm=True)
        self.refine = ConvBlock(out_channels, out_channels, stride=1, linear=False, use_norm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder stage."""
        return self.refine(self.down(x))


class ConditioningFusion(nn.Module):
    """Vendored CodeSLAM multiplicative conditioning fusion."""

    def __init__(
        self,
        depth_channels: int,
        image_channels: int,
        out_channels: int,
        *,
        linear: bool,
        use_norm: bool,
    ) -> None:
        """Initialize the image projection and mixing block."""
        super().__init__()
        self.image_projection = nn.Conv2d(image_channels, depth_channels, kernel_size=1, bias=True)
        self.mix = ConvBlock(
            depth_channels + image_channels + depth_channels,
            out_channels,
            stride=1,
            linear=linear,
            use_norm=use_norm,
        )

    def forward(self, depth_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """Fuse depth and image features."""
        projected = self.image_projection(image_features)
        fused = torch.cat([depth_features, image_features, depth_features * projected], dim=1)
        return self.mix(fused)


class UpsampleStage(nn.Module):
    """Vendored CodeSLAM upsampling stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        linear: bool,
        use_norm: bool,
        use_deconvolution: bool = False,
    ) -> None:
        """Initialize an interpolation or deconvolution upsampler and projection block."""
        super().__init__()
        self.linear = linear
        self.use_deconvolution = use_deconvolution
        if use_deconvolution:
            self.deconvolution = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            )
        self.project = ConvBlock(
            in_channels, out_channels, stride=1, linear=linear, use_norm=use_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the upsampling stage."""
        if self.use_deconvolution:
            x = self.deconvolution(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.project(x)


class DecoderState:
    """Vendored CodeSLAM decoder state."""

    def __init__(
        self, proximity_pyramid: list[torch.Tensor], scale_pyramid: list[torch.Tensor]
    ) -> None:
        """Initialize the decoder state."""
        self.proximity_pyramid = proximity_pyramid
        self.scale_pyramid = scale_pyramid


class CodeSLAMDepthModel(nn.Module):
    """Vendored conditioned depth auto-encoder described in the CodeSLAM paper."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the CodeSLAM image/depth encoder, latent code, and decoder."""
        super().__init__()
        self.config = config or ModelConfig()

        image_channels = [
            self.config.base_channels,
            self.config.base_channels * 2,
            self.config.base_channels * 4,
            self.config.base_channels * 8,
            self.config.base_channels * 16,
        ]
        self._image_channels = image_channels

        self.image_encoder = nn.ModuleList()
        in_channels = 1
        for out_channels in image_channels:
            self.image_encoder.append(EncoderStage(in_channels, out_channels))
            in_channels = out_channels

        self.depth_encoder = nn.ModuleList()
        self.depth_fusions = nn.ModuleList()
        in_channels = 1
        for out_channels, image_channels_at_level in zip(image_channels, image_channels):
            self.depth_encoder.append(EncoderStage(in_channels, out_channels))
            self.depth_fusions.append(
                ConditioningFusion(
                    out_channels, image_channels_at_level, out_channels, linear=False, use_norm=True
                )
            )
            in_channels = out_channels

        bottleneck_height = self.config.input_height // 32
        bottleneck_width = self.config.input_width // 32
        self._bottleneck_shape = (image_channels[-1], bottleneck_height, bottleneck_width)
        bottleneck_dim = image_channels[-1] * bottleneck_height * bottleneck_width

        self.posterior_hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_dim, self.config.latent_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.posterior_mean = nn.Linear(self.config.latent_hidden_dim, self.config.code_dim)
        self.posterior_logvar = nn.Linear(self.config.latent_hidden_dim, self.config.code_dim)

        self.code_to_bottleneck = nn.Linear(self.config.code_dim, bottleneck_dim)
        self.image_bottleneck_bias = nn.Conv2d(
            image_channels[-1], image_channels[-1], kernel_size=1, bias=True
        )

        decoder_channels = [
            image_channels[-2],
            image_channels[-3],
            image_channels[-4],
            image_channels[-5],
        ]
        skip_channels = [
            image_channels[-2],
            image_channels[-3],
            image_channels[-4],
            image_channels[-5],
        ]

        self.decoder_up = nn.ModuleList()
        self.decoder_fusions = nn.ModuleList()
        self.proximity_heads = nn.ModuleList()

        decoder_in = image_channels[-1]
        for index, (out_channels, skip_channels_at_level) in enumerate(
            zip(decoder_channels, skip_channels)
        ):
            self.decoder_up.append(
                UpsampleStage(
                    decoder_in,
                    out_channels,
                    linear=self.config.linear_decoder,
                    use_norm=not self.config.linear_decoder,
                    use_deconvolution=index == len(decoder_channels) - 1,
                )
            )
            self.decoder_fusions.append(
                ConditioningFusion(
                    out_channels,
                    skip_channels_at_level,
                    out_channels,
                    linear=self.config.linear_decoder,
                    use_norm=not self.config.linear_decoder,
                )
            )
            self.proximity_heads.append(nn.Conv2d(out_channels, 1, kernel_size=3, padding=1))
            decoder_in = out_channels

        self.uncertainty_seed = ConvBlock(
            image_channels[-1], decoder_channels[0], stride=1, linear=False, use_norm=True
        )
        self.uncertainty_up = nn.ModuleList()
        self.uncertainty_fusions = nn.ModuleList()
        self.uncertainty_heads = nn.ModuleList()

        uncertainty_in = decoder_channels[0]
        for index, (out_channels, skip_channels_at_level) in enumerate(
            zip(decoder_channels, skip_channels)
        ):
            self.uncertainty_up.append(
                UpsampleStage(
                    uncertainty_in,
                    out_channels,
                    linear=False,
                    use_norm=True,
                    use_deconvolution=index == len(decoder_channels) - 1,
                )
            )
            self.uncertainty_fusions.append(
                ConditioningFusion(
                    out_channels, skip_channels_at_level, out_channels, linear=False, use_norm=True
                )
            )
            self.uncertainty_heads.append(nn.Conv2d(out_channels, 1, kernel_size=3, padding=1))
            uncertainty_in = out_channels

    def _ensure_grayscale(self, intensity: torch.Tensor) -> torch.Tensor:
        """Convert RGB intensity to grayscale if needed."""
        if intensity.ndim == 3:
            intensity = intensity.unsqueeze(0)
        if intensity.shape[1] == 1:
            return intensity
        if intensity.shape[1] == 3:
            r, g, b = intensity[:, 0:1], intensity[:, 1:2], intensity[:, 2:3]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b
        raise ValueError(f"Expected 1 or 3 intensity channels, got {intensity.shape[1]}")

    def encode_image(self, intensity: torch.Tensor) -> list[torch.Tensor]:
        """Encode intensity into a feature pyramid."""
        intensity = self._ensure_grayscale(intensity)
        features = []
        x = intensity
        for stage in self.image_encoder:
            x = stage(x)
            features.append(x)
        return features

    def encode_depth(
        self,
        proximity: torch.Tensor,
        image_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode depth proximity conditioned on image features."""
        x = proximity
        for stage, fusion, image_feature in zip(
            self.depth_encoder, self.depth_fusions, image_features
        ):
            x = stage(x)
            x = fusion(x, image_feature)
        posterior_hidden = self.posterior_hidden(x)
        return self.posterior_mean(posterior_hidden), self.posterior_logvar(posterior_hidden)

    def reparameterize(
        self, mean: torch.Tensor, logvar: torch.Tensor, sample_posterior: bool
    ) -> torch.Tensor:
        """Sample or return the posterior mean latent code."""
        if not sample_posterior:
            return mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode_from_features(
        self, image_features: list[torch.Tensor], code: torch.Tensor
    ) -> DecoderState:
        """Decode a latent code using precomputed image features."""
        batch_size = code.shape[0]
        channels, height, width = self._bottleneck_shape
        x = self.code_to_bottleneck(code).view(batch_size, channels, height, width)
        x = x + self.image_bottleneck_bias(image_features[-1])

        proximity_pyramid: list[torch.Tensor] = []
        skip_features = list(reversed(image_features[:-1]))
        for up, fusion, head, skip in zip(
            self.decoder_up,
            self.decoder_fusions,
            self.proximity_heads,
            skip_features,
        ):
            x = up(x)
            x = fusion(x, skip)
            proximity_pyramid.append(head(x))

        uncertainty = self.uncertainty_seed(image_features[-1])
        scale_pyramid: list[torch.Tensor] = []
        for up, fusion, head, skip in zip(
            self.uncertainty_up,
            self.uncertainty_fusions,
            self.uncertainty_heads,
            skip_features,
        ):
            uncertainty = up(uncertainty)
            uncertainty = fusion(uncertainty, skip)
            scale_pyramid.append(head(uncertainty))

        return DecoderState(proximity_pyramid=proximity_pyramid, scale_pyramid=scale_pyramid)

    def decode(self, intensity: torch.Tensor, code: torch.Tensor) -> DecoderState:
        """Decode a latent code from raw intensity input."""
        image_features = self.encode_image(intensity)
        return self.decode_from_features(image_features, code)

    def predict_from_image_features(
        self, image_features: list[torch.Tensor], code: torch.Tensor
    ) -> DepthPrediction:
        """Predict depth, proximity, and scale pyramids from features and code."""
        decoded = self.decode_from_features(image_features, code)
        scale_pyramid = [
            positive_scale(scale, self.config.min_uncertainty) for scale in decoded.scale_pyramid
        ]
        depth_pyramid = [
            proximity_to_depth(proximity, self.config.proximity_transition).clamp(
                self.config.min_depth,
                self.config.max_depth,
            )
            for proximity in decoded.proximity_pyramid
        ]
        return DepthPrediction(
            code=code,
            proximity_pyramid=decoded.proximity_pyramid,
            depth_pyramid=depth_pyramid,
            scale_pyramid=scale_pyramid,
            posterior_mean=None,
            posterior_logvar=None,
        )

    def zero_code(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create the default zero latent code."""
        return torch.zeros(batch_size, self.config.code_dim, device=device, dtype=dtype)

    def forward(
        self,
        intensity: torch.Tensor,
        depth: torch.Tensor | None = None,
        *,
        code: torch.Tensor | None = None,
        sample_posterior: bool = True,
    ) -> DepthPrediction:
        """Run CodeSLAM depth prediction."""
        intensity = self._ensure_grayscale(intensity)
        image_features = self.encode_image(intensity)

        posterior_mean = None
        posterior_logvar = None
        if code is None:
            if depth is None:
                code = self.zero_code(intensity.shape[0], intensity.device, intensity.dtype)
            else:
                proximity = depth_to_proximity(depth, self.config.proximity_transition)
                posterior_mean, posterior_logvar = self.encode_depth(proximity, image_features)
                code = self.reparameterize(posterior_mean, posterior_logvar, sample_posterior)

        prediction = self.predict_from_image_features(image_features, code)
        prediction.posterior_mean = posterior_mean
        prediction.posterior_logvar = posterior_logvar
        return prediction


def build_cldnn_radio() -> CLDNNL:
    """Build a traceable vendored CLDNN radio backbone."""
    return CLDNNL(frame_length=32, num_classes=4).eval()


def example_input_cldnn_radio() -> torch.Tensor:
    """Return a sample CLDNN radio input tensor."""
    return torch.randn(1, 1, 2, 32)


def build_codeslam() -> CodeSLAMDepthModel:
    """Build a traceable vendored CodeSLAM depth model."""
    return CodeSLAMDepthModel(ModelConfig()).eval()


def example_input_codeslam() -> torch.Tensor:
    """Return a sample CodeSLAM intensity tensor."""
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("CLDNN Radio", "build_cldnn_radio", "example_input_cldnn_radio", 2016, "CV8-CLDNN-RADIO"),
    (
        "CLDNN-AMC (CNN-LSTM-DNN for radio)",
        "build_cldnn_radio",
        "example_input_cldnn_radio",
        2016,
        "CV8-CLDNN-AMC",
    ),
    ("CodeSLAM", "build_codeslam", "example_input_codeslam", 2018, "CV8-CODESLAM"),
]
