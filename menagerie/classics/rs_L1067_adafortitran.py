# SOURCE: vendored from https://github.com/BerkIGuler/AdaFortiTran @ master
# (src/models/adafortitran.py, src/models/fortitran.py, src/models/linear.py,
#  src/models/blocks/{channel_adaptivity,encoders,enhancers,patch_processors,
#  positional_encodings}.py, src/config/schemas.py)
#
# AdaFortiTran / FortiTran (ICC 2025): Adaptive Hybrid CNN-Transformer Channel
# Estimator for OFDM systems under high Doppler shift. The classes below are the
# REAL model code from the official repo -- a linear pilot-to-OFDM-grid upsampler,
# a small CNN "ConvEnhancer" refinement block, a patch-embedding + Transformer
# encoder for long-range dependencies, patch reconstruction, and (for the
# adaptive "Ada" variant) a channel-condition MLP token encoder (SNR / delay
# spread / Doppler shift) concatenated into the transformer sequence. No
# architecture was altered; only the package-relative imports
# (`from src.config.schemas import ...`, `from .blocks import ...`) were
# collapsed into this single file, and the pydantic `SystemConfig`/`ModelConfig`
# schemas are the real validated config classes from the repo (pydantic is an
# installed base lib).

import torch
import torch.nn as nn
from typing import List, Literal, Optional, Self, Tuple

from pydantic import BaseModel, Field, model_validator

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# src/config/schemas.py (real pydantic config classes, unmodified)
# ---------------------------------------------------------------------------
class OFDMParams(BaseModel):
    num_scs: int = Field(..., gt=0, description="Number of OFDM subcarriers")
    num_symbols: int = Field(..., gt=0, description="Number of OFDM symbols")


class PilotParams(BaseModel):
    num_scs: int = Field(..., gt=0, description="Number of pilots across sub-carriers")
    num_symbols: int = Field(..., gt=0, description="Number of pilots across OFDM symbols")


class SystemConfig(BaseModel):
    """System configuration for OFDM and pilot parameters."""

    ofdm: OFDMParams
    pilot: PilotParams

    @model_validator(mode="after")
    def validate_pilot_constraints(self) -> Self:
        if self.pilot.num_scs > self.ofdm.num_scs:
            raise ValueError(
                f"Pilot sub-carriers ({self.pilot.num_scs}) cannot exceed "
                f"OFDM sub-carriers ({self.ofdm.num_scs})"
            )
        if self.pilot.num_symbols > self.ofdm.num_symbols:
            raise ValueError(
                f"Pilot symbols ({self.pilot.num_symbols}) cannot exceed "
                f"OFDM symbols ({self.ofdm.num_symbols})"
            )
        return self

    model_config = {"extra": "forbid"}


class BaseConfig(BaseModel):
    pass


class ModelConfig(BaseConfig):
    model_type: Literal["linear", "fortitran", "adafortitran"] = Field(default="linear")
    patch_size: Optional[Tuple[int, int]] = Field(default=None)
    num_layers: Optional[int] = Field(default=None, gt=0)
    model_dim: Optional[int] = Field(default=None, gt=0)
    num_head: Optional[int] = Field(default=None, gt=0)
    activation: Literal["relu", "gelu"] = Field(default="gelu")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    max_seq_len: int = Field(default=512, gt=0)
    pos_encoding_type: Literal["learnable", "sinusoidal"] = Field(default="learnable")
    adaptive_token_length: Optional[int] = Field(default=None, gt=0)
    channel_adaptivity_hidden_sizes: Optional[List[int]] = Field(default=None)

    @model_validator(mode="after")
    def validate_model_specific_requirements(self) -> Self:
        if self.model_type == "linear":
            pass
        elif self.model_type in ["fortitran", "adafortitran"]:
            required_fields = ["patch_size", "num_layers", "model_dim", "num_head"]
            for field in required_fields:
                if getattr(self, field) is None:
                    raise ValueError(f"{field} is required for {self.model_type} model")
            if self.model_type == "adafortitran":
                if self.channel_adaptivity_hidden_sizes is None:
                    raise ValueError(
                        "channel_adaptivity_hidden_sizes is required for AdaFortiTran model"
                    )
                if self.adaptive_token_length is None:
                    raise ValueError("adaptive_token_length is required for AdaFortiTran model")
            elif self.model_type == "fortitran":
                if self.channel_adaptivity_hidden_sizes is not None:
                    raise ValueError(
                        "channel_adaptivity_hidden_sizes should not be provided for FortiTran model"
                    )
                if self.adaptive_token_length is not None:
                    raise ValueError(
                        "adaptive_token_length should not be provided for FortiTran model"
                    )
        return self

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# src/models/blocks/positional_encodings.py (real code)
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding"""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for transformers."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# src/models/blocks/encoders.py (real code)
# ---------------------------------------------------------------------------
class TransformerEncoderForChannels(nn.Module):
    """Transformer encoder for channels"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_dim: int = 128,
        num_head: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
        num_layers: int = 3,
        max_len: int = 512,
        pos_encoding_type: str = "learnable",
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, model_dim)
        if pos_encoding_type == "learnable":
            self.positional_encoding = LearnablePositionalEncoding(max_len, model_dim)
        elif pos_encoding_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(max_len, model_dim)
        else:
            raise ValueError("pos_encoding_type must be 'learnable' or 'sinusoidal'")

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_head,
            dim_feedforward=2 * model_dim,
            activation=activation,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.linear_2 = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return self.linear_2(x)


# ---------------------------------------------------------------------------
# src/models/blocks/channel_adaptivity.py (real code)
# ---------------------------------------------------------------------------
class ChannelAdapter(nn.Module):
    """Nonlinear encoder for channel condition tokens."""

    def __init__(self, hidden_sizes: Tuple[int, int, int]) -> None:
        super().__init__()
        self.snr_encoder = self._create_mlp(hidden_sizes)
        self.ds_encoder = self._create_mlp(hidden_sizes)
        self.dop_encoder = self._create_mlp(hidden_sizes)

    @staticmethod
    def _create_mlp(hidden_sizes: Tuple[int, int, int]) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
        )

    def forward(
        self,
        snr: torch.Tensor,
        delay_spread: torch.Tensor,
        doppler_shift: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = snr.shape[0]
        snr_emb = torch.reshape(self.snr_encoder(snr), (batch_size, -1, 2))
        ds_emb = torch.reshape(self.ds_encoder(delay_spread), (batch_size, -1, 2))
        dop_emb = torch.reshape(self.dop_encoder(doppler_shift), (batch_size, -1, 2))
        return torch.cat((snr_emb, ds_emb, dop_emb), dim=2)


# ---------------------------------------------------------------------------
# src/models/blocks/patch_processors.py (real code)
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Transform channel matrix into patch embeddings (sequence of flattened vectors)"""

    def __init__(self, patch_size: Tuple[int, int] = (3, 2)) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unfold(torch.unsqueeze(x, dim=1))
        return torch.permute(x, dims=(0, 2, 1))


class InversePatchEmbedding(nn.Module):
    """Transform patch embeddings back to original matrix format."""

    def __init__(
        self,
        output_size: Tuple[int, int] = (120, 14),
        patch_size: Tuple[int, int] = (3, 2),
    ) -> None:
        super().__init__()
        self.fold = nn.Fold(output_size=output_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, dims=(0, 2, 1))
        x = self.fold(x)
        return torch.squeeze(x, dim=1)


# ---------------------------------------------------------------------------
# src/models/blocks/enhancers.py (real code)
# ---------------------------------------------------------------------------
class ConvEnhancer(nn.Module):
    """Convolutional enhancement network with 1->8->32->8->1 channel pattern."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


# ---------------------------------------------------------------------------
# src/models/fortitran.py (real code)
# ---------------------------------------------------------------------------
class BaseFortiTranEstimator(nn.Module):
    """Base Hybrid CNN-Transformer Channel Estimator for OFDM Systems."""

    def __init__(
        self,
        system_config: SystemConfig,
        model_config: ModelConfig,
        device: str = "cpu",
        use_channel_adaptation: bool = False,
    ) -> None:
        super().__init__()

        self.system_config = system_config
        self.model_config = model_config
        self.use_channel_adaptation = use_channel_adaptation
        self.device = torch.device(device)

        self._setup_dimensions()
        self._build_architecture()
        self.to(self.device)

    def _setup_dimensions(self) -> None:
        self.ofdm_size = (
            self.system_config.ofdm.num_scs,
            self.system_config.ofdm.num_symbols,
        )
        self.pilot_size = (
            self.system_config.pilot.num_scs,
            self.system_config.pilot.num_symbols,
        )
        self.pilot_features = self.pilot_size[0] * self.pilot_size[1]
        self.ofdm_features = self.ofdm_size[0] * self.ofdm_size[1]
        self.patch_length = self.model_config.patch_size[0] * self.model_config.patch_size[1]

        if self.use_channel_adaptation:
            if self.model_config.adaptive_token_length is None:
                raise ValueError(
                    "adaptive_token_length must be set when channel adaptation is enabled"
                )
            self.transformer_input_dim = self.patch_length + self.model_config.adaptive_token_length
        else:
            self.transformer_input_dim = self.patch_length

    def _build_architecture(self) -> None:
        self.pilot_upsampler = nn.Linear(self.pilot_features, self.ofdm_features)
        self.initial_enhancer = ConvEnhancer()
        self.patch_embedder = PatchEmbedding(self.model_config.patch_size)

        if self.use_channel_adaptation:
            if self.model_config.channel_adaptivity_hidden_sizes is None:
                raise ValueError(
                    "channel_adaptivity_hidden_sizes must be set when channel adaptation is enabled"
                )
            hidden_sizes = tuple(self.model_config.channel_adaptivity_hidden_sizes)
            if len(hidden_sizes) != 3:
                raise ValueError("channel_adaptivity_hidden_sizes must have exactly 3 values")
            self.channel_adapter = ChannelAdapter(hidden_sizes)

        transformer_output_dim = self.patch_length

        self.transformer_encoder = TransformerEncoderForChannels(
            input_dim=self.transformer_input_dim,
            output_dim=transformer_output_dim,
            model_dim=self.model_config.model_dim,
            num_head=self.model_config.num_head,
            activation=self.model_config.activation,
            dropout=self.model_config.dropout,
            num_layers=self.model_config.num_layers,
            max_len=self.model_config.max_seq_len,
            pos_encoding_type=self.model_config.pos_encoding_type,
        )

        self.patch_reconstructor = InversePatchEmbedding(
            self.ofdm_size, self.model_config.patch_size
        )
        self.final_refiner = ConvEnhancer()

    def forward(
        self, pilot_symbols: torch.Tensor, meta_data: Optional[Tuple] = None
    ) -> torch.Tensor:
        if self.use_channel_adaptation and meta_data is None:
            raise ValueError("meta_data is required when channel adaptation is enabled")

        channel_conditions = None
        if self.use_channel_adaptation and meta_data is not None:
            _, snr, delay_spread, max_dop_shift, _, _ = meta_data
            channel_conditions = [
                tensor.to(self.device) for tensor in (snr, delay_spread, max_dop_shift)
            ]

        pilot_symbols = pilot_symbols.to(self.device)

        real_estimate = self._forward_real_valued(pilot_symbols.real, channel_conditions)
        imag_estimate = self._forward_real_valued(pilot_symbols.imag, channel_conditions)

        channel_estimate = torch.complex(real_estimate, imag_estimate)

        return channel_estimate

    def _forward_real_valued(
        self,
        x: torch.Tensor,
        channel_conditions: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        if x.dim() > 2:
            x = x.view(batch_size, -1)

        upsampled = self.pilot_upsampler(x)
        upsampled_2d = upsampled.view(batch_size, 1, *self.ofdm_size)

        conv_enhanced = torch.squeeze(self.initial_enhancer(upsampled_2d), dim=1)

        patch_embeddings = self.patch_embedder(conv_enhanced)

        if self.use_channel_adaptation and channel_conditions is not None:
            encoded_channel_condition = self.channel_adapter(*channel_conditions)
            transformer_input = torch.cat((patch_embeddings, encoded_channel_condition), dim=2)
        else:
            transformer_input = patch_embeddings

        transformer_output = self.transformer_encoder(transformer_input)

        reconstructed = self.patch_reconstructor(transformer_output)

        residual_combined = conv_enhanced + reconstructed

        refined_output = torch.squeeze(
            self.final_refiner(torch.unsqueeze(residual_combined, dim=1)), dim=1
        )

        return refined_output


class FortiTranEstimator(BaseFortiTranEstimator):
    """Standard Hybrid CNN-Transformer Channel Estimator for OFDM Systems (no channel adaptation)."""

    def __init__(
        self, system_config: SystemConfig, model_config: ModelConfig, device: str = "cpu"
    ) -> None:
        super().__init__(system_config, model_config, device=device, use_channel_adaptation=False)


# ---------------------------------------------------------------------------
# src/models/adafortitran.py (real code)
# ---------------------------------------------------------------------------
class AdaFortiTranEstimator(BaseFortiTranEstimator):
    """Adaptive Hybrid CNN-Transformer Channel Estimator for OFDM Systems with channel adaptation."""

    def __init__(
        self, system_config: SystemConfig, model_config: ModelConfig, device: str = "cpu"
    ) -> None:
        super().__init__(system_config, model_config, device=device, use_channel_adaptation=True)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------
# Tiny OFDM grid: ofdm=(6,4) divisible by patch_size=(3,2) -> patch_length=6,
# num_patches = (6//3)*(4//2) = 4. Real config/adafortitran.yaml values used
# for architecture knobs (patch_size, num_layers, model_dim, num_head,
# activation, dropout, pos_encoding_type), shrunk (num_layers, model_dim) only
# for a fast CPU trace.
_OFDM_SIZE = (6, 4)
_PILOT_SIZE = (3, 2)
_BATCH = 2


def _build_configs(adaptive: bool) -> Tuple[SystemConfig, ModelConfig]:
    system_config = SystemConfig(
        ofdm={"num_scs": _OFDM_SIZE[0], "num_symbols": _OFDM_SIZE[1]},
        pilot={"num_scs": _PILOT_SIZE[0], "num_symbols": _PILOT_SIZE[1]},
    )
    if adaptive:
        model_config = ModelConfig(
            model_type="adafortitran",
            patch_size=(3, 2),
            num_layers=2,
            model_dim=8,
            num_head=2,
            activation="gelu",
            dropout=0.1,
            max_seq_len=32,
            pos_encoding_type="learnable",
            channel_adaptivity_hidden_sizes=[4, 8, 8],
            adaptive_token_length=6,
        )
    else:
        model_config = ModelConfig(
            model_type="fortitran",
            patch_size=(3, 2),
            num_layers=2,
            model_dim=8,
            num_head=2,
            activation="gelu",
            dropout=0.1,
            max_seq_len=32,
            pos_encoding_type="learnable",
        )
    return system_config, model_config


def build_adafortitran():
    torch.manual_seed(0)
    system_config, model_config = _build_configs(adaptive=True)
    model = AdaFortiTranEstimator(system_config, model_config, device="cpu")
    model.eval()
    return model


def example_input_adafortitran():
    torch.manual_seed(0)
    real = torch.randn(_BATCH, *_PILOT_SIZE)
    imag = torch.randn(_BATCH, *_PILOT_SIZE)
    pilot_symbols = torch.complex(real, imag)
    snr = torch.randn(_BATCH, 1)
    delay_spread = torch.randn(_BATCH, 1)
    max_dop_shift = torch.randn(_BATCH, 1)
    meta_data = (None, snr, delay_spread, max_dop_shift, None, None)
    return (pilot_symbols, meta_data)


def build_fortitran():
    torch.manual_seed(0)
    system_config, model_config = _build_configs(adaptive=False)
    model = FortiTranEstimator(system_config, model_config, device="cpu")
    model.eval()
    return model


def example_input_fortitran():
    torch.manual_seed(0)
    real = torch.randn(_BATCH, *_PILOT_SIZE)
    imag = torch.randn(_BATCH, *_PILOT_SIZE)
    pilot_symbols = torch.complex(real, imag)
    return (pilot_symbols,)


MENAGERIE_ENTRIES = [
    ("AdaFortiTran", "build_adafortitran", "example_input_adafortitran", 2025, MENAGERIE_ZOO),
    ("FortiTran", "build_fortitran", "example_input_fortitran", 2025, MENAGERIE_ZOO),
]
