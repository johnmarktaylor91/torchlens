"""AudioCraft, AudioLM, AudioSeal, and Bark compact token-audio models.

Sources: Meta AudioCraft docs/GitHub, MusicGen/MAGNeT/JASCO papers and model
docs, AudioLM (Google, 2022), AudioSeal (Meta, 2024), and Hugging Face Bark docs.

The modules below are random-init reconstructions intended for base-environment
TorchLens rendering: codebook-token language models, masked-token refinement,
conditioning fusion, multi-band diffusion decoding, neural watermarking, and
Bark's semantic/coarse/fine GPT stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLM(nn.Module):
    """Compact autoregressive codebook-token Transformer."""

    def __init__(self, vocab: int = 128, dim: int = 48, codebooks: int = 4) -> None:
        """Initialize token, codebook, and conditioning projections."""

        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.codebook = nn.Embedding(codebooks, dim)
        self.cond = nn.Linear(16, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=96, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, vocab)
        self.codebooks = codebooks

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Score interleaved acoustic tokens conditioned on a text/audio vector."""

        tokens, cond = data
        pos_codebook = torch.arange(tokens.shape[1], device=tokens.device) % self.codebooks
        x = self.token(tokens.clamp(0, self.token.num_embeddings - 1))
        x = x + self.codebook(pos_codebook).unsqueeze(0)
        x = x + self.cond(cond).unsqueeze(1)
        mask = torch.full(
            (tokens.shape[1], tokens.shape[1]), float("-inf"), device=tokens.device
        ).triu(1)
        return self.head(self.blocks(x, mask=mask))


class MaskedTokenRefiner(nn.Module):
    """MAGNeT-style non-autoregressive masked codebook refiner."""

    def __init__(self, vocab: int = 128, dim: int = 48) -> None:
        """Initialize masked-token Transformer."""

        super().__init__()
        self.embed = nn.Embedding(vocab + 1, dim)
        self.cond = nn.Linear(16, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=96, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, vocab)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Refine masked audio tokens in parallel."""

        tokens, cond = data
        mask_id = torch.full_like(tokens[:, ::4], self.embed.num_embeddings - 1)
        mixed = tokens.clone()
        mixed[:, ::4] = mask_id
        x = self.embed(mixed.clamp(0, self.embed.num_embeddings - 1))
        x = x + self.cond(cond).unsqueeze(1)
        return self.head(self.blocks(x))


class JascoTokenLM(nn.Module):
    """JASCO-style flow-matching music model with time-aligned controls."""

    def __init__(self, vocab: int = 128, dim: int = 48) -> None:
        """Initialize token and multi-condition projections."""

        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.time = nn.Linear(1, dim)
        self.chord = nn.Embedding(24, 16)
        self.chord_proj = nn.Linear(16, dim)
        self.drum = nn.Conv1d(1, dim, kernel_size=5, padding=2)
        self.melody = nn.Conv1d(12, dim, kernel_size=3, padding=1)
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=96, batch_first=True)
        self.flow = nn.TransformerEncoder(layer, num_layers=2)
        self.velocity = nn.Linear(dim, vocab)

    def forward(
        self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Predict flow velocity from tokens and time-aligned controls."""

        tokens, chords, drums, melody, t = data
        chord_cond = self.chord_proj(self.chord(chords))
        drum_cond = self.drum(drums.unsqueeze(1)).transpose(1, 2)
        melody_cond = self.melody(melody).transpose(1, 2)
        controls = chord_cond + drum_cond + melody_cond
        x = self.token(tokens.clamp(0, self.token.num_embeddings - 1))
        x = x + controls[:, : x.shape[1]] + self.time(t[:, None]).unsqueeze(1)
        return self.velocity(self.flow(x))


class MultiBandDiffusion(nn.Module):
    """Compact MultiBandDiffusion decoder over sub-band waveforms."""

    def __init__(self, bands: int = 4, hidden: int = 32) -> None:
        """Initialize a small band-conditioned diffusion denoiser."""

        super().__init__()
        self.code = nn.Embedding(128, hidden)
        self.time = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.down = nn.Conv1d(bands + hidden, hidden, 5, padding=2)
        self.mid = nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2)
        self.up = nn.Conv1d(hidden, bands, 3, padding=1)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Denoise noisy multi-band audio conditioned on codec tokens and timestep."""

        bands, codes, timestep = data
        cond = self.code(codes).mean(dim=1).unsqueeze(-1).expand(-1, -1, bands.shape[-1])
        cond = cond + self.time(timestep[:, None]).unsqueeze(-1)
        x = torch.cat((bands, cond), dim=1)
        return bands + self.up(F.gelu(self.mid(F.gelu(self.down(x)))))


class AudioSealGenerator(nn.Module):
    """AudioSeal-style localized neural watermark generator."""

    def __init__(self, nbits: int = 16) -> None:
        """Initialize SEANet-like encoder/decoder and message projection."""

        super().__init__()
        self.msg = nn.Linear(nbits, 16)
        self.enc = nn.Conv1d(1, 32, 7, padding=3)
        self.mid = nn.Conv1d(48, 32, 5, padding=2)
        self.dec = nn.Conv1d(32, 1, 7, padding=3)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Embed message bits into a waveform."""

        wav, bits = data
        msg = self.msg(bits).unsqueeze(-1).expand(-1, -1, wav.shape[-1])
        feat = torch.cat((F.gelu(self.enc(wav.unsqueeze(1))), msg), dim=1)
        return wav + 0.01 * torch.tanh(self.dec(F.gelu(self.mid(feat)))[:, 0])


class AudioSealDetector(nn.Module):
    """AudioSeal-style watermark detector and bit reader."""

    def __init__(self, nbits: int = 16) -> None:
        """Initialize convolutional detector."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 24, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(24, 32, 5, padding=2),
            nn.GELU(),
        )
        self.localization = nn.Conv1d(32, 1, 1)
        self.presence = nn.Linear(32, 1)
        self.bits = nn.Linear(32, nbits)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect sample-level watermark presence and recover message logits."""

        feat = self.net(wav.unsqueeze(1))
        local_map = self.localization(feat).squeeze(1)
        pooled = feat.mean(dim=-1)
        return torch.sigmoid(local_map), self.presence(pooled), self.bits(pooled)


class MelodyConditionedMusicGen(nn.Module):
    """MusicGen Melody with an audio/chromagram melody conditioning path."""

    def __init__(self, vocab: int = 128, dim: int = 48) -> None:
        """Initialize token LM plus chromagram encoder."""

        super().__init__()
        self.lm = TokenLM(vocab, dim, codebooks=4)
        self.chroma = nn.Conv1d(12, 16, kernel_size=5, padding=2)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Score audio tokens conditioned on text and melody chroma."""

        tokens, text_cond, chroma = data
        melody_cond = self.chroma(chroma).mean(dim=-1)
        return self.lm((tokens, text_cond + melody_cond))


class StyleConditionedMusicGen(nn.Module):
    """MusicGen Style with a reference-audio style encoder."""

    def __init__(self, vocab: int = 128, dim: int = 48) -> None:
        """Initialize token LM plus reference style encoder."""

        super().__init__()
        self.lm = TokenLM(vocab, dim, codebooks=4)
        self.style = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.GELU(), nn.Conv1d(16, 16, 5, padding=2)
        )
        self.gate = nn.Linear(16, 16)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Score audio tokens conditioned on text and reference style."""

        tokens, text_cond, reference = data
        style_cond = self.style(reference.unsqueeze(1)).mean(dim=-1)
        return self.lm((tokens, text_cond + torch.tanh(self.gate(style_cond))))


class BarkGPT(nn.Module):
    """Bark semantic/coarse/fine GPT-like token model."""

    def __init__(self, vocab: int = 256, dim: int = 48, codebook_idx: int = 0) -> None:
        """Initialize Bark GPT stage."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.codebook = nn.Parameter(torch.full((1, 1, dim), float(codebook_idx)))
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=96, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict next Bark semantic or acoustic tokens."""

        x = self.embed(tokens.clamp(0, self.embed.num_embeddings - 1)) + self.codebook
        mask = torch.full(
            (tokens.shape[1], tokens.shape[1]), float("-inf"), device=tokens.device
        ).triu(1)
        return self.head(self.blocks(x, mask=mask))


class BarkFineTransformer(nn.Module):
    """Bark fine stage with non-causal parallel acoustic-code refinement."""

    def __init__(self, vocab: int = 256, dim: int = 48, codebooks: int = 4) -> None:
        """Initialize bidirectional acoustic refiner."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.codebook = nn.Embedding(codebooks, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=96, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, vocab)
        self.codebooks = codebooks

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Refine all acoustic codebooks in parallel without a causal mask."""

        code = torch.arange(tokens.shape[1], device=tokens.device) % self.codebooks
        x = self.embed(tokens.clamp(0, self.embed.num_embeddings - 1)) + self.codebook(
            code
        ).unsqueeze(0)
        return self.head(self.blocks(x))


class AudioLMCascade(nn.Module):
    """AudioLM semantic-to-acoustic token cascade."""

    def __init__(self) -> None:
        """Initialize semantic and acoustic token LMs."""

        super().__init__()
        self.semantic = TokenLM(vocab=128, dim=48, codebooks=1)
        self.acoustic = TokenLM(vocab=128, dim=48, codebooks=4)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run semantic then acoustic token modeling."""

        sem, ac, cond = data
        sem_logits = self.semantic((sem, cond)).mean(dim=1)
        return self.acoustic((ac, sem_logits[:, :16]))


def build_audiogen() -> nn.Module:
    """Build AudioGen-style text-to-audio token LM."""

    return TokenLM()


def build_musicgen() -> nn.Module:
    """Build MusicGen-style single-stage EnCodec token LM."""

    return TokenLM(codebooks=4)


def build_musicgen_melody() -> nn.Module:
    """Build MusicGen Melody-style conditioned token LM."""

    return MelodyConditionedMusicGen()


def build_musicgen_style() -> nn.Module:
    """Build MusicGen Style-style conditioned token LM."""

    return StyleConditionedMusicGen()


def build_magnet() -> nn.Module:
    """Build MAGNeT-style masked token model."""

    return MaskedTokenRefiner()


def build_jasco() -> nn.Module:
    """Build JASCO-style controllable music token LM."""

    return JascoTokenLM()


def build_multibanddiffusion() -> nn.Module:
    """Build MultiBandDiffusion-style decoder."""

    return MultiBandDiffusion()


def build_audioseal_generator() -> nn.Module:
    """Build AudioSeal watermark generator."""

    return AudioSealGenerator()


def build_audioseal_detector() -> nn.Module:
    """Build AudioSeal watermark detector."""

    return AudioSealDetector()


def build_audiolm() -> nn.Module:
    """Build AudioLM semantic-acoustic cascade."""

    return AudioLMCascade()


def build_bark_text() -> nn.Module:
    """Build Bark text/semantic GPT."""

    return BarkGPT(codebook_idx=0)


def build_bark_fine() -> nn.Module:
    """Build Bark fine non-causal acoustic refiner."""

    return BarkFineTransformer()


def build_bark_coarse() -> nn.Module:
    """Build Bark coarse acoustic GPT."""

    return BarkGPT(codebook_idx=1)


def example_tokens() -> tuple[torch.Tensor, torch.Tensor]:
    """Return token LM inputs."""

    return torch.randint(0, 128, (1, 16)), torch.randn(1, 16)


def example_musicgen_melody() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MusicGen Melody token, text, and chromagram inputs."""

    return torch.randint(0, 128, (1, 16)), torch.randn(1, 16), torch.randn(1, 12, 16)


def example_musicgen_style() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MusicGen Style token, text, and reference-audio inputs."""

    return torch.randint(0, 128, (1, 16)), torch.randn(1, 16), torch.randn(1, 64)


def example_jasco() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return JASCO control inputs."""

    return (
        torch.randint(0, 128, (1, 16)),
        torch.randint(0, 24, (1, 16)),
        torch.randn(1, 16),
        torch.randn(1, 12, 16),
        torch.tensor([0.35]),
    )


def example_bands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return multi-band diffusion inputs."""

    return torch.randn(1, 4, 64), torch.randint(0, 128, (1, 12)), torch.tensor([0.4])


def example_watermark() -> tuple[torch.Tensor, torch.Tensor]:
    """Return AudioSeal generator inputs."""

    return torch.randn(1, 256), torch.rand(1, 16)


def example_detector() -> torch.Tensor:
    """Return AudioSeal detector input."""

    return torch.randn(1, 256)


def example_bark() -> torch.Tensor:
    """Return Bark token input."""

    return torch.randint(0, 256, (1, 16))


def example_audiolm() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return AudioLM cascade inputs."""

    return torch.randint(0, 128, (1, 10)), torch.randint(0, 128, (1, 16)), torch.randn(1, 16)


MENAGERIE_ENTRIES = [
    ("audiocraft_audiogen", "build_audiogen", "example_tokens", "2023", "audio"),
    ("audiocraft.AudioSeal", "build_audioseal_detector", "example_detector", "2024", "audio"),
    ("audiocraft_jasco", "build_jasco", "example_jasco", "2024", "audio"),
    ("audiocraft_magnet", "build_magnet", "example_tokens", "2024", "audio"),
    ("audiocraft_multibanddiffusion", "build_multibanddiffusion", "example_bands", "2023", "audio"),
    ("audiocraft_musicgen", "build_musicgen", "example_tokens", "2023", "audio"),
    (
        "audiocraft_musicgen_melody",
        "build_musicgen_melody",
        "example_musicgen_melody",
        "2023",
        "audio",
    ),
    ("audiocraft.MusicGenStyle", "build_musicgen_style", "example_musicgen_style", "2024", "audio"),
    (
        "audiocraft_musicgen_style",
        "build_musicgen_style",
        "example_musicgen_style",
        "2024",
        "audio",
    ),
    ("AudioLM-SemanticAcousticTokenLM", "build_audiolm", "example_audiolm", "2022", "audio"),
    ("audioseal", "build_audioseal_detector", "example_detector", "2024", "audio"),
    ("AudioSealDetector", "build_audioseal_detector", "example_detector", "2024", "audio"),
    ("AudioSealWM", "build_audioseal_generator", "example_watermark", "2024", "audio"),
    ("bark_coarse_transformer", "build_bark_coarse", "example_bark", "2023", "audio"),
    ("bark_fine_transformer", "build_bark_fine", "example_bark", "2023", "audio"),
    ("bark_text_transformer", "build_bark_text", "example_bark", "2023", "audio"),
]
