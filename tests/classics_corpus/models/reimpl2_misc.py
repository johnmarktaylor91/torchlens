"""Miscellaneous compact reimplementations for dependency-gated targets."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FedProxCNN(nn.Module):
    """Small MNIST CNN commonly used with FedProx/FedAvg experiments."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize convolutional classifier.

        Parameters
        ----------
        classes:
            Output class count.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify client images.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.net(x)


class LeNet300100(nn.Module):
    """LeNet-300-100 multilayer perceptron for MNIST."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize 300 and 100 unit hidden layers.

        Parameters
        ----------
        classes:
            Output class count.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify MNIST images.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.net(x)


class JordanRNNClassifier(nn.Module):
    """Jordan recurrent network with output-fed context units."""

    def __init__(self, in_features: int = 6, hidden: int = 12, classes: int = 4) -> None:
        """Initialize recurrent and output projections.

        Parameters
        ----------
        in_features:
            Input feature count per step.
        hidden:
            Hidden state size.
        classes:
            Output class count.
        """

        super().__init__()
        self.in_proj = nn.Linear(in_features + classes, hidden)
        self.out_proj = nn.Linear(hidden, classes)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Process a sequence with output feedback context.

        Parameters
        ----------
        seq:
            Sequence tensor ``(batch, time, features)``.

        Returns
        -------
        torch.Tensor
            Final class logits.
        """

        context = torch.zeros(seq.shape[0], self.out_proj.out_features, device=seq.device)
        for step in range(seq.shape[1]):
            hidden = torch.tanh(self.in_proj(torch.cat([seq[:, step], context], dim=1)))
            context = torch.softmax(self.out_proj(hidden), dim=-1)
        return context


class GloVeBilinear(nn.Module):
    """GloVe global log-bilinear co-occurrence scorer."""

    def __init__(self, vocab: int = 32, dim: int = 16) -> None:
        """Initialize word/context embeddings and biases.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Embedding dimension.
        """

        super().__init__()
        self.word = nn.Embedding(vocab, dim)
        self.context = nn.Embedding(vocab, dim)
        self.word_bias = nn.Embedding(vocab, 1)
        self.context_bias = nn.Embedding(vocab, 1)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        """Score word-context co-occurrence pairs.

        Parameters
        ----------
        pairs:
            Integer pairs ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Log co-occurrence predictions.
        """

        word_ids, ctx_ids = pairs[:, 0], pairs[:, 1]
        score = (self.word(word_ids) * self.context(ctx_ids)).sum(dim=-1)
        bias = self.word_bias(word_ids).squeeze(-1) + self.context_bias(ctx_ids).squeeze(-1)
        return score + bias


class _InvertibleOneByOne(nn.Module):
    """Invertible 1x1 convolution used by Glow/WaveGlow flows."""

    def __init__(self, channels: int) -> None:
        """Initialize an orthogonal mixing matrix.

        Parameters
        ----------
        channels:
            Channel count.
        """

        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        self.weight = nn.Parameter(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix channels with a 1x1 convolution.

        Parameters
        ----------
        x:
            Audio tensor.

        Returns
        -------
        torch.Tensor
            Mixed audio tensor.
        """

        return F.conv1d(x, self.weight.unsqueeze(-1))


class WaveGlowCompact(nn.Module):
    """Compact WaveGlow flow with invertible 1x1 convs and affine couplings."""

    def __init__(self, channels: int = 8, mel_channels: int = 16, flows: int = 3) -> None:
        """Initialize flow steps.

        Parameters
        ----------
        channels:
            Audio group channels.
        mel_channels:
            Conditioning mel channels.
        flows:
            Number of flow steps.
        """

        super().__init__()
        self.pre = nn.Conv1d(1, channels, 4, stride=4)
        self.cond = nn.Conv1d(mel_channels, channels // 2, 1)
        self.mix = nn.ModuleList([_InvertibleOneByOne(channels) for _ in range(flows)])
        self.coupling = nn.ModuleList(
            [nn.Conv1d(channels, channels, 3, padding=1) for _ in range(flows)]
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Transform waveform to a latent flow representation.

        Parameters
        ----------
        inputs:
            Tuple of waveform and mel conditioning.

        Returns
        -------
        torch.Tensor
            Flow latent tensor.
        """

        audio, mel = inputs
        z = self.pre(audio)
        cond = F.interpolate(self.cond(mel), size=z.shape[-1], mode="nearest")
        for mix, coupling in zip(self.mix, self.coupling):
            z = mix(z)
            z_keep, z_change = z.chunk(2, dim=1)
            shift, log_scale = coupling(torch.cat((z_keep, cond), dim=1)).chunk(2, dim=1)
            z_change = z_change * torch.exp(torch.tanh(log_scale)) + shift
            z = torch.cat((z_keep, z_change), dim=1)
        return z


class GradTTSCompact(nn.Module):
    """Compact Grad-TTS text encoder plus score-based mel decoder."""

    def __init__(self, vocab: int = 48, hidden: int = 32, mel: int = 16) -> None:
        """Initialize encoder, duration predictor, and score network.

        Parameters
        ----------
        vocab:
            Text vocabulary size.
        hidden:
            Hidden width.
        mel:
            Mel channel count.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, 4, hidden * 2, batch_first=True),
            num_layers=1,
        )
        self.duration = nn.Linear(hidden, 1)
        self.prior = nn.Linear(hidden, mel)
        self.score = nn.Sequential(
            nn.Conv1d(mel + mel, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, mel, 3, padding=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate compact mel scores from text tokens.

        Parameters
        ----------
        tokens:
            Text token ids.

        Returns
        -------
        torch.Tensor
            Denoised mel estimate.
        """

        enc = self.encoder(self.embed(tokens))
        _dur = F.softplus(self.duration(enc))
        prior = self.prior(enc).transpose(1, 2)
        noise = torch.randn_like(prior)
        score = self.score(torch.cat([noise, prior], dim=1))
        return noise - 0.1 * score


class FreGANGenerator(nn.Module):
    """Fre-GAN-style resolution-connected vocoder generator."""

    def __init__(self, mel: int = 16, channels: int = 32) -> None:
        """Initialize upsampling and resolution connections.

        Parameters
        ----------
        mel:
            Mel channel count.
        channels:
            Hidden channel count.
        """

        super().__init__()
        self.pre = nn.Conv1d(mel, channels, 7, padding=3)
        self.up1 = nn.ConvTranspose1d(channels, channels // 2, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(channels // 2, channels // 4, 4, stride=2, padding=1)
        self.low = nn.Conv1d(channels, channels // 4, 1)
        self.out = nn.Conv1d(channels // 4, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from mel features.

        Parameters
        ----------
        mel:
            Mel spectrogram.

        Returns
        -------
        torch.Tensor
            Waveform estimate.
        """

        h0 = F.leaky_relu(self.pre(mel), 0.1)
        h1 = F.leaky_relu(self.up1(h0), 0.1)
        h2 = F.leaky_relu(self.up2(h1), 0.1)
        skip = F.interpolate(self.low(h0), size=h2.shape[-1], mode="nearest")
        return torch.tanh(self.out(h2 + skip))


def build_fedprox_cnn() -> nn.Module:
    """Build FedProx CNN.

    Returns
    -------
    nn.Module
        CNN model.
    """

    return FedProxCNN()


def build_lenet_300_100() -> nn.Module:
    """Build LeNet-300-100.

    Returns
    -------
    nn.Module
        MLP model.
    """

    return LeNet300100()


def build_jordan_rnn_classifier() -> nn.Module:
    """Build Jordan RNN classifier.

    Returns
    -------
    nn.Module
        Jordan RNN.
    """

    return JordanRNNClassifier()


def build_glove_bilinear() -> nn.Module:
    """Build GloVe bilinear scorer.

    Returns
    -------
    nn.Module
        GloVe scorer.
    """

    return GloVeBilinear()


def build_waveglow() -> nn.Module:
    """Build compact WaveGlow.

    Returns
    -------
    nn.Module
        WaveGlow flow.
    """

    return WaveGlowCompact()


def build_grad_tts() -> nn.Module:
    """Build compact Grad-TTS.

    Returns
    -------
    nn.Module
        Grad-TTS model.
    """

    return GradTTSCompact()


def build_fregan_generator() -> nn.Module:
    """Build compact Fre-GAN generator.

    Returns
    -------
    nn.Module
        Fre-GAN generator.
    """

    return FreGANGenerator()


def example_mnist() -> torch.Tensor:
    """Create MNIST-like input.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(2, 1, 28, 28)


def example_sequence() -> torch.Tensor:
    """Create sequence input.

    Returns
    -------
    torch.Tensor
        Sequence tensor.
    """

    return torch.randn(2, 5, 6)


def example_pairs() -> torch.Tensor:
    """Create word-context pairs.

    Returns
    -------
    torch.Tensor
        Pair ids.
    """

    return torch.tensor([[1, 2], [3, 4], [5, 6]])


def example_waveglow() -> tuple[torch.Tensor, torch.Tensor]:
    """Create waveform and mel conditioning.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Audio and mel tensors.
    """

    return torch.randn(1, 1, 128), torch.randn(1, 16, 32)


def example_tokens() -> torch.Tensor:
    """Create token ids.

    Returns
    -------
    torch.Tensor
        Token tensor.
    """

    return torch.randint(0, 48, (1, 10))


def example_mel() -> torch.Tensor:
    """Create mel spectrogram.

    Returns
    -------
    torch.Tensor
        Mel tensor.
    """

    return torch.randn(1, 16, 16)


MENAGERIE_ENTRIES = [
    ("FedProx-CNN", "build_fedprox_cnn", "example_mnist", "2018", "E5"),
    ("LeNet-300-100", "build_lenet_300_100", "example_mnist", "1998", "E3"),
    ("JordanRNNClassifier", "build_jordan_rnn_classifier", "example_sequence", "1986", "E3"),
    ("GloVe-Bilinear", "build_glove_bilinear", "example_pairs", "2014", "E4"),
    ("waveglow", "build_waveglow", "example_waveglow", "2018", "E5"),
    ("Grad-TTS", "build_grad_tts", "example_tokens", "2021", "E5"),
    ("fregan_generator", "build_fregan_generator", "example_mel", "2021", "E5"),
]
