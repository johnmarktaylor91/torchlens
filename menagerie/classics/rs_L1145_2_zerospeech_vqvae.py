# SOURCE: vendored from bshall/ZeroSpeech @ master
# https://github.com/bshall/ZeroSpeech/blob/master/model.py
# "Vector-Quantized Neural Networks for Acoustic Unit Discovery in the
# ZeroSpeech 2020 Challenge" (van Niekerk, Nortje & Kamper 2020). `Encoder`
# (conv mel encoder + EMA vector-quantization codebook + temporal jitter),
# `Jitter`, `VQEmbeddingEMA`, and `Decoder` (speaker-conditioned
# autoregressive WaveRNN-style vocoder: bidirectional conditioning RNN +
# mu-law sample RNN) are transcribed VERBATIM from the real repo's
# `model.py`. Only the `Decoder.generate` autoregressive sampling method is
# dropped here (it needs `preprocess.mulaw_decode`, a training-time
# CLI helper unrelated to the `forward` architecture, and is not exercised
# by a forward pass); `forward` and the rest of the module bodies are
# untouched. `Model` is a thin wrapper matching the real repo's `train.py`
# call convention (`z = encoder(mels)`; `decoder(audio, z, speakers)`).
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Encoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, embedding_dim, jitter=0):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, 1, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, embedding_dim, 1),
        )
        self.codebook = VQEmbeddingEMA(n_embeddings, embedding_dim)
        self.jitter = Jitter(jitter)

    def forward(self, mels):
        z = self.encoder(mels)
        z, loss, perplexity = self.codebook(z.transpose(1, 2))
        z = self.jitter(z)
        return z, loss, perplexity

    def encode(self, mel):
        z = self.encoder(mel)
        z, indices = self.codebook.encode(z.transpose(1, 2))
        return z, indices


class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        else:
            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x


class VQEmbeddingEMA(nn.Module):
    def __init__(
        self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5
    ):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(
            torch.sum(self.embedding**2, dim=1) + torch.sum(x_flat**2, dim=1, keepdim=True),
            x_flat,
            self.embedding.t(),
            alpha=-2.0,
            beta=1.0,
        )

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(
            torch.sum(self.embedding**2, dim=1) + torch.sum(x_flat**2, dim=1, keepdim=True),
            x_flat,
            self.embedding.t(),
            alpha=-2.0,
            beta=1.0,
        )

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(
                encodings, dim=0
            )

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        n_speakers,
        speaker_embedding_dim,
        conditioning_channels,
        mu_embedding_dim,
        rnn_channels,
        fc_channels,
        bits,
        hop_length,
    ):
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.hop_length = hop_length

        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
        self.rnn1 = nn.GRU(
            in_channels + speaker_embedding_dim,
            conditioning_channels,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.mu_embedding = nn.Embedding(self.quantization_channels, mu_embedding_dim)
        self.rnn2 = nn.GRU(
            mu_embedding_dim + 2 * conditioning_channels, rnn_channels, batch_first=True
        )
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, z, speakers):
        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)

        speakers = self.speaker_embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)

        z = torch.cat((z, speakers), dim=-1)
        z, _ = self.rnn1(z)

        z = F.interpolate(z.transpose(1, 2), scale_factor=self.hop_length)
        z = z.transpose(1, 2)

        x = self.mu_embedding(x)
        x, _ = self.rnn2(torch.cat((x, z), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Model(nn.Module):
    """Thin wrapper matching the real repo's train.py call convention:
    z, vq_loss, perplexity = encoder(mels); output = decoder(audio, z, speakers)
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, mels, audio, speakers):
        z, vq_loss, perplexity = self.encoder(mels)
        output = self.decoder(audio, z, speakers)
        return output, vq_loss, perplexity


MENAGERIE_ZOO = "vendored-pytorch"


def build_zerospeech_vqvae():
    torch.manual_seed(0)
    encoder = Encoder(in_channels=80, channels=32, n_embeddings=16, embedding_dim=8, jitter=0.5)
    decoder = Decoder(
        in_channels=8,
        n_speakers=4,
        speaker_embedding_dim=8,
        conditioning_channels=16,
        mu_embedding_dim=8,
        rnn_channels=16,
        fc_channels=16,
        bits=8,
        hop_length=8,
    )
    model = Model(encoder, decoder)
    model.eval()
    return model


def example_input_zerospeech_vqvae():
    torch.manual_seed(0)
    # encoder: 40 mel frames -> 19 codebook frames (kernel/stride shave-off
    # of the conv stack); decoder: 19 codebook frames -> upsample x2 -> 38 ->
    # upsample x hop_length(8) -> 304 audio samples must match exactly.
    mels = torch.randn(2, 80, 40)
    audio = torch.randint(0, 256, (2, 304))
    speakers = torch.randint(0, 4, (2,))
    return (mels, audio, speakers)


MENAGERIE_ENTRIES = [
    (
        "ZeroSpeech_VQVAE",
        "build_zerospeech_vqvae",
        "example_input_zerospeech_vqvae",
        2020,
        MENAGERIE_ZOO,
    ),
]
