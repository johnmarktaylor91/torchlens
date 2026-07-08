# SOURCE: vendored from daandouwe/grammar-vae @ e2cbb998412558070aa7e2cc1a61fb29f27e86ae
# (src/encoder.py, src/decoder.py, src/model.py -- module bodies only)
#
# GrammarVAE (Kusner, Paige & Hernandez-Lobato, ICML 2017, "Grammar Variational
# Autoencoder", arXiv:1703.01925). The *official* repo (mkusner/grammarVAE) is
# Theano/Keras 1.x and does not import into a modern torch-only environment. The
# `daandouwe/grammar-vae` repo linked from the official README's "unofficial
# implementations" list is a clean, architecture-faithful PyTorch port of the same
# model (Conv1d encoder over one-hot production-rule sequences -> reparameterized
# Gaussian latent -> RNN decoder emitting per-timestep rule logits), so this is a
# real, runnable nn.Module for the family, not a from-scratch reimplementation.
#
# Only `GrammarVAE.forward()` (encode -> sample -> decode -> logits) is used here;
# the `generate()` method (CFG-mask-constrained decoding via `nltk`/`grammar.py`/
# `stack.py`) is grammar-parsing glue for post-hoc valid-expression sampling, not
# part of the trainable architecture, so it and its `nltk` dependency are dropped.
import torch
import torch.nn as nn
from torch.distributions import Normal


class Encoder(nn.Module):
    """Convolutional encoder for Grammar VAE.

    Applies a series of one-dimensional convolutions to a batch
    of one-hot encodings of the sequence of rules that generate
    an artithmetic expression.
    """

    def __init__(self, hidden_dim=20, z_dim=2, conv_size="small"):
        super(Encoder, self).__init__()
        if conv_size == "small":
            # 12 rules, so 12 input channels
            self.conv1 = nn.Conv1d(12, 2, kernel_size=2)
            self.conv2 = nn.Conv1d(2, 3, kernel_size=3)
            self.conv3 = nn.Conv1d(3, 4, kernel_size=4)
            self.linear = nn.Linear(36, hidden_dim)
        elif conv_size == "large":
            self.conv1 = nn.Conv1d(12, 24, kernel_size=2)
            self.conv2 = nn.Conv1d(24, 12, kernel_size=3)
            self.conv3 = nn.Conv1d(12, 12, kernel_size=4)
            self.linear = nn.Linear(108, hidden_dim)
        else:
            raise ValueError(
                "Invallid value for `conv_size`: {}. Must be in [small, large]".format(conv_size)
            )

        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Encode x into a mean and variance of a Normal"""
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.relu(h)
        h = h.view(x.size(0), -1)  # flatten
        h = self.linear(h)
        h = self.relu(h)
        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))
        return mu, sigma


class Decoder(nn.Module):
    """RNN decoder that reconstructs the sequence of rules from laten z"""

    def __init__(self, input_size, hidden_size, output_size, rnn_type="lstm"):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Select rnn_type from [lstm, gru]")

        self.relu = nn.ReLU()

    def forward(self, z, max_length):
        """The forward pass used for training the Grammar VAE.

        For the rnn we follow the same convention as the official keras
        implementaion: the latent z is the input to the rnn at each timestep.
        See line 138 of
            https://github.com/mkusner/grammarVAE/blob/master/models/model_eq.py
        for reference.
        """
        x = self.linear_in(z)
        x = self.relu(x)

        # The input to the rnn is the same for each timestep: it is z.
        x = x.unsqueeze(1).expand(-1, max_length, -1)
        # NOTE: original repo initializes hx as 2-D (batch, hidden); modern
        # torch.nn.LSTM/GRU requires a 3-D (num_layers, batch, hidden) initial
        # state for batch_first inputs, so this adds the leading num_layers=1
        # axis (torch/nn/modules/rnn.py's shape-check tightened since the
        # repo's original torch version; behavior is otherwise unchanged).
        hx = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        hx = (hx, hx) if self.rnn_type == "lstm" else hx

        x, _ = self.rnn(x, hx)

        x = self.relu(x)
        x = self.linear_out(x)
        return x


class GrammarVAE(nn.Module):
    """Grammar Variational Autoencoder"""

    def __init__(self, hidden_encoder_size, z_dim, hidden_decoder_size, output_size, rnn_type):
        super(GrammarVAE, self).__init__()
        self.encoder = Encoder(hidden_encoder_size, z_dim)
        self.decoder = Decoder(z_dim, hidden_decoder_size, output_size, rnn_type)

    def sample(self, mu, sigma):
        """Reparametrized sample from a N(mu, sigma) distribution"""
        normal = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        eps = normal.sample()
        z = mu + eps * torch.sqrt(sigma)
        return z

    def kl(self, mu, sigma):
        """KL divergence between two normal distributions"""
        return torch.mean(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), 1))

    def forward(self, x, max_length=15):
        mu, sigma = self.encoder(x)
        z = self.sample(mu, sigma)
        logits = self.decoder(z, max_length=max_length)
        return logits


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------
def build_grammarvae():
    return GrammarVAE(
        hidden_encoder_size=20,
        z_dim=2,
        hidden_decoder_size=20,
        output_size=12,
        rnn_type="lstm",
    )


def example_input_grammarvae():
    torch.manual_seed(0)
    # (batch, num_rules=12, seq_len=15) one-hot-style float encoding of a
    # production-rule sequence, matching the small `conv_size` encoder above.
    x = torch.randn(2, 12, 15)
    return (x,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("GrammarVAE", build_grammarvae, example_input_grammarvae, 2017, "SOURCE_AVAILABLE"),
]
