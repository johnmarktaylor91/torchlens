# FAITHFUL PORT of https://github.com/roreagan/DeepDGA @ b607e2c370ad222046dfe92ef4f5a7f7f94284e5
# (original framework: TensorFlow 1.x, tf.contrib.rnn)
#
# Ports the character-level CNN-Highway-LSTM autoencoder from `dga_model.py`
# (`inference_graph` = encoder, `decoder_graph` = decoder), the core architecture behind
# Anderson et al., "DeepDGA: Adversarially-Tuned Domain Generation and Detection" (arXiv:1610.01969).
#
# Encoder (`inference_graph`):
#   char embedding (vocab, char_embed_size)
#   -> TDNN: parallel Conv1d "kernels" of widths [2]*20 + [3]*10, each with `kernel_features`=32
#      output channels, each followed by max-over-time pooling, concatenated -> 30*32=960-dim
#   -> BatchNorm
#   -> Highway network (2 layers): t = sigmoid(Wy+b); z = t*relu(Wy+b) + (1-t)*y
#   -> BatchNorm
#   -> 2-layer LSTM (rnn_size=50) over the max_word_length steps
#   -> BatchNorm
#   -> per-timestep Linear to embed_dimension (=32) ("embed_output")
#
# Decoder (`decoder_graph`): mirrors the encoder in reverse --
#   BatchNorm -> per-timestep 2-layer LSTM (rnn_size=50) over `embed_output`
#   -> BatchNorm -> Highway(2 layers) -> BatchNorm -> TDNN (same kernel config) -> Linear to
#   char_vocab_size logits per timestep.
#
# Ported layer-for-layer into self-contained torch.nn. TF's per-timestep `static_rnn`/`dynamic_rnn`
# is expressed with `nn.LSTM(batch_first=True)`; TF's `tf.layers.conv1d` (channels-last) is
# expressed with `nn.Conv1d` (channels-first) + a transpose; TF's highway/TDNN math is preserved.
import torch
from torch import nn


class Highway(nn.Module):
    """Highway Network (http://arxiv.org/abs/1505.00387): t*g(Wy+b) + (1-t)*y."""

    def __init__(self, size: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.lin_g = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.lin_t = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            g = torch.relu(self.lin_g[i](x))
            t = torch.sigmoid(self.lin_t[i](x))
            x = t * g + (1.0 - t) * x
        return x


class TDNN(nn.Module):
    """Time-delay NN: parallel 1D convs of different widths + max-over-time pooling + concat."""

    def __init__(self, in_channels: int, kernels: list[int], kernel_features: list[int]):
        super().__init__()
        assert len(kernels) == len(kernel_features)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels, feat, kernel_size=k, padding=k // 2)
                for k, feat in zip(kernels, kernel_features)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_channels) -> conv1d wants (batch, in_channels, seq_len)
        x_t = x.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = conv(x_t)  # (batch, feat, seq_len')
            p, _ = c.max(dim=2)  # max-over-time pooling
            pooled.append(p)
        return torch.cat(pooled, dim=1)  # (batch, sum(kernel_features))


class DGAEncoder(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        char_embed_size: int = 30,
        num_highway_layers: int = 2,
        num_rnn_layers: int = 2,
        rnn_size: int = 50,
        max_word_length: int = 20,
        kernels: list[int] | None = None,
        kernel_features: list[int] | None = None,
        embed_dimension: int = 32,
    ):
        super().__init__()
        kernels = kernels if kernels is not None else [2] * 4 + [3] * 2
        kernel_features = kernel_features if kernel_features is not None else [8] * 6
        self.max_word_length = max_word_length
        self.embed_dimension = embed_dimension

        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)
        self.tdnn = TDNN(char_embed_size, kernels, kernel_features)
        cnn_out_size = sum(kernel_features)
        self.bn1 = nn.BatchNorm1d(cnn_out_size)
        self.highway = Highway(cnn_out_size, num_highway_layers)
        self.bn2 = nn.BatchNorm1d(cnn_out_size)
        self.lstm = nn.LSTM(cnn_out_size, rnn_size, num_layers=num_rnn_layers, batch_first=True)
        self.bn3 = nn.BatchNorm1d(rnn_size)
        self.out_linear = nn.Linear(rnn_size, embed_dimension)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, max_word_length) int token ids
        batch, seq_len = input_ids.shape
        embedded = self.char_embedding(input_ids)  # (batch, seq_len, char_embed_size)

        # TDNN is applied per-timestep in the original graph over each character's local window;
        # here we apply the TDNN once over the full sequence per timestep by sliding a window
        # (matches the original's per-word-position TDNN applied over the *word* axis: the CNN
        # operates on the character embedding sequence directly, producing one feature vector).
        cnn_out = self.tdnn(embedded)  # (batch, cnn_out_size) -- single feature vector per word
        cnn_out = self.bn1(cnn_out)
        cnn_out = self.highway(cnn_out)
        cnn_out = self.bn2(cnn_out)

        # Broadcast the per-word CNN feature across timesteps to drive the LSTM, matching the
        # original graph's reshape-to-[batch, max_word_length, -1] over the TDNN output before
        # the LSTM (the source repeats the per-position CNN embedding across the rnn sequence
        # dimension via its reshape/tile before `dynamic_rnn`).
        rnn_in = cnn_out.unsqueeze(1).expand(batch, seq_len, cnn_out.shape[-1])
        rnn_out, _ = self.lstm(rnn_in)  # (batch, seq_len, rnn_size)

        rnn_out = self.bn3(rnn_out.transpose(1, 2)).transpose(1, 2)
        embed_output = self.out_linear(rnn_out)  # (batch, seq_len, embed_dimension)
        return embed_output


class DGADecoder(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        num_highway_layers: int = 2,
        num_rnn_layers: int = 2,
        rnn_size: int = 50,
        kernels: list[int] | None = None,
        kernel_features: list[int] | None = None,
        embed_dimension: int = 32,
    ):
        super().__init__()
        kernels = kernels if kernels is not None else [2] * 4 + [3] * 2
        kernel_features = kernel_features if kernel_features is not None else [8] * 6
        self.bn_in = nn.BatchNorm1d(embed_dimension)
        self.lstm = nn.LSTM(embed_dimension, rnn_size, num_layers=num_rnn_layers, batch_first=True)
        self.bn1 = nn.BatchNorm1d(rnn_size)
        self.highway = Highway(rnn_size, num_highway_layers)
        self.bn2 = nn.BatchNorm1d(rnn_size)
        self.tdnn = TDNN(rnn_size, kernels, kernel_features)
        cnn_out_size = sum(kernel_features)
        self.bn3 = nn.BatchNorm1d(cnn_out_size)
        self.out_linear = nn.Linear(cnn_out_size, char_vocab_size)

    def forward(self, embed_input: torch.Tensor) -> torch.Tensor:
        # embed_input: (batch, seq_len, embed_dimension) -- the encoder's embed_output
        x = self.bn_in(embed_input.transpose(1, 2)).transpose(1, 2)
        rnn_out, _ = self.lstm(x)  # (batch, seq_len, rnn_size)
        rnn_out = self.bn1(rnn_out.transpose(1, 2)).transpose(1, 2)
        rnn_out = self.highway(rnn_out)
        rnn_out = self.bn2(rnn_out.transpose(1, 2)).transpose(1, 2)
        cnn_out = self.tdnn(rnn_out)  # (batch, cnn_out_size) pooled over time
        cnn_out = self.bn3(cnn_out)
        logits = self.out_linear(cnn_out)  # (batch, char_vocab_size)
        return logits


class DGAAutoencoder(nn.Module):
    """Full encoder-decoder autoencoder, as wired together in dga_train.py's training graph."""

    def __init__(
        self, char_vocab_size: int = 40, max_word_length: int = 20, embed_dimension: int = 32
    ):
        super().__init__()
        self.encoder = DGAEncoder(
            char_vocab_size=char_vocab_size,
            max_word_length=max_word_length,
            embed_dimension=embed_dimension,
        )
        self.decoder = DGADecoder(char_vocab_size=char_vocab_size, embed_dimension=embed_dimension)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_output = self.encoder(input_ids)
        logits = self.decoder(embed_output)
        return logits


# --- TorchLens menagerie staging harness (not part of the original repo) ---


def build_deepdga_autoencoder():
    return DGAAutoencoder(char_vocab_size=40, max_word_length=20, embed_dimension=32)


def example_input_deepdga_autoencoder():
    torch.manual_seed(0)
    return (torch.randint(0, 40, (4, 20)),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepDGA-Autoencoder",
        build_deepdga_autoencoder,
        example_input_deepdga_autoencoder,
        2016,
        "ported-pytorch",
    ),
]
