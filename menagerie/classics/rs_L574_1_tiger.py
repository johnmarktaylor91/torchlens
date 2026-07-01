# FAITHFUL PORT of daklab/tiger @ main (original framework: TensorFlow / Keras)
# Original: layers.py (SequenceSequentialWithNonSequenceBypass, OneHotInputParser,
#   AlignOneHotEncoding1D) + models.py (Tiger1D)
# TIGER (Targeted Inhibition of Gene Expression via gRNA design) predicts CRISPR-Cas13d
# guide-RNA knockdown efficiency from one-hot-encoded target/guide nucleotide sequences.
# The real repo is TensorFlow/Keras (tf.keras.Sequential + custom Keras Layer subclasses)
# and is not reasonably installable alongside the torch-based base env used here, so the
# Tiger1D architecture (the primary/simplest of the released Tiger variants) is transcribed
# faithfully into base-env torch below: the same one-hot input parsing/splitting, the same
# guide-sequence alignment/concatenation, and the same Conv1D -> MaxPool1D -> Flatten ->
# Dropout -> Dense(128) -> Dense(32) -> Dense(1) stack as models.py:Tiger1D.
# Reference: https://github.com/daklab/tiger
"""TIGER: CNN-based on-target Cas13d guide-RNA efficacy predictor (Tiger1D variant)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# --- ported from layers.py: OneHotInputParser.call ---
class OneHotInputParser(nn.Module):
    def __init__(
        self,
        target_len: int,
        context_5p: int,
        context_3p: int,
        use_guide_seq: bool,
        pad_guide_seq: bool,
    ):
        super().__init__()
        self.target_len = context_5p + target_len + context_3p
        self.guide_len = (self.target_len if pad_guide_seq else target_len) if use_guide_seq else 0

    def forward(self, x):
        n = x.shape[0]
        target_one_hot = x[:, : 4 * self.target_len]
        target_one_hot = target_one_hot.reshape(n, self.target_len, 4)
        guide_one_hot = x[:, 4 * self.target_len : 4 * (self.target_len + self.guide_len)]
        guide_one_hot = guide_one_hot.reshape(n, self.guide_len, 4)
        non_sequence_features = x[:, 4 * (self.target_len + self.guide_len) :]
        return target_one_hot, guide_one_hot, non_sequence_features


# --- ported from layers.py: AlignOneHotEncoding1D.call ---
class AlignOneHotEncoding1D(nn.Module):
    def __init__(self, use_guide_seq: bool):
        super().__init__()
        self.use_guide_seq = use_guide_seq

    def forward(self, target_one_hot, guide_one_hot):
        if not self.use_guide_seq:
            return target_one_hot
        return torch.cat([target_one_hot, guide_one_hot], dim=-1)


# --- ported from models.py: Tiger1D ---
# (SequenceSequentialWithNonSequenceBypass fused directly into forward() below,
# matching the original's data flow: input_parser -> sequence_layers -> concat
# with non-sequence features -> dense head.)
class Tiger1D(nn.Module):
    def __init__(
        self,
        target_len: int,
        context_5p: int,
        context_3p: int,
        use_guide_seq: bool,
        num_scalar_feats: int = 0,
    ):
        super().__init__()
        self.input_parser = OneHotInputParser(
            target_len, context_5p, context_3p, use_guide_seq, pad_guide_seq=True
        )
        self.align = AlignOneHotEncoding1D(use_guide_seq)
        seq_len = self.input_parser.target_len
        in_channels = 8 if use_guide_seq else 4

        # Conv1D(filters=64, kernel_size=4, padding='same') x2 -> MaxPool1D(2, padding='same')
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(64, 64, kernel_size=4, padding="same")
        self.dropout_seq = nn.Dropout(0.25)

        pooled_len = -(-seq_len // 2)  # ceil(seq_len / 2), matches Keras 'same' pooling
        flat_dim = 64 * pooled_len + num_scalar_feats

        self.dense1 = nn.Linear(flat_dim, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(0.1)
        self.dense3 = nn.Linear(32, 1)

    def forward(self, x):
        target_one_hot, guide_one_hot, non_sequence_features = self.input_parser(x)
        seq = self.align(target_one_hot, guide_one_hot)  # (N, L, C)
        seq = seq.transpose(1, 2)  # (N, C, L) for Conv1d

        seq = F.relu(self.conv1(seq))
        seq = F.relu(self.conv2(seq))
        seq = F.max_pool1d(seq, kernel_size=2, ceil_mode=True)
        seq = seq.flatten(1)
        seq = self.dropout_seq(seq)

        x = torch.cat([seq, non_sequence_features], dim=-1)
        x = torch.sigmoid(self.dense1(x))
        x = self.dropout1(x)
        x = torch.sigmoid(self.dense2(x))
        x = self.dropout2(x)
        x = self.dense3(x)
        return x


def build_tiger1d():
    target_len = 8
    context_5p = 2
    context_3p = 2
    use_guide_seq = True
    model = Tiger1D(target_len, context_5p, context_3p, use_guide_seq, num_scalar_feats=0)
    model.eval()
    return model


def example_input_tiger1d():
    seq_len = 12  # context_5p + target_len + context_3p
    guide_len = seq_len  # pad_guide_seq=True
    n = 4 * (seq_len + guide_len)
    return torch.rand(1, n)


MENAGERIE_ENTRIES = [
    (
        "TIGER (Cas13d guide efficacy CNN)",
        "build_tiger1d",
        "example_input_tiger1d",
        2023,
        "ported-pytorch",
    ),
]
