# SOURCE: vendored from dlehdgus99/essay_scoring @ main
# (train.ipynb, code cells defining `Attention` and `EssayScoringModel`, copied
# verbatim -- the repo ships the model only as inline notebook cells, no separate
# .py model file)
#
# Prompt-agnostic automated essay scoring (AES) model: a Conv1d feature extractor
# over one-hot POS-tag embeddings, an attention-pooling layer after the conv, a
# bidirectional LSTM over the pooled conv features, a second attention-pooling
# layer after the LSTM, then a 3-layer MLP regression head that scores the essay
# after concatenating the LSTM-attention output with a vector of hand-engineered
# readability/spelling/linguistic/sentiment features (textstat, spellchecker,
# spaCy POS/tense, VADER sentiment -- computed outside the model at data-prep
# time, so the model itself takes the already-flattened feature vector as input).
# Only base-lib deps at the nn.Module level: torch. (The queue notes this
# candidate as "DANN-based domain adaptation"; the actual repo shipped is this
# CNN+BiLSTM+attention regressor with no adversarial/domain-discriminator branch
# -- vendored as the real code actually present in the repo, not the queue's
# description.)

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)
        return context.squeeze(1), attn_weights


class EssayScoringModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, feature_dim, dropout_prob):
        super(EssayScoringModel, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding="same")
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm_attention = Attention(hidden_dim * 2)
        self.conv1_attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim * 2 + feature_dim, 256)  # Increased capacity
        self.fc2 = nn.Linear(256, 128)  # Added another fully connected layer
        self.fc3 = nn.Linear(128, 1)

    def forward(self, pos_tags, features):
        # Embedding layer
        # Shape: (batch_size, seq_length, num_classes)
        embeds = F.one_hot(pos_tags, num_classes=50).float()

        # Convolutional layer
        # Shape: (batch_size, hidden_dim, seq_length)
        conv_out = F.relu(self.conv1(embeds.transpose(1, 2)))
        conv_out = self.dropout(conv_out)

        # Attention pooling after convolution
        attn_conv_out = self.conv1_attention(conv_out.transpose(1, 2))[
            0
        ]  # Shape: (batch_size, hidden_dim)

        # LSTM layer
        # Shape: (batch_size, 1, hidden_dim)
        lstm_out, _ = self.lstm(attn_conv_out.unsqueeze(1))
        lstm_out = self.dropout(lstm_out)

        # Attention pooling after LSTM
        # Shape: (batch_size, hidden_dim * 2)
        attn_lstm_out = self.lstm_attention(lstm_out)[0]

        # Concat LSTM-attention output with hand-engineered feature vector
        combined = torch.cat((attn_lstm_out, features), dim=1)
        combined = self.dropout(combined)

        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        return out


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_essay_scoring():
    """Tiny EssayScoringModel for tracing. Architecture is unmodified from the
    real repo; feature_dim matches the 44-entry `features_to_use` list built by
    extract_all_feature() in the real notebook."""
    model = EssayScoringModel(embedding_dim=50, hidden_dim=16, feature_dim=44, dropout_prob=0.0)
    model.eval()
    return model


def example_input_essay_scoring():
    batch, seq_len, feature_dim = 2, 12, 44
    pos_tags = torch.randint(0, 50, (batch, seq_len))
    features = torch.randn(batch, feature_dim)
    return (pos_tags, features)


MENAGERIE_ENTRIES = [
    (
        "EssayScoringCNNBiLSTM",
        build_essay_scoring,
        example_input_essay_scoring,
        2024,
        "vendored-pytorch",
    ),
]
