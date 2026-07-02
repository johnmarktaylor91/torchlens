# SOURCE: vendored from logpai/deep-loglizer @ main
#
# Files combined below (imports/paths adjusted only, architecture untouched):
#   deeploglizer/models/lstm.py        -> Attention, LSTM
#   deeploglizer/models/base_model.py  -> Embedder (minimal slice needed by LSTM.forward)
#
# LogRobust (FSE 2019, Zhang et al., "Robust Log-Based Anomaly Detection on Unstable
# Log Data"): a bidirectional attention-LSTM log-sequence classifier. It is the flagship
# "attention BiLSTM" model shipped inside the deep-loglizer toolkit (logpai/deep-loglizer),
# which is the reference reimplementation the LogRobust paper's technique is packaged
# under (`LSTM(..., use_attention=True, num_directions=2)`). The `base_model.py` training
# harness (evaluate/fit/save loops, pandas/sklearn eval bookkeeping) is intentionally NOT
# vendored -- only the `Embedder` module it defines, which is a direct architectural
# dependency of `LSTM.forward`. `ForcastBasedModel.__init__` is also reproduced in trimmed
# form (drops the disk/logging/pandas plumbing) purely so the real `LSTM` class can be
# constructed unmodified.

import math

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# deeploglizer/models/base_model.py (Embedder + trimmed ForcastBasedModel base)
# ---------------------------------------------------------------------------
class Embedder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrain_matrix=None,
        freeze=False,
        use_tfidf=False,
    ):
        super(Embedder, self).__init__()
        self.use_tfidf = use_tfidf
        if pretrain_matrix is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                pretrain_matrix, padding_idx=1, freeze=freeze
            )
        else:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)

    def forward(self, x):
        if self.use_tfidf:
            return torch.matmul(x, self.embedding_layer.weight.double())
        else:
            return self.embedding_layer(x.long())


class ForcastBasedModel(nn.Module):
    """Trimmed from deeploglizer.models.base_model.ForcastBasedModel: keeps only the
    embedder construction that LSTM.__init__ relies on via super().__init__(); drops
    the training-harness bookkeeping (model_save_path, device, patience, evaluate/fit/
    save/load) which is orthogonal to the traced architecture."""

    def __init__(
        self,
        meta_data,
        feature_type,
        label_type,
        embedding_dim,
        freeze=False,
        use_tfidf=False,
        **kwargs,
    ):
        super(ForcastBasedModel, self).__init__()
        self.feature_type = feature_type
        self.label_type = label_type
        if feature_type in ["sequentials", "semantics"]:
            self.embedder = Embedder(
                meta_data["vocab_size"],
                embedding_dim=embedding_dim,
                pretrain_matrix=meta_data.get("pretrain_matrix", None),
                freeze=freeze,
                use_tfidf=use_tfidf,
            )


# ---------------------------------------------------------------------------
# deeploglizer/models/lstm.py
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, input_size, max_seq_len):
        super(Attention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(max_seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(max_seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.zeros(self.atten_bias)

    def forward(self, lstm_input):
        input_tensor = lstm_input.transpose(1, 0)  # f x b x d

        input_tensor = torch.bmm(input_tensor, self.atten_w) + self.atten_bias  # f x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()

        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), lstm_input).squeeze()

        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)


class LSTM(ForcastBasedModel):
    def __init__(
        self,
        meta_data,
        hidden_size=100,
        num_directions=2,
        num_layers=1,
        window_size=None,
        use_attention=False,
        embedding_dim=16,
        feature_type="sequentials",
        label_type="next_log",
        topk=5,
        use_tfidf=False,
        freeze=False,
        **kwargs,
    ):
        super().__init__(
            meta_data=meta_data,
            feature_type=feature_type,
            label_type=label_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
        )
        num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.window_size = window_size
        self.use_attention = use_attention
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=(self.num_directions == 2),
        )
        if self.use_attention:
            assert window_size is not None, "window size must be set if use attention"
            self.attn = Attention(hidden_size * num_directions, window_size)
        self.prediction_layer = nn.Linear(self.hidden_size * self.num_directions, num_labels)

    def forward(self, x):
        # Simplified from the original dict-based forward (input_dict with
        # "features"/"window_labels"/"window_anomalies") to a plain tensor-in,
        # logits-out forward for tracing: takes the raw integer token-window tensor
        # and returns classification logits over the vocabulary, exactly the same
        # embedder -> LSTM -> (attention) -> prediction_layer path as upstream.
        x = self.embedder(x)

        if self.feature_type == "semantics":
            if not self.use_tfidf:
                x = x.sum(dim=-2)  # add tf-idf

        outputs, _ = self.rnn(x.float())

        if self.use_attention:
            representation = self.attn(outputs)
        else:
            representation = outputs[:, -1, :]

        logits = self.prediction_layer(representation)
        return logits


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_VOCAB_SIZE = 40
_NUM_LABELS = 20
_WINDOW_SIZE = 10
_EMBEDDING_DIM = 8
_HIDDEN_SIZE = 12
_BATCH = 2


def build_logrobust():
    meta_data = {"vocab_size": _VOCAB_SIZE, "num_labels": _NUM_LABELS}
    return LSTM(
        meta_data=meta_data,
        hidden_size=_HIDDEN_SIZE,
        num_directions=2,
        num_layers=1,
        window_size=_WINDOW_SIZE,
        use_attention=True,
        embedding_dim=_EMBEDDING_DIM,
        feature_type="sequentials",
        label_type="anomaly",
    )


def example_input_logrobust():
    return torch.randint(0, _VOCAB_SIZE, (_BATCH, _WINDOW_SIZE))


MENAGERIE_ENTRIES = [
    (
        "LogRobust",
        build_logrobust,
        example_input_logrobust,
        2019,
        "VENDOR",
    ),
]
