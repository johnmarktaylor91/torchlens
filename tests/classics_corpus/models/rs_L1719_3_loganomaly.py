# SOURCE: vendored from https://github.com/logpai/deep-loglizer @ main
# (deeploglizer/models/{lstm.py, base_model.py}). LogAnomaly: template2vec +
# quantitative + sequential-pattern LSTM for log anomaly detection (Meng et
# al., IJCAI 2019, "LogAnomaly: Unsupervised Detection of Sequential and
# Quantitative Anomalies in Unstructured Logs"). deep-loglizer is logpai's
# official benchmark reimplementation toolkit (this LSTM+Attention model is
# the toolkit's LogAnomaly-family entry, `deeploglizer/models/lstm.py`).
#
# Vendored real repo code: `Attention` and `LSTM` (deeploglizer/models/lstm.py)
# are copied verbatim. `LSTM` subclasses `ForcastBasedModel`
# (deeploglizer/models/base_model.py); only its architecture-relevant
# `__init__` (which builds the shared `Embedder`) is vendored here -- the
# `evaluate`/`fit`/`save_model`/`load_model` methods are the training/eval
# loop (require sklearn+pandas and disk I/O), not architecture, and are
# dropped. No layer, dimension, or dataflow in `Embedder`/`ForcastBasedModel.
# __init__`/`Attention`/`LSTM.forward` was changed.

import math

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# deeploglizer/models/base_model.py (architecture-relevant subset)
# ---------------------------------------------------------------------------
class Embedder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, pretrain_matrix=None, freeze=False, use_tfidf=False
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
    def __init__(
        self,
        meta_data,
        model_save_path,
        feature_type,
        label_type,
        eval_type,
        topk,
        use_tfidf,
        embedding_dim,
        freeze=False,
        gpu=-1,
        anomaly_ratio=None,
        patience=3,
        **kwargs,
    ):
        super(ForcastBasedModel, self).__init__()
        self.device = torch.device("cpu")
        self.topk = topk
        self.meta_data = meta_data
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.anomaly_ratio = anomaly_ratio
        self.patience = patience
        self.time_tracker = {}

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
        model_save_path="./lstm_models",
        feature_type="sequentials",
        label_type="next_log",
        eval_type="session",
        topk=5,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
        **kwargs,
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
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
        self.criterion = nn.CrossEntropyLoss()
        self.prediction_layer = nn.Linear(self.hidden_size * self.num_directions, num_labels)

    def forward(self, input_dict):
        if self.label_type == "anomaly":
            y = input_dict["window_anomalies"].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict["window_labels"].long().view(-1)
        self.batch_size = y.size()[0]
        x = input_dict["features"]
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
        y_pred = logits.softmax(dim=-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict


_VOCAB_SIZE = 20
_NUM_LABELS = 20
_WINDOW_SIZE = 8


def build_loganomaly():
    meta_data = {"vocab_size": _VOCAB_SIZE, "num_labels": _NUM_LABELS}
    return LSTM(
        meta_data,
        hidden_size=16,
        num_directions=2,
        num_layers=1,
        window_size=_WINDOW_SIZE,
        use_attention=True,
        embedding_dim=8,
        feature_type="sequentials",
        label_type="next_log",
    )


def example_input_loganomaly():
    # single positional `input_dict` arg, matching the real repo's dataloader
    # batch format for the "sequentials" + "next_log" LogAnomaly configuration.
    batch_size = 4
    input_dict = {
        "features": torch.randint(0, _VOCAB_SIZE, (batch_size, _WINDOW_SIZE)),
        "window_labels": torch.randint(0, _NUM_LABELS, (batch_size,)),
        "window_anomalies": torch.randint(0, 2, (batch_size,)),
    }
    return (input_dict,)


MENAGERIE_ENTRIES = [
    (
        "LogAnomaly",
        build_loganomaly,
        example_input_loganomaly,
        2019,
        MENAGERIE_ZOO,
    ),
]
