# SOURCE: vendored from thunlp/TopJudge @ master
# https://raw.githubusercontent.com/thunlp/TopJudge/master/net/model/model/multi_lstm_seq.py
# https://raw.githubusercontent.com/thunlp/TopJudge/master/net/model/encoder/lstm_encoder.py
# https://raw.githubusercontent.com/thunlp/TopJudge/master/net/model/decoder/lstm_decoder.py
# https://raw.githubusercontent.com/thunlp/TopJudge/master/net/model/layer/attention.py
# https://raw.githubusercontent.com/thunlp/TopJudge/master/net/utils.py (generate_graph only)
#
# Zhong, Zhang, Liu, Sun 2018 (EMNLP 2018) "Legal Judgment Prediction via Topological Learning"
# -- casts multi-task legal-judgment prediction (law article / charge / prison term) as a
# forward pass over a small topological DAG of subtasks with explicit dependency edges (e.g.
# charge depends on article; term depends on both), so each downstream task's `LSTMCell`
# state is updated from every upstream task it depends on via per-edge learned hidden/cell
# gates before its own linear classification head fires. `MultiLSTMSeq` = the paper's
# LSTM-encoder + topological-decoder configuration (`net/work.py` model_list["lstm"] +
# `topjudge` graph config): a document `LSTMEncoder` (sentence-then-document 2-level LSTM,
# max-pooled) feeds the shared "fact" representation into `LSTMDecoder`, whose
# `generate_graph`-parsed adjacency drives the topological multi-task `LSTMCell` propagation
# copied verbatim from `net/model/decoder/lstm_decoder.py`.
#
# `LSTMEncoder`, `Attention`, `LSTMDecoder`, `MultiLSTMSeq`, and `generate_graph` are copied
# verbatim from the real files. The only change is replacing the real `configparser.Config`
# object (loaded from `.config` files + `net/loader.get_num_classes`, which reads
# `crit.txt`/`law.txt` label-frequency tables under a MIMIC-style CAIL legal corpus data
# directory that ships with the repo, not architecture) with a tiny in-memory `_TinyConfig`
# stand-in exposing the same `getint`/getfloat`/`get`/`getboolean` accessor contract used by
# every real forward() below, and a `_tiny_num_classes` table replacing
# `net.loader.get_num_classes`. No computation in `LSTMEncoder.forward`, `LSTMDecoder.forward`,
# `Attention.forward`, or `generate_graph` is changed.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- generate_graph, copied verbatim from net/utils.py (pure adjacency parsing, no arch) ----
def generate_graph(config):
    s = config.get("data", "graph")
    arr = s.replace("[", "").replace("]", "").split(",")
    graph = []
    n = 0
    if s == "[]":
        arr = []
        n = 3
    for a in range(0, len(arr)):
        arr[a] = arr[a].replace("(", "").replace(")", "").split(" ")
        arr[a][0] = int(arr[a][0])
        arr[a][1] = int(arr[a][1])
        n = max(n, max(arr[a][0], arr[a][1]))

    n += 1
    for a in range(0, n):
        graph.append([])
        for b in range(0, n):
            graph[a].append(False)

    for a in range(0, len(arr)):
        graph[arr[a][0]][arr[a][1]] = True

    return graph


class _TinyConfig:
    """Stand-in for the real configparser-backed `Config` object (loaded from `.config`
    files under `config/cail/...` in the real repo): a plain dict-of-dicts exposing the
    same getint/getfloat/get/getboolean accessor contract the vendored forward() methods
    call. No architecture behavior lives here -- just config plumbing."""

    def __init__(self, data):
        self._data = data

    def getint(self, section, key):
        return int(self._data[section][key])

    def getfloat(self, section, key):
        return float(self._data[section][key])

    def getboolean(self, section, key):
        return bool(self._data[section][key])

    def get(self, section, key):
        return self._data[section][key]


_TINY_NUM_CLASSES = {"law": 5, "crit": 4, "time": 11}


def _tiny_num_classes(task_name):
    return _TINY_NUM_CLASSES[task_name]


# ---- Attention, copied verbatim from net/model/layer/attention.py ----
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        pass

    def forward(self, feature, hidden):
        feature = feature.view(feature.size(0), -1, 1)
        ratio = torch.bmm(hidden, feature)
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).view(ratio.size(0), -1, 1)
        result = torch.bmm(hidden.transpose(1, 2), ratio)
        result = result.view(result.size(0), -1)

        return result


# ---- LSTMEncoder, copied verbatim from net/model/encoder/lstm_encoder.py ----
class LSTMEncoder(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTMEncoder, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm_sentence = nn.LSTM(
            self.data_size,
            self.hidden_dim,
            batch_first=True,
            num_layers=config.getint("net", "num_layers"),
        )
        self.lstm_document = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            batch_first=True,
            num_layers=config.getint("net", "num_layers"),
        )
        self.feature_len = self.hidden_dim

    def init_hidden(self, config, usegpu):
        if torch.cuda.is_available() and usegpu:
            self.sentence_hidden = (
                torch.autograd.Variable(
                    torch.zeros(
                        config.getint("net", "num_layers"),
                        config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                        self.hidden_dim,
                    ).cuda()
                ),
                torch.autograd.Variable(
                    torch.zeros(
                        config.getint("net", "num_layers"),
                        config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                        self.hidden_dim,
                    ).cuda()
                ),
            )
            self.document_hidden = (
                torch.autograd.Variable(
                    torch.zeros(
                        config.getint("net", "num_layers"),
                        config.getint("data", "batch_size"),
                        self.hidden_dim,
                    ).cuda()
                ),
                torch.autograd.Variable(
                    torch.zeros(
                        config.getint("net", "num_layers"),
                        config.getint("data", "batch_size"),
                        self.hidden_dim,
                    ).cuda()
                ),
            )
        else:
            self.sentence_hidden = (
                torch.autograd.Variable(
                    torch.zeros(
                        1,
                        config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                        self.hidden_dim,
                    )
                ),
                torch.autograd.Variable(
                    torch.zeros(
                        1,
                        config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                        self.hidden_dim,
                    )
                ),
            )
            self.document_hidden = (
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)
                ),
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)
                ),
            )

    def forward(self, x, doc_len, config):
        x = x.view(
            config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
            config.getint("data", "sentence_len"),
            config.getint("data", "vec_size"),
        )

        sentence_out, self.sentence_hidden = self.lstm_sentence(x, self.sentence_hidden)
        temp_out = []
        if config.get("net", "method") == "LAST":
            for a in range(0, len(sentence_out)):
                idx = a // config.getint("data", "sentence_num")
                idy = a % config.getint("data", "sentence_num")
                temp_out.append(sentence_out[a][doc_len[idx][idy + 2] - 1])
            sentence_out = torch.stack(temp_out)
        elif config.get("net", "method") == "MAX":
            sentence_out = sentence_out.contiguous().view(
                config.getint("data", "batch_size"),
                config.getint("data", "sentence_num"),
                config.getint("data", "sentence_len"),
                config.getint("net", "hidden_size"),
            )
            sentence_out = torch.max(sentence_out, dim=2)[0]
            sentence_out = sentence_out.view(
                config.getint("data", "batch_size"),
                config.getint("data", "sentence_num"),
                config.getint("net", "hidden_size"),
            )
        sentence_out = sentence_out.view(
            config.getint("data", "batch_size"),
            config.getint("data", "sentence_num"),
            self.hidden_dim,
        )

        lstm_out, self.document_hidden = self.lstm_document(sentence_out, self.document_hidden)

        self.attention = lstm_out

        if config.get("net", "method") == "LAST":
            outv = []
            for a in range(0, len(doc_len)):
                outv.append(lstm_out[a][doc_len[a][1] - 1])
            lstm_out = torch.cat(outv)
        elif config.get("net", "method") == "MAX":
            lstm_out = torch.max(lstm_out, dim=1)[0]

        return lstm_out


# ---- LSTMDecoder, copied verbatim from net/model/decoder/lstm_decoder.py ----
class LSTMDecoder(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTMDecoder, self).__init__()
        self.feature_len = config.getint("net", "hidden_size")

        features = config.getint("net", "hidden_size")
        self.hidden_dim = features
        self.outfc = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(features, _tiny_num_classes(x)))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]
        for x in task_name:
            self.cell_list.append(
                nn.LSTMCell(
                    config.getint("net", "hidden_size"), config.getint("net", "hidden_size")
                )
            )

        self.hidden_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.attention = Attention(config)
        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, config, usegpu):
        self.hidden_list = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for a in range(0, len(task_name) + 1):
            if torch.cuda.is_available() and usegpu:
                self.hidden_list.append(
                    (
                        torch.autograd.Variable(
                            torch.zeros(config.getint("data", "batch_size"), self.hidden_dim).cuda()
                        ),
                        torch.autograd.Variable(
                            torch.zeros(config.getint("data", "batch_size"), self.hidden_dim).cuda()
                        ),
                    )
                )
            else:
                self.hidden_list.append(
                    (
                        torch.autograd.Variable(
                            torch.zeros(config.getint("data", "batch_size"), self.hidden_dim)
                        ),
                        torch.autograd.Variable(
                            torch.zeros(config.getint("data", "batch_size"), self.hidden_dim)
                        ),
                    )
                )

    def forward(self, x, doc_len, config, attention):
        fc_input = x
        outputs = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        graph = generate_graph(config)

        first = []
        for a in range(0, len(task_name) + 1):
            first.append(True)
        for a in range(1, len(task_name) + 1):
            h, c = self.cell_list[a](fc_input, self.hidden_list[a])
            for b in range(1, len(task_name) + 1):
                if graph[a][b]:
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = h, c
                    else:
                        hp = hp + self.hidden_state_fc_list[a][b](h)
                        cp = cp + self.cell_state_fc_list[a][b](c)
                    self.hidden_list[b] = (hp, cp)
            if config.getboolean("net", "attention"):
                h = self.attention(h, attention)
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(
                        config.getint("data", "batch_size"), -1
                    )
                )
            else:
                outputs.append(self.outfc[a - 1](h).view(config.getint("data", "batch_size"), -1))

        return outputs


# ---- MultiLSTMSeq, copied verbatim from net/model/model/multi_lstm_seq.py ----
class MultiLSTMSeq(nn.Module):
    def __init__(self, config, usegpu):
        super(MultiLSTMSeq, self).__init__()

        self.encoder = LSTMEncoder(config, usegpu)
        self.decoder = LSTMDecoder(config, usegpu)
        self.trans_linear = nn.Linear(self.encoder.feature_len, self.decoder.feature_len)
        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))

    def init_hidden(self, config, usegpu):
        self.encoder.init_hidden(config, usegpu)
        self.decoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        if self.encoder.feature_len != self.decoder.feature_len:
            x = self.trans_linear(x)
        x = self.dropout(x)
        x = self.decoder(x, doc_len, config, self.encoder.attention)

        return x


class TopJudgeTraceAdapter(nn.Module):
    """Thin adapter: the real `MultiLSTMSeq.forward` needs a non-tensor `config` object and
    a `doc_len` bookkeeping structure (unused by the MAX-pooling path exercised here) as
    positional args alongside the input tensor, and the model keeps its recurrent hidden
    state as `self.*_hidden`/`self.hidden_list` instance attributes re-initialized via
    `init_hidden` (not passed through forward). This wrapper closes over the (non-tensor)
    config + calls `init_hidden` once per forward (mirroring how the real `net/work.py`
    training loop re-inits hidden state every batch) so the traced signature is a single
    tensor in, list-of-tensors out. No architecture computation is added; every op is a
    call into the unmodified `MultiLSTMSeq.forward`."""

    def __init__(self, model, config, usegpu, doc_len):
        super().__init__()
        self.model = model
        self.config = config
        self.usegpu = usegpu
        self.doc_len = doc_len

    def forward(self, x):
        self.model.init_hidden(self.config, self.usegpu)
        outputs = self.model.forward(x, self.doc_len, self.config)
        return torch.cat(outputs, dim=1)


def _build_config(batch_size, sentence_num, sentence_len, vec_size, hidden_size):
    return _TinyConfig(
        {
            "data": {
                "vec_size": vec_size,
                "batch_size": batch_size,
                "sentence_num": sentence_num,
                "sentence_len": sentence_len,
                "type_of_label": "law,crit,time",
                "graph": "[(1 2),(1 3),(2 3)]",
            },
            "net": {
                "hidden_size": hidden_size,
                "num_layers": 1,
                "method": "MAX",
                "attention": False,
                "more_fc": False,
            },
            "train": {
                "dropout": 0.1,
            },
        }
    )


def build_topjudge():
    torch.manual_seed(0)
    batch_size, sentence_num, sentence_len, vec_size, hidden_size = 2, 3, 4, 8, 16
    config = _build_config(batch_size, sentence_num, sentence_len, vec_size, hidden_size)
    model = MultiLSTMSeq(config, usegpu=False)
    model.eval()
    doc_len = [[sentence_len] * (sentence_num + 2) for _ in range(batch_size)]
    return TopJudgeTraceAdapter(model, config, usegpu=False, doc_len=doc_len)


def example_input_topjudge():
    torch.manual_seed(1)
    batch_size, sentence_num, sentence_len, vec_size = 2, 3, 4, 8
    return torch.randn(batch_size, sentence_num, sentence_len, vec_size)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("TopJudge", "build_topjudge", "example_input_topjudge", 2018, "vendored"),
]
