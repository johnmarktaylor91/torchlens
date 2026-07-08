# SOURCE: vendored from ThuYShao/BERT-PLI-IJCAI2020 @ master
#
# File combined below (imports/paths adjusted only, architecture untouched):
#   model/nlp/AttenRNN.py -> Attention, AttentionRNN
#
# BERT-PLI (IJCAI 2020): "BERT with Paragraph-Level Interaction" for legal case
# retrieval. The full pipeline is two-stage: (1) a BERT encoder scores/embeds
# query-paragraph x candidate-paragraph pairs, producing a per-query-paragraph
# sequence of pooled BERT vectors (shape B x M x 768, M = max_para_q); (2) this
# AttentionRNN module -- a bidirectional LSTM/GRU over the paragraph-interaction
# sequence, max-pooled and fed through a learned-query attention layer -- rolls
# those per-paragraph BERT vectors into a single case-level relevance score.
# AttentionRNN is the architecturally distinctive contribution of BERT-PLI (the
# BERT stage itself is stock transformers.BertModel, i.e. rung-1 material used
# upstream in the original pipeline); it is vendored standalone here since it is
# pure torch with no external deps and takes precomputed BERT-paragraph features
# as its input tensor.
#
# The original module reads several fields off a `config` (configparser) object
# and keeps `self.hidden`/`init_hidden` as instance-mutated LSTM/GRU initial state
# (matching the original repo's non-batch-first hidden-state bookkeeping). Both
# are kept verbatim; only the config-object plumbing is replaced with a tiny
# dataclass-like namespace so the module can be built standalone.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        pass

    def forward(self, feature, hidden):
        # hidden: B * M * H, feature: B * H * 1
        ratio = torch.bmm(hidden, feature)
        # ratio: B * M * 1
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).unsqueeze(2)
        # result: B * H
        result = torch.bmm(hidden.permute(0, 2, 1), ratio)
        result = result.view(result.size(0), -1)
        return result


class AttentionRNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(AttentionRNN, self).__init__()

        self.input_dim = 768
        self.hidden_dim = config.getint("model", "hidden_dim")
        self.dropout_rnn = config.getfloat("model", "dropout_rnn")
        self.dropout_fc = config.getfloat("model", "dropout_fc")
        self.bidirectional = config.getboolean("model", "bidirectional")
        if self.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.num_layers = config.getint("model", "num_layers")
        self.output_dim = config.getint("model", "output_dim")
        self.max_para_q = config.getint("model", "max_para_q")

        if config.get("model", "rnn") == "lstm":
            self.rnn = nn.LSTM(
                self.input_dim,
                self.hidden_dim,
                batch_first=True,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout=self.dropout_rnn,
            )
        else:
            self.rnn = nn.GRU(
                self.input_dim,
                self.hidden_dim,
                batch_first=True,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout=self.dropout_rnn,
            )

        self.max_pool = nn.MaxPool1d(kernel_size=self.max_para_q)
        self.fc_a = nn.Linear(self.hidden_dim * self.direction, self.hidden_dim * self.direction)
        self.attention = Attention(config)
        self.fc_f = nn.Linear(self.hidden_dim * self.direction, self.output_dim)
        #         self.soft_max = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.weight = self.init_weight(config, gpu_list)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)

    def init_weight(self, config, gpu_list):
        try:
            label_weight = config.getfloat("model", "label_weight")
        except Exception:
            return None
        weight_lst = torch.ones(self.output_dim)
        weight_lst[-1] = label_weight
        if torch.cuda.is_available() and len(gpu_list) > 0:
            weight_lst = weight_lst.cuda()
        return weight_lst

    def init_hidden(self, config, batch_size, gpu_list):
        if torch.cuda.is_available() and len(gpu_list) > 0:
            if config.get("model", "rnn") == "lstm":
                self.hidden = (
                    torch.autograd.Variable(
                        torch.zeros(
                            (self.direction * self.num_layers, batch_size, self.hidden_dim)
                        ).cuda()
                    ),
                    torch.autograd.Variable(
                        torch.zeros(
                            (self.direction * self.num_layers, batch_size, self.hidden_dim)
                        ).cuda()
                    ),
                )
            else:
                self.hidden = torch.autograd.Variable(
                    torch.zeros(
                        (self.direction * self.num_layers, batch_size, self.hidden_dim)
                    ).cuda()
                )
        else:
            if config.get("model", "rnn") == "lstm":
                self.hidden = (
                    torch.autograd.Variable(
                        torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim))
                    ),
                    torch.autograd.Variable(
                        torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim))
                    ),
                )
            else:
                self.hidden = torch.autograd.Variable(
                    torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim))
                )

    def init_multi_gpu(self, device, config, *args, **params):
        self.rnn = nn.DataParallel(self.rnn, device_ids=device)
        self.max_pool = nn.DataParallel(self.max_pool, device_ids=device)
        self.fc_a = nn.DataParallel(self.fc_a, device_ids=device)
        self.attention = nn.DataParallel(self.attention, device_ids=device)
        self.fc_f = nn.DataParallel(self.fc_f, device_ids=device)

    #         self.soft_max = nn.DataParallel(self.soft_max, device_ids=device)

    def forward(self, x):
        # x: B * M * I  (M = max_para_q paragraph slots, I = 768 BERT feature dim)
        batch_size = x.size()[0]
        self.init_hidden(_TL_CONFIG, batch_size, [])  # 2 * B * H

        rnn_out, self.hidden = self.rnn(x, self.hidden)  # rnn_out: B * M * 2H, hidden: 2 * B * H
        tmp_rnn = rnn_out.permute(0, 2, 1)  # B * 2H * M

        feature = self.max_pool(tmp_rnn)  # B * 2H * 1
        feature = feature.squeeze(2)  # B * 2H
        feature = self.fc_a(feature)  # B * 2H
        feature = feature.unsqueeze(2)  # B * 2H * 1

        atten_out = self.attention(feature, rnn_out)  # B * (2H)
        atten_out = self.dropout(atten_out)
        y = self.fc_f(atten_out)
        y = y.view(y.size()[0], -1)
        return y


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
# The original module pulls hyperparameters out of a configparser-style object
# (see config/nlp/AttenLSTM.config / AttenGRU.config in the source repo); a tiny
# stand-in namespace replicates that interface with getint/getfloat/getboolean/get.
class _TinyConfig:
    def __init__(self, values):
        self._values = values

    def getint(self, section, key):
        return int(self._values[key])

    def getfloat(self, section, key):
        return float(self._values[key])

    def getboolean(self, section, key):
        return bool(self._values[key])

    def get(self, section, key):
        return self._values[key]


_MAX_PARA_Q = 4

_TL_CONFIG = _TinyConfig(
    {
        "hidden_dim": 8,
        "dropout_rnn": 0.0,
        "dropout_fc": 0.0,
        "bidirectional": True,
        "num_layers": 1,
        "output_dim": 2,
        "max_para_q": _MAX_PARA_Q,
        "rnn": "lstm",
    }
)


def build_bert_pli_attentionrnn():
    model = AttentionRNN(_TL_CONFIG, gpu_list=[])
    model.eval()
    return model


def example_input_bert_pli_attentionrnn():
    return torch.randn(2, _MAX_PARA_Q, 768)


MENAGERIE_ENTRIES = [
    (
        "BERT-PLI (AttentionRNN paragraph-interaction aggregator)",
        build_bert_pli_attentionrnn,
        example_input_bert_pli_attentionrnn,
        2020,
        "VENDOR",
    ),
]
