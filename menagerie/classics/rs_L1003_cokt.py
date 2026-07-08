# SOURCE: vendored from john1226966735/CoKT @ main
# (CoKT/Code/DKT_QE.py + CoKT/Code/CoKT.py, copied verbatim)
#
# CoKT (Collaborative Embedding for Knowledge Tracing, WSDM 2022): the CoKT class
# subclasses DKT_QE's "question-level DKT" base model (InputModule fuses a question
# embedding with the correctness label via a learned 0/1 selection matrix, an
# LSTM/GRU/RNN KnowledgeStateModule tracks the running knowledge state, and a
# PredictModule scores the next question) and swaps InputModule's plain
# nn.Embedding question-embedding layer for LoadFusePretrainEmb, which fuses one or
# more pre-trained (node2vec) question-embedding tables via a learned linear fusion
# layer. In the real repo the node2vec tables are loaded from disk; here
# example_input_cokt() supplies random "pre_emb_list" tensors of the same shape so
# the real forward path (LoadFusePretrainEmb.forward -> InputModule -> LSTM ->
# PredictModule) traces unmodified. Only base-lib deps: torch.

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

MENAGERIE_ZOO = "vendored-pytorch"

DEVICE = torch.device("cpu")


# ---- CoKT/Code/DKT_QE.py (verbatim, DEVICE globals folded into module-level DEVICE) ----


class DKT_QuesEmb(nn.Module, ABC):
    def __init__(self, args, data):
        super(DKT_QuesEmb, self).__init__()
        self.input_module = InputModule(data["num_ques"], args.emb_dim)
        self.ks_module = KnowledgeStateModule(
            args.rnn_mode, self.input_module.input_dim, args.hidden_dim, args.rnn_num_layer
        )
        self.predict_module = PredictModule(args.hidden_dim, args.emb_dim, args.exercise_dim)

    def forward(self, seq_lens, pad_curr, pad_answer, pad_next):
        # get interact/question embedding
        interact_emb, next_ques_emb = self.input_module(pad_curr, pad_answer, pad_next)
        # update knowledge state
        ks_emb = self.ks_module(interact_emb)
        # predict
        pad_predict = self.predict_module(ks_emb, next_ques_emb)
        pack_predict = pack_padded_sequence(pad_predict, seq_lens, enforce_sorted=True)
        return pack_predict


class InputModule(nn.Module, ABC):
    def __init__(self, num_ques, ques_emb_dim):
        super(InputModule, self).__init__()
        self.input_dim = 2 * ques_emb_dim
        # for question embedding
        self.ques_emb_layer = nn.Embedding(num_ques, ques_emb_dim)

        # for fusing question embedding and correctness
        self.transform_matrix = torch.zeros(2, self.input_dim, device=DEVICE)
        self.transform_matrix[0][ques_emb_dim:] = 1.0
        self.transform_matrix[1][:ques_emb_dim] = 1.0

    def forward(self, pad_curr, pad_answer, pad_next):  # [seq, bs]
        # get embedding of question
        curr_ques_emb = self.ques_emb_layer(
            pad_curr
        )  # get current questions' embeddings, [seq, bs, dim]
        next_ques_emb = self.ques_emb_layer(
            pad_next
        )  # get predict questions' embeddings, [seq, bs, dim]

        # concatenate zero vector in front of or behind curr_ques_emb according to correctness
        answer_emb = F.embedding(pad_answer, self.transform_matrix)
        interact_emb = torch.cat((curr_ques_emb, curr_ques_emb), -1) * answer_emb
        return interact_emb, next_ques_emb


class KnowledgeStateModule(nn.Module, ABC):
    def __init__(self, rnn_mode, input_dim, hidden_dim, num_layer):
        super(KnowledgeStateModule, self).__init__()
        assert rnn_mode in ["lstm", "rnn", "gru"]
        if rnn_mode == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=False)
        elif rnn_mode == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layer, batch_first=False)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layer, batch_first=False)

    def forward(self, pad_interact_emb):
        pad_ks_emb, _ = self.rnn(pad_interact_emb)
        return pad_ks_emb


class PredictModule(nn.Module, ABC):
    def __init__(self, ks_dim, question_dim, exercise_dim):
        super(PredictModule, self).__init__()
        self.h2y = nn.Linear(ks_dim + question_dim, exercise_dim)
        self.y2o = nn.Linear(exercise_dim, 1)

    def forward(self, ks_emb, question_emb):
        y = F.relu(self.h2y(torch.cat((ks_emb, question_emb), -1)))
        prediction = torch.sigmoid(self.y2o(y)).squeeze(-1)
        return prediction


# ---- CoKT/Code/CoKT.py (verbatim) ----


class CoKT(DKT_QuesEmb, ABC):
    def __init__(self, args, data):
        super(CoKT, self).__init__(args, data)
        self.input_module.ques_emb_layer = LoadFusePretrainEmb(args, data)


class LoadFusePretrainEmb(nn.Module, ABC):
    def __init__(self, args, data):
        super(LoadFusePretrainEmb, self).__init__()
        self.num_graph = len(args.used_graphs)
        self.pre_emb_list = data["pre_emb_list"]

        if self.num_graph > 1:
            concat_dim = sum([emb.size(-1) for emb in self.pre_emb_list])
            self.fuse_emb_layer = nn.Linear(concat_dim, args.emb_dim)

    def forward(self, pad_ques):
        # get embedding of present question
        batch_emb_list = []
        for emb_mat in self.pre_emb_list:
            batch_emb_list.append(F.embedding(pad_ques, emb_mat))
        if self.num_graph > 1:
            fused_ques_emb = F.relu(self.fuse_emb_layer(torch.cat(batch_emb_list, dim=-1)))
        else:
            fused_ques_emb = batch_emb_list[0]
        return fused_ques_emb


# ---- tiny build/example (architecture unmodified from the real repo) ----


class _Args:
    def __init__(self):
        self.emb_dim = 8
        self.hidden_dim = 8
        self.rnn_mode = "lstm"
        self.rnn_num_layer = 1
        self.exercise_dim = 8
        self.used_graphs = ["cw", "pk"]  # 2 graphs -> exercises the fuse_emb_layer branch
        self.device = "cpu"


def build_cokt():
    """Tiny CoKT (2 fused node2vec-style pretrained embedding tables) for tracing.
    Architecture is unmodified from the real repo."""
    num_ques = 20
    args = _Args()
    pre_emb_list = [torch.randn(num_ques, args.emb_dim) for _ in args.used_graphs]
    data = {"num_ques": num_ques, "pre_emb_list": pre_emb_list}
    model = CoKT(args, data)
    model.eval()
    return model


def example_input_cokt():
    seq, bs = 6, 2
    seq_lens = torch.tensor(sorted([seq] * bs, reverse=True), dtype=torch.long)
    pad_curr = torch.randint(0, 20, (seq, bs))
    pad_answer = torch.randint(0, 2, (seq, bs))
    pad_next = torch.randint(0, 20, (seq, bs))
    return (seq_lens, pad_curr, pad_answer, pad_next)


MENAGERIE_ENTRIES = [
    ("CoKT", build_cokt, example_input_cokt, 2022, "vendored-pytorch"),
]
