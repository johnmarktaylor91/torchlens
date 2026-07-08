# SOURCE: vendored from SageAgastya/SmashRNN @ master
# https://raw.githubusercontent.com/SageAgastya/SmashRNN/master/SmashRnn.py
#
# Jiang, Zhang, Ma, Karyappa, Yang, Bendersky, Najork 2019 (WWW 2019) "Semantic Text
# Matching for Long-Form Documents" -- Siamese Multi-depth Attention-based Hierarchical RNN
# (SMASH RNN) for long-document-to-long-document semantic matching. The original Google
# Research paper never released official code; `SageAgastya/SmashRNN` is a public unofficial
# PyTorch reimplementation of the multi-level (word/sentence/paragraph) attention-hierarchy
# architecture, still real trainable-nn-module code (not written by us from a paper
# description). Three parallel attention encoders (`WordEncoder`, `SentEncoder`,
# `ParaEncoder`) each pool a padded `(batch, para, sent, word, dim)` pre-embedded-token
# tensor up to a single document vector via softmax-attention at their respective depth
# (word-only flat-attention; word->sentence via GRU + 2-level attention; word->sentence->
# paragraph via 2x GRU + 3-level attention); `MashRNN` concatenates the three depth-specific
# document representations; `Siamese` couples two `MashRNN` towers (shared weights, true
# Siamese) with a small MLP + sigmoid to score document-pair similarity.
#
# `WordEncoder`, `SentEncoder`, `ParaEncoder`, `MashRNN`, and `Siamese` are copied verbatim
# from the real `SmashRnn.py`, MINUS the module-level ELMo/TensorFlow-Hub embedding pipeline
# (`Embedder`/`Pad`/the `tf.Graph()` + `hub.Module(...)` block at file-scope), which is real
# but non-architecture preprocessing plumbing (network download + a live TF session to turn
# raw text into `(bs, sent_cnt, word_cnt, 1024)` ELMo vectors) with a hard runtime dependency
# on `tensorflow`/`tensorflow_hub`, which are not in the base env and are unrelated to the
# `nn.Module` architecture itself -- the vendored classes below already operate purely on
# pre-embedded float tensors, exactly the same input contract the real `Pad.__call__` (built
# from real ELMo embeddings) would produce. `WordEncoder.__init__` in the real file assigns
# `self.hidden_layers = hidden_layers` from an undefined name (a bug in the real code -- the
# ctor never declares a `hidden_layers` param, and the attribute is never read anywhere in
# `WordEncoder`); that dead, broken line is dropped as a minimal constructor fix, not an
# architecture change (every op in `forward` is untouched). Separately, the real
# `MashRNN.forward` does `torch.cat([X1, X2, X3], dim=1)` where `X1` (from `WordEncoder`,
# whose `.view(1, x1, -1)` always fixes a size-1 middle dim) comes out 3D `(1, 1, dim)`
# while `X2`/`X3` (from `SentEncoder`/`ParaEncoder`) come out 2D `(1, dim)` -- a real
# shape-contract bug in the unofficial repo (the concat as originally written would crash;
# the repo's own `__main__` never actually exercises `Siamese.forward`). The minimal,
# non-architectural fix applied below is `X1.squeeze(1)` before the concat, which changes
# no arithmetic (attention weights, sums, GRUs are untouched) -- it only reconciles the
# mismatched tensor rank so the three depth-specific document vectors concatenate as the
# paper (and every op's own inline shape comment, e.g. `# (1,1024)`) describes.

import torch
import torch.nn as nn


class WordEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1):
        # input_dim = input dim at each time step = emb_size
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.upw = nn.Linear(hidden_dim, output_dim)  # 1024x1
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):  # padded elmo embed
        x1, _x2 = (X.shape[0] * X.shape[1] * X.shape[2], X.shape[3])  # (4*10*50, 1024)
        X = X.view(1, x1, -1)  # (1, 4*10*50, 1024)
        out = X
        upkji = self.tanh(self.fc(out))  # (bs,2000,1024)
        alpha = self.softmax(self.upw(upkji))  # (bs,2000,1)
        first_level_attention = torch.sum(alpha * out, dim=1, keepdim=True)  # (bs,1,1024)
        return first_level_attention


class SentEncoder(nn.Module):
    def __init__(
        self, input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1, keep_prob=0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.GRU = nn.GRU(
            input_dim,
            int(hidden_dim / 2),
            hidden_layers,
            batch_first=True,
            dropout=keep_prob,
            bidirectional=True,
        )
        self.upw = nn.Linear(hidden_dim, output_dim)
        self.ups = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        x1, x2, x3 = (X.shape[0] * X.shape[1], X.shape[2], X.shape[3])  # (4*10,50,1024)
        X = X.view(x1, x2, x3)
        out = X
        upkj = self.tanh(self.fc(out))  # u(p)(kj) = (bs,40,50,1024)
        alpha = self.softmax(self.upw(upkj))
        first_level_attention = torch.sum(alpha * out, dim=-2).unsqueeze(0)  # (bs,40,1024)
        out, _ = self.GRU(first_level_attention)
        upk = self.tanh(self.fc1(out))
        alpha = self.softmax(self.ups(upk))
        second_level_attention = torch.sum(alpha * out, dim=-2)
        return second_level_attention


class ParaEncoder(nn.Module):
    def __init__(
        self, input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1, keep_prob=0.2
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.GRU_Sent = nn.GRU(
            input_dim,
            int(hidden_dim / 2),
            hidden_layers,
            batch_first=True,
            dropout=keep_prob,
            bidirectional=True,
        )
        self.GRU_para = nn.GRU(
            input_dim,
            int(hidden_dim / 2),
            hidden_layers,
            batch_first=True,
            dropout=keep_prob,
            bidirectional=True,
        )
        self.FC1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-2),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-2),
        )
        self.FC3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-2),
        )

    def forward(self, X):  # (4,10,50,1024)
        out = X  # (4,10,50,1024)
        alpha_word = self.FC1(out)
        first_level_attention = torch.sum(alpha_word * out, dim=-2)
        first_level_attention, _ = self.GRU_Sent(first_level_attention)
        alpha_sent = self.FC2(first_level_attention)
        second_level_attention = torch.sum(alpha_sent * first_level_attention, dim=-2).unsqueeze(0)
        second_level_attention, _ = self.GRU_para(second_level_attention)
        alpha_para = self.FC3(second_level_attention)
        third_level_output = torch.sum(alpha_para * second_level_attention, dim=-2)
        return third_level_output


class MashRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.X1 = WordEncoder()
        self.X2 = SentEncoder()
        self.X3 = ParaEncoder()

    def forward(self, X):
        X1 = self.X1(X)  # (1,1024)
        X2 = self.X2(X)  # (1,1024)
        X3 = self.X3(X)  # (1,1024)
        X1 = X1.squeeze(1)  # minimal shape-bug fix (see header note): (1,1,1024) -> (1,1024)
        mashRNN = torch.cat([X1, X2, X3], dim=1)  # (1, 3*1024)
        return mashRNN


class Siamese(nn.Module):  # Siamese (Unsupervised, we don't have labels)
    def __init__(self):
        super().__init__()
        self.mashRNN = MashRNN()
        self.linear = nn.Sequential(nn.Linear(2 * 3072, 1024), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        out1 = self.mashRNN(x1)
        out2 = self.mashRNN(x2)
        cat = torch.cat([out1, out2], dim=1)
        out = self.linear(cat)
        out = self.out(out)
        return out


def build_smashrnn():
    torch.manual_seed(0)
    model = Siamese()
    model.eval()
    return model


def example_input_smashrnn():
    torch.manual_seed(1)
    # (batch, sent_or_para_count, word_count, elmo_dim) -- the real code's own shape
    # convention (see the `(4*10*50,1024)` / `(4*10,50,1024)` / `(4,10,50,1024)` comments
    # copied verbatim in WordEncoder/SentEncoder/ParaEncoder.forward above: a 4D
    # `(bs, sent_cnt, word_cnt, dim)` padded-embedding tensor, matching what the real
    # `Pad.__call__` (built from real ELMo embeddings) would produce). `Siamese`/`MashRNN`
    # construct their sub-encoders with the real classes' default `input_dim=1024` (the
    # real ELMo embedding width), so the example input keeps that dim to match; only
    # batch/sent_cnt/word_cnt are shrunk for a tiny trace-only example.
    batch, sent_cnt, word_cnt, dim = 1, 2, 3, 1024
    x1 = torch.randn(batch, sent_cnt, word_cnt, dim)
    x2 = torch.randn(batch, sent_cnt, word_cnt, dim)
    return (x1, x2)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SMASH-RNN", "build_smashrnn", "example_input_smashrnn", 2019, "vendored"),
]
