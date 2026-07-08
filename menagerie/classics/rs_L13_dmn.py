# SOURCE: vendored from jhyuklee/dmn-pytorch @ 36e19dd7d92698d0e157b8b769d4b3e6ad5b16fb
# https://raw.githubusercontent.com/jhyuklee/dmn-pytorch/36e19dd7d92698d0e157b8b769d4b3e6ad5b16fb/model.py
#
# Kumar et al. 2016 "Ask Me Anything: Dynamic Memory Networks for Natural Language
# Processing" -- original DMN architecture (input/question GRU encoders, GRU-cell
# episodic-memory attention gate, GRU-cell answer decoder). The upstream file hardcodes
# `.cuda()` everywhere (no CPU path existed at all) and relies on an external `config`
# object + a `word2vec`-style `idx2vec` numpy array passed at construction time. Both are
# minimally adapted here so the module is constructible/traceable on CPU with random
# init; every layer, op, and control-flow decision is otherwise identical to the source.
#
# Minimal, non-architectural changes made:
#   - `.cuda()` calls replaced with `.to(device)` using the parameter's own device
#     (buffers now follow whatever device the module is moved to, instead of being
#     hardcoded to CUDA).
#   - `F.tanh` / `F.sigmoid` (removed in modern torch) replaced with `torch.tanh` /
#     `torch.sigmoid`, which are numerically identical drop-ins for the same ops.
#   - Optimizer/checkpoint/training-loop code (irrelevant to the traced architecture)
#     dropped; only the `nn.Module` graph is kept.
#   - `config`/`idx2vec` replaced by a tiny `SimpleNamespace` config + random init vector
#     built in `build_dmn()`, so the model constructs without any external dataset file.

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DMN(nn.Module):
    def __init__(self, config, idx2vec, set_num):
        super(DMN, self).__init__()
        self.config = config
        self.set_num = set_num

        # embedding layers
        self.word_embed = nn.Embedding(config.word_vocab_size, config.word_embed_dim, padding_idx=0)

        # dimensions according to settings
        self.s_rnn_idim = config.word_embed_dim
        self.q_rnn_idim = config.word_embed_dim
        self.e_cell_idim = config.s_rnn_hdim
        self.m_cell_idim = config.e_cell_hdim
        self.a_cell_idim = config.q_rnn_hdim + config.word_vocab_size
        self.z_dim = config.s_rnn_hdim * 4

        # rnn layers
        self.s_rnn = nn.GRU(self.s_rnn_idim, config.s_rnn_hdim, batch_first=True)
        self.q_rnn = nn.GRU(self.q_rnn_idim, config.q_rnn_hdim, batch_first=True)
        self.e_cell = nn.GRUCell(self.e_cell_idim, config.e_cell_hdim)
        self.m_cell = nn.GRUCell(self.m_cell_idim, config.m_cell_hdim)
        self.a_cell = nn.GRUCell(self.a_cell_idim, config.a_cell_hdim)

        # linear layers
        self.out = nn.Linear(config.m_cell_hdim, config.word_vocab_size, bias=False)
        self.g1 = nn.Linear(self.z_dim, config.g1_dim)
        self.g2 = nn.Linear(config.g1_dim, 1)

        # initialization
        self.init_word_embed(idx2vec)

    def init_word_embed(self, idx2vec):
        self.word_embed.weight.data.copy_(torch.from_numpy(np.array(idx2vec)))
        self.word_embed.weight.requires_grad = False

    def init_rnn_h(self, batch_size):
        device = self.word_embed.weight.device
        return torch.zeros(
            self.config.s_rnn_ln * 1, batch_size, self.config.s_rnn_hdim, device=device
        )

    def init_cell_h(self, batch_size):
        device = self.word_embed.weight.device
        return torch.zeros(batch_size, self.config.s_rnn_hdim, device=device)

    def input_module(self, stories, s_lens):
        device = self.word_embed.weight.device
        word_embed = F.dropout(self.word_embed(stories), self.config.word_dr)
        init_s_rnn_h = self.init_rnn_h(stories.size(0))
        gru_out, _ = self.s_rnn(word_embed, init_s_rnn_h)
        gru_out = gru_out.contiguous().view(-1, self.config.s_rnn_hdim).cpu()
        s_lens_offset = (
            torch.arange(0, stories.size(0)).type(torch.LongTensor)
            * self.config.max_slen[self.set_num]
        ).unsqueeze(1)
        s_lens = (torch.clamp(s_lens.cpu() + s_lens_offset - 1, min=0)).view(-1)
        selected = (
            gru_out[s_lens, :]
            .view(-1, self.config.max_sentnum[self.set_num], self.config.s_rnn_hdim)
            .to(device)
        )
        return selected

    def question_module(self, questions, q_lens):
        device = self.word_embed.weight.device
        word_embed = F.dropout(self.word_embed(questions), self.config.word_dr)
        init_q_rnn_h = self.init_rnn_h(questions.size(0))
        gru_out, _ = self.q_rnn(word_embed, init_q_rnn_h)
        gru_out = gru_out.contiguous().view(-1, self.config.q_rnn_hdim).cpu()
        q_lens = (
            torch.arange(0, questions.size(0)).type(torch.LongTensor)
            * self.config.max_qlen[self.set_num]
            + q_lens.cpu()
            - 1
        )
        selected = gru_out[q_lens, :].view(-1, self.config.q_rnn_hdim).to(device)

        return selected

    def episodic_memory_module(self, s_rep, q_rep, e_lens, memory):
        device = self.word_embed.weight.device
        # expand s_rep to have sentinel
        sentinel = torch.zeros(s_rep.size(0), 1, self.config.s_rnn_hdim, device=device)
        s_rep = torch.cat((s_rep, sentinel), 1)
        q_rep = q_rep.unsqueeze(1).expand_as(s_rep)
        memory = memory.unsqueeze(1).expand_as(s_rep)
        Z = torch.cat(
            [s_rep * q_rep, s_rep * memory, torch.abs(s_rep - q_rep), torch.abs(s_rep - memory)], 2
        )
        G = self.g2(torch.tanh(self.g1(Z.view(-1, self.z_dim))))
        G_s = torch.sigmoid(G).view(-1, self.config.max_sentnum[self.set_num] + 1).unsqueeze(2)
        G_s = torch.transpose(G_s, 0, 1).contiguous()
        s_rep = torch.transpose(s_rep, 0, 1).contiguous()

        e_rnn_h = self.init_cell_h(s_rep.size(1))
        hiddens = []
        for step, (gg, ss) in enumerate(zip(G_s, s_rep)):
            e_rnn_h = gg * self.e_cell(ss, e_rnn_h) + (1 - gg) * e_rnn_h
            hiddens.append(e_rnn_h)
        hiddens = (
            torch.transpose(torch.stack(hiddens), 0, 1)
            .contiguous()
            .view(-1, self.config.e_cell_hdim)
            .cpu()
        )
        e_lens = (
            torch.arange(0, s_rep.size(1)).type(torch.LongTensor)
            * (self.config.max_sentnum[self.set_num] + 1)
            + e_lens.cpu()
            - 1
        )
        selected = hiddens[e_lens, :].view(-1, self.config.e_cell_hdim).to(device)
        return selected, G.view(-1, self.config.max_sentnum[self.set_num] + 1)

    def answer_module(self, q_rep, memory):
        y = F.softmax(self.out(memory), dim=-1)
        a_rnn_h = memory
        ys = []
        for step in range(self.config.max_alen):
            a_rnn_h = self.a_cell(torch.cat((y, q_rep), 1), a_rnn_h)
            z = self.out(a_rnn_h)
            y = F.softmax(z, dim=-1)
            ys.append(z)
        ys = torch.transpose(torch.stack(ys), 0, 1).contiguous()
        return ys

    def forward(self, stories, questions, s_lens, q_lens, e_lens):
        s_rep = self.input_module(stories, s_lens)
        q_rep = self.question_module(questions, q_lens)

        memory = q_rep  # initial memory
        gates = []
        for episode in range(self.config.max_episode):
            e_rep, gate = self.episodic_memory_module(s_rep, q_rep, e_lens, memory)
            gates.append(gate)
            memory = self.m_cell(e_rep, memory)
        gates = torch.transpose(torch.stack(gates), 0, 1).contiguous()
        outputs = self.answer_module(q_rep, memory)

        return outputs, gates


def build_dmn():
    config = SimpleNamespace(
        word_vocab_size=40,
        word_embed_dim=16,
        s_rnn_hdim=20,
        q_rnn_hdim=20,
        e_cell_hdim=20,
        m_cell_hdim=20,
        a_cell_hdim=20,
        g1_dim=24,
        word_dr=0.0,
        s_rnn_ln=1,
        max_slen={0: 6},
        max_sentnum={0: 4},
        max_qlen={0: 5},
        max_episode=2,
        max_alen=2,
    )
    idx2vec = np.random.randn(config.word_vocab_size, config.word_embed_dim).astype(np.float32)
    return DMN(config, idx2vec, set_num=0)


def example_input_dmn():
    # `stories` is a FLAT token sequence (batch, max_slen); `s_lens` holds the
    # (1-indexed, ascending) end-of-sentence positions within that flat sequence,
    # padded with 0 up to `max_sentnum` -- matching dmn-pytorch's `dataset.py`
    # (`s_lengths` = positions of the "." token, `pad_sent_word` pads the whole
    # flat story to `max_slen`), NOT a per-sentence `(batch, sentnum, slen)` tensor.
    batch = 2
    max_sentnum = 4
    max_slen = 20
    max_qlen = 5
    stories = torch.randint(1, 40, (batch, max_slen))
    questions = torch.randint(1, 40, (batch, max_qlen))
    sentence_len = max_slen // max_sentnum
    s_lens = torch.stack(
        [torch.arange(1, max_sentnum + 1) * sentence_len for _ in range(batch)]
    ).long()
    q_lens = torch.full((batch,), max_qlen, dtype=torch.long)
    e_lens = torch.full((batch,), max_sentnum, dtype=torch.long)
    return (stories, questions, s_lens, q_lens, e_lens)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Dynamic Memory Network (DMN)", "build_dmn", "example_input_dmn", 2016, "vendored"),
]
