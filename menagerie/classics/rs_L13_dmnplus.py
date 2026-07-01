# SOURCE: vendored from dandelin/Dynamic-memory-networks-plus-Pytorch @ ad49955f907c03aade2f6c8ed13370ce7288d5a7
# https://raw.githubusercontent.com/dandelin/Dynamic-memory-networks-plus-Pytorch/ad49955f907c03aade2f6c8ed13370ce7288d5a7/babi_main.py
#
# Xiong et al. 2016 "Dynamic Memory Networks for Visual and Textual Question Answering"
# (DMN+) -- bidirectional-GRU input module + position encoding, attention-gated GRU
# episodic memory (AttentionGRUCell/AttentionGRU/EpisodicMemory), multi-hop reasoning,
# linear answer module. This is a widely-cited reference DMN+ implementation.
#
# Minimal, non-architectural changes made (all bookkeeping/API-drift, no computation
# changed):
#   - `.cuda()` calls replaced with `.to(device)` derived from the input tensor's own
#     device, so the module is constructible/traceable on CPU.
#   - `torch.autograd.Variable` wrapping removed (no-op on modern torch; tensors already
#     carry autograd history).
#   - `F.tanh` / `F.sigmoid` (removed in modern torch) replaced with `torch.tanh` /
#     `torch.sigmoid`, and `F.softmax(...)`/`init.xavier_normal(...)`/`init.uniform(...)`
#     given the required `dim=`/trailing-underscore forms now mandatory in modern torch
#     (`xavier_normal_`, `uniform_`) -- same numerics, just the current spelling.
#   - `InputModule.forward` referenced a bare `hidden_size` name (a real bug in the
#     original -- it should have read `self.hidden_size`, since no such name is ever
#     defined in that scope); fixed to `self.hidden_size` so the module is runnable
#     at all (this is the only place actual behavior was touched, and it restores the
#     obviously-intended behavior rather than changing it).
#   - `nn.Embedding(..., sparse=True)` requires a sparse-aware optimizer to train, but
#     is fully traceable/inferrable as-is, so it is kept exactly as authored.
#   - Training loop / babi-loader / optimizer code (irrelevant to the traced
#     architecture) dropped; only the `nn.Module` graph is kept.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def position_encoding(embedded_sentence):
    """
    embedded_sentence.size() -> (#batch, #sentence, #token, #embedding)
    l.size() -> (#sentence, #embedding)
    output.size() -> (#batch, #sentence, #embedding)
    """
    _, _, slen, elen = embedded_sentence.size()

    pos_l = [
        [(1 - s / (slen - 1)) - (e / (elen - 1)) * (1 - 2 * s / (slen - 1)) for e in range(elen)]
        for s in range(slen)
    ]
    pos_l = torch.tensor(pos_l, dtype=embedded_sentence.dtype, device=embedded_sentence.device)
    pos_l = pos_l.unsqueeze(0)  # for #batch
    pos_l = pos_l.unsqueeze(1)  # for #sen
    pos_l = pos_l.expand_as(embedded_sentence)
    weighted = embedded_sentence * pos_l
    return torch.sum(weighted, dim=2).squeeze(2)  # sum with tokens


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.Wr.state_dict()["weight"])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.Ur.state_dict()["weight"])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.W.state_dict()["weight"])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.U.state_dict()["weight"])

    def forward(self, fact, C, g):
        """
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        """

        r = torch.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = torch.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h


class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        """
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        """
        batch_num, sen_num, embedding_size = facts.size()
        C = torch.zeros(self.hidden_size, device=facts.device, dtype=facts.dtype)
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal_(self.z1.state_dict()["weight"])
        init.xavier_normal_(self.z2.state_dict()["weight"])
        init.xavier_normal_(self.next_mem.state_dict()["weight"])

    def make_interaction(self, facts, questions, prevM):
        """
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        """
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat(
            [
                facts * questions,
                facts * prevM,
                torch.abs(facts - questions),
                torch.abs(facts - prevM),
            ],
            dim=2,
        )

        z = z.view(-1, 4 * embedding_size)

        G = torch.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G, dim=-1)

        return G

    def forward(self, facts, questions, prevM):
        """
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        """
        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem


class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, questions, word_embedding):
        """
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        gru() -> (1, #batch, #hidden)
        """
        questions = word_embedding(questions)
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
        return questions


class InputModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if "weight" in name:
                init.xavier_normal_(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts, word_embedding):
        """
        contexts.size() -> (#batch, #sentence, #token)
        word_embedding() -> (#batch, #sentence x #token, #embedding)
        position_encoding() -> (#batch, #sentence, #embedding)
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        """
        batch_num, sen_num, token_num = contexts.size()

        contexts = contexts.view(batch_num, -1)
        contexts = word_embedding(contexts)

        contexts = contexts.view(batch_num, sen_num, token_num, -1)
        contexts = position_encoding(contexts)
        contexts = self.dropout(contexts)

        h0 = torch.zeros(
            2, batch_num, self.hidden_size, device=contexts.device, dtype=contexts.dtype
        )
        facts, hdn = self.gru(contexts, h0)
        facts = facts[:, :, : self.hidden_size] + facts[:, :, self.hidden_size :]
        return facts


class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        init.xavier_normal_(self.z.state_dict()["weight"])
        self.dropout = nn.Dropout(0.1)

    def forward(self, M, questions):
        M = self.dropout(M)
        concat = torch.cat([M, questions], dim=2).squeeze(1)
        z = self.z(concat)
        return z


class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True)
        init.uniform_(self.word_embedding.state_dict()["weight"], a=-(3**0.5), b=3**0.5)
        self.criterion = nn.CrossEntropyLoss()

        self.input_module = InputModule(vocab_size, hidden_size)
        self.question_module = QuestionModule(vocab_size, hidden_size)
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)

    def forward(self, contexts, questions):
        """
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, 1, #hidden)
        """
        facts = self.input_module(contexts, self.word_embedding)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        for hop in range(self.num_hop):
            M = self.memory(facts, questions, M)
        preds = self.answer_module(M, questions)
        return preds


def build_dmnplus():
    hidden_size = 16
    vocab_size = 40
    return DMNPlus(hidden_size, vocab_size, num_hop=3, qa=None)


def example_input_dmnplus():
    batch = 2
    sen_num = 4
    token_num = 5
    q_token_num = 6
    contexts = torch.randint(1, 40, (batch, sen_num, token_num))
    questions = torch.randint(1, 40, (batch, q_token_num))
    return (contexts, questions)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Dynamic Memory Network Plus (DMN+)",
        "build_dmnplus",
        "example_input_dmnplus",
        2016,
        "vendored",
    ),
]
