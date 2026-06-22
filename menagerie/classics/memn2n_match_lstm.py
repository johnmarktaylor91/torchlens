"""End-to-End Memory Networks (MemN2N) and Match-LSTM for bAbI / Reading Comprehension.

--- MemN2N ---
Sukhbaatar et al. (2015), "End-To-End Memory Networks". NeurIPS 2015. arXiv:1503.08895.
Source: https://github.com/facebook/MemNN

Distinctive primitive: multi-hop soft memory reads.
  1. Story sentences -> memory slot embeddings (m_i via embedding A).
  2. Query -> query embedding (u via embedding B).
  3. Attention: p_i = softmax(u . m_i) over N memory slots.
  4. Output read: o = sum_i p_i * c_i  (output embeddings C).
  5. u_new = u + o  (residual update).
  Repeat for K hops; final answer = argmax(W (u_K + o_K)).

--- Match-LSTM ---
Wang & Jiang (2016), "Machine Comprehension Using Match-LSTM and Answer Pointer".
IACL 2017. arXiv:1608.07905.

Distinctive primitive: attention-weighted matching LSTM.
  1. Passage and question encoded by separate BiLSTMs.
  2. For each passage token t, compute attention over question tokens: alpha_t = softmax(W * h_t^P).
  3. Context vector: z_t = sum_i alpha_t,i * h_i^Q.
  4. Match-LSTM: LSTM over [h_t^P; z_t] -- the matched sequence.
  5. Pointer network / output: linear over match-LSTM states -> span predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# 1.  End-to-End Memory Network (MemN2N) -- bAbI QA
# ==============================================================


class MemN2N(nn.Module):
    """Multi-hop end-to-end memory network.

    Compact config: vocab=20, embed_dim=16, n_hops=3, mem_size=5 sentences.
    Input: story sentence indices (B, n_sents, sent_len) + query indices (B, query_len).
    Wrapped to take a single (B, n_sents*sent_len + query_len) int tensor.
    """

    def __init__(
        self,
        vocab_size: int = 20,
        embed_dim: int = 16,
        n_hops: int = 3,
        n_sents: int = 5,
        sent_len: int = 4,
        query_len: int = 3,
        n_answers: int = 20,
    ) -> None:
        super().__init__()
        self.n_hops = n_hops
        self.n_sents = n_sents
        self.sent_len = sent_len
        self.query_len = query_len
        self.embed_dim = embed_dim

        # Separate embeddings per hop (adjacent weight tying simplified: all share)
        self.A_embs = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(n_hops)])
        self.C_embs = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(n_hops)])
        self.B_emb = nn.Embedding(vocab_size, embed_dim)  # query embedding
        self.W_out = nn.Linear(embed_dim, n_answers, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_sents*sent_len + query_len) int64

        First n_sents*sent_len tokens = story, last query_len = query.
        """
        B = x.shape[0]
        n_story = self.n_sents * self.sent_len
        story = x[:, :n_story].view(B, self.n_sents, self.sent_len)
        query = x[:, n_story:]  # (B, query_len)

        # Query embedding: mean over query positions
        u = self.B_emb(query).mean(dim=1)  # (B, embed_dim)

        for hop in range(self.n_hops):
            # Memory embeddings: (B, n_sents, embed_dim) = mean over words
            m = self.A_embs[hop](story).mean(dim=2)  # (B, n_sents, D)
            # Attention: p = softmax(u . m^T)
            attn = torch.bmm(m, u.unsqueeze(2)).squeeze(2)  # (B, n_sents)
            p = F.softmax(attn, dim=1)  # (B, n_sents)
            # Output read
            c = self.C_embs[hop](story).mean(dim=2)  # (B, n_sents, D)
            o = torch.bmm(p.unsqueeze(1), c).squeeze(1)  # (B, D)
            u = u + o  # residual update

        return self.W_out(u)  # (B, n_answers) logits


class _MemN2NWrapper(nn.Module):
    def __init__(self, base: MemN2N) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


def build_memn2n_babi() -> nn.Module:
    return _MemN2NWrapper(
        MemN2N(vocab_size=20, embed_dim=16, n_hops=3, n_sents=5, sent_len=4, query_len=3)
    ).eval()


def example_input_memn2n() -> torch.Tensor:
    """(1, 23) int64 -- 5 sents * 4 words + 3 query words."""
    return torch.randint(0, 20, (1, 23))


# ==============================================================
# 2.  Match-LSTM for Reading Comprehension
# ==============================================================


class MatchLSTMRC(nn.Module):
    """Match-LSTM reading comprehension block.

    Passage tokens -> passage BiLSTM encoder.
    Question tokens -> question BiLSTM encoder.
    For each passage token: attend over question -> concat -> MatchLSTM.
    Output: span start/end logit (simplified: linear over match-LSTM states).

    Wrapped to take a single (B, P + Q) int tensor, first P = passage, last Q = question.
    """

    def __init__(
        self,
        vocab_size: int = 30,
        embed_dim: int = 16,
        hidden: int = 24,
        passage_len: int = 8,
        question_len: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.passage_len = passage_len
        self.question_len = question_len
        self.enc_passage = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.enc_question = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        # Attention: W_q projects question hidden, W_p projects passage hidden
        self.W_q = nn.Linear(2 * hidden, hidden, bias=False)
        self.W_p = nn.Linear(2 * hidden, hidden, bias=False)
        self.v_attn = nn.Linear(hidden, 1, bias=False)
        # Match-LSTM input = [passage_h; context_vector]
        self.match_lstm = nn.LSTM(4 * hidden, hidden, batch_first=True, bidirectional=False)
        # Span head: start logit over passage positions
        self.span_head = nn.Linear(hidden, 2)  # start + end for each position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, P+Q) int64 -> (B, P, 2) span logits"""
        B = x.shape[0]
        P, Q = self.passage_len, self.question_len
        p_idx = x[:, :P]  # (B, P)
        q_idx = x[:, P:]  # (B, Q)

        p_emb = self.embed(p_idx)  # (B, P, E)
        q_emb = self.embed(q_idx)  # (B, Q, E)

        p_enc, _ = self.enc_passage(p_emb)  # (B, P, 2H)
        q_enc, _ = self.enc_question(q_emb)  # (B, Q, 2H)

        # Match at each passage position
        match_inputs = []
        for t in range(P):
            h_p_t = p_enc[:, t, :]  # (B, 2H)
            # Attention score over question: (B, Q)
            # score_i = v^T tanh(W_q * q_i + W_p * h_p_t)
            W_q_q = self.W_q(q_enc)  # (B, Q, H)
            W_p_h = self.W_p(h_p_t).unsqueeze(1).expand_as(W_q_q)  # (B, Q, H)
            score = self.v_attn(torch.tanh(W_q_q + W_p_h)).squeeze(-1)  # (B, Q)
            alpha = F.softmax(score, dim=1)  # (B, Q)
            context = torch.bmm(alpha.unsqueeze(1), q_enc).squeeze(1)  # (B, 2H)
            match_inputs.append(torch.cat([h_p_t, context], dim=-1))  # (B, 4H)

        match_seq = torch.stack(match_inputs, dim=1)  # (B, P, 4H)
        match_out, _ = self.match_lstm(match_seq)  # (B, P, H)
        return self.span_head(match_out)  # (B, P, 2)


class _MatchLSTMWrapper(nn.Module):
    def __init__(self, base: MatchLSTMRC) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


def build_match_lstm_rc() -> nn.Module:
    return _MatchLSTMWrapper(
        MatchLSTMRC(vocab_size=30, embed_dim=16, hidden=24, passage_len=8, question_len=4)
    ).eval()


def example_input_match_lstm() -> torch.Tensor:
    """(1, 12) int64 -- 8 passage + 4 question tokens."""
    return torch.randint(0, 30, (1, 12))


MENAGERIE_ENTRIES = [
    (
        "MemN2N (End-to-End Memory Network: multi-hop soft memory reads for bAbI QA)",
        "build_memn2n_babi",
        "example_input_memn2n",
        "2015",
        "DC",
    ),
    (
        "Match-LSTM RC (Match-LSTM for reading comprehension: attended question context + matching LSTM)",
        "build_match_lstm_rc",
        "example_input_match_lstm",
        "2016",
        "DC",
    ),
]
