# SOURCE: vendored from yxonic/DTransformer @ main
# (DTransformer/model.py, copied verbatim)
#
# DTransformer (WWW 2023): a knowledge-tracing model that stacks several
# DTransformerLayer self-/cross-attention blocks (a Transformer-style multi-head
# attention with a learned exponential-decay temporal-distance bias, "gammas", and
# a max-out normalization on the attention scores) over question/response
# embeddings, then reads a set of learned "knowledge" query vectors (know_params)
# out of the resulting representation via a second attention pooling step
# (readout) before an MLP prediction head. Ships as a module (not a recipe row)
# because DTransformer.forward consumes 3 real tensors (q_emb, s_emb, lens); we
# trace via the model's own .predict(q, s, pid) entry point, which internally
# builds q_emb/s_emb from raw question/response index tensors -- so
# example_input_dtransformer() supplies the real (q, s) index tensors DTransformer
# is trained on. Only base-lib deps: torch, numpy (numpy imported by the original
# but unused at inference time here).

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

MIN_SEQ_LEN = 5


class DTransformer(nn.Module):
    def __init__(
        self,
        n_questions,
        n_pid=0,
        d_model=128,
        d_fc=256,
        n_heads=8,
        n_know=16,
        n_layers=1,
        dropout=0.05,
        lambda_cl=0.1,
        proj=False,
        hard_neg=True,
        window=1,
        shortcut=False,
    ):
        super().__init__()
        self.n_questions = n_questions
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        self.n_heads = n_heads
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, n_heads, dropout)
        self.block3 = DTransformerLayer(d_model, n_heads, dropout)
        self.block4 = DTransformerLayer(d_model, n_heads, dropout, kq_same=False)
        self.block5 = DTransformerLayer(d_model, n_heads, dropout)

        self.n_know = n_know
        self.know_params = nn.Parameter(torch.empty(n_know, d_model))
        torch.nn.init.uniform_(self.know_params, -1.0, 1.0)

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, d_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc // 2, 1),
        )

        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

        self.dropout_rate = dropout
        self.lambda_cl = lambda_cl
        self.hard_neg = hard_neg
        self.shortcut = shortcut
        self.n_layers = n_layers
        self.window = window

    def forward(self, q_emb, s_emb, lens):
        if self.shortcut:
            # AKT
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, scores = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            return self.block3(hq, hq, hs, lens, peek_cur=False), scores, None

        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, peek_cur=True)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block1(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block2(hq, hq, hs, lens, peek_cur=True)
        else:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)

        bs, seqlen, d_model = p.size()
        n_know = self.n_know

        query = (
            self.know_params[None, :, None, :]
            .expand(bs, -1, seqlen, -1)
            .contiguous()
            .view(bs * n_know, seqlen, d_model)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)

        z, k_scores = self.block4(
            query, hq, p, torch.repeat_interleave(lens, n_know), peek_cur=False
        )
        z = (
            z.view(bs, n_know, seqlen, d_model)  # unpack dimensions
            .transpose(1, 2)  # (bs, seqlen, n_know, d_model)
            .contiguous()
            .view(bs, seqlen, -1)
        )
        k_scores = (
            k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (bs, n_heads, seqlen, n_know, seqlen)
            .contiguous()
        )
        return z, q_scores, k_scores

    def embedding(self, q, s, pid=None):
        lens = (s >= 0).sum(dim=1)
        # set prediction mask
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)

        q_emb = self.q_embed(q)
        s_emb = self.s_embed(s) + q_emb

        p_diff = 0.0

        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)

            q_diff_emb = self.q_diff_embed(q)
            q_emb += q_diff_emb * p_diff

            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb += s_diff_emb * p_diff

        return q_emb, s_emb, lens, p_diff

    def readout(self, z, query):
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )
        value = z.reshape(bs * seqlen, self.n_know, -1)

        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1),
        ).view(bs * seqlen, 1, self.n_know)
        alpha = torch.softmax(beta, -1)
        return torch.matmul(alpha, value).view(bs, seqlen, -1)

    def predict(self, q, s, pid=None, n=1):
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid)
        z, q_scores, k_scores = self(q_emb, s_emb, lens)

        # predict T+N
        if self.shortcut:
            assert n == 1, "AKT does not support T+N prediction"
            h = z
        else:
            query = q_emb[:, n - 1 :, :]
            h = self.readout(z[:, : query.size(1), :], query)

        y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

        if pid is not None:
            return y, z, q_emb, (p_diff**2).mean() * 1e-3, (q_scores, k_scores)
        else:
            return y, z, q_emb, 0.0, (q_scores, k_scores)

    def get_loss(self, q, s, pid=None):
        logits, _, _, reg_loss, _ = self.predict(q, s, pid)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return (
            F.binary_cross_entropy_with_logits(masked_logits, masked_labels, reduction="mean")
            + reg_loss
        )

    def get_cl_loss(self, q, s, pid=None):
        bs = s.size(0)

        # skip CL for batches that are too short
        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:
            return self.get_loss(q, s, pid)

        # augmentation
        q_ = q.clone()
        s_ = s.clone()

        if pid is not None:
            pid_ = pid.clone()
        else:
            pid_ = None

        # manipulate order
        for b in range(bs):
            idx = random.sample(range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate)))
            for i in idx:
                q_[b, i], q_[b, i + 1] = q_[b, i + 1], q_[b, i]
                s_[b, i], s_[b, i + 1] = s_[b, i + 1], s_[b, i]
                if pid_ is not None:
                    pid_[b, i], pid_[b, i + 1] = pid_[b, i + 1], pid_[b, i]

        # hard negative
        s_flip = s.clone() if self.hard_neg else s_
        for b in range(bs):
            # manipulate score
            idx = random.sample(range(lens[b]), max(1, int(lens[b] * self.dropout_rate)))
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]
        if not self.hard_neg:
            s_ = s_flip

        # model
        logits, z_1, q_emb, reg_loss, _ = self.predict(q, s, pid)
        masked_logits = logits[s >= 0]

        _, z_2, *_ = self.predict(q_, s_, pid_)

        if self.hard_neg:
            _, z_3, *_ = self.predict(q, s_flip, pid)

        # CL loss
        input = self.sim(z_1[:, :minlen, :], z_2[:, :minlen, :])
        if self.hard_neg:
            hard_neg = self.sim(z_1[:, :minlen, :], z_3[:, :minlen, :])
            input = torch.cat([input, hard_neg], dim=1)
        target = torch.arange(s.size(0))[:, None].to(self.know_params.device).expand(-1, minlen)
        cl_loss = F.cross_entropy(input, target)

        # prediction loss
        masked_labels = s[s >= 0].float()
        pred_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="mean"
        )

        for i in range(1, self.window):
            label = s[:, i:]
            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            pred_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )
        pred_loss /= self.window

        # TODO: weights
        return pred_loss + cl_loss * self.lambda_cl + reg_loss, pred_loss, cl_loss

    def sim(self, z1, z2):
        bs, seqlen, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seqlen, self.n_know, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seqlen, self.n_know, -1)
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05

    def tracing(self, q, s, pid=None):
        # add fake q, s, pid to generate the last tracing result
        pad = torch.tensor([0]).to(self.know_params.device)
        q = torch.cat([q, pad], dim=0).unsqueeze(0)
        s = torch.cat([s, pad], dim=0).unsqueeze(0)
        if pid is not None:
            pid = torch.cat([pid, pad], dim=0).unsqueeze(0)

        with torch.no_grad():
            # q_emb: (bs, seq_len, d_model)
            # z: (bs, seq_len, n_know * d_model)
            # know_params: (n_know, d_model)->(n_know, 1, d_model)
            q_emb, s_emb, lens, _ = self.embedding(q, s, pid)
            z, _, _ = self(q_emb, s_emb, lens)
            query = self.know_params.unsqueeze(1).expand(-1, z.size(1), -1).contiguous()
            z = z.expand(self.n_know, -1, -1).contiguous()
            h = self.readout(z, query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
            y = torch.sigmoid(y)

        return y


class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False):
        # construct mask
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.device())

        # mask manipulation
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()

            for b in range(query.size(0)):
                # sample for each batch
                if lens[b] < MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate)))
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0

        # apply transformer layer
        query_, scores = self.masked_attn_head(query, key, values, mask, maxout=not peek_cur)
        query = query + self.dropout(query_)
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, maxout=False):
        bs = q.size(0)

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        v_, scores = attention(q, k, v, mask, self.gammas, maxout)

        # concatenate heads and put through final linear layer
        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output, scores


def attention(q, k, v, mask, gamma=None, maxout=False):
    # attention score with scaled dot production
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    # include temporal effect
    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect

    # normalize attention score
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)  # set to hard zero to avoid leakage

    # max-out scores (bs, n_heads, seqlen, seqlen)
    if maxout:
        scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
        scores *= scale

    # calculate output
    output = torch.matmul(scores, v)
    return output, scores


# ---- tiny build/example (architecture unmodified from the real repo) ----


class _DTransformerWrapper(nn.Module):
    """Thin wrapper routing through the real DTransformer.predict() entry point
    (the actual inference path in DTransformer/eval.py), so tracing sees the real
    q/s -> embedding -> attention-stack -> readout -> MLP forward path end to end."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, q, s, pid):
        y, z, q_emb, reg_loss, scores = self.model.predict(q, s, pid)
        return y


def build_dtransformer():
    """Tiny DTransformer (n_pid>0 branch, so the p_diff/q_diff/s_diff difficulty
    embeddings are exercised too) for tracing. Architecture is unmodified from the
    real repo."""
    model = DTransformer(
        n_questions=20,
        n_pid=15,
        d_model=16,
        d_fc=32,
        n_heads=4,
        n_know=4,
        n_layers=1,
        dropout=0.0,
    )
    model.eval()
    return _DTransformerWrapper(model)


def example_input_dtransformer():
    bs, seqlen = 2, 8
    q = torch.randint(0, 20, (bs, seqlen))
    s = torch.randint(0, 2, (bs, seqlen))
    pid = torch.randint(0, 15, (bs, seqlen))
    return (q, s, pid)


MENAGERIE_ENTRIES = [
    ("DTransformer", build_dtransformer, example_input_dtransformer, 2023, "vendored-pytorch"),
]
