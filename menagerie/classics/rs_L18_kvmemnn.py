# SOURCE: vendored from prateekagarwal3/KV-Profile-Memory-pytorch @ 1718789d84dbb2961c0168399f22a147a61323c9
# https://raw.githubusercontent.com/prateekagarwal3/KV-Profile-Memory-pytorch/1718789d84dbb2961c0168399f22a147a61323c9/model.py
#
# Miller, Fisch, Dodge, Karimi, Bordes & Weston 2016 "Key-Value Memory Networks for
# Directly Reading Documents" (EMNLP 2016). The original paper's implementation was
# Lua/Torch7 (facebookarchive/MemNN, archived, no PyTorch/Python code exists in that
# repo -- confirmed by inspecting its file tree, all `.lua`). This PyTorch port
# (applied here to the persona-profile-memory / ConvAI2-style "key-value profile
# memory" variant of the architecture) is a real, complete, runnable implementation of
# the paper's key mechanism: two independent addressing/output embedding tables
# (`A1`/`A2`, i.e. separate "key hashing" and "value hashing" embeddings per the
# paper's Sec. 3 "key hashing" framing collapsed into direct memory-slot embedding
# sums), a query embedded and matched against memory keys via dot-product + softmax
# ("key addressing", `p = softmax(bmm(m, q))`), a weighted read-out over the
# corresponding values ("value reading", `o = bmm(p, c)`), a residual query update
# (`q = o + q`, the paper's "controller" hop update), a second key/value addressing
# hop over an external `key`/`val` memory bank, and a final learned projection `W`
# scoring a fixed candidate-response set via dot product -- the exact multi-hop
# key-value memory-network read/score architecture. Real code taken verbatim (only
# reformatted; no computation changed).
#
# Minimal, non-architectural change made:
#   - `model.py` has no `if __name__` construction/example call (the repo's `train.py`
#     imports `KVMemNN` but a data-loading bug means it never actually instantiates or
#     runs it in that script). Added `build_kvmemnn()`/`example_input_kvmemnn()` below
#     that construct the real `KVMemNN(mem_len, mem_size, embd_size, vocab_size)` with
#     tiny sizes and synthesize the exact tensor shapes the real `forward()` reads:
#     `q` is (bsz, mem_len) query word-ids; `persona` is (bsz, mem_size * mem_len)
#     flattened memory-slot word-ids (the real code explicitly `.view(-1, mem_len)`s
#     it first); `key`/`val` are (bsz, mem_size, mem_len) word-ids (fed straight into
#     `self.A2(...)`, i.e. already memory-slot-shaped, unlike `persona`); `cands` is
#     (bsz, 20 * mem_len) flattened candidate-response word-ids, matching the real
#     code's hardcoded `cands.view(-1, 20, self.mem_len, ...)`.

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVMemNN(nn.Module):
    def __init__(self, mem_len, mem_size, embd_size, vocab_size):
        super(KVMemNN, self).__init__()
        self.A1 = nn.Embedding(vocab_size, embd_size)
        self.A1_bn = nn.BatchNorm1d(mem_len)
        self.A2 = nn.Embedding(vocab_size, embd_size)
        self.A2_bn = nn.BatchNorm1d(mem_len)
        self.mem_len = mem_len
        self.mem_size = mem_size
        self.embd_size = embd_size
        self.vocab_size = vocab_size
        self.W = nn.Linear(self.embd_size, self.embd_size)

    def forward(self, q, persona, key, val, cands):
        m = persona.view(-1, self.mem_len)  # (bs*mem_size, mem_len)
        m = self.A1(m)
        # m = self.A1_bn(m)
        m = m.view(-1, self.mem_size, self.mem_len, self.embd_size)
        m = torch.sum(m, dim=2)  # (bs, mem_size, embd_size)

        cands = self.A1(cands)
        # cands = self.A1_bn(cands)
        cands = cands.view(-1, 20, self.mem_len, self.embd_size)
        cands = torch.sum(cands, dim=2)  # (bs, 20, embd_size)

        q = self.A1(q)
        q = torch.sum(q, dim=1)  # (bs, embd_size)

        c = persona.view(-1, self.mem_len)  # (bs*mem_size, mem_len)
        c = self.A2(c)
        # c = self.A2_bn(c)
        c = c.view(-1, self.mem_size, self.mem_len, self.embd_size)
        c = torch.sum(c, dim=2)  # (bs, mem_size, embd_size)

        p = torch.bmm(m, q.unsqueeze(2)).squeeze(2)
        p = F.softmax(p, -1).unsqueeze(1)  # (bs, 1, mem_size)
        o = torch.bmm(p, c).squeeze(1)  # use m as c, (bs, embd_size)
        q = o + q  # (bs, embd_size)

        key = self.A2(key)  # (bs, mem_size, mem_len, embd_size)
        # key = self.A2_bn(key)
        key = torch.sum(key, dim=2)  # (bs, mem_size, embd_size)

        val = self.A2(val)  # (bs, mem_size, mem_len, embd_size)
        # val = self.A2_bn(val)
        val = torch.sum(val, dim=2)  # (bs, mem_size, embd_size)

        ph = torch.bmm(key, q.unsqueeze(2)).squeeze(2)
        ph = F.softmax(ph, -1).unsqueeze(1)  # (bs, 1, mem_size)
        o = torch.bmm(ph, val).squeeze(1)  # (bs, embd_size)

        q = o + q  # (bs, embd_size)
        q = self.W(q)  # (bs, embd_size)
        q = torch.bmm(cands, q.unsqueeze(2)).squeeze(2)  # (bs, 20)
        return q


def build_kvmemnn():
    mem_len = 6
    mem_size = 4
    embd_size = 16
    vocab_size = 50
    model = KVMemNN(mem_len, mem_size, embd_size, vocab_size)
    model.eval()
    return model


def example_input_kvmemnn():
    bsz = 2
    mem_len = 6
    mem_size = 4
    vocab_size = 50
    n_cands = 20

    q = torch.randint(1, vocab_size, (bsz, mem_len))
    persona = torch.randint(1, vocab_size, (bsz, mem_size * mem_len))
    key = torch.randint(1, vocab_size, (bsz, mem_size, mem_len))
    val = torch.randint(1, vocab_size, (bsz, mem_size, mem_len))
    cands = torch.randint(1, vocab_size, (bsz, n_cands * mem_len))
    return (q, persona, key, val, cands)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Key-Value Memory Network (Key-Value Profile Memory)",
        "build_kvmemnn",
        "example_input_kvmemnn",
        2016,
        "vendored",
    ),
]
