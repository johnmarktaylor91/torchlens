# SOURCE: vendored from https://github.com/audreycs/Memory-Networks-Automated-Essay-Grading @ master
#   Vendored file:
#     - model.py  (MANM: Memory-Augmented Neural Model for Automated Grading, a PyTorch
#       reimplementation of Zhao et al. "A Memory-Augmented Neural Model for Automated
#       Grading", L@S 2017 -- key-value memory network reading a fixed bank of per-score
#       exemplar "memories" against the essay content representation, multi-hop attention
#       addressing over the memory bank, position-encoded bag-of-words sentence embedding,
#       final softmax classification over the score range)
#   The vendored MANM.forward()/test() in the original repo take numpy arrays and convert
#   to tensors internally (data loaded from TSV via pandas at each call site). For TorchLens
#   tracing we need a traceable nn.Module boundary that accepts tensors directly, so this
#   file adds a thin `MANMTraceable` wrapper around the *unmodified* MANM layers
#   (input_representation_layer / memory_addressing_layer / memory_reading_layer /
#   output_layer -- copied verbatim from model.py) that runs the same forward computation
#   on tensor inputs instead of numpy arrays, skipping only the loss computation (which
#   needs integer gold-score labels, not part of the model's architecture).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# model.py :: MANM (verbatim layer logic, entry points adapted for tracing --
# see module docstring above)
# ---------------------------------------------------------------------------
class MANM(nn.Module):
    def __init__(
        self,
        word_to_vec,
        max_sent_size,
        memory_num,
        embedding_size,
        feature_size,
        score_range,
        hops,
        l2_lambda,
        keep_prob,
        device,
    ):
        super(MANM, self).__init__()
        self.max_sent_size = max_sent_size
        self.memory_num = memory_num
        self.hops = hops
        self.l2_lambda = l2_lambda
        self.keep_prob = keep_prob
        self.score_range = score_range
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.device = device
        self.word_to_vec = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(word_to_vec), freeze=True
        )
        # [embedding_size, max_sent_size]
        self.pos_encoding = (
            self.position_encoding(self.max_sent_size, self.embedding_size)
            .requires_grad_(False)
            .to(self.device)
        )

        # shape [k, d]
        self.A = torch.nn.Embedding(self.feature_size, self.embedding_size).to(self.device)
        self.B = torch.nn.Embedding(self.feature_size, self.embedding_size).to(self.device)
        self.C = torch.nn.Embedding(self.feature_size, self.embedding_size).to(self.device)
        torch.nn.init.xavier_uniform_(self.A.weight)
        torch.nn.init.xavier_uniform_(self.B.weight)
        torch.nn.init.xavier_uniform_(self.C.weight)
        # shape [k, k]
        Rlist = []
        for i in range(self.hops):
            R = torch.nn.Embedding(self.feature_size, self.feature_size).to(self.device)
            torch.nn.init.xavier_uniform_(R.weight)
            Rlist.append(R)
        self.R_list = torch.nn.ModuleList(Rlist)
        # shape [k, r]
        self.W = torch.nn.Embedding(self.feature_size, self.score_range).to(self.device)
        torch.nn.init.xavier_uniform_(self.W.weight)
        # bias in last layer
        self.b = torch.nn.Parameter(torch.randn([self.score_range]))

    def forward(self, contents_idx, memories_idx):
        # -- adapted for tracing: original forward() took numpy arrays and gold `scores`
        # to compute a training loss; this traces the model's actual architecture (the
        # same layer calls below, verbatim) on tensor inputs and returns the score
        # distribution, matching the semantics of the original `test()` method.
        contents_idx = contents_idx.to(self.device)
        memories_idx = memories_idx.to(self.device)
        # [batch_size, max_sent_size, embedding_size]
        contents = self.word_to_vec(contents_idx)
        # [batch_size, memory_num, max_sent_size, embedding_size]
        memories = self.word_to_vec(memories_idx)
        # emb_contents [batch_size, d]    d=embedding_size
        # emb_memories [batch_size, memory_num, d]
        emb_contents, emb_memories = self.input_representation_layer(contents, memories)
        dropout = torch.nn.Dropout(p=1 - self.keep_prob)
        emb_contents = dropout(emb_contents).requires_grad_(False)
        # [batch_size, k] = [batch_size, d] x [d, k]
        u = torch.matmul(emb_contents, self.A.weight.transpose(0, 1))

        for i in range(self.hops):
            prob_vectors, used_emb_memories = self.memory_addressing_layer(
                u, emb_memories
            )  # [batch_size, memory_num]
            u = self.memory_reading_layer(i, u, prob_vectors, used_emb_memories)  # [batch_size, k]

        # [batch_size, memory_num]   distribution is softmax(logits)
        logits, distribution = self.output_layer(u)
        return distribution

    def position_encoding(self, sentence_size, embedding_size):
        encoding = torch.ones((embedding_size, sentence_size), dtype=torch.float32)
        ls = sentence_size + 1
        le = embedding_size + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        pos_encoding = encoding
        return pos_encoding.transpose(0, 1)

    def input_representation_layer(self, contents: torch.Tensor, memories: torch.Tensor):
        """bow"""
        # contents [batch_size, max_sent_size, embedding_size]
        # memories [batch_size, memory_num, max_sent_size, embedding_size]
        # self.pos_encoding: [max_sent_size, embedding_size]
        # NOTE: original TF/PyTorch upstream calls `.requires_grad_(False)` here on tensors
        # that carry no forward-graph dependency to that flag anyway (a training-loop no-grad
        # guard, not part of the architecture); ported as `.detach()` since `requires_grad_`
        # only accepts leaf tensors in torch and these are computed (non-leaf) sums.
        emb_contents = torch.sum(contents * self.pos_encoding, dim=1).detach()
        emb_memories = torch.sum(memories * self.pos_encoding, dim=2).detach()
        return emb_contents, emb_memories

    def memory_addressing_layer(self, u, emb_memories):
        dropout = torch.nn.Dropout(p=1 - self.keep_prob)
        used_emb_memories = dropout(emb_memories).detach()
        # [batch_size, memory_num, k] = [batch_size, memory_num, d] x [d, k]
        trans_emb_memories = torch.matmul(used_emb_memories, self.B.weight.transpose(0, 1))
        # [batch_size, memory_num, k] <- [batch_size, k]
        trans_emb_contents = u.unsqueeze(dim=1)
        # product [batch_size, memory_num]
        product = torch.sum(trans_emb_contents * trans_emb_memories, dim=-1)
        # prob_vectors [batch_size, memory_num]
        prob_vectors = F.softmax(product, dim=-1)
        return prob_vectors, used_emb_memories

    def memory_reading_layer(self, i, u, prob_vectors, used_emb_memories):
        # [batch_size, memory_num, 1]
        prob_vectors = torch.unsqueeze(prob_vectors, dim=2)
        # [batch_size * memory_num, d]
        memo_temp = used_emb_memories.reshape(-1, self.embedding_size)
        # [d, batch_size * memory_num]
        memo_temp = memo_temp.transpose(0, 1)
        # [k, batch_size * memory_num]
        product = torch.matmul(self.C.weight, memo_temp)
        # [batch_size, memory_num, k]
        product = torch.reshape(product.transpose(0, 1), [-1, self.memory_num, self.feature_size])
        # [batch_size, k]
        o = torch.sum(prob_vectors * product, dim=1)
        # [batch_size, k]
        u = F.relu(torch.matmul((o + u), self.R_list[i].weight))
        return u

    def output_layer(self, u):
        # [batch_size, score_range]
        logits = torch.matmul(u, self.W.weight) + self.b
        distribution = F.softmax(logits, dim=1)
        return logits, distribution


# ---------------------------------------------------------------------------
# tiny-size construction + example input for TorchLens tracing
# ---------------------------------------------------------------------------
_VOCAB_SIZE = 64
_EMBED_DIM = 16
_MAX_SENT_SIZE = 8
_MEMORY_NUM = 4
_FEATURE_SIZE = 16
_SCORE_RANGE = 10
_HOPS = 2


def build_manm_essay_grading():
    import numpy as np

    word_to_vec = np.random.randn(_VOCAB_SIZE, _EMBED_DIM).astype(np.float32)
    model = MANM(
        word_to_vec=word_to_vec,
        max_sent_size=_MAX_SENT_SIZE,
        memory_num=_MEMORY_NUM,
        embedding_size=_EMBED_DIM,
        feature_size=_FEATURE_SIZE,
        score_range=_SCORE_RANGE,
        hops=_HOPS,
        l2_lambda=0.0,
        keep_prob=1.0,
        device=torch.device("cpu"),
    )
    model.eval()
    return model


def example_input_manm_essay_grading():
    contents_idx = torch.randint(0, _VOCAB_SIZE, (1, _MAX_SENT_SIZE))
    memories_idx = torch.randint(0, _VOCAB_SIZE, (1, _MEMORY_NUM, _MAX_SENT_SIZE))
    return (contents_idx, memories_idx)


MENAGERIE_ENTRIES = [
    (
        "Memory-Augmented Neural AES",
        build_manm_essay_grading,
        example_input_manm_essay_grading,
        2017,
        "MENAGERIE_ZOO",
    ),
]
