# FAITHFUL PORT of insomnia1996/deep-lawyer @ master (original framework: Keras 1.x / TF1.x)
# https://raw.githubusercontent.com/insomnia1996/deep-lawyer/master/code/final_selfmodel.py
# (`build_final`, lines 67-238) + https://raw.githubusercontent.com/insomnia1996/deep-lawyer/master/config.py
#
# MPBFN -- "Legal Judgment Prediction via Multi-Perspective Bi-Feedback Network" (Yang, Jia,
# Xiao, Zhou, Long, Wang, Jiang; IJCAI 2019, https://arxiv.org/abs/1905.03969). The repo
# (insomnia1996/deep-lawyer) ships the real Keras 1.x model definition in
# `code/final_selfmodel.py:build_final`, but it is written against a `keras.engine.topology`
# / `keras.layers.Merge`-era API (Keras <2.0 / TF1.x graph-mode Functional API) that no
# longer exists in the installed Keras 2.13 (`keras.engine.topology` was removed, `Merge` was
# removed in Keras 2.0) -- the real code cannot run in the base env, so per the ladder this is
# transcribed FAITHFULLY into self-contained torch (every layer/mechanism as in the actual
# `build_final` graph, not a paper-level gist).
#
# Architecture (mirrors `build_final` verbatim, functional-graph steps re-expressed as torch
# submodules/forward-graph nodes):
#   1. `embedding_layer` (word embedding) -> `cnninput` reshape -> 4 parallel Conv2d branches
#      (kernel heights 2,3,3,3 all spanning the full embedding width, i.e. 1D-conv-over-time
#      via a full-width 2D kernel) each followed by full-length max-pool-over-time -> `feature`
#      = concat of the 4 pooled branches (a TextCNN-style multi-kernel encoder), then dropout.
#   2. First-pass predictions: `accu_preds1 = softmax(Dense(feature))`,
#      `law_preds1 = softmax(Dense(feature))`.
#   3. Bi-feedback round 1: `accu_preds1` dotted against the accusation-label embedding table
#      -> `law_preds2` (sigmoid Dense, no bias); symmetrically `law_preds1` dotted against the
#      law-label embedding table -> `accu_preds2` (sigmoid Dense, no bias). This is the
#      "bi-feedback" mechanism: each task's first-pass prediction seeds a *refinement* signal
#      for the *other* task via its label-embedding table.
#   4. `accu_preds3 = normalize(accu_preds1 * accu_preds2)`,
#      `law_preds3 = normalize(law_preds1 * law_preds2)` -- fused final label predictions
#      (elementwise product then L1-renormalize, matching `normalize()`/`Lambda` in the
#      original).
#   5. `accu_merge2`/`law_merge2`: the *fused* predictions (`accu_preds3`/`law_preds3`)
#      re-dotted against the same label-embedding tables and projected (`Dense(..., 'elu')`)
#      to `hid_size` -- the "perspective" embeddings used to condition the pair-attention path.
#   6. A separate "pair" branch: `total_pair` (word, word) / (word, number) token pairs, each
#      side embedded (word-embedding for word tokens, a small digit-embedding table for
#      per-digit numeric tokens, masked/gated by is-word / is-number indicator tensors and
#      summed), concatenated into a length-2 sequence per pair, and run through a shared LSTM
#      (`TimeDistributed(LSTM(hid_size))`) to get one `hid_size` vector per pair.
#   7. The pair vectors are attention-pooled twice -- once against `law_merge2_d` (softmax
#      dot-product attention -> `feature_pair_law`) and once against `accu_merge2_d`
#      (-> `feature_pair_accu`, computed in the original but only `feature_pair_law` is
#      actually consumed downstream, matching the real code's unused `feature_accu` branch
#      that stays commented out).
#   8. `feature_law = elu(Dense(concat(feature, feature_pair_law)))`; `accu_fix = feature *
#      accu_merge2_d`; `law_fix = feature_law * law_merge2_d` -- perspective-fixed features
#      for the term (sentence-length) head.
#   9. `term_preds1 = softmax(Dense(accu_fix))`, `term_preds2 = softmax(Dense(law_fix))`,
#      `term_preds3 = normalize(term_preds1 * term_preds2)` -- the same bi-feedback-fusion
#      pattern applied to the term/penalty task, conditioned on both the accusation and law
#      perspectives.
#   Outputs: `(accu_preds3, law_preds3, term_preds3)`, exactly as the real
#   `Model(inputs=[...], outputs=[accu_preds3, law_preds3, term_preds3])`.
#
# `Config` hyperparameters are lifted from the real `config.py` (`hid_size`, `word_embed_size`,
# `total_num`, `total_pair`, `num_accu_liu`, `num_law_liu`, `num_term_liu`) and only
# down-scaled for menagerie's tiny trace budget (`word_vocab_size`, `total_num`, `total_pair`,
# `hid_size`, `num_*_liu` are all shrunk; the *ratios/roles* -- `hid_size/4` per CNN branch,
# `word_embed_size/8` per numeric-digit slot -- are preserved exactly).

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, eps=1e-7):
    """Mirrors the real `normalize()` Lambda: L1-renormalize along the last axis."""
    return x / (x.sum(dim=-1, keepdim=True) + eps)


class MPBFN(nn.Module):
    def __init__(
        self,
        word_vocab_size=200,
        word_embed_size=32,
        hid_size=32,
        total_num=24,
        total_pair=6,
        num_accu=10,
        num_law=9,
        num_term=6,
    ):
        super().__init__()
        assert hid_size % 4 == 0
        assert word_embed_size % 8 == 0

        self.word_embed_size = word_embed_size
        self.hid_size = hid_size
        self.total_num = total_num
        self.total_pair = total_pair
        self.num_accu = num_accu
        self.num_law = num_law
        self.num_term = num_term

        branch = hid_size // 4

        # --- word / label / digit embedding tables (Embedding(...) layers) ---
        self.embedding = nn.Embedding(word_vocab_size, word_embed_size)
        self.law_embedding = nn.Embedding(num_law, hid_size)
        self.accu_embedding = nn.Embedding(num_accu, hid_size)
        self.num_embedding = nn.Embedding(10, word_embed_size // 8)

        # --- multi-kernel CNN feature extractor (Conv2D(kernel=(k, word_embed_size)) + full
        #     length max-pool, k in {2, 3, 3, 3}) ---
        self.cnn2 = nn.Conv2d(1, branch, kernel_size=(2, word_embed_size))
        self.cnn3 = nn.Conv2d(1, branch, kernel_size=(3, word_embed_size))
        self.cnn4 = nn.Conv2d(1, branch, kernel_size=(3, word_embed_size))
        self.cnn5 = nn.Conv2d(1, branch, kernel_size=(3, word_embed_size))
        self.feature_dropout = nn.Dropout(0.5)

        # --- first-pass task heads ---
        self.accu_preds1_fc = nn.Linear(hid_size, num_accu)
        self.law_preds1_fc = nn.Linear(hid_size, num_law)

        # --- bi-feedback refinement heads: project the *other* task's label-embedding
        #     dot-product (a hid_size vector) to this task's label space (no bias, matches
        #     `Dense(..., use_bias=False)`) ---
        self.law_preds2_fc = nn.Linear(hid_size, num_law, bias=False)
        self.accu_preds2_fc = nn.Linear(hid_size, num_accu, bias=False)

        # --- perspective projections (Dense(hid_size, 'elu')) ---
        self.accu_merge2_d_fc = nn.Linear(hid_size, hid_size)
        self.law_merge2_d_fc = nn.Linear(hid_size, hid_size)

        # --- pair branch ---
        self.pair_num_fc = nn.Linear(word_embed_size, word_embed_size)
        self.pair_lstm = nn.LSTM(word_embed_size, hid_size, batch_first=True)
        self.pair_d_fc = nn.Linear(hid_size, hid_size, bias=False)

        self.feature_law_fc = nn.Linear(2 * hid_size, hid_size)

        # --- term (penalty) heads ---
        self.term_preds1_fc = nn.Linear(hid_size, num_term)
        self.term_preds2_fc = nn.Linear(hid_size, num_term)

    def _cnn_branch(self, conv, cnninput):
        # cnninput: [batch, 1, total_num, word_embed_size]
        out = conv(cnninput)  # [batch, branch, total_num - k + 1, 1]
        out = F.relu(out)
        pooled = F.max_pool2d(out, kernel_size=(out.shape[2], 1))  # [batch, branch, 1, 1]
        return pooled.view(pooled.shape[0], -1)

    def forward(self, data):
        sentence = data["sentence"]  # [batch, total_num] long
        accu_ids = data["accu_ids"]  # [batch, num_accu] long, label-table row ids
        law_ids = data["law_ids"]  # [batch, num_law] long
        pair_front_word = data["pair_front_word"]  # [batch, total_pair] long
        pair_last_word = data["pair_last_word"]  # [batch, total_pair] long
        pair_front_isword = data["pair_front_isword"]  # [batch, total_pair] float
        pair_last_isword = data["pair_last_isword"]  # [batch, total_pair] float
        pair_front_num = data["pair_front_num"]  # [batch, total_pair, 8] long
        pair_last_num = data["pair_last_num"]  # [batch, total_pair, 8] long
        pair_front_isnum = data["pair_front_isnum"]  # [batch, total_pair] float
        pair_last_isnum = data["pair_last_isnum"]  # [batch, total_pair] float

        batch = sentence.shape[0]

        # --- CNN feature extraction ---
        embedded = self.embedding(sentence)  # [batch, total_num, word_embed_size]
        cnninput = embedded.unsqueeze(1)  # [batch, 1, total_num, word_embed_size]
        pooling2 = self._cnn_branch(self.cnn2, cnninput)
        pooling3 = self._cnn_branch(self.cnn3, cnninput)
        pooling4 = self._cnn_branch(self.cnn4, cnninput)
        pooling5 = self._cnn_branch(self.cnn5, cnninput)
        feature = torch.cat([pooling2, pooling3, pooling4, pooling5], dim=-1)  # [batch, hid_size]
        feature = self.feature_dropout(feature)

        # --- first-pass predictions ---
        accu_preds1 = F.softmax(self.accu_preds1_fc(feature), dim=-1)  # [batch, num_accu]
        law_preds1 = F.softmax(self.law_preds1_fc(feature), dim=-1)  # [batch, num_law]

        # --- label-embedding lookups (dynamic per-sample label tables, as in the real
        #     `Embedding(...)(accu_input)` / `Embedding(...)(law_input)` calls) ---
        accu_embedded = self.accu_embedding(accu_ids)  # [batch, num_accu, hid_size]
        law_embedded = self.law_embedding(law_ids)  # [batch, num_law, hid_size]

        # --- bi-feedback round 1: cross-task refinement via label-embedding dot products ---
        accu_merge = torch.einsum("bn,bnd->bd", accu_preds1, accu_embedded)  # [batch, hid_size]
        law_preds2 = torch.sigmoid(self.law_preds2_fc(accu_merge))  # [batch, num_law]

        law_merge = torch.einsum("bn,bnd->bd", law_preds1, law_embedded)  # [batch, hid_size]
        accu_preds2 = torch.sigmoid(self.accu_preds2_fc(law_merge))  # [batch, num_accu]

        # --- fused predictions ---
        accu_preds3 = normalize(accu_preds1 * accu_preds2)
        law_preds3 = normalize(law_preds1 * law_preds2)

        # --- perspective projections from the fused predictions ---
        accu_merge2 = torch.einsum("bn,bnd->bd", accu_preds3, accu_embedded)
        accu_merge2_d = F.elu(self.accu_merge2_d_fc(accu_merge2))

        law_merge2 = torch.einsum("bn,bnd->bd", law_preds3, law_embedded)
        law_merge2_d = F.elu(self.law_merge2_d_fc(law_merge2))

        # --- pair branch ---
        pair_front_w = self.embedding(pair_front_word) * pair_front_isword.unsqueeze(-1)
        pair_last_w = self.embedding(pair_last_word) * pair_last_isword.unsqueeze(-1)

        pair_front_n = self.num_embedding(pair_front_num).reshape(
            batch, self.total_pair, self.word_embed_size
        )
        pair_front_n = F.elu(self.pair_num_fc(pair_front_n)) * pair_front_isnum.unsqueeze(-1)
        pair_last_n = self.num_embedding(pair_last_num).reshape(
            batch, self.total_pair, self.word_embed_size
        )
        pair_last_n = F.elu(self.pair_num_fc(pair_last_n)) * pair_last_isnum.unsqueeze(-1)

        pair_front = pair_front_n + pair_front_w  # [batch, total_pair, word_embed_size]
        pair_last = pair_last_n + pair_last_w

        # concat then reshape to a length-2 sequence per pair (TimeDistributed(LSTM))
        pair_seq = torch.stack([pair_front, pair_last], dim=2)  # [batch, total_pair, 2, dim]
        pair_seq = pair_seq.reshape(batch * self.total_pair, 2, self.word_embed_size)
        _, (h_n, _) = self.pair_lstm(pair_seq)
        pair = h_n[-1].reshape(
            batch, self.total_pair, self.hid_size
        )  # [batch, total_pair, hid_size]

        pair_d = torch.tanh(self.pair_d_fc(pair))  # [batch, total_pair, hid_size]

        weight_law = torch.einsum("bph,bh->bp", pair_d, law_merge2_d)
        weight_law = F.softmax(weight_law, dim=-1)
        feature_pair_law = torch.einsum("bp,bph->bh", weight_law, pair)

        weight_accu = torch.einsum("bph,bh->bp", pair_d, accu_merge2_d)
        weight_accu = F.softmax(weight_accu, dim=-1)
        # matches the real code: feature_pair_accu is computed but never consumed downstream
        _feature_pair_accu = torch.einsum("bp,bph->bh", weight_accu, pair)

        feature_law = torch.cat([feature, feature_pair_law], dim=-1)
        feature_law = F.elu(self.feature_law_fc(feature_law))

        accu_fix = feature * accu_merge2_d
        law_fix = feature_law * law_merge2_d

        term_preds1 = F.softmax(self.term_preds1_fc(accu_fix), dim=-1)
        term_preds2 = F.softmax(self.term_preds2_fc(law_fix), dim=-1)
        term_preds3 = normalize(term_preds1 * term_preds2)

        return {
            "accu_preds": accu_preds3,
            "law_preds": law_preds3,
            "term_preds": term_preds3,
        }


def build_mpbfn():
    model = MPBFN(
        word_vocab_size=200,
        word_embed_size=32,
        hid_size=32,
        total_num=24,
        total_pair=6,
        num_accu=10,
        num_law=9,
        num_term=6,
    )
    model.eval()
    return model


def example_input_mpbfn():
    torch.manual_seed(0)
    batch = 2
    total_num = 24
    total_pair = 6
    num_accu = 10
    num_law = 9
    vocab = 200

    data = {
        "sentence": torch.randint(0, vocab, (batch, total_num)),
        "accu_ids": torch.arange(num_accu).unsqueeze(0).repeat(batch, 1),
        "law_ids": torch.arange(num_law).unsqueeze(0).repeat(batch, 1),
        "pair_front_word": torch.randint(0, vocab, (batch, total_pair)),
        "pair_last_word": torch.randint(0, vocab, (batch, total_pair)),
        "pair_front_isword": torch.ones(batch, total_pair),
        "pair_last_isword": torch.ones(batch, total_pair),
        "pair_front_num": torch.randint(0, 10, (batch, total_pair, 8)),
        "pair_last_num": torch.randint(0, 10, (batch, total_pair, 8)),
        "pair_front_isnum": torch.zeros(batch, total_pair),
        "pair_last_isnum": torch.zeros(batch, total_pair),
    }
    return (data,)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("MPBFN", "build_mpbfn", "example_input_mpbfn", 2019, "ported"),
]
