# SOURCE: vendored from andy194673/nlg-sclstm-multiwoz @ master
#
# https://github.com/andy194673/nlg-sclstm-multiwoz
# https://raw.githubusercontent.com/andy194673/nlg-sclstm-multiwoz/master/model/layers/decoder_deep.py
# https://raw.githubusercontent.com/andy194673/nlg-sclstm-multiwoz/master/model/lm_deep.py
#
# Wen et al. 2015 "Semantically Conditioned LSTM-based Natural Language
# Generation for Spoken Dialogue Systems" (SCLSTM). This is andy194673's
# PyTorch reimplementation used for the MultiWOZ NLG baseline (the original
# authors' RNNLG toolkit, shawnwun/RNNLG, is Theano/TF). `DecoderDeep`
# (model/layers/decoder_deep.py) and `LM_deep` (model/lm_deep.py) are copied
# verbatim -- the SC-LSTM gating recurrence (`_step`/`rnn_step`), the
# dialogue-act "reading gate" `dt` mechanism (`w2h_r`/`h2h_r`/`dc`), and the
# `n_layer`-deep stacked-decoder concatenation scheme are unchanged.
#
# Minimal, non-architectural adaptations for base-env CPU tracing:
#   - `USE_CUDA = True` -> `False`, and every `.cuda()` call site guarded
#     behind that flag (the upstream module hardcodes `.cuda()` on every
#     Linear layer at construction time and on tensors during forward;
#     removing that hardcoded GPU requirement is a device/environment fix,
#     not an architecture change).
#   - `forward()` is called in the upstream `gen=False` (teacher-forced
#     training) branch, which is the real tensor computation path: the
#     `dataset` argument is only used for `dataset.word2index['SOS_token']`
#     (to build the initial one-hot SOS input) and `dataset.index2word[idx]`
#     (to accumulate a human-readable decoded string in `logits2words`,
#     which does not feed back into the traced tensor computation when
#     `gen=False` -- `vocab_t = input_var[:, t, :]`, i.e. real teacher-forced
#     targets, not the decoded string). A minimal `_TinyVocab` stub supplies
#     exactly those two lookup tables + `batch_size`, mirroring the
#     `dataset`/`Dataset` object the real training script (`run_woz3.py`)
#     passes in -- this is data plumbing, not a change to the SC-LSTM
#     architecture itself.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = False


class DecoderDeep(nn.Module):
    def __init__(
        self, dec_type, input_size, output_size, hidden_size, d_size, n_layer=1, dropout=0.5
    ):
        super(DecoderDeep, self).__init__()
        self.dec_type = dec_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_size = d_size
        self.n_layer = n_layer
        self.dropout = dropout

        assert d_size is not None
        # NOTE: using modulelist instead of python list
        self.w2h, self.h2h = nn.ModuleList(), nn.ModuleList()
        self.w2h_r, self.h2h_r = nn.ModuleList(), nn.ModuleList()
        self.dc = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                w2h_i = nn.Linear(input_size, hidden_size * 4)
                w2h_r_i = nn.Linear(input_size, d_size)
            else:
                w2h_i = nn.Linear(input_size + i * hidden_size, hidden_size * 4)
                w2h_r_i = nn.Linear(input_size + i * hidden_size, d_size)
            if USE_CUDA:
                w2h_i, w2h_r_i = w2h_i.cuda(), w2h_r_i.cuda()
            self.w2h.append(w2h_i)
            self.w2h_r.append(w2h_r_i)

            h2h_i = nn.Linear(hidden_size, hidden_size * 4)
            h2h_r_i = nn.Linear(hidden_size, d_size)
            dc_i = nn.Linear(d_size, hidden_size, bias=False)
            if USE_CUDA:
                h2h_i, h2h_r_i, dc_i = h2h_i.cuda(), h2h_r_i.cuda(), dc_i.cuda()
            self.h2h.append(h2h_i)
            self.h2h_r.append(h2h_r_i)
            self.dc.append(dc_i)

        self.out = nn.Linear(hidden_size * n_layer, output_size)

    def _step(self, input_t, last_hidden, last_cell, last_dt, layer_idx):
        """
        * Do feedforward for one step in one layer in sclstm *
        Args:
                input_t: (batch_size, hidden_size)
                last_hidden: (batch_size, hidden_size)
                last_cell: (batch_size, hidden_size)
        Return:
                cell, hidden, dt at this time step, all: (batch_size, hidden_size)
        """
        # get all gates
        w2h = self.w2h[layer_idx](input_t)  # (batch_size, hidden_size*4)
        w2h = torch.split(w2h, self.hidden_size, dim=1)  # (batch_size, hidden_size) * 4
        h2h = self.h2h[layer_idx](last_hidden[layer_idx])
        h2h = torch.split(h2h, self.hidden_size, dim=1)

        gate_i = F.sigmoid(w2h[0] + h2h[0])  # (batch_size, hidden_size)
        gate_f = F.sigmoid(w2h[1] + h2h[1])
        gate_o = F.sigmoid(w2h[2] + h2h[2])

        # updata dt
        alpha = 1.0 / self.n_layer
        # NOTE: avoid inplace operation which will cause backprop error on graph
        _gate_r = 0
        for i in range(self.n_layer):
            _gate_r += alpha * self.h2h_r[i](last_hidden[i])
        gate_r = F.sigmoid(self.w2h_r[layer_idx](input_t) + _gate_r)

        dt = gate_r * last_dt

        cell_hat = F.tanh(w2h[3] + h2h[3])
        cell = gate_f * last_cell + gate_i * cell_hat + F.tanh(self.dc[layer_idx](dt))
        hidden = gate_o * F.tanh(cell)

        return hidden, cell, dt

    def rnn_step(self, vocab_t, last_hidden, last_cell, last_dt, gen=False):
        """
        run a step over all layers in sclstm
        """
        cur_hidden, cur_cell, cur_dt = [], [], []
        output_hidden = []
        for i in range(self.n_layer):
            # prepare input_t
            if i == 0:
                input_t = vocab_t
                assert input_t.size(1) == self.input_size
            else:
                pre_hidden = torch.cat(output_hidden, dim=1)
                input_t = torch.cat((vocab_t, pre_hidden), dim=1)
                assert input_t.size(1) == self.input_size + i * self.hidden_size

            _hidden, _cell, _dt = self._step(input_t, last_hidden, last_cell[i], last_dt[i], i)
            cur_hidden.append(_hidden)
            cur_cell.append(_cell)
            cur_dt.append(_dt)
            if gen:
                output_hidden.append(_hidden.clone())
            else:
                output_hidden.append(F.dropout(_hidden.clone(), p=self.dropout, training=True))

        last_hidden, last_cell, last_dt = cur_hidden, cur_cell, cur_dt
        if not gen:
            for i in range(self.n_layer):
                last_hidden[i] = F.dropout(last_hidden[i], p=self.dropout, training=True)
        output = self.out(torch.cat(last_hidden, dim=1))
        return output, last_hidden, last_cell, last_dt

    def forward(
        self, input_var, dataset, init_hidden=None, init_feat=None, gen=False, sample_size=1
    ):
        """
        Args:
                input_var: (batch_size, max_len, emb_size)
                hidden: (batch_size, hidden_size) if exist
                feat: (batch_size, feat_size) if exist
        Return:
                output_prob: (batch_size, max_len, output_size)
        """
        batch_size = input_var.size(0)
        max_len = 55 if gen else input_var.size(1)

        self.output_prob = Variable(torch.zeros(batch_size, max_len, self.output_size))
        if USE_CUDA:
            self.output_prob = self.output_prob.cuda()

        # container for last cell, hidden for each layer
        last_hidden, last_cell, last_dt = [], [], []
        for i in range(self.n_layer):
            last_hidden.append(init_hidden.clone())
            last_cell.append(
                init_hidden.clone()
            )  # create a new variable with same content, but new history
            last_dt.append(init_feat.clone())

        decoded_words = ["" for k in range(batch_size)]
        vocab_t = self.get_onehot("SOS_token", dataset, batch_size=batch_size)
        for t in range(max_len):
            output, last_hidden, last_cell, last_dt = self.rnn_step(
                vocab_t, last_hidden, last_cell, last_dt, gen=gen
            )

            self.output_prob[:, t, :] = output
            previous_out = self.logits2words(output, decoded_words, dataset, sample_size)
            vocab_t = previous_out if gen else input_var[:, t, :]  # (batch_size, output_size)

        if gen:
            decoded_words = self.truncate(decoded_words)
        return self.output_prob, decoded_words

    def truncate(self, decoded_words):
        res = []
        for s in decoded_words:
            s = s.split()
            idx = s.index("EOS_token") if "EOS_token" in s else len(s)
            res.append(" ".join(s[:idx]))
        return res

    def get_onehot(self, word, dataset, batch_size=1):
        res = [
            [1 if index == dataset.word2index[word] else 0 for index in range(self.input_size)]
            for b in range(batch_size)
        ]
        res = Variable(torch.FloatTensor(res))
        if USE_CUDA:
            res = res.cuda()
        return res  # (batch_size, input_size)

    def logits2words(self, output, decoded_words, dataset, sample_size):
        """
        * Decode words from logits output at a time step AND put decoded words in final results *
        * take argmax if sample size == 1
        """
        import numpy as np

        batch_size = output.size(0)
        if sample_size == 1:  # take argmax directly w/o sampling
            topv, topi = F.softmax(output, dim=1).data.topk(1)  # both (batch_size, 1)
        else:  # sample over word distribution
            topv, topi = [], []
            word_dis = F.softmax(output, dim=1)  # (batch_size, output_size)

            n_candidate = 3
            word_dis_sort, idx_of_idx = torch.sort(word_dis, dim=1, descending=True)
            word_dis_sort = word_dis_sort[:, :n_candidate]
            idx_of_idx = idx_of_idx[:, :n_candidate]
            sample_idx = torch.multinomial(word_dis_sort, 1)  # (batch_size,)
            for b in range(batch_size):
                i = int(sample_idx[b])
                idx = int(idx_of_idx[b][i])
                prob = float(word_dis[b][idx])
                topi.append(idx)
                topv.append(prob)

            topv = torch.FloatTensor(topv).view(batch_size, 1)
            topi = torch.LongTensor(topi).view(batch_size, 1)

        decoded_words_t = np.zeros((batch_size, self.output_size))
        for b in range(batch_size):
            idx = topi[b][0].item()
            word = dataset.index2word[idx]
            decoded_words[b] += word + " "
            decoded_words_t[b][idx] = 1
        decoded_words_t = Variable(torch.from_numpy(decoded_words_t.astype(np.float32)))

        if USE_CUDA:
            decoded_words_t = decoded_words_t.cuda()

        return decoded_words_t


class LM_deep(nn.Module):
    def __init__(
        self,
        dec_type,
        input_size,
        output_size,
        hidden_size,
        d_size,
        n_layer=1,
        dropout=0.5,
        lr=0.001,
    ):
        super(LM_deep, self).__init__()
        self.dec_type = dec_type
        self.hidden_size = hidden_size
        self.dec = DecoderDeep(
            dec_type,
            input_size,
            output_size,
            hidden_size,
            d_size=d_size,
            n_layer=n_layer,
            dropout=dropout,
        )
        self.set_solver(lr)

    def forward(self, input_var, dataset, feats_var, gen=False, beam_search=False, beam_size=1):
        batch_size = dataset.batch_size
        if self.dec_type == "sclstm":
            init_hidden = Variable(torch.zeros(batch_size, self.hidden_size))
            if USE_CUDA:
                init_hidden = init_hidden.cuda()
            """
            train/valid (gen=False, beam_search=False, beam_size=1)
            test w/o beam_search (gen=True, beam_search=False, beam_size=beam_size)
            test w/i beam_search (gen=True, beam_search=True, beam_size=beam_size)
            """
            if beam_search:
                assert gen
                decoded_words = self.dec.beam_search(
                    input_var,
                    dataset,
                    init_hidden=init_hidden,
                    init_feat=feats_var,
                    gen=gen,
                    beam_size=beam_size,
                )
                return decoded_words

            # w/o beam_search
            sample_size = beam_size
            decoded_words = [[] for _ in range(batch_size)]
            for sample_idx in range(sample_size):  # over generation
                self.output_prob, gens = self.dec(
                    input_var,
                    dataset,
                    init_hidden=init_hidden,
                    init_feat=feats_var,
                    gen=gen,
                    sample_size=sample_size,
                )
                for batch_idx in range(batch_size):
                    decoded_words[batch_idx].append(gens[batch_idx])

            return decoded_words

        else:  # TODO: vanilla lstm
            pass

    def set_solver(self, lr):
        if self.dec_type == "sclstm":
            self.solver = torch.optim.Adam(self.dec.parameters(), lr=lr)
        else:
            self.solver = torch.optim.Adam(
                [{"params": self.dec.parameters()}, {"params": self.feat2hidden.parameters()}],
                lr=lr,
            )


# --- minimal, non-architectural test harness (mirrors the `dataset`/`Dataset`
# object that run_woz3.py passes into LM_deep.forward at training time) ---
class _TinyVocab:
    def __init__(self, vocab_size, batch_size):
        self.batch_size = batch_size
        words = ["SOS_token", "EOS_token"] + [f"w{i}" for i in range(vocab_size - 2)]
        self.word2index = {w: i for i, w in enumerate(words)}
        self.index2word = {i: w for i, w in enumerate(words)}


class SCLSTMTraceWrapper(nn.Module):
    """Thin tensor-in/tensor-out wrapper around the real LM_deep so it can be
    traced: LM_deep.forward's `dataset` argument is a fixed, non-tensor
    vocab-lookup object (see header note), not a model input, so it is closed
    over here rather than passed through forward().
    """

    def __init__(self, lm_deep, dataset):
        super().__init__()
        self.lm_deep = lm_deep
        self.dataset = dataset

    def forward(self, input_var, feats_var):
        self.lm_deep(input_var, self.dataset, feats_var, gen=False)
        return self.lm_deep.output_prob


def build_sclstm():
    dec_type = "sclstm"
    vocab_size = 40
    hidden_size = 16
    d_size = 12  # len of 1-hot dialogue-act feature vector (do_size + da_size + sv_size)
    n_layer = 1
    batch_size = 2
    lm_deep = LM_deep(
        dec_type,
        vocab_size,
        vocab_size,
        hidden_size,
        d_size,
        n_layer=n_layer,
        dropout=0.5,
        lr=0.001,
    )
    dataset = _TinyVocab(vocab_size, batch_size)
    return SCLSTMTraceWrapper(lm_deep, dataset)


def example_input_sclstm():
    batch_size = 2
    max_len = 5
    vocab_size = 40
    d_size = 12

    # teacher-forced target sequence, one-hot over vocab at each timestep
    target_idx = torch.randint(0, vocab_size, (batch_size, max_len))
    input_var = F.one_hot(target_idx, num_classes=vocab_size).float()
    feats_var = torch.rand(batch_size, d_size)

    return (input_var, feats_var)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "SCLSTM (Semantically Conditioned LSTM for NLG)",
        "build_sclstm",
        "example_input_sclstm",
        2015,
        "vendored",
    ),
]
