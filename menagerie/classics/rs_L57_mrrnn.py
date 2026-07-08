# SOURCE: vendored from Ri0S/mrrnn-pytorch @ master (commit at fetch time)
# https://raw.githubusercontent.com/Ri0S/mrrnn-pytorch/master/models.py
# https://raw.githubusercontent.com/Ri0S/mrrnn-pytorch/master/modules.py
#
# Serban et al. 2017 (AAAI) "Multiresolution Recurrent Neural Networks" (MrRNN) --
# a hierarchical dialogue model with a word-level utterance encoder (`BaseEncoder`,
# a bidirectional GRU), a session-level recurrent state over utterances
# (`SessionEncoder`, a GRU), a "coarse" side-channel (`Coarse`) that predicts and
# encodes a coarse noun/entity sequence per utterance (`CoarseEncoder` +
# `Coarse.dec`), and a maxout GRU decoder (`Decoder`, teacher-forcing branch
# `do_decode_tc`) that generates the fine-grained utterance conditioned on the
# concatenation of session state and coarse hidden state. This is the actual
# `Seq2Seq`("mrrnn" mode) architecture from `models.py`/`modules.py` -- unchanged.
#
# `Seq2Seq`, `Coarse`, `BaseEncoder`, `CoarseEncoder`, `SessionEncoder`, `Decoder`,
# `max_out` are copied verbatim (module bodies unchanged) from `models.py` and
# `modules.py`. No architectural code was rewritten; only these mechanical,
# import-isolation changes were made:
#   - The real `configs.py` builds its `Config` object by unpickling 4 vocabulary
#     files from `./data/*.pkl` and parsing `sys.argv` via argparse at import time.
#     That is replaced here by a plain `SimpleNamespace` (`_build_config()`) carrying
#     the exact same fields the model code reads (`model`, `mode`, `num_layers`,
#     `bidirectional`, `embedding_size`, `encoder_hidden_size`, `context_hidden_size`,
#     `decoder_hidden_size`, `dropout`, `syllable_size`, `word_size`, `device`) --
#     same values (matching upstream argparse defaults), just supplied directly
#     instead of via unpickled vocab files + CLI flags.
#   - `modules.py`/`models.py` read a single module-level `config` object
#     (`from configs import config`); this file instead threads a `config` argument
#     explicitly into every constructor/forward that needs it (`BaseEncoder`,
#     `CoarseEncoder`, `SessionEncoder`, `Decoder`, `Coarse`, `Seq2Seq`), so the
#     module has no import-time global state. The per-call `config.<field>` reads
#     inside method bodies are otherwise identical to upstream.
#   - `nn.CrossEntropyLoss(..., size_average=False)` (a PyTorch kwarg removed in
#     modern torch, replaced by `reduction=`) -> `nn.CrossEntropyLoss(ignore_index=0,
#     reduction='sum')`, the documented modern equivalent of `size_average=False`.
#   - This build exercises `config.mode == 'train'` (teacher-forcing decode path,
#     `do_decode_tc`), which is the code path upstream itself takes for training --
#     not a new branch.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from types import SimpleNamespace


def max_out(x):
    if len(x.size()) == 2:
        s1, s2 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2 // 2, 2)
        x, _ = torch.max(x, 2)

    elif len(x.size()) == 3:
        s1, s2, s3 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2, s3 // 2, 2)
        x, _ = torch.max(x, 3)

    return x


class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, config):
        super(BaseEncoder, self).__init__()
        self.config = config
        self.hid_size = hid_size
        self.num_lyr = config.num_layers
        self.drop = nn.Dropout(config.dropout)
        self.direction = 2 if config.bidirectional else 1
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=0, sparse=False)
        self.rnn = nn.GRU(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=self.num_lyr,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout,
        )

    def forward(self, inp, length):
        length, indices = length.sort(descending=True)
        x = inp.index_select(0, indices)

        bt_siz, seq_len = x.size(0), x.size(1)

        h_0 = torch.zeros(
            self.direction * self.num_lyr,
            bt_siz,
            self.hid_size,
            device=self.config.device,
            dtype=torch.float32,
        )
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)
        x_emb = pack_padded_sequence(x_emb, length, True)

        x_o, x_hid = self.rnn(x_emb, h_0)
        if self.direction == 2:
            x_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(x_hid[2 * i : 2 * i + 2, :, :], 0, keepdim=True)
                x_hids.append(x_hid_temp)
            x_hid = torch.cat(x_hids, 0)

        x_hid = x_hid[self.num_lyr - 1, :, :].unsqueeze(0)
        x_hid = x_hid.transpose(0, 1)
        _, reverse_indices = indices.sort()
        x_hid = x_hid.index_select(0, reverse_indices)
        return x_hid


class CoarseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, config):
        super(CoarseEncoder, self).__init__()
        self.config = config
        self.hid_size = hid_size
        self.num_lyr = config.num_layers
        self.drop = nn.Dropout(config.dropout)
        self.direction = 2 if config.bidirectional else 1
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=0, sparse=False)
        self.rnn = nn.GRU(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=self.num_lyr,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout,
        )

    def forward(self, inp, length):
        length, indices = length.sort(descending=True)
        x = pad_sequence([inp[idx] for idx in indices], True, 0)

        bt_siz, seq_len = x.size(0), x.size(1)

        h_0 = torch.zeros(
            self.direction * self.num_lyr,
            bt_siz,
            self.hid_size,
            device=self.config.device,
            dtype=torch.float32,
        )
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)
        x_emb = pack_padded_sequence(x_emb, length, True)

        x_o, x_hid = self.rnn(x_emb, h_0)
        if self.direction == 2:
            x_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(x_hid[2 * i : 2 * i + 2, :, :], 0, keepdim=True)
                x_hids.append(x_hid_temp)
            x_hid = torch.cat(x_hids, 0)

        x_hid = x_hid[self.num_lyr - 1, :, :].unsqueeze(0)
        x_hid = x_hid.transpose(0, 1)
        _, reverse_indices = indices.sort()
        x_hid = x_hid.index_select(0, reverse_indices)
        return x_hid


class SessionEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, config, mode=None):
        super(SessionEncoder, self).__init__()
        self.config = config
        self.mode = mode
        self.hid_size = hid_size
        self.rnn = nn.GRU(
            hidden_size=hid_size,
            input_size=inp_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
            dropout=config.dropout,
        )

    def forward(self, x, h_0):
        if h_0 is None:
            h_0 = torch.zeros(
                1, x.size(0), self.hid_size, device=self.config.device, dtype=torch.float32
            )

        h_o, h_n = self.rnn(x, h_0)
        h_n = h_n.view(x.size(0), -1, self.hid_size)
        if self.mode == "coarse":
            return h_o
        return h_n


class Decoder(nn.Module):
    def __init__(self, config, mode=None):
        super(Decoder, self).__init__()
        self.config = config
        self.mode = mode
        self.emb_size = config.embedding_size
        self.hid_size = config.decoder_hidden_size
        self.num_lyr = 1
        self.teacher_forcing = True

        self.drop = nn.Dropout(config.dropout)
        self.tanh = nn.Tanh()

        if mode == "coarse":
            self.embed_in = nn.Embedding(
                config.word_size, self.emb_size, padding_idx=0, sparse=False
            )
            self.embed_out = nn.Linear(self.emb_size, config.word_size, bias=False)
        else:
            self.embed_in = nn.Embedding(
                config.syllable_size, self.emb_size, padding_idx=0, sparse=False
            )
            self.embed_out = nn.Linear(self.emb_size, config.syllable_size, bias=False)

        self.rnn = nn.GRU(
            hidden_size=self.hid_size,
            input_size=self.emb_size,
            num_layers=self.num_lyr,
            batch_first=True,
            dropout=config.dropout,
        )

        if config.model == "mrrnn" and mode is None:
            self.ses_to_dec = nn.Linear(config.context_hidden_size * 2, self.hid_size)
            self.ses_inf = nn.Linear(config.context_hidden_size * 2, self.emb_size * 2, False)
        else:
            self.ses_to_dec = nn.Linear(config.context_hidden_size, self.hid_size)
            self.ses_inf = nn.Linear(config.context_hidden_size, self.emb_size * 2, False)
        self.dec_inf = nn.Linear(self.hid_size, self.emb_size * 2, False)
        self.emb_inf = nn.Linear(self.emb_size, self.emb_size * 2, True)
        self.tc_ratio = 0.2

    def do_decode_tc(self, ses_encoding, target, length):
        length, indices = length.sort(descending=True)
        x = target.index_select(0, indices)

        target_emb = self.embed_in(x)
        target_emb = self.drop(target_emb)
        emb_inf_vec = self.emb_inf(target_emb)
        target_emb = pack_padded_sequence(target_emb, length, True)

        init_hidn = self.tanh(self.ses_to_dec(ses_encoding))
        init_hidn = init_hidn.view(self.num_lyr, target.size(0), self.hid_size)

        hid_o, _ = self.rnn(target_emb, init_hidn)
        hid_o = pad_packed_sequence(hid_o, True, 0, target.size(1))

        _, reverse_indices = indices.sort()
        hid_o = hid_o[0].index_select(0, reverse_indices)

        dec_hid_vec = self.dec_inf(hid_o)
        ses_inf_vec = self.ses_inf(ses_encoding)
        total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec

        hid_o_mx = max_out(total_hid_o)
        hid_o_mx = self.embed_out(hid_o_mx)

        if self.mode == "coarse":
            return hid_o_mx, hid_o_mx.max(2)[1]
        else:
            return hid_o_mx

    def do_decode(self, siz, seq_len, ses_encoding, target, length):
        ses_inf_vec = self.ses_inf(ses_encoding)
        ses_encoding = self.tanh(self.ses_to_dec(ses_encoding))
        hid_n, preds, lm_preds = ses_encoding, [], []

        hid_n = hid_n.view(self.num_lyr, siz, self.hid_size)
        inp_tok = torch.ones(siz, 1, device=self.config.device, dtype=torch.int64).new_full(
            (siz, 1), 2
        )
        out_words = torch.ones(siz, 1, device=self.config.device, dtype=torch.int64).new_full(
            (siz, 1), 2
        )
        for i in range(seq_len):
            inp_tok_vec = self.embed_in(inp_tok)
            emb_inf_vec = self.emb_inf(inp_tok_vec)

            inp_tok_vec = self.drop(inp_tok_vec)

            hid_o, hid_n = self.rnn(inp_tok_vec, hid_n)
            dec_hid_vec = self.dec_inf(hid_o)

            total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec
            hid_o_mx = max_out(total_hid_o)

            hid_o_mx = self.embed_out(hid_o_mx)
            _, inp_tok = torch.max(hid_o_mx, dim=2)
            out_words = torch.cat((out_words, inp_tok), 1)

        return out_words

    def forward(self, input, length):
        if len(input) == 1:
            ses_encoding = input
            x = None
        elif len(input) == 2:
            ses_encoding, x = input
        else:
            ses_encoding, x, beam = input

        siz, seq_len = x.size(0), x.size(1)

        if len(input) == 3 and beam is None:
            dec_o = self.do_decode(siz, length, ses_encoding, None, None)

        elif self.teacher_forcing:
            dec_o = self.do_decode_tc(ses_encoding, x, length)
        else:
            dec_o = self.do_decode(siz, seq_len, ses_encoding, x, length)

        return dec_o

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val

    def get_teacher_forcing(self):
        return self.teacher_forcing

    def set_tc_ratio(self, new_val):
        self.tc_ratio = new_val

    def get_tc_ratio(self):
        return self.tc_ratio


class Coarse(nn.Module):
    def __init__(self, config):
        super(Coarse, self).__init__()
        self.config = config
        self.base_enc = BaseEncoder(
            config.word_size, config.embedding_size, config.encoder_hidden_size, config
        )
        self.ses_enc = SessionEncoder(
            config.context_hidden_size, config.encoder_hidden_size, config, "coarse"
        )
        self.dec = Decoder(config, "coarse")
        self.coarse_enc = CoarseEncoder(
            config.word_size, config.embedding_size, config.encoder_hidden_size, config
        )
        # NOTE: `size_average=False` (removed in modern torch) -> `reduction='sum'`,
        # the documented modern equivalent.
        self.criteria = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    def forward(self, coarse_data, coarse_length):
        us = coarse_data[:-1].transpose(0, 1).contiguous().clone()
        us_length = coarse_length[:-1].transpose(0, 1).contiguous().clone()
        qu_seq = self.base_enc(us.view(-1, us.size(2)), us_length.view(-1)).view(
            us.size(0), us.size(1), -1
        )
        final_session_o = self.ses_enc(qu_seq, None)
        loss = 0
        pred_words = [
            torch.tensor([], device=self.config.device, dtype=torch.int64)
            for _ in range(coarse_data.size(1))
        ]
        for i in range(final_session_o.size(1)):  # sequence
            preds, words = self.dec(
                (final_session_o[:, i : i + 1, :], coarse_data[i + 1, :, :]),
                coarse_length[i + 1, :],
            )
            loss = loss + self.criteria(
                preds.view(-1, self.config.word_size), coarse_data[i + 1].contiguous().view(-1)
            )
            words = [
                words[idx][: coarse_length[i + 1][idx]] for idx in range(coarse_data.size(1))
            ]  # batch
            pred_words = [
                torch.cat((pred_words[idx], words[idx])) for idx in range(coarse_data.size(1))
            ]

        coarse_hidden = self.coarse_enc(pred_words, coarse_length[1:].sum(0))
        total_length = coarse_length[1:].sum()
        return coarse_hidden, loss / total_length.item()


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.base_enc = BaseEncoder(
            config.syllable_size, config.embedding_size, config.encoder_hidden_size, config
        )
        self.ses_enc = SessionEncoder(
            config.context_hidden_size, config.encoder_hidden_size, config
        )
        self.dec = Decoder(config)
        if config.model.lower() == "mrrnn":
            self.coarse = Coarse(config)

    def forward(self, syllable_data, syllable_length, coarse_data, coarse_length):
        us = syllable_data[:-1].transpose(0, 1).contiguous().clone()
        us_length = syllable_length[:-1].transpose(0, 1).contiguous().clone()
        gold = syllable_data[-1]
        gold_length = syllable_length[-1]
        qu_seq = self.base_enc(us.view(-1, us.size(2)), us_length.view(-1)).view(
            us.size(0), us.size(1), -1
        )
        final_session_o = self.ses_enc(qu_seq, None)
        loss = None
        if self.config.model.lower() == "mrrnn":
            coarse_hidden, loss = self.coarse(coarse_data, coarse_length)
            final_session_o = torch.cat((final_session_o, coarse_hidden), 2)

        preds = self.dec((final_session_o, gold), gold_length)

        if self.config.model.lower() == "mrrnn":
            return preds, loss

        return preds


def _build_config():
    return SimpleNamespace(
        model="mrrnn",
        mode="train",
        num_layers=1,
        bidirectional=True,
        embedding_size=24,
        encoder_hidden_size=32,
        context_hidden_size=32,
        decoder_hidden_size=40,
        dropout=0.0,
        syllable_size=40,
        word_size=30,
        device=torch.device("cpu"),
    )


def build_mrrnn():
    config = _build_config()
    return Seq2Seq(config)


def example_input_mrrnn():
    # Shapes follow `Seq2Seq.forward`'s slicing convention: dim0 = utterance-in-session
    # index (last one is the "gold" utterance to reconstruct), dim1 = batch,
    # dim2 = syllable/word position within an utterance.
    n_utts = 3  # >=2 so `[:-1]` (context) is non-empty and `[-1]` (gold) exists
    batch = 2
    syl_len = 5
    word_len = 4

    syllable_data = torch.randint(1, 40, (n_utts, batch, syl_len))
    syllable_length = torch.full((n_utts, batch), syl_len, dtype=torch.int64)

    coarse_data = torch.randint(1, 30, (n_utts, batch, word_len))
    coarse_length = torch.full((n_utts, batch), word_len, dtype=torch.int64)

    return (syllable_data, syllable_length, coarse_data, coarse_length)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MrRNN", "build_mrrnn", "example_input_mrrnn", 2017, "vendored"),
]
