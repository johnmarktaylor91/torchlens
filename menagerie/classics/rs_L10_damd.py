# SOURCE: vendored from thu-spmi/damd-multiwoz @ 9b0456d7e590fb5de77ec81e967e8010487eeb56
# (damd_net.py, one encoder + three parallel span/response decoders with pointer-generator
# copy mechanism for task-oriented dialogue; Zhang et al., AAAI 2020 "Task-Oriented Dialog
# Systems that Consider Multiple Appropriate Responses under the Same Context").
#
# The real nn.Module classes below (Attn, LayerNormalization, MultiLayerGRUwithLN,
# biGRUencoder, Copy, DomainSpanDecoder, BeliefSpanDecoder, ActSpanDecoder, ResponseDecoder,
# ActSelectionModel, DAMD, get_final_scores, init_gru, label_smoothing) are copied verbatim
# from the official repo's damd_net.py. Only two things were changed to make the file
# import/run standalone in a base torch env, with no architectural edits:
#   1. `import utils` / `from config import global_config as cfg` -> replaced with a tiny
#      local `cfg` object (`_DamdConfig`, same field names/defaults as the repo's
#      `config.py:_Config._multiwoz_damd_init`) and an inlined `cuda_` no-op helper
#      (repo's `utils.cuda_`, trivial one-liner) so nothing needs the original repo's
#      data/vocab/config machinery on disk.
#   2. The repo's `reader.vocab.vocab_size` / `vocab_size_oov` contract is satisfied with a
#      minimal fake `_TinyVocab` (2 attributes only) instead of building the full SWDA/MultiWOZ
#      `Vocab` class, since DAMD's constructor only reads those two integers off `reader.vocab`.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cuda_(var):
    # repo's utils.cuda_: moves to GPU only if cfg.cuda; kept CPU-only here (no arch change).
    return var


class _DamdConfig:
    """Mirrors config.py:_Config._multiwoz_damd_init field names/defaults (data/eval-only
    fields such as file paths are omitted; every field the model code itself reads is kept)."""

    def __init__(self):
        self.vocab_size = 48
        self.embed_size = 16
        self.hidden_size = 24
        self.pointer_dim = 6
        self.enc_layer_num = 1
        self.dec_layer_num = 1
        self.dropout = 0.0
        self.layer_norm = False
        self.skip_connect = False
        self.encoder_share = False
        self.attn_param_share = False
        self.copy_param_share = False
        self.enable_aspn = True
        self.use_pvaspn = False
        self.enable_bspn = True
        self.bspn_mode = "bsdx"
        self.enable_dspn = False
        self.enable_dst = False
        self.label_smoothing = 0.0
        self.max_span_length = 8
        self.max_nl_length = 8
        self.teacher_force = 100
        self.beam_width = 2
        self.nbest = 2
        self.multi_acts_training = False


cfg = _DamdConfig()


def init_gru(gru):
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    gru.apply(weight_reset)
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i : i + gru.hidden_size], gain=1)


def label_smoothing(labels, smoothing_rate, vocab_size_oov):
    with torch.no_grad():
        confidence = 1.0 - smoothing_rate
        low_confidence = (1.0 - confidence) / labels.new_tensor(vocab_size_oov - 1)
        y_tensor = labels.data
        y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
        n_dims = vocab_size_oov
        y_one_hot = (
            torch.zeros(y_tensor.size()[0], n_dims)
            .fill_(low_confidence)
            .scatter_(1, y_tensor, confidence)
        )
        y_one_hot = cuda_(y_one_hot.view(*labels.shape, -1))
    return y_one_hot


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        attn_energies = self.score(hidden, encoder_outputs)  # [B,T,H]
        if mask is None:
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        else:
            attn_energies.masked_fill_(mask, -1e20)
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]

        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context  # [B,1, H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # [B,T,H]
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = self.v(energy).transpose(1, 2)  # [B,1,T]
        return energy


class LayerNormalization(nn.Module):
    """Layer normalization module"""

    def __init__(self, hidden_size, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out


class MultiLayerGRUwithLN(nn.Module):
    """multi-layer GRU with layer normalization"""

    def __init__(
        self,
        input_size,
        hidden_size,
        layer_num=1,
        bidirec=False,
        layer_norm=False,
        skip_connect=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirec = bidirec
        self.layer_norm = layer_norm
        self.skip_connect = skip_connect
        self.dropout = dropout
        self.model_layers = nn.ModuleDict()
        self.all_weights = []
        for l in range(self.layer_num):  # noqa: E741 -- verbatim var name from vendored source
            if l == 0:
                gru = nn.GRU(
                    self.input_size,
                    self.hidden_size,
                    num_layers=1,
                    dropout=self.dropout,
                    bidirectional=self.bidirec,
                    batch_first=True,
                )
            else:
                input_size = self.hidden_size if not self.bidirec else 2 * self.hidden_size
                gru = nn.GRU(
                    input_size,
                    self.hidden_size,
                    num_layers=1,
                    dropout=self.dropout,
                    bidirectional=self.bidirec,
                    batch_first=True,
                )
            self.model_layers["gru_" + str(l)] = gru
            self.all_weights.extend(gru.all_weights)
            if self.layer_norm:
                output_size = self.hidden_size if not self.bidirec else 2 * self.hidden_size
                ln = nn.LayerNorm(output_size)
                self.model_layers["ln_" + str(l)] = ln

    def forward(self, inputs, hidden=None):
        batch_size = inputs.size()[0]
        in_l, last_input = inputs, None
        hs = []
        if hidden:
            hiddens = hidden.view(self.layer_num, self.bidirec, batch_size, self.hidden_size)
        for l in range(self.layer_num):  # noqa: E741 -- verbatim var name from vendored source
            init_hs = hiddens[l] if hidden else None
            in_l, hs_l = self.model_layers["gru_" + str(l)](in_l, init_hs)
            hs.append(hs_l)
            if self.layer_norm:
                in_l = self.model_layers["ln_" + str(l)](in_l)
            if self.dropout > 0 and l < (self.layer_num - 1):
                in_l = F.dropout(in_l)
            if self.skip_connect and last_input is not None:
                in_l = in_l + last_input
            last_input = in_l
        hs = torch.cat(hs, 0)
        return in_l, hs


class biGRUencoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.hidden_size = cfg.hidden_size
        if cfg.enc_layer_num == 1:
            self.gru = nn.GRU(
                self.embed_size,
                cfg.hidden_size,
                cfg.enc_layer_num,
                dropout=cfg.dropout,
                bidirectional=True,
                batch_first=True,
            )
        else:
            self.gru = MultiLayerGRUwithLN(
                self.embed_size,
                cfg.hidden_size,
                cfg.enc_layer_num,
                bidirec=True,
                layer_norm=cfg.layer_norm,
                skip_connect=cfg.skip_connect,
                dropout=cfg.dropout,
            )
        init_gru(self.gru)

    def forward(self, input_seqs, hidden=None):
        embedded = self.embedding(input_seqs)
        outputs, hidden = self.gru(embedded, hidden)
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        return outputs, hidden


class Copy(nn.Module):
    def __init__(self, hidden_size, copy_weight=1.0):
        super().__init__()
        self.Wcopy = nn.Linear(hidden_size, hidden_size)
        self.copy_weight = copy_weight

    def forward(self, enc_out_hs, dec_hs):
        raw_cp_score = torch.tanh(self.Wcopy(enc_out_hs))  # [B,Tenc,H]
        raw_cp_score = torch.einsum("beh,bdh->bde", raw_cp_score, dec_hs)  # [B, Tdec, Tenc]
        return raw_cp_score * self.copy_weight


def get_final_scores(raw_scores, word_onehot_input, input_idx_oov, vocab_size_oov):
    for idx, raw_sc in enumerate(raw_scores):
        if idx == 0:
            continue
        one_hot = word_onehot_input[idx - 1]  # [B, Tenc_i, V+Tenc_i]
        cps = torch.einsum("imj,ijn->imn", raw_sc, one_hot)  # [B, Tdec, V+Tenc_i]
        raw_scores[idx] = cps

    cum_idx = [score.size(2) for score in raw_scores]
    for i in range(len(cum_idx) - 1):
        cum_idx[i + 1] += cum_idx[i]
    cum_idx.insert(0, 0)

    logsoftmax = torch.nn.LogSoftmax(dim=2)
    normalized_scores = logsoftmax(torch.cat(raw_scores, dim=2))  # [B,Tdec, V+V+Tenc1+V+Tenc2+...]

    gen_score = normalized_scores[:, :, cum_idx[0] : cum_idx[1]]  # [B, Tdec, V]
    Tdec = gen_score.size(1)
    B = gen_score.size(0)
    V = gen_score.size(2)

    total_score = cuda_(torch.zeros(B, Tdec, vocab_size_oov)).fill_(
        -1e20
    )  # [B, Tdec, vocab_size_oov]
    c_to_g_scores = []
    for i in range(1, len(cum_idx) - 1):
        cps = normalized_scores[:, :, cum_idx[i] : cum_idx[i + 1]]  # [B, Tdec, V+Tenc_i]
        c_to_g_scores.append(cps[:, :, :V])
        cp_score = cps[:, :, V:]
        avail_copy_idx = (input_idx_oov[i - 1] >= V).nonzero()
        for idx in avail_copy_idx:
            b, t = idx[0], idx[1]
            ts = total_score[b, :, input_idx_oov[i - 1][b, t]].view(Tdec, 1)
            cs = cp_score[b, :, t].view(Tdec, 1)
            total_score[b, :, input_idx_oov[i - 1][b, t]] = torch.logsumexp(
                torch.cat([ts, cs], 1), 1
            )

    gen_score = torch.logsumexp(torch.stack([gen_score] + c_to_g_scores, 3), 3)
    total_score[:, :, :V] = gen_score
    return total_score.contiguous()  # [B, Tdec, vocab_size_oov]


class DomainSpanDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, Wgen=None, dropout=0.0):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        self.gru = nn.GRU(
            3 * cfg.hidden_size + self.embed_size,
            cfg.hidden_size,
            cfg.dec_layer_num,
            dropout=cfg.dropout,
            batch_first=True,
        )
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen

        self.attn_user = Attn(cfg.hidden_size)
        self.attn_pvresp = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)
        self.attn_pvdspn = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_pvdspn = Copy(cfg.hidden_size)

    def forward(
        self, inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step, mode="train"
    ):
        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        gru_input.append(embed_last_w)

        if first_step:
            self.mask_user = (inputs["user"] == 0).unsqueeze(1)  # [B,1,T]
            self.mask_pvresp = (inputs["pv_resp"] == 0).unsqueeze(1)  # [B,1,T]
            self.mask_pvdspn = (inputs["pv_dspn"] == 0).unsqueeze(1)  # [B,1,T]
        if mode == "test" and not first_step:
            self.mask_pvresp = (inputs["pv_resp"] == 0).unsqueeze(1)  # [B,1,T]
            self.mask_pvdspn = (inputs["pv_dspn"] == 0).unsqueeze(1)  # [B,1,T]

        context_user = self.attn_user(dec_last_h, hidden_states["user"], self.mask_user)
        gru_input.append(context_user)
        if not first_turn:
            context_pvresp = self.attn_pvresp(dec_last_h, hidden_states["resp"], self.mask_pvresp)
            context_pvdspn = self.attn_pvdspn(dec_last_h, hidden_states["dspn"], self.mask_pvdspn)
        else:
            batch_size = inputs["user"].size(0)
            context_pvresp = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))
            context_pvdspn = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))
        gru_input.append(context_pvresp)
        gru_input.append(context_pvdspn)

        gru_out, dec_last_h = self.gru(
            torch.cat(gru_input, 2), dec_last_h
        )  # [B, 1, H], [n_layer, B, H]
        return dec_last_h

    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False):
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)  # [B, Tdec, H]
        raw_scores.append(raw_gen_score)

        if not first_turn:
            raw_cp_score_dspn = self.cp_pvdspn(hidden_states["dspn"], dec_hs)  # [B,Ta]
            raw_cp_score_dspn.masked_fill_(self.mask_pvdspn.repeat(1, Tdec, 1), -1e20)
            raw_scores.append(raw_cp_score_dspn)
            word_onehot_input.append(inputs["pv_dspn_onehot"])
            input_idx_oov.append(inputs["pv_dspn_nounk"])

        probs = get_final_scores(raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov)

        return probs


class BeliefSpanDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, bspn_mode, Wgen=None, dropout=0.0):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        self.bspn_mode = bspn_mode

        self.gru = nn.GRU(
            3 * cfg.hidden_size + self.embed_size,
            cfg.hidden_size,
            cfg.dec_layer_num,
            dropout=cfg.dropout,
            batch_first=True,
        )
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen

        self.attn_user = Attn(cfg.hidden_size)
        self.attn_pvresp = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)
        self.attn_pvbspn = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_user = Copy(cfg.hidden_size, 1.0)
        self.cp_pvresp = self.cp_user if cfg.copy_param_share else Copy(cfg.hidden_size)
        self.cp_pvbspn = self.cp_user if cfg.copy_param_share else Copy(cfg.hidden_size, 1.0)

        self.mask_user = None
        self.mask_pvresp = None
        self.mask_pvbspn = None

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)  # input dropout

    def forward(
        self, inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step, mode="train"
    ):
        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        gru_input.append(embed_last_w)

        if first_step:
            self.mask_user = (inputs["user"] == 0).unsqueeze(1)  # [B,1,T]
            self.mask_pvresp = (inputs["pv_resp"] == 0).unsqueeze(1)  # [B,1,T]
            self.mask_pvbspn = (inputs["pv_" + self.bspn_mode] == 0).unsqueeze(1)  # [B,1,T]
        if mode == "test" and not first_step:
            self.mask_pvresp = (inputs["pv_resp"] == 0).unsqueeze(1)  # [B,1,T]
            self.mask_pvbspn = (inputs["pv_" + self.bspn_mode] == 0).unsqueeze(1)  # [B,1,T]

        context_user = self.attn_user(dec_last_h, hidden_states["user"], self.mask_user)
        gru_input.append(context_user)
        if not first_turn:
            context_pvresp = self.attn_pvresp(dec_last_h, hidden_states["resp"], self.mask_pvresp)
            context_pvbspn = self.attn_pvbspn(
                dec_last_h, hidden_states[self.bspn_mode], self.mask_pvbspn
            )
        else:
            batch_size = inputs["user"].size(0)
            context_pvresp = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))
            context_pvbspn = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))
        gru_input.append(context_pvresp)
        gru_input.append(context_pvbspn)

        gru_out, dec_last_h = self.gru(
            torch.cat(gru_input, 2), dec_last_h
        )  # [B, 1, H], [n_layer, B, H]
        return dec_last_h

    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False):
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)  # [B, Tdec, V]
        raw_scores.append(raw_gen_score)

        raw_cp_score_user = self.cp_user(hidden_states["user"], dec_hs)  # [B, Tdec,Tu]
        raw_cp_score_user.masked_fill_(self.mask_user.repeat(1, Tdec, 1), -1e20)
        raw_scores.append(raw_cp_score_user)
        word_onehot_input.append(inputs["user_onehot"])
        input_idx_oov.append(inputs["user_nounk"])

        if not first_turn:
            raw_cp_score_pvresp = self.cp_pvresp(hidden_states["resp"], dec_hs)  # [B, Tdec,Tr]
            raw_cp_score_pvresp.masked_fill_(self.mask_pvresp.repeat(1, Tdec, 1), -1e20)
            raw_scores.append(raw_cp_score_pvresp)
            word_onehot_input.append(inputs["pv_resp_onehot"])
            input_idx_oov.append(inputs["pv_resp_nounk"])

            raw_cp_score_pvbspn = self.cp_pvbspn(
                hidden_states[self.bspn_mode], dec_hs
            )  # [B, Tdec, Tb]
            raw_cp_score_pvbspn.masked_fill_(self.mask_pvbspn.repeat(1, Tdec, 1), -1e20)
            raw_scores.append(raw_cp_score_pvbspn)
            word_onehot_input.append(inputs["pv_%s_onehot" % self.bspn_mode])
            input_idx_oov.append(inputs["pv_%s_nounk" % self.bspn_mode])

        probs = get_final_scores(
            raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov
        )  # [B, V_oov]

        return probs


class ActSpanDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, Wgen=None, dropout=0.0):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        input_dim = cfg.hidden_size + self.embed_size + cfg.pointer_dim
        if cfg.use_pvaspn:
            input_dim += cfg.hidden_size
        if cfg.enable_bspn:
            input_dim += cfg.hidden_size
        if cfg.enable_dspn:
            input_dim += cfg.hidden_size

        self.gru = nn.GRU(
            input_dim, cfg.hidden_size, cfg.dec_layer_num, dropout=cfg.dropout, batch_first=True
        )
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen

        self.attn_usdx = Attn(cfg.hidden_size)
        if cfg.enable_bspn:
            self.attn_bspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)
        if cfg.enable_dspn:
            self.attn_dspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)
        self.attn_pvaspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_pvaspn = Copy(cfg.hidden_size)
        self.cp_dspn = self.cp_pvaspn if cfg.copy_param_share else Copy(cfg.hidden_size)
        self.cp_bspn = self.cp_pvaspn if cfg.copy_param_share else Copy(cfg.hidden_size)

        self.mask_usdx = None
        self.mask_bspn = None
        self.mask_dspn = None
        self.mask_pvaspn = None

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(cfg.dropout)  # input dropout

    def forward(
        self,
        inputs,
        hidden_states,
        dec_last_w,
        dec_last_h,
        first_turn,
        first_step,
        bidx=None,
        mode="train",
    ):
        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        gru_input.append(embed_last_w)

        if first_step:
            self.mask_usdx = (inputs["usdx"] == 0).unsqueeze(1)  # [B,1,T]
            self.mask_pvaspn = (inputs["pv_aspn"] == 0).unsqueeze(1)  # [B,1,T]
            if cfg.enable_bspn:
                self.mask_bspn = (inputs[cfg.bspn_mode] == 0).unsqueeze(1)  # [B,1,T]
            if cfg.enable_dspn:
                self.mask_dspn = (inputs["dspn"] == 0).unsqueeze(1)  # [B,1,T]
        if mode == "test" and not first_step:
            self.mask_pvaspn = (inputs["pv_aspn"] == 0).unsqueeze(1)  # [B,1,T]
            if cfg.enable_bspn:
                self.mask_bspn = (inputs[cfg.bspn_mode] == 0).unsqueeze(1)  # [B,1,T]
            if cfg.enable_dspn:
                self.mask_dspn = (inputs["dspn"] == 0).unsqueeze(1)  # [B,1,T]

        if bidx is None:
            context_usdx = self.attn_usdx(dec_last_h, hidden_states["usdx"], self.mask_usdx)
        else:
            context_usdx = self.attn_usdx(
                dec_last_h, hidden_states["usdx"][bidx], self.mask_usdx[bidx]
            )
        gru_input.append(context_usdx)
        if cfg.enable_bspn:
            if bidx is None:
                context_bspn = self.attn_bspn(
                    dec_last_h, hidden_states[cfg.bspn_mode], self.mask_bspn
                )
            else:
                context_bspn = self.attn_bspn(
                    dec_last_h, hidden_states[cfg.bspn_mode][bidx], self.mask_bspn[bidx]
                )
            gru_input.append(context_bspn)
        if cfg.enable_dspn:
            if bidx is None:
                context_dspn = self.attn_dspn(dec_last_h, hidden_states["dspn"], self.mask_dspn)
            else:
                context_dspn = self.attn_dspn(
                    dec_last_h, hidden_states["dspn"][bidx], self.mask_dspn[bidx]
                )
            gru_input.append(context_dspn)
        if cfg.use_pvaspn:
            if not first_turn:
                if bidx is None:
                    context_pvaspn = self.attn_pvaspn(
                        dec_last_h, hidden_states["aspn"], self.mask_pvaspn
                    )
                else:
                    context_pvaspn = self.attn_pvaspn(
                        dec_last_h, hidden_states["aspn"][bidx], self.mask_pvaspn[bidx]
                    )
            else:
                if bidx is None:
                    context_pvaspn = cuda_(torch.zeros(inputs["user"].size(0), 1, cfg.hidden_size))
                else:
                    context_pvaspn = cuda_(torch.zeros(1, 1, cfg.hidden_size))
            gru_input.append(context_pvaspn)

        if bidx is None:
            gru_input.append(inputs["db"].unsqueeze(1))
        else:
            gru_input.append(inputs["db"][bidx].unsqueeze(1))

        gru_out, dec_last_h = self.gru(
            torch.cat(gru_input, 2), dec_last_h
        )  # [B, 1, H], [n_layer, B, H]
        return dec_last_h

    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False, bidx=None):
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)  # [B, Tdec, H]
        raw_scores.append(raw_gen_score)

        if cfg.enable_bspn:
            if bidx is None:
                raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode], dec_hs)  # [B,Tb]
                raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1, Tdec, 1), -1e20)
                raw_scores.append(raw_cp_score_bspn)
                word_onehot_input.append(inputs[cfg.bspn_mode + "_onehot"])
                input_idx_oov.append(inputs[cfg.bspn_mode + "_nounk"])
            else:
                raw_cp_score_bspn = self.cp_bspn(
                    hidden_states[cfg.bspn_mode][bidx], dec_hs
                )  # [B,Tb]
                raw_cp_score_bspn.masked_fill_(self.mask_bspn[bidx].repeat(1, Tdec, 1), -1e20)
                raw_scores.append(raw_cp_score_bspn)
                word_onehot_input.append(inputs[cfg.bspn_mode + "_onehot"][bidx])
                input_idx_oov.append(inputs[cfg.bspn_mode + "_nounk"][bidx])

        if cfg.enable_dspn:
            if bidx is None:
                raw_cp_score_dspn = self.cp_dspn(hidden_states["dspn"], dec_hs)  # [B,Tb]
                raw_cp_score_dspn.masked_fill_(self.mask_dspn.repeat(1, Tdec, 1), -1e20)
                raw_scores.append(raw_cp_score_dspn)
                word_onehot_input.append(inputs["dspn_onehot"])
                input_idx_oov.append(inputs["dspn_nounk"])
            else:
                raw_cp_score_dspn = self.cp_dspn(hidden_states["dspn"][bidx], dec_hs)  # [B,Tb]
                raw_cp_score_dspn.masked_fill_(self.mask_dspn[bidx].repeat(1, Tdec, 1), -1e20)
                raw_scores.append(raw_cp_score_dspn)
                word_onehot_input.append(inputs["dspn_onehot"][bidx])
                input_idx_oov.append(inputs["dspn_nounk"][bidx])

        if not first_turn and cfg.use_pvaspn:
            if bidx is None:
                raw_cp_score_aspn = self.cp_pvaspn(hidden_states["aspn"], dec_hs)  # [B,Ta]
                raw_cp_score_aspn.masked_fill_(self.mask_pvaspn.repeat(1, Tdec, 1), -1e20)
                raw_scores.append(raw_cp_score_aspn)
                word_onehot_input.append(inputs["pv_aspn_onehot"])
                input_idx_oov.append(inputs["pv_aspn_nounk"])
            else:
                raw_cp_score_aspn = self.cp_pvaspn(hidden_states["aspn"][bidx], dec_hs)  # [B,Ta]
                raw_cp_score_aspn.masked_fill_(self.mask_pvaspn[bidx].repeat(1, Tdec, 1), -1e20)
                raw_scores.append(raw_cp_score_aspn)
                word_onehot_input.append(inputs["pv_aspn_onehot"][bidx])
                input_idx_oov.append(inputs["pv_aspn_nounk"][bidx])

        probs = get_final_scores(raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov)

        return probs


class ResponseDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, Wgen=None, dropout=0.0):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        gru_input_size = cfg.hidden_size + self.embed_size + cfg.pointer_dim
        if cfg.enable_bspn:
            gru_input_size += cfg.hidden_size
        if cfg.enable_aspn:
            gru_input_size += cfg.hidden_size

        self.gru = nn.GRU(
            gru_input_size,
            cfg.hidden_size,
            cfg.dec_layer_num,
            dropout=cfg.dropout,
            batch_first=True,
        )
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen
        self.attn_usdx = Attn(cfg.hidden_size)
        if cfg.enable_bspn:
            self.attn_bspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)
        if cfg.enable_aspn:
            self.attn_aspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_usdx = Copy(cfg.hidden_size)
        if cfg.enable_bspn:
            self.cp_bspn = self.cp_usdx if cfg.copy_param_share else Copy(cfg.hidden_size)
        if cfg.enable_aspn:
            self.cp_aspn = self.cp_usdx if cfg.copy_param_share else Copy(cfg.hidden_size)

        self.mask_usdx = None
        self.mask_bspn = None
        if cfg.enable_aspn:
            self.mask_aspn = None

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)  # input dropout

    def forward(
        self, inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step, mode="train"
    ):
        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        gru_input.append(embed_last_w)

        if first_step:
            self.mask_usdx = (inputs["usdx"] == 0).unsqueeze(1)  # [B,1,T]
            if cfg.enable_bspn:
                self.mask_bspn = (inputs[cfg.bspn_mode] == 0).unsqueeze(1)  # [B,1,T]
            if cfg.enable_aspn:
                self.mask_aspn = (inputs["aspn"] == 0).unsqueeze(1)  # [B,1,T]
        if mode == "test" and not first_step:
            if cfg.enable_bspn:
                self.mask_bspn = (inputs[cfg.bspn_mode] == 0).unsqueeze(1)  # [B,1,T]
            if cfg.enable_aspn:
                self.mask_aspn = (inputs["aspn"] == 0).unsqueeze(1)  # [B,1,T]

        context_usdx = self.attn_usdx(dec_last_h, hidden_states["usdx"], self.mask_usdx)
        gru_input.append(context_usdx)
        if cfg.enable_bspn:
            context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.bspn_mode], self.mask_bspn)
            gru_input.append(context_bspn)
        if cfg.enable_aspn:
            context_aspn = self.attn_aspn(dec_last_h, hidden_states["aspn"], self.mask_aspn)
            gru_input.append(context_aspn)

        gru_input.append(inputs["db"].unsqueeze(1))

        gru_out, dec_last_h = self.gru(
            torch.cat(gru_input, 2), dec_last_h
        )  # [B, 1, H], [n_layer, B, H]

        return dec_last_h

    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False):
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)  # [B, Tdec, H]
        raw_scores.append(raw_gen_score)

        raw_cp_score_usdx = self.cp_usdx(hidden_states["usdx"], dec_hs)  # [B,Tu]
        raw_cp_score_usdx.masked_fill_(self.mask_usdx.repeat(1, Tdec, 1), -1e20)
        raw_scores.append(raw_cp_score_usdx)
        word_onehot_input.append(inputs["usdx_onehot"])
        input_idx_oov.append(inputs["usdx_nounk"])

        if cfg.enable_bspn:
            raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode], dec_hs)  # [B,Tb]
            raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1, Tdec, 1), -1e20)
            raw_scores.append(raw_cp_score_bspn)
            word_onehot_input.append(inputs[cfg.bspn_mode + "_onehot"])
            input_idx_oov.append(inputs[cfg.bspn_mode + "_nounk"])

        if cfg.enable_aspn:
            raw_cp_score_aspn = self.cp_aspn(hidden_states["aspn"], dec_hs)  # [B,Ta]
            raw_cp_score_aspn.masked_fill_(self.mask_aspn.repeat(1, Tdec, 1), -1e20)
            raw_scores.append(raw_cp_score_aspn)
            word_onehot_input.append(inputs["aspn_onehot"])
            input_idx_oov.append(inputs["aspn_nounk"])

        probs = get_final_scores(raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov)

        return probs


class ActSelectionModel(nn.Module):
    def __init__(self, hidden_size, length, nbest):
        super().__init__()
        self.nbest = nbest
        self.hidden_size = hidden_size
        self.length = length
        self.W1 = nn.Linear(hidden_size * length, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, hiddens_batch):
        batch_size = hiddens_batch.size()[0]
        logits = hiddens_batch.view(batch_size, self.nbest, -1)
        logits = self.W2(nn.ReLU()(self.W1(logits))).view(batch_size)
        logprob = self.logsoftmax(logits)  # [B,nbest]
        return logprob


class _TinyVocab:
    """Minimal stand-in for the repo's utils.Vocab: DAMD.__init__ only reads these two ints
    off reader.vocab (see damd_net.py DAMD.__init__: self.vocab_size / self.vsize_oov)."""

    def __init__(self, vocab_size, vocab_size_oov):
        self.vocab_size = vocab_size
        self.vocab_size_oov = vocab_size_oov


class _TinyReader:
    def __init__(self, vocab_size, vocab_size_oov):
        self.vocab = _TinyVocab(vocab_size, vocab_size_oov)


class DAMD(nn.Module):
    def __init__(self, reader):
        super().__init__()
        self.reader = reader
        self.vocab = self.reader.vocab
        self.vocab_size = self.vocab.vocab_size
        self.vsize_oov = self.vocab.vocab_size_oov
        self.embed_size = cfg.embed_size
        self.hidden_size = cfg.hidden_size
        self.n_layer = cfg.dec_layer_num
        self.dropout = cfg.dropout
        self.max_span_len = cfg.max_span_length
        self.max_nl_len = cfg.max_nl_length
        self.teacher_force = cfg.teacher_force
        self.label_smth = cfg.label_smoothing
        self.beam_width = cfg.beam_width
        self.nbest = cfg.nbest

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        self.user_encoder = biGRUencoder(self.embedding)
        if cfg.encoder_share:
            self.usdx_encoder = self.user_encoder
        else:
            self.usdx_encoder = biGRUencoder(self.embedding)
        self.span_encoder = biGRUencoder(self.embedding)

        Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if cfg.copy_param_share else None

        self.decoders = {}
        if cfg.enable_dspn:
            self.dspn_decoder = DomainSpanDecoder(
                self.embedding, self.vsize_oov, Wgen=Wgen, dropout=self.dropout
            )
            self.decoders["dspn"] = self.dspn_decoder
        if cfg.enable_bspn:
            self.bspn_decoder = BeliefSpanDecoder(
                self.embedding, self.vsize_oov, cfg.bspn_mode, Wgen=Wgen, dropout=self.dropout
            )
            self.decoders[cfg.bspn_mode] = self.bspn_decoder
        if cfg.enable_aspn:
            self.aspn_decoder = ActSpanDecoder(
                self.embedding, self.vsize_oov, Wgen=Wgen, dropout=self.dropout
            )
            self.decoders["aspn"] = self.aspn_decoder
        self.resp_decoder = ResponseDecoder(
            self.embedding, self.vsize_oov, Wgen=Wgen, dropout=self.dropout
        )
        self.decoders["resp"] = self.resp_decoder

        if cfg.enable_dst and cfg.bspn_mode == "bsdx":
            self.dst_decoder = BeliefSpanDecoder(
                self.embedding, self.vsize_oov, "bspn", Wgen=Wgen, dropout=self.dropout
            )
            self.decoders["bspn"] = self.dst_decoder

        self.nllloss = nn.NLLLoss(ignore_index=0)

        self.go_idx = {"bspn": 3, "bsdx": 3, "aspn": 4, "dspn": 9, "resp": 1}
        self.eos_idx = {"bspn": 7, "bsdx": 7, "aspn": 8, "dspn": 10, "resp": 6}
        self.teacher_forcing_decode = {
            "bspn": False,
            "bsdx": False,
            "aspn": False,
            "dspn": False,
            "resp": False,
        }
        self.limited_vocab_decode = {
            "bspn": False,
            "bsdx": False,
            "aspn": False,
            "dspn": False,
            "resp": False,
        }

    def forward(self, inputs, hidden_states, first_turn, mode="train"):
        # Registers only the shared (non-loss) forward computation: encoders + all enabled
        # decoders run for one decode step each (matches the repo's train_forward loop body,
        # unrolled to n_step iterations of train_decode). Loss/backprop bookkeeping
        # (supervised_loss) is training-harness code, not architecture, so it is omitted here.
        user_enc, user_enc_last_h = self.user_encoder(inputs["user"])
        usdx_enc, usdx_enc_last_h = self.usdx_encoder(inputs["usdx"])
        resp_enc, resp_enc_last_h = self.usdx_encoder(inputs["pv_resp"])
        hidden_states["user"] = user_enc
        hidden_states["usdx"] = usdx_enc
        hidden_states["resp"] = resp_enc

        probs = {}

        def train_decode(name, init_hidden, hidden_states, probs):
            batch_size = inputs["user"].size(0)
            dec_last_w = cuda_(torch.ones(batch_size, 1).long() * self.go_idx[name])
            dec_last_h = (init_hidden[-1] + init_hidden[-2]).unsqueeze(0)

            decode_step = inputs[name].size(1)
            hiddens = []
            for t in range(decode_step):
                first_step = t == 0
                dec_last_h = self.decoders[name](
                    inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step
                )
                hiddens.append(dec_last_h)
                dec_last_w = inputs[name][:, t].view(-1, 1)

            dec_hs = torch.cat(hiddens, dim=0).transpose(0, 1)  # [1,B,H] ---> [B,T,H]
            probs[name] = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn)
            if name != "resp":
                hidden_states[name] = dec_hs
            return hidden_states, probs

        if cfg.enable_bspn:
            bspn_enc, _ = self.span_encoder(inputs["pv_" + cfg.bspn_mode])
            hidden_states[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == "bspn" else usdx_enc_last_h
            hidden_states, probs = train_decode(cfg.bspn_mode, init_hidden, hidden_states, probs)

        if cfg.enable_aspn:
            aspn_enc, _ = self.span_encoder(inputs["pv_aspn"])
            hidden_states["aspn"] = aspn_enc
            hidden_states, probs = train_decode("aspn", usdx_enc_last_h, hidden_states, probs)

        hidden_states, probs = train_decode("resp", usdx_enc_last_h, hidden_states, probs)

        return probs


def _tiny_inputs(batch_size=2, T=3, Tdec=2, vocab_size=48, vocab_size_oov=52, hidden_size=24):
    torch.manual_seed(0)

    def toks(t=T):
        x = torch.randint(1, vocab_size, (batch_size, t))
        return x

    def onehot(t, vs_oov):
        width = vocab_size + t
        oh = torch.zeros(batch_size, t, width)
        idx = torch.randint(0, width, (batch_size, t, 1))
        oh.scatter_(2, idx, 1.0)
        return oh

    def nounk(t):
        return torch.randint(0, vocab_size + t, (batch_size, t))

    inputs = {}
    for key in ["user", "usdx", "pv_resp", "pv_bsdx", "aspn", "bsdx", "pv_aspn"]:
        inputs[key] = toks(T)
    for key in ["user_onehot"]:
        inputs[key] = onehot(T, vocab_size_oov)
    inputs["user_nounk"] = nounk(T)
    inputs["pv_resp_onehot"] = onehot(T, vocab_size_oov)
    inputs["pv_resp_nounk"] = nounk(T)
    inputs["pv_bsdx_onehot"] = onehot(T, vocab_size_oov)
    inputs["pv_bsdx_nounk"] = nounk(T)
    # bsdx/aspn hidden_states are overwritten with the *decoder's own* dec_hs (length Tdec)
    # right after each is decoded (train_decode: "if name != 'resp': hidden_states[name] = dec_hs"),
    # so downstream copy-attention against them must be sized Tdec, not the encoder length T.
    inputs["bsdx_onehot"] = onehot(Tdec, vocab_size_oov)
    inputs["bsdx_nounk"] = nounk(Tdec)
    inputs["aspn_onehot"] = onehot(Tdec, vocab_size_oov)
    inputs["aspn_nounk"] = nounk(Tdec)
    inputs["usdx_onehot"] = onehot(T, vocab_size_oov)
    inputs["usdx_nounk"] = nounk(T)
    inputs["db"] = torch.zeros(batch_size, 6)  # cfg.pointer_dim (fixed at 6 in the repo's config)
    # decode-length targets (train_decode iterates over inputs[name].size(1))
    inputs["bsdx"] = toks(Tdec)
    inputs["aspn"] = toks(Tdec)
    inputs["resp"] = toks(Tdec)
    return inputs


def build_damd_reader():
    return _TinyReader(vocab_size=48, vocab_size_oov=52)


class DamdWrapper(nn.Module):
    """Wraps DAMD's dict-in/dict-out forward into a single-tensor-friendly call so torchlens
    can trace it directly (DAMD.forward's real signature and computation are unchanged)."""

    def __init__(self):
        super().__init__()
        self.damd = DAMD(build_damd_reader())

    def forward(self, dummy):
        inputs = _tiny_inputs()
        hidden_states = {}
        probs = self.damd(inputs, hidden_states, first_turn=True, mode="train")
        return probs["resp"]


def build_damd():
    return DamdWrapper()


def example_input_damd():
    return torch.zeros(1)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DAMD", "build_damd", "example_input_damd", 2020, "vendored-pytorch"),
]
