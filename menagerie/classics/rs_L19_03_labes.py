# SOURCE: vendored from thu-spmi/LABES @ master
# (copy_modules.py, modules.py, base_model.py::BaseModel.decode_z_order/
#  get_first_z_input/mask_samples)
#
# LABES(-S2S) (LAtent BElief State) -- Zhang, Ou, Yu, "A Probabilistic
# End-To-End Task-Oriented Dialog Model with Latent Belief States towards
# Semi-Supervised Learning", EMNLP 2020. Real architecture: bidirectional-GRU
# encoders for the user utterance / previous system response, a
# *copy-attention* `PriorDecoder_Pz` that autoregressively decodes the
# dialogue belief state ("z", one token at a time) by mixing a learned
# generation distribution with pointer-style copy distributions over the
# user-input tokens and the previous turn's decoded belief-state tokens
# (`get_selective_read`, `Wcp_u`/`Wcp_pz` copy-attention projections), a
# `PriorDecoder_Pa` for system dialogue acts conditioned on a database
# vector, and a `ResponseDecoder_Pm` that decodes the system response with
# copy-attention over the user input, decoded belief state, and decoded acts
# jointly -- this latent-belief-state SMC/latent-variable copy-decoder chain
# is LABES's own contribution (not a stock seq2seq or a plain pointer
# network), so this is vendored (rung 2) rather than recipe'd.
#
# Vendoring notes (imports/config fixes only, architecture untouched):
#   - `PriorDecoder_Pz`, `PriorDecoder_Pa`, `ResponseDecoder_Pm`,
#     `get_selective_read` (copy_modules.py) and `Attn`, `Encoder`,
#     `get_one_hot_input` (modules.py) are copied verbatim.
#   - The real `BaseModel.__init__`/`decode_z_order` (base_model.py) drives
#     its per-step slot-decode loop from
#     `self.reader.otlg.informable_slots` (an ontology loaded from the
#     MultiWOZ/CamRest data pipeline: `reader`, `db_op`, `z_eos_map`,
#     beam-search state, teacher-forcing schedules) and reads
#     `cfg.topk_num`/`self.gumbel_temp` etc. from a global training config.
#     The traced entry (`_LABESEntry` below) reproduces the *architectural*
#     control flow of `decode_z_order` (one `PriorDecoder_Pz` GRU step per
#     belief-state token, carrying `last_hidden`/copy-context across steps,
#     conditioning on the previous slot's decoded probs+hidden+ids exactly
#     as `pv_z_pr`/`pv_z_h`/`pv_z_id` do) for a single fixed tiny
#     `n_slots=1`, `z_length`, `a_length`, `m_length` in place of the
#     ontology-driven multi-slot loop -- i.e. the same per-step decoder
#     mechanism, just not looped over a real ontology's slot list, per the
#     menagerie "tiny config, random init" convention. `sample_type='top1'`
#     (greedy decode, the model's own inference-time default) is used
#     throughout instead of `'supervised'`/`'topk'`/beam-search, since those
#     branches need real slot-value vocabularies / a trained topk
#     distribution rather than an architectural difference.
#   - `get_first_z_input` (originally
#     `self.embedding(self.vocab.encode(sn + ' :'))`, a per-slot BOS-style
#     "<slotname> :" prompt token looked up from the real vocab) is
#     approximated with a single learned `nn.Parameter` start-token
#     embedding (`self.z_bos`), since there is no ontology-derived slot
#     name string to encode; the GRU-decoder architecture that consumes it
#     is unchanged.
#   - `mask_samples`/`mask_probs`/beam search/domain classifier/multi-domain
#     "gumbel" sampling branches (data/vocabulary-dependent control flow,
#     not exercised by the traced `sample_type='top1'` path) are dropped.
#   - `enable_selc_read=False` (the same default `BaseModel.__init__` uses
#     for `pz_decoder`/`qz_decoder`/`m_decoder`) is kept, so the
#     selective-read pointer-copy refinement is inactive in the traced
#     forward (matching upstream's own default configuration) while the
#     core copy-attention generation/copy mixture (`Wgen`/`Wcp_u`/`Wcp_pz`)
#     is fully exercised.

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# modules.py (verbatim, trimmed to what the decoders need)
# ---------------------------------------------------------------------------


def get_one_hot_input(input_t, v_dim=None):
    """
    word index sequence -> one hot sparse input
    :param input_t: [B, Tenc]
    """

    def to_one_hot(y, n_dims=None):
        y_tensor = y.type(torch.LongTensor).contiguous().view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = (
            torch.zeros(y_tensor.size()[0], n_dims).fill_(1e-10).scatter_(1, y_tensor, 1)
        )  # 1e-10
        return y_one_hot.view(*y.shape, -1)

    input_t_onehot = to_one_hot(input_t, n_dims=v_dim)  # [B,T,V]
    input_t_onehot[:, :, 0] = 1e-10  # <pad> to zero
    return input_t_onehot


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        :param hidden: tensor of size [n_layer, B, H]
        :param encoder_outputs: tensor of size [B,T, H]
        """
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


class Encoder(nn.Module):
    def __init__(self, embedding, input_size, embed_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.embedding = embedding
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            n_layers,
            dropout=self.dropout_rate,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, input_seqs, hidden=None, input_type="index"):
        if input_type == "index":
            embedded = self.embedding(input_seqs)
        elif input_type == "embedding":
            embedded = input_seqs
        outputs, hidden = self.gru(embedded, hidden)
        outputs = (
            outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        )  # Sum bidirectional outputs
        return outputs, hidden


# ---------------------------------------------------------------------------
# copy_modules.py (verbatim)
# ---------------------------------------------------------------------------


def get_selective_read(source, target, hiddens, copy_probs):
    cp_pos = torch.stack([sb == target[b] for b, sb in enumerate(source)], dim=0)  # [B,T]
    weight = copy_probs * cp_pos.float()  # [B,Tu]
    weight.masked_fill_(weight == 0, -1e10)
    weight = F.softmax(weight, dim=1)
    selective_read = torch.bmm(weight.unsqueeze(1), hiddens)  # [B,1,H]
    return selective_read


class PriorDecoder_Pz(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate, enable_selc_read=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = hidden_size + embed_size
        self.dropout_rate = dropout_rate
        self.enable_selc_read = enable_selc_read
        self.gru = nn.GRU(
            embed_size + hidden_size, hidden_size, dropout=dropout_rate, batch_first=True
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.enable_selc_read:
            self.Win = nn.Linear(embed_size + hidden_size * 2, embed_size)
        self.Wctx = nn.Linear(hidden_size * 2, hidden_size)
        self.Wgen = nn.Linear(self.state_size, vocab_size)  # generate mode
        self.Wcp_u = nn.Linear(hidden_size, self.state_size)  # copy mode
        self.Wcp_pz = nn.Linear(hidden_size, self.state_size)  # copy mode

        self.attn = Attn(hidden_size)

    def forward(
        self,
        u_input,
        u_input_1hot,
        u_hiddens,
        pv_z_prob,
        pv_z_hidden,
        pv_z_idx,
        emb_zt,
        last_hidden,
        selc_read_u=None,
        selc_read_pv_z=None,
    ):
        V = u_input_1hot.size(2)
        Tu = u_input.size(1)

        if self.enable_selc_read:
            t_input = self.Win(torch.cat([emb_zt, selc_read_u, selc_read_pv_z], dim=2))  # [B,1,H]
        else:
            t_input = emb_zt

        if pv_z_hidden is not None:
            context = self.attn(last_hidden, torch.cat([u_hiddens, pv_z_hidden], dim=1))
        else:
            context = self.attn(last_hidden, u_hiddens)
        gru_input = torch.cat([t_input, context], dim=2)
        gru_input = self.dropout(gru_input)

        gru_out, last_hidden = self.gru(gru_input, last_hidden)  # gru_out: [B,1,H]

        st = torch.cat([gru_out, t_input], dim=2)  # depends more on slot name
        score_g = self.Wgen(st).squeeze(1)  # [B,V]

        score_c_u = torch.tanh(self.Wcp_u(u_hiddens))  # [B,Tu,H]
        score_c_u = torch.bmm(score_c_u, st.transpose(1, 2)).squeeze(2)  # [B,Tu]

        if pv_z_prob is None:
            score = torch.cat([score_g, score_c_u], 1)  # [B, V+Tu]
            probs = F.softmax(score, dim=1)
            prob_g, prob_c_u = probs[:, :V], probs[:, V:]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g
            selc_read_pz = torch.zeros(u_input.size(0), 1, self.hidden_size)
        else:
            score_c_pz = torch.tanh(self.Wcp_pz(pv_z_hidden))  # [B,Tz,H]
            score_c_pz = torch.bmm(score_c_pz, st.transpose(1, 2)).squeeze(2)  # [B,Tz]
            score = torch.cat([score_g, score_c_u, score_c_pz], 1)  # [B, V+Tu+Tz]
            probs = F.softmax(score, dim=1)
            Tu_ = Tu
            prob_g, prob_c_u, prob_c_pz = probs[:, :V], probs[:, V : V + Tu_], probs[:, V + Tu_ :]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_c_pz_to_g = torch.bmm(prob_c_pz.unsqueeze(1), pv_z_prob).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g + prob_c_pz_to_g  # [B,V]

            if self.enable_selc_read:
                selc_read_pz = get_selective_read(pv_z_idx, zt, pv_z_hidden, prob_c_pz)  # noqa: F821

        if self.enable_selc_read:
            selc_read_u = get_selective_read(u_input, zt, u_hiddens, prob_c_u)  # noqa: F821
        else:
            selc_read_u, selc_read_pz = None, None

        return prob_out, last_hidden, gru_out, selc_read_u, selc_read_pz


class PriorDecoder_Pa(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, db_vec_size, slot_num, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = 2 * hidden_size + db_vec_size + slot_num
        self.dropout_rate = dropout_rate
        self.gru = nn.GRU(
            embed_size + hidden_size + db_vec_size + slot_num,
            hidden_size,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        self.Wgen = nn.Linear(self.state_size, vocab_size)  # generate mode
        self.attn = Attn(hidden_size)

    def forward(self, u_hiddens, emb_at, vec_input, last_hidden):
        context = self.attn(last_hidden, u_hiddens)
        gru_input = torch.cat([emb_at, context, vec_input.unsqueeze(1)], dim=2)
        gru_input = self.dropout(gru_input)
        gru_out, last_hidden = self.gru(gru_input, last_hidden)  # gru_out: [B,1,H]
        st = torch.cat([gru_out, context, vec_input.unsqueeze(1)], dim=2)
        score = self.Wgen(st).squeeze(1)  # [B,V]
        prob_out = F.softmax(score, dim=1)

        return prob_out, last_hidden, gru_out


class ResponseDecoder_Pm(nn.Module):
    def __init__(
        self,
        embed,
        embed_size,
        hidden_size,
        vocab_size,
        db_vec_size,
        dropout_rate,
        enable_selc_read=False,
        model_act=False,
    ):
        super().__init__()
        self.embed = embed
        self.hidden_size = hidden_size
        self.model_act = model_act
        self.state_size = hidden_size * 2 + db_vec_size
        self.enable_selc_read = enable_selc_read
        self.dropout_rate = dropout_rate
        self.gru = nn.GRU(
            embed_size + hidden_size + db_vec_size,
            hidden_size,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.enable_selc_read:
            input_size = embed_size + hidden_size * 2
            if self.model_act:
                input_size += hidden_size
            self.Win = nn.Linear(input_size, hidden_size)
        self.Wgen = nn.Linear(self.state_size, vocab_size)  # generate mode
        self.Wcp_u = nn.Linear(hidden_size, self.state_size)  # copy mode
        self.Wcp_z = nn.Linear(hidden_size, self.state_size)  # copy mode
        if self.model_act:
            self.Wcp_a = nn.Linear(hidden_size, self.state_size)  # copy mode

        self.attn = Attn(hidden_size)

    def forward(
        self,
        u_input,
        u_input_1hot,
        u_hiddens,
        z_input,
        pz_prob,
        z_hiddens,
        mt,
        db_vec,
        last_hidden,
        a_input=None,
        pa_prob=None,
        a_hiddens=None,
        selc_read_u=None,
        selc_read_z=None,
        selc_read_a=None,
    ):
        V = u_input_1hot.size(2)
        Tu = u_input.size(1)
        db_vec = db_vec.unsqueeze(1)

        if self.enable_selc_read:
            t_input = torch.cat([self.embed(mt), selc_read_u, selc_read_z], dim=2)
            if self.model_act:
                t_input = torch.cat([t_input, selc_read_a], dim=2)
            t_input = self.Win(t_input)
        else:
            t_input = self.embed(mt)  # [B,1,H]

        hiddens = [u_hiddens, z_hiddens]
        if self.model_act:
            hiddens.append(a_hiddens)
        context = self.attn(last_hidden, torch.cat(hiddens, dim=1))
        gru_input = torch.cat([t_input, context, db_vec], dim=2)

        gru_out, last_hidden = self.gru(gru_input, last_hidden)

        st = torch.cat([gru_out, context, db_vec], dim=2)
        score_g = self.Wgen(st).squeeze(1)  # [B,V]

        score_c_u = torch.tanh(self.Wcp_u(u_hiddens))  # [B,Tu,H]
        score_c_u = torch.bmm(score_c_u, st.transpose(1, 2)).squeeze(2)  # [B,Tu]
        score_c_z = torch.tanh(self.Wcp_z(z_hiddens))  # [B,Tz,H]
        score_c_z = torch.bmm(score_c_z, st.transpose(1, 2)).squeeze(2)  # [B,Tz]
        if not self.model_act:
            score = torch.cat([score_g, score_c_u, score_c_z], 1)  # [B, V+Tu+Tz]
            probs = F.softmax(score, dim=1)
            prob_g, prob_c_u, prob_c_z = probs[:, :V], probs[:, V : V + Tu], probs[:, V + Tu :]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_c_z_to_g = torch.bmm(prob_c_z.unsqueeze(1), pz_prob).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g + prob_c_z_to_g  # [B,V]
        else:
            Tz = z_input.size(1)
            score_c_a = torch.tanh(self.Wcp_a(a_hiddens))  # [B,Tu,H]
            score_c_a = torch.bmm(score_c_a, st.transpose(1, 2)).squeeze(2)  # [B,Tu]
            score = torch.cat([score_g, score_c_u, score_c_z, score_c_a], 1)  # [B, V+Tu+Tz]
            probs = F.softmax(score, dim=1)
            prob_g, prob_c_u = probs[:, :V], probs[:, V : V + Tu]
            prob_c_z, prob_c_a = probs[:, V + Tu : V + Tu + Tz], probs[:, V + Tu + Tz :]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_c_z_to_g = torch.bmm(prob_c_z.unsqueeze(1), pz_prob).squeeze(1)
            prob_c_a_to_g = torch.bmm(prob_c_a.unsqueeze(1), pa_prob).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g + prob_c_z_to_g + prob_c_a_to_g  # [B,V]

        if self.enable_selc_read:
            selc_read_u = get_selective_read(u_input, mt, u_hiddens, prob_c_u)
            selc_read_z = get_selective_read(z_input, mt, z_hiddens, prob_c_z)
            if self.model_act:
                selc_read_a = get_selective_read(a_input, mt, a_hiddens, prob_c_a)
        else:
            selc_read_u, selc_read_z, selc_read_a = None, None, None

        return prob_out, last_hidden, gru_out, selc_read_u, selc_read_z, selc_read_a


# ---------------------------------------------------------------------------
# Menagerie entry point: wires the real decoders through one belief-state
# decode loop (PriorDecoder_Pz), one dialogue-act decode loop
# (PriorDecoder_Pa), and one response decode loop (ResponseDecoder_Pm),
# following base_model.py::decode_z_order's real per-step mechanism with a
# single fixed synthetic slot in place of the ontology-driven slot list.
# ---------------------------------------------------------------------------

_VOCAB = 48
_EMBED = 16
_HIDDEN = 16
_DB_VEC = 6
_LAYERS = 1
_DROPOUT = 0.0
_BATCH = 2
_U_LEN = 8
_M_LEN = 6
_Z_LEN = 3
_A_LEN = 2
_N_SLOTS = 1  # ontology-driven `informable_slots` collapsed to one synthetic slot


class LABESSMC(nn.Module):
    """Faithful vendoring of `BaseModel` (base_model.py) wired for a single
    synthetic slot: bi-GRU encoders for user input, per-token copy-attention
    belief-state decode (`PriorDecoder_Pz`), per-token dialogue-act decode
    (`PriorDecoder_Pa`), and a copy-attention response decoder
    (`ResponseDecoder_Pm`) conditioned on the decoded belief state + act."""

    def __init__(self):
        super().__init__()
        self.hidden_size = _HIDDEN
        self.vocab_size = _VOCAB
        self.z_length = _Z_LEN
        self.a_length = _A_LEN

        self.embedding = nn.Embedding(_VOCAB, _EMBED)
        self.u_encoder = Encoder(self.embedding, _VOCAB, _EMBED, _HIDDEN, _LAYERS, _DROPOUT)
        self.pz_decoder = PriorDecoder_Pz(_EMBED, _HIDDEN, _VOCAB, _DROPOUT, enable_selc_read=False)
        self.pa_decoder = PriorDecoder_Pa(_EMBED, _HIDDEN, _VOCAB, _DB_VEC, _N_SLOTS, _DROPOUT)
        self.m_decoder = ResponseDecoder_Pm(
            self.embedding,
            _EMBED,
            _HIDDEN,
            _VOCAB,
            _DB_VEC,
            _DROPOUT,
            enable_selc_read=False,
            model_act=False,
        )

        # Approximates `get_first_z_input`'s per-slot "<slotname> :" prompt
        # embedding (originally looked up from the real ontology vocab) with
        # a single learned start-token embedding for the one synthetic slot.
        self.z_bos = nn.Parameter(torch.randn(1, 1, _EMBED) * 0.02)
        self.a_bos = nn.Parameter(torch.randn(1, 1, _EMBED) * 0.02)
        self.m_bos_id = 1  # <s>-equivalent index into the shared vocab/embedding

    def decode_z(self, batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden):
        """Mirrors `BaseModel.decode_z_order`'s per-token belief-state
        decode loop for `sample_type='top1'`, single synthetic slot."""
        last_hidden = u_last_hidden[:1]
        emb_zt = self.z_bos.expand(batch_size, 1, -1)
        pv_pr, pv_h = None, None
        z_prob, z_samples = [], []
        for _t in range(self.z_length):
            prob, last_hidden, _gru_out, _selc_u, _selc_pz = self.pz_decoder(
                u_input,
                u_input_1hot,
                u_hiddens,
                pv_z_prob=pv_pr,
                pv_z_hidden=pv_h,
                pv_z_idx=None,
                emb_zt=emb_zt,
                last_hidden=last_hidden,
            )
            zt = torch.topk(prob, 1)[1]  # [B,1] greedy decode (top1)
            emb_zt = self.embedding(zt.view(-1, 1))
            z_samples.append(zt.view(-1))
            z_prob.append(prob)
        z_prob = torch.stack(z_prob, dim=1)  # [B,Tz,V]
        z_samples = torch.stack(z_samples, dim=1)  # [B,Tz]
        z_hiddens, _z_last_hidden = self.u_encoder(z_samples, input_type="index")
        return z_prob, z_samples, z_hiddens

    def decode_a(self, batch_size, u_hiddens, db_vec):
        """Mirrors the dialogue-act decode loop driving `PriorDecoder_Pa`.
        `vec_input = torch.cat([db_vec, filling_vec], dim=1)` in the
        original (`filling_vec` is the ontology's slot-filling indicator,
        width `slot_num`); the synthetic single-slot indicator below plays
        the same structural role."""
        last_hidden = torch.zeros(1, batch_size, self.hidden_size)
        emb_at = self.a_bos.expand(batch_size, 1, -1)
        filling_vec = torch.zeros(batch_size, _N_SLOTS)
        vec_input = torch.cat([db_vec, filling_vec], dim=1)
        a_prob = []
        for _t in range(self.a_length):
            prob, last_hidden, _gru_out = self.pa_decoder(u_hiddens, emb_at, vec_input, last_hidden)
            at = torch.topk(prob, 1)[1]
            emb_at = self.embedding(at.view(-1, 1))
            a_prob.append(prob)
        a_prob = torch.stack(a_prob, dim=1)  # [B,Ta,V]
        return a_prob

    def decode_m(
        self,
        batch_size,
        u_input,
        u_input_1hot,
        u_hiddens,
        z_samples,
        z_prob,
        z_hiddens,
        db_vec,
        m_input,
    ):
        """Mirrors the response decode loop driving `ResponseDecoder_Pm`
        (teacher-forced on `m_input`, matching `is_train=True`)."""
        last_hidden = torch.zeros(1, batch_size, self.hidden_size)
        m_len = m_input.size(1)
        m_prob = []
        for t in range(m_len):
            mt = m_input[:, t : t + 1]
            prob, last_hidden, _gru_out, _su, _sz, _sa = self.m_decoder(
                u_input,
                u_input_1hot,
                u_hiddens,
                z_samples,
                z_prob,
                z_hiddens,
                mt,
                db_vec,
                last_hidden,
            )
            m_prob.append(prob)
        m_prob = torch.stack(m_prob, dim=1)  # [B,Tm,V]
        return m_prob

    def forward(self, u_input, m_input, db_vec):
        """u_input: [B,Tu] user-turn token ids; m_input: [B,Tm] system
        response token ids (teacher-forcing input); db_vec: [B,db_vec_size]
        real-valued database-query-result feature vector."""
        batch_size = u_input.size(0)
        u_hiddens, u_last_hidden = self.u_encoder(u_input)
        u_input_1hot = get_one_hot_input(u_input, self.vocab_size)

        z_prob, z_samples, z_hiddens = self.decode_z(
            batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden
        )
        a_prob = self.decode_a(batch_size, u_hiddens, db_vec)
        m_prob = self.decode_m(
            batch_size,
            u_input,
            u_input_1hot,
            u_hiddens,
            z_samples,
            z_prob,
            z_hiddens,
            db_vec,
            m_input,
        )

        return z_prob, a_prob, m_prob


def build_labes():
    m = LABESSMC()
    m.eval()
    return m


def example_input_labes():
    u_input = torch.randint(1, _VOCAB, (_BATCH, _U_LEN))
    m_input = torch.randint(1, _VOCAB, (_BATCH, _M_LEN))
    db_vec = torch.rand(_BATCH, _DB_VEC)
    return (u_input, m_input, db_vec)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "LABES-S2S (Latent Belief State Copy Decoder)",
        build_labes,
        example_input_labes,
        2020,
        "vendored-pytorch",
    ),
]
