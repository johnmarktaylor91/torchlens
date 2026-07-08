# SOURCE: vendored from WING-NUS/sequicity @ master
#
# https://github.com/WING-NUS/sequicity
# https://raw.githubusercontent.com/WING-NUS/sequicity/master/tsd_net.py
# https://raw.githubusercontent.com/WING-NUS/sequicity/master/config.py
# https://raw.githubusercontent.com/WING-NUS/sequicity/master/model.py
#
# Lei et al. 2018 "Sequicity: Simplifying Task-oriented Dialogue Systems with
# Single Sequence-to-Sequence Architectures" (ACL 2018). The `TSD` model
# ("Two Stage CopyNet": a bidirectional-GRU utterance encoder feeding a
# belief-span (`z`) CopyNet decoder, whose output in turn feeds a response
# (`m`) CopyNet decoder gated by DB-degree features) is copied verbatim from
# `tsd_net.py`, along with its submodules `Attn`, `SimpleDynamicEncoder`,
# `BSpanDecoder`, `ResponseDecoder`, and the helper functions `cuda_`, `toss_`,
# `get_sparse_input_aug`, `init_gru`. The dual sparse-copy-augmented-softmax
# mechanism (`u_copy_score`/`pv_z_copy_score`/`z_copy_score` computed via
# `get_sparse_input_aug`/`get_sparse_selective_input` + matmul against decoder
# hidden states) and the CopyNet-style GRU decoding loops are unchanged.
#
# Minimal, non-architectural adaptations for base-env CPU tracing:
#   - `from config import global_config as cfg` -> a self-contained `_Config`
#     stub below, populated with the same field values `config.py`'s
#     `_camrest_tsdf_init()` uses (scaled down: `vocab_size`/`hidden_size`/
#     `embedding_size` shrunk for a fast trace; `cuda = False` so `cuda_()`
#     is a no-op) -- `cfg` is a plain settings object in the original repo,
#     not part of the TSD architecture.
#   - `TSD.forward(mode='train', ...)` is the call path exercised here (the
#     real end-to-end training forward pass: encode -> z-decode -> supervised
#     loss over z and m). The `mode='test'`/`'rl'` branches additionally
#     require `self.reader.db_degree_handler` (a dataset-specific SQL/JSON
#     DB lookup unrelated to the neural architecture) and are not exercised;
#     `self.reader` is passed as `None` since `forward_turn(mode='train')`
#     never dereferences it. `self.vocab` is a minimal `_TinyVocab` stub
#     supplying the `.encode()` lookups (`EOS_Z2`, `EOS_M`, `<unk>`) that the
#     real `reader.Vocab.encode` provides -- see `reader.py`'s `Vocab` class,
#     which this mirrors exactly (same special-token ids 0-3).

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class _Config:
    """Minimal stand-in for config.py's global_config, values matching
    _camrest_tsdf_init() (scaled down for a fast trace)."""

    def __init__(self):
        self.cuda = False
        self.eos_m_token = "EOS_M"
        self.vocab_size = 40
        self.embedding_size = 16
        self.hidden_size = 16
        self.degree_size = 5
        self.layer_num = 1
        self.dropout_rate = 0.0
        self.z_length = 4
        self.max_ts = 8
        self.teacher_force = 100
        self.beam_search = False
        self.use_positional_embedding = False
        self.dataset = "camrest"


cfg = _Config()


def cuda_(var):
    return var.cuda() if cfg.cuda else var


def toss_(p):
    return random.randint(0, 99) <= p


def nan(v):
    if type(v) is float:
        return v == float("nan")
    return np.isnan(np.sum(v.data.cpu().numpy()))


def get_sparse_input_aug(x_input_np):
    """
    sparse input of
    :param x_input_np: [T,B]
    :return: Numpy array: [B,T,aug_V]
    """
    ignore_index = [0]
    unk = 2
    result = np.zeros(
        (x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]),
        dtype=np.float32,
    )
    result.fill(1e-10)
    for t in range(x_input_np.shape[0]):
        for b in range(x_input_np.shape[1]):
            w = x_input_np[t][b]
            if w not in ignore_index:
                if w != unk:
                    result[t][b][x_input_np[t][b]] = 1.0
                else:
                    result[t][b][cfg.vocab_size + t] = 1.0
    result_np = result.transpose((1, 0, 2))
    result = torch.from_numpy(result_np).float()
    return result


def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i : i + gru.hidden_size], gain=1)


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(
        self, hidden, encoder_outputs, mask=False, inp_seqs=None, stop_tok=None, normalize=True
    ):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(hidden, encoder_outputs)
        if True or not mask:
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        else:
            mask_idx = []
            for b in range(inp_seqs.shape[1]):
                for t in range(inp_seqs.shape[0] + 1):
                    if t == inp_seqs.shape[0] or inp_seqs[t, b] in stop_tok:
                        mask_idx.append(t)
                        break
            mask = []
            for mask_len in mask_idx:
                mask.append([1.0] * mask_len + [0.0] * (inp_seqs.shape[0] - mask_len))
            mask = cuda_(Variable(torch.FloatTensor(mask)))  # [B,T]
            attn_energies = attn_energies * mask.unsqueeze(1)
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]

        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context.transpose(0, 1)  # [1,B,H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = F.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy


class SimpleDynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(
            embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True
        )
        init_gru(self.gru)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)  # noqa: F841 (verbatim upstream; kept unused for fidelity)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden, embedded


class BSpanDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        if cfg.use_positional_embedding:
            self.positional_embedding = nn.Embedding(cfg.max_ts + 1, embed_size)
            init_pos_emb = self.position_encoding_init(cfg.max_ts + 1, embed_size)
            self.positional_embedding.weight.data = init_pos_emb
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, dropout=dropout_rate)
        self.proj = nn.Linear(hidden_size * 2, vocab_size)

        self.attn_u = Attn(hidden_size)
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate

        self.inp_dropout = nn.Dropout(self.dropout_rate)

        init_gru(self.gru)
        self.vocab = vocab

    def position_encoding_init(self, n_position, d_pos_vec):
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
                if pos != 0
                else np.zeros(d_pos_vec)
                for pos in range(n_position)
            ]
        )

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(
        self,
        u_enc_out,
        z_tm1,
        last_hidden,
        u_input_np,
        pv_z_enc_out,
        prev_z_input_np,
        u_emb,
        pv_z_emb,
        position,
    ):
        sparse_u_input = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)

        if pv_z_enc_out is not None:
            context = self.attn_u(
                last_hidden,
                torch.cat([pv_z_enc_out, u_enc_out], dim=0),
                mask=True,
                inp_seqs=np.concatenate([prev_z_input_np, u_input_np], 0),
                stop_tok=[self.vocab.encode("EOS_M")],
            )
        else:
            context = self.attn_u(
                last_hidden,
                u_enc_out,
                mask=True,
                inp_seqs=u_input_np,
                stop_tok=[self.vocab.encode("EOS_M")],
            )
        embed_z = self.emb(z_tm1)

        if cfg.use_positional_embedding:  # defaulty not used
            position_label = [position] * u_enc_out.size(1)  # [B]
            position_label = cuda_(Variable(torch.LongTensor(position_label))).view(1, -1)  # [1,B]
            pos_emb = self.positional_embedding(position_label)
            embed_z = embed_z + pos_emb

        gru_in = torch.cat([embed_z, context], 2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)
        gen_score = self.proj(torch.cat([gru_out, context], 2)).squeeze(0)
        u_copy_score = F.tanh(self.proj_copy1(u_enc_out.transpose(0, 1)))  # [B,T,H]
        u_copy_score = torch.matmul(u_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
        u_copy_score = u_copy_score.cpu()
        u_copy_score_max = torch.max(u_copy_score, dim=1, keepdim=True)[0]
        u_copy_score = torch.exp(u_copy_score - u_copy_score_max)  # [B,T]
        u_copy_score = (
            torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(1)
            + u_copy_score_max
        )  # [B,V]
        u_copy_score = cuda_(u_copy_score)
        if pv_z_enc_out is None:
            scores = F.softmax(torch.cat([gen_score, u_copy_score], dim=1), dim=1)
            gen_score, u_copy_score = scores[:, : cfg.vocab_size], scores[:, cfg.vocab_size :]
            proba = gen_score + u_copy_score[:, : cfg.vocab_size]  # [B,V]
            proba = torch.cat([proba, u_copy_score[:, cfg.vocab_size :]], 1)
        else:
            sparse_pv_z_input = Variable(get_sparse_input_aug(prev_z_input_np), requires_grad=False)
            pv_z_copy_score = F.tanh(self.proj_copy2(pv_z_enc_out.transpose(0, 1)))  # [B,T,H]
            pv_z_copy_score = torch.matmul(
                pv_z_copy_score, gru_out.squeeze(0).unsqueeze(2)
            ).squeeze(2)
            pv_z_copy_score = pv_z_copy_score.cpu()
            pv_z_copy_score_max = torch.max(pv_z_copy_score, dim=1, keepdim=True)[0]
            pv_z_copy_score = torch.exp(pv_z_copy_score - pv_z_copy_score_max)  # [B,T]
            pv_z_copy_score = (
                torch.log(torch.bmm(pv_z_copy_score.unsqueeze(1), sparse_pv_z_input)).squeeze(1)
                + pv_z_copy_score_max
            )  # [B,V]
            pv_z_copy_score = cuda_(pv_z_copy_score)
            scores = F.softmax(torch.cat([gen_score, u_copy_score, pv_z_copy_score], dim=1), dim=1)
            gen_score, u_copy_score, pv_z_copy_score = (
                scores[:, : cfg.vocab_size],
                scores[:, cfg.vocab_size : 2 * cfg.vocab_size + u_input_np.shape[0]],
                scores[:, 2 * cfg.vocab_size + u_input_np.shape[0] :],
            )
            proba = (
                gen_score + u_copy_score[:, : cfg.vocab_size] + pv_z_copy_score[:, : cfg.vocab_size]
            )  # [B,V]
            proba = torch.cat(
                [proba, pv_z_copy_score[:, cfg.vocab_size :], u_copy_score[:, cfg.vocab_size :]], 1
            )
        return gru_out, last_hidden, proba


class ResponseDecoder(nn.Module):
    def __init__(
        self, embed_size, hidden_size, vocab_size, degree_size, dropout_rate, gru, proj, emb, vocab
    ):
        super().__init__()
        self.emb = emb
        self.attn_z = Attn(hidden_size)
        self.attn_u = Attn(hidden_size)
        self.gru = gru
        init_gru(self.gru)
        self.proj = proj
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate

        self.vocab = vocab

    def get_sparse_selective_input(self, x_input_np):
        result = np.zeros(
            (x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]),
            dtype=np.float32,
        )
        result.fill(1e-10)
        reqs = ["address", "phone", "postcode", "pricerange", "area"]
        for t in range(x_input_np.shape[0] - 1):
            for b in range(x_input_np.shape[1]):
                w = x_input_np[t][b]
                word = self.vocab.decode(w)
                if word in reqs:
                    slot = self.vocab.encode(word + "_SLOT")
                    result[t + 1][b][slot] = 1.0
                else:
                    if w == 2 or w >= cfg.vocab_size:
                        result[t + 1][b][cfg.vocab_size + t] = 5.0
                    else:
                        result[t + 1][b][w] = 1.0
        result_np = result.transpose((1, 0, 2))
        result = torch.from_numpy(result_np).float()
        return result

    def forward(
        self, z_enc_out, u_enc_out, u_input_np, m_t_input, degree_input, last_hidden, z_input_np
    ):
        sparse_z_input = Variable(self.get_sparse_selective_input(z_input_np), requires_grad=False)

        m_embed = self.emb(m_t_input)
        z_context = self.attn_z(
            last_hidden,
            z_enc_out,
            mask=True,
            stop_tok=[self.vocab.encode("EOS_Z2")],
            inp_seqs=z_input_np,
        )
        u_context = self.attn_u(
            last_hidden,
            u_enc_out,
            mask=True,
            stop_tok=[self.vocab.encode("EOS_M")],
            inp_seqs=u_input_np,
        )
        gru_in = torch.cat([m_embed, u_context, z_context, degree_input.unsqueeze(0)], dim=2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)
        gen_score = self.proj(torch.cat([z_context, u_context, gru_out], 2)).squeeze(0)
        z_copy_score = F.tanh(self.proj_copy2(z_enc_out.transpose(0, 1)))
        z_copy_score = torch.matmul(z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
        z_copy_score = z_copy_score.cpu()
        z_copy_score_max = torch.max(z_copy_score, dim=1, keepdim=True)[0]
        z_copy_score = torch.exp(z_copy_score - z_copy_score_max)  # [B,T]
        z_copy_score = (
            torch.log(torch.bmm(z_copy_score.unsqueeze(1), sparse_z_input)).squeeze(1)
            + z_copy_score_max
        )  # [B,V]
        z_copy_score = cuda_(z_copy_score)

        scores = F.softmax(torch.cat([gen_score, z_copy_score], dim=1), dim=1)
        gen_score, z_copy_score = scores[:, : cfg.vocab_size], scores[:, cfg.vocab_size :]
        proba = gen_score + z_copy_score[:, : cfg.vocab_size]  # [B,V]
        proba = torch.cat([proba, z_copy_score[:, cfg.vocab_size :]], 1)
        return proba, last_hidden, gru_out


class TSD(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        degree_size,
        layer_num,
        dropout_rate,
        z_length,
        max_ts,
        beam_search=False,
        teacher_force=100,
        **kwargs,
    ):
        super().__init__()
        self.vocab = kwargs["vocab"]
        self.reader = kwargs["reader"]
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.dec_gru = nn.GRU(
            degree_size + embed_size + hidden_size * 2, hidden_size, dropout=dropout_rate
        )
        self.proj = nn.Linear(hidden_size * 3, vocab_size)
        self.u_encoder = SimpleDynamicEncoder(
            vocab_size, embed_size, hidden_size, layer_num, dropout_rate
        )
        self.z_decoder = BSpanDecoder(embed_size, hidden_size, vocab_size, dropout_rate, self.vocab)
        self.m_decoder = ResponseDecoder(
            embed_size,
            hidden_size,
            vocab_size,
            degree_size,
            dropout_rate,
            self.dec_gru,
            self.proj,
            self.emb,
            self.vocab,
        )
        self.embed_size = embed_size

        self.z_length = z_length
        self.max_ts = max_ts
        self.beam_search = beam_search
        self.teacher_force = teacher_force

        self.pr_loss = nn.NLLLoss(ignore_index=0)
        self.dec_loss = nn.NLLLoss(ignore_index=0)

        self.saved_log_policy = []

        if self.beam_search:
            self.beam_size = kwargs["beam_size"]
            self.eos_token_idx = kwargs["eos_token_idx"]

    def forward(
        self,
        u_input,
        u_input_np,
        m_input,
        m_input_np,
        z_input,
        u_len,
        m_len,
        turn_states,
        degree_input,
        mode,
        **kwargs,
    ):
        if mode == "train" or mode == "valid":
            pz_proba, pm_dec_proba, turn_states = self.forward_turn(
                u_input,
                u_len,
                m_input=m_input,
                m_len=m_len,
                z_input=z_input,
                mode="train",
                turn_states=turn_states,
                degree_input=degree_input,
                u_input_np=u_input_np,
                m_input_np=m_input_np,
                **kwargs,
            )
            loss, pr_loss, m_loss = self.supervised_loss(
                torch.log(pz_proba), torch.log(pm_dec_proba), z_input, m_input
            )
            return loss, pr_loss, m_loss, turn_states

        elif mode == "test":
            m_output_index, pz_index, turn_states = self.forward_turn(
                u_input,
                u_len=u_len,
                mode="test",
                turn_states=turn_states,
                degree_input=degree_input,
                u_input_np=u_input_np,
                m_input_np=m_input_np,
                **kwargs,
            )
            return m_output_index, pz_index, turn_states
        elif mode == "rl":
            loss = self.forward_turn(
                u_input,
                u_len=u_len,
                is_train=False,
                mode="rl",
                turn_states=turn_states,
                degree_input=degree_input,
                u_input_np=u_input_np,
                m_input_np=m_input_np,
                **kwargs,
            )
            return loss

    def forward_turn(
        self,
        u_input,
        u_len,
        turn_states,
        mode,
        degree_input,
        u_input_np,
        m_input_np=None,
        m_input=None,
        m_len=None,
        z_input=None,
        **kwargs,
    ):
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        :param u_input_np:
        :param m_input_np:
        :param u_len:
        :param turn_states:
        :param is_train:
        :param u_input: [T,B]
        :param m_input: [T,B]
        :param z_input: [T,B]
        :return:
        """
        prev_z_input = kwargs.get("prev_z_input", None)
        prev_z_input_np = kwargs.get("prev_z_input_np", None)
        prev_z_len = kwargs.get("prev_z_len", None)
        pv_z_emb = None
        batch_size = u_input.size(1)
        pv_z_enc_out = None

        if prev_z_input is not None:
            pv_z_enc_out, _, pv_z_emb = self.u_encoder(prev_z_input, prev_z_len)
        u_enc_out, u_enc_hidden, u_emb = self.u_encoder(u_input, u_len)
        last_hidden = u_enc_hidden[:-1]
        z_tm1 = cuda_(Variable(torch.ones(1, batch_size).long() * 3))  # GO_2 token
        m_tm1 = cuda_(Variable(torch.ones(1, batch_size).long()))  # GO token
        if mode == "train":
            pz_dec_outs = []
            pz_proba = []
            z_length = z_input.size(0) if z_input is not None else self.z_length  # GO token
            hiddens = [None] * batch_size
            for t in range(z_length):
                pz_dec_out, last_hidden, proba = self.z_decoder(
                    u_enc_out=u_enc_out,
                    u_input_np=u_input_np,
                    z_tm1=z_tm1,
                    last_hidden=last_hidden,
                    pv_z_enc_out=pv_z_enc_out,
                    prev_z_input_np=prev_z_input_np,
                    u_emb=u_emb,
                    pv_z_emb=pv_z_emb,
                    position=t,
                )
                pz_proba.append(proba)
                pz_dec_outs.append(pz_dec_out)
                z_np = z_tm1.view(-1).cpu().data.numpy()
                for i in range(batch_size):
                    if z_np[i] == self.vocab.encode("EOS_Z2"):
                        hiddens[i] = last_hidden[:, i, :]
                z_tm1 = z_input[t].view(1, -1)
            for i in range(batch_size):
                if hiddens[i] is None:
                    hiddens[i] = last_hidden[:, i, :]
            last_hidden = torch.stack(hiddens, dim=1)

            z_input_np = z_input.cpu().data.numpy()

            pz_dec_outs = torch.cat(pz_dec_outs, dim=0)  # [Tz,B,H]
            pz_proba = torch.stack(pz_proba, dim=0)
            # P(m|z,u)
            pm_dec_proba, m_dec_outs = [], []
            m_length = m_input.size(0)  # Tm
            for t in range(m_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden, dec_out = self.m_decoder(
                    pz_dec_outs, u_enc_out, u_input_np, m_tm1, degree_input, last_hidden, z_input_np
                )
                if teacher_forcing:
                    m_tm1 = m_input[t].view(1, -1)
                else:
                    _, m_tm1 = torch.topk(proba, 1)
                    m_tm1 = m_tm1.view(1, -1)
                pm_dec_proba.append(proba)
                m_dec_outs.append(dec_out)

            pm_dec_proba = torch.stack(pm_dec_proba, dim=0)  # [T,B,V]
            return pz_proba, pm_dec_proba, None

    def supervised_loss(self, pz_proba, pm_dec_proba, z_input, m_input):
        pz_proba, pm_dec_proba = (
            pz_proba[:, :, : cfg.vocab_size].contiguous(),
            pm_dec_proba[:, :, : cfg.vocab_size].contiguous(),
        )
        pr_loss = self.pr_loss(pz_proba.view(-1, pz_proba.size(2)), z_input.view(-1))
        m_loss = self.dec_loss(pm_dec_proba.view(-1, pm_dec_proba.size(2)), m_input.view(-1))

        loss = pr_loss + m_loss
        return loss, pr_loss, m_loss


# --- minimal, non-architectural test harness (mirrors reader.Vocab's fixed
# special-token ids: <pad>=0, <go>=1, <unk>=2, <go2>=3) ---
class _TinyVocab:
    def __init__(self, vocab_size):
        words = ["<pad>", "<go>", "<unk>", "<go2>", "EOS_Z2", "EOS_M"] + [
            f"w{i}" for i in range(vocab_size - 6)
        ]
        self._item2idx = {w: i for i, w in enumerate(words)}
        self._idx2item = {i: w for i, w in enumerate(words)}

    def encode(self, item):
        return self._item2idx.get(item, self._item2idx["<unk>"])

    def decode(self, idx):
        return self._idx2item.get(idx, "ITEM_%d" % idx)


class SequicityTraceWrapper(nn.Module):
    """Thin tensor-in/tensor-out wrapper around the real TSD so it can be
    traced: TSD.forward's `mode` (str), `turn_states` (None/dict), and the
    `*_np` numpy-array views of the tensor inputs are fixed, non-tensor
    training-loop plumbing (see header note), not model inputs, so they are
    closed over / recomputed here rather than passed through forward().
    """

    def __init__(self, tsd):
        super().__init__()
        self.tsd = tsd

    def forward(self, u_input, m_input, z_input, degree_input):
        u_input_np = u_input.detach().cpu().numpy()
        m_input_np = m_input.detach().cpu().numpy()
        u_len = np.array([u_input.size(0)] * u_input.size(1))
        m_len = np.array([m_input.size(0)] * m_input.size(1))
        loss, pr_loss, m_loss, _turn_states = self.tsd(
            u_input,
            u_input_np,
            m_input,
            m_input_np,
            z_input,
            u_len,
            m_len,
            None,
            degree_input,
            "train",
        )
        return loss, pr_loss, m_loss


def build_sequicity():
    vocab = _TinyVocab(cfg.vocab_size)
    tsd = TSD(
        embed_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        degree_size=cfg.degree_size,
        layer_num=cfg.layer_num,
        dropout_rate=cfg.dropout_rate,
        z_length=cfg.z_length,
        max_ts=cfg.max_ts,
        beam_search=False,
        teacher_force=cfg.teacher_force,
        eos_token_idx=vocab.encode("EOS_M"),
        vocab=vocab,
        reader=None,
    )
    return SequicityTraceWrapper(tsd)


def example_input_sequicity():
    batch_size = 2
    u_len_t, z_len_t, m_len_t = 6, cfg.z_length, 5

    u_input = torch.randint(4, cfg.vocab_size, (u_len_t, batch_size)).long()
    z_input = torch.randint(4, cfg.vocab_size, (z_len_t, batch_size)).long()
    z_input[-1, :] = 4  # EOS_Z2
    m_input = torch.randint(4, cfg.vocab_size, (m_len_t, batch_size)).long()
    degree_input = torch.zeros(batch_size, cfg.degree_size)

    return (u_input, m_input, z_input, degree_input)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Sequicity (Two-Stage CopyNet dialogue)",
        "build_sequicity",
        "example_input_sequicity",
        2018,
        "vendored",
    ),
]
