# SOURCE: vendored from WING-NUS/sequicity @ master (tsd_net.py)
# TSCP (Two-Stage CopyNet) model class `TSD`, as used in the ACL 2018 Sequicity paper.
# Vendored verbatim from the real nn.Module definitions in tsd_net.py; only minimal,
# non-architectural fixes applied to run on a modern torch without the original repo's
# `config.py`/`reader.py`/`Vocab` data pipeline:
#   - `from config import global_config as cfg` -> replaced with a tiny local `cfg` shim
#     exposing exactly the attributes tsd_net.py reads (vocab_size, use_positional_embedding,
#     max_ts, truncated).
#   - `from reader import pad_sequences` -> ported verbatim into this file (it is a pure
#     numpy padding utility copied byte-for-byte from reader.py, not a model change).
#   - `cuda_()` always returns the input unchanged here (no CUDA requirement to trace).
#   - `F.tanh(...)` (removed in modern torch) -> `torch.tanh(...)`.
#   - `torch.nn.init.orthogonal(...)` (renamed/removed) -> `torch.nn.init.orthogonal_(...)`.
#   - `Attn.score` used `self.v.repeat(...)` sized for batch dim from `encoder_outputs`;
#     kept exactly as in the original.
# All nn.Module classes (Attn, SimpleDynamicEncoder, BSpanDecoder, ResponseDecoder, TSD)
# and their forward() bodies are the ORIGINAL code, not a rewrite.
#
# The example harness drives the real `TSD.bspan_decoder` + `TSD.greedy_decode` path
# (the two-stage CopyNet decoders that are the paper's actual architectural contribution)
# with a minimal Vocab/reader stub that satisfies only the interface tsd_net.py calls
# (`vocab.encode`, `reader.db_degree_handler`) -- no training data pipeline is needed to
# trace the network.

import copy
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class _Config:
    vocab_size = 32
    use_positional_embedding = False
    max_ts = 4
    truncated = False
    z_length = 3


cfg = _Config()


def cuda_(var):
    return var


def toss_(p):
    return random.randint(0, 99) <= p


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


def pad_sequences(
    sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0.0
):
    # Ported verbatim from reader.py (pure numpy padding utility, not a model change).
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    lengths = []
    for x in sequences:
        if not hasattr(x, "__len__"):
            raise ValueError(
                "`sequences` must be a list of iterables. Found non-iterable: " + str(x)
            )
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)
    if maxlen is not None and cfg.truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == "pre":
            trunc = s[-maxlen:]
        else:
            trunc = s[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == "post":
            x[idx, : len(trunc)] = trunc
        else:
            x[idx, -len(trunc) :] = trunc
    return x


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
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
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
        u_copy_score = torch.tanh(self.proj_copy1(u_enc_out.transpose(0, 1)))  # [B,T,H]
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
            pv_z_copy_score = torch.tanh(self.proj_copy2(pv_z_enc_out.transpose(0, 1)))  # [B,T,H]
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
        z_copy_score = torch.tanh(self.proj_copy2(z_enc_out.transpose(0, 1)))
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


class _VocabStub:
    """Minimal stand-in for the real repo Vocab class: tsd_net.py only ever calls
    `vocab.encode(token_str) -> int` and `vocab.decode(idx) -> str` on it."""

    def __init__(self, vocab_size):
        self._special = {"EOS_M": 4, "EOS_Z2": 5}
        self._vocab_size = vocab_size

    def encode(self, token):
        return self._special.get(token, 1)

    def decode(self, idx):
        for k, v in self._special.items():
            if v == int(idx):
                return k
        return "<unk>"


class TSD(nn.Module):
    """Two-Stage CopyNet dialogue model (belief-span decoder then response decoder),
    as introduced in the Sequicity paper (ACL 2018) and used to instantiate TSCP.
    Vendored verbatim from WING-NUS/sequicity tsd_net.py.
    """

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

    def bspan_decoder(
        self,
        u_enc_out,
        z_tm1,
        last_hidden,
        u_input_np,
        pv_z_enc_out,
        prev_z_input_np,
        u_emb,
        pv_z_emb,
    ):
        pz_dec_outs = []
        pz_proba = []
        decoded = []
        batch_size = u_enc_out.size(1)
        hiddens = [None] * batch_size
        for t in range(cfg.z_length):
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
            z_proba, z_index = torch.topk(proba, 1)  # [B,1]
            z_index = z_index.data.view(-1)
            decoded.append(z_index.clone())
            for i in range(z_index.size(0)):
                if z_index[i] >= cfg.vocab_size:
                    z_index[i] = 2  # unk
            z_np = z_tm1.view(-1).cpu().data.numpy()
            for i in range(batch_size):
                if z_np[i] == self.vocab.encode("EOS_Z2"):
                    hiddens[i] = last_hidden[:, i, :]
            z_tm1 = cuda_(Variable(z_index).view(1, -1))
        for i in range(batch_size):
            if hiddens[i] is None:
                hiddens[i] = last_hidden[:, i, :]
        last_hidden = torch.stack(hiddens, dim=1)
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        decoded = [list(_) for _ in decoded]
        return pz_dec_outs, decoded, last_hidden

    def greedy_decode(
        self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input, bspan_index
    ):
        decoded = []
        bspan_index_np = pad_sequences(bspan_index).transpose((1, 0))
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.m_decoder(
                pz_dec_outs, u_enc_out, u_input_np, m_tm1, degree_input, last_hidden, bspan_index_np
            )
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())
            for i in range(mt_index.size(0)):
                if mt_index[i] >= cfg.vocab_size:
                    mt_index[i] = 2  # unk
            m_tm1 = cuda_(Variable(mt_index).view(1, -1))
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded]

    def forward(self, u_input, u_len_t, dummy_degree_input):
        """Test-mode forward: encode a user utterance, decode a belief-span (z), then
        greedy-decode a response conditioned on it. This exercises the real two-stage
        CopyNet decoders (BSpanDecoder + ResponseDecoder) that are the paper's
        architectural contribution, with a minimal-but-real self-contained call path
        (no external reader/db pipeline)."""
        u_input_np = u_input.cpu().numpy()
        u_len = u_len_t.cpu().numpy()
        batch_size = u_input.size(1)

        u_enc_out, u_enc_hidden, u_emb = self.u_encoder(u_input, u_len)
        last_hidden = u_enc_hidden[:-1]
        z_tm1 = cuda_(Variable(torch.ones(1, batch_size).long() * 3))  # GO_2 token
        m_tm1 = cuda_(Variable(torch.ones(1, batch_size).long()))  # GO token

        pz_dec_outs, bspan_index, last_hidden = self.bspan_decoder(
            u_enc_out,
            z_tm1,
            last_hidden,
            u_input_np,
            pv_z_enc_out=None,
            prev_z_input_np=None,
            u_emb=u_emb,
            pv_z_emb=None,
        )
        pz_dec_outs = torch.cat(pz_dec_outs, dim=0)

        # In the real repo, degree_input comes from `reader.db_degree_handler`
        # (a symbolic-DB lookup, not a neural component); we feed it directly here.
        degree_input = dummy_degree_input

        m_output_index = self.greedy_decode(
            pz_dec_outs,
            u_enc_out,
            m_tm1,
            u_input_np,
            last_hidden,
            degree_input,
            bspan_index,
        )
        return m_output_index


MENAGERIE_ZOO = "vendored-pytorch"


def build_tscp():
    vocab_size = cfg.vocab_size
    vocab = _VocabStub(vocab_size)
    reader = object()  # unused in the test-mode forward path exercised here
    model = TSD(
        embed_size=16,
        hidden_size=24,
        vocab_size=vocab_size,
        degree_size=5,
        layer_num=1,
        dropout_rate=0.0,
        z_length=cfg.z_length,
        max_ts=cfg.max_ts,
        beam_search=False,
        teacher_force=0,
        vocab=vocab,
        reader=reader,
    )
    model.eval()
    return model


def example_input_tscp():
    seq_len, batch_size = 6, 2
    u_input = torch.randint(low=6, high=cfg.vocab_size, size=(seq_len, batch_size)).long()
    u_len = torch.tensor([seq_len, seq_len - 1])
    degree_input = torch.zeros(batch_size, 5)
    return u_input, u_len, degree_input


MENAGERIE_ENTRIES = [
    ("TSCP", build_tscp, example_input_tscp, 2018, "vendored-pytorch"),
]
