# SOURCE: vendored from magic282/NQG @ master
# https://raw.githubusercontent.com/magic282/NQG/master/seq2seq_pt/s2s/Models.py
# https://raw.githubusercontent.com/magic282/NQG/master/seq2seq_pt/s2s/modules/ConcatAttention.py
# https://raw.githubusercontent.com/magic282/NQG/master/seq2seq_pt/s2s/modules/Maxout.py
# https://raw.githubusercontent.com/magic282/NQG/master/seq2seq_pt/s2s/Constants.py (PAD id only)
#
# Zhou et al. 2017 "Neural Question Generation from Text: A Preliminary Study" (NQG) -- a
# BiGRU encoder (with extra BIO-tag and answer-position feature embeddings concatenated onto the
# word embedding, `Encoder`) + a GRU decoder with input-feeding, Bahdanau-style concat attention
# (`ConcatAttention`), a maxout readout layer (`MaxOut`), and a copy-mechanism gate (`copySwitch`)
# that at every decoder step outputs a probability of copying the attended source word instead of
# generating from the target vocabulary (`NMTModel`/`Decoder`). This is real, already torch-only
# repo code (the seq2seq_pt/ tree has no non-base dependencies); it was written for PyTorch ~0.4 so
# it uses deprecated-but-still-functional APIs directly. No architectural change was made; the only
# edits are mechanical modernization needed to run under current torch:
#   - `torch.autograd.Variable(..., volatile=False)` (removed keyword in modern torch) -> the plain
#     tensor itself (`Variable` on a tensor is a no-op since torch 0.4; `volatile` was removed in the
#     same release and has no replacement needed for eager inference).
#   - `F.sigmoid(...)` (removed function) -> `torch.sigmoid(...)` (same op, current spelling).
#   - `NMTModel.forward`'s original signature unpacked a single 4-tuple-of-tuples "batch" object
#     built by `s2s.Dataset`; `build_nqg()`/`example_input_nqg()` below construct that same nested
#     structure directly (word ids + BIO ids + a list of one feature-id tensor + lengths) so the
#     model can be traced without vendoring the full `Dataset`/data-loading pipeline.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

PAD = 0


# ---- s2s/modules/ConcatAttention.py ----
class ConcatAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ConcatAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(
                context.size(0), context.size(1), -1
            )  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        tmp20 = self.tanh(tmp10)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(
            tmp20.size(0), tmp20.size(1)
        )  # batch x sourceL
        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1000000)
        score = self.sm(energy)
        score_m = score.view(score.size(0), 1, score.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(score_m, context).squeeze(1)  # batch x dim

        return weightedContext, score, precompute

    def extra_repr(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.att_dim)
            + " * "
            + "("
            + str(self.attend_dim)
            + "->"
            + str(self.att_dim)
            + " + "
            + str(self.query_dim)
            + "->"
            + str(self.att_dim)
            + ")"
            + ")"
        )


# ---- s2s/modules/Maxout.py ----
class MaxOut(nn.Module):
    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, input):
        input_size = list(input.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        input = input.view(*output_size)
        input, idx = input.max(last_dim, keepdim=True)
        output = input.squeeze(last_dim)

        return output

    def extra_repr(self):
        return self.__class__.__name__ + "({0})".format(self.pool_size)


# ---- s2s/Models.py ----
class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.enc_rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts, opt.word_vec_size, padding_idx=PAD)
        self.bio_lut = nn.Embedding(8, 16, padding_idx=PAD)  # TODO: Fix this magic number
        self.feat_lut = nn.Embedding(64, 16, padding_idx=PAD)  # TODO: Fix this magic number
        input_size = input_size + 16 + 16 * 3
        self.rnn = nn.GRU(
            input_size,
            self.hidden_size,
            num_layers=opt.layers,
            dropout=opt.dropout,
            bidirectional=opt.brnn,
        )

    def forward(self, input, bio, feats, hidden=None):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths)
        """
        lengths = input[-1].data.view(-1).tolist()  # lengths data is wrapped inside a Variable
        wordEmb = self.word_lut(input[0])
        bioEmb = self.bio_lut(bio[0])
        featsEmb = [self.feat_lut(feat) for feat in feats[0]]
        featsEmb = torch.cat(featsEmb, dim=-1)
        input_emb = torch.cat((wordEmb, bioEmb, featsEmb), dim=-1)
        emb = pack(input_emb, lengths, enforce_sorted=False)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class Decoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts, opt.word_vec_size, padding_idx=PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear(
            (opt.enc_rnn_size + opt.dec_rnn_size + opt.word_vec_size), opt.dec_rnn_size
        )
        self.maxout = MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.copySwitch = nn.Linear(opt.enc_rnn_size + opt.dec_rnn_size, 1)

        self.hidden_size = opt.dec_rnn_size

    def forward(self, input, hidden, context, src_pad_mask, init_att):
        emb = self.word_lut(input)

        g_outputs = []
        c_outputs = []
        copyGateOutputs = []
        cur_context = init_att
        self.attn.applyMask(src_pad_mask)
        precompute = None
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context], 1)
            output, hidden = self.rnn(input_emb, hidden)
            cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

            copyProb = self.copySwitch(torch.cat((output, cur_context), dim=1))
            copyProb = torch.sigmoid(copyProb)

            readout = self.readout(torch.cat((emb_t, output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
            c_outputs += [attn]
            copyGateOutputs += [copyProb]
        g_outputs = torch.stack(g_outputs)
        c_outputs = torch.stack(c_outputs)
        copyGateOutputs = torch.stack(copyGateOutputs)
        return g_outputs, c_outputs, copyGateOutputs, hidden, attn, cur_context


class DecInit(nn.Module):
    def __init__(self, opt):
        super(DecInit, self).__init__()
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = opt.enc_rnn_size
        self.dec_rnn_size = opt.dec_rnn_size
        self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)
        self.tanh = nn.Tanh()

    def forward(self, last_enc_h):
        return self.tanh(self.initer(last_enc_h))


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, decIniter):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return context.data.new(*h_size).zero_()

    def forward(self, input):
        """
        (wrap(srcBatch), lengths), \
               (wrap(bioBatch), lengths), ((wrap(x) for x in featBatches), lengths), \
               (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
               indices
        """
        src = input[0]
        tgt = input[3][0][:-1]  # exclude last target from inputs
        src_pad_mask = src[0].data.eq(PAD).transpose(0, 1).float()
        bio = input[1]
        feats = input[2]
        enc_hidden, context = self.encoder(src, bio, feats)

        init_att = self.make_init_att(context)
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden

        g_out, c_out, c_gate_out, dec_hidden, _attn, _attention_vector = self.decoder(
            tgt, enc_hidden, context, src_pad_mask, init_att
        )

        return g_out, c_out, c_gate_out


class _Opt:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_nqg():
    vocab_size = 40
    opt = _Opt(
        layers=1,
        brnn=True,
        enc_rnn_size=32,
        dec_rnn_size=32,
        word_vec_size=16,
        dropout=0.0,
        input_feed=True,
        att_vec_size=16,
        maxout_pool_size=2,
    )
    encoder = Encoder(opt, vocab_size)
    decoder = Decoder(opt, vocab_size)
    dec_initer = DecInit(opt)
    model = NMTModel(encoder, decoder, dec_initer)
    model.eval()
    return model


def example_input_nqg():
    batch = 3
    src_len = 6
    tgt_len = 5

    src_words = torch.randint(1, 40, (src_len, batch))
    src_lengths = torch.tensor([src_len] * batch)

    bio_words = torch.randint(1, 8, (src_len, batch))
    bio_lengths = src_lengths

    # Encoder.forward concatenates one embedded feature per entry in feats[0] (3 answer-position-style
    # feature streams in the original NQG data pipeline, each embedded to feat_lut's 16 dims).
    feat_words = [torch.randint(1, 64, (src_len, batch)) for _ in range(3)]
    feat_lengths = src_lengths

    tgt_words = torch.randint(1, 40, (tgt_len, batch))
    copy_switch = torch.zeros(tgt_len, batch)
    copy_tgt = torch.zeros(tgt_len, batch, dtype=torch.long)

    src = (src_words, src_lengths)
    bio = (bio_words, bio_lengths)
    feats = (feat_words, feat_lengths)
    tgt = (tgt_words, copy_switch, copy_tgt)
    indices = torch.arange(batch)

    batch_input = (src, bio, feats, tgt, indices)
    return (batch_input,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("NQG", "build_nqg", "example_input_nqg", 2017, "vendored"),
]
