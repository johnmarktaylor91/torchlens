# SOURCE: vendored from https://github.com/moonlightlane/QG-Net @ master
# (OpenNMT-py/onmt/Models.py + OpenNMT-py/onmt/modules/{GlobalAttention,Embeddings,
# StackedRNN,UtilClass}.py)
#
# QG-Net: A Data-Driven Question Generation Model for Educational Content
# (Chen, Wu & Chi Yang, L@S 2018). The official repo does not add any custom
# nn.Module code of its own: it vendors a 2018-era snapshot of OpenNMT-py
# (v0.4, pre torch.nn.functional softmax(dim=) API) and trains it via
# `OpenNMT-py/train.py` with the flags in the repo's `train.sh`:
#   -encoder_type brnn -decoder_type rnn -rnn_type LSTM -input_feed 1 -copy_attn
# i.e. QG-Net's "model" *is* an unmodified OpenNMT-py NMTModel: a bidirectional
# LSTM RNNEncoder feeding an InputFeedRNNDecoder (stacked LSTMCells with Luong
# global attention at every decode step, plus a second copy-attention head over
# the encoder context for extractive copy from the source passage).
#
# The classes below are that vendored OpenNMT-py 0.4 model code, copied
# faithfully. Only minimal, non-architectural changes were made to run on a
# modern torch: `torch.autograd.Variable` wrapping was dropped (a no-op on
# modern torch), `nn.Softmax()` calls were given an explicit `dim=` (the 2018
# API defaulted to the last dim; behavior is unchanged), the boolean mask
# arithmetic in `GlobalAttention.forward` (`1 - mask` on a bool/uint8 tensor)
# was rewritten with `~mask` (same semantics, since torch now rejects `1 -
# BoolTensor`), and the repo-relative `onmt.*` imports were inlined into this
# single file. No layer, dimension, or control-flow logic was altered.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


def aeq(*args):
    """Assert all arguments have the same value (onmt.Utils.aeq)."""
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), "Not all arguments have the same value: " + str(
        args
    )


def sequence_mask(lengths, max_len=None):
    """Creates a boolean mask from sequence lengths (onmt.Utils.sequence_mask)."""
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))


# ---------------------------------------------------------------------------
# onmt/modules/UtilClass.py
# ---------------------------------------------------------------------------


class Bottle(nn.Module):
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.contiguous().view(size[0], size[1], -1)


class BottleLinear(Bottle, nn.Linear):
    pass


class Elementwise(nn.ModuleList):
    """A simple network container: parameters are a list of modules; inputs
    are a 3d tensor whose last dimension is the same length as the list;
    outputs are the result of applying modules to inputs elementwise."""

    def __init__(self, merge=None, *args):
        assert merge in [None, "first", "concat", "sum", "mlp"]
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, input):
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.merge == "first":
            return outputs[0]
        elif self.merge == "concat" or self.merge == "mlp":
            return torch.cat(outputs, 2)
        elif self.merge == "sum":
            return sum(outputs)
        else:
            return outputs


# ---------------------------------------------------------------------------
# onmt/modules/Embeddings.py
# ---------------------------------------------------------------------------


class Embeddings(nn.Module):
    """Words embeddings dictionary for encoder/decoder."""

    def __init__(
        self,
        word_vec_size,
        position_encoding,
        feat_merge,
        feat_vec_exponent,
        feat_vec_size,
        dropout,
        word_padding_idx,
        feat_padding_idx,
        word_vocab_size,
        feat_vocab_sizes=[],
    ):
        self.word_padding_idx = word_padding_idx

        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        if feat_merge == "sum":
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab**feat_vec_exponent) for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad) for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        self.embedding_size = sum(emb_dims) if feat_merge == "concat" else word_vec_size

        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module("emb_luts", emb_luts)

        if feat_merge == "mlp":
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(BottleLinear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module("mlp", mlp)

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def forward(self, input):
        """input (LongTensor): len x batch x nfeat -> emb: len x batch x embedding_size"""
        in_length, in_batch, nfeat = input.size()
        aeq(nfeat, len(self.emb_luts))

        emb = self.make_embedding(input)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)

        return emb


# ---------------------------------------------------------------------------
# onmt/modules/StackedRNN.py
# ---------------------------------------------------------------------------


class StackedLSTM(nn.Module):
    """Stacked LSTM used by the decoder for input feeding."""

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


# ---------------------------------------------------------------------------
# onmt/modules/GlobalAttention.py
# ---------------------------------------------------------------------------


class GlobalAttention(nn.Module):
    """Luong (dot/general) or Bahdanau (mlp) global attention."""

    def __init__(self, dim, coverage=False, attn_type="dot"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ["dot", "general", "mlp"], "Please select a valid attention type."

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = BottleLinear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = BottleLinear(dim, 1, bias=False)
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input, context, context_lengths=None, coverage=None):
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            context = context + self.linear_cover(cover).view_as(context)
            context = self.tanh(context)

        align = self.score(input, context)

        if context_lengths is not None:
            mask = sequence_mask(context_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align = align.masked_fill(~mask, -float("inf"))

        align_vectors = self.sm(align.view(batch * targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        c = torch.bmm(align_vectors, context)

        concat_c = torch.cat([c, input], 2).view(batch * targetL, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, align_vectors


# ---------------------------------------------------------------------------
# onmt/Models.py
# ---------------------------------------------------------------------------


class EncoderBase(nn.Module):
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            (n_batch_,) = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        raise NotImplementedError


class RNNEncoder(EncoderBase):
    """The standard (bidirectional) RNN encoder."""

    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, dropout, embeddings):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        self.rnn = getattr(nn, rnn_type)(
            input_size=embeddings.embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, input, lengths=None, hidden=None):
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        outputs, hidden_t = self.rnn(emb, hidden)

        return hidden_t, outputs


class RNNDecoderBase(nn.Module):
    def __init__(
        self,
        rnn_type,
        bidirectional_encoder,
        num_layers,
        hidden_size,
        attn_type,
        coverage_attn,
        context_gate,
        copy_attn,
        dropout,
        embeddings,
    ):
        super(RNNDecoderBase, self).__init__()

        self.decoder_type = "rnn"
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size, num_layers, dropout)

        self.context_gate = None

        self._coverage = coverage_attn
        self.attn = GlobalAttention(hidden_size, coverage=coverage_attn, attn_type=attn_type)

        self._copy = False
        if copy_attn:
            self.copy_attn = GlobalAttention(hidden_size, attn_type=attn_type)
            self._copy = True

    def forward(self, input, context, state, context_lengths=None):
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)

        hidden, outputs, attns, coverage = self._run_forward_pass(
            input, context, state, context_lengths=context_lengths
        )

        final_output = outputs[-1]
        state.update_state(
            hidden,
            final_output.unsqueeze(0),
            coverage.unsqueeze(0) if coverage is not None else None,
        )

        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # LSTM
            return RNNDecoderState(
                context,
                self.hidden_size,
                tuple([self._fix_enc_hidden(enc_hidden[i]) for i in range(len(enc_hidden))]),
            )
        else:  # GRU
            return RNNDecoderState(context, self.hidden_size, self._fix_enc_hidden(enc_hidden))


class InputFeedRNNDecoder(RNNDecoderBase):
    """RNN decoder with input feeding and (optionally) copy attention."""

    def _run_forward_pass(self, input, context, state, context_lengths=None):
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)

        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) if state.coverage is not None else None

        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output, context.transpose(0, 1), context_lengths=context_lengths
            )
            output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            if self._coverage:
                coverage = coverage + attn if coverage is not None else attn
                attns["coverage"] += [coverage]

            if self._copy:
                _, copy_attn = self.copy_attn(output, context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        assert rnn_type == "LSTM", "Only LSTM is used for QG-Net's input-feed decoder"
        stacked_cell = StackedLSTM
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """The encoder + decoder Neural Machine Translation Model."""

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(
            tgt, context, enc_state if dec_state is None else dec_state
        )
        if self.multigpu:
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = context.new_zeros(*h_size).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_qgnet():
    """Small-scale QG-Net NMTModel, matching train.sh's flags:
    -encoder_type brnn -decoder_type rnn -rnn_type LSTM -input_feed 1 -copy_attn
    (tiny dims so it traces fast; architecture unchanged)."""
    src_vocab = 48
    tgt_vocab = 40
    word_vec_size = 16
    rnn_size = 24  # opt.rnn_size; must be even for brnn

    src_embeddings = Embeddings(
        word_vec_size=word_vec_size,
        position_encoding=False,
        feat_merge="concat",
        feat_vec_exponent=0.7,
        feat_vec_size=0,
        dropout=0.0,
        word_padding_idx=0,
        feat_padding_idx=[],
        word_vocab_size=src_vocab,
        feat_vocab_sizes=[],
    )
    tgt_embeddings = Embeddings(
        word_vec_size=word_vec_size,
        position_encoding=False,
        feat_merge="concat",
        feat_vec_exponent=0.7,
        feat_vec_size=0,
        dropout=0.0,
        word_padding_idx=0,
        feat_padding_idx=[],
        word_vocab_size=tgt_vocab,
        feat_vocab_sizes=[],
    )

    encoder = RNNEncoder(
        rnn_type="LSTM",
        bidirectional=True,
        num_layers=2,
        hidden_size=rnn_size,
        dropout=0.0,
        embeddings=src_embeddings,
    )
    decoder = InputFeedRNNDecoder(
        rnn_type="LSTM",
        bidirectional_encoder=True,
        num_layers=2,
        hidden_size=rnn_size,
        attn_type="general",
        coverage_attn=False,
        context_gate=None,
        copy_attn=True,
        dropout=0.0,
        embeddings=tgt_embeddings,
    )

    model = NMTModel(encoder, decoder)
    model.eval()
    return model


def example_input_qgnet():
    """(src, tgt, lengths) matching NMTModel.forward's expected shapes:
    src/tgt: len x batch x nfeat (nfeat=1, no extra features); lengths: batch."""
    src_len, tgt_len, batch = 10, 8, 2
    src = torch.randint(1, 48, (src_len, batch, 1), dtype=torch.long)
    tgt = torch.randint(1, 40, (tgt_len, batch, 1), dtype=torch.long)
    lengths = torch.tensor([src_len, src_len - 2], dtype=torch.long)
    return src, tgt, lengths


MENAGERIE_ENTRIES = [
    ("QG-Net", "build_qgnet", "example_input_qgnet", 2018, MENAGERIE_ZOO),
]
