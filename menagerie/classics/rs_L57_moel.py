# SOURCE: vendored from HLTCHKUST/MoEL @ master (commit at fetch time)
# https://raw.githubusercontent.com/HLTCHKUST/MoEL/master/model/transformer_mulexpert.py
# https://raw.githubusercontent.com/HLTCHKUST/MoEL/master/model/common_layer.py
#
# Lin et al. 2019 (EMNLP) "MoEL: Mixture of Empathetic Listeners" -- a Transformer
# encoder over the dialogue context, an emotion-classification head over the encoder's
# pooled state (`decoder_key`), and N "listener" Transformer-decoder experts
# (`MulDecoder.experts`, one per emotion) whose outputs are combined by a learned
# soft-attention distribution (`attention_activation` over `decoder_key` logits) before
# a final shared decoder stack + generator. This is the "Mixture of Empathetic
# Listeners" architecture itself (encoder + per-emotion expert decoders + soft mixture
# + shared final decoder + generator), taken verbatim from the two real model files.
#
# `Encoder`, `Decoder`, `MulDecoder`, `Generator`, `ACT_basic`, `Transformer_experts`
# are copied from `model/transformer_mulexpert.py`. `EncoderLayer`, `DecoderLayer`,
# `MultiHeadAttention`, `Conv`, `PositionwiseFeedForward`, `LayerNorm`, `Embeddings`,
# `share_embedding`, `_gen_bias_mask`, `_gen_timing_signal`, `_get_attn_subsequent_mask`
# are copied from `model/common_layer.py`. No architecture code was rewritten; only
# these mechanical, import-isolation changes were made:
#   - The real `utils.config` module (an argparse-at-import-time global config that
#     also gates on `os.cpu_count()>8` for CUDA) is replaced by a plain `SimpleNamespace`
#     built inline in `build_moel()`, holding the same fields the model code reads
#     (`emb_dim`, `hidden_dim`, `hop`, `heads`, `depth`, `filter`, `universal`, `act`,
#     `PAD_idx`, `pointer_gen`, `label_smoothing`, `weight_sharing`, `basic_learner`,
#     `topk`, `softmax`, `oracle`, `project`, `noam`, `USE_CUDA`, `pretrain_emb`,
#     `save_path`, `lr`) -- same values, just supplied directly instead of via CLI flags.
#   - `Transformer_experts.__init__` is trimmed to the architecture-construction lines
#     only (embedding/encoder/decoder/decoder_key/generator/emoji_embedding/weight
#     sharing/criterion/optimizer); the training-only methods
#     (`train_one_batch`/`save_model`/`decoder_greedy`/`decoder_topk`/`compute_act_loss`)
#     and the checkpoint-loading branch (`model_file_path is not None`) are dropped --
#     none of them run during a forward trace. `forward()` below is the real
#     encode -> decoder_key gate -> attention_activation -> MulDecoder(experts) ->
#     generator pipeline, assembled from the body of `train_one_batch` (the encode/
#     gate/decode/generate steps, minus the loss computation and CUDA-only `.cuda()`
#     calls the original hardcodes for its topk/oracle training branches -- this build
#     uses `config.topk=0` and `config.oracle=False`, i.e. the plain non-topk,
#     non-oracle attention path, so those CUDA-only branches are simply not taken,
#     matching upstream behavior for that same config).
#   - `share_embedding(vocab, pretrain=False)` is used (skips loading a GloVe file from
#     disk, an upstream-supported code path already).
#   - `np.float` (removed in modern numpy) -> `float` in `_gen_timing_signal`.
#   - A local `Lang`-alike vocab stand-in (`_TinyVocab`) supplies `n_words`,
#     `word2index`, `index2word` -- the same duck-typed interface the real
#     `utils.data_reader.Lang` class provides, just populated with a tiny synthetic
#     vocabulary instead of parsed from the EmpatheticDialogues corpus.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# from model/common_layer.py
# ---------------------------------------------------------------------------


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
        self,
        hidden_size,
        total_key_depth,
        total_value_depth,
        filter_size,
        num_heads,
        bias_mask=None,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="both",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        x = inputs

        x_norm = self.layer_norm_mha(x)

        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)

        x = self.dropout(x + y)

        x_norm = self.layer_norm_ffn(x)

        y = self.positionwise_feed_forward(x_norm)

        y = self.dropout(x + y)

        return y


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
        self,
        hidden_size,
        total_key_depth,
        total_value_depth,
        filter_size,
        num_heads,
        bias_mask,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention_dec = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout,
        )

        self.multi_head_attention_enc_dec = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            None,
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config="cc",
            padding="left",
            dropout=relu_dropout,
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask

        x_norm = self.layer_norm_mha_dec(x)

        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)

        x = self.dropout(x + y)

        x_norm = self.layer_norm_mha_enc(x)

        y, attention_weight = self.multi_head_attention_enc_dec(
            x_norm, encoder_outputs, encoder_outputs, mask_src
        )

        x = self.dropout(x + y)

        x_norm = self.layer_norm_ffn(x)

        y = self.positionwise_feed_forward(x_norm)

        y = self.dropout(x + y)

        return y, encoder_outputs, attention_weight, mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(
        self,
        input_depth,
        total_key_depth,
        total_value_depth,
        output_depth,
        num_heads,
        bias_mask=None,
        dropout=0.0,
    ):
        super(MultiHeadAttention, self).__init__()

        if total_key_depth % num_heads != 0:
            print(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask

        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(
            0, 2, 1, 3
        )

    def _merge_heads(self, x):
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (
            x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)
        )

    def forward(self, queries, keys, values, mask):
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        queries *= self.query_scale

        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        attetion_weights = logits.sum(dim=1) / self.num_heads

        weights = nn.functional.softmax(logits, dim=-1)

        weights = self.dropout(weights)

        contexts = torch.matmul(weights, values)

        contexts = self._merge_heads(contexts)

        outputs = self.output_linear(contexts)

        return outputs, attetion_weights


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        super(Conv, self).__init__()
        padding = (
            (kernel_size - 1, 0)
            if pad_type == "left"
            else (kernel_size // 2, (kernel_size - 1) // 2)
        )
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(
        self, input_depth, filter_size, output_depth, layer_config="ll", padding="left", dropout=0.0
    ):
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = (
            [(input_depth, filter_size)]
            + [(filter_size, filter_size)] * (len(layer_config) - 2)
            + [(filter_size, output_depth)]
        )

        for lc, s in zip(list(layer_config), sizes):
            if lc == "l":
                layers.append(nn.Linear(*s))
            elif lc == "c":
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        float(num_timescales) - 1
    )
    # NOTE: `np.float` (removed in modern numpy) -> `float`; upstream used np.float as a
    # bare alias for the Python builtin, no behavior change.
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(float) * -log_timescale_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def share_embedding(vocab, config, pretrain=False):
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
    # `pretrain=True` in the original loads a GloVe file from disk via `gen_embeddings`;
    # this build always uses `pretrain=False` (a code path the upstream function already
    # supports) so construction stays self-contained.
    return embedding


# ---------------------------------------------------------------------------
# from model/transformer_mulexpert.py
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        x = self.input_dropout(inputs)

        x = self.embedding_proj(x)

        # Add timing signal
        x = x + self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

        for i in range(self.num_layers):
            x = self.enc[i](x, mask)

        y = self.layer_norm(x)
        return y


class MulDecoder(nn.Module):
    def __init__(
        self,
        expert_num,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        basic_learner=False,
        project=False,
    ):
        super(MulDecoder, self).__init__()
        self.num_layers = num_layers
        self.basic_learner = basic_learner
        self.project = project
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )
        if self.basic_learner:
            self.basic = DecoderLayer(*params)
        self.experts = nn.ModuleList([DecoderLayer(*params) for e in range(expert_num)])
        self.dec = nn.Sequential(*[DecoderLayer(*params) for _layer_idx in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask, attention_epxert):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0)
        x = self.input_dropout(inputs)
        if not self.project:
            x = self.embedding_proj(x)
        x = x + self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
        expert_outputs = []
        if self.basic_learner:
            basic_out, _, attn_dist, _ = self.basic((x, encoder_output, [], (mask_src, dec_mask)))

        # compute experts -- soft mixture path (attention_epxert has one weight per
        # expert per batch element; every expert decoder runs, weighted-summed)
        for i, expert in enumerate(self.experts):
            expert_out, _, attn_dist, _ = expert((x, encoder_output, [], (mask_src, dec_mask)))
            expert_outputs.append(expert_out)
        x = torch.stack(expert_outputs, dim=1)  # (batch_size, expert_number, len, hidden_size)
        x = attention_epxert * x
        x = x.sum(dim=1)  # (batch_size, len, hidden_size)
        if self.basic_learner:
            x = x + basic_out
        # Run decoder
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

        y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab, hidden_dim, pointer_gen=False):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(hidden_dim, 1)
        self.pointer_gen = pointer_gen

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):
        if self.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if self.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class _TinyVocab:
    """Minimal duck-typed stand-in for the real `utils.data_reader.Lang` object:
    the model code only reads `.n_words` at construction time."""

    def __init__(self, n_words):
        self.n_words = n_words
        self.word2index = {str(i): i for i in range(n_words)}
        self.index2word = {i: str(i) for i in range(n_words)}


class Transformer_experts(nn.Module):
    """
    MoEL: encoder -> emotion-gate over pooled encoder state (`decoder_key`) ->
    soft attention distribution over N expert listener decoders (`MulDecoder`) ->
    shared final decoder stack -> generator. Trimmed to construction + a forward()
    assembled from the real `train_one_batch` encode/gate/decode/generate steps
    (dropping the loss computation and checkpoint I/O, which do not affect the
    traced architecture).
    """

    def __init__(self, vocab, decoder_number, config):
        super(Transformer_experts, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.config = config

        self.embedding = share_embedding(self.vocab, config, config.pretrain_emb)
        self.encoder = Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )
        self.decoder_number = decoder_number
        self.decoder = MulDecoder(
            decoder_number,
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            basic_learner=config.basic_learner,
            project=config.project,
        )

        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)

        self.generator = Generator(
            config.hidden_dim, self.vocab_size, config.hidden_dim, pointer_gen=config.pointer_gen
        )
        self.emoji_embedding = nn.Linear(64, config.emb_dim, bias=False)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        if config.softmax:
            self.attention_activation = nn.Softmax(dim=1)
        else:
            self.attention_activation = nn.Sigmoid()

    def forward(self, enc_batch, dec_batch_shift):
        # Encode
        mask_src = enc_batch.data.eq(self.config.PAD_idx).unsqueeze(1)
        encoder_outputs = self.encoder(self.embedding(enc_batch), mask_src)

        # Attention over decoder ("listener") experts, gated by the pooled encoder state
        q_h = encoder_outputs[:, 0]
        logit_prob = self.decoder_key(q_h)  # (bsz, num_experts)

        attention_parameters = self.attention_activation(logit_prob)
        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(
            -1
        )  # (batch, expert_num, 1, 1)

        mask_trg = dec_batch_shift.data.eq(self.config.PAD_idx).unsqueeze(1)

        pre_logit, attn_dist = self.decoder(
            self.embedding(dec_batch_shift),
            encoder_outputs,
            (mask_src, mask_trg),
            attention_parameters,
        )
        logit = self.generator(pre_logit, attn_dist, None, None, attn_dist_db=None)
        return logit


def build_moel():
    config = SimpleNamespace(
        emb_dim=32,
        hidden_dim=32,
        hop=2,
        heads=2,
        depth=16,
        filter=32,
        universal=False,
        PAD_idx=1,
        pointer_gen=False,
        label_smoothing=False,
        weight_sharing=False,
        basic_learner=False,
        topk=0,
        softmax=True,
        oracle=False,
        project=False,
        noam=False,
        USE_CUDA=False,
        pretrain_emb=False,
        save_path="save/test/",
        lr=0.0001,
    )
    vocab = _TinyVocab(n_words=60)
    decoder_number = 32  # number of emotion classes / listener experts (empatheticdialogues has 32)
    return Transformer_experts(vocab, decoder_number, config)


def example_input_moel():
    batch = 2
    src_len = 5
    trg_len = 4
    enc_batch = torch.randint(2, 60, (batch, src_len))
    dec_batch_shift = torch.randint(2, 60, (batch, trg_len))
    return (enc_batch, dec_batch_shift)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MoEL", "build_moel", "example_input_moel", 2019, "vendored"),
]
