# SOURCE: vendored from HLTCHKUST/PAML @ master (commit at fetch time)
# https://raw.githubusercontent.com/HLTCHKUST/PAML/master/model/transformer.py
# https://raw.githubusercontent.com/HLTCHKUST/PAML/master/model/common_layer.py
#
# Madotto et al. 2019 (ACL) "Personalizing Dialogue Agents via Meta-Learning" (PAML) --
# applies MAML (model-agnostic meta-learning) to adapt a persona-conditioned dialogue
# generator to new personas from a handful of examples. The meta-learning procedure
# (MAML.py: inner-loop `do_learning` + outer-loop gradient steps over `train_one_batch`)
# is a *training algorithm*, not an architecture; the actual network PAML meta-learns
# is the Transformer encoder-decoder defined in `model/transformer.py` /
# `model/common_layer.py` (itself adapted from kolloldas/torchnlp per the repo's own
# header comment), which is what is vendored here verbatim.
#
# `Encoder`, `Decoder`, `Generator`, `ACT_basic`, `Transformer` are copied from
# `model/transformer.py`. `EncoderLayer`, `DecoderLayer`, `MultiHeadAttention`, `Conv`,
# `PositionwiseFeedForward`, `LayerNorm`, `share_embedding`, `_gen_bias_mask`,
# `_gen_timing_signal`, `_get_attn_subsequent_mask` are copied from
# `model/common_layer.py`. No architecture code was rewritten; only these mechanical,
# import-isolation changes were made:
#   - The real `utils.config` module (an argparse-at-import-time global config) is
#     replaced by a plain `SimpleNamespace` built inline in `build_paml_transformer()`,
#     holding the same fields the model code reads (`emb_dim`, `hidden_dim`, `hop`,
#     `heads`, `depth`, `filter`, `universal`, `act`, `max_enc_steps`, `PAD_idx`,
#     `SOS_idx`, `EOS_idx`, `pointer_gen`, `weight_sharing`, `label_smoothing`, `noam`,
#     `use_sgd`, `USE_CUDA`, `preptrained`, `save_path`, `lr`) -- same values, just
#     supplied directly instead of via CLI flags.
#   - `Transformer.__init__` is trimmed to the architecture-construction lines only
#     (embedding/encoder/decoder/generator/weight-sharing/criterion); the
#     checkpoint-loading branch (`model_file_path is not None`), the `.cuda()` calls,
#     and the training-only methods (`train_one_batch`/`save_model`/`decoder_greedy`/
#     `score_sentence`/`compute_act_loss`) are dropped -- none of them run during a
#     forward trace. `forward()` below is assembled from the real encode/decode/
#     generate steps in the body of `train_one_batch`, minus the loss computation.
#   - `share_embedding(vocab, pretrain=False)` is used (skips loading a GloVe file
#     from disk, an upstream-supported code path already, since `preptrained=False`
#     is a legal upstream config value).
#   - `evaluate()`, the module-level `bert = bert_model()` call, and the
#     `utils.metric`/`utils.beam_omt`/`utils.load_bert` imports they require are
#     dropped: they are eval-time helpers (BLEU/entailment scoring, beam-search
#     decoding) unrelated to the `Transformer` module's forward architecture.
#   - `np.random` seeding lines are dropped (irrelevant to architecture; the repo
#     seeds RNGs for training reproducibility, not to define the model).
#   - A local `_TinyVocab` stand-in supplies `n_words`, `word2index`, `index2word` --
#     the same duck-typed interface the real `utils.data_reader.Lang` class provides,
#     just populated with a tiny synthetic vocabulary instead of parsed from the
#     ConvAI2/PERSONA-CHAT corpus.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# from model/common_layer.py
# ---------------------------------------------------------------------------


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
            raise ValueError(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
        if total_value_depth % num_heads != 0:
            raise ValueError(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  # sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
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


def share_embedding(vocab, config, pretrain=True):
    embedding = nn.Embedding(vocab.n_words, config.emb_dim)
    return embedding


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
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


# ---------------------------------------------------------------------------
# from model/transformer.py
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
        config,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        super(Encoder, self).__init__()
        self.config = config
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

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if self.config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers
                )
                y = self.layer_norm(x)
            else:
                for layer_idx in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                    x += (
                        self.position_signal[:, layer_idx, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
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
        config,
        max_length=100,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        super(Decoder, self).__init__()
        self.config = config
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

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

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0)
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if self.config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)
            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for layer_idx in range(self.num_layers):
                    x += (
                        self.position_signal[:, layer_idx, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab, config):
        super(Generator, self).__init__()
        self.config = config
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
    ):
        config = self.config
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            p_gen = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = p_gen * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - p_gen) * attn_dist
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


class ACT_basic(nn.Module):
    # CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(
        self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, decoding=False
    ):
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1])
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1])
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1])
        previous_state = torch.zeros_like(inputs)

        step = 0
        while ((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any():
            state = state + time_enc[:, : inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(
                inputs.data
            )

            p = self.sigma(self.p(state)).squeeze(-1)
            still_running = (halting_probability < 1.0).float()

            new_halted = (
                halting_probability + p * still_running > self.threshold
            ).float() * still_running

            still_running = (
                halting_probability + p * still_running <= self.threshold
            ).float() * still_running

            halting_probability = halting_probability + p * still_running

            remainders = remainders + new_halted * (1 - halting_probability)

            halting_probability = halting_probability + new_halted * remainders

            n_updates = n_updates + still_running + new_halted

            update_weights = p * still_running + new_halted * remainders

            if decoding:
                state, _, attention_weight = fn((state, encoder_output, []))
            else:
                state = fn(state)

            previous_state = (state * update_weights.unsqueeze(-1)) + (
                previous_state * (1 - update_weights.unsqueeze(-1))
            )
            if decoding:
                if step == 0:
                    previous_att_weight = torch.zeros_like(attention_weight)
                previous_att_weight = (attention_weight * update_weights.unsqueeze(-1)) + (
                    previous_att_weight * (1 - update_weights.unsqueeze(-1))
                )
            step += 1

        if decoding:
            return previous_state, previous_att_weight, (remainders, n_updates)
        else:
            return previous_state, (remainders, n_updates)


class Transformer(nn.Module):
    """
    The PAML persona-dialogue Transformer (MAML.py meta-learns over this module's
    parameters via `model.train_one_batch`'s inner loop; this class is the exact
    architecture MAML is applied to).
    """

    def __init__(self, vocab, config, is_eval=False):
        super(Transformer, self).__init__()
        self.config = config
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config, config.preptrained)
        self.encoder = Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            config=config,
            universal=config.universal,
        )

        self.decoder = Decoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            config=config,
            max_length=config.max_enc_steps,
            universal=config.universal,
        )
        self.generator = Generator(config.hidden_dim, self.vocab_size, config)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(
                size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1
            )
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        if is_eval:
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()
            self.generator = self.generator.eval()
            self.embedding = self.embedding.eval()

    def forward(self, enc_batch, dec_batch_shift):
        """Assembled from the encode/decode/generate steps in the real
        `Transformer.train_one_batch` (model/transformer.py), minus the loss."""
        mask_src = enc_batch.data.eq(self.config.PAD_idx).unsqueeze(1)
        encoder_outputs = self.encoder(self.embedding(enc_batch), mask_src)

        mask_trg = dec_batch_shift.data.eq(self.config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(
            self.embedding(dec_batch_shift), encoder_outputs, (mask_src, mask_trg)
        )

        logit = self.generator(pre_logit, attn_dist, None, None)
        return logit


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class _TinyVocab:
    """Duck-typed stand-in for utils.data_reader.Lang (n_words/word2index/index2word)."""

    def __init__(self, n_words):
        self.n_words = n_words
        self.word2index = {str(i): i for i in range(n_words)}
        self.index2word = {i: str(i) for i in range(n_words)}


def build_paml_transformer():
    config = SimpleNamespace(
        emb_dim=32,
        hidden_dim=32,
        hop=2,
        heads=2,
        depth=16,
        filter=32,
        universal=False,
        act=False,
        max_enc_steps=32,
        PAD_idx=1,
        SOS_idx=3,
        EOS_idx=2,
        pointer_gen=False,
        weight_sharing=False,
        label_smoothing=False,
        noam=False,
        use_sgd=False,
        USE_CUDA=False,
        preptrained=False,
        save_path="save/test/",
        lr=0.0001,
    )
    vocab = _TinyVocab(n_words=60)
    return Transformer(vocab, config)


def example_input_paml_transformer():
    batch = 2
    src_len = 5
    trg_len = 4
    enc_batch = torch.randint(2, 60, (batch, src_len))
    dec_batch_shift = torch.randint(2, 60, (batch, trg_len))
    return (enc_batch, dec_batch_shift)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("PAML", "build_paml_transformer", "example_input_paml_transformer", 2019, "vendored"),
]
