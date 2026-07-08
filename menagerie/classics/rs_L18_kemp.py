# SOURCE: vendored from qtli/KEMP @ b9a4fd58fa48556856c83d3d04125422d15e0b36
# https://raw.githubusercontent.com/qtli/KEMP/b9a4fd58fa48556856c83d3d04125422d15e0b36/code/KEMP.py
# https://raw.githubusercontent.com/qtli/KEMP/b9a4fd58fa48556856c83d3d04125422d15e0b36/code/common_layer.py
#
# Li et al. 2022 "Knowledge Bridging for Empathetic Dialogue Generation" (AAAI 2022,
# a.k.a. KEMP). Real architecture: a custom T2T-style Transformer `Encoder` runs over
# the dialogue-context embeddings; before encoding, a `concept_graph` cross-attention
# module (`W_q`/`W_k`/`W_v`/`graph_out` + LayerNorm) fuses ConceptNet concept-word
# embeddings into the context representation ("knowledge bridging" / emotional context
# graph). The fused sequence goes through the shared `Encoder`, an `identify` linear
# layer distills a VAD-weighted (valence-arousal-dominance) pooled context into an
# emotion-category logit, that logit is embedded and fed as the decoder's start-of-
# sequence token, and a custom `Decoder` (masked self-attn + cross-attn against the
# fused encoder memory, with an extra "emotion_contexts" cross-attention path baked
# into `MultiHeadAttention.forward`) autoregressively produces hidden states that a
# pointer-generator `Generator` head turns into vocabulary logits. All Transformer
# building blocks (`EncoderLayer`/`DecoderLayer`/`MultiHeadAttention` w/ VAD bias term
# and emotion-context fusion/`PositionwiseFeedForward`/`LayerNorm`, T2T sinusoidal
# timing signal) are the real code from code/common_layer.py, taken verbatim.
#
# Minimal, non-architectural changes made (bookkeeping / import-time-side-effect
# removal only; no computation changed):
#   - The real `KEMP.__init__` takes a `(word2index, word2count, index2word, n_words)`
#     vocab tuple, builds an `nn.NLLLoss`/`LabelSmoothing` criterion, an Adam/NoamOpt
#     optimizer, creates a checkpoint save directory on disk, and optionally loads a
#     `model_file_path` checkpoint. None of that is part of the traced architecture
#     (it is training/checkpoint bookkeeping); the vendored `KEMP.__init__` here keeps
#     the exact module graph (`embedding`, `encoder`, concept-graph linears, `identify`,
#     `emotion_embedding`, `decoder`, `decoder_key`, `generator`, optional
#     `embedding_proj_in`, optional weight-tying) and drops the loss/optimizer/
#     checkpoint-IO construction.
#   - `args` is a tiny local `_Config` namespace carrying the exact attribute names the
#     real `Encoder`/`Decoder`/`KEMP`/`Generator`/`MultiHeadAttention`/
#     `_get_attn_subsequent_mask` code reads (`emb_dim`, `hidden_dim`, `hop`, `heads`,
#     `depth`, `filter`, `universal`, `max_seq_length`, `PAD_idx`, `pointer_gen`,
#     `projection`, `weight_sharing`, `dropout`, `model`, `USE_CUDA`, `device`,
#     `pretrain_emb`, `emb_file`) -- values only, no behavior invented. This replaces
#     the real repo's `argparse`-based `utils/config.py` (which parses sys.argv at
#     import time and crashes with no CLI args).
#   - `share_embedding`'s real code loads a pretrained-vectors file from disk when
#     `pretrain=True` and `emb_dim` is 50/200/300; kept the function body verbatim but
#     the config here uses `pretrain_emb=False` (random init) so the module is
#     constructible with no vectors file on disk -- a data/loading concern, not an
#     architecture change.
#   - A minimal stand-in vocab tuple `(word2index, word2count, index2word, n_words)`
#     replaces the real repo's `pickle`-loaded vocab (only `n_words`/`word2index` are
#     read by the traced path, exactly as in the real `KEMP.__init__`).
#   - Dropped training-loop / greedy-decoding / checkpoint-save methods
#     (`train_one_batch`, `decoder_greedy`, `save_model`, `compute_act_loss`); kept the
#     real `KEMP.__init__` module graph and `concept_graph`/`Encoder`/`Decoder`/
#     `Generator` forward paths (the traced architecture) verbatim, wired together in a
#     small `forward()` that mirrors the real `train_one_batch`'s pre-decoder + decoder
#     + generator pipeline exactly (embed -> concept_graph fusion -> encode -> VAD-
#     weighted emotion identify -> decode -> generate).

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Config:
    """Stand-in for utils/config.py (real module executes argparse at import time)."""

    PAD_idx = 1
    SOS_idx = 3
    EOS_idx = 2
    emb_dim = 32
    hidden_dim = 32
    hop = 2
    heads = 2
    depth = 16
    filter = 24
    universal = False
    act = False
    pointer_gen = False
    projection = False
    weight_sharing = False
    label_smoothing = False
    noam = False
    pretrain_emb = False
    emb_file = None
    attn_loss = False
    model = "KEMP"
    max_seq_length = 64
    dropout = 0.0
    USE_CUDA = False
    use_cuda = False
    device = torch.device("cpu")
    save_path = "/tmp/kemp_menagerie_scratch"


args = _Config()


# ---- code/common_layer.py (real Transformer building blocks, verbatim) ----


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
        args,
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
        self.args = args
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
        if self.args.model not in ["KEMP", "wo_ECE", "wo_EDD"]:
            x, encoder_outputs, attention_weight, mask = inputs
            pred_emotion, emotion_contexts = None, None
        else:
            x, encoder_outputs, pred_emotion, emotion_contexts, attention_weight, mask = inputs
        mask_src, dec_mask = mask

        x_norm = self.layer_norm_mha_dec(x)
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)
        x = self.dropout(x + y)

        x_norm = self.layer_norm_mha_enc(x)
        y, attention_weight = self.multi_head_attention_enc_dec(
            x_norm,
            encoder_outputs,
            encoder_outputs,
            mask_src,
            emotion_contexts=emotion_contexts,
        )
        x = self.dropout(x + y)

        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_feed_forward(x_norm)
        y = self.dropout(x + y)

        return y, encoder_outputs, pred_emotion, emotion_contexts, attention_weight, mask


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
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask

        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.emotion_output_linear = nn.Linear(2 * output_depth, output_depth, bias=False)

        self.W_vad = nn.Parameter(torch.zeros(1))

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

    def forward(self, queries, keys, values, mask, vad=None, emotion_contexts=None):
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        queries = queries * self.query_scale

        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if vad is not None:
            vad = vad.unsqueeze(1).unsqueeze(1)
            vad_weights = vad.repeat(1, self.num_heads, logits.size(2), 1)
            logits = logits + self.W_vad * vad_weights

        if mask is not None:
            mask = mask.unsqueeze(1)
            logits = logits.masked_fill(mask, -1e18)

        attetion_weights = logits.sum(dim=1) / self.num_heads

        weights = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(weights)

        contexts = torch.matmul(weights, values)
        contexts = self._merge_heads(contexts)

        outputs = self.output_linear(contexts)

        if emotion_contexts is not None:
            emotion_contexts = emotion_contexts.unsqueeze(1).repeat(1, outputs.size(1), 1)
            outputs = torch.cat((outputs, emotion_contexts), dim=2)
            outputs = self.emotion_output_linear(outputs)

        return outputs, torch.softmax(attetion_weights, dim=-1)


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
        self,
        input_depth,
        filter_size,
        output_depth,
        layer_config="ll",
        padding="left",
        dropout=0.0,
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
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(float) * -log_timescale_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _get_attn_subsequent_mask(args, size):
    """
    Get an attention mask to avoid using the subsequent info.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if args.USE_CUDA:
        return subsequent_mask.to(args.device)
    else:
        return subsequent_mask


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def share_embedding(args, n_words, word2index, pretrain=True):
    embedding = Embeddings(n_words, args.emb_dim, padding_idx=args.PAD_idx)
    # real code loads a pretrained-vectors file from disk here when `pretrain` and
    # emb_dim in [50, 200, 300]; skipped (config default pretrain_emb=False), which is
    # a data/loading concern rather than an architecture change.
    return embedding


# ---- code/KEMP.py (real KEMP architecture, forward path verbatim) ----


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        args,
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
        concept=False,
    ):
        super(Encoder, self).__init__()
        self.args = args
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

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
        self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        x = x + self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

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
        args,
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
        universal=False,
    ):
        super(Decoder, self).__init__()
        self.args = args
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        self.mask = _get_attn_subsequent_mask(self.args, max_length)

        params = (
            args,
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.dec = nn.Sequential(*[DecoderLayer(*params) for _ in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_loss = nn.MSELoss()

    def forward(
        self,
        inputs,
        encoder_output,
        mask=None,
        pred_emotion=None,
        emotion_contexts=None,
        context_vad=None,
    ):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg.bool() + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)].bool(),
            0,
        )
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        x = x + self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

        y, _, pred_emotion, emotion_contexts, attn_dist, _ = self.dec(
            (x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src, dec_mask))
        )

        loss_att = 0.0
        if context_vad is not None:
            src_attn_dist = torch.mean(attn_dist, dim=1)
            loss_att = self.attn_loss(src_attn_dist, context_vad)

        y = self.layer_norm(y)

        return y, attn_dist, loss_att


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, args, d_model, vocab):
        super(Generator, self).__init__()
        self.args = args
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(self.args.hidden_dim, 1)

    def forward(
        self,
        x,
        pred_emotion=None,
        emotion_context=None,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
    ):
        if self.args.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        if emotion_context is not None:
            pred_emotion = pred_emotion.repeat(1, x.size(1), 1)
            x = torch.cat((x, pred_emotion), dim=2)
            logit = self.emo_proj(x)
        else:
            logit = self.proj(x)

        if self.args.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )

            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)

            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class KEMP(nn.Module):
    def __init__(self, args, vocab, decoder_number):
        super(KEMP, self).__init__()
        self.args = args
        self.vocab = vocab
        word2index, word2count, index2word, n_words = vocab
        self.word2index = word2index
        self.word2count = word2count
        self.index2word = index2word
        self.vocab_size = n_words

        self.embedding = share_embedding(args, n_words, word2index, self.args.pretrain_emb)
        self.encoder = Encoder(
            args,
            self.args.emb_dim,
            self.args.hidden_dim,
            num_layers=self.args.hop,
            num_heads=self.args.heads,
            total_key_depth=self.args.depth,
            total_value_depth=self.args.depth,
            max_length=args.max_seq_length,
            filter_size=self.args.filter,
            universal=self.args.universal,
        )

        ## GRAPH
        self.dropout = args.dropout
        self.W_q = nn.Linear(args.emb_dim, args.emb_dim)
        self.W_k = nn.Linear(args.emb_dim, args.emb_dim)
        self.W_v = nn.Linear(args.emb_dim, args.emb_dim)
        self.graph_out = nn.Linear(args.emb_dim, args.emb_dim)
        self.graph_layer_norm = LayerNorm(args.hidden_dim)

        ## emotional signal distilling
        self.identify = nn.Linear(args.emb_dim, decoder_number, bias=False)
        self.activation = nn.Softmax(dim=1)

        ## multiple decoders
        self.emotion_embedding = nn.Linear(decoder_number, args.emb_dim)
        self.decoder = Decoder(
            args,
            args.emb_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.hop,
            num_heads=args.heads,
            total_key_depth=args.depth,
            total_value_depth=args.depth,
            filter_size=args.filter,
            max_length=args.max_seq_length,
        )

        self.decoder_key = nn.Linear(args.hidden_dim, decoder_number, bias=False)
        self.generator = Generator(args, args.hidden_dim, self.vocab_size)
        if args.projection:
            self.embedding_proj_in = nn.Linear(args.emb_dim, args.hidden_dim, bias=False)
        if args.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

    def concept_graph(self, context, concept, adjacency_mask):
        """
        :param context: (bsz, max_context_len, embed_dim)
        :param concept: (bsz, max_concept_len, embed_dim)
        :param adjacency_mask: (bsz, max_context_len, max_context_len + max_concpet_len)
        """
        target = context
        src = torch.cat((target, concept), dim=1)

        q = self.W_q(target)
        k, v = self.W_k(src), self.W_v(src)
        attn_weights_ori = torch.bmm(q, k.transpose(1, 2))

        adjacency_mask = adjacency_mask.bool()
        attn_weights_ori = attn_weights_ori.masked_fill(adjacency_mask, 1e-24)
        attn_weights = torch.softmax(attn_weights_ori, dim=-1)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = self.graph_out(attn)

        attn = F.dropout(attn, p=self.dropout, training=self.training)
        new_context = self.graph_layer_norm(target + attn)

        new_context = torch.cat((new_context, concept), dim=1)
        return new_context

    def forward(self, batch):
        """Mirrors the real train_one_batch's pre-decoder + decoder + generator pipeline."""
        enc_batch = batch["context_batch"]
        enc_batch_extend_vocab = batch["context_ext_batch"]
        enc_vad_batch = batch["context_vad"]
        concept_input = batch["concept_batch"]
        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch["concept_vad_batch"]
        dec_batch = batch["target_batch"]

        mask_src = enc_batch.data.eq(self.args.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_context"])
        src_emb = self.embedding(enc_batch) + emb_mask
        src_vad = enc_vad_batch

        mask_con = concept_input.data.eq(self.args.PAD_idx).unsqueeze(1)
        con_mask = self.embedding(batch["mask_concept"])
        con_emb = self.embedding(concept_input) + con_mask

        ## Knowledge Update
        src_emb = self.concept_graph(src_emb, con_emb, batch["adjacency_mask_batch"])
        mask_src = torch.cat((mask_src, mask_con), dim=2)
        src_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)

        ## Encode - context & concept
        encoder_outputs = self.encoder(src_emb, mask_src)

        ## emotional signal distilling
        src_vad = torch.softmax(src_vad, dim=-1)
        emotion_context_vad = src_vad.unsqueeze(2)
        emotion_context_vad = emotion_context_vad.repeat(1, 1, self.args.emb_dim)
        emotion_context = torch.sum(emotion_context_vad * encoder_outputs, dim=1)
        _emotion_contexts = (
            emotion_context_vad * encoder_outputs
        )  # unused downstream, matches real train_one_batch

        emotion_logit = self.identify(emotion_context)

        ## Decode
        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)
        dec_emb = self.embedding(dec_batch[:, :-1])
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)

        mask_trg = dec_batch.data.eq(self.args.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist, _ = self.decoder(
            inputs=dec_emb,
            encoder_output=encoder_outputs,
            mask=(mask_src, mask_trg),
            pred_emotion=None,
            emotion_contexts=emotion_context,
            context_vad=src_vad,
        )

        enc_batch_extend_vocab = torch.cat((enc_batch_extend_vocab, concept_ext_input), dim=1)
        logit = self.generator(pre_logit, None, None, attn_dist, None, None)

        return logit, emotion_logit


def build_kemp():
    n_words = 60
    word2index = {str(i): i for i in range(n_words)}
    word2count = {str(i): 1 for i in range(n_words)}
    index2word = {str(i): str(i) for i in range(n_words)}
    vocab = (word2index, word2count, index2word, n_words)
    model = KEMP(args, vocab, decoder_number=32)
    model.eval()
    return model


def example_input_kemp():
    batch_size, ctx_len, con_len, tgt_len = 2, 6, 4, 5
    vocab_n = 60

    def toks(length):
        return torch.randint(2, vocab_n, (batch_size, length))

    context_batch = toks(ctx_len)
    concept_batch = toks(con_len)
    src_len = ctx_len + con_len
    batch = {
        "context_batch": context_batch,
        "context_ext_batch": context_batch.clone(),
        "context_vad": torch.rand(batch_size, ctx_len),
        "concept_batch": concept_batch,
        "concept_ext_batch": concept_batch.clone(),
        "concept_vad_batch": torch.rand(batch_size, con_len),
        "mask_context": torch.zeros(batch_size, ctx_len, dtype=torch.long),
        "mask_concept": torch.zeros(batch_size, con_len, dtype=torch.long),
        "adjacency_mask_batch": torch.zeros(batch_size, ctx_len, src_len, dtype=torch.bool),
        "target_batch": toks(tgt_len),
    }
    return (batch,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "KEMP (Knowledge-Enriched Emotional Dialogue / Knowledge Bridging)",
        "build_kemp",
        "example_input_kemp",
        2022,
        "vendored",
    ),
]
