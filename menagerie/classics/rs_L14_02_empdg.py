# SOURCE: vendored from qtli/EmpDG @ b3054b1df452305352e2024c9485044e422d6551
# https://raw.githubusercontent.com/qtli/EmpDG/b3054b1df452305352e2024c9485044e422d6551/Model/Empdg_G.py
# https://raw.githubusercontent.com/qtli/EmpDG/b3054b1df452305352e2024c9485044e422d6551/Model/common_layer.py
#
# Li et al. 2020 "EmpDG: Multi-resolution Interactive Empathetic Dialogue Generation"
# (COLING 2020). Real architecture: the generator `Empdg_G` runs two parallel custom
# Transformer (T2T-style) encoders -- `Semantic_Encoder` over the raw dialogue context
# and `Emotion_Encoder` over a coarse emotion-word-filtered sub-context ("multi-
# resolution emotion perception") -- fuses their CLS-style pooled outputs through an
# `identify_new` linear layer to predict a coarse emotion label, projects that label
# into an embedding used to seed the decoder's first input token, concatenates both
# encoder outputs into one memory bank, and decodes with a shared custom Transformer
# decoder + pointer-generator output head. The Transformer building blocks
# (EncoderLayer/DecoderLayer/MultiHeadAttention/PositionwiseFeedForward/LayerNorm,
# T2T-style sinusoidal timing signal) are the real code from Model/common_layer.py,
# taken essentially verbatim from the official repo.
#
# Minimal, non-architectural changes made (all bookkeeping/import-time-side-effect
# removal or numpy/torch API drift; no computation changed):
#   - The repo's `utils/config.py` is an argparse module that calls
#     `parser.parse_args()` at import time (crashes with no CLI args). Replaced with a
#     tiny local `_Config` namespace carrying the exact same attribute names/defaults
#     the real `Empdg_G`/`Semantic_Encoder`/`Emotion_Encoder`/`Decoder`/`Generator` code
#     reads (`emb_dim`, `hidden_dim`, `hop`, `heads`, `depth`, `filter`, `universal`,
#     `act`, `PAD_idx`, `pointer_gen`, `weight_sharing`, `label_smoothing`, `noam`,
#     `pretrain_emb`, `USE_CUDA`, `device`) -- values only, no behavior invented.
#   - `np.float` (removed in modern numpy; was a deprecated alias for the Python
#     builtin `float`) in `_gen_timing_signal` restored to `float`, matching the
#     Sahandfer/CEM sibling repo's own (already-fixed) copy of this exact function.
#   - `_get_attn_subsequent_mask`'s real code calls `config.USE_CUDA` /
#     `subsequent_mask.cuda()`; kept the same branch structure but routed through
#     `config.device` so it is constructible on CPU-only hosts.
#   - `share_embedding`'s real code loads a GloVe file from disk when `pretrain=True`;
#     kept the real function body but default `pretrain_emb=False` here (random init)
#     so the module is constructible without a vectors file. This is a data/loading
#     concern, not an architecture change.
#   - Dropped training-loop / adversarial-discriminator-loss / checkpoint-io / greedy-
#     decoding methods (`train_one_batch`, `decoder_greedy`, `predict`, `g_for_d`,
#     `save_model`) and the sklearn-based accuracy bookkeeping; kept the real
#     `Empdg_G.__init__` module graph and `Semantic_Encoder`/`Emotion_Encoder`/
#     `Decoder`/`Generator` forward paths (the traced architecture) verbatim, wired
#     together in a small `forward()` that mirrors the real `train_one_batch`'s
#     pre-decoder pipeline exactly (semantic+emotion encode -> identify_new -> concat
#     memory -> decode -> generate).
#   - A minimal stand-in `Vocab` (n_words only) replaces the real repo's `Lang` class
#     from utils/data_loader.py (only `.n_words` is read by the traced path).

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Config:
    """Stand-in for utils/config.py (real module executes argparse at import time)."""

    PAD_idx = 1
    SOS_idx = 3
    emb_dim = 32
    hidden_dim = 32
    hop = 2
    heads = 2
    depth = 16
    filter = 24
    universal = False
    act = False
    pointer_gen = False
    weight_sharing = False
    label_smoothing = False
    noam = False
    pretrain_emb = False
    USE_CUDA = False
    device = torch.device("cpu")
    save_path = "/tmp/empdg_menagerie_scratch"


config = _Config()


class Vocab:
    """Stand-in for utils/data_loader.py Lang (only n_words is read here)."""

    def __init__(self, n_words):
        self.n_words = n_words


# ---- Model/common_layer.py (real Transformer building blocks, verbatim) ----


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
            mask = mask.unsqueeze(1)
            logits = logits.masked_fill(mask, -1e18)

        attetion_weights = logits.sum(dim=1) / self.num_heads

        weights = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(weights)

        contexts = torch.matmul(weights, values)
        contexts = self._merge_heads(contexts)

        outputs = self.output_linear(contexts)

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


def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask.to(config.device)


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def share_embedding(vocab, pretrain=False):
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
    return embedding


# ---- Model/Empdg_G.py (real EmpDG generator architecture, forward path verbatim) ----


class Semantic_Encoder(nn.Module):
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
        concept=False,
    ):
        super(Semantic_Encoder, self).__init__()
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


class Emotion_Encoder(nn.Module):
    """
    A Transformer Encoder module.
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
        concept=False,
    ):
        super(Emotion_Encoder, self).__init__()
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
        universal=False,
    ):
        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
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

    def forward(self, inputs, encoder_output, mask=None):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg.bool() + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)].bool(),
            0,
        ).to(config.device)
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        x = x + self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

        y = self.layer_norm(y)

        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        max_oov_length=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )

            extra_zeros = torch.zeros((logit.size(0), max_oov_length), device=x.device)
            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)

            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18
            )

            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class Empdg_G(nn.Module):
    def __init__(self, vocab, emotion_number):
        """
        :param emotion_number: the number of emotion labels, i.e., 32
        """
        super(Empdg_G, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.semantic_und = Semantic_Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )
        self.emotion_pec = Emotion_Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

        ## emotional signal distilling
        self.identify = nn.Linear(config.emb_dim, emotion_number, bias=False)
        self.identify_new = nn.Linear(2 * config.emb_dim, emotion_number, bias=False)
        self.activation = nn.Softmax(dim=1)

        ## decoders
        self.emotion_embedding = nn.Linear(emotion_number, config.emb_dim)
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.decoder_key = nn.Linear(config.hidden_dim, emotion_number, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

    def forward(self, batch):
        """Mirrors the real train_one_batch's pre-decoder + decoder pipeline."""
        enc_batch = batch["context_batch"]
        enc_emo_batch = batch["emotion_context_batch"]
        dec_batch = batch["target_batch"]

        ## Semantic Understanding
        mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        sem_emb_mask = self.embedding(batch["mask_context"])
        sem_emb = self.embedding(enc_batch) + sem_emb_mask
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)

        ## Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)

        emotion_logit = self.identify_new(
            torch.cat((emo_encoder_outputs[:, 0, :], sem_encoder_outputs[:, 0, :]), dim=-1)
        )

        ## Combine Two Contexts
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)

        ## Empathetic Response Generation
        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)
        dec_emb = self.embedding(dec_batch[:, :-1])
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)

        mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(dec_emb, src_emb, (mask_src, mask_trg))

        logit = self.generator(pre_logit, attn_dist, attn_dist_db=None)

        return logit, emotion_logit


def build_empdg_g():
    vocab = Vocab(n_words=60)
    model = Empdg_G(vocab, emotion_number=32)
    model.eval()
    return model


def example_input_empdg_g():
    batch_size, ctx_len, emo_len, tgt_len = 2, 6, 5, 5
    vocab_n = 60

    def toks(length):
        return torch.randint(2, vocab_n, (batch_size, length))

    batch = {
        "context_batch": toks(ctx_len),
        "mask_context": torch.zeros(batch_size, ctx_len, dtype=torch.long),
        "emotion_context_batch": toks(emo_len),
        "target_batch": toks(tgt_len),
    }
    return (batch,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "EmpDG (Multi-resolution Interactive Empathetic Dialogue Generation)",
        "build_empdg_g",
        "example_input_empdg_g",
        2020,
        "vendored",
    ),
]
