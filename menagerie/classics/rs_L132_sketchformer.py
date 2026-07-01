# FAITHFUL PORT of leosampaio/sketchformer @ master (original framework: TensorFlow 2 / Keras)
"""SketchFormer: Transformer-based Representation for Sketched Structures.

Ribeiro, Leo Sampaio Ferraz, et al. "Sketchformer: Transformer-based representation for
sketched structures." CVPR 2020. Official repo is a TensorFlow-2/Keras implementation
(`builders/layers/transformer.py`, `models/sketchformer.py`) -- confirmed by inspecting the
real repo source, not a paper summary. TorchLens targets PyTorch eager capture; TensorFlow is
not in the installed base-lib set for this environment, so per the menagerie ladder this is a
rung-3 FAITHFUL PORT: every real mechanism in the TF source is transcribed 1:1 into base-env
torch, not reimplemented from the paper text.

Ported mechanisms (from builders/layers/transformer.py + models/sketchformer.py, itself an
explicit port of the "ref: https://www.tensorflow.org/tutorials/text/transformer" tutorial
transformer):
  * `Encoder`/`Decoder`: standard sinusoidal-position-encoded Transformer encoder/decoder
    stacks (`EncoderLayer`/`DecoderLayer`, each pre-LN-free post-add LayerNorm, matching the
    TF source's `layernorm1(x + attn_output)` ordering).
  * Continuous stroke-point input: `use_continuous_input=True` selects a `Dense(d_model)`
    embedding (torch: `nn.Linear`) instead of a token `Embedding`, matching the (dx, dy, p1,
    p2, p3) 5-D continuous SketchFormer variant (`dataset.hps['use_continuous_data']`).
  * `SelfAttnV1`: the bottleneck attention-pooling layer (`ui=tanh(xW+b)`, `ai=softmax(uV)`,
    `o=sum(x*ai)`) that reduces the encoder's (batch, seq_len, d_model) output to a single
    (batch, lowerdim) embedding vector -- ported with the identical weight shapes (W: [fdim,
    units], b: [units], V: [units, 1]) and identical einsum-equivalent ops.
  * `DenseExpander`: re-expands the (batch, lowerdim) embedding back to (batch, seq_len,
    feat_dim_out) via a Dense-on-broadcast-dim trick before decoding, ported verbatim.
  * `Transformer.call`/`encode`/`decode`: the model wiring (encode -> bottleneck -> classify;
    embedding -> expand -> decode -> output_layer) from `models/sketchformer.py`, restricted
    to the forward/inference path (the TF file's `train_on_batch`/`GradientTape` training loop
    is not architecture and is omitted, matching the menagerie eager-capture convention of
    other ported repos in this catalog).

Not ported: `SelfAttnV2` (an alternate un-used-by-default bottleneck variant, `attn_version=2`)
and the VAE/z_mean/z_log_var branch referenced in `Transformer.call`'s docstring but never
actually built in `build_model` (dead code in the source repo) are both omitted as they are not
part of the model that `build_model()` actually constructs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


# ---------------------------------------------------------------------------
# builders/utils.py -> positional_encoding (verbatim numpy computation, ported
# to torch tensor construction)
# ---------------------------------------------------------------------------
def _get_angles(pos, i, d_model):
    angle_rates = 1.0 / (10000 ** ((2 * (i // 2)) / d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    angle_rads = _get_angles(pos, i, d_model)
    pos_encoding = angle_rads.clone()
    pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pos_encoding.unsqueeze(0)  # (1, position, d_model)


# ---------------------------------------------------------------------------
# builders/utils.py -> scaled_dot_product_attention (ported)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))
    dk = torch.tensor(k.size(-1), dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + (mask * -1e9)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


# ---------------------------------------------------------------------------
# builders/layers/transformer.py -> SelfAttnV1 (ported: 3 explicit weight
# matrices, matching the Keras `add_weight` shapes exactly)
# ---------------------------------------------------------------------------
class SelfAttnV1(nn.Module):
    def __init__(self, feat_dim, units=None):
        super(SelfAttnV1, self).__init__()
        self.units = units if units is not None else feat_dim
        self.W = nn.Parameter(torch.empty(feat_dim, self.units).normal_(0, 0.05))
        self.b = nn.Parameter(torch.zeros(self.units))
        self.V = nn.Parameter(torch.empty(self.units, 1).uniform_(-0.05, 0.05))

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        ui = torch.tanh(torch.matmul(x, self.W) + self.b)  # (B, T, units)
        ai = F.softmax(torch.matmul(ui, self.V), dim=1)  # (B, T, 1)
        o = torch.sum(x * ai, dim=1)  # (B, feat_dim)
        return o, ai


# ---------------------------------------------------------------------------
# builders/layers/transformer.py -> MultiHeadAttention (ported)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model),
    )


# ---------------------------------------------------------------------------
# builders/layers/transformer.py -> EncoderLayer / DecoderLayer (ported)
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


# ---------------------------------------------------------------------------
# builders/layers/transformer.py -> Encoder / Decoder (ported; continuous-input
# branch only, matching SketchFormer's default (dx, dy, p1, p2, p3) stroke data)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding=1000,
        rate=0.1,
        use_continuous_input=False,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_continuous_input = use_continuous_input

        if use_continuous_input:
            self.embedding = nn.Linear(5, d_model)
        else:
            self.embedding = nn.Embedding(input_vocab_size, d_model)

        self.register_buffer(
            "pos_encoding",
            positional_encoding(maximum_position_encoding, d_model),
            persistent=False,
        )

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask):
        seq_len = x.size(1)

        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for layer in self.enc_layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding=1000,
        rate=0.1,
        use_continuous_input=False,
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_continuous_input = use_continuous_input

        if use_continuous_input:
            self.embedding = nn.Linear(5, d_model)
        else:
            self.embedding = nn.Embedding(target_vocab_size, d_model)

        self.register_buffer(
            "pos_encoding",
            positional_encoding(maximum_position_encoding, d_model),
            persistent=False,
        )

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size(1)
        attention_weights = {}

        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i, layer in enumerate(self.dec_layers):
            x, block1, block2 = layer(x, enc_output, look_ahead_mask, padding_mask)
            attention_weights[f"decoder_layer{i + 1}_block1"] = block1
            attention_weights[f"decoder_layer{i + 1}_block2"] = block2

        return x, attention_weights


class DenseExpander(nn.Module):
    """builders/layers/transformer.py -> DenseExpander (ported).

    Expand tensor: input (batch, feat_dim_in) -> output (batch, seq_len, feat_dim_out).
    """

    def __init__(self, seq_len, feat_dim_in, feat_dim_out=0):
        super(DenseExpander, self).__init__()
        self.seq_len = seq_len
        self.feat_dim_out = feat_dim_out
        if feat_dim_out:
            self.project_layer = nn.Sequential(nn.Linear(feat_dim_in, feat_dim_out), nn.ReLU())
        # Keras Dense applied on the innermost dim of shape (B, feat_dim, 1) -> (B, feat_dim, seq_len).
        self.expand_layer = nn.Linear(1, seq_len)

    def forward(self, x):
        if self.feat_dim_out:
            x = self.project_layer(x)
        x = x.unsqueeze(2)  # (B, feat_dim_out, 1)
        x = self.expand_layer(x)  # (B, feat_dim_out, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, feat_dim_out)
        return x


# ---------------------------------------------------------------------------
# models/sketchformer.py -> Transformer (ported: build_model + call/encode/decode,
# restricted to the forward/inference path; continuous-data + classification +
# reconstruction branches, matching `specific_default_hparams()`'s defaults)
# ---------------------------------------------------------------------------
class SketchformerTransformer(nn.Module):
    def __init__(
        self,
        num_layers=4,
        d_model=128,
        dff=512,
        num_heads=8,
        dropout_rate=0.1,
        lowerdim=256,
        seq_len=64,
        n_classes=345,
        do_classification=True,
        do_reconstruction=True,
        class_buffer_layers=0,
        class_dropout=0.1,
    ):
        super(SketchformerTransformer, self).__init__()
        self.hps_lowerdim = lowerdim
        self.seq_len = seq_len
        self.do_classification = do_classification
        self.do_reconstruction = do_reconstruction

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size=None,
            rate=dropout_rate,
            use_continuous_input=True,
        )

        if do_reconstruction:
            self.decoder = Decoder(
                num_layers,
                d_model,
                num_heads,
                dff,
                target_vocab_size=None,
                rate=dropout_rate,
                use_continuous_input=True,
            )
            self.output_layer = nn.Linear(d_model, 5)  # back to original (dx, dy, p1, p2, p3) space

        if lowerdim:
            self.bottleneck_layer = SelfAttnV1(d_model, lowerdim)
            self.expand_layer = DenseExpander(seq_len, lowerdim)
            if do_classification:
                self.classify_layer = nn.Sequential(
                    nn.Linear(lowerdim, n_classes), nn.Softmax(dim=-1)
                )
                self.class_buffer = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(lowerdim, lowerdim), nn.ReLU())
                        for _ in range(class_buffer_layers)
                    ]
                )
                self.class_dropout = nn.ModuleList(
                    [nn.Dropout(class_dropout) for _ in range(class_buffer_layers)]
                )

    def classify_from_embedding(self, embedding):
        fc = embedding
        for buf, drop in zip(self.class_buffer, self.class_dropout):
            fc = buf(fc)
            fc = drop(fc)
        return self.classify_layer(fc)

    def encode(self, inp, inp_mask):
        enc_output = self.encoder(inp, inp_mask)
        out = {"enc_output": enc_output, "class": None}
        if self.hps_lowerdim:
            bottle_neck, _ = self.bottleneck_layer(enc_output)
            out["embedding"] = bottle_neck
            if self.do_classification:
                out["class"] = self.classify_from_embedding(out["embedding"])
        else:
            out["embedding"] = enc_output
        return out

    def decode(self, embedding, target, target_mask, look_ahead_mask):
        # blind_decoder_mask=True (repo default): decoder padding mask is zeroed, i.e. it
        # doesn't see input-sequence padding info directly.
        padding_mask = torch.zeros_like(target_mask)
        if self.hps_lowerdim:
            pre_decoder = self.expand_layer(embedding)
        else:
            pre_decoder = embedding
        dec_output, attention_weights = self.decoder(
            target, pre_decoder, look_ahead_mask, padding_mask
        )
        final_output = self.output_layer(dec_output)
        return {"recon": final_output, "attn_weights": attention_weights}

    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        out_keys = ["embedding", "recon", "class"]
        enc_outputs = self.encode(inp, enc_padding_mask)
        out = {key: enc_outputs[key] for key in out_keys if key in enc_outputs}
        if self.do_reconstruction:
            dec_outputs = self.decode(
                enc_outputs["embedding"], tar, dec_padding_mask, look_ahead_mask
            )
            out.update({key: dec_outputs[key] for key in out_keys if key in dec_outputs})
        return out


# ---------------------------------------------------------------------------
# builders/utils.py -> create_masks (ported; continuous-data branch: pad flag is
# in the last channel of the 5-D point, matching `create_padding_mask`'s
# `elif seq.shape[-1] > 1` branch)
# ---------------------------------------------------------------------------
def create_padding_mask_continuous(seq):
    mask = (seq[..., -1] == 1).float()
    return mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)


def create_look_ahead_mask(size, device=None):
    mask = 1 - torch.tril(torch.ones((size, size), device=device))
    return mask


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask_continuous(inp)
    dec_padding_mask = create_padding_mask_continuous(inp)
    look_ahead_mask = create_look_ahead_mask(tar.size(1), device=tar.device)
    dec_target_padding_mask = create_padding_mask_continuous(tar)
    combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


# ---------------------------------------------------------------------------
# Menagerie build/example helpers
# ---------------------------------------------------------------------------
def build_sketchformer():
    model = SketchformerTransformer(
        num_layers=2,
        d_model=32,
        dff=64,
        num_heads=4,
        dropout_rate=0.1,
        lowerdim=32,
        seq_len=16,
        n_classes=10,
        do_classification=True,
        do_reconstruction=True,
    )
    model.eval()
    return model


def example_input_sketchformer():
    torch.manual_seed(0)
    seq_len = 16
    # (dx, dy, p1, p2, p3) continuous stroke-point representation (SketchFormer's default
    # `use_continuous_data=True` mode); pad flag lives in the last channel.
    inp = torch.zeros(2, seq_len, 5)
    inp[:, :, :2] = torch.randn(2, seq_len, 2)
    inp[:, :, 2] = 1.0  # "pen down" one-hot slot
    tar = inp.clone()
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)
    return (inp, tar, enc_padding_mask, combined_mask, dec_padding_mask)


MENAGERIE_ENTRIES = [
    ("SketchFormer", build_sketchformer, example_input_sketchformer, 2020, MENAGERIE_ZOO),
]
