# SOURCE: vendored from wangxr0526/Parrot @ master
# Files: models/model_layer.py (custom multi-head-attention-with-weights TransformerDecoder),
#        models/parrot_model.py (ParrotConditionModel, a real BertForSequenceClassification
#        subclass with a custom transformer decoder head for reaction-condition prediction).
# ParrotConditionModel's actual training wrapper (ParrotConditionPredictionModel) subclasses
# simpletransformers' SmilesClassificationModel and is not vendored (it is a training/data
# harness, not part of the traced architecture, and depends on rxnfp/simpletransformers which
# are not installed base libs). ParrotConditionModel itself only depends on transformers
# (BertModel/BertConfig/BertForSequenceClassification), which IS a base lib.
#
# IMPORT-LEVEL FIX ONLY (no architecture change): the original models/model_layer.py imports
# torch.nn.functional._scaled_dot_product_attention, a private PyTorch API removed in current
# torch (torch 2.8 here). It is replaced below with `_legacy_scaled_dot_product_attention`, a
# drop-in reimplementation of the exact same math the old private function performed (plain
# softmax attention that also returns the per-head attention weights, which the modern public
# `torch.nn.functional.scaled_dot_product_attention` does not expose) -- see PyTorch's own
# documented "efficient implementation equivalent" for scaled_dot_product_attention. This is a
# minimal compat shim for a deleted private symbol, not a change to Parrot's architecture.

import math
import warnings
from collections import defaultdict
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import _in_projection, _in_projection_packed, linear
from torch.nn.modules import TransformerDecoder as OrgTransformerDecoder
from torch.nn.modules import TransformerDecoderLayer as OrgTransformerDecoderLayer
from torch.nn.modules.activation import MultiheadAttention as _TorchMultiheadAttention
from torch.nn.modules.normalization import LayerNorm
from torch.overrides import handle_torch_function, has_torch_function
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel

MENAGERIE_ZOO = "vendored-pytorch"

BOS, EOS, PAD, MASK = "[BOS]", "[EOS]", "[PAD]", "[MASK]"


def _legacy_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p):
    """Drop-in replacement for the removed torch.nn.functional._scaled_dot_product_attention.

    Reproduces the same math (plain softmax attention, returning both the output and the
    per-head attention weights) that the deleted private PyTorch function performed.
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    output = torch.bmm(attn, v)
    return output, attn


# ---- models/model_layer.py ----
def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Modified from the pytorch source so that it can output the attention weights of multiple heads instead of the average.
    """

    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, (
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    )
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, (
        f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    )
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], (
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        )
    else:
        assert key.shape == value.shape, (
            f"key shape {key.shape} does not match value shape {value.shape}"
        )

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, (
            "use_separate_proj_weight is True but q_proj_weight is None"
        )
        assert k_proj_weight is not None, (
            "use_separate_proj_weight is True but k_proj_weight is None"
        )
        assert v_proj_weight is not None, (
            "use_separate_proj_weight is True but v_proj_weight is None"
        )
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v
        )

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, (
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            )
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, (
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        )
        assert static_k.size(2) == head_dim, (
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        )
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, (
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        )
        assert static_v.size(2) == head_dim, (
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        )
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), (
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        )
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _legacy_scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # output the attention weights for each head.
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class MultiheadAttention(_TorchMultiheadAttention):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class TransformerDecoderLayer(OrgTransformerDecoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=0.00001,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None,
        output_attention=False,
    ) -> None:
        self.output_attention = output_attention
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            sa_block_x, sa_block_attention = self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask
            )
            x = x + sa_block_x
            mha_block_x, mha_block_attention = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + mha_block_x
            x = x + self._ff_block(self.norm3(x))
        else:
            sa_block_x, sa_block_attention = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + sa_block_x)
            mha_block_x, mha_block_attention = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask
            )
            x = self.norm2(x + mha_block_x)
            x = self.norm3(x + self._ff_block(x))

        return x, {"cross_attn": mha_block_attention, "decoder_self_attn": sa_block_attention}

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x, attention_weight = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.output_attention,
        )
        return self.dropout1(x), attention_weight

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x, attention_weight = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.output_attention,
        )

        return self.dropout2(x), attention_weight


class TransformerDecoder(OrgTransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, output_attention=False):
        self.output_attention = output_attention
        super().__init__(decoder_layer, num_layers, norm)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        self.attention_weights = defaultdict(list)

        for mod in self.layers:
            output, attn_dict = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if self.output_attention:
                for key in attn_dict:
                    self.attention_weights[key].append(attn_dict[key])

        if self.norm is not None:
            output = self.norm(output)
        if self.output_attention:
            return output, self.attention_weights
        else:
            return output, None


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# ---- models/parrot_model.py ----
class ParrotConditionModel(BertForSequenceClassification):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        num_decoder_layers = config.num_decoder_layers
        nhead = config.nhead
        tgt_vocab_size = config.tgt_vocab_size
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        d_model = config.d_model
        self.use_temperature = config.use_temperature
        device = None
        dtype = None

        if hasattr(config, "output_attention"):
            self.output_attention = config.output_attention
        else:
            self.output_attention = False

        self.bert = BertModel(config)
        activation = F.relu
        layer_norm_eps = 1e-5
        factory_kwargs = {"device": device, "dtype": dtype}

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first=True,
            norm_first=False,
            **factory_kwargs,
            output_attention=self.output_attention,
        )
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size=d_model)
        self.positional_encoding = PositionalEncoding(emb_size=d_model, dropout=dropout)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, output_attention=self.output_attention
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=config.condition_label_mapping[1][PAD]
        )
        if self.use_temperature:
            self.memory_regression_layer = nn.Sequential(
                nn.Linear(self.config.max_position_embeddings * self.config.hidden_size, d_model),
                nn.ReLU(),
            )
            self.regression_layer1 = nn.Sequential(nn.Linear(d_model * 5, d_model), nn.ReLU())
            self.regression_layer2 = nn.Linear(2 * d_model, 1)
            self.reg_loss_fn = torch.nn.MSELoss()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        label_input=None,
        label_mask=None,
        label_padding_mask=None,
        labels=None,
        memory_key_padding_mask=None,
        temperature=None,
    ):
        if memory_key_padding_mask is None:
            memory_key_padding_mask = attention_mask == 0
        memory = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[
            0
        ]
        outs, attention_weights = self.decoder(
            self.positional_encoding(self.tgt_tok_emb(label_input)),
            memory,
            tgt_mask=label_mask,
            tgt_key_padding_mask=label_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(outs)

        labels_out = labels[:, 1:]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), labels_out.reshape(-1))

        if self.use_temperature:
            temp_memory = memory.reshape(
                -1, self.config.max_position_embeddings * self.config.hidden_size
            )
            temp_memory = self.memory_regression_layer(temp_memory)

            temp_out = outs[:, :-1, :]
            temp_out = temp_out.reshape(-1, temp_out.size(1) * temp_out.size(2))
            temp_out = self.regression_layer1(temp_out)

            temp_out = torch.cat([temp_memory, temp_out], dim=1)
            temp_out = self.regression_layer2(temp_out)

            loss_reg = self.reg_loss_fn(temp_out.reshape(-1), temperature.reshape(-1))

            return loss, logits, attention_weights, loss_reg, temp_out
        return loss, logits, attention_weights

    def encode(self, input_ids):
        return self.bert(input_ids)[0]

    def decode(self, tgt, memory, tgt_mask, memory_key_padding_mask):
        decoder_output, attention_weightes = self.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return decoder_output, attention_weightes

    def decode_temperature(self, memory, decoder_output):
        temp_memory = memory.reshape(
            -1, self.config.max_position_embeddings * self.config.hidden_size
        )
        temp_memory = self.memory_regression_layer(temp_memory)

        temp_out = decoder_output
        temp_out = temp_out.reshape(-1, temp_out.size(1) * temp_out.size(2))
        temp_out = self.regression_layer1(temp_out)

        temp_out = torch.cat([temp_memory, temp_out], dim=1)
        temp_out = self.regression_layer2(temp_out)
        return temp_out


def build_parrot():
    max_position_embeddings = 16
    hidden_size = 32
    condition_label_mapping = ({}, {PAD: 0, BOS: 1, EOS: 2, MASK: 3})
    config = BertConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=max_position_embeddings,
        num_labels=5,
    )
    config.num_decoder_layers = 2
    config.nhead = 4
    config.tgt_vocab_size = 8
    config.dim_feedforward = 64
    config.dropout = 0.1
    config.d_model = hidden_size
    config.use_temperature = False
    config.output_attention = True
    config.condition_label_mapping = condition_label_mapping
    return ParrotConditionModel(config)


def example_input_parrot():
    torch.manual_seed(0)
    batch = 2
    src_len = 16
    tgt_len = 5
    input_ids = torch.randint(0, 64, (batch, src_len))
    attention_mask = torch.ones(batch, src_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch, src_len, dtype=torch.long)
    label_input = torch.randint(0, 8, (batch, tgt_len))
    labels = torch.randint(0, 8, (batch, tgt_len + 1))
    label_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf")), diagonal=1)
    label_padding_mask = torch.zeros(batch, tgt_len, dtype=torch.bool)
    # positional order matches ParrotConditionModel.forward's signature
    return (
        input_ids,
        attention_mask,
        token_type_ids,
        label_input,
        label_mask,
        label_padding_mask,
        labels,
    )


MENAGERIE_ENTRIES = [
    ("Parrot_ConditionModel", "build_parrot", "example_input_parrot", "2023", "vendored-pytorch"),
]
