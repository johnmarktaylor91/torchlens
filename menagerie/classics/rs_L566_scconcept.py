# SOURCE: vendored from theislab/scConcept @ main
#   src/concept/model.py (ContrastiveModel + GeneEncoder + ContinuousValueEncoder +
#       PositionalEncoding + GeneExpressionDecoder)
#   src/concept/modules/transformer.py (TransformerEncoder)
#   src/concept/modules/flash_attention_layer.py (FlashTransformerEncoderLayer)
#   src/concept/modules/mha.py (FallbackMHA)
#
# scConcept: single/multi-species single-cell foundation model. A gene-token + continuous
# expression-value transformer encoder (contrastive + masked-value-prediction
# pretraining), with an optional FlashAttention fast path (`flash_attn`, an optional
# extra) that transparently falls back to `FallbackMHA` (real code path in the repo,
# taken automatically when `flash_attn` isn't installed -- exactly the code path
# exercised here). `ContrastiveModel` genuinely is a `lightning.LightningModule`
# subclass in the real repo; `lightning`/`wandb`/`torchmetrics`/`torch.distributed` are
# all importable in this environment so the class is vendored UNMODIFIED except for:
# (1) relative `.modules.*` imports rewritten to the local classes below (no `concept`
# package installed), (2) `pad_input`/`unpad_input` (only reachable inside the
# `flash_attention=True` branch of `_encode`, which this module never exercises) are not
# vendored to avoid pulling in `einops`-only-for-that-path plumbing that adds nothing
# when `flash_attention=False`.
import logging
import math
from collections import defaultdict
from typing import List, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torchmetrics.classification import BinaryAccuracy

logger = logging.getLogger(__name__)

FLASH_ATTN_AVAILABLE = False  # flash_attn is an optional extra; not installed here


# ---- src/concept/modules/mha.py -----------------------------------------------------
class FallbackMHA(nn.Module):
    """Subset of flash_attn.modules.mha.MHA with a compatible checkpoint schema."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if use_flash_attn:
            raise ValueError("FallbackMHA only supports use_flash_attn=False")
        if num_heads_kv not in (None, num_heads):
            raise NotImplementedError("FallbackMHA does not support grouped-query attention")
        if any(
            [
                dwconv,
                rotary_emb_dim != 0,
                use_alibi,
                window_size != (-1, -1),
                fused_bias_fc,
                return_residual,
                checkpointing,
                layer_idx is not None,
                rotary_emb_base != 10000.0,
                rotary_emb_scale_base is not None,
                rotary_emb_interleaved,
            ]
        ):
            raise NotImplementedError(
                "FallbackMHA only implements the attention features used in this project"
            )

        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        self.num_heads = num_heads
        self.num_heads_kv = num_heads if num_heads_kv is None else num_heads_kv
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout

        if self.cross_attn:
            kv_dim = 2 * self.head_dim * self.num_heads_kv
            self.Wq = nn.Linear(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.Wkv = nn.Linear(embed_dim, kv_dim, bias=qkv_proj_bias, **factory_kwargs)
        else:
            qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
            self.Wqkv = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        if mixer_subset is not None or inference_params is not None:
            raise NotImplementedError(
                "FallbackMHA only supports plain training/inference-free attention"
            )
        if x_kv is not None and not self.cross_attn:
            raise ValueError("x_kv is only supported for cross-attention")
        if cu_seqlens is not None or max_seqlen is not None:
            raise ValueError(
                "FallbackMHA only supports padded inputs without cu_seqlens/max_seqlen"
            )
        if kwargs:
            unexpected_args = ", ".join(sorted(kwargs))
            raise TypeError(f"FallbackMHA got unsupported arguments: {unexpected_args}")
        if x.dim() != 3:
            raise ValueError(
                "FallbackMHA expects padded batch-first inputs of shape (batch, seq, dim)"
            )

        batch_size, seqlen, _ = x.shape
        if self.cross_attn:
            kv_input = x if x_kv is None else x_kv
            if kv_input.dim() != 3 or kv_input.shape[0] != batch_size:
                raise ValueError("x_kv must have shape (batch, seq, dim)")
            kv_seqlen = kv_input.shape[1]
            q = self.Wq(x).view(batch_size, seqlen, self.num_heads, self.head_dim)
            kv = self.Wkv(kv_input).view(batch_size, kv_seqlen, 2, self.num_heads_kv, self.head_dim)
            k, v = kv.unbind(dim=2)
        else:
            kv_seqlen = seqlen
            qkv = self.Wqkv(x).view(batch_size, seqlen, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, kv_seqlen):
                raise ValueError("key_padding_mask must have shape (batch, seq)")
            attn_mask = torch.zeros(
                batch_size,
                1,
                1,
                kv_seqlen,
                dtype=q.dtype,
                device=q.device,
            )
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(q.dtype).min
            )

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, seqlen, self.embed_dim)
        out = self.out_proj(context)
        return out if not self.return_residual else (out, x)


def _resolve_mha_cls(use_flash_attn):
    if use_flash_attn:
        from flash_attn.modules.mha import MHA

        return MHA
    return FallbackMHA


# ---- src/concept/modules/flash_attention_layer.py ------------------------------------
class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    Modified from torch.nn.TransformerEncoderLayer to support FlashAttention (falls
    back to FallbackMHA when use_flash_attn=False)."""

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        use_flash_attn=True,
        device=None,
        dtype=None,
        norm_scheme="pre",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        mha_cls = _resolve_mha_cls(use_flash_attn)
        self.use_flash_attn = use_flash_attn
        self.self_attn = mha_cls(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            **factory_kwargs,
        )
        if not hasattr(self.self_attn, "batch_first"):
            self.self_attn.batch_first = batch_first
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if self.norm_scheme == "pre":
            attn_src = self.norm1(src)
            src2 = self._self_attention(
                attn_src,
                key_padding_mask=key_padding_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            src = src + self.dropout1(src2)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src2)
        else:
            src2 = self._self_attention(
                src, key_padding_mask=key_padding_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
        return src

    def _self_attention(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[Tensor] = None,
    ) -> Tensor:
        if self.use_flash_attn:
            flash_key_padding_mask = None if key_padding_mask is None else ~key_padding_mask
            return self.self_attn(
                src,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                key_padding_mask=flash_key_padding_mask,
            )
        return self.self_attn(src, key_padding_mask=key_padding_mask)


# ---- src/concept/modules/transformer.py ----------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        import copy

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[Tensor] = None,
        key_padding_mask=None,
        is_causal=False,
    ):
        output = src
        for mod in self.layers:
            output = mod(
                output,
                mask,
                is_causal=is_causal,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                key_padding_mask=key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


# ---- src/concept/model.py -------------------------------------------------------------
class GeneEncoder(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict,
        dim_gene_embs: int,
        dim_model: int,
        padding_idx: Optional[int] = None,
        pretrained_vocabularies: Optional[dict] = None,
        freeze_pretrained_vocabulary: bool = None,
    ):
        super().__init__()
        self.dim_gene_embs = dim_gene_embs
        self.dim_model = dim_model
        self.pretrained_vocabulary_available = pretrained_vocabularies is not None

        self.learnable_embs = nn.ModuleDict()

        pretrained_dim = None
        if self.pretrained_vocabulary_available:
            assert freeze_pretrained_vocabulary is not None, (
                "freeze_pretrained_vocabulary must be provided if pretrained_vocabularies is provided"
            )
            self.pretrained_embs = nn.ModuleDict()

            for species, n_genes in vocab_sizes.items():
                pretrained_vocab = pretrained_vocabularies[species]
                assert pretrained_vocab.shape[0] == n_genes
                if pretrained_dim is None:
                    pretrained_dim = pretrained_vocab.shape[1]
                else:
                    assert pretrained_vocab.shape[1] == pretrained_dim

                self.pretrained_embs[species] = nn.Embedding.from_pretrained(
                    pretrained_vocab, freeze=freeze_pretrained_vocabulary, padding_idx=padding_idx
                )
                self.learnable_embs[species] = nn.Embedding(
                    n_genes,
                    dim_gene_embs,
                    padding_idx=padding_idx,
                    _weight=torch.zeros(n_genes, dim_gene_embs, dtype=torch.float),
                )
            self.adapter1 = nn.Linear(pretrained_dim, dim_gene_embs, bias=True)
        else:
            for species, n_genes in vocab_sizes.items():
                self.learnable_embs[species] = nn.Embedding(
                    n_genes, dim_gene_embs, padding_idx=padding_idx
                )
            self.adapter1 = None

        self.adapter2 = nn.Linear(dim_gene_embs, dim_model, bias=True)
        self.enc_norm = nn.LayerNorm(dim_model)

    def forward(self, x: Tensor, species: str, add_learnable_embs: bool = False) -> Tensor:
        x_learnable = self.learnable_embs[species](x)
        if self.pretrained_vocabulary_available:
            x_pretrained = self.adapter1(self.pretrained_embs[species](x))
            x = x_pretrained + x_learnable * int(add_learnable_embs)
        else:
            x = x_learnable
        x = self.adapter2(x)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x.float().unsqueeze(-1)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
        if x.dim() == 3:
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len, :].to(x.device)
        elif x.dim() == 2:
            assert seqlens is not None, "Sequence lengths must be provided for 2D input tensor."
            cum = seqlens.cumsum(0)
            offsets = torch.repeat_interleave(cum - seqlens, seqlens, output_size=x.size(0))
            indices = torch.arange(x.size(0), device=x.device) - offsets
            x = x + self.pe[0, indices, :].to(x.device)

        return x


class GeneExpressionDecoder(nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.Linear(dim_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).squeeze(-1)


class ContrastiveModel(L.LightningModule):
    """Lightning module implementing scConcept pretraining and adaptation workflows.

    The model combines gene/value encoders with a transformer backbone and supports
    contrastive and MLM-style objectives used for single-cell representation learning.
    """

    def __init__(
        self,
        config,
        pad_token_id: int,
        cls_token_id: int,
        vocab_sizes: dict,
        pretrained_vocabularies: Optional[dict] = None,
        precomp_embs_key: str = None,
        world_size: int = 1,
        val_loader_names=[],
        obs_keys: List[str] = [],
        debug: bool = False,
    ):
        if config["mlm_loss_weight"] > 0:
            assert config["decoder_head"] == True, "Decoder head must be enabled for MLM loss"  # noqa: E712

        super().__init__()
        self.debug = debug
        self.flash_attention = config["flash_attention"]
        if self.flash_attention and not FLASH_ATTN_AVAILABLE:
            logger.warning(
                "flash_attention=True requires the optional 'flash_attn' package to be installed. "
                "Falling back to non flash implementation."
            )
            self.flash_attention = False
        self.dim_gene_embs = config.get("dim_gene_embs", config["dim_model"])
        self.dim_model = config["dim_model"]
        self.dim_hid = config["dim_hid"]
        self.num_head = config["num_head"]
        self.nlayers = config["nlayers"]
        self.dropout = config["dropout"]
        self.decoder_head = config["decoder_head"]
        self.input_encoding = config["input_encoding"]
        self.MASK_VALUE = config["mask_value"]
        self.CLS_VALUE = config["cls_value"]
        self.PAD_TOKEN_ID = pad_token_id
        self.CLS_TOKEN_ID = cls_token_id
        self.masking_rate = config["training"]["masking_rate"]
        self.lr = config["training"]["lr"]
        self.weight_decay = config["training"]["weight_decay"]
        self.optimizer_class = config["training"]["optimizer_class"]
        self.scheduler = config["training"]["scheduler"]
        self.warmup = config["training"]["warmup"]
        self.max_steps = config["training"]["max_steps"]
        self.min_lr = config["training"]["min_lr"]
        self.log_every_n_steps = config["training"].get("log_every_n_steps", 100)
        self.values_only_sanity_check = config["values_only_sanity_check"]
        self.data_loading_speed_sanity_check = config["data_loading_speed_sanity_check"]
        self.norm_scheme = config.get("norm_scheme", "post")
        self.activation = config.get("activation", "relu")
        self.mlm_loss_weight = config["mlm_loss_weight"]
        self.cont_loss_weight = config["cont_loss_weight"]
        self.contrastive_loss = config["contrastive_loss"]
        self.loss_switch_step = config["loss_switch_step"]
        self.logit_scale_init_value = config["logit_scale_init_value"]
        self.projection_dim = config["projection_dim"]
        self.pe_max_len = config["pe_max_len"]
        self.precomp_embs_key = precomp_embs_key
        self.world_size = world_size
        self.val_loader_names = val_loader_names
        self.obs_keys = list(obs_keys)
        assert self.contrastive_loss in ["binary", "multiclass"]
        self.LOGGING_STEP = False

        self.use_learnable_embs_freq = config["training"]["use_learnable_embs_freq"]
        freeze_pretrained = config["training"]["freeze_pretrained_vocabulary"]

        self.gene_token_encoder = GeneEncoder(
            vocab_sizes,
            self.dim_gene_embs,
            self.dim_model,
            padding_idx=pad_token_id,
            pretrained_vocabularies=pretrained_vocabularies,
            freeze_pretrained_vocabulary=freeze_pretrained,
        )

        self.active_species: Optional[str] = None

        self.cls_embedding = nn.Parameter(torch.zeros(self.dim_model))

        if self.input_encoding == "value_encoding":
            self.value_encoder = ContinuousValueEncoder(self.dim_model, dropout=0.0)
        elif self.input_encoding == "rank_encoding":
            self.positional_encoder = PositionalEncoding(self.dim_model, max_len=self.pe_max_len)

        encoder_layers = FlashTransformerEncoderLayer(
            self.dim_model,
            self.num_head,
            self.dim_hid,
            self.dropout,
            batch_first=True,
            use_flash_attn=self.flash_attention,
            norm_scheme=self.norm_scheme,
            activation=self.activation,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

        if self.decoder_head:
            self.expression_decoder = GeneExpressionDecoder(self.dim_model)

        self.binarcy_accuracy = BinaryAccuracy()
        self.logit_scale = nn.Parameter(
            torch.tensor(float(self.logit_scale_init_value)), requires_grad=True
        )
        if self.projection_dim:
            self.projection = nn.Linear(self.dim_model, self.projection_dim, bias=False)

        self.sample_stats = {"train": [], "val": defaultdict(list)}
        self.logit_masks = {}
        self.stage = None

        if config["training"].get("train_vocab_only", None):
            self.requires_grad_(False)
            for learnbale_embs in self.gene_token_encoder.learnable_embs.values():
                learnbale_embs.weight.requires_grad_(True)
            self.use_learnable_embs_freq = 1.0

    def _encode(
        self,
        tokens: Tensor,
        values: Tensor,
        seq_lengths: List[int] = None,
        return_padded_embeddings: bool = False,
    ) -> Tensor:
        src_key_padding_mask = tokens == self.PAD_TOKEN_ID

        # NOTE: flash_attention branch (unpad_input/pad_input) intentionally not
        # vendored -- self.flash_attention is always False in this staged module
        # (flash_attn is an optional extra, not installed).
        gene_embs = self._encode_gene_tokens(tokens)
        gene_embs[:, 0, :] = self.cls_embedding.to(dtype=gene_embs.dtype)
        if self.input_encoding == "rank_encoding":
            total_embs = self.positional_encoder(gene_embs)
        else:
            value_embs = self._encode_values(values)
            total_embs = gene_embs + value_embs

        embs_padded = self.transformer_encoder(total_embs, key_padding_mask=src_key_padding_mask)
        cell_embs = embs_padded[:, 0, :]

        return embs_padded, cell_embs

    def forward(self, input_tokens, input_values, seq_lengths: List[int] = None):
        embs_padded, cell_embs = self._encode(
            input_tokens,
            input_values,
            seq_lengths,
            return_padded_embeddings=self.decoder_head is True,
        )
        pred = self.expression_decoder(embs_padded) if self.decoder_head else None

        return pred, embs_padded, cell_embs

    def set_active_species(self, species: str) -> None:
        self.active_species = species

    def _encode_gene_tokens(self, tokens: Tensor) -> Tensor:
        if self.stage is None:
            self.stage = "predict"

        assert self.active_species is not None, (
            "active_species must be set via set_active_species() before encoding."
        )
        species = self.active_species

        add_learnable_embs = False
        if (
            self.gene_token_encoder.pretrained_vocabulary_available
            and self.use_learnable_embs_freq is not None
        ):
            if self.stage == "train" and not self.LOGGING_STEP:
                add_learnable_embs = int(
                    (self.global_step + 1) * self.use_learnable_embs_freq
                ) > int(self.global_step * self.use_learnable_embs_freq)
            elif self.stage == "predict":
                add_learnable_embs = bool(self.use_learnable_embs_freq)
            else:
                add_learnable_embs = True

        return self.gene_token_encoder(tokens, species, add_learnable_embs=add_learnable_embs)

    def _encode_values(self, values: Tensor) -> Tensor:
        return self.value_encoder(values)


def build_scconcept():
    config = {
        "flash_attention": False,
        "dim_model": 32,
        "dim_hid": 64,
        "num_head": 4,
        "nlayers": 2,
        "dropout": 0.0,
        "decoder_head": True,
        "input_encoding": "value_encoding",
        "mask_value": -1.0,
        "cls_value": 0.0,
        "mlm_loss_weight": 0.0,
        "cont_loss_weight": 1.0,
        "contrastive_loss": "binary",
        "loss_switch_step": 0,
        "logit_scale_init_value": 2.6592,
        "projection_dim": None,
        "pe_max_len": 256,
        "values_only_sanity_check": False,
        "data_loading_speed_sanity_check": False,
        "training": {
            "masking_rate": 0.15,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "optimizer_class": torch.optim.AdamW,
            "scheduler": None,
            "warmup": 0,
            "max_steps": 100,
            "min_lr": 1e-5,
            "use_learnable_embs_freq": None,
            "freeze_pretrained_vocabulary": None,
        },
    }
    model = ContrastiveModel(
        config,
        pad_token_id=0,
        cls_token_id=1,
        vocab_sizes={"human": 64},
    )
    model.set_active_species("human")
    model.stage = "predict"
    return model.eval()


def example_input_scconcept():
    batch_size = 4
    seq_len = 12
    tokens = torch.randint(1, 64, (batch_size, seq_len))
    tokens[:, 0] = 1  # CLS token
    values = torch.rand(batch_size, seq_len)
    return (tokens, values)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("scConcept", "build_scconcept", "example_input_scconcept", 2026, MENAGERIE_ZOO),
]
