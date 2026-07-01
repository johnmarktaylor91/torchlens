# SOURCE: vendored from AlibabaResearch/DAMO-ConvAI (space-2 subtree) @ main
#
# SPACE-2's `UnifiedTransformer` (space/models/unified_transformer.py) is the real
# tree-structured semi-supervised contrastive pretraining backbone for task-oriented
# dialogue (embedder + stacked pre-LN-free transformer blocks + optional MLM/subspace/
# pool heads + supervised-contrastive loss), transcribed verbatim from the official
# DAMO-ConvAI/space-2 repo along with its module dependencies (`Embedder`, `Subspace`,
# `TransformerBlock`, `MultiheadAttention`, `FeedForward`, `SupConLoss`,
# `ModelBase`/`HParams`). `IntentUnifiedTransformer` (intent_unified_transformer.py)
# extends it with the intent-classification head used for the banking77/clinc/hwu
# intent-recognition finetuning task the "SPACE" queue entry lists. All files import
# only torch (+ argparse/json for the HParams cmdline-argument scaffolding, base
# stdlib); no architecture code was changed. `ModelBase`'s registry/argparse
# machinery and the `reader`/`generator`/`.cuda()` training-script plumbing are
# trimmed for the staging harness -- the traced call path is the model's real
# `.infer()` -> `_infer()` -> `_mask_encoder_network()` inference path, which only
# consumes plain tensors (no dataset reader needed for a forward pass).

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- space/modules/multihead_attention.py ---


class MultiheadAttention(nn.Module):
    """
    Multi head attention layer.
    """

    def __init__(self, hidden_dim, num_heads, dropout):
        assert hidden_dim % num_heads == 0
        super(MultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.linear_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        return

    def _split_heads(self, x, is_key=False):
        x = x.reshape(x.size(0), x.size(1), self.num_heads, self.head_dim)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), self.hidden_dim)
        return x

    def _attn(self, query, key, value, mask):
        scores = torch.matmul(query, key)
        scores = scores * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.num_heads, 1, 1)
            scores.masked_fill_(mask.bool(), float("-inf"))

        attn = self.softmax(scores)
        attn = self.dropout_layer(attn)

        if mask is not None:
            attn.masked_fill_(mask.bool(), 0.0)

        out = torch.matmul(attn, value)
        return out

    def forward(self, inp, mask=None, cache=None):
        """Forward process of self attention."""
        qkv = self.linear_qkv(inp)
        query, key, value = torch.split(qkv, self.hidden_dim, dim=2)

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        if cache is not None:
            if "key" in cache and "value" in cache:
                key = torch.cat([cache["key"], key], dim=3)
                value = torch.cat([cache["value"], value], dim=2)
            cache["key"] = key
            cache["value"] = value

        out = self._attn(query, key, value, mask)
        out = self._merge_heads(out)
        out = self.linear_out(out)
        return out


# --- space/modules/feedforward.py ---


class FeedForward(nn.Module):
    """
    Positional feed forward layer.
    """

    def __init__(self, hidden_dim, inner_dim, dropout):
        super(FeedForward, self).__init__()

        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.linear_hidden = nn.Sequential(nn.Linear(hidden_dim, inner_dim), nn.GELU())
        self.linear_out = nn.Linear(inner_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        return

    def forward(self, x):
        out = self.linear_hidden(x)
        out = self.dropout_layer(out)
        out = self.linear_out(out)
        return out


# --- space/modules/transformer_block.py ---


class TransformerBlock(nn.Module):
    """
    Transformer block module.
    """

    def __init__(self, hidden_dim, num_heads, dropout, attn_dropout, ff_dropout):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=attn_dropout
        )
        self.attn_norm = nn.LayerNorm(
            normalized_shape=hidden_dim, eps=1e-12, elementwise_affine=True
        )
        self.ff = FeedForward(hidden_dim=hidden_dim, inner_dim=4 * hidden_dim, dropout=ff_dropout)
        self.ff_norm = nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-12, elementwise_affine=True)
        self.dropout_layer = nn.Dropout(p=dropout)
        return

    def forward(self, inp, mask=None, cache=None):
        """Forward process on one transformer layer."""
        attn_out = self.attn(inp, mask, cache)
        attn_out = self.dropout_layer(attn_out)
        attn_out = self.attn_norm(attn_out + inp)

        ff_out = self.ff(attn_out)
        ff_out = self.dropout_layer(ff_out)
        ff_out = self.ff_norm(ff_out + attn_out)

        return ff_out


# --- space/modules/embedder.py ---


class Embedder(nn.Module):
    """
    Composite embedding layer.
    """

    def __init__(
        self,
        hidden_dim,
        num_token_embeddings,
        num_pos_embeddings,
        num_type_embeddings,
        num_turn_embeddings,
        padding_idx=None,
        dropout=0.1,
        pos_trainable=False,
    ):
        super(Embedder, self).__init__()

        self.token_embedding = nn.Embedding(num_token_embeddings, hidden_dim)
        self.pos_embedding = nn.Embedding(num_pos_embeddings, hidden_dim)
        self.pos_embedding.weight.requires_grad = pos_trainable
        self.type_embedding = nn.Embedding(num_type_embeddings, hidden_dim)
        self.turn_embedding = nn.Embedding(num_turn_embeddings, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.turn_embedding.weight)
        return

    def forward(self, token_inp, pos_inp=None, type_inp=None, turn_inp=None):
        embed = self.token_embedding(token_inp)
        if pos_inp is not None:
            embed += self.pos_embedding(pos_inp)
        if type_inp is not None:
            embed += self.type_embedding(type_inp)
        if turn_inp is not None:
            embed += self.turn_embedding(turn_inp)
        embed = self.dropout_layer(embed)
        return embed


# --- space/modules/subspace.py ---


class Subspace(nn.Module):
    """
    Subspace.
    """

    subspaces = ["D", "I", "S", "V", "DI", "IS", "SV", "DIS", "ISV", "DISV"]

    def __init__(self, hidden_dim, subspace_dim, trigger_subspaces):
        super(Subspace, self).__init__()

        self.trigger_subspaces = (
            trigger_subspaces.split(",") if trigger_subspaces else self.subspaces
        )
        self.trigger_indices = [
            self.subspaces.index(subspace) for subspace in self.trigger_subspaces
        ]
        self.hidden_dim = hidden_dim
        self.subspace_dim = subspace_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.subspace_dim * len(self.subspaces)), nn.Tanh()
        )

    def forward(self, x):
        out = self.projection(x)
        out = out.reshape(x.size(0), len(self.subspaces), self.subspace_dim)
        out = torch.cat([out[:, index : index + 1, :] for index in self.trigger_indices], dim=1)
        return out


# --- space/utils/criterions.py (SupConLoss only; constructed but not exercised by a
#     plain forward/infer pass) ---


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature


# --- space/models/model_base.py (ModelBase, trimmed of registry/argparse/cuda plumbing) ---


class ModelBase(nn.Module):
    """
    Basic model wrapper (registry/argparse/`.cuda()` plumbing trimmed for staging;
    `use_gpu`/`gpu`/`init_checkpoint` are still read from hparams as in the real code).
    """

    def __init__(self, hparams):
        super(ModelBase, self).__init__()
        self.init_checkpoint = hparams.init_checkpoint
        self.use_gpu = hparams.use_gpu
        self.gpu = hparams.gpu
        return

    def forward(self, inputs, is_training=False, with_label=False, data_file=None):
        if is_training:
            self.train()
        else:
            self.eval()

        outputs = self._forward(inputs, is_training, with_label=with_label)
        metrics = self._collect_metrics(inputs, outputs, with_label=with_label, data_file=data_file)

        return metrics

    def infer(self, inputs):
        self.eval()
        results = self._infer(inputs)
        results = {name: results[name].cpu().detach().numpy() for name in results}
        return results


# --- space/models/unified_transformer.py ---


class UnifiedTransformer(ModelBase):
    """
    Implement unified transformer.
    """

    def __init__(self, hparams, dtype="float32"):
        super(UnifiedTransformer, self).__init__(hparams)
        self.num_token_embeddings = hparams.num_token_embeddings
        self.num_pos_embeddings = hparams.num_pos_embeddings
        self.num_type_embeddings = hparams.num_type_embeddings
        self.num_turn_embeddings = hparams.num_turn_embeddings
        self.temperature = hparams.temperature
        self.hidden_dim = hparams.hidden_dim
        self.subspace_dim = hparams.subspace_dim
        self.num_heads = hparams.num_heads
        self.num_layers = hparams.num_layers
        self.padding_idx = hparams.padding_idx
        self.dropout = hparams.dropout
        self.embed_dropout = hparams.embed_dropout
        self.attn_dropout = hparams.attn_dropout
        self.ff_dropout = hparams.ff_dropout
        self.mlm_ratio = hparams.mlm_ratio
        self.pos_trainable = hparams.pos_trainable
        self.bidirectional_context = hparams.bidirectional_context
        self.label_smooth = hparams.label_smooth
        self.initializer_range = hparams.initializer_range
        self.token_loss = hparams.token_loss
        self.learning_method = hparams.learning_method
        self.trigger_subspaces = hparams.trigger_subspaces
        self.with_contrastive = hparams.with_contrastive
        self.with_project = hparams.with_project
        self.with_pool = hparams.with_pool
        self.with_mlm = hparams.with_mlm
        self.with_cls = hparams.with_cls
        self._dtype = dtype

        self.embedder = Embedder(
            self.hidden_dim,
            self.num_token_embeddings,
            self.num_pos_embeddings,
            self.num_type_embeddings,
            self.num_turn_embeddings,
            padding_idx=self.padding_idx,
            dropout=self.embed_dropout,
            pos_trainable=self.pos_trainable,
        )
        self.embed_layer_norm = nn.LayerNorm(
            normalized_shape=self.hidden_dim, eps=1e-12, elementwise_affine=True
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.dropout,
                    self.attn_dropout,
                    self.ff_dropout,
                )
                for _ in range(hparams.num_layers)
            ]
        )

        if self.with_mlm:
            self.mlm_transform = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(normalized_shape=self.hidden_dim, eps=1e-12, elementwise_affine=True),
            )
            self.mlm_bias = nn.Parameter(torch.zeros(self.num_token_embeddings))

        if self.with_project:
            self.subspace = Subspace(
                hidden_dim=self.hidden_dim,
                subspace_dim=self.subspace_dim,
                trigger_subspaces=self.trigger_subspaces,
            )

        if self.with_pool:
            self.pooler = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh())

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.bce_loss = nn.BCELoss(reduction="none")
        self.nll_loss = nn.NLLLoss(ignore_index=self.padding_idx, reduction="none")
        self.contrastive_loss = SupConLoss(temperature=self.temperature)
        self._create_parameters()

        self.max_grad_norm = hparams.max_grad_norm
        if self.max_grad_norm is not None:
            self.grad_clip = self.max_grad_norm
        else:
            self.grad_clip = None
        self.weight_decay = hparams.weight_decay
        return

    def _create_parameters(self):
        """Create model's paramters."""
        import numpy as np

        sequence_mask = np.tri(self.num_pos_embeddings, self.num_pos_embeddings, dtype=self._dtype)
        self.sequence_mask = torch.tensor(sequence_mask)
        return

    def _create_mask(self, input_mask, append_head=False, auto_regressive=False):
        seq_len = input_mask.shape[1]

        input_mask = input_mask.float()
        mask1 = input_mask.unsqueeze(-1).repeat(1, 1, seq_len)
        mask2 = mask1.permute(0, 2, 1)
        mask = mask1 * mask2

        if append_head:
            mask = torch.cat([mask[:, :1, :], mask], dim=1)
            mask = torch.cat([mask[:, :, :1], mask], dim=2)
            seq_len += 1

        if auto_regressive:
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            seq_mask = seq_mask.to(mask.device)
            mask = mask * seq_mask

        mask = 1 - mask
        return mask

    def _mask_encoder_network(
        self, input_token, input_mask, input_pos=None, input_type=None, input_turn=None
    ):
        mask_embed = self.embedder.token_embedding.weight[0]
        mask_embed = mask_embed.unsqueeze(0).unsqueeze(0)
        mask_embed = mask_embed.repeat(input_token.shape[0], 1, 1)
        input_embed = self.embedder(input_token, input_pos, input_type, input_turn)
        embed = torch.cat([mask_embed, input_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        mask = self._create_mask(
            input_mask, auto_regressive=not self.bidirectional_context, append_head=True
        )

        for layer in self.layers:
            embed = layer(embed, mask, None)

        latent_embed = embed[:, 0]
        enc_embed = embed[:, 1:]
        return latent_embed, enc_embed

    def _forward(self, inputs, is_training, with_label):
        raise NotImplementedError(
            "training-time `_forward` needs a dataset reader; use `_infer`/`infer`"
        )

    def _collect_metrics(self, inputs, outputs, with_label, data_file):
        raise NotImplementedError(
            "training-time `_collect_metrics` needs score matrices; use `_infer`/`infer`"
        )

    def _infer(self, inputs):
        """Real inference process of model."""
        results = {}

        latent_embed, enc_embed = self._mask_encoder_network(
            input_token=inputs["src_token"],
            input_mask=inputs["src_mask"],
            input_pos=inputs["src_pos"],
            input_type=inputs["src_type"],
            input_turn=inputs["src_turn"],
        )
        features = latent_embed
        if self.with_project:
            features = self.subspace(latent_embed).squeeze(1)
            features = F.normalize(features, dim=-1, p=2)
        results["features"] = features
        results["ids"] = inputs["ids"]

        return results


# --- space/models/intent_unified_transformer.py ---


class IntentUnifiedTransformer(UnifiedTransformer):
    """
    Implement intent unified transformer.
    """

    def __init__(self, hparams, dtype="float32"):
        super(IntentUnifiedTransformer, self).__init__(hparams, dtype=dtype)
        self.example = hparams.example
        self.num_intent = hparams.num_intent
        self.with_rdrop = hparams.with_rdrop
        self.kl_ratio = hparams.kl_ratio
        if self.example:
            self.loss_fct = nn.NLLLoss()
        else:
            self.intent_classifier = nn.Linear(self.hidden_dim, self.num_intent)
            self.loss_fct = nn.CrossEntropyLoss()
        return

    def _infer(self, inputs):
        """Real inference process of model (intent-classification variant)."""
        results = {}

        if self.with_cls:
            latent_embed, enc_embed = self._mask_encoder_network(
                input_token=inputs["src_token"],
                input_mask=inputs["src_mask"],
                input_pos=inputs["src_pos"],
                input_type=inputs["src_type"],
                input_turn=inputs["src_turn"],
            )
            features = latent_embed
        else:
            raise NotImplementedError(
                "encoder-decoder branch needs tgt_* inputs; not exercised here"
            )

        if self.with_project:
            features = self.subspace(features).squeeze(1)
        elif self.with_pool:
            features = self.pooler(features)

        if self.example:
            raise NotImplementedError(
                "example-driven inference needs an example bank; not exercised here"
            )
        else:
            intent_logits = self.intent_classifier(features)
            intent_probs = self.softmax(intent_logits)
            results["intent_probs"] = intent_probs

        return results


# --- staging harness (tiny sizes; not part of the real repo) ---


class _HParams(dict):
    """Minimal stand-in for space.args.HParams: attribute access over a dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


def _base_hparams():
    return _HParams(
        init_checkpoint=None,
        use_gpu=False,
        gpu=0,
        num_token_embeddings=64,
        num_pos_embeddings=32,
        num_type_embeddings=2,
        num_turn_embeddings=8,
        temperature=0.07,
        hidden_dim=16,
        subspace_dim=8,
        num_heads=2,
        num_layers=2,
        padding_idx=0,
        dropout=0.1,
        embed_dropout=0.0,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlm_ratio=0.1,
        pos_trainable=True,
        bidirectional_context=True,
        label_smooth=0.0,
        initializer_range=0.02,
        token_loss=False,
        learning_method="semi",
        trigger_subspaces="",
        with_contrastive=True,
        with_project=True,
        with_pool=False,
        with_mlm=False,
        with_cls=True,
        max_grad_norm=5.0,
        weight_decay=0.0,
    )


def _example_infer_inputs():
    B, L = 2, 10
    return {
        "src_token": torch.randint(1, 64, (B, L)),
        "src_mask": torch.ones(B, L, dtype=torch.long),
        "src_pos": torch.arange(L).unsqueeze(0).expand(B, L).clone(),
        "src_type": torch.zeros(B, L, dtype=torch.long),
        "src_turn": torch.zeros(B, L, dtype=torch.long),
        "ids": torch.arange(B),
    }


class _UnifiedTransformerInferWrapper(nn.Module):
    """Thin tensor-in/tensor-out wrapper around the real model's `_infer` path
    (UnifiedTransformer.forward has an incompatible (dict, bool, bool) signature for
    tracing; `_infer` is the real inference call HuggingFace-style modules expose via
    `.infer()`)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_token, src_mask, src_pos, src_type, src_turn):
        inputs = {
            "src_token": src_token,
            "src_mask": src_mask,
            "src_pos": src_pos,
            "src_type": src_type,
            "src_turn": src_turn,
            "ids": torch.arange(src_token.shape[0]),
        }
        out = self.model._infer(inputs)
        return out["features"]


class _IntentUnifiedTransformerInferWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_token, src_mask, src_pos, src_type, src_turn):
        inputs = {
            "src_token": src_token,
            "src_mask": src_mask,
            "src_pos": src_pos,
            "src_type": src_type,
            "src_turn": src_turn,
        }
        out = self.model._infer(inputs)
        return out["intent_probs"]


def build_space2_unified():
    hparams = _base_hparams()
    model = UnifiedTransformer(hparams)
    return _UnifiedTransformerInferWrapper(model)


def example_input_space2_unified():
    inp = _example_infer_inputs()
    return (inp["src_token"], inp["src_mask"], inp["src_pos"], inp["src_type"], inp["src_turn"])


def build_space2_intent():
    hparams = _base_hparams()
    # Real intent-finetuning scripts (scripts/{banking,clinc,hwu}/train.sh) set
    # WITH_PROJECT=false / WITH_CLS=true for the intent-classification head, unlike
    # the contrastive-pretraining defaults used by `build_space2_unified` above.
    hparams["with_project"] = False
    hparams["with_pool"] = False
    hparams["with_cls"] = True
    hparams["example"] = False
    hparams["num_intent"] = 12
    hparams["with_rdrop"] = False
    hparams["kl_ratio"] = 5.0
    model = IntentUnifiedTransformer(hparams)
    return _IntentUnifiedTransformerInferWrapper(model)


def example_input_space2_intent():
    inp = _example_infer_inputs()
    return (inp["src_token"], inp["src_mask"], inp["src_pos"], inp["src_type"], inp["src_turn"])


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "SPACE-2-UnifiedTransformer",
        "build_space2_unified",
        "example_input_space2_unified",
        2022,
        "vendored",
    ),
    (
        "SPACE-2-IntentUnifiedTransformer",
        "build_space2_intent",
        "example_input_space2_intent",
        2022,
        "vendored",
    ),
]
