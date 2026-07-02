# SOURCE: vendored from BenevolentAI/MolBERT @ main
# (https://github.com/BenevolentAI/MolBERT, `molbert/models/base.py` +
#  `molbert/models/smiles.py` + `molbert/tasks/tasks.py` + `molbert/tasks/heads.py` +
#  `molbert/utils/lm_utils.py`)
#
# MolBERT (Fabian et al., "Molecular representation learning with language models and
# domain-relevant auxiliary tasks", 2020) is a BERT-family masked-language model
# pretrained on SMILES strings with two auxiliary tasks (same-molecule-pair prediction
# and physicochemical-property regression). Architecture: a standard `BertModel` encoder
# (`SuperPositionalBertModel` below) whose only deviation from stock BERT is swapping the
# learned position embedding for a fixed sinusoidal one (`SuperPositionalBertEmbeddings`,
# copied from the real repo's `pytorch_transformers`-style `PositionalEmbedding`, itself
# transcribed from HF's now-removed `transfo_xl` model -- see below), wrapped by
# `FlexibleBertModel` which attaches a `nn.ModuleList` of task heads
# (`MaskedLMTask`/`IsSameTask`/`PhyschemTask`) on top of the shared encoder output.
# `SmilesMolbertModel.get_config`/`get_tasks` (verbatim below) is the concrete
# instantiation used for the real "smiles" training entrypoint.
#
# All class bodies (SuperPositionalEmbedding, SuperPositionalBertEmbeddings,
# SuperPositionalBertModel, FlexibleBertModel, BaseTask/MaskedLMTask/IsSameTask/
# PhyschemTask, IsSameHead/PhysChemHead, BertConfigExtras) are copied verbatim from the
# real repo files, with two narrow, load-bearing adaptations for transformers-version
# drift (the real repo pins an old `pytorch_transformers`/early-`transformers` release
# whose internal module layout has since moved):
#   1. Import paths: `transformers.modeling_bert.{BertEncoder,BertPooler,
#      BertLMPredictionHead}` -> `transformers.models.bert.modeling_bert.{...}` (same
#      classes, just relocated as transformers reorganized into per-model submodules).
#   2. `transformers.modeling_transfo_xl.PositionalEmbedding` was removed entirely when
#      HF deprecated/dropped the transfo_xl model family. Its real body (10 lines: a
#      registered `inv_freq` buffer + sinusoidal `forward`) is transcribed verbatim from
#      the last available transformers release that shipped it
#      (https://raw.githubusercontent.com/huggingface/transformers/v4.46.0/src/transformers/models/deprecated/transfo_xl/modeling_transfo_xl.py,
#      class `PositionalEmbedding`) since `SuperPositionalEmbedding` only overrides
#      `forward` and needs the base class's `__init__`/`inv_freq` buffer.
#   3. `BertModel.__call__` in the real repo's transformers version returns a bare
#      `(sequence_output, pooled_output)` tuple; the modern `transformers` (>=4.x)
#      `BertModel.forward` returns a `BaseModelOutputWithPoolingAndCrossAttentions` by
#      default. `SuperPositionalBertModel.forward` below passes `return_dict=False`
#      (a real, first-class `BertModel` kwarg -- not new behavior) to restore the
#      tuple-returning call convention `FlexibleBertModel.forward` expects, with zero
#      change to what tensors are computed.
# No layer, weight shape, or forward-pass mechanism was altered.

import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertLMPredictionHead,
    BertPooler,
    BertPreTrainedModel,
)

MENAGERIE_ZOO = "vendored-pytorch"


# --------------------------------------------------------------------------------------
# transformers.models.deprecated.transfo_xl.modeling_transfo_xl.PositionalEmbedding
# (verbatim, transcribed since the transfo_xl model was fully removed from modern
# transformers; see header note 2 above)
# --------------------------------------------------------------------------------------


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


# --------------------------------------------------------------------------------------
# molbert/models/base.py (verbatim)
# --------------------------------------------------------------------------------------


class SuperPositionalEmbedding(PositionalEmbedding):
    """
    Same as PositionalEmbedding in XLTransformer, BUT
    has a different handling of the batch dimension that avoids cumbersome dimension shuffling
    """

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = pos_emb.unsqueeze(0)
        if bsz is not None:
            pos_emb = pos_emb.expand(bsz, -1, -1)
        return pos_emb


class SuperPositionalBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings, BUT
    uses non-learnt (computed) positional embeddings
    """

    def __init__(self, config):
        super(SuperPositionalBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = SuperPositionalEmbedding(config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # do word embedding first to determine its type (float or half)
        words_embeddings = self.word_embeddings(input_ids)

        # if position_ids or token_type_ids were not provided, used defaults
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=words_embeddings.dtype, device=words_embeddings.device
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = words_embeddings
        position_embeddings = self.position_embeddings(position_ids, input_ids.size(0))
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SuperPositionalBertModel(nn.Module):
    """
    Same as BertModel, BUT
    uses SuperPositionalBertEmbeddings instead of BertEmbeddings

    NOTE: the real repo subclasses transformers' BertModel directly
    (``class SuperPositionalBertModel(BertModel)``). This staging module instead
    composes a BertEncoder/BertPooler pair directly (still the identical real HF
    classes) because modern ``BertModel.__init__`` performs extra setup
    (``self.embeddings = BertEmbeddings(config)`` plus internal bookkeeping) that
    subclassing would require re-deriving; the forward computation graph
    (embeddings -> encoder -> pooler) is byte-identical to the real
    ``SuperPositionalBertModel(BertModel)`` forward pass.
    """

    def __init__(self, config):
        super(SuperPositionalBertModel, self).__init__()
        self.config = config
        self.embeddings = SuperPositionalBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            extended_attention_mask.dtype
        ).min

        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask, return_dict=False
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class FlexibleBertModel(BertPreTrainedModel):
    """
    General BERT model with tasks to specify
    """

    def __init__(self, config, tasks: nn.ModuleList):
        super().__init__(config)
        self.bert = SuperPositionalBertModel(config)

        self.tasks = tasks

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )

        return {task.name: task(sequence_output, pooled_output) for task in self.tasks}


# --------------------------------------------------------------------------------------
# molbert/tasks/tasks.py + molbert/tasks/heads.py (verbatim)
# --------------------------------------------------------------------------------------


class IsSameHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_same_clf = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 2),
        )

    def forward(self, pooled_output):
        return self.is_same_clf(pooled_output)


class PhysChemHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.physchem_clf = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_physchem_properties),
        )

    def forward(self, pooled_output):
        return self.physchem_clf(pooled_output)


class BaseTask(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, sequence_output, pooled_output):
        raise NotImplementedError

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        raise NotImplementedError


class PhyschemTask(BaseTask):
    def __init__(self, name, config):
        super().__init__(name)
        self.loss = MSELoss()

        self.physchem_head = PhysChemHead(config)

    def forward(self, sequence_output, pooled_output):
        return self.physchem_head(pooled_output)

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        return self.loss(batch_predictions[self.name], batch_labels[self.name])


class MaskedLMTask(BaseTask):
    def __init__(self, name, config):
        super().__init__(name)
        self.loss = CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = config.vocab_size
        self.masked_lm_head = BertLMPredictionHead(config)

    def forward(self, sequence_output, pooled_output):
        return self.masked_lm_head(sequence_output)

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        return self.loss(
            batch_predictions["masked_lm"].view(-1, self.vocab_size),
            batch_labels["lm_label_ids"].view(-1),
        )


class IsSameTask(BaseTask):
    def __init__(self, name, config):
        super().__init__(name)
        self.loss = CrossEntropyLoss(ignore_index=-1)
        self.is_same_head = IsSameHead(config)

    def forward(self, sequence_output, pooled_output):
        return self.is_same_head(pooled_output)

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        return self.loss(batch_predictions[self.name].view(-1, 2), batch_labels[self.name].view(-1))


# --------------------------------------------------------------------------------------
# molbert/utils/lm_utils.py :: BertConfigExtras (verbatim, adapted to modern BertConfig
# kwarg name -- vocab_size_or_config_json_file was BertConfig's old positional arg name,
# renamed to vocab_size upstream; same field, same semantics)
# --------------------------------------------------------------------------------------


class BertConfigExtras(BertConfig):
    """
    Same as BertConfig, BUT
    adds any kwarg as a member field
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
        extra_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "num_physchem_properties",
                "named_descriptor_set",
                "is_same_smiles",
            }
        }
        super(BertConfigExtras, self).__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            **extra_kwargs,
        )

        for k, v in kwargs.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------------------
# molbert/models/smiles.py :: SmilesMolbertModel.get_config/get_tasks (verbatim logic,
# re-expressed as a plain builder function rather than reading from a PyTorch-Lightning
# hparams Namespace)
# --------------------------------------------------------------------------------------


def build_molbert():
    torch.manual_seed(0)

    # tiny config, matching SmilesMolbertModel.get_config()'s hparams.tiny branch
    config = BertConfigExtras(
        vocab_size=42,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_physchem_properties=5,
        named_descriptor_set="all",
        is_same_smiles=True,
    )

    tasks = nn.ModuleList(
        [
            MaskedLMTask(name="masked_lm", config=config),
            IsSameTask(name="is_same", config=config),
            PhyschemTask(name="physchem_props", config=config),
        ]
    )

    return FlexibleBertModel(config, tasks)


def example_input_molbert():
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 16, 42
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return (input_ids, token_type_ids, attention_mask)


MENAGERIE_ENTRIES = [
    ("MolBERT", build_molbert, example_input_molbert, 2020, MENAGERIE_ZOO),
]
