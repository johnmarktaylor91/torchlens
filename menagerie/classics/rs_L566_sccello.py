# SOURCE: vendored from DeepGraphLearning/scCello @ main
#   sccello/src/model_prototype_contrastive.py
#
# scCello: single-cell foundation model with a modified BERT encoder
# (PrototypeContrastiveModel -- custom embeddings that add legacy tf-class/tf-superclass/
# expbin embedding tables on top of standard word+position embeddings) plus a
# cell-ontology-aware contrastive pretraining objective. This staged module vendors the
# fine-tuning classification head `PrototypeContrastiveForSequenceClassification`, which
# is architecturally distinct from a stock BertModel (custom embeddings class swapped into
# the encoder) and is fully self-contained for a forward pass in eval mode: it never
# touches `sccello.src.cell_ontology` (only used by the pretraining MLM class's
# contrastive loss heads, which require prestored cell-ontology-graph pickles unavailable
# outside the repo's data pipeline) nor `wandb`/`ipdb`/`torch.distributed` (all gated
# behind `if self.training` logging branches). Imports fixed minimally: dropped
# `wandb`/`ipdb`/`torchmetrics.functional.classification`/`torch.distributed` (only used by
# the training-time logging branch, unreachable in eval mode) and the `sccello.src`
# package import.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEncoder,
    BertIntermediate,
    BertLayer,
    BertModel,
    BertOutput,
    BertPreTrainedModel,
    BertSelfAttention,
    BertSelfOutput,
)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        return first_token_tensor


class PrototypeContrastiveEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # legacy
        self.tf_class_embeddings = nn.Embedding(1, config.hidden_size, scale_grad_by_freq=True)
        self.tf_superclass_embeddings = nn.Embedding(1, config.hidden_size, scale_grad_by_freq=True)
        self.expbin_embeddings = nn.Embedding(
            1, config.hidden_size, scale_grad_by_freq=True, padding_idx=config.pad_token_id
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # legacy
        tf_class_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        tf_superclass_ids = torch.zeros(
            input_shape, dtype=torch.long, device=self.position_ids.device
        )
        expbin_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        tf_class_embeddings = self.tf_class_embeddings(tf_class_ids)
        tf_superclass_embeddings = self.tf_superclass_embeddings(tf_superclass_ids)
        expbin_embeddings = self.expbin_embeddings(expbin_ids)

        embeddings = (
            inputs_embeds + tf_class_embeddings + tf_superclass_embeddings + expbin_embeddings
        )
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PrototypeContrastiveSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)


class PrototypeContrastiveAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        nn.Module.__init__(self)
        self.self = PrototypeContrastiveSelfAttention(
            config, position_embedding_type=position_embedding_type
        )
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()


class PrototypeContrastiveLayer(BertLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PrototypeContrastiveAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = PrototypeContrastiveAttention(
                config, position_embedding_type="absolute"
            )
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


class PrototypeContrastiveEncoder(BertEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList(
            [PrototypeContrastiveLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False


class PrototypeContrastiveModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = PrototypeContrastiveEmbeddings(config)
        self.encoder = PrototypeContrastiveEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()


class PrototypeContrastiveOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.activations import ACT2FN

        class _PredictionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
                self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                assert isinstance(config.hidden_act, str)
                self.transform_act_fn = ACT2FN[config.hidden_act]
                self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.bias = nn.Parameter(torch.zeros(config.hidden_size))
                self.dense_2.bias = self.bias

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                hidden_states = self.dense_1(hidden_states)
                hidden_states = self.layer_norm(hidden_states)
                hidden_states = self.transform_act_fn(hidden_states)
                hidden_states = self.dense_2(hidden_states)
                return hidden_states

        self.predictions = _PredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return self.predictions(sequence_output)


class PrototypeContrastiveForSequenceClassification(BertPreTrainedModel):
    def __init__(
        self,
        config,
        data_source=None,
        normalize_flag=False,
        pass_cell_cls=False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = PrototypeContrastiveModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.total_logging_steps = config.total_logging_steps
        self.step_count = 0

        self.data_source = data_source
        self.normalize_flag = normalize_flag
        self.pass_cell_cls = pass_cell_cls
        if self.pass_cell_cls:
            self.cell_cls = PrototypeContrastiveOnlyMLMHead(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # [bsz, L, dim], [bsz, dim]

        pooled_output = outputs[1]
        if self.pass_cell_cls:
            pooled_output = self.cell_cls(pooled_output)
        if self.normalize_flag:
            pooled_output = F.normalize(pooled_output, p=2, dim=-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # NOTE: original repo logs to wandb here `if dist.get_rank() == 0 and self.training`;
        # dropped entirely (unreachable in eval mode, and torch.distributed is not
        # initialized outside a real training run).

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def build_sccello():
    config = BertConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
        pad_token_id=0,
        num_labels=5,
    )
    config.total_logging_steps = 100
    model = PrototypeContrastiveForSequenceClassification(
        config, data_source="test", normalize_flag=True
    )
    return model.eval()


def example_input_sccello():
    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(1, 256, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return (input_ids, attention_mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("scCello", "build_sccello", "example_input_sccello", 2024, MENAGERIE_ZOO),
]
