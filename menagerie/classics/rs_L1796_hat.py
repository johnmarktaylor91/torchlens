# SOURCE: vendored from coastalcph/hierarchical-transformers @ main
# https://raw.githubusercontent.com/coastalcph/hierarchical-transformers/main/models/hat/modelling_hat.py
# https://raw.githubusercontent.com/coastalcph/hierarchical-transformers/main/models/hat/configuration_hat.py
#
# Chalkidis, Dai, Fergadiotis, Malakasiotis, Elliott 2022 (EMNLP Findings) "An Exploration
# of Hierarchical Attention Transformers for Efficient Long Document Classification" --
# HAT (Hierarchical Attention Transformer). A RoBERTa-style token embedding feeds a stack
# of `HATLayer`s; each layer optionally runs (a) a SENTENCE encoder (a standard
# `RobertaAttention`+`RobertaIntermediate`+`RobertaOutput` transformer block, run
# independently per-sentence after `transform_tokens2sentences` reshapes the flat token
# sequence into `[batch*num_sentences, sentence_length, hidden]` segments) and/or (b) a
# DOCUMENT encoder (the same transformer-block machinery, but run over the sequence of
# per-sentence "representative tokens" -- i.e. token 0 of every sentence segment -- with a
# learned sentence-position embedding added first). After the document encoder, the
# updated representative tokens are written back into the flat token sequence, so
# document-level context propagates back down into the token stream for the next layer --
# the alternating sentence/document attention (controlled per-layer by
# `config.encoder_layout[str(idx)]`) that gives the architecture its "hierarchical
# attention" name. `HATForSequenceClassification` pools the final representative tokens
# (`HATPooler`, default max-pooling over sentence positions) into a document embedding and
# applies a linear classifier head.
#
# `HATConfig`, `HATEmbeddings`, `transform_tokens2sentences`, `transform_masks2sentences`,
# `transform_sentences2tokens`, `HATLayer`, `TransformerLayer`, `HATEncoder`,
# `HATPreTrainedModel`, `HATPooler`, `HATModel`, `HATForSequenceClassification`,
# `create_position_ids_from_input_ids` are copied verbatim from the real
# `modelling_hat.py`/`configuration_hat.py` (only the unused pretraining heads --
# `HATForMaskedLM`, BoW/VICReg/SimCLR pretraining variants, token-classification,
# QA, multiple-choice -- and the doc-string decorators are dropped; the sentence/document
# hierarchical attention mechanism itself is untouched). `RobertaAttention`,
# `RobertaIntermediate`, `RobertaOutput` are imported directly from the installed
# `transformers` library, exactly as the real file does. No architectural changes.
#
# `max_sentence_length` is not a `HATConfig.__init__` parameter -- the real repo's
# `convert_bert_to_htf.py`/`convert_roberta_to_htf.py` set it as an ad-hoc post-construction
# attribute (`htf_config.max_sentence_length = MAX_SENTENCE_LENGTH`), which `PretrainedConfig`
# permits since it stores unknown kwargs as plain attributes; published HF configs
# (e.g. `kiddothe2b/hierarchical-transformer-base-4096`) confirm both `max_sentence_length`
# and `max_sentence_size` are set (same value). We reproduce that exact real-world
# construction pattern here rather than inventing a different config surface.

import torch
import torch.utils.checkpoint
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
)
from transformers import PretrainedConfig


def transform_tokens2sentences(hidden_states, num_sentences, max_sentence_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(
        hidden_states,
        (hidden_states.size(0), num_sentences, max_sentence_length, hidden_states.size(-1)),
    )
    # squash segments into sequence into a single axis (samples * segments, max_segment_length, hidden_size)
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_sentences, max_sentence_length, seg_hidden_states.size(-1)
    )

    return hidden_states_reshape


def transform_masks2sentences(hidden_states, num_sentences, max_sentence_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(
        hidden_states, (hidden_states.size(0), 1, 1, num_sentences, max_sentence_length)
    )
    # squash segments into sequence into a single axis (samples * segments, 1, 1, max_segment_length)
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_sentences, 1, 1, seg_hidden_states.size(-1)
    )

    return hidden_states_reshape


def transform_sentences2tokens(seg_hidden_states, num_sentences, max_sentence_length):
    # transform squashed sequence into segments
    hidden_states = seg_hidden_states.contiguous().view(
        seg_hidden_states.size(0) // num_sentences,
        num_sentences,
        max_sentence_length,
        seg_hidden_states.size(-1),
    )
    # transform segments into sequence
    hidden_states = hidden_states.contiguous().view(
        hidden_states.size(0), num_sentences * max_sentence_length, hidden_states.size(-1)
    )
    return hidden_states


@dataclass
class BaseModelOutputWithSentenceAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    sentence_attentions: Optional[Tuple[torch.FloatTensor]] = None


class HATConfig(PretrainedConfig):
    """Real HATConfig from configuration_hat.py, copied verbatim."""

    model_type = "hierarchical-transformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        max_sentences=64,
        max_sentence_size=128,
        model_max_length=8192,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        encoder_layout=None,
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sentences = max_sentences
        self.max_sentence_size = max_sentence_size
        self.model_max_length = model_max_length
        self.encoder_layout = encoder_layout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class HATEmbeddings(nn.Module):
    """Same as BertEmbeddings with a tiny tweak for positional embeddings indexing."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.position_embeddings = nn.Embedding(
            config.max_sentence_length + self.padding_idx + 1,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            torch.arange(self.padding_idx + 1, config.max_sentence_length + self.padding_idx + 1)
            .repeat(config.max_sentences)
            .expand((1, -1)),
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, self.position_ids
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class HATLayer(nn.Module):
    def __init__(self, config, use_sentence_encoder=True, use_document_encoder=True):
        super().__init__()
        self.max_sentence_length = config.max_sentence_length
        self.max_sentences = config.max_sentences
        self.hidden_size = config.hidden_size
        self.use_document_encoder = use_document_encoder
        self.use_sentence_encoder = use_sentence_encoder
        if self.use_sentence_encoder:
            self.sentence_encoder = TransformerLayer(config)
        if self.use_document_encoder:
            self.document_encoder = TransformerLayer(config)
            self.position_embeddings = nn.Embedding(
                config.max_sentences + 1, config.hidden_size, padding_idx=config.pad_token_id
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_sentences=None,
        output_attentions=False,
    ):
        sentence_outputs = (None, None)
        if self.use_sentence_encoder:
            # transform sequences to sentences
            sentence_inputs = transform_tokens2sentences(
                hidden_states,
                num_sentences=num_sentences,
                max_sentence_length=self.max_sentence_length,
            )
            sentence_masks = transform_masks2sentences(
                attention_mask,
                num_sentences=num_sentences,
                max_sentence_length=self.max_sentence_length,
            )

            sentence_outputs = self.sentence_encoder(
                sentence_inputs, sentence_masks, output_attentions=output_attentions
            )

            # transform sentences to tokens
            outputs = transform_sentences2tokens(
                sentence_outputs[0],
                num_sentences=num_sentences,
                max_sentence_length=self.max_sentence_length,
            )
        else:
            outputs = hidden_states

        document_outputs = (None, None)
        if self.use_document_encoder:
            # gather sentence representative tokens
            sentence_global_tokens = outputs[:, :: self.max_sentence_length].clone()
            sentence_attention_mask = attention_mask[:, :, :, :: self.max_sentence_length].clone()

            sentence_positions = torch.arange(1, num_sentences + 1).repeat(outputs.size(0), 1).to(
                outputs.device
            ) * (sentence_attention_mask.reshape(-1, num_sentences) >= -100).int().to(
                outputs.device
            )
            sentence_global_tokens += self.position_embeddings(sentence_positions)

            document_outputs = self.document_encoder(
                sentence_global_tokens, sentence_attention_mask, output_attentions=output_attentions
            )

            # replace sentence representative tokens
            outputs[:, :: self.max_sentence_length] = document_outputs[0]

        if output_attentions:
            return outputs, sentence_outputs[1], document_outputs[1]

        return outputs, None


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class HATEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                HATLayer(
                    config,
                    use_sentence_encoder=self.config.encoder_layout[str(idx)]["sentence_encoder"],
                    use_document_encoder=self.config.encoder_layout[str(idx)]["document_encoder"],
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_sentences=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_sentence_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                num_sentences,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_sentence_attentions = all_sentence_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_sentence_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithSentenceAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            sentence_attentions=all_sentence_attentions,
        )


class HATPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HATConfig
    base_model_prefix = "hat"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AttentivePooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_dropout = config.hidden_dropout_prob
        self.lin_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, inputs):
        lin_out = self.lin_proj(inputs)
        attention_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
        attention_weights_normalized = torch.softmax(attention_weights, -1)
        return torch.sum(attention_weights_normalized.unsqueeze(-1) * inputs, 1)


class HATPooler(nn.Module):
    def __init__(self, config, pooling="max"):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooling = pooling
        if self.pooling == "attentive":
            self.attentive_pooling = AttentivePooling(config)
        self.activation = nn.Tanh()
        self.max_sentence_length = config.max_sentence_length

    def forward(self, hidden_states):
        if self.pooling == "attentive":
            pooled_output = self.attentive_pooling(hidden_states)
        else:
            pooled_output = torch.max(hidden_states, dim=1)[0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class HATModel(HATPreTrainedModel):
    """The bare HAT Model transformer outputting raw hidden-states without any specific head on top."""

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->HAT
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = HATEmbeddings(config)
        self.encoder = HATEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # Compute number of sentences
        num_batch_sentences = input_ids.shape[-1] // self.config.max_sentence_length

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            num_sentences=num_batch_sentences,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output) + encoder_outputs[1:]

        return BaseModelOutputWithSentenceAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            sentence_attentions=encoder_outputs.sentence_attentions,
        )


class HATForSequenceClassification(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, pooling="max"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.max_sentence_length = config.max_sentence_length
        self.pooling = pooling

        self.hi_transformer = HATModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.pooler = HATPooler(config, pooling=pooling)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.pooling == "first":
            pooled_output = self.pooler(torch.unsqueeze(sequence_output[:, 0, :], 1))
        elif self.pooling == "last":
            pooled_output = self.pooler(
                torch.unsqueeze(sequence_output[:, -self.max_sentence_length, :], 1)
            )
        else:
            pooled_output = self.pooler(sequence_output[:, :: self.max_sentence_length])

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
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, position_ids):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    mask = input_ids.ne(padding_idx).int()
    return position_ids[:, : input_ids.size(1)].repeat(input_ids.size(0), 1) * mask


def build_hat():
    max_sentence_length = 8
    max_sentences = 4
    num_hidden_layers = 2
    encoder_layout = {
        str(i): {"sentence_encoder": True, "document_encoder": (i == num_hidden_layers - 1)}
        for i in range(num_hidden_layers)
    }
    config = HATConfig(
        vocab_size=200,
        hidden_size=16,
        max_sentences=max_sentences,
        max_sentence_size=max_sentence_length,
        model_max_length=max_sentence_length * max_sentences,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=max_sentence_length + 2,
        type_vocab_size=1,
        pad_token_id=0,
        encoder_layout=encoder_layout,
        num_labels=3,
    )
    config.max_sentence_length = max_sentence_length
    return HATForSequenceClassification(config, pooling="max")


def example_input_hat():
    max_sentence_length = 8
    max_sentences = 2
    seq_len = max_sentence_length * max_sentences
    input_ids = torch.randint(1, 200, (2, seq_len))
    attention_mask = torch.ones(2, seq_len)
    return (input_ids, attention_mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("HAT", "build_hat", "example_input_hat", 2022, "vendored"),
]
