# SOURCE: vendored from facebookresearch/PAQ @ main (commit at fetch time)
# https://raw.githubusercontent.com/facebookresearch/PAQ/main/paq/generation/answer_extractor/span2D_model.py
#
# Lewis et al. 2021 (NeurIPS) "PAQ: 65 Million Probably-Asked Questions and What You
# Can Do With Them" -- PAQ is a QA-pair generation pipeline (passage selection ->
# question generation -> answer extraction -> filtering) plus RePAQ, a dense
# retriever/reranker over the resulting 65M QA cache. The retriever
# (`paq/retrievers/retriever_utils.py::RetrieverEncoder`) and reranker
# (`paq/rerankers/rerank.py::load_reranker`) are thin, architecturally-unmodified
# wrappers around real HF `AutoModel`/`AutoModelForMultipleChoice` classes (rung-1
# territory, not what's vendored here). The genuine architectural contribution in
# the PAQ codebase is the *answer extractor* used during QA-pair generation:
# `AnswerSpanExtractor2DModel`, a `BertPreTrainedModel` subclass that predicts
# answer spans jointly over a 2D (start x end) grid instead of the usual separate
# start/end classifiers -- a real added head/mechanism, not just BERT usage. That
# class (plus its `ModelOutput` dataclass and `postprocess_span2d_output` helper) is
# vendored here verbatim from `paq/generation/answer_extractor/span2D_model.py`,
# which imports nothing beyond `torch`/`transformers` (both base libs already
# installed) -- no import-isolation changes were needed at all beyond the module
# path itself. One mechanical lint cleanup: the original computes an unused local
# `feature_null_score = span_logits[0, 0]` inside `postprocess_span2d_output`
# (dead code upstream too -- never read again in that function); it is dropped
# here since it does not affect the traced architecture or any returned value.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import math

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, ModuleList

from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.utils import ModelOutput

import numpy as np


@dataclass
class AnswerSpanExtractor2DModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        span_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, sequence_length)`):
            Span scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    span_logits: torch.FloatTensor = None
    span_masks: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class AnswerSpanExtractor2DModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        # Linear mapping for start and end representation
        self.start_outputs = nn.Linear(config.hidden_size, config.span_output_size)
        self.end_outputs = nn.Linear(config.hidden_size, config.span_output_size)
        prev_out_size = config.span_output_size * 2

        # Add final MLP output layers to produce probabilities
        self.output_mlp = None
        mlp_sizes = getattr(config, "output_mlp_sizes", None)
        if mlp_sizes and len(mlp_sizes) > 0:
            self.output_mlp = ModuleList()
            for cur_size in mlp_sizes:
                self.output_mlp.append(nn.Linear(prev_out_size, cur_size))
                self.output_mlp.append(nn.ReLU())
                prev_out_size = cur_size

        self.span_outputs = nn.Linear(prev_out_size, 1)

        self.max_answer_length = getattr(config, "max_answer_length", 30)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_num_answers)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_num_answers)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        start_hidden = self.start_outputs(sequence_output)  # [B, L, D]
        end_hidden = self.end_outputs(sequence_output)  # [B, L, D]

        sequence_length = sequence_output.shape[1]
        start_hidden = start_hidden.unsqueeze(2).expand(-1, -1, sequence_length, -1)  # [B, L, L, D]
        end_hidden = end_hidden.unsqueeze(1).expand(-1, sequence_length, -1, -1)  # [B, L, L, D]
        # Concat the start and end representation to form span representation
        span_hidden = torch.cat((start_hidden, end_hidden), -1)  # [B, L, L, D*2]

        # Run MLP layers
        if self.output_mlp is not None:
            for layer in self.output_mlp:
                span_hidden = layer(span_hidden)  # [B, L, L, ?]

        span_logits = self.span_outputs(span_hidden)  # [B, L, L, 1]
        span_logits = span_logits.squeeze(-1)  # [B, L, L]

        span_masks = torch.einsum("bi,bj->bij", attention_mask, attention_mask)  # [B, L, L]
        span_masks = torch.triu(span_masks)
        span_masks = torch.tril(span_masks, diagonal=self.max_answer_length)

        def _convert_to_span_matrix(start_positions, end_positions):
            span_labels = torch.zeros_like(span_logits)  # [B, L, L]
            for i, (start_post, end_post) in enumerate(zip(start_positions, end_positions)):
                for start_idx, end_idx in zip(start_post, end_post):
                    if 0 <= start_idx and 0 <= end_idx:  # we use -1 as null indicator
                        assert start_idx < sequence_length and end_idx < sequence_length
                        span_labels[i, start_idx, end_idx] = 1.0
                    else:
                        break
            return span_labels

        total_loss = None
        if start_positions is not None and end_positions is not None:
            span_labels = _convert_to_span_matrix(start_positions, end_positions)

            loss_fct = BCEWithLogitsLoss(weight=span_masks, reduction="sum")
            total_loss = loss_fct(span_logits, span_labels)  # / torch.sum(span_masks)

        if not return_dict:
            output = (span_logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return AnswerSpanExtractor2DModelOutput(
            loss=total_loss,
            span_logits=span_logits,
            span_masks=span_masks,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def postprocess_span2d_output(
    span2D_output: AnswerSpanExtractor2DModelOutput,
    features,
    max_answer_length,
    passage: str,
    n_best_size: int,
) -> List[Dict]:
    all_span_logits = span2D_output.span_logits.detach().cpu().numpy()
    all_span_masks = span2D_output.span_masks.detach().cpu().numpy()

    prelim_predictions = []
    # Looping through all the features associated to the current example.
    for feature_index in range(len(all_span_logits)):
        span_logits = all_span_logits[feature_index]
        span_masks = all_span_masks[feature_index]
        span_logits += -100 * (1 - span_masks)  # mask the span logits

        offset_mapping = features["offset_mapping"][feature_index]
        token_is_max_context = None

        start_indexes, end_indexes = np.unravel_index(
            np.argsort(span_logits, axis=None)[
                -1 : -n_best_size - 10 : -1
            ],  # a buffer of 10 in case some are invalid
            span_logits.shape,
        )
        start_indexes, end_indexes = start_indexes.tolist(), end_indexes.tolist()
        for start_index, end_index in zip(start_indexes, end_indexes):
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue
            if token_is_max_context is not None and not token_is_max_context.get(
                str(start_index), False
            ):
                continue
            prelim_predictions.append(
                {
                    "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                    "score": span_logits[start_index, end_index],
                }
            )

    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

    for pred in predictions:
        offsets = pred.pop("offsets")
        pred["text"] = passage[offsets[0] : offsets[1]]
        pred["start"] = offsets[0]
        pred["end"] = offsets[1]

    if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
        predictions.insert(0, {"text": "null", "score": -100.0, "start": 0, "end": 0})

    for pred in predictions:
        score = pred.get("score")
        pred["score"] = sigmoid(score)

    return predictions


def build_paq_span_extractor():
    config = BertConfig(
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    config.span_output_size = 16
    config.output_mlp_sizes = [16]
    config.max_answer_length = 10
    return AnswerSpanExtractor2DModel(config)


def example_input_paq_span_extractor():
    batch = 2
    seq_len = 12
    input_ids = torch.randint(0, 99, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    return (input_ids, attention_mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "PAQ RePAQ Answer Span Extractor (2D)",
        "build_paq_span_extractor",
        "example_input_paq_span_extractor",
        2021,
        "vendored",
    ),
]
