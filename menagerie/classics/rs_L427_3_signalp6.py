# SOURCE: vendored from fteufel/signalp-6.0 @ main
#
# SignalP 6.0's `BertSequenceTaggingCRF` (src/signalp6/models/bert_crf.py) wraps a
# HuggingFace `transformers.BertModel` protein-LM backbone with a linear
# hidden->emissions projection and a linear-chain CRF (multi_tag_crf.py, itself a
# vendored/extended fork of JeppeHallgren/pytorch-crf, credited in the original file's
# own header comment) for sequence tagging + global signal-peptide-type
# classification. Both files use only torch + transformers (installed base libs);
# transcribed verbatim (forward/CRF math/global-label aggregation unchanged). Only the
# `ProteinBertTokenizer`/`kingdom`-embedding convenience wrapper class and CLI-only
# imports were trimmed, and `use_kingdom_id`/`use_crf`/`use_region_labels` are left at
# their real default config values (kingdom id off, plain multi-state CRF tagging on).

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel

# --- multi_tag_crf.py (CRF, vendored from JeppeHallgren/pytorch-crf, per original header) ---


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    """

    def __init__(
        self,
        num_tags: int,
        batch_first: bool = False,
        constrain_every: bool = False,
        allowed_transitions: List[Tuple[int, int]] = None,
        allowed_start: List[int] = None,
        allowed_end: List[int] = None,
        include_start_end_transitions: bool = False,
        init_weight: float = 0.577350,
    ) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.constrain_every = constrain_every
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.transition_constraint = allowed_transitions is not None
        self.init_weight = init_weight
        self.include_start_end_transitions = True
        if self.include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.empty(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.empty(num_tags))

            if allowed_transitions is None:  # All transitions are valid.
                constraint_start_mask = torch.empty(num_tags).fill_(1.0)
                constraint_end_mask = torch.empty(num_tags).fill_(1.0)
            else:
                constraint_start_mask = torch.empty(num_tags).fill_(0.0)
                constraint_end_mask = torch.empty(num_tags).fill_(0.0)
                constraint_start_mask[allowed_start] = 1.0
                constraint_end_mask[allowed_end] = 1.0
            self._constraint_start_mask = torch.nn.Parameter(
                constraint_start_mask, requires_grad=False
            )
            self._constraint_end_mask = torch.nn.Parameter(constraint_end_mask, requires_grad=False)

        constraint_mask = None
        if allowed_transitions is None:  # All transitions are valid.
            constraint_mask = torch.empty(num_tags, num_tags).fill_(1.0)
        else:
            constraint_mask = torch.empty(num_tags, num_tags).fill_(0.0)
            for i, j in allowed_transitions:
                constraint_mask[i, j] = 1.0
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -self.init_weight, self.init_weight)

        if self.include_start_end_transitions:
            nn.init.uniform_(self.start_transitions, -self.init_weight, self.init_weight)
            nn.init.uniform_(self.end_transitions, -self.init_weight, self.init_weight)

        if self.transition_constraint:
            self.do_transition_constraint()

    def do_transition_constraint(self):
        inf = torch.as_tensor(-20, dtype=self.transitions.dtype)
        inf_matrix = torch.empty(self.transitions.shape).fill_(inf).to(self.transitions.device)
        self.transitions.data = torch.where(
            self._constraint_mask.byte(), self.transitions, inf_matrix
        )

        if self.include_start_end_transitions:
            inf_vector = (
                torch.empty(self.start_transitions.shape)
                .fill_(inf)
                .to(self.start_transitions.device)
            )
            self.start_transitions.data = torch.where(
                self._constraint_start_mask.byte(), self.start_transitions, inf_vector
            )
            self.end_transitions.data = torch.where(
                self._constraint_end_mask.byte(), self.end_transitions, inf_vector
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        tag_bitmap: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        if tag_bitmap is not None:
            self._validate(emissions, tags=tag_bitmap[:, :, 0], mask=mask)
        else:
            self._validate(emissions, tags=tags, mask=mask)

        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if tag_bitmap is not None:
                tag_bitmap = tag_bitmap.transpose(0, 1)
            else:
                tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if self.constrain_every:
            self.do_transition_constraint()

        if tag_bitmap is not None:
            log_numerator = self._compute_seq_score_multi_tag(emissions, tag_bitmap, mask)
        else:
            log_numerator = self._compute_seq_score(emissions, tags, mask)
        log_denominator = self._compute_log_normalizer(emissions, mask)
        llh = log_numerator - log_denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        assert reduction == "token_mean"
        return llh.sum() / mask.float().sum()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None,
        init_state_vector: Optional[torch.LongTensor] = None,
        forced_steps: int = 2,
        no_mask_label: int = 0,
    ) -> List[List[int]]:
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if init_state_vector is not None:
            paths = self._viterbi_decode_force_states(
                emissions, mask, init_state_vector, forced_steps, no_mask_label
            )
        else:
            paths = self._viterbi_decode(emissions, mask)

        return paths

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(f"emissions must have dimension of 3, got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, got {emissions.size(2)}"
            )

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_seq_score(
        self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        if self.include_start_end_transitions:
            score = self.start_transitions[tags[0]]
            score += emissions[0, torch.arange(batch_size), tags[0]]
        else:
            score = emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        if self.include_start_end_transitions:
            score += self.end_transitions[last_tags]

        return score

    def _compute_seq_score_multi_tag(
        self,
        emissions: torch.Tensor,
        tag_bitmap: torch.LongTensor,
        mask: torch.ByteTensor,
    ) -> torch.Tensor:
        inf_matrix = (
            torch.empty(emissions.shape)
            .fill_(torch.finfo(emissions.dtype).min)
            .to(emissions.device)
        )
        filtered_inputs = torch.where(tag_bitmap.byte(), emissions, inf_matrix)

        seq_score = self._compute_log_normalizer(filtered_inputs, mask)

        return seq_score

    def _compute_log_normalizer(
        self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        if self.include_start_end_transitions:
            score = self.start_transitions + emissions[0]
        else:
            score = emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        if self.include_start_end_transitions:
            score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor
    ) -> List[List[int]]:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        if self.transition_constraint:
            start_transitions_temp = self.start_transitions.data.clone()
            end_transitions_temp = self.end_transitions.data.clone()
            transitions_temp = self.transitions.data.clone()
            self.do_transition_constraint()

        if self.include_start_end_transitions:
            score = self.start_transitions + emissions[0]
        else:
            score = emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        if self.include_start_end_transitions:
            score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        if self.transition_constraint:
            self.start_transitions.data = start_transitions_temp
            self.end_transitions.data = end_transitions_temp
            self.transitions.data = transitions_temp

        return best_tags_list

    def _compute_log_alpha(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor, run_backwards: bool
    ) -> torch.FloatTensor:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.size()[:2] == mask.size()
        assert emissions.size(2) == self.num_tags
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()
        broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
        emissions_broadcast = emissions.unsqueeze(2)
        seq_iterator = range(1, seq_length)

        if run_backwards:
            broadcast_transitions = broadcast_transitions.transpose(1, 2)
            emissions_broadcast = emissions_broadcast.transpose(2, 3)

            if self.include_start_end_transitions:
                log_prob = [self.end_transitions.expand(emissions.size(1), -1)]
            else:
                log_prob = [torch.zeros_like(emissions[0])]

            seq_iterator = reversed(seq_iterator)
        else:
            if self.include_start_end_transitions:
                log_prob = [emissions[0] + self.start_transitions.view(1, -1)]
            else:
                log_prob = [torch.zeros_like(emissions[0])]

        for i in seq_iterator:
            broadcast_log_prob = log_prob[-1].unsqueeze(2)  # (batch_size, num_tags, 1)
            score = broadcast_log_prob + broadcast_transitions + emissions_broadcast[i]
            score = self._log_sum_exp(score, dim=1)
            log_prob.append(
                score * mask[i].unsqueeze(1) + log_prob[-1] * (1.0 - mask[i]).unsqueeze(1)
            )

        if run_backwards:
            log_prob.reverse()

        return torch.stack(log_prob)

    def compute_marginal_probabilities(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor
    ) -> torch.FloatTensor:
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        alpha = self._compute_log_alpha(emissions, mask, run_backwards=False)
        beta = self._compute_log_alpha(emissions, mask, run_backwards=True)
        if self.include_start_end_transitions:
            z = torch.logsumexp(alpha[alpha.size(0) - 1] + self.end_transitions, dim=1)
        else:
            z = torch.logsumexp(alpha[alpha.size(0) - 1], dim=1)
        prob = alpha + beta - z.view(1, -1, 1)

        if self.batch_first:
            prob = prob.transpose(0, 1)

        marginals = torch.exp(prob)
        return marginals

    @staticmethod
    def _log_sum_exp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        offset, _ = tensor.max(dim)
        broadcast_offset = offset.unsqueeze(dim)
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))
        return offset + safe_log_sum_exp

    def _viterbi_decode_force_states(
        self,
        emissions: torch.FloatTensor,
        mask: torch.ByteTensor,
        init_state_vector=torch.LongTensor,
        forced_steps: int = 2,
        no_mask_label: int = 0,
    ) -> List[List[int]]:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        if self.transition_constraint:
            start_transitions_temp = self.start_transitions.data.clone()
            end_transitions_temp = self.end_transitions.data.clone()
            transitions_temp = self.transitions.data.clone()
            self.do_transition_constraint()

        if self.include_start_end_transitions:
            score = self.start_transitions + emissions[0]
        else:
            score = emissions[0]
        history = []

        init_steps_mask = torch.nn.functional.one_hot(init_state_vector, num_classes=self.num_tags)
        dont_mask_idx = torch.where(init_state_vector == no_mask_label)[0]

        init_steps_mask[dont_mask_idx] = 1
        init_steps_mask[dont_mask_idx] = 0

        score = score + init_steps_mask * 10000

        for i in range(1, forced_steps):
            broadcast_score = score.unsqueeze(2)
            emissions_fixed = emissions[i] + init_steps_mask * 100000
            broadcast_emission = emissions_fixed.unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        for i in range(forced_steps, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        if self.include_start_end_transitions:
            score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        if self.transition_constraint:
            self.start_transitions.data = start_transitions_temp
            self.end_transitions.data = end_transitions_temp
            self.transitions.data = transitions_temp

        return best_tags_list


# --- bert_crf.py (BertSequenceTaggingCRF) ---

SIGNALP6_CLASS_LABEL_MAP = [
    [0, 1, 2],
    [3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22],
    [23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36],
]


class BertSequenceTaggingCRF(BertPreTrainedModel):
    """Sequence tagging and global label prediction model (like SignalP).
    LM output goes through a linear layer with classifier_hidden_size before being projected to num_labels outputs.
    These outputs then either go into the CRF as emissions, or to softmax as direct probabilities.
    config.use_crf controls this.

    Inputs are batch first.
       Loss is sum between global sequence label crossentropy and position wise tags crossentropy.
       Optionally use CRF.
    """

    def __init__(self, config):
        super().__init__(config)

        self.use_kingdom_id = config.use_kingdom_id if hasattr(config, "use_kingdom_id") else False

        if self.use_kingdom_id:
            self.kingdom_embedding = nn.Embedding(4, config.kingdom_embed_size)

        self.bert = BertModel(config=config)
        self.lm_output_dropout = nn.Dropout(
            config.lm_output_dropout if hasattr(config, "lm_output_dropout") else 0
        )
        self.kingdom_id_as_token = (
            config.kingdom_id_as_token if hasattr(config, "kingdom_id_as_token") else False
        )
        self.type_id_as_token = (
            config.type_id_as_token if hasattr(config, "type_id_as_token") else False
        )

        self.crf_input_length = 70

        self.outputs_to_emissions = nn.Linear(
            config.hidden_size
            if self.use_kingdom_id is False
            else config.hidden_size + config.kingdom_embed_size,
            config.num_labels,
        )

        self.num_global_labels = (
            config.num_global_labels if hasattr(config, "num_global_labels") else config.num_labels
        )
        self.num_labels = config.num_labels
        self.class_label_mapping = (
            config.class_label_mapping
            if hasattr(config, "class_label_mapping")
            else SIGNALP6_CLASS_LABEL_MAP
        )
        assert len(self.class_label_mapping) == self.num_global_labels, (
            "defined number of classes and class-label mapping do not agree."
        )

        self.allowed_crf_transitions = (
            config.allowed_crf_transitions if hasattr(config, "allowed_crf_transitions") else None
        )
        self.allowed_crf_starts = (
            config.allowed_crf_starts if hasattr(config, "allowed_crf_starts") else None
        )
        self.allowed_crf_ends = (
            config.allowed_crf_ends if hasattr(config, "allowed_crf_ends") else None
        )

        self.crf = CRF(
            num_tags=config.num_labels,
            batch_first=True,
            allowed_transitions=self.allowed_crf_transitions,
            allowed_start=self.allowed_crf_starts,
            allowed_end=self.allowed_crf_ends,
        )
        self.sp_region_tagging = (
            config.use_region_labels if hasattr(config, "use_region_labels") else False
        )
        self.use_large_crf = True

        self.crf_scaling_factor = (
            config.crf_scaling_factor if hasattr(config, "crf_scaling_factor") else 1
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        kingdom_ids=None,
        input_mask=None,
        targets=None,
        targets_bitmap=None,
        global_targets=None,
        inputs_embeds=None,
        sample_weights=None,
        return_emissions=False,
        force_states=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if targets is not None and targets_bitmap is not None:
            raise ValueError("You cannot specify both targets and targets_bitmap at the same time")

        outputs = self.bert(input_ids, attention_mask=input_mask, inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        sequence_output, input_mask = self._trim_transformer_output(sequence_output, input_mask)
        if self.kingdom_id_as_token:
            sequence_output = sequence_output[:, 1:, :]
            input_mask = input_mask[:, 1:] if input_mask is not None else None
        if self.type_id_as_token:
            sequence_output = sequence_output[:, 1:, :]
            input_mask = input_mask[:, 1:] if input_mask is not None else None

        if targets is not None:
            sequence_output = sequence_output[:, : targets.shape[1], :]
            input_mask = input_mask[:, : targets.shape[1]] if input_mask is not None else None
        else:
            sequence_output = sequence_output[:, : self.crf_input_length, :]
            input_mask = input_mask[:, : self.crf_input_length] if input_mask is not None else None

        sequence_output = self.lm_output_dropout(sequence_output)

        if self.use_kingdom_id:
            ids_emb = self.kingdom_embedding(kingdom_ids)
            ids_emb = ids_emb.unsqueeze(1).repeat(1, sequence_output.shape[1], 1)
            sequence_output = torch.cat([sequence_output, ids_emb], dim=-1)

        prediction_logits = self.outputs_to_emissions(sequence_output)

        if targets is not None:
            log_likelihood = self.crf(
                emissions=prediction_logits,
                tags=targets,
                tag_bitmap=None,
                mask=input_mask.byte(),
                reduction="mean",
            )
            neg_log_likelihood = -log_likelihood * self.crf_scaling_factor
        elif targets_bitmap is not None:
            log_likelihood = self.crf(
                emissions=prediction_logits,
                tags=None,
                tag_bitmap=targets_bitmap,
                mask=input_mask.byte(),
                reduction="mean",
            )
            neg_log_likelihood = -log_likelihood * self.crf_scaling_factor
        else:
            neg_log_likelihood = 0

        probs = self.crf.compute_marginal_probabilities(
            emissions=prediction_logits, mask=input_mask.byte()
        )

        if self.sp_region_tagging:
            global_probs = self.compute_global_labels_multistate(probs, input_mask)
        else:
            global_probs = self.compute_global_labels(probs, input_mask)

        global_log_probs = torch.log(global_probs)

        preds = self.predict_global_labels(global_probs, kingdom_ids, weights=None)

        if force_states:
            init_states = self.inital_state_labels_from_global_labels(preds)
        else:
            init_states = None
        viterbi_paths = self.crf.decode(
            emissions=prediction_logits,
            mask=input_mask.byte(),
            init_state_vector=init_states,
        )

        max_pad_len = max([len(x) for x in viterbi_paths])
        pos_preds = [x + [-1] * (max_pad_len - len(x)) for x in viterbi_paths]
        pos_preds = torch.tensor(pos_preds, device=probs.device)

        outputs = (global_probs, probs, pos_preds)

        losses = neg_log_likelihood

        if global_targets is not None:
            loss_fct = nn.NLLLoss(
                ignore_index=-1,
                reduction="none" if sample_weights is not None else "mean",
            )
            global_loss = loss_fct(
                global_log_probs.view(-1, self.num_global_labels),
                global_targets.view(-1),
            )

            if sample_weights is not None:
                global_loss = global_loss * sample_weights
                global_loss = global_loss.mean()

            losses = losses + global_loss

        if targets is not None or global_targets is not None or targets_bitmap is not None:
            outputs = (losses,) + outputs

        if return_emissions:
            outputs = outputs + (prediction_logits, input_mask)

        return outputs

    @staticmethod
    def _trim_transformer_output(hidden_states, input_mask):
        """Helper function to remove CLS, SEP tokens after passing through transformer"""
        hidden_states = hidden_states[:, 1:, :]

        if input_mask is not None:
            input_mask = input_mask[:, 1:]
            true_seq_lens = input_mask.sum(dim=1) - 1

            mask_list = []
            output_list = []
            for i in range(input_mask.shape[0]):
                mask_list.append(input_mask[i, : true_seq_lens[i]])
                output_list.append(hidden_states[i, : true_seq_lens[i], :])

            mask_out = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True)
            hidden_out = torch.nn.utils.rnn.pad_sequence(output_list, batch_first=True)
        else:
            hidden_out = hidden_states[:, :-1, :]
            mask_out = None

        return hidden_out, mask_out

    def compute_global_labels(self, probs, mask):
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

        summed_probs = (probs * mask.unsqueeze(-1)).sum(dim=1)
        sequence_lengths = mask.sum(dim=1)
        global_probs = summed_probs / sequence_lengths.unsqueeze(-1)

        no_sp = global_probs[:, 0:3].sum(dim=1)
        spi = global_probs[:, 3:7].sum(dim=1)

        if self.num_global_labels > 2:
            spii = global_probs[:, 7:11].sum(dim=1)
            tat = global_probs[:, 11:15].sum(dim=1)
            tat_spi = global_probs[:, 15:19].sum(dim=1)
            spiii = global_probs[:, 19:].sum(dim=1)
            return torch.stack([no_sp, spi, spii, tat, tat_spi, spiii], dim=-1)
        else:
            return torch.stack([no_sp, spi], dim=-1)

    def compute_global_labels_multistate(self, probs, mask):
        """Aggregates probabilities for region-tagging CRF output"""
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

        summed_probs = (probs * mask.unsqueeze(-1)).sum(dim=1)
        sequence_lengths = mask.sum(dim=1)
        global_probs = summed_probs / sequence_lengths.unsqueeze(-1)

        global_probs_list = []
        for class_indices in self.class_label_mapping:
            summed_probs = global_probs[:, class_indices].sum(dim=1)
            global_probs_list.append(summed_probs)

        return torch.stack(global_probs_list, dim=-1)

    def predict_global_labels(self, probs, kingdom_ids, weights=None):
        if self.use_kingdom_id:
            eukarya_idx = torch.where(kingdom_ids == 0)[0]
            summed_sp_probs = probs[eukarya_idx, 1:].sum(dim=1)
            probs[eukarya_idx, 1] = summed_sp_probs
            probs[eukarya_idx, 2:] = 0

        if weights is not None:
            probs = probs * weights
        preds = probs.argmax(dim=1)

        return preds

    @staticmethod
    def inital_state_labels_from_global_labels(preds):
        initial_states = torch.zeros_like(preds)
        initial_states[preds == 0] = 0
        initial_states[preds == 1] = 3
        initial_states[preds == 2] = 9
        initial_states[preds == 3] = 16
        initial_states[preds == 4] = 23
        initial_states[preds == 5] = 31

        return initial_states


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_signalp6():
    # Real config (config.json in the released checkpoint) uses a ProtTrans-style
    # ProtBert vocab (~30 tokens), hidden_size=1024, 30 layers, 37 CRF tags
    # (SIGNALP6_CLASS_LABEL_MAP spans 0..36), num_global_labels=6. Shrunk to a tiny
    # BERT backbone for the staging trace; num_labels/class_label_mapping kept at
    # the real 37-tag / 6-class scheme so compute_global_labels' slicing is valid.
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        num_labels=37,
        num_global_labels=6,
        use_kingdom_id=False,
        lm_output_dropout=0.1,
        use_region_labels=False,
    )
    return BertSequenceTaggingCRF(config)


def example_input_signalp6():
    B, L = 2, 74  # >= CLS + crf_input_length(70) + SEP so trimming stays non-empty
    input_ids = torch.randint(1, 32, (B, L))
    input_mask = torch.ones(B, L, dtype=torch.long)
    return (input_ids, None, input_mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SignalP6-BertCRF", "build_signalp6", "example_input_signalp6", 2022, "vendored"),
]
