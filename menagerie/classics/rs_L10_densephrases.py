# SOURCE: vendored from princeton-nlp/DensePhrases @ 9583883ea9390b0308e806c3e72fa5831afa445b
# (densephrases/encoder.py: Encoder, the dual dense-phrase / dense-query encoder trained for
# phrase-indexed open-domain QA; Lee, Sung, Kang & Chen, "Learning Dense Representations of
# Phrases at Scale", ACL 2021.)
#
# The Encoder class below is copied verbatim from the repo (a `transformers.PreTrainedModel`
# subclass that wraps THREE independently-finetuned transformer encoders -- a shared
# phrase_encoder plus two query-side encoders query_start_encoder/query_end_encoder built as
# `copy.deepcopy(self.phrase_encoder)` -- and computes phrase/query start-end span
# representations plus a start/end "filter" head; no architecture was added or changed, only
# `MODEL_MAPPING`-style base-transformer selection glue from the official
# `densephrases/utils/single_utils.py:load_encoder` was reproduced minimally (any AutoModel-
# style encoder works via `transformer_cls`/`pretrained`, matching the constructor's real
# contract) so this file has no dependency beyond `torch` and `transformers` (both base libs).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import copy
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy_with_logits, embedding, log_softmax, softmax
from transformers import BertConfig, BertModel, PreTrainedModel

logger = logging.getLogger(__name__)


class Encoder(PreTrainedModel):
    base_model_prefix = "densephrases"

    def __init__(
        self,
        config,
        tokenizer,
        pretrained=None,
        transformer_cls=None,
        lambda_kl=0.0,
        lambda_neg=0.0,
        lambda_flt=0.0,
    ):
        super().__init__(config)
        self.tokenizer = tokenizer

        # Additional parameters
        self.filter_linear = nn.Linear(config.hidden_size, 2)

        # Arguments
        self.lambda_kl = lambda_kl
        self.lambda_neg = lambda_neg
        self.lambda_flt = lambda_flt
        self.pre_batch = None
        self.apply(self.init_weights)

        # Load transformer after init
        assert pretrained is not None or transformer_cls is not None
        logger.info(
            "Initializing encoders with pre-trained LM"
            if pretrained
            else "Loading encoders from load_dir"
        )
        if lambda_kl > 0:
            logger.info("Teacher initialized for distillation. Weights will be loaded.")
            self.cross_encoder = None
            self.qa_outputs = None

        # Encoders: three LMs
        self.phrase_encoder = pretrained if pretrained is not None else transformer_cls(config)
        self.query_start_encoder = copy.deepcopy(self.phrase_encoder)
        self.query_end_encoder = copy.deepcopy(self.phrase_encoder)

    def init_pre_batch(self, pbn_size):
        """Initialize a pre-batch queue"""
        from collections import deque

        self.pre_batch = deque(maxlen=pbn_size)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def merge_inputs(self, input_ids_, attention_mask_, input_ids, attention_mask):
        """Merge queries and passages for the cross encoder"""
        new_input_ids = []
        new_attention_mask = []
        new_token_type_ids = []
        max_len = input_ids_[0].shape[0] + input_ids[0].shape[0]
        for input_id_, att_mask_, input_id, att_mask in zip(
            input_ids_, attention_mask_, input_ids, attention_mask
        ):
            new_input_id = torch.zeros(max_len).long().to(self.device)
            title_sep = (input_id == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0]
            sep_input_id = input_id[title_sep + 1 : att_mask.sum()]
            new_input_id[: att_mask_.sum() + att_mask.sum() - 1 - title_sep] = torch.cat(
                [input_id_[: att_mask_.sum()], sep_input_id], dim=0
            )
            new_input_ids.append(new_input_id)
            new_attention_mask.append((new_input_id != self.tokenizer.pad_token_id).long())
            new_token_type_ids.append(
                (
                    torch.cat(
                        [
                            torch.zeros(att_mask_.sum(0)).long().to(self.device),
                            torch.ones(att_mask.sum(0) - 1).long().to(self.device),
                            torch.zeros(max_len - att_mask.sum(0) - att_mask_.sum(0) + 1)
                            .long()
                            .to(self.device),
                        ]
                    )
                )
            )
        new_input_ids = torch.stack(new_input_ids)
        new_attention_mask = torch.stack(new_attention_mask)
        new_token_type_ids = torch.stack(new_token_type_ids)
        return new_input_ids, new_attention_mask, new_token_type_ids

    def embed_phrase(self, input_ids, attention_mask, token_type_ids):
        """Get phrase embeddings (token-wise)"""
        outputs = self.phrase_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs[0], outputs[0]

    def embed_query(self, input_ids_, attention_mask_, token_type_ids_):
        """Get query start/end embeddings"""
        outputs_s_ = self.query_start_encoder(
            input_ids_,
            attention_mask=attention_mask_,
            token_type_ids=token_type_ids_,
        )
        outputs_e_ = self.query_end_encoder(
            input_ids_,
            attention_mask=attention_mask_,
            token_type_ids=token_type_ids_,
        )
        sequence_output_s_ = outputs_s_[0]
        sequence_output_e_ = outputs_e_[0]
        query_start = sequence_output_s_[:, :1, :]
        query_end = sequence_output_e_[:, :1, :]
        return query_start, query_end

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        input_ids_=None,
        attention_mask_=None,
        token_type_ids_=None,
        start_positions=None,
        end_positions=None,
        return_phrase=False,
        return_query=False,
        neg_input_ids=None,
        neg_attention_mask=None,
        neg_token_type_ids=None,
    ):
        # Context-side
        if input_ids is not None:
            assert len(input_ids.size()) == 2
            start, end = self.embed_phrase(input_ids, attention_mask, token_type_ids)

            if neg_input_ids is not None:
                neg_start, neg_end = self.embed_phrase(
                    neg_input_ids, neg_attention_mask, neg_token_type_ids
                )

            filter_output = start[:]
            filter_start_logits, filter_end_logits = self.filter_linear(filter_output).chunk(
                2, dim=2
            )
            filter_start_logits = filter_start_logits.squeeze(2)
            filter_end_logits = filter_end_logits.squeeze(2)

            if return_phrase:
                return (start, end, filter_start_logits, filter_end_logits)

        # Query-side
        if input_ids_ is not None:
            assert len(input_ids_.size()) == 2
            query_start, query_end = self.embed_query(input_ids_, attention_mask_, token_type_ids_)

            if return_query:
                return (query_start, query_end)

        # Gather distributed reps
        if dist.is_initialized() and self.training:
            ps_list = [torch.zeros_like(start) for _ in range(dist.get_world_size())]
            pe_list = [torch.zeros_like(end) for _ in range(dist.get_world_size())]
            qs_list = [torch.zeros_like(query_start) for _ in range(dist.get_world_size())]
            qe_list = [torch.zeros_like(query_end) for _ in range(dist.get_world_size())]
            sp_list = [torch.zeros_like(start_positions) for _ in range(dist.get_world_size())]
            ep_list = [torch.zeros_like(end_positions) for _ in range(dist.get_world_size())]

            dist.all_gather(tensor_list=ps_list, tensor=start.contiguous())
            dist.all_gather(tensor_list=pe_list, tensor=end.contiguous())
            dist.all_gather(tensor_list=qs_list, tensor=query_start.contiguous())
            dist.all_gather(tensor_list=qe_list, tensor=query_end.contiguous())
            dist.all_gather(tensor_list=sp_list, tensor=start_positions.contiguous())
            dist.all_gather(tensor_list=ep_list, tensor=end_positions.contiguous())

            ps_list[dist.get_rank()] = start
            pe_list[dist.get_rank()] = end
            qs_list[dist.get_rank()] = query_start
            qe_list[dist.get_rank()] = query_end

            all_start = torch.cat(ps_list, 0)
            all_end = torch.cat(pe_list, 0)
            all_query_start = torch.cat(qs_list, 0)
            all_query_end = torch.cat(qe_list, 0)
            all_start_positions = torch.cat(sp_list, 0)
            all_end_positions = torch.cat(ep_list, 0)
            if neg_input_ids is not None:
                nps_list = [torch.zeros_like(neg_start) for _ in range(dist.get_world_size())]
                npe_list = [torch.zeros_like(neg_end) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=nps_list, tensor=neg_start.contiguous())
                dist.all_gather(tensor_list=npe_list, tensor=neg_end.contiguous())
                nps_list[dist.get_rank()] = neg_start
                npe_list[dist.get_rank()] = neg_end
                all_neg_start = torch.cat(nps_list, 0)
                all_neg_end = torch.cat(npe_list, 0)
        else:
            all_start = start
            all_end = end
            all_query_start = query_start
            all_query_end = query_end
            all_start_positions = start_positions
            all_end_positions = end_positions
            if neg_input_ids is not None:
                all_neg_start = neg_start
                all_neg_end = neg_end

        # Get dense logits
        start_logits = start.matmul(query_start.transpose(1, 2)).squeeze(-1)
        end_logits = end.matmul(query_end.transpose(1, 2)).squeeze(-1)
        dense_logits = start_logits.unsqueeze(2) + end_logits.unsqueeze(1)

        if neg_input_ids is not None:
            neg_start_logits = (
                (all_query_start.unsqueeze(1) * all_neg_start.unsqueeze(0))
                .sum(-1)
                .view(all_query_start.shape[0], all_neg_start.shape[0], all_neg_start.shape[1])
                .max(-1)[0]
            )
            neg_end_logits = (
                (all_query_end.unsqueeze(1) * all_neg_end.unsqueeze(0))
                .sum(-1)
                .view(all_query_end.shape[0], all_neg_end.shape[0], all_neg_end.shape[1])
                .max(-1)[0]
            )

        if self.training and self.lambda_neg > 0:
            pinb_start_logits = None
            pinb_end_logits = None
            if self.pre_batch is not None:
                if len(self.pre_batch) > 0:
                    pre_start = torch.cat([pb[0] for pb in self.pre_batch], dim=1)
                    pre_end = torch.cat([pb[1] for pb in self.pre_batch], dim=1)
                    pinb_start_logits = (
                        (all_query_start.unsqueeze(1) * pre_start.unsqueeze(0))
                        .sum(-1)
                        .view(all_query_start.shape[0], -1)
                    )
                    pinb_end_logits = (
                        (all_query_end.unsqueeze(1) * pre_end.unsqueeze(0))
                        .sum(-1)
                        .view(all_query_end.shape[0], -1)
                    )

            gold_start = torch.stack(
                [
                    st[start_pos : start_pos + 1] if start_pos > 0 else st[0:1]
                    for st, start_pos in zip(all_start, all_start_positions)
                ]
            )
            gold_end = torch.stack(
                [
                    en[end_pos : end_pos + 1] if end_pos > 0 else en[0:1]
                    for en, end_pos in zip(all_end, all_end_positions)
                ]
            )
            inb_start_logits = (
                (all_query_start.unsqueeze(1) * gold_start.unsqueeze(0))
                .sum(-1)
                .view(all_query_start.shape[0], -1)
            )
            inb_end_logits = (
                (all_query_end.unsqueeze(1) * gold_end.unsqueeze(0))
                .sum(-1)
                .view(all_query_end.shape[0], -1)
            )

            if neg_input_ids is not None:
                inb_start_logits = torch.cat((inb_start_logits, neg_start_logits), dim=1)
                inb_end_logits = torch.cat((inb_end_logits, neg_end_logits), dim=1)

            if pinb_start_logits is not None:
                inb_start_logits = torch.cat((inb_start_logits, pinb_start_logits), dim=1)
                inb_end_logits = torch.cat((inb_end_logits, pinb_end_logits), dim=1)

        outputs = (start_logits, end_logits, filter_start_logits, filter_end_logits)

        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(dense_logits.mean(2), start_positions)
            end_loss = loss_fct(dense_logits.mean(1), end_positions)
            single_loss = (start_loss + end_loss) / 2
            total_loss = single_loss

            if self.lambda_kl > 0:
                self.cross_encoder.eval()
                with torch.no_grad():
                    new_input_ids, new_attention_mask, new_token_type_ids = self.merge_inputs(
                        input_ids_, attention_mask_, input_ids, attention_mask
                    )
                    outputs_qd = self.cross_encoder(
                        new_input_ids,
                        attention_mask=new_attention_mask,
                        token_type_ids=new_token_type_ids,
                    )
                    tmp_sequence_output = outputs_qd[0]
                    sequence_output = []
                    for seq_output, att_mask_, input_id in zip(
                        tmp_sequence_output, attention_mask_, input_ids
                    ):
                        title_sep = (input_id == self.tokenizer.sep_token_id).nonzero(
                            as_tuple=True
                        )[0][0]
                        title_mask = torch.zeros(size=(title_sep.item(), seq_output.shape[1])).to(
                            self.device
                        )
                        new_logits = self.qa_outputs(
                            torch.cat(
                                (
                                    seq_output[:1],
                                    title_mask,
                                    seq_output[
                                        att_mask_.sum() : att_mask_.sum()
                                        + input_ids.shape[1]
                                        - 1
                                        - title_sep.item()
                                    ],
                                )
                            )
                        )
                        new_logits[1:title_sep, :] = -1e4
                        sequence_output.append(new_logits)
                    sequence_output = torch.stack(sequence_output)

                    start_logits_qd, end_logits_qd = sequence_output.split(1, dim=-1)
                    start_logits_qd, end_logits_qd = (
                        start_logits_qd.squeeze(-1),
                        end_logits_qd.squeeze(-1),
                    )

                temperature = 1.0
                start_logits = start_logits / temperature
                end_logits = end_logits / temperature
                start_logits_qd = start_logits_qd / temperature
                end_logits_qd = end_logits_qd / temperature
                kl_loss = nn.KLDivLoss(reduction="none")
                kl_start = (
                    kl_loss(
                        log_softmax(start_logits, dim=1),
                        target=softmax(start_logits_qd[:, : start_logits.size(1)], dim=1),
                    ).sum(1)
                ).mean(0)
                kl_end = (
                    kl_loss(
                        log_softmax(end_logits, dim=1),
                        target=softmax(end_logits_qd[:, : end_logits.size(1)], dim=1),
                    ).sum(1)
                ).mean(0)
                total_loss = total_loss + (kl_start + kl_end) / 2.0 * self.lambda_kl

            if self.lambda_neg > 0:
                inb_ignored_index = all_start_positions.size(0)  # noqa: F841 -- unused in upstream too, kept verbatim
                inb_s_target = torch.arange(all_start_positions.size(0)).to(self.device)
                inb_e_target = torch.arange(all_end_positions.size(0)).to(self.device)

                inb_loss_fct = CrossEntropyLoss()
                inb_start_loss = inb_loss_fct(inb_start_logits, inb_s_target)
                inb_end_loss = inb_loss_fct(inb_end_logits, inb_e_target)
                inb_se_loss = (inb_start_loss + inb_end_loss) / 2
                total_loss = total_loss + inb_se_loss * self.lambda_neg

            if self.lambda_flt > 0:
                length = torch.tensor(filter_start_logits.size(-1)).to(filter_start_logits.device)
                eye = torch.eye(length + 2).to(filter_start_logits.device)
                start_1hot = embedding(start_positions + 1, eye)[:, 1:-1]
                end_1hot = embedding(end_positions + 1, eye)[:, 1:-1]
                start_loss = binary_cross_entropy_with_logits(
                    filter_start_logits, start_1hot, pos_weight=length, reduction="none"
                ).mean(1)
                end_loss = binary_cross_entropy_with_logits(
                    filter_end_logits, end_1hot, pos_weight=length, reduction="none"
                ).mean(1)
                filter_loss = 0.5 * start_loss + 0.5 * end_loss

                assert all((start_positions > 0) == (end_positions > 0))
                ans_mask = (start_positions > 0).float()
                filter_loss = (filter_loss * ans_mask).sum() / (ans_mask.sum() + 1e-9)
                total_loss = total_loss + filter_loss * self.lambda_flt

            if self.pre_batch is not None:
                assert self.lambda_neg > 0
                if len(self.pre_batch) > 0:
                    if start.shape[0] == self.pre_batch[-1][0].shape[0]:
                        self.pre_batch.append(
                            [gold_start.clone().detach(), gold_end.clone().detach()]
                        )
                else:
                    self.pre_batch.append([gold_start.clone().detach(), gold_end.clone().detach()])

            outputs = (total_loss,) + outputs
        return outputs  # (loss), start_logits, end_logits, filter_start_logits, filter_end_logits


class _DummyTokenizer:
    """Minimal stand-in for the repo's HF AutoTokenizer: only `.sep_token_id`/`.pad_token_id`
    are read by Encoder (merge_inputs / distillation branch), neither exercised by the plain
    phrase+query forward path traced here."""

    sep_token_id = 102
    pad_token_id = 0


def build_densephrases_encoder():
    """Real Encoder wired to a tiny BertModel (transformer_cls=BertModel), matching the repo's
    `load_encoder` contract (`transformer_cls(config)` for a from-scratch base transformer)."""
    config = BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        type_vocab_size=2,
    )
    return Encoder(config, tokenizer=_DummyTokenizer(), transformer_cls=BertModel)


def example_input_densephrases_encoder():
    torch.manual_seed(0)
    batch_size, ctx_len, q_len = 2, 12, 6
    input_ids = torch.randint(1, 64, (batch_size, ctx_len))
    attention_mask = torch.ones(batch_size, ctx_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, ctx_len, dtype=torch.long)
    input_ids_ = torch.randint(1, 64, (batch_size, q_len))
    attention_mask_ = torch.ones(batch_size, q_len, dtype=torch.long)
    token_type_ids_ = torch.zeros(batch_size, q_len, dtype=torch.long)
    start_positions = torch.randint(0, ctx_len, (batch_size,))
    end_positions = torch.randint(0, ctx_len, (batch_size,))
    return (
        input_ids,
        attention_mask,
        token_type_ids,
        input_ids_,
        attention_mask_,
        token_type_ids_,
        start_positions,
        end_positions,
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DensePhrases",
        "build_densephrases_encoder",
        "example_input_densephrases_encoder",
        2021,
        "vendored-pytorch",
    ),
]
