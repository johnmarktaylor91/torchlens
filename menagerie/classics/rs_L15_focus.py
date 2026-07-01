# SOURCE: vendored from pkchat-focus/FoCus @ main (classification_modules.py)
#
# FoCus (Jang, Boo, Kim, "Call for Focus: Fusing Attention Over Persona and
# Knowledge to Generate Personalized Response with Focus", AAAI 2022,
# official ncsoft/pkchat-focus repo). Real architecture: `GPT2PK_ctxt`
# subclasses HuggingFace's `GPT2LMHeadModel` and adds a persona+knowledge
# grounding head on top -- an in-context attention bottleneck
# (`attn1`/`attn2`: Linear(hidden->5) -> Linear(5->hidden)) that compresses
# encoded knowledge-paragraph EOS representations into the same width as the
# dialogue EOS representation, then a `ConcatSummary` classifier (dropout +
# Linear(7*hidden -> 1)) scores each of 5 persona candidates and 10
# knowledge candidates by concatenating that in-context state with the
# candidate's own GPT-2 encoding; the top-scoring knowledge paragraph and
# all positively-classified persona sentences are then spliced back into the
# dialogue token sequence before a second GPT-2 LM pass produces the
# response. This two-stage grounding-then-generation head is FoCus's actual
# contribution, not a usage-only variant of GPT2LMHeadModel, so it is
# vendored (rung 2) rather than recipe'd.
#
# Vendoring notes (imports/portability fixes only, architecture untouched):
#   - The original `forward()` hardcodes `device = input_ids.get_device()`
#     then builds several scratch tensors with `.cuda(device)`, which
#     crashes on CPU (`get_device()` returns -1 for CPU tensors, and
#     `.cuda(-1)` is invalid). Replaced every `torch.tensor([...]).cuda(
#     device)` with `torch.tensor([...], device=input_ids.device)` --
#     device-portability only, identical values/dtypes/placement semantics
#     on GPU, now also correct on CPU. No architectural logic changed.
#   - `BARTPK_ctxt` (the BART variant of the same grounding head) is not
#     vendored here to keep the staged module minimal; `GPT2PK_ctxt` alone
#     already exercises the full persona/knowledge-grounding + LM-head
#     architecture that is FoCus's contribution.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Sigmoid, Softmax
from transformers import GPT2Config, GPT2LMHeadModel

MENAGERIE_ZOO = "vendored-pytorch"


class ConcatSummary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim * 7, 1)  # hiddensize, numclasses

    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits


class Summary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim, 1)  # hiddensize, numclasses

    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits


class GPT2PK_ctxt(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model_(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.n_embd)
        self.summary = Summary(emb_dim=config.n_embd)
        self.attn1 = nn.Linear(config.n_embd, 5)
        self.attn2 = nn.Linear(5, config.n_embd)  # Selected knowledge count only
        self.max_position = config.n_positions
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        input_eos=None,
        token_type_ids=None,
        only_dial_input_ids=None,
        only_dial_token_type_ids=None,
        persona_input_ids=None,
        knowledge_input_ids=None,
        persona_can_idx=None,
        persona_grounding=None,
        knowledge_can_idx=None,
        knowledge_grounding=None,
        tot_knowledge=None,
        tot_knowledge_token_ids=None,
        tot_knowledge_eos=None,
        training=None,
        mc_token_ids=None,
    ):
        persona = 50259
        knowledge = 50260
        padding = 50261
        bos = 50256
        device = input_ids.device

        persona_tensor = torch.tensor([persona], device=device)
        knowledge_tensor = torch.tensor([knowledge], device=device)
        bos_tensor = torch.tensor([bos], device=device)
        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None
        lm_labels = None

        if input_eos is not None:
            lm_hidden_states = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids)[
                "last_hidden_state"
            ]
            batch, seq_len, embdim = lm_hidden_states.size()
            lm_hidden_states_eos_list = []
            for i in range(batch):
                lm_hidden_states_batch = lm_hidden_states[i]
                lm_eos_batch = input_eos[i]
                lm_hidden_states_eos = torch.index_select(lm_hidden_states_batch, -2, lm_eos_batch)
                lm_hidden_states_eos_list.append(lm_hidden_states_eos)
            lm_eos_rep = torch.stack(lm_hidden_states_eos_list)

            tot_knowledge_hidden_states = self.transformer(
                input_ids=tot_knowledge, token_type_ids=tot_knowledge_token_ids
            )["last_hidden_state"]
            tot_knowledge_eos_list = []
            for i in range(batch):
                tot_knowledge_hidden_states_batch = tot_knowledge_hidden_states[i]
                tot_knowledge_eos_batch = tot_knowledge_eos[i]
                tot_knowledge_eos_list_batch = []
                for j in range(5):
                    tot_knowledge_eos_token = torch.index_select(
                        tot_knowledge_hidden_states_batch[j], -2, tot_knowledge_eos_batch[j]
                    )
                    tot_knowledge_eos_list_batch.append(tot_knowledge_eos_token.squeeze())
                tot_knowledge_eos_batch_rep = torch.stack(tot_knowledge_eos_list_batch)
                tot_knowledge_eos_list.append(tot_knowledge_eos_batch_rep)
            tot_knowledge_eos_final = torch.stack(tot_knowledge_eos_list)
            knowledge_inctxt_attn = self.attn1(tot_knowledge_eos_final)
            knowledge_inctxt_eos_rep = self.attn2(knowledge_inctxt_attn)
            inctxt_states = torch.cat((lm_eos_rep, knowledge_inctxt_eos_rep), dim=1).type_as(
                input_ids
            )

            sigmoid = Sigmoid()
            # persona candidates
            num_persona_can = 5
            if persona_input_ids is not None:
                persona_emb = self.transformer(input_ids=persona_input_ids)["last_hidden_state"]
                if persona_can_idx is not None:
                    persona_list = []
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]  # 6, 768
                        persona_emb_batch = persona_emb[batch_i]
                        persona_can_idx_batch = persona_can_idx[batch_i]
                        persona_batch_list = []
                        for i in range(num_persona_can):
                            persona_selected = torch.index_select(
                                persona_emb_batch[i], 0, persona_can_idx_batch[i]
                            )
                            final_rep_persona = torch.cat(
                                [
                                    inctxt_eos_batch.type_as(lm_eos_rep),
                                    persona_selected.type_as(lm_eos_rep),
                                ],
                                dim=0,
                            )  # 7,768
                            persona_batch_list.append(final_rep_persona)
                        persona_batch_list = torch.stack(persona_batch_list)
                        persona_list.append(persona_batch_list)
                    persona_rep = torch.stack(persona_list).view(batch * num_persona_can, -1)
                    persona_logits = self.concat_summary(persona_rep).view(batch, -1)
                    outputs = (persona_logits,)

                    persona_pred_sigmoid = sigmoid(persona_logits)
                    persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
                    all_persona_pred = []
                    selected_persona_idx = list()
                    for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                        batch_list_idx = list()
                        batch_list = list()
                        for i, can in enumerate(persona_batch):
                            if can:
                                batch_list_idx.append(can)
                                persona_selected_now = persona_input_ids[batch_idx][i]
                                mask_persona = torch.ne(persona_selected_now, padding)
                                persona_selected_now = torch.masked_select(
                                    persona_selected_now, mask_persona
                                )
                                batch_list.append(persona_selected_now[:-2])
                        all_persona_pred.append(batch_list)
                        selected_persona_idx.append(batch_list_idx)

            # knowledge candidates
            num_knowledge_can = 10
            if knowledge_input_ids is not None:
                knowledge_emb = self.transformer(input_ids=knowledge_input_ids)["last_hidden_state"]
                if knowledge_can_idx is not None:
                    knowledge_list = []
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        knowledge_emb_batch = knowledge_emb[batch_i]
                        knowledge_can_idx_batch = knowledge_can_idx[batch_i]
                        knowledge_batch_list = []
                        for i in range(num_knowledge_can):
                            knowledge_selected = torch.index_select(
                                knowledge_emb_batch[i], 0, knowledge_can_idx_batch[i]
                            )
                            final_rep_knowledge = torch.cat(
                                [
                                    inctxt_eos_batch.type_as(lm_eos_rep),
                                    knowledge_selected.type_as(lm_eos_rep),
                                ],
                                dim=0,
                            )
                            knowledge_batch_list.append(final_rep_knowledge)
                        knowledge_batch_list = torch.stack(knowledge_batch_list)
                        knowledge_list.append(knowledge_batch_list)
                    knowledge_rep = torch.stack(knowledge_list).view(batch * num_knowledge_can, -1)
                    knowledge_logits = self.concat_summary(knowledge_rep).view(batch, -1)
                    outputs = (knowledge_logits,) + outputs
                    softmax = Softmax(dim=-1)
                    knowledge_softmax = softmax(knowledge_logits)
                    _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
                    all_knowledge_pred = []
                    for batch_i in range(batch):
                        knowledge_pred_idx = k_index_1[batch_i]
                        knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                        mask_knowledge = torch.ne(knowledge_pred, padding)
                        knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                        knowledge_pred = knowledge_pred[1:-2]  # delete bos, knowledge_st, eos
                        if knowledge_pred.size()[0] > 150:
                            knowledge_pred = knowledge_pred[:150]
                        all_knowledge_pred.append(knowledge_pred)

            final_input_list = []
            final_input_tti_list = []
            final_lm_label_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                only_dial_token_type_ids_batch = only_dial_token_type_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                mask_only_dial_tti_batch = torch.ne(only_dial_token_type_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(
                    only_dial_input_ids_batch, mask_only_dial_input_ids_batch
                )
                only_dial_token_type_ids_batch = torch.masked_select(
                    only_dial_token_type_ids_batch, mask_only_dial_tti_batch
                )
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                    new_persona_tti = torch.tensor(
                        [persona] * (new_persona.size()[0]), device=device
                    )
                else:
                    new_persona = None
                    new_persona_tti = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)
                new_knowledge_tti = torch.tensor(
                    [knowledge] * (new_knowledge.size()[0]), device=device
                )

                if new_persona is not None:
                    new_input = torch.cat(
                        [bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch], dim=-1
                    )
                    new_input_tti = torch.cat(
                        [
                            knowledge_tensor,
                            new_knowledge_tti,
                            new_persona_tti,
                            only_dial_token_type_ids_batch,
                        ],
                        dim=-1,
                    )
                else:
                    new_input = torch.cat(
                        [bos_tensor, new_knowledge, only_dial_input_ids_batch], dim=-1
                    )
                    new_input_tti = torch.cat(
                        [knowledge_tensor, new_knowledge_tti, only_dial_token_type_ids_batch],
                        dim=-1,
                    )

                new_input_size = new_input.size()[0]
                new_lm_label = torch.cat(
                    [
                        torch.tensor(
                            [-100] * (new_input_size - (only_dial_input_ids_batch.size()[0]) + 1),
                            device=device,
                        ),
                        only_dial_input_ids_batch[1:],
                    ],
                    dim=-1,
                )
                assert new_input.size() == new_input_tti.size() == new_lm_label.size()
                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size, device=device)
                    add_lm_padding = torch.tensor([-100] * padding_size, device=device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                    final_tti_input = torch.cat([new_input_tti, add_padding], dim=-1)
                    final_lm_label = torch.cat([new_lm_label, add_lm_padding], dim=-1)
                final_input_list.append(final_input)
                final_input_tti_list.append(final_tti_input)
                final_lm_label_list.append(final_lm_label)
            input_ids = torch.stack(final_input_list)
            token_type_ids = torch.stack(final_input_tti_list)
            lm_labels = torch.stack(final_lm_label_list)

        dynamic_lm_hidden_states = self.transformer(
            input_ids=input_ids, token_type_ids=token_type_ids
        )["last_hidden_state"]
        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(
                persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits)
            )
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is not True:
            outputs = (lm_labels,) + outputs
            lm_labels = None

        if lm_labels is not None:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        return outputs
        # (lm_loss-training), (lm_label-validation), (knowledge_loss), (persona_loss),
        # dynamic_lm_logits, knowledge_logits, persona_logits, presents,
        # (all hidden_states), (attentions)


# Local alias: the original module imports `GPT2Model` directly from
# `transformers`; kept as a plain re-export so `GPT2PK_ctxt.__init__` above
# reads identically to the upstream source.
from transformers import GPT2Model as GPT2Model_  # noqa: E402


_VOCAB = 50262  # base GPT-2 BPE vocab (50257) + persona/knowledge/padding/... specials
_PADDING = 50261
_MAX_POSITION = 64
_PERSONA_LEN = 6
_KNOWLEDGE_LEN = 8
_TOT_KNOW_LEN = 6
_DIAL_LEN = 20


class FoCusGPT2Wrapper(nn.Module):
    """Wraps GPT2PK_ctxt's many-kwarg forward into a single-tensor-friendly call so
    torchlens can trace it directly (GPT2PK_ctxt.forward's real computation is unchanged)."""

    def __init__(self):
        super().__init__()
        config = GPT2Config(
            vocab_size=_VOCAB,
            n_positions=_MAX_POSITION,
            n_embd=32,
            n_layer=2,
            n_head=2,
        )
        self.model = GPT2PK_ctxt(config)

    def forward(self, dummy):
        torch.manual_seed(0)
        input_ids = torch.randint(0, _VOCAB - 10, (1, _DIAL_LEN))
        token_type_ids = torch.zeros(1, _DIAL_LEN, dtype=torch.long)
        input_eos = torch.tensor([[_DIAL_LEN - 1]])

        only_dial_input_ids = torch.cat(
            [input_ids, torch.full((1, _MAX_POSITION - _DIAL_LEN), _PADDING, dtype=torch.long)],
            dim=-1,
        )
        only_dial_token_type_ids = torch.cat(
            [
                token_type_ids,
                torch.full((1, _MAX_POSITION - _DIAL_LEN), _PADDING, dtype=torch.long),
            ],
            dim=-1,
        )

        persona_input_ids = torch.randint(0, _VOCAB - 10, (1, 5, _PERSONA_LEN))
        persona_can_idx = torch.randint(0, _PERSONA_LEN, (1, 5, 1))

        knowledge_input_ids = torch.randint(0, _VOCAB - 10, (1, 10, _KNOWLEDGE_LEN))
        knowledge_can_idx = torch.randint(0, _KNOWLEDGE_LEN, (1, 10, 1))

        tot_knowledge = torch.randint(0, _VOCAB - 10, (1, 5, _TOT_KNOW_LEN))
        tot_knowledge_token_ids = torch.zeros(1, 5, _TOT_KNOW_LEN, dtype=torch.long)
        tot_knowledge_eos = torch.full((1, 5, 1), _TOT_KNOW_LEN - 1, dtype=torch.long)

        return self.model(
            input_ids=input_ids,
            input_eos=input_eos,
            token_type_ids=token_type_ids,
            only_dial_input_ids=only_dial_input_ids,
            only_dial_token_type_ids=only_dial_token_type_ids,
            persona_input_ids=persona_input_ids,
            persona_can_idx=persona_can_idx,
            knowledge_input_ids=knowledge_input_ids,
            knowledge_can_idx=knowledge_can_idx,
            tot_knowledge=tot_knowledge,
            tot_knowledge_token_ids=tot_knowledge_token_ids,
            tot_knowledge_eos=tot_knowledge_eos,
            training=False,
        )


def build_focus_gpt2():
    """Tiny random-init GPT2PK_ctxt (FoCus persona+knowledge grounding head)."""
    model = FoCusGPT2Wrapper()
    model.eval()
    return model


def example_input_focus_gpt2():
    return torch.zeros(1)


MENAGERIE_ENTRIES = [
    (
        "FoCus (Persona+Knowledge Grounded GPT-2)",
        "build_focus_gpt2",
        "example_input_focus_gpt2",
        2022,
        "vendored-pytorch",
    ),
]
