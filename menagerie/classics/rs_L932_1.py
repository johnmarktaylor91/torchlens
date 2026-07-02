# SOURCE: vendored from datamol-io/safe @ main (safe/trainer/model.py)
#
# SAFE-GPT (SAFEDoubleHeadsModel): a dual-head GPT2 model for the SAFE (Sequential
# Attachment-based Fragment Embedding) molecular string representation. Adds a
# PropertyHead multi-task regression head alongside the standard GPT2 LM head.
# Covers both queue candidates "SAFE encoding" and "SAFE-GPT" (same repo/paper,
# datamol-io/safe -- POTENTIAL_DEDUP noted in queue.tsv; one vendored model covers both).
#
# Only base-lib deps used here (torch, transformers) -- the real `safe` package's extra
# deps (datamol, fsspec, loguru, tokenizers-trainer utilities) are for its custom
# tokenizer/data pipeline, not the model class itself, so they are omitted.

from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import GPT2Config, GPT2DoubleHeadsModel, PretrainedConfig
from transformers.activations import get_activation
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

MENAGERIE_ZOO = "vendored-pytorch"


class PropertyHead(torch.nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)

            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string, or `None` to add no activation.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "cls_index")
        self.summary = torch.nn.Identity()
        last_hidden_size = config.hidden_size

        if getattr(config, "summary_hidden_size", None) and config.summary_hidden_size > 0:
            self.summary = nn.Linear(config.hidden_size, config.summary_hidden_size)
            last_hidden_size = config.summary_hidden_size

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = (
            get_activation(activation_string) if activation_string else nn.Identity()
        )

        self.out = torch.nn.Identity()
        if getattr(config, "num_labels", None) and config.num_labels > 0:
            num_labels = config.num_labels
            self.out = nn.Linear(last_hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        cls_index: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states: `torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`)
                The hidden states of the last layer.
            cls_index: `torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]`
                where ... are optional leading dimensions of `hidden_states`, *optional*
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            batch_size = hidden_states.shape[0]
            output = hidden_states.squeeze()[torch.arange(batch_size), cls_index]
        else:
            raise NotImplementedError

        output = self.summary(output)
        output = self.activation(output)
        return self.out(output)


class SAFEDoubleHeadsModel(GPT2DoubleHeadsModel):
    """The safe model is a dual head GPT2 model with a language modeling head and an optional multi-task regression head"""

    def __init__(self, config):
        self.num_labels = getattr(config, "num_labels", None)
        super().__init__(config)
        self.config.num_labels = self.num_labels
        del self.multiple_choice_head
        self.multiple_choice_head = PropertyHead(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inputs: Optional[Any] = None,  # do not remove because of trainer
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, GPT2DoubleHeadsModelOutput]:
        r"""

        Args:
            mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
                Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
                1]`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to
                `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
            mc_labels (`torch.LongTensor` of shape `(batch_size, n_tasks)`, *optional*):
                Labels for computing the supervized loss for regularization.
            inputs: List of inputs, put here because the trainer removes information not in signature
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if mc_token_ids is None and self.config.pad_token_id is not None and input_ids is not None:
            mc_token_ids = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(
                lm_logits.device
            )

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        mc_loss = None
        mc_logits = None
        if mc_labels is not None and getattr(self.config, "num_labels", 0) > 0:
            mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
            mc_labels = mc_labels.to(mc_logits.device)
            loss_fct = MSELoss()
            mc_loss = loss_fct(
                mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1, mc_logits.size(-1))
            )

        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            return (
                lm_loss,
                mc_loss,
            ) + output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


def build_safe_gpt():
    """Tiny SAFE-GPT (SAFEDoubleHeadsModel) for tracing: small GPT2 backbone + property head."""
    config = GPT2Config(
        vocab_size=200,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        num_labels=3,
        summary_type="cls_index",
        pad_token_id=0,
    )
    model = SAFEDoubleHeadsModel(config)
    model.eval()
    return model


def example_input_safe_gpt():
    input_ids = torch.randint(1, 200, (1, 16), dtype=torch.long)
    return (input_ids,)


MENAGERIE_ENTRIES = [
    ("SAFE-GPT", build_safe_gpt, example_input_safe_gpt, 2024, "vendored-pytorch"),
]
