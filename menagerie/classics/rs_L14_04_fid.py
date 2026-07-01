# SOURCE: vendored from facebookresearch/FiD @ fe769f30e3714e22476910ee39ea0054dd7921de
# https://raw.githubusercontent.com/facebookresearch/FiD/fe769f30e3714e22476910ee39ea0054dd7921de/src/model.py
#
# Izacard & Grave 2021 "Leveraging Passage Retrieval with Generative Models for Open
# Domain Question Answering" (Fusion-in-Decoder, ICLR 2021 / EACL 2021). Real
# architecture: subclasses HuggingFace's real `T5ForConditionalGeneration`, but wraps
# the encoder in an `EncoderWrapper` that folds the "number of retrieved passages"
# dimension into the batch dimension before running each passage independently
# through the (checkpoint-wrappable) T5 encoder stack, then reshapes the per-passage
# encoder outputs back out and concatenates them along the sequence-length axis --
# so the T5 decoder attends, via cross-attention, over the fused concatenation of
# every passage's encoding at once ("fusion in decoder"). This is the actual
# `nn.Module` architecture from the official repo, not a re-description: the encoder
# reshape/wrap logic, checkpoint wrapper, and decoder/generator plumbing are the real
# code.
#
# Minimal, non-architectural changes made (only HF `transformers` version-drift
# shims; no computation changed):
#   - The 2021-era `transformers` this repo targeted had T5 encoder sub-blocks return
#     plain tuples; current `transformers` (4.x) T5Stack expects/returns a
#     `BaseModelOutputWithPastAndCrossAttentions` and the top-level
#     `T5ForConditionalGeneration.forward` reads `encoder_outputs.last_hidden_state`
#     (attribute access, not tuple indexing). `EncoderWrapper.forward` now wraps its
#     final concatenated hidden state in `BaseModelOutputWithPastAndCrossAttentions`
#     instead of returning a bare tuple -- an output-type adapter for the current HF
#     API, not an architecture change (same tensor, same shape, same fusion).
#   - Dropped the retriever-training-only pieces (`RetrieverConfig`, `Retriever`,
#     `overwrite_forward_crossattention`/`cross_attention_forward`,
#     `get_crossattention_scores`) that patch private T5 cross-attention internals
#     for cross-attention-score distillation -- optional debug/distillation-only
#     surface unrelated to the traced FiD reader forward pass; the real `FiDT5`
#     reader architecture (`wrap_encoder`/`EncoderWrapper`/`CheckpointWrapper`/
#     `forward`) is kept intact and verbatim in spirit (only the output-type shim
#     above touches behavior).

import torch
import transformers
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        self.n_passages = 1
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        last_hidden_state = outputs[0].view(bsz, self.n_passages * passage_length, -1)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state)


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, *rest, **kwargs):
        # NOTE (compat shim, not an architecture change): the 2021-era transformers
        # this repo targeted called each encoder block with exactly 3 positional
        # args (hidden_states, attention_mask, position_bias). Current transformers
        # T5Stack passes additional positional args (e.g. encoder_hidden_states,
        # encoder_extended_attention_mask, encoder_decoder_position_bias for
        # gradient-checkpointing compatibility) even on encoder-only blocks. The
        # trailing positional args are captured in `*rest` and forwarded through
        # unchanged -- same passthrough wrapper, just accepting the current T5
        # block calling convention instead of the old fixed 3-positional-arg one.
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [], dtype=torch.float, device=output[0].device, requires_grad=True
                )
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias,
                *rest,
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, *rest, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


class FiDTraceWrapper(nn.Module):
    """Thin positional-args adapter around FiDT5 for trace-harness calling
    convention (the harness calls the example/build pair with two positional
    args; the real FiDT5.forward takes input_ids/attention_mask plus
    keyword-only T5 args like decoder_input_ids via **kwargs). No architecture
    change -- forwards straight into the real FiDT5 module."""

    def __init__(self, fid_model):
        super().__init__()
        self.fid = fid_model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        return self.fid(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )


def build_fidt5():
    config = transformers.T5Config(
        vocab_size=64,
        d_model=16,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=8,
        pad_token_id=0,
        decoder_start_token_id=0,
    )
    model = FiDT5(config)
    model.eval()
    return FiDTraceWrapper(model)


def example_input_fidt5():
    n_passages = 2
    input_ids = torch.randint(1, 64, (1, n_passages, 6))
    attention_mask = torch.ones(1, n_passages, 6, dtype=torch.long)
    decoder_input_ids = torch.randint(1, 64, (1, 4))
    return (input_ids, attention_mask, decoder_input_ids)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "FiD (Fusion-in-Decoder)",
        "build_fidt5",
        "example_input_fidt5",
        2021,
        "vendored",
    ),
]
