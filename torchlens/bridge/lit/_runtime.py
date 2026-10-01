"""Tokenization, padding conventions, state guard, and the traced forward.

The padding conventions are ONE coherent set per path (LIT-panel memo D9):
classification right-pads throughout; the causal-LM adapter uses TWO
tokenizations -- LEFT-padded for one batched ``generate()`` (right-padded
batched generation fails by fluently echoing the prompt) and RIGHT-padded for
one traced forward with ``use_cache=False`` (there ``position_ids`` are
unnecessary and the naive pooling index is correct). Mixing halves is silent
wrongness whose worst measured case is cosine 0.40 -- which is why the gates
assert ``cos == 1.0``, never "close enough". Padding is applied manually from
per-example encodings so the user's tokenizer object is never mutated
(memo D18 hygiene).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from . import _refusals


def validate_tokenizer(tokenizer: Any) -> None:
    """Validate the tokenizer contract at construction time.

    Parameters
    ----------
    tokenizer:
        The user's tokenizer; must be callable, decode ids, and carry a pad or
        eos token id so batches can be padded without mutating it.

    Raises
    ------
    torchlens._errors.InvalidArgumentError
        Code ``lit_tokenizer_invalid``.
    """

    if tokenizer is None or not callable(tokenizer):
        _refusals.refuse_tokenizer_invalid(
            f"tokenizer is {type(tokenizer).__name__}; the bridge needs a callable "
            "HF-style tokenizer to execute LIT's edited text",
            remedy="Pass the checkpoint's tokenizer as the second positional argument.",
        )
    if pad_token_id(tokenizer) is None:
        _refusals.refuse_tokenizer_invalid(
            "tokenizer exposes neither pad_token_id nor eos_token_id, so ragged LIT "
            "requests cannot be padded",
            remedy="Use a tokenizer with a pad or eos token (the bridge pads with "
            "pad_token_id, falling back to eos_token_id, without mutating the "
            "tokenizer).",
        )


def pad_token_id(tokenizer: Any) -> int | None:
    """Return the padding id: ``pad_token_id``, else ``eos_token_id``.

    Parameters
    ----------
    tokenizer:
        The user's tokenizer.

    Returns
    -------
    int | None
        The id to pad with, or None when the tokenizer has neither.
    """

    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    return int(eos) if eos is not None else None


def encode_texts(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    """Encode each text unpadded (specials included, truncated to the model max).

    Parameters
    ----------
    tokenizer:
        The user's tokenizer.
    texts:
        One string per LIT example.

    Returns
    -------
    list[list[int]]
        Per-example token id lists.
    """

    return [list(tokenizer(text, truncation=True)["input_ids"]) for text in texts]


def pad_batch(
    id_lists: list[list[int]],
    pad_id: int,
    side: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-example id lists into ``input_ids`` and ``attention_mask``.

    Parameters
    ----------
    id_lists:
        Per-example token ids.
    pad_id:
        Padding token id.
    side:
        ``"right"`` or ``"left"``.
    device:
        Target device for both tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(input_ids, attention_mask)``, both ``[batch, max_len]`` int64.
    """

    width = max(len(ids) for ids in id_lists)
    rows: list[list[int]] = []
    masks: list[list[int]] = []
    for ids in id_lists:
        fill = [pad_id] * (width - len(ids))
        ones = [1] * len(ids)
        zeros = [0] * (width - len(ids))
        if side == "right":
            rows.append(ids + fill)
            masks.append(ones + zeros)
        else:
            rows.append(fill + ids)
            masks.append(zeros + ones)
    input_ids = torch.tensor(rows, dtype=torch.int64, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.int64, device=device)
    return input_ids, attention_mask


def model_device(net: nn.Module) -> torch.device:
    """Return the device request tensors should be created on.

    Parameters
    ----------
    net:
        The live model.

    Returns
    -------
    torch.device
        The first parameter's device, or CPU for parameterless modules.
    """

    for param in net.parameters():
        return param.device
    return torch.device("cpu")


@contextmanager
def operation_state(net: nn.Module) -> Iterator[None]:
    """Enter eval mode for one operation and restore every module's prior mode.

    The restore runs after success AND failure (memo D18); parameters,
    gradients, the tokenizer, and device placement are never written by the
    bridge, so recording per-module ``training`` flags is the complete guard.

    Parameters
    ----------
    net:
        The live model.

    Yields
    ------
    None
        With the model in eval mode.
    """

    modes = [(module, module.training) for module in net.modules()]
    net.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def traced_forward(net: nn.Module, input_kwargs: dict[str, Any], save: Any) -> Any:
    """Run one traced forward and demand a COMPLETE settlement.

    Parameters
    ----------
    net:
        The live model, already under the operation-state guard.
    input_kwargs:
        Forward keyword arguments (tensors plus flags like ``use_cache``).
    save:
        TorchLens ``save=`` selection (None keeps the save-everything default).

    Returns
    -------
    Any
        The finished TorchLens ``Trace``.

    Raises
    ------
    torchlens._errors.RecordBindingError
        Code ``lit_capture_incomplete`` when the capture settles non-COMPLETE.
    """

    from torchlens import trace as tl_trace

    log = tl_trace(net, [], input_kwargs=input_kwargs, save=save)
    status = getattr(log.outcome, "status", None)
    if getattr(status, "name", str(status)) != "COMPLETE":
        _refusals.refuse_capture_incomplete(status)
    return log


def extract_logits(task: str, output: Any) -> torch.Tensor:
    """Extract the logits tensor from a reconstructed model output.

    Parameters
    ----------
    task:
        The adapter task (for teaching messages).
    output:
        ``Trace.reconstruct_output()``'s value: a tensor, an HF ``ModelOutput``
        with ``.logits``, or a tuple whose first element is the logits.

    Returns
    -------
    torch.Tensor
        The logits tensor.

    Raises
    ------
    torchlens._errors.RecordBindingError
        Code ``lit_model_output_unsupported``.
    """

    value = output
    if hasattr(value, "logits"):
        value = value.logits
    elif isinstance(value, (tuple, list)) and value:
        value = value[0]
    if not isinstance(value, torch.Tensor):
        _refusals.refuse_model_output_unsupported(
            task, f"reconstructed output is {type(output).__name__} with no tensor logits"
        )
    expected = 2 if task == "classification" else 3
    if value.ndim != expected:
        _refusals.refuse_model_output_unsupported(
            task,
            f"logits have rank {value.ndim} (shape {tuple(value.shape)}); "
            f"this task needs rank {expected}",
        )
    return value


def unpadded_lengths(attention_mask: torch.Tensor) -> list[int]:
    """Return per-row real-token counts from an attention mask.

    Parameters
    ----------
    attention_mask:
        ``[batch, tokens]`` mask.

    Returns
    -------
    list[int]
        Number of real tokens per row.
    """

    return [int(n) for n in attention_mask.sum(dim=1).tolist()]
