"""Record chosen sites across one real Hugging Face ``generate()`` call.

DOCUMENTED-UNSTABLE: the ``generate_kwargs=`` door on :func:`torchlens.record`,
:class:`GenerationRecording` and :class:`GenerationStep` ship as unstable
spellings.

``tl.record(model, input_ids, save=..., generate_kwargs={...})`` runs the
model's own ``generate`` once, with its KV cache, and records the selected
sites at every forward that generation runs: the prefill, then one forward per
decode step. One fastlog :class:`~torchlens.fastlog.Recorder` spans the whole
call; each model forward is one Recorder pass. With the cache on, a decode
forward computes only the new position, so each step records only that
position and the total cost grows linearly with the number of generated
tokens. Step 0 is the prefill forward, the same convention as the bound decode
step (``torchlens.intervention.at_step``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import nn

from .._errors import InvalidArgumentError
from ._recorder import Recorder
from .types import ActivationRecord, Recording

#: Generation settings that decide which tokens are produced, recorded with the result.
TOKEN_CHOICE_SETTINGS: tuple[str, ...] = (
    "do_sample",
    "num_beams",
    "num_return_sequences",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "min_p",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "max_new_tokens",
    "min_new_tokens",
    "max_length",
    "eos_token_id",
    "pad_token_id",
    "use_cache",
)

#: What a generation recording never holds, disclosed on every result.
NOT_RECORDED: tuple[str, ...] = (
    "KV-cache tensors (past_key_values) are not saved; only the selected sites are.",
    "Logits processing, sampling and stopping run between forwards, outside any recorded forward.",
    "The last generated token is never fed back through the model, so no step computes "
    "activations at its position.",
    "A decode step sees earlier positions only through the cache; their activations are "
    "the earlier steps' records.",
)

#: Recorder options that cannot hold across a generation, with the reason for each.
_UNSUPPORTED_RECORD_OPTIONS: tuple[tuple[str, str], ...] = (
    ("intervene", "interventions inside a recorded generation are not built"),
    ("halt", "a halted forward returns no output for generate() to continue from"),
)

_ASSISTED_KEYS: tuple[str, ...] = ("assistant_model", "prompt_lookup_num_tokens")


@dataclass(frozen=True)
class GenerationStep:
    """One forward of a recorded generation (DOCUMENTED-UNSTABLE).

    Attributes
    ----------
    step:
        0 for the prefill forward, then 1, 2, ... for each decode forward.
    pass_index:
        The Recorder pass that holds this forward's records
        (``recording.by_pass[pass_index]``).
    kind:
        ``"prefill"`` or ``"decode"``.
    positions:
        Absolute sequence positions this forward computed; a cached decode
        step computes exactly one.
    n_ops:
        Operations this forward executed (every capture event, retained or not).
    n_records:
        Activation records retained for this forward.
    saved_bytes:
        Bytes of retained in-memory payloads for this forward.
    """

    step: int
    pass_index: int
    kind: Literal["prefill", "decode"]
    positions: range
    n_ops: int
    n_records: int
    saved_bytes: int


@dataclass(frozen=True)
class GenerationRecording:
    """The selected sites of every forward one ``generate()`` call ran (DOCUMENTED-UNSTABLE).

    Attributes
    ----------
    recording:
        The multi-pass fastlog recording; one pass per model forward.
    steps:
        One :class:`GenerationStep` per forward, in execution order.
    sequences:
        The prompt followed by the generated tokens, as ``generate()`` returned them.
    prompt_length:
        Number of prompt positions.
    settings:
        The generation settings that decide which tokens are produced (resolved
        from the call and the model's generation config), plus the torch seed
        and a digest of the torch RNG state at the start of generation.
    not_recorded:
        What the recording never holds, in plain words.
    """

    recording: Recording
    steps: tuple[GenerationStep, ...]
    sequences: torch.Tensor
    prompt_length: int
    settings: Mapping[str, Any]
    not_recorded: tuple[str, ...] = NOT_RECORDED

    @property
    def n_steps(self) -> int:
        """Return the number of recorded forwards (prefill included)."""

        return len(self.steps)

    def records_at(self, step: int) -> list[ActivationRecord]:
        """Return the activation records of one generation step.

        Parameters
        ----------
        step:
            0 for the prefill forward, ``k`` for the k-th decode forward.

        Returns
        -------
        list[ActivationRecord]
            The step's retained records, in execution order.
        """

        if not 0 <= step < len(self.steps):
            raise InvalidArgumentError(
                f"generation step {step} does not exist; this recording has steps "
                f"0 to {len(self.steps) - 1} (0 is the prefill)",
                code="record_generation_step_missing",
                remedy=f"pass a step from 0 to {len(self.steps) - 1}",
            )
        pass_index = self.steps[step].pass_index
        records = self.recording.records
        return [records[index] for index in self.recording.by_pass.get(pass_index, [])]

    def step_of_position(self, position: int) -> GenerationStep:
        """Return the step whose forward computed an absolute sequence position.

        Parameters
        ----------
        position:
            Absolute position in ``sequences`` (0 is the first prompt token).

        Returns
        -------
        GenerationStep
            The step that computed ``position``.
        """

        for entry in self.steps:
            if position in entry.positions:
                return entry
        last = self.steps[-1].positions.stop - 1 if self.steps else -1
        raise InvalidArgumentError(
            f"no recorded forward computed position {position}; recorded positions are "
            f"0 to {last} (the last generated token is never fed back through the model)",
            code="record_generation_step_missing",
            remedy=f"pass a position from 0 to {last}",
        )


def _refuse(reason: str, remedy: str) -> InvalidArgumentError:
    """Build the one generation-recorder refusal."""

    return InvalidArgumentError(
        f"tl.record(..., generate_kwargs=...) refuses: {reason}",
        code="record_generation_unsupported",
        remedy=remedy,
        reason=reason,
    )


def _setting(key: str, generate_kwargs: Mapping[str, Any], base: Any) -> Any:
    """Resolve one generation setting from the call, else the generation config."""

    if key in generate_kwargs:
        return generate_kwargs[key]
    return getattr(base, key, None)


def _resolved_settings(model: nn.Module, generate_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return the token-choice settings this generation will run with."""

    base = generate_kwargs.get("generation_config", getattr(model, "generation_config", None))
    return {key: _setting(key, generate_kwargs, base) for key in TOKEN_CHOICE_SETTINGS}


def _positional(input_args: Any) -> tuple[Any, ...]:
    """Return the prompt's positional arguments as a tuple."""

    if input_args is None:
        return ()
    if isinstance(input_args, (list, tuple)):
        return tuple(input_args)
    return (input_args,)


def _batch_size(input_args: Any, input_kwargs: Mapping[str, Any]) -> int | None:
    """Return the prompt batch size, when the prompt is a tensor."""

    candidates = [*_positional(input_args), input_kwargs.get("input_ids")]
    for value in candidates:
        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.dim() > 1 else 1
    return None


def preflight_generation(
    model: nn.Module,
    input_args: Any,
    input_kwargs: Mapping[str, Any],
    generate_kwargs: Mapping[str, Any],
    record_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse a generation the recorder cannot hold; return the resolved settings.

    Raises
    ------
    InvalidArgumentError
        ``record_generation_unsupported`` naming what is unsupported.
    """

    if not callable(getattr(model, "generate", None)):
        raise _refuse(
            f"{type(model).__name__} has no callable generate(); the recorder runs the "
            "model's own generate and never invents one",
            "record one forward per step yourself: tl.record(model, ids, save=...)",
        )
    config = getattr(model, "config", None)
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise _refuse(
            "encoder-decoder models are not supported (the encoder runs outside the "
            "decoder forwards the recorder holds)",
            "record the decoder forwards yourself: tl.record(model, inputs, save=...)",
        )
    for option, reason in _UNSUPPORTED_RECORD_OPTIONS:
        if record_options.get(option) is not None:
            raise _refuse(reason, f"drop {option}= from the tl.record call")
    if record_options.get("on_forward_error", "raise") != "raise":
        raise _refuse(
            "partial forward-error modes cannot continue a generation",
            'drop on_forward_error= (the default "raise" holds)',
        )
    settings = _resolved_settings(model, generate_kwargs)
    if settings["use_cache"] is False:
        raise _refuse(
            "use_cache=False recomputes the whole prefix every step, so the cost is no "
            "longer linear in generated tokens",
            "drop use_cache=False from generate_kwargs",
        )
    if (settings["num_beams"] or 1) > 1:
        raise _refuse(
            "beam search (num_beams > 1) is not supported",
            "drop num_beams= from generate_kwargs (greedy or sampled decoding)",
        )
    if (settings["num_return_sequences"] or 1) > 1:
        raise _refuse(
            "more than one return sequence is not supported",
            "drop num_return_sequences= from generate_kwargs",
        )
    for key in _ASSISTED_KEYS:
        if generate_kwargs.get(key) is not None:
            raise _refuse(
                f"assisted decoding ({key}=) is not supported",
                f"drop {key}= from generate_kwargs",
            )
    batch = _batch_size(input_args, input_kwargs)
    if batch is not None and batch > 1:
        raise _refuse(
            f"a batch of {batch} prompts is not supported (padding-aware batching is not built)",
            "record one prompt at a time: input_ids[i : i + 1]",
        )
    return settings


def _rng_digest() -> str:
    """Return a short digest of the CPU torch RNG state."""

    state = torch.get_rng_state().numpy().tobytes()
    return hashlib.sha256(state).hexdigest()[:16]


def _step_window(kwargs: Mapping[str, Any], step: int) -> range:
    """Return the absolute positions one generate-issued forward computes."""

    ids = kwargs.get("input_ids")
    if not isinstance(ids, torch.Tensor):
        ids = kwargs.get("inputs_embeds")
        length = int(ids.shape[-2]) if isinstance(ids, torch.Tensor) else 0
    else:
        length = int(ids.shape[-1])
    cache = kwargs.get("past_key_values")
    start = 0
    if cache is not None and callable(getattr(cache, "get_seq_length", None)):
        start = int(cache.get_seq_length())
    elif isinstance(kwargs.get("cache_position"), torch.Tensor):
        start = int(kwargs["cache_position"].reshape(-1)[0])
    if step > 0 and start == 0:
        raise _refuse(
            "the model ran a decode step without a KV cache (it recomputed the whole "
            "prefix), so the cost is no longer linear in generated tokens",
            "use a model whose generate() passes past_key_values to its forward",
        )
    return range(start, start + length)


@contextmanager
def _forwards_through(
    model: nn.Module, recorder: Recorder, windows: list[tuple[int, range]]
) -> Iterator[None]:
    """Route each outermost ``model.forward`` call through ``recorder.log``."""

    had_instance_forward = "forward" in model.__dict__
    saved_forward = model.__dict__.get("forward")
    original: Callable[..., Any] = model.forward
    depth = [0]

    def forward(*args: Any, **kwargs: Any) -> Any:
        if depth[0]:
            return original(*args, **kwargs)
        windows.append((recorder._next_pass_index, _step_window(kwargs, len(windows))))
        depth[0] += 1
        try:
            return recorder.log(args, kwargs)
        finally:
            depth[0] -= 1

    object.__setattr__(model, "forward", forward)
    try:
        yield
    finally:
        if had_instance_forward:
            object.__setattr__(model, "forward", saved_forward)
        else:
            del model.__dict__["forward"]


def _steps(recording: Recording, windows: list[tuple[int, range]]) -> tuple[GenerationStep, ...]:
    """Build one :class:`GenerationStep` per recorded forward."""

    ops_by_pass: dict[int, int] = {}
    for row in recording.raw_metadata():
        pass_index = int(row["pass_index"])
        ops_by_pass[pass_index] = ops_by_pass.get(pass_index, 0) + 1
    steps = []
    for step, (pass_index, window) in enumerate(windows):
        indexes = recording.by_pass.get(pass_index, [])
        payloads = (recording.records[index].ram_payload for index in indexes)
        steps.append(
            GenerationStep(
                step=step,
                pass_index=pass_index,
                kind="prefill" if step == 0 else "decode",
                positions=window,
                n_ops=ops_by_pass.get(pass_index, 0),
                n_records=len(indexes),
                saved_bytes=sum(
                    payload.numel() * payload.element_size()
                    for payload in payloads
                    if payload is not None
                ),
            )
        )
    return tuple(steps)


def _sequences(output: Any) -> torch.Tensor:
    """Return the token sequences from a ``generate()`` return value."""

    if isinstance(output, torch.Tensor):
        return output
    sequences = getattr(output, "sequences", None)
    if isinstance(sequences, torch.Tensor):
        return sequences
    raise TypeError(f"generate() returned {type(output).__name__} without token sequences")


def record_generation(
    model: nn.Module,
    input_args: Any,
    input_kwargs: Mapping[str, Any] | None,
    generate_kwargs: Mapping[str, Any],
    recorder_kwargs: dict[str, Any],
) -> tuple[Any, GenerationRecording]:
    """Run ``model.generate`` once and record every forward it runs.

    Parameters
    ----------
    model:
        A model with a Hugging Face style ``generate`` that passes its KV cache
        to ``forward``.
    input_args, input_kwargs:
        Prompt arguments forwarded to ``generate``.
    generate_kwargs:
        Generation settings forwarded verbatim to ``generate``.
    recorder_kwargs:
        The ``Recorder`` options of the ``tl.record`` call.

    Returns
    -------
    tuple[Any, GenerationRecording]
        ``generate()``'s own return value and the recording.
    """

    prompt_kwargs = dict(input_kwargs or {})
    settings = preflight_generation(
        model, input_args, prompt_kwargs, generate_kwargs, recorder_kwargs
    )
    random_seed = recorder_kwargs.pop("random_seed", None)
    if random_seed is not None:
        # Seed once for the whole generation; a per-forward reseed would change
        # which tokens a sampled generation produces.
        torch.manual_seed(random_seed)
    settings = {
        **settings,
        "random_seed": random_seed,
        "torch_initial_seed": torch.initial_seed(),
        "torch_rng_state_sha256": _rng_digest(),
    }
    windows: list[tuple[int, range]] = []
    with Recorder(model, **recorder_kwargs) as recorder:
        with _forwards_through(model, recorder, windows):
            generate: Callable[..., Any] = getattr(model, "generate")
            output = generate(*_positional(input_args), **prompt_kwargs, **generate_kwargs)
    recording = recorder.recording
    steps = _steps(recording, windows)
    prompt_length = steps[0].positions.stop if steps else 0
    return output, GenerationRecording(
        recording=recording,
        steps=steps,
        sequences=_sequences(output),
        prompt_length=prompt_length,
        settings=MappingProxyType(settings),
    )
