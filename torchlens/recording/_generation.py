"""Record chosen sites across one real Hugging Face ``generate()`` call.

DOCUMENTED-UNSTABLE: the ``generate_kwargs=`` door on :func:`torchlens.record`,
:class:`GenerationRecording` and :class:`GenerationStep` ship as unstable
spellings.

``tl.record(model, input_ids, save=..., generate_kwargs={...})`` runs the
model's own ``generate`` once, with its KV cache, and records the selected
sites at every forward that generation runs: the prefill, then one forward per
decode step. One fastlog :class:`~torchlens.recording.Recorder` spans the whole
call; each model forward is one Recorder pass. With the cache on, a decode
forward computes only the new position, so each step records only that
position and the total cost grows linearly with the number of generated
tokens. Step 0 is the prefill forward, the same convention as the bound decode
step (``torchlens.at_step``).
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, cast

import torch
from torch import nn

from .._errors import InvalidArgumentError
from .._ir.selector_eval import normalize_selector_like
from ._capture_door import _capture_without_seed, _unfold_capture_options
from ._generation_options import GenerationOptions
from ._generation_stream import TokenSeries, _StepSink, site_selector
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
    "KV-cache tensors (past_key_values) are not saved; only the selected sites are "
    "(GenerationOptions(kv_sites=) keeps the declared layers' new slices per token).",
    "Logits processing, sampling and stopping run between forwards, outside any recorded forward.",
    "The last generated token is never fed back through the model, so no step computes "
    "activations at its position.",
    "A decode step sees earlier positions only through the cache; their activations are "
    "the earlier steps' records.",
)

#: Recorder options that cannot hold across a generation, with the reason for each.
_UNSUPPORTED_RECORD_OPTIONS: tuple[tuple[str, str], ...] = (
    ("intervene", "interventions inside a recorded generation are not built"),
    ("stop_after", "a stopped forward returns no output for generate() to continue from"),
)

_ASSISTED_KEYS: tuple[str, ...] = ("assistant_model", "prompt_lookup_num_tokens")


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
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
    options:
        The :class:`GenerationOptions` the recording ran with (``None`` for the
        plain door, which keeps every selected site at every token).
    """

    recording: Recording
    steps: tuple[GenerationStep, ...]
    sequences: torch.Tensor
    prompt_length: int
    settings: Mapping[str, Any]
    not_recorded: tuple[str, ...] = NOT_RECORDED
    options: GenerationOptions | None = None
    _sink: Any = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def output(self) -> Any:
        """Return what ``generate()`` returned, exactly (``recording.output``).

        Held in memory only; a recording loaded from disk refuses with
        ``output_not_stored``.
        """

        return self.recording.output

    @property
    def n_steps(self) -> int:
        """Return the number of recorded forwards (prefill included)."""

        return len(self.steps)

    @property
    def series_names(self) -> tuple[str, ...]:
        """Return the per-token series this recording answers (reducer sites, KV slices)."""

        return () if self._sink is None else tuple(self._sink.series_names)

    @property
    def peak_saved_bytes(self) -> int:
        """Return the most payload bytes held in memory at once during the generation."""

        if self._sink is None:
            return sum(step.saved_bytes for step in self.steps)
        return int(self._sink.peak_bytes)

    @property
    def evicted_steps(self) -> Mapping[int, str]:
        """Return the forwards the ring buffer evicted, step to ``"spilled"`` or ``"dropped"``."""

        return MappingProxyType({} if self._sink is None else dict(self._sink.evicted))

    def series(self, name: str) -> TokenSeries:
        """Return one per-token series: a reducer site's values or a KV slice.

        Parameters
        ----------
        name:
            A site address passed in ``GenerationOptions(reducers=)`` or a KV
            series name, ``kv.<layer>.key`` or ``kv.<layer>.value``.

        Returns
        -------
        TokenSeries
            The values over every readable position (spilled forwards are read
            back from disk; dropped ones are absent from the front).
        """

        if self._sink is None:
            raise InvalidArgumentError(
                f"no per-token series named {name!r}: this recording ran without "
                "generation=GenerationOptions(...)",
                code="record_generation_step_missing",
                remedy="pass generation=GenerationOptions(reducers={...}) or kv_sites=",
            )
        return cast("TokenSeries", self._sink.series(name))

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


#: The message shape every generation-recorder refusal shares.
_REFUSAL = "tl.record(..., generate_kwargs=...) refuses: {}"


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


def _model_refusal(model: nn.Module, record_options: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return ``(reason, remedy)`` when the model or record options cannot hold a generation."""

    if not callable(getattr(model, "generate", None)):
        return (
            f"{type(model).__name__} has no callable generate(); the recorder runs the "
            "model's own generate and never invents one",
            "record one forward per step yourself: tl.record(model, ids, save=...)",
        )
    if bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False)):
        return (
            "encoder-decoder models are not supported (the encoder runs outside the "
            "decoder forwards the recorder holds)",
            "record the decoder forwards yourself: tl.record(model, inputs, save=...)",
        )
    for option, reason in _UNSUPPORTED_RECORD_OPTIONS:
        if record_options.get(option) is not None:
            return reason, f"drop {option}= from the tl.record call"
    if record_options.get("on_forward_error", "raise") != "raise":
        return (
            "partial forward-error modes cannot continue a generation",
            'drop on_forward_error= (the default "raise" holds)',
        )
    return None


def _decoding_refusal(
    settings: Mapping[str, Any], generate_kwargs: Mapping[str, Any], batch: int | None
) -> tuple[str, str] | None:
    """Return ``(reason, remedy)`` when the decoding settings cannot be recorded."""

    if settings["use_cache"] is False:
        return (
            "use_cache=False recomputes the whole prefix every step, so the cost is no "
            "longer linear in generated tokens",
            "drop use_cache=False from generate_kwargs",
        )
    if (settings["num_beams"] or 1) > 1:
        return (
            "beam search (num_beams > 1) is not supported",
            "drop num_beams= from generate_kwargs (greedy or sampled decoding)",
        )
    if (settings["num_return_sequences"] or 1) > 1:
        return (
            "more than one return sequence is not supported",
            "drop num_return_sequences= from generate_kwargs",
        )
    for key in _ASSISTED_KEYS:
        if generate_kwargs.get(key) is not None:
            return (
                f"assisted decoding ({key}=) is not supported",
                f"drop {key}= from generate_kwargs",
            )
    if batch is not None and batch > 1:
        return (
            f"a batch of {batch} prompts is not supported (padding-aware batching is not built)",
            "record one prompt at a time: input_ids[i : i + 1]",
        )
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

    settings = _resolved_settings(model, generate_kwargs)
    refusal = _model_refusal(model, record_options) or _decoding_refusal(
        settings, generate_kwargs, _batch_size(input_args, input_kwargs)
    )
    if refusal is not None:
        reason, remedy = refusal
        raise InvalidArgumentError(
            _REFUSAL.format(reason),
            code="record_generation_unsupported",
            remedy=remedy,
            reason=reason,
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
        reason = (
            "the model ran a decode step without a KV cache (it recomputed the whole "
            "prefix), so the cost is no longer linear in generated tokens"
        )
        raise InvalidArgumentError(
            _REFUSAL.format(reason),
            code="record_generation_unsupported",
            remedy="use a model whose generate() passes past_key_values to its forward",
            reason=reason,
        )
    return range(start, start + length)


class _ForwardRoute:
    """The state of one routed ``model.forward``, kept out of the model's object graph.

    Each capture ends by walking what the model reaches, the closure cells of
    its ``forward`` included, to clear TorchLens tensor metadata. A closure that
    held the Recorder directly would make that walk visit everything recorded so
    far on every pass, so its cost would grow with the step count. The walk
    descends an object only through ``__dict__``; this slotted holder has none.
    """

    __slots__ = ("depth", "original", "recorder", "sink", "windows")

    def __init__(
        self,
        recorder: Recorder,
        windows: list[tuple[int, range]],
        original: Callable[..., Any],
        sink: _StepSink | None = None,
    ) -> None:
        self.recorder = recorder
        self.windows = windows
        self.original = original
        self.sink = sink
        self.depth = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.depth:
            return self.original(*args, **kwargs)
        step = len(self.windows)
        pass_index, window = self.recorder._next_pass_index, _step_window(kwargs, step)
        self.windows.append((pass_index, window))
        self.depth += 1
        try:
            output = self.recorder(args, kwargs)
        finally:
            self.depth -= 1
        if self.sink is not None:
            self.sink.after_pass(step, pass_index, window, kwargs)
        return output


@contextmanager
def _forwards_through(
    model: nn.Module,
    recorder: Recorder,
    windows: list[tuple[int, range]],
    sink: _StepSink | None = None,
) -> Iterator[None]:
    """Route each outermost ``model.forward`` call through the recorder call."""

    had_instance_forward = "forward" in model.__dict__
    saved_forward = model.__dict__.get("forward")
    route = _ForwardRoute(recorder, windows, model.forward, sink)

    def forward(*args: Any, **kwargs: Any) -> Any:
        """Hand the call to the route, which records outermost forwards only."""

        return route(*args, **kwargs)

    object.__setattr__(model, "forward", forward)
    try:
        yield
    finally:
        if had_instance_forward:
            object.__setattr__(model, "forward", saved_forward)
        else:
            del model.__dict__["forward"]


def _steps(
    recording: Recording,
    windows: list[tuple[int, range]],
    bytes_by_step: list[int] | None = None,
) -> tuple[GenerationStep, ...]:
    """Build one :class:`GenerationStep` per recorded forward."""

    ops_by_pass: dict[int, int] = {}
    for row in recording.raw_metadata():
        pass_index = int(row["pass_number"])
        ops_by_pass[pass_index] = ops_by_pass.get(pass_index, 0) + 1
    steps = []
    for step, (pass_index, window) in enumerate(windows):
        indexes = recording.by_pass.get(pass_index, [])
        payloads = (recording.records[index].ram_payload for index in indexes)
        if bytes_by_step is not None and step < len(bytes_by_step):
            saved_bytes = bytes_by_step[step]
        else:
            saved_bytes = sum(
                payload.numel() * payload.element_size()
                for payload in payloads
                if payload is not None
            )
        steps.append(
            GenerationStep(
                step=step,
                pass_index=pass_index,
                kind="prefill" if step == 0 else "decode",
                positions=window,
                n_ops=ops_by_pass.get(pass_index, 0),
                n_records=len(indexes),
                saved_bytes=saved_bytes,
            )
        )
    return tuple(steps)


def _keep_nothing(ctx: Any) -> bool:
    """Keep no activation: a KV-sites-only recording reads only the cache.

    A plain callable, not a selector: a selector that fires zero times warns.
    """

    return False


def _generation_save(save: Any, generation: GenerationOptions) -> Any:
    """Return the save a widened generation recording runs under."""

    if generation.reducers:
        return _save_with_sites(save, tuple(generation.reducers))
    if save is None and generation.kv_sites:
        return _keep_nothing
    return save


def _save_with_sites(save: Any, sites: tuple[str, ...]) -> Any:
    """Return ``save`` widened to a selector that always keeps the declared sites."""

    selector = site_selector(sites)
    if save is None:
        return selector
    return normalize_selector_like(save, lifecycle="capture") | selector


def _max_new_tokens(settings: Mapping[str, Any], prompt_length: int | None) -> int | None:
    """Return how many tokens the generation may add, when the settings say."""

    if settings.get("max_new_tokens") is not None:
        return int(settings["max_new_tokens"])
    if settings.get("max_length") is not None and prompt_length is not None:
        return max(int(settings["max_length"]) - prompt_length, 0)
    return None


def _sequences(output: Any) -> torch.Tensor:
    """Return the token sequences from a ``generate()`` return value."""

    if isinstance(output, torch.Tensor):
        return output
    sequences = getattr(output, "sequences", None)
    if isinstance(sequences, torch.Tensor):
        return sequences
    raise TypeError(f"generate() returned {type(output).__name__} without token sequences")


def _seeded_settings(
    settings: dict[str, Any], recorder_kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Seed torch once for the whole generation.

    Returns ``settings`` with the RNG facts and the recorder options with the seed
    taken out of ``capture=`` (the recorder must not reseed per forward).
    """

    capture = recorder_kwargs.get("capture")
    random_seed = _unfold_capture_options(capture)["seed"]
    if random_seed is not None:
        # Seed once for the whole generation; a per-forward reseed would change
        # which tokens a sampled generation produces.
        torch.manual_seed(random_seed)
        recorder_kwargs = {**recorder_kwargs, "capture": _capture_without_seed(capture)}
    settings = {
        **settings,
        "random_seed": random_seed,
        "torch_initial_seed": torch.initial_seed(),
        "torch_rng_state_sha256": _rng_digest(),
    }
    return settings, recorder_kwargs


def record_generation(
    model: nn.Module,
    input_args: Any,
    input_kwargs: Mapping[str, Any] | None,
    generate_kwargs: Mapping[str, Any],
    recorder_kwargs: dict[str, Any],
    generation: GenerationOptions | None = None,
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
    generation:
        Per-token reducers, KV-cache slices, ring buffer, spill and the
        retained-byte guard (``None`` keeps every selected site, unguarded).

    Returns
    -------
    tuple[Any, GenerationRecording]
        ``generate()``'s own return value and the recording.
    """

    prompt_kwargs = dict(input_kwargs or {})
    settings = preflight_generation(
        model, input_args, prompt_kwargs, generate_kwargs, recorder_kwargs
    )
    settings, recorder_kwargs = _seeded_settings(settings, recorder_kwargs)
    windows: list[tuple[int, range]] = []
    if generation is not None:
        recorder_kwargs["save"] = _generation_save(recorder_kwargs.get("save"), generation)
    sink: _StepSink | None = None
    with Recorder(model, **recorder_kwargs) as recorder:
        if generation is not None:
            sink = _StepSink(generation, recorder, partial(_max_new_tokens, settings))
        with _forwards_through(model, recorder, windows, sink):
            generate = cast("Callable[..., Any]", model.generate)
            output = generate(*_positional(input_args), **prompt_kwargs, **generate_kwargs)
    recording = recorder.recording
    steps = _steps(recording, windows, None if sink is None else sink.bytes_by_step)
    return output, GenerationRecording(
        recording=recording,
        steps=steps,
        sequences=_sequences(output),
        prompt_length=steps[0].positions.stop if steps else 0,
        settings=MappingProxyType(settings),
        options=generation,
        _sink=sink,
    )


def enrich_generation(generation: GenerationRecording, postprocess: str) -> GenerationRecording:
    """Return ``generation`` with its recording enriched by ``postprocess`` (``"none"`` keeps it)."""

    if postprocess == "none":
        return generation
    return dataclasses.replace(generation, recording=generation.recording.enrich(postprocess))
