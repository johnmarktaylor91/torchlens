"""The ``input_size=`` grammar and the deterministic synthesis kernel (B6).

Grammar (quickstart memo D4, SOL's verbatim): one flat positive-integer tuple
is one shape; a sequence of shapes is multiple positional tensors; a mapping
is forward-keyword-to-shape. Batch is included exactly as written. Zero,
negative, symbolic, and unknown bindings refuse before capture.

Dtype and value recipes are FAIL-CLOSED static facts: an unambiguous
embedding consumer permits vocab-bounded integer IDs; an unambiguous float
entry permits uniform [0, 1); ambiguity teaches an override -- never
dtype-discovery by a parade of failed forwards. Synthesis uses a local
seed-0 generator that advances no process RNG and never moves the model.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import torch
from torch import nn

from .._errors import InvalidArgumentError

__tl_layer__ = "L2"

#: Default synthesis seed (memo D4: "a local seed-0 generator").
SYNTHESIS_SEED = 0

#: Closed keyword-name fact table for the mapping form. Fail-closed: a
#: mapping key outside this table falls back to model-level facts, and
#: ambiguity refuses with the override teach rather than probing.
_IDS_KEYWORDS = frozenset({"input_ids", "decoder_input_ids"})
_MASK_KEYWORDS = frozenset({"attention_mask", "decoder_attention_mask"})
_IMAGE_KEYWORDS = frozenset({"pixel_values", "pixel_values_videos"})


@dataclass(frozen=True)
class InputSpec:
    """Explicit per-slot override for the declared rung (the taught remedy).

    ``input_size=`` accepts an ``InputSpec`` anywhere it accepts a bare shape
    tuple, so a caller can pin the dtype and value recipe when the static
    facts are ambiguous (the fail-closed teach names this spelling).

    Attributes
    ----------
    shape:
        Tensor shape, batch included exactly as written.
    dtype:
        Optional dtype override (``None`` = decided by the static facts).
    low / high:
        Optional value-range override. For integer dtypes the recipe is
        ``randint(low, high)``; for float dtypes ``uniform [low, high)``.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype | None = None
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class SlotSpec:
    """One resolved input slot: shape plus its settled synthesis recipe."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    recipe: str  # "uniform" | "randint" | "ones"
    low: float
    high: float
    keyword: str | None = None  # None = positional


@dataclass(frozen=True)
class ParsedInputSize:
    """The parsed ``input_size=`` argument: positional and keyword slots."""

    positional: tuple[SlotSpec, ...]
    keyword: tuple[SlotSpec, ...]


def _refuse(problem: str, *, code: str = "input_size_invalid", **context: Any) -> NoReturn:
    """Raise the typed grammar refusal with the ladder's remedy attached."""

    raise InvalidArgumentError(
        problem,
        code=code,
        remedy=(
            "pass one flat positive-int tuple for one input, a sequence of "
            "shape tuples for several positional inputs, or a mapping of "
            "forward keyword names to shapes; use "
            "torchlens.quickstart.InputSpec(shape, dtype=..., low=..., high=...) "
            "to pin a slot's dtype and value recipe -- or pass a real input, "
            "which is always the better answer"
        ),
        **context,
    )


def _validate_shape(raw: Any) -> tuple[int, ...]:
    """Validate one shape: a non-empty flat tuple of positive real ints."""

    if not isinstance(raw, (tuple, list)) or len(raw) == 0:
        _refuse(
            f"input_size= shape {raw!r} is not a non-empty tuple of positive integers.",
            value=repr(raw),
        )
    dims: list[int] = []
    for dim in raw:
        if isinstance(dim, bool) or not isinstance(dim, int):
            _refuse(
                f"input_size= dimension {dim!r} is not a plain positive integer "
                "(symbolic and non-integer dimensions cannot be synthesized).",
                value=repr(dim),
            )
        if dim <= 0:
            _refuse(
                f"input_size= dimension {dim} is not positive; a zero or negative "
                "dimension has no synthesizable tensor.",
                value=dim,
            )
        dims.append(int(dim))
    return tuple(dims)


def _is_single_flat_shape(value: Any) -> bool:
    """Return whether ``value`` spells ONE shape (a flat sequence of ints)."""

    if not isinstance(value, (tuple, list)) or len(value) == 0:
        return False
    return all(isinstance(dim, int) and not isinstance(dim, bool) for dim in value)


def _first_module(model: nn.Module, kinds: tuple[type, ...]) -> nn.Module | None:
    """Return the first module of any of ``kinds`` in ``model.modules()`` order."""

    for module in model.modules():
        if isinstance(module, kinds):
            return module
    return None


def _vocab_bound(model: nn.Module) -> int | None:
    """Return the SAFE vocab bound: the smallest ``num_embeddings`` on the model.

    Every id strictly below the smallest embedding table is valid for every
    embedding lookup it could reach, so synthesized ids are always inside the
    executed vocabulary (fail-closed: representative ids are the real input's
    job, valid ids are this kernel's).
    """

    sizes = [
        int(module.num_embeddings)
        for module in model.modules()
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag))
    ]
    return min(sizes) if sizes else None


def _model_entry_fact(model: nn.Module) -> str | None:
    """Classify the model's entry consumer: ``"ids"``, ``"float"``, or ``None``.

    The first module in ``model.modules()`` order among the known entry
    families decides (mirrors the shipped inference priors): an
    ``nn.Embedding`` entry permits vocab-bounded integer ids; a
    Conv/Linear/RNN entry permits uniform floats. No known entry is honest
    ambiguity and refuses upstream.
    """

    entry = _first_module(
        model,
        (
            nn.Embedding,
            nn.EmbeddingBag,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Linear,
            nn.RNN,
            nn.LSTM,
            nn.GRU,
        ),
    )
    if entry is None:
        return None
    if isinstance(entry, (nn.Embedding, nn.EmbeddingBag)):
        return "ids"
    return "float"


def _fact_for_keyword(
    keyword: str, model: nn.Module
) -> tuple[str, float, float, torch.dtype] | None:
    """Return the (recipe, low, high, dtype) fact for a known keyword name."""

    if keyword in _IDS_KEYWORDS:
        bound = _vocab_bound(model)
        if bound is None or bound < 1:
            return None
        return ("randint", 0.0, float(bound), torch.int64)
    if keyword in _MASK_KEYWORDS:
        return ("ones", 1.0, 1.0, torch.int64)
    if keyword in _IMAGE_KEYWORDS:
        return ("uniform", 0.0, 1.0, torch.get_default_dtype())
    return None


def _fact_for_model(model: nn.Module, *, slot: str) -> tuple[str, float, float, torch.dtype]:
    """Return the model-level (recipe, low, high, dtype) fact, or refuse.

    Fail-closed (memo D4): ambiguity refuses with the ``InputSpec`` override
    teach; dtype is never discovered by failed forwards.
    """

    fact = _model_entry_fact(model)
    if fact == "ids":
        bound = _vocab_bound(model)
        if bound is not None and bound >= 1:
            return ("randint", 0.0, float(bound), torch.int64)
    if fact == "float":
        return ("uniform", 0.0, 1.0, torch.get_default_dtype())
    _refuse(
        f"Could not settle a dtype and value recipe for input slot {slot} from "
        "static model facts alone (no unambiguous embedding or float entry "
        "module was found), and TorchLens never discovers dtypes by running "
        "failed forwards.",
        code="input_dtype_ambiguous",
        slot=slot,
    )


def _resolve_slot(raw: Any, model: nn.Module, *, keyword: str | None, slot: str) -> SlotSpec:
    """Resolve one grammar entry (shape or ``InputSpec``) to a ``SlotSpec``."""

    if isinstance(raw, InputSpec):
        shape = _validate_shape(raw.shape)
        if raw.dtype is not None:
            recipe = "randint" if not raw.dtype.is_floating_point else "uniform"
            low = raw.low if raw.low is not None else 0.0
            default_high = 2.0 if recipe == "randint" else 1.0
            high = raw.high if raw.high is not None else default_high
            return SlotSpec(shape, raw.dtype, recipe, float(low), float(high), keyword)
        fact = (_fact_for_keyword(keyword, model) if keyword else None) or _fact_for_model(
            model, slot=slot
        )
        recipe, low, high, dtype = fact
        if raw.low is not None:
            low = float(raw.low)
        if raw.high is not None:
            high = float(raw.high)
        return SlotSpec(shape, dtype, recipe, low, high, keyword)
    shape = _validate_shape(raw)
    fact = (_fact_for_keyword(keyword, model) if keyword else None) or _fact_for_model(
        model, slot=slot
    )
    recipe, low, high, dtype = fact
    return SlotSpec(shape, dtype, recipe, low, high, keyword)


def _forward_keyword_names(model: nn.Module) -> tuple[frozenset[str], bool]:
    """Return the forward signature's keyword names and VAR_KEYWORD flag.

    Uninspectable forwards return ``(frozenset(), True)`` so unknown-binding
    validation degrades open (the forward itself will refuse a genuinely
    wrong name) rather than refusing a legal call on missing metadata.
    """

    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return frozenset(), True
    names: set[str] = set()
    has_var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(parameter.name)
    return frozenset(names), has_var_keyword


def parse_input_size(input_size: Any, model: nn.Module) -> ParsedInputSize:
    """Parse and validate the ``input_size=`` argument (the D4 grammar).

    Parameters
    ----------
    input_size:
        One flat positive-int tuple (one positional tensor), a sequence of
        shape tuples / ``InputSpec`` entries (several positional tensors), or
        a mapping of forward keyword names to shapes / ``InputSpec`` entries.
    model:
        The model, consulted for static dtype facts and (mapping form) the
        forward signature.

    Returns
    -------
    ParsedInputSize
        Fully settled slot specs; every refusal has already happened.

    Raises
    ------
    torchlens._errors.InvalidArgumentError
        Codes ``input_size_invalid`` (malformed grammar),
        ``input_size_unknown_binding`` (mapping key the forward does not
        accept), and ``input_dtype_ambiguous`` (no static dtype fact).
    """

    if isinstance(input_size, InputSpec):
        return ParsedInputSize(
            positional=(_resolve_slot(input_size, model, keyword=None, slot="0"),),
            keyword=(),
        )
    if isinstance(input_size, Mapping):
        return _parse_mapping_form(input_size, model)
    if _is_single_flat_shape(input_size):
        return ParsedInputSize(
            positional=(_resolve_slot(input_size, model, keyword=None, slot="0"),),
            keyword=(),
        )
    if isinstance(input_size, Sequence) and not isinstance(input_size, (str, bytes)):
        slots = tuple(
            _resolve_slot(entry, model, keyword=None, slot=str(index))
            for index, entry in enumerate(input_size)
        )
        if not slots:
            _refuse("input_size= is empty; there is no shape to synthesize.")
        return ParsedInputSize(positional=slots, keyword=())
    _refuse(
        f"input_size= {input_size!r} is not a shape tuple, a sequence of shape "
        "tuples, or a mapping of forward keyword names to shapes.",
        value=repr(input_size),
    )


def _parse_mapping_form(input_size: Mapping[Any, Any], model: nn.Module) -> ParsedInputSize:
    """Parse the mapping (forward-keyword-to-shape) form."""

    if not input_size:
        _refuse("input_size= mapping is empty; there is no shape to synthesize.")
    known_names, has_var_keyword = _forward_keyword_names(model)
    slots: list[SlotSpec] = []
    for key, raw in input_size.items():
        if not isinstance(key, str):
            _refuse(
                f"input_size= mapping key {key!r} is not a forward keyword name string.",
                value=repr(key),
            )
        if known_names and not has_var_keyword and key not in known_names:
            _refuse(
                f"input_size= binds {key!r}, but the model's forward accepts no such "
                f"keyword (known keywords: {sorted(known_names)}).",
                code="input_size_unknown_binding",
                binding=key,
                known=sorted(known_names),
            )
        slots.append(_resolve_slot(raw, model, keyword=key, slot=key))
    return ParsedInputSize(positional=(), keyword=tuple(slots))


def _model_device(model: nn.Module) -> torch.device:
    """Return the device synthesized inputs should land on (never moves the model)."""

    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _synthesize_slot(
    slot: SlotSpec, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    """Synthesize one tensor for ``slot`` on CPU, then transfer to ``device``.

    CPU generation keeps values byte-deterministic across target devices; the
    transfer moves the INPUT only -- the model is never touched.
    """

    if slot.recipe == "ones":
        tensor = torch.ones(slot.shape, dtype=slot.dtype)
    elif slot.recipe == "randint":
        tensor = torch.randint(
            int(slot.low), int(slot.high), slot.shape, generator=generator, dtype=slot.dtype
        )
    else:
        span = slot.high - slot.low
        tensor = torch.rand(slot.shape, generator=generator, dtype=slot.dtype) * span + slot.low
    return tensor.to(device) if device.type != "cpu" else tensor


def synthesize(
    parsed: ParsedInputSize, model: nn.Module, *, seed: int = SYNTHESIS_SEED
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    """Synthesize the declared rung's concrete tensors deterministically.

    Uses a LOCAL ``torch.Generator`` seeded with ``seed``: the process-global
    RNG is never read or advanced, and repeated calls are byte-identical.

    Returns
    -------
    tuple
        ``(positional_tensors, keyword_tensors)`` ready for the concrete
        capture primitive.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    device = _model_device(model)
    positional = tuple(_synthesize_slot(slot, generator, device) for slot in parsed.positional)
    keyword = {
        slot.keyword: _synthesize_slot(slot, generator, device)
        for slot in parsed.keyword
        if slot.keyword is not None
    }
    return positional, keyword
