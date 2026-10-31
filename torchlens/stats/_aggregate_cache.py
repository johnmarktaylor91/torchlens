"""``tl.stats.aggregate(..., cache=True)``: skip the dataset loop on a rerun (DOCUMENTED-UNSTABLE).

A cached call keys the run, looks the key up in the cache directory, and on
a hit restores every accumulator's saved final state in place and returns
its results without running the model. On a miss it runs the loop as usual
and saves the final states under the key (``tl_stats_state_v1``, the same
format as ``tl.stats.save_fitted``).

The key is a SHA-256 over:

- the format id, the torchlens version and the torch version;
- the model: its tensor contents, ``requires_grad`` and ``training`` flags,
  and its implementation (forward code, plain attributes, registered
  hooks), the same two fingerprints the ``tl.trace(cache=True)`` key uses;
- the data: every batch the dataloader yields, in order, by value (tensors
  by content, containers by structure);
- the accumulators: each metric's name and its whole starting state
  (configuration included), plus the code and weights of any ``encoder=``;
- ``target=`` and the code of ``loss_fn=``.

When a part cannot be keyed by value the call refuses
(``stats_aggregate_cache_unkeyable``) instead of guessing: a one-shot
iterator (it cannot be read twice), a batch leaf of an opaque type, a
``Quantile(seed=None)`` (fresh entropy every run), a model with modules in
training mode (a hit would skip their state updates and dropout draws), a
metric that is not a ``tl.stats`` accumulator, or an ``encoder=`` /
``loss_fn=`` whose state is not visible (a callable object, or a function
referencing an opaque global). A dataloader that yields different batches
on the keying pass and the folding pass warns once
(``stats_aggregate_cache_data_unstable``) and stores nothing.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import types
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import nn

from .._errors import InvalidArgumentError
from ..errors import TorchLensWarning
from ._streaming import StreamingStat

__tl_layer__ = "L5"

#: Format id folded into every cache key (bump when the key's meaning changes).
CACHE_KEY_SCHEMA = "tl_aggregate_cache_v1"
_DOCS = "docs/reference/stats.md, Save, reload and cache"
_VALUE_DEPTH_CEILING = 64
_CALLABLE_DEPTH_CEILING = 4


class _Unkeyable(Exception):
    """Internal signal: one part of the run cannot be keyed by value."""

    def __init__(self, part: str, why: str, remedy: str) -> None:
        super().__init__(why)
        self.part = part
        self.why = why
        self.remedy = remedy


def _refuse_unkeyable(reason: _Unkeyable) -> NoReturn:
    """Raise the user-facing refusal for an unkeyable run."""

    raise InvalidArgumentError(
        f"tl.stats.aggregate(cache=True) cannot key {reason.part}: {reason.why}. A cache key "
        "that does not cover everything the result depends on could return a stale "
        f"result, so the call refuses instead. See {_DOCS}.",
        code="stats_aggregate_cache_unkeyable",
        remedy=reason.remedy,
        part=reason.part,
    )


_NOT_A_LEAF = object()


def _leaf_key(value: Any, where: str) -> object:
    """Key a scalar, tensor or array by value; ``_NOT_A_LEAF`` for anything else."""

    from .._capture_fingerprint import _hash_tensor_content

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, torch.Tensor):
        if value.is_meta or value.layout != torch.strided:
            raise _Unkeyable(where, "a tensor without readable contents", _REMEDY_DATA)
        return ("tensor", _hash_tensor_content(value))
    if isinstance(value, (torch.dtype, torch.device, torch.Size)):
        return ("torch", str(value))
    if type(value).__module__ == "numpy" and hasattr(value, "dtype") and hasattr(value, "shape"):
        return _ndarray_key(value, where)
    return _NOT_A_LEAF


def _value_key(value: Any, where: str, depth: int = 0) -> object:
    """Return a by-value key fragment, or raise :class:`_Unkeyable`."""

    if depth > _VALUE_DEPTH_CEILING:
        raise _Unkeyable(where, "the value nests deeper than 64 levels", _REMEDY_DATA)
    leaf = _leaf_key(value, where)
    if leaf is not _NOT_A_LEAF:
        return leaf
    if isinstance(value, Mapping):
        items = sorted(
            (
                repr(_value_key(key, where, depth + 1)),
                repr(_value_key(item, where, depth + 1)),
            )
            for key, item in value.items()
        )
        return ("mapping", type(value).__name__, tuple(items))
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_value_key(item, where, depth + 1) for item in value),
        )
    cls = type(value)
    raise _Unkeyable(
        where,
        f"it holds a {cls.__module__}.{cls.__qualname__}, whose contents the key cannot read",
        _REMEDY_DATA,
    )


def _ndarray_key(value: Any, where: str) -> object:
    """Key a NumPy array by content, refusing object arrays (they hold pointers)."""

    import numpy as np

    if getattr(value.dtype, "hasobject", False):
        raise _Unkeyable(where, "a NumPy array of Python objects", _REMEDY_DATA)
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256(contiguous.data).hexdigest()
    return ("ndarray", tuple(value.shape), str(value.dtype), digest)


_REMEDY_DATA = (
    "yield tensors, numbers, strings and lists, tuples or dicts of them from a re-iterable "
    "loader, for example tl.stats.aggregate(model, [x[:4], x[4:]], metrics, cache=True); "
    "or drop cache="
)
_REMEDY_CALLABLE = (
    "pass an nn.Module, one of its methods, or a plain function whose globals are modules, "
    "tensors or other such functions, for example encoder=sae.encode; or drop cache="
)


def _module_key(module: nn.Module) -> object:
    """Key a module by content and implementation (the trace cache's fingerprints)."""

    from .._capture_state_helpers import (
        _fingerprint_model_content,
        _fingerprint_model_implementation,
    )

    return (
        "module",
        _fingerprint_model_content(module),
        _fingerprint_model_implementation(module),
    )


def _code_names(code: types.CodeType) -> set[str]:
    """Return the global names a code object and its nested code objects reference."""

    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


def _function_key(func: types.FunctionType, where: str, depth: int) -> object:
    """Key a plain function: its code, defaults, closure cells and referenced globals."""

    from .._capture_fingerprint import _hash_code_object_into

    hasher = hashlib.sha256()
    _hash_code_object_into(hasher, func.__code__)
    defaults = tuple(_member_key(item, where, depth + 1) for item in func.__defaults__ or ())
    kwdefaults = tuple(
        (name, _member_key(item, where, depth + 1))
        for name, item in sorted((func.__kwdefaults__ or {}).items())
    )
    cells: list[object] = []
    for cell in func.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:
            cells.append("<empty-cell>")
            continue
        cells.append(_member_key(contents, where, depth + 1))
    referenced = tuple(
        (name, _member_key(func.__globals__[name], where, depth + 1))
        for name in sorted(_code_names(func.__code__))
        if name in func.__globals__
    )
    return ("function", hasher.hexdigest(), defaults, kwdefaults, tuple(cells), referenced)


def _callable_key(func: Any, where: str, depth: int = 0) -> object:  # noqa: PLR0911 -- one branch per callable kind
    """Return a key for an ``encoder=`` / ``loss_fn=`` callable, or raise :class:`_Unkeyable`."""

    if depth > _CALLABLE_DEPTH_CEILING:
        raise _Unkeyable(where, "its functions nest deeper than four calls", _REMEDY_CALLABLE)
    if isinstance(func, nn.Module):
        return _module_key(func)
    if isinstance(func, functools.partial):
        return (
            "partial",
            _callable_key(func.func, where, depth + 1),
            tuple(_member_key(item, where, depth + 1) for item in func.args),
            tuple(
                (name, _member_key(item, where, depth + 1))
                for name, item in sorted(func.keywords.items())
            ),
        )
    if inspect.ismethod(func):
        return (
            "method",
            _callable_key(func.__func__, where, depth + 1),
            _member_key(func.__self__, where, depth + 1),
        )
    if isinstance(func, types.FunctionType):
        return _function_key(func, where, depth)
    if isinstance(func, (types.BuiltinFunctionType, types.BuiltinMethodType)) and (
        func.__self__ is None or isinstance(func.__self__, types.ModuleType)
    ):
        return ("builtin", getattr(func, "__module__", None), func.__qualname__)
    cls = type(func)
    raise _Unkeyable(
        where,
        f"it is a {cls.__module__}.{cls.__qualname__}, whose state the key cannot read",
        _REMEDY_CALLABLE,
    )


def _member_key(value: Any, where: str, depth: int) -> object:
    """Key a value a callable closes over or references by name."""

    if isinstance(value, nn.Module):
        return _module_key(value)
    if isinstance(value, types.ModuleType):
        return ("pymodule", value.__name__)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if callable(value) and not isinstance(value, torch.Tensor):
        return _callable_key(value, where, depth)
    try:
        return _value_key(value, where)
    except _Unkeyable as exc:
        raise _Unkeyable(where, exc.why, _REMEDY_CALLABLE) from exc


def _accumulators(stat: Any) -> Iterator[Any]:
    """Yield ``stat`` and, for an ``Aggregator``, every child accumulator."""

    from ._aggregate import Aggregator

    yield stat
    if isinstance(stat, Aggregator):
        for child in stat.stats:
            yield from _accumulators(child)


def _metrics_key(metrics: Mapping[str, StreamingStat]) -> object:
    """Key every metric's name, starting state and encoder."""

    from ._per_unit import Quantile
    from ._state_io import encode_state, is_saveable_state, state_digest

    encoders: list[object] = []
    for name, stat in metrics.items():
        if not is_saveable_state(stat):
            cls = type(stat)
            raise _Unkeyable(
                f"metric {name!r}",
                f"it is a {cls.__module__}.{cls.__qualname__}, not a tl.stats accumulator, "
                "so its state cannot be saved",
                "use tl.stats accumulators (Mean, Quantile, TopK, ...) with cache=, or drop cache=",
            )
        for part in _accumulators(stat):
            if isinstance(part, Quantile) and part.seed is None:
                raise _Unkeyable(
                    f"metric {name!r}",
                    "Quantile(seed=None) samples from fresh entropy, so no two runs match",
                    "seed it: tl.stats.Quantile(seed=0), or drop cache=",
                )
            encoder = getattr(part, "encoder", None)
            if encoder is not None:
                encoders.append(_callable_key(encoder, f"the encoder= of metric {name!r}"))
    state_json, tensors, _paths = encode_state(dict(metrics))
    return (state_digest(state_json, tensors), tuple(encoders))


def _batch_digest(index: int, batch: Any) -> bytes:
    """Return the digest of one batch, keyed by position."""

    fragment = _value_key(batch, f"batch {index} of the dataloader")
    return hashlib.sha256(repr((index, fragment)).encode("utf-8")).digest()


class _HashingLoader:
    """Re-iterable wrapper that digests every batch it yields."""

    def __init__(self, dataloader: Iterable[Any]) -> None:
        self._dataloader = dataloader
        self.hasher = hashlib.sha256()

    def __iter__(self) -> Iterator[Any]:
        for index, batch in enumerate(self._dataloader):
            self.hasher.update(_batch_digest(index, batch))
            yield batch


def _data_key(dataloader: Iterable[Any]) -> str:
    """Digest every batch the dataloader yields, restoring the global random state after."""

    from ._aggregate import _rng_restore, _rng_snapshot

    if isinstance(dataloader, Iterator):
        raise _Unkeyable(
            "the dataloader",
            f"it is a one-shot {type(dataloader).__name__}, which cannot be read once to key "
            "and again to fold",
            _REMEDY_DATA,
        )
    snapshot = _rng_snapshot()
    if snapshot is None:
        raise _Unkeyable(
            "the dataloader",
            "the global random state cannot be saved, so reading the data to key it could "
            "change what the loop sees",
            "drop cache=",
        )
    try:
        hashing = _HashingLoader(dataloader)
        for _batch in hashing:
            pass
        return hashing.hasher.hexdigest()
    finally:
        _rng_restore(snapshot)


def _run_key(
    model: nn.Module,
    metrics: Mapping[str, StreamingStat],
    target: str,
    loss_fn: Callable[..., torch.Tensor] | None,
    data_digest: str,
) -> str:
    """Combine every keyed part into the cache key."""

    import torchlens

    if any(module.training for module in model.modules()):
        raise _Unkeyable(
            "the model",
            "some modules are in training mode, and a cache hit would skip their state "
            "updates (BatchNorm running statistics) and dropout draws",
            "call model.eval() first, or drop cache=",
        )
    payload = {
        "schema": CACHE_KEY_SCHEMA,
        "torchlens": str(torchlens.__version__),
        "torch": str(torch.version.__version__),
        "model": repr(_module_key(model)),
        "metrics": repr(_metrics_key(metrics)),
        "target": target,
        "loss_fn": repr(None if loss_fn is None else _callable_key(loss_fn, "loss_fn=")),
        "data": data_digest,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_root(cache_dir: str | Path | None) -> Path:
    """Create (private to this user) and return the aggregate cache directory."""

    from .._capture_state_helpers import _capture_cache_dir
    from ..user_funcs import _harden_capture_cache_dir

    base = _capture_cache_dir(cache_dir)
    created = not base.exists()
    root = base / "aggregate"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if created:
        _harden_capture_cache_dir(base)
    _harden_capture_cache_dir(root)
    return root


def _restore_hit(entry: Path, metrics: Mapping[str, StreamingStat]) -> bool:
    """Restore a cache entry into ``metrics`` in place; ``False`` when it is unusable."""

    from safetensors import safe_open

    from ._state_io import can_merge_state, merge_state_into, read_state

    if not entry.is_file():
        return False
    try:
        with safe_open(str(entry), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            names = list(handle.keys())
            tensors = {key: handle.get_tensor(key) for key in names}
        loaded = read_state(entry, metadata, tensors, encoder=None, require_encoder_match=False)
    except Exception:  # noqa: BLE001 -- any unreadable entry is a cache miss, never an error
        # A damaged or foreign entry (a refused digest, an unreadable file) is a
        # miss: the loop recomputes and overwrites it.
        return False
    if not isinstance(loaded, dict) or list(loaded) != list(metrics):
        return False
    if not all(can_merge_state(metrics[name], loaded[name]) for name in metrics):
        return False
    for name, stat in metrics.items():
        merge_state_into(stat, loaded[name])
    return True


def _warn_unstable_data() -> None:
    """Warn once that the dataloader yielded different batches on two passes."""

    from ._per_unit import _WARNED_CODES

    code = "stats_aggregate_cache_data_unstable"
    if code in _WARNED_CODES:
        return
    _WARNED_CODES.add(code)
    warnings.warn(
        TorchLensWarning(
            "tl.stats.aggregate(cache=True) read different batches from the dataloader when keying "
            "the run and when folding it (a shuffling sampler with its own generator, or a "
            "loader that changes between passes), so the result was returned but not cached; "
            f"this warning fires once per session. See {_DOCS}. Remedy: pass a loader that "
            "yields the same batches every pass, for example DataLoader(dataset, shuffle=False), "
            "or drop cache=",
            code=code,
        ),
        stacklevel=4,
    )


def run_cached(  # noqa: PLR0913 -- aggregate's subject arguments plus its keywords
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
    *,
    target: str,
    loss_fn: Callable[..., torch.Tensor] | None,
    cache_dir: str | Path | None,
    loop: Callable[[Iterable[Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Return the cached results for this run, or run ``loop`` and cache them."""

    from ._state_io import write_state

    try:
        data_digest = _data_key(dataloader)
        key = _run_key(model, metrics, target, loss_fn, data_digest)
    except _Unkeyable as reason:
        _refuse_unkeyable(reason)
    entry = _cache_root(cache_dir) / f"{key}.safetensors"
    if _restore_hit(entry, metrics):
        return {name: stat.compute() for name, stat in metrics.items()}
    hashing = _HashingLoader(dataloader)
    results = loop(hashing)
    if hashing.hasher.hexdigest() != data_digest:
        _warn_unstable_data()
        return results
    write_state(dict(metrics), entry)
    return results
