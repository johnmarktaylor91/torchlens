"""Per-token series, ring buffer, disk spill and cost guard of the generation recorder.

The :class:`_StepSink` runs after every forward that ``generate()`` issues:
it replaces the declared sites' payloads by their reducers' values, reads the
declared KV-cache slices, evicts the forwards the ring buffer no longer holds
(spilling them to disk when asked), and checks the retained-byte budget.
:class:`TokenSeries` is what ``GenerationRecording.series(name)`` returns.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, NoReturn

import torch

from .._errors import InvalidArgumentError
from .._ir.selector_eval import module_union_matches
from ._generation_options import GenerationOptions
from .types import ActivationRecord, Recording

__all__ = ["TokenSeries"]

BUDGET_EXCEEDED = "record_generation_budget_exceeded"
REDUCER_INVALID = "record_generation_reducer_invalid"
_REFUSAL = "tl.record(..., generate_kwargs=...) refuses: {}"


@dataclasses.dataclass(frozen=True)
class TokenSeries:
    """One per-token series of a generation recording (DOCUMENTED-UNSTABLE).

    Attributes
    ----------
    name:
        The reducer site address or the KV series name (``kv.<layer>.key``).
    positions:
        The absolute sequence positions the series covers, prompt positions
        first; forwards the ring buffer dropped are absent from the front.
    values:
        The values, one row per position along the first axis.
    """

    name: str
    positions: range
    values: torch.Tensor

    def __len__(self) -> int:
        return len(self.positions)


def _nbytes(tensor: torch.Tensor | None) -> int:
    """Return the payload bytes of ``tensor`` (0 for ``None``)."""

    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _cache_layers(cache: Any) -> int:
    """Return how many decoder layers a KV cache holds (0 when unreadable)."""

    for attr in ("layers", "key_cache"):
        entries = getattr(cache, attr, None)
        if entries is not None:
            return len(entries)
    return len(cache) if isinstance(cache, (tuple, list)) else 0


def _cache_layer_kv(cache: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return one layer's ``(keys, values)`` from a transformers cache of any vintage."""

    layers = getattr(cache, "layers", None)
    if layers is not None:
        entry = layers[layer]
        keys, values = getattr(entry, "keys", None), getattr(entry, "values", None)
    elif getattr(cache, "key_cache", None) is not None:
        keys, values = cache.key_cache[layer], cache.value_cache[layer]
    elif isinstance(cache, (tuple, list)):
        keys, values = cache[layer][0], cache[layer][1]
    else:
        return None
    if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
        return keys, values
    return None


def _kv_slice(tensor: torch.Tensor, window: range) -> torch.Tensor:
    """Return the ``[positions, heads, head_dim]`` slice of one cache tensor."""

    return torch.movedim(tensor[0, :, window.start : window.stop, :], 1, 0).detach().clone()


class _StepSink:
    """Per-forward bookkeeping of a widened generation recording."""

    def __init__(
        self,
        options: GenerationOptions,
        recording: Recording,
        max_new_tokens: Callable[[int], int | None],
    ) -> None:
        self.options = options
        self.recording = recording
        self.max_new_tokens = max_new_tokens
        self.windows: list[range] = []
        self.pass_of_step: list[int] = []
        self.values: dict[str, dict[int, torch.Tensor]] = {site: {} for site in options.reducers}
        self.bytes_by_step: list[int] = []
        self.evicted: dict[int, str] = {}
        self.spill_files: dict[int, Path] = {}
        self.retained_bytes = 0
        self.peak_bytes = 0
        self._site_sets = {site: frozenset({site}) for site in options.reducers}

    # -- per forward -------------------------------------------------------------------

    def after_pass(
        self, step: int, pass_index: int, window: range, kwargs: Mapping[str, Any]
    ) -> None:
        """Reduce, read the KV slices, evict and guard after one recorded forward."""

        self.windows.append(window)
        self.pass_of_step.append(pass_index)
        step_bytes = self._reduce(step, pass_index, window)
        step_bytes += self._read_kv(step, window, kwargs.get("past_key_values"))
        self.bytes_by_step.append(step_bytes)
        self.retained_bytes += step_bytes
        self._evict(step)
        self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        self._guard(step)

    def _site_of(self, record: ActivationRecord) -> str | None:
        """Return the declared reducer site ``record`` is the output of, if any."""

        address = record.ctx.address
        if address in self.options.reducers:
            return str(address)
        for site, addresses in self._site_sets.items():
            if module_union_matches(record.ctx, addresses):
                return site
        return None

    def _reduce(self, step: int, pass_index: int, window: range) -> int:
        """Replace the declared sites' payloads of one pass by their reduced values."""

        records = self.recording.records
        seen: set[str] = set()
        total = 0
        for index in self.recording.by_pass.get(pass_index, []):
            record = records[index]
            site = self._site_of(record)
            if site is None:
                total += _nbytes(record.ram_payload) + _nbytes(record.transformed_ram_payload)
                continue
            if site in seen:
                self._refuse(
                    f"site {site!r} produced more than one record in forward {step}",
                    "declare a site whose module runs once per forward",
                )
            seen.add(site)
            value = self._reduced_value(site, record, window, step)
            records[index] = dataclasses.replace(
                record,
                ram_payload=value,
                transformed_ram_payload=None,
                metadata={
                    **record.metadata,
                    "reducer": getattr(self.options.reducers[site], "name", "callable"),
                    "reduced_from_shape": tuple(record.ctx.shape or ()),
                },
            )
            self.values[site][step] = value
            total += _nbytes(value)
        missing = set(self.options.reducers) - seen
        if missing:
            self._refuse(
                f"forward {step} recorded no output for the declared site(s) {sorted(missing)}",
                "declare module addresses the model runs on every forward "
                "(model.get_submodule(address) must exist)",
            )
        return total

    def _reduced_value(
        self, site: str, record: ActivationRecord, window: range, step: int
    ) -> torch.Tensor:
        """Run one reducer and check its position-major contract."""

        payload = (
            record.ram_payload if record.ram_payload is not None else record.transformed_ram_payload
        )
        if payload is None:
            self._refuse(
                f"site {site!r} kept no payload in forward {step} (save_out=False?)",
                "let the save predicate keep the site's payload",
            )
        value = self.options.reducers[site](payload)
        if not isinstance(value, torch.Tensor) or value.dim() == 0 or value.shape[0] != len(window):
            got = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            self._refuse(
                f"the reducer of {site!r} returned {got} for a forward of {len(window)} "
                f"position(s); a reducer returns a tensor whose first axis runs over the "
                f"forward's positions",
                "return one row per position (torchlens.recording.reducers.norm/top_k/probe do)",
            )
        return value.detach()

    def _read_kv(self, step: int, window: range, cache: Any) -> int:
        """Keep the declared layers' new KV-cache slices for one forward."""

        declared = self.options.kv_sites
        if declared == () or declared is False:
            return 0
        if cache is None or _cache_layers(cache) == 0:
            self._refuse(
                f"kv_sites= was declared but forward {step} received no readable KV cache "
                f"(past_key_values is {type(cache).__name__})",
                "use a model whose generate() passes a transformers Cache to forward",
            )
        n_layers = _cache_layers(cache)
        layers = range(n_layers) if isinstance(declared, bool) else declared
        total = 0
        for layer in layers:
            if layer >= n_layers:
                self._refuse(
                    f"kv_sites names layer {layer} but the cache holds {n_layers} layer(s)",
                    f"pass layer indexes below {n_layers}",
                )
            pair = _cache_layer_kv(cache, layer)
            if pair is None or pair[0].shape[-2] < window.stop:
                self._refuse(
                    f"the KV cache of layer {layer} does not hold positions {window.start} to "
                    f"{window.stop - 1} after forward {step} (a sliding or offloaded cache?)",
                    "record with a plain DynamicCache (the generate() default)",
                )
            for part, tensor in zip(("key", "value"), pair, strict=True):
                name = f"kv.{layer}.{part}"
                value = _kv_slice(tensor, window)
                self.values.setdefault(name, {})[step] = value
                total += _nbytes(value)
        return total

    # -- ring buffer and spill -----------------------------------------------------------

    def _evict(self, step: int) -> None:
        """Spill or drop the forward the ring buffer no longer holds."""

        keep_last = self.options.keep_last
        if keep_last is None:
            return
        old = step - keep_last
        if old < 0 or old in self.evicted:
            return
        payloads: dict[str, torch.Tensor] = {}
        for name, by_step in self.values.items():
            value = by_step.pop(old, None)
            if value is not None:
                payloads[name] = value
        records = self.recording.records
        for index in self.recording.by_pass.get(self.pass_of_step[old], []):
            record = records[index]
            if record.ctx.address in self.options.reducers:
                key = str(record.ctx.address)
            else:
                key = str(record.ctx.label)
                for payload in (record.ram_payload, record.transformed_ram_payload):
                    if payload is not None:
                        payloads[key] = payload
            records[index] = self._evicted_record(record, old, key)
        fate = "dropped"
        if self.options.spill is not None:
            path = Path(self.options.spill)
            path.mkdir(parents=True, exist_ok=True)
            file = path / f"step-{old:06d}.pt"
            torch.save({key: value.contiguous() for key, value in payloads.items()}, file)
            self.spill_files[old] = file
            fate = "spilled"
        self.evicted[old] = fate
        self.retained_bytes -= self.bytes_by_step[old]

    def _evicted_record(self, record: ActivationRecord, step: int, key: str) -> ActivationRecord:
        """Return ``record`` with its payloads released and its fate in the metadata."""

        fate = "spilled" if self.options.spill is not None else "dropped"
        metadata = {**record.metadata, "payload": fate, "spill_key": key}
        if self.options.spill is not None:
            metadata["spill_file"] = str(Path(self.options.spill) / f"step-{step:06d}.pt")
        return dataclasses.replace(
            record, ram_payload=None, transformed_ram_payload=None, metadata=metadata
        )

    # -- guard ---------------------------------------------------------------------------

    def _guard(self, step: int) -> None:
        """Refuse when the retained bytes exceed, or will exceed, the budget."""

        budget = self.options.max_saved_bytes
        if budget is None:
            return
        if self.retained_bytes > budget:
            self._budget_refusal(
                f"forward {step} brought the retained payloads to {self.retained_bytes:,} "
                f"bytes, over the {budget:,}-byte budget"
            )
        if step != 1 or not self.options.keeps_everything:
            return
        new_tokens = self.max_new_tokens(self.windows[0].stop)
        if new_tokens is None:
            return
        projected = self.bytes_by_step[0] + self.bytes_by_step[1] * new_tokens
        if projected > budget:
            self._budget_refusal(
                f"the prefill retained {self.bytes_by_step[0]:,} bytes and the first decode "
                f"step {self.bytes_by_step[1]:,}, so {new_tokens:,} new tokens would retain "
                f"about {projected:,} bytes, over the {budget:,}-byte budget"
            )

    def _budget_refusal(self, reason: str) -> NoReturn:
        raise InvalidArgumentError(
            _REFUSAL.format(reason),
            code=BUDGET_EXCEEDED,
            remedy=(
                "reduce the declared sites (GenerationOptions(reducers=)), bound the ring "
                "(keep_last=, with spill= to keep the rest on disk), or raise max_saved_bytes="
            ),
            reason=reason,
        )

    def _refuse(self, reason: str, remedy: str) -> NoReturn:
        raise InvalidArgumentError(
            _REFUSAL.format(reason), code=REDUCER_INVALID, remedy=remedy, reason=reason
        )

    # -- reading -------------------------------------------------------------------------

    @property
    def series_names(self) -> tuple[str, ...]:
        """Return the names ``series()`` answers, reducer sites first."""

        return tuple(self.values)

    def _rows(self, name: str) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield ``(step, value)`` for every step still readable, in step order."""

        in_memory = self.values.get(name, {})
        for step in range(len(self.windows)):
            if step in in_memory:
                yield step, in_memory[step]
            elif self.evicted.get(step) == "spilled":
                loaded = torch.load(self.spill_files[step], weights_only=True)
                if name in loaded:
                    yield step, loaded[name]

    def series(self, name: str) -> TokenSeries:
        """Return the per-token series ``name`` over every readable position."""

        if name not in self.values:
            raise InvalidArgumentError(
                f"no per-token series named {name!r}; this recording has "
                f"{list(self.values) or 'none (declare reducers= or kv_sites=)'}",
                code="record_generation_step_missing",
                remedy="pass a reducer site address or a kv.<layer>.key / kv.<layer>.value name",
            )
        rows = list(self._rows(name))
        if not rows:
            raise InvalidArgumentError(
                f"every forward of series {name!r} was dropped by the ring buffer",
                code="record_generation_step_missing",
                remedy="pass spill= beside keep_last= to keep evicted forwards on disk",
            )
        start = self.windows[rows[0][0]].start
        stop = self.windows[rows[-1][0]].stop
        values = torch.cat([value for _, value in rows], dim=0)
        return TokenSeries(name=name, positions=range(start, stop), values=values)


def site_predicate(sites: tuple[str, ...]) -> Callable[[Any], bool]:
    """Return a save predicate that keeps the module outputs of ``sites``."""

    addresses = frozenset(sites)

    def keep(ctx: Any) -> bool:
        return module_union_matches(ctx, addresses)

    return keep
