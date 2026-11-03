"""Grouped options of the streaming generation recorder (DOCUMENTED-UNSTABLE spellings).

``tl.record(model, ids, generate_kwargs={...}, generation=GenerationOptions(...))``
widens the generation recorder from "keep the declared sites at every token"
to per-token reducers, KV-cache slices as sites, a ring buffer with disk
spill, and a guarded memory cost at thousands of tokens.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

import torch

from .._errors import InvalidArgumentError

__all__ = ["DEFAULT_MAX_SAVED_BYTES", "GenerationOptions"]

#: The retained-payload budget a widened generation recording holds by default.
DEFAULT_MAX_SAVED_BYTES: int = 512 * 2**20

ReducerFn = Callable[[torch.Tensor], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class GenerationOptions:
    """What a generation recording keeps per token, and how much it may hold.

    Parameters
    ----------
    reducers:
        Site address to reducer. After each forward the site's payload is
        replaced by the reducer's value (``torchlens.recording.reducers`` has
        ``norm``, ``top_k`` and ``probe``; any callable from the payload to a
        tensor whose first axis runs over that forward's positions works).
        The declared sites are recorded even without a ``save=``.
    kv_sites:
        Decoder layer indexes whose new KV-cache slices are kept per token,
        read from the cache the model's ``generate`` passes to each forward
        (series ``kv.<layer>.key`` and ``kv.<layer>.value``, shaped
        ``[positions, kv_heads, head_dim]``). ``True`` keeps every layer.
    keep_last:
        Ring buffer: keep the payloads of the last ``keep_last`` forwards in
        memory and spill or drop the older ones (``None`` keeps everything).
    spill:
        Directory for the payloads the ring buffer evicts, one ``torch.save``
        file per step; ``series()`` reads them back. Without it, evicted
        payloads are dropped.
    max_saved_bytes:
        Budget for the retained payload bytes (in memory, net of spills).
        After the first decode step the recorder projects the cost of the
        whole generation from the prefill and per-step bytes and refuses
        early when it would exceed the budget; the running total is checked
        at every step. ``None`` disables the guard.

    Whatever the options do not keep is released as each forward is reduced: a
    pass's unkept payloads (its model inputs, the KV cache ``generate`` feeds back
    among them) and the logits of every pass but the newest, so the recording's
    memory grows with the number of tokens, not with its square.
    """

    reducers: Mapping[str, ReducerFn] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    kv_sites: tuple[int, ...] | bool = ()
    keep_last: int | None = None
    spill: str | Path | None = None
    max_saved_bytes: int | None = DEFAULT_MAX_SAVED_BYTES

    def __post_init__(self) -> None:
        reducers = MappingProxyType(dict(self.reducers))
        for site, reducer in reducers.items():
            if not isinstance(site, str) or not callable(reducer):
                raise InvalidArgumentError(
                    "GenerationOptions(reducers=) maps site addresses (str) to callables; got "
                    f"{site!r}: {reducer!r}",
                    code="record_generation_options_invalid",
                    remedy="pass {address: torchlens.recording.reducers.norm()} or your own callable",
                )
        object.__setattr__(self, "reducers", reducers)
        if self.kv_sites is not True and self.kv_sites is not False:
            sites = tuple(int(index) for index in self.kv_sites)
            if any(index < 0 for index in sites):
                raise InvalidArgumentError(
                    f"GenerationOptions(kv_sites=) takes layer indexes >= 0 or True; got {sites}",
                    code="record_generation_options_invalid",
                    remedy="pass decoder layer indexes such as (0, 5) or True for every layer",
                )
            object.__setattr__(self, "kv_sites", sites)
        if self.keep_last is not None and self.keep_last < 1:
            raise InvalidArgumentError(
                f"GenerationOptions(keep_last=) needs at least 1 forward; got {self.keep_last}",
                code="record_generation_options_invalid",
                remedy="pass keep_last=None to keep everything or a positive count",
            )
        if self.spill is not None and self.keep_last is None:
            raise InvalidArgumentError(
                "GenerationOptions(spill=) needs keep_last= (the ring buffer decides what spills)",
                code="record_generation_options_invalid",
                remedy="pass keep_last=N with spill=, or drop spill=",
            )
        if self.max_saved_bytes is not None and self.max_saved_bytes < 1:
            raise InvalidArgumentError(
                f"GenerationOptions(max_saved_bytes=) needs a positive budget or None; got "
                f"{self.max_saved_bytes}",
                code="record_generation_options_invalid",
                remedy="pass a byte budget, or None to disable the guard",
            )

    @property
    def keeps_everything(self) -> bool:
        """Return whether no ring buffer bounds the retained forwards."""

        return self.keep_last is None
