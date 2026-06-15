"""Per-invocation backward pass records and accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
import weakref

from .._io import FieldPolicy, TLSPEC_VERSION, default_fill_state, read_tlspec_version
from ..quantities import Duration
from ._accessor_base import Accessor

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BackwardPass:
    """Projected metadata for one backward engine invocation.

    Parameters
    ----------
    pass_index:
        One-based global backward invocation number.
    trigger:
        Entry point that opened this backward pass.
    implicit:
        Whether this pass was inferred from orphan tensor-hook emissions.
    outer_context:
        Optional outer TorchLens trigger context when this pass was nested.
    """

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "pass_index": FieldPolicy.KEEP,
        "trigger": FieldPolicy.KEEP,
        "implicit": FieldPolicy.KEEP,
        "outer_context": FieldPolicy.KEEP,
        "backward_call_context": FieldPolicy.KEEP,
        "root_grad_fn_ids": FieldPolicy.KEEP,
        "root_meta": FieldPolicy.KEEP,
        "root_grad_arguments": FieldPolicy.KEEP,
        "inputs_subset": FieldPolicy.KEEP,
        "order": FieldPolicy.KEEP,
        "origin_backward_pass": FieldPolicy.KEEP,
        "engine_flags": FieldPolicy.KEEP,
        "save_grads_policy": FieldPolicy.KEEP,
        "duration": FieldPolicy.KEEP,
        "peak_memory": FieldPolicy.KEEP,
        "status": FieldPolicy.KEEP,
        "order_attribution_coverage": FieldPolicy.KEEP,
        "grad_fn_calls": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
    }

    pass_index: int
    trigger: str
    implicit: bool
    outer_context: str | None = None
    backward_call_context: object | None = None
    root_grad_fn_ids: list[int] = field(default_factory=list)
    root_meta: tuple[object, ...] = field(default_factory=tuple)
    root_grad_arguments: object | None = None
    inputs_subset: tuple[object, ...] = field(default_factory=tuple)
    order: int | None = None
    origin_backward_pass: int | None = None
    engine_flags: dict[str, object] | None = None
    save_grads_policy: str | None = None
    duration: Duration | None = None
    peak_memory: int | None = None
    status: str = "ok"
    order_attribution_coverage: float | None = None
    grad_fn_calls: list[Any] = field(default_factory=list)
    _source_trace_ref: Any = None

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with an IO format marker."""

        state = self.__dict__.copy()
        state["_source_trace_ref"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and fill fields added in newer versions."""

        read_tlspec_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults={
                "outer_context": None,
                "backward_call_context": None,
                "root_grad_fn_ids": [],
                "root_meta": (),
                "root_grad_arguments": None,
                "inputs_subset": (),
                "order": None,
                "origin_backward_pass": None,
                "engine_flags": None,
                "save_grads_policy": None,
                "duration": None,
                "peak_memory": None,
                "status": "ok",
                "order_attribution_coverage": None,
                "grad_fn_calls": [],
                "_source_trace_ref": None,
            },
        )
        state.pop("call_context", None)
        self.__dict__.update(state)

    @property
    def source_trace(self) -> Any:
        """Return the owning Trace if it is still alive."""

        ref = self._source_trace_ref
        return None if ref is None else ref()

    @source_trace.setter
    def source_trace(self, value: Any) -> None:
        """Set the owning Trace weakref."""

        self._source_trace_ref = weakref.ref(value) if value is not None else None

    @property
    def trace(self) -> Any:
        """Alias for the owning Trace."""

        return self.source_trace

    @property
    def ordinal_index(self) -> int:
        """Return this pass's 0-based position in ``trace.backward_passes``."""

        return self.pass_index - 1

    def to_pandas(self) -> "pd.DataFrame":
        """Export this backward pass as a one-row DataFrame."""

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        row = {
            "pass_index": self.pass_index,
            "trigger": self.trigger,
            "implicit": self.implicit,
            "outer_context": self.outer_context,
            "backward_call_context": self.backward_call_context,
            "order": self.order,
            "origin_backward_pass": self.origin_backward_pass,
            "duration": self.duration,
            "peak_memory": self.peak_memory,
            "status": self.status,
            "order_attribution_coverage": self.order_attribution_coverage,
            "num_grad_fn_calls": len(self.grad_fn_calls),
        }
        return pd.DataFrame([row], columns=list(row))


class BackwardPassAccessor(Accessor[BackwardPass]):
    """Dict-like accessor for projected backward pass records."""

    def __init__(self, backward_passes: dict[int, BackwardPass]) -> None:
        """Initialize the accessor from pass-index keyed records."""

        items = [backward_passes[index] for index in sorted(backward_passes)]
        super().__init__(
            {str(pass_record.pass_index): pass_record for pass_record in items},
            item_list=items,
        )

    def for_pass(self, pass_index: int) -> BackwardPass:
        """Return a pass by its one-based global pass number."""

        key = str(pass_index)
        if key not in self._dict:
            available = [pass_record.pass_index for pass_record in self._list]
            raise KeyError(f"Backward pass {pass_index} not found. Available passes: {available}.")
        return self._dict[key]
