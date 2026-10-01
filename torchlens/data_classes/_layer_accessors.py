"""OpAccessor and LayerAccessor: dict-like lookup over ops and layers.

Split out of ``layer.py`` under the R43 file-size ratchet (the ``_layer_spec.py``
precedent). The accessors are a self-contained lookup surface: this module
never imports ``layer.py`` at module level (the one shared helper,
``_layer_log_to_row``, is imported inside ``LayerAccessor.to_pandas``), so the
split introduces no import cycle. ``layer.py`` re-exports both names, keeping
every historical import site (``from .layer import OpAccessor``) and pickle
qualname resolution working.
"""

import weakref
from typing import TYPE_CHECKING, Optional

from .._errors import AmbiguousOpLookupError
from .._io import FieldPolicy
from ..constants import LAYER_LOG_FIELD_ORDER
from ._accessor_base import Accessor, attach_source_honesty

if TYPE_CHECKING:
    import pandas as pd

    from .layer import Layer
    from .op import Op
    from .trace import Trace


class OpAccessor(Accessor["Op"]):
    """Scoped dict-like accessor for the Op entries owned by one Layer."""

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(self, ops: dict[int, "Op"] | None = None) -> None:
        """Initialize the accessor.

        Parameters
        ----------
        ops:
            Mapping from 1-based pass index to Op.
        """

        ops = ops or {}
        super().__init__(ops, item_list=[op for _, op in sorted(ops.items())])

    def __getitem__(self, key: int | str) -> "Op":
        """Return an Op by 0-based position or pass-qualified label."""

        if isinstance(key, int):
            return self._list[key]
        resolved = self._resolve_substring(key)
        if resolved is not None:
            return resolved
        raise KeyError(f"Op '{key}' not found in scoped Layer ops.")

    def __setitem__(self, key: int, value: "Op") -> None:
        """Set an Op by 1-based pass index."""

        self._dict[key] = value
        self._list = [op for _, op in sorted(self._dict.items())]

    def __contains__(self, key: object) -> bool:
        """Return whether key resolves to an Op."""

        if isinstance(key, int):
            return -len(self._list) <= key < len(self._list)
        if isinstance(key, str):
            try:
                self[key]
            except (KeyError, ValueError):
                return False
            return True
        return False

    # BREAKING (lovely bug 27, C02; MIGRATIONS entry owed): iteration now
    # inherits the base value semantics (yields Ops), where the deleted
    # override yielded 1-BASED call-index ints while ``[]`` indexed 0-based
    # Ops -- a silent off-by-one on multi-pass layers. ``get`` and ``repr``
    # share the one 0-based/pass-qualified basis ``__getitem__`` uses.
    def get(self, key: int | str, default: "Op | None" = None) -> "Op | None":
        """Return an Op by 0-based position or label, or ``default``."""

        try:
            return self[key]
        except (KeyError, IndexError, ValueError):
            return default

    def __repr__(self) -> str:
        """Return a bounded summary teaching the REAL index basis."""

        labels = [str(getattr(op, "label", getattr(op, "layer_label", "?"))) for op in self._list]
        shown = ", ".join(repr(label) for label in labels[:5])
        suffix = ", ..." if len(labels) > 5 else ""
        return (
            f"OpAccessor with {len(self._list)} ops "
            f"(0-based positions or pass-qualified labels): [{shown}{suffix}]"
        )

    def _resolve_substring(self, key: str) -> "Op | None":
        """Resolve Op by any scoped layer-label variant."""
        if len(self._dict) == 1:
            only_op = next(iter(self._dict.values()))
            if key in {
                only_op.layer_label,
                only_op.layer_label_short,
                only_op._label_raw,
                only_op.raw_label,
            }:
                return only_op
        parent_matches = [
            op_log
            for op_log in self._dict.values()
            if key in {op_log.layer_label, op_log.layer_label_short}
        ]
        if len(parent_matches) > 1:
            parent_label = parent_matches[0].layer_label
            qualified = ", ".join(op_log.label for op_log in parent_matches[:10])
            suffix = "..." if len(parent_matches) > 10 else ""
            raise AmbiguousOpLookupError(
                f"Layer '{parent_label}' has {len(parent_matches)} ops. Use a 0-based "
                "integer position or a pass-qualified label like "
                f"'{parent_label}:1'. Available Op labels: {qualified}{suffix}."
            )
        for op_log in self._dict.values():
            if key in {
                op_log.label,
                op_log.label_short,
                op_log._label_raw,
                op_log.raw_label,
            }:
                return op_log
        return None


class LayerAccessor(Accessor["Layer"]):
    """Dict-like accessor for Layer objects.

    Supports indexing by:
    * **layer label** (str) -- exact match against no-pass label.
    * **ordinal index** (int) -- position in execution order.
    * **pass notation** (str ``"conv2d_1_1:2"``) -- strips the pass
      suffix and returns the parent Layer.

    Available as ``trace.layers``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def _composition_note(self) -> str | None:
        """Composition breakdown for the one-line card: ops vs buffers."""

        buffer_count = sum(1 for layer in self._list if getattr(layer, "is_buffer", False))
        if not buffer_count:
            return None
        return f"{len(self._list) - buffer_count} ops, {buffer_count} buffers"

    def __init__(
        self,
        layer_logs: dict[str, "Layer"],
        source_trace: Optional["Trace"] = None,
    ) -> None:
        """Initialize an accessor over aggregate layer logs.

        Parameters
        ----------
        layer_logs:
            Mapping from layer labels to aggregate ``Layer`` objects.
        source_trace:
            Trace that owns the layer logs, if still reachable.
        """

        source_ref = weakref.ref(source_trace) if source_trace is not None else None
        super().__init__(layer_logs, source_ref=source_ref)

    def _resolve_pass_qualified(self, key: str) -> "Layer | None":
        """Resolve ``layer_label:pass`` notation to the parent Layer."""
        base, _, pass_str = key.rpartition(":")
        try:
            int(pass_str)
        except ValueError:
            return None
        return self._resolve_substring(base)

    def _resolve_substring(self, key: str) -> "Layer | None":
        """Resolve exact long or short Layer labels."""
        if key in self._dict:
            return self._dict[key]
        matches = [
            layer
            for layer in self._list
            if key in {layer.layer_label, layer.layer_label, layer.layer_label_short}
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousOpLookupError(
                f"Layer lookup '{key}' is ambiguous across {len(matches)} Layers. "
                "Use the full Layer label."
            )
        return None

    def _suggest(self, key: str) -> list[str]:
        """Return similar layer labels from the source Trace."""
        source_ref = getattr(self, "_source_ref", None)
        source = source_ref() if source_ref is not None else None
        if source is not None and hasattr(source, "find_layers"):
            return source.find_layers(str(key))
        return []

    def by_operator(self, operator: str | None = None) -> dict[str, int] | list[str]:
        """Group layers by Torch operator name.

        Parameters
        ----------
        operator:
            Optional operator name. When supplied, matching layer labels are returned.

        Returns
        -------
        Dict[str, int] | List[str]
            Counts by operator, or labels for one operator.
        """

        if operator is not None:
            return [
                layer.layer_label
                for layer in self._list
                if (layer.func_name or layer.layer_type) == operator
            ]
        counts: dict[str, int] = {}
        for layer in self._list:
            key = str(layer.func_name or layer.layer_type)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_module(self, module: str | None = None) -> dict[str, int] | list[str]:
        """Group layers by containing module address.

        Parameters
        ----------
        module:
            Optional module address. When supplied, matching layer labels are returned.

        Returns
        -------
        Dict[str, int] | List[str]
            Counts by module, or labels for one module.
        """

        if module is not None:
            return [
                layer.layer_label
                for layer in self._list
                if layer.module == module or module in getattr(layer, "modules", [])
            ]
        counts: dict[str, int] = {}
        for layer in self._list:
            key = str(layer.module or "self")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_module_and_operator(
        self,
        module: str | None = None,
        operator: str | None = None,
    ) -> dict[tuple[str, str], int] | list[str]:
        """Group layers by module and operator.

        Parameters
        ----------
        module:
            Optional module address filter.
        operator:
            Optional operator-name filter.

        Returns
        -------
        Dict[Tuple[str, str], int] | List[str]
            Counts by ``(module, operator)`` or labels matching both filters.
        """

        if module is not None and operator is not None:
            return [
                layer.layer_label
                for layer in self._list
                if (layer.module == module or module in getattr(layer, "modules", []))
                and (layer.func_name or layer.layer_type) == operator
            ]
        counts: dict[tuple[str, str], int] = {}
        for layer in self._list:
            key = (str(layer.module or "self"), str(layer.func_name or layer.layer_type))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def total(self) -> int:
        """Return the number of aggregate layers.

        Returns
        -------
        int
            Number of layer logs.
        """

        return len(self)

    # F10 (lovely bug 6): the per-member dump repr is gone -- 553 lines on real
    # gpt2. LayerAccessor inherits the one-line composition card from Accessor
    # (with the ops/buffers breakdown via ``_composition_note``); ``str()``
    # shows the first bounded members with an exact remainder.

    def to_pandas(self) -> "pd.DataFrame":
        """One row per unique layer (aggregate view), ordered by ``LAYER_LOG_FIELD_ORDER``.

        Builds each row the same way as ``Layer.to_pandas()`` so every field
        in ``LAYER_LOG_FIELD_ORDER`` is exported -- this used to hand-roll a
        12-field subset that silently dropped most populated Layer fields.
        Per-pass fields (``_MULTI_PASS_PER_CALL_LAYER_FIELDS``) are reported
        as ``None`` for multi-pass (recurrent) layers instead of raising.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e
        from .layer import _layer_log_to_row

        if not self._list:
            return pd.DataFrame(columns=LAYER_LOG_FIELD_ORDER)
        rows = [_layer_log_to_row(ll) for ll in self._list]
        frame = pd.DataFrame(rows, columns=LAYER_LOG_FIELD_ORDER)
        return attach_source_honesty(frame, self._list)


# The R43 split is a private relocation: both classes keep their historical
# PUBLIC identity (`torchlens.data_classes.layer`) so reprs, oracle surface
# dumps, and the save side of new pickles spell the pre-split module path
# (the legacy-artifact golden pins it; the unpickle allowlist accepts both).
OpAccessor.__module__ = "torchlens.data_classes.layer"
LayerAccessor.__module__ = "torchlens.data_classes.layer"
