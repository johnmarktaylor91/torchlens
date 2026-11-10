"""Call-site family of the shared op fields: the code context and the call-site chain.

One of the per-family modules split out of ``_ops_shared_fields`` along the
families of ``torchlens/_trace_core/field_table.py`` (here ``site``), so a lane
changing how the call site is observed edits this module alone.
"""

from typing import TYPE_CHECKING, Any

from ..._utils.introspection import _get_code_context

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_predicate_call_site_chain",
    "_set_call_site_fields",
)


def _set_call_site_fields(self: "Trace", fields_dict: dict[str, Any], func_call_id: int) -> None:
    """Record the call's ``code_context`` and stage its call-site chain for the row commit."""

    code_context_cache = getattr(self, "_code_context_cache", None)
    if code_context_cache is None:
        code_context_cache = {}
        self._code_context_cache = code_context_cache
    call_site_out: list[Any] = []
    fields_dict["code_context"] = _get_code_context(
        self.num_context_lines,
        source_loading_enabled=self._options_save_source_context,
        disable_col_offset=False,
        context_cache=code_context_cache,
        call_site_out=call_site_out,
    )
    self._raw_graph_ws._pending_call_site = (func_call_id, call_site_out[0])


def _predicate_call_site_chain(trace: "Trace") -> tuple[Any, ...]:
    """Return the call-site chain of the torch call being recorded.

    The same frame walk exhaustive capture runs for its code context
    (:func:`_set_call_site_fields`), asked only for the chain: no source lines, no
    column offsets, and the per-capture anchor in ``trace._code_context_cache``
    stops the walk at the capture call. Empty when no Python frame lies
    inside the innermost stateful module call. The key and class string it
    feeds are the ones exhaustive capture mints (``_ops_activations``), so the
    cooked Trace groups by call site (``callsite_v2``) with the same result.
    """

    context_cache = trace._code_context_cache
    call_site_out: list[Any] = []
    _get_code_context(
        0,
        source_loading_enabled=False,
        disable_col_offset=True,
        context_cache=context_cache,
        call_site_out=call_site_out,
    )
    return call_site_out[0] if call_site_out else ()
