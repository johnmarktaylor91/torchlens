"""Read sites inside modules and facet-named frozen sets for ``build_graph``.

A transcoder key is a module name (the transcoder reads the module's call
input) or an op label (it reads that op's output: a read point inside a
module, such as Gemma Scope's read of the pre-feedforward norm's value
before its ``(1 + w)`` gain). An op-label read names the MLP outputs it
writes through mapping-valued ``W_dec`` and ``b_dec`` and must run before
every one of them. A ``frozen=`` facet selector (``tl.facet("pattern")``)
holds the op behind that facet in every module that serves it, when the
facet is the whole output of one op.
"""

from __future__ import annotations

from typing import Any

from .._result import AttributionError
from ..onebackward._accessor import ReadEdgeIndex
from ..onebackward._targets import resolve_site_edge
from ._transcoders import Reader

__all__ = ["check_upstream", "facet_labels", "is_module", "op_read_label"]

_DOCS_POINTER = "See docs/reference/attribution.md, section Attribution graphs."


def _site_refusal(message: str, *, code: str) -> AttributionError:
    """Build one coded read-site or frozen-facet refusal ending in its docs pointer."""

    return AttributionError(f"{message} {_DOCS_POINTER}", code=code)


def is_module(trace: Any, name: str) -> bool:
    """Return whether ``name`` is a module of the capture."""

    try:
        trace.modules[name]
    except Exception:  # noqa: BLE001 - any lookup failure means "not a module name"
        return False
    return True


def op_read_label(index: ReadEdgeIndex, reader: Reader) -> str:
    """Return the edge label of an op-label read site, or refuse.

    Refuses a key that names no addressable op, and an op-label key whose
    ``W_dec`` is one tensor (it names no MLP output to write).
    """

    site = reader.site
    if site not in index.edges and f"{site}:1" not in index.edges:
        example = next(iter(index.edges), "mul_1_3:1")
        raise _site_refusal(
            f"The transcoder read site {site!r} names neither a module nor an op of this "
            "capture. Remedy: key the transcoder by the MLP module it replaces, from "
            "dict(model.named_modules()), or by an op label of this capture (for example "
            f"{example!r}; list them with [str(op.label) for op in trace]).",
            code="attribution_graph_site_not_found",
        )
    if any(write == site for write, _decoder in reader.decoders):
        raise _site_refusal(
            f"The transcoder keyed by the op label {site!r} gives W_dec as one tensor, "
            "which names no MLP output to write. Remedy: give W_dec and b_dec as mappings "
            "from the MLP output module name to tensors, for example "
            "W_dec={'model.layers.0.mlp': W_dec} and b_dec={'model.layers.0.mlp': b_dec}.",
            code="attribution_graph_transcoder_invalid",
        )
    return resolve_site_edge(index, site, role="target").label


def check_upstream(
    trace: Any, index: ReadEdgeIndex, reader: Reader, read: str, writes: dict[str, str]
) -> None:
    """Refuse an op-label read site that does not run before every MLP output it writes."""

    from ...receptive_field._path import ancestor_labels

    for write, _decoder in reader.decoders:
        if read not in ancestor_labels(trace, writes[write]):
            raise _site_refusal(
                f"The transcoder read site {reader.site!r} does not run before {write!r}, "
                "an MLP output it writes, so its features would read their own output. "
                "Remedy: key the transcoder by an op that runs before every MLP output it "
                f"writes (for {write!r}, for example {_upstream_example(trace, index, write)!r}).",
                code="attribution_graph_read_site_not_upstream",
            )


def _upstream_example(trace: Any, index: ReadEdgeIndex, write: str) -> str:
    """Return an op label that runs before ``write``: its module's call input."""

    ops = list(getattr(trace.modules[write], "input_ops", None) or ())
    if len(ops) != 1:
        return "<an op label from [str(op.label) for op in trace]>"
    return resolve_site_edge(index, str(ops[0]), role="target").label


def facet_labels(trace: Any, index: ReadEdgeIndex, selector: Any) -> list[str]:
    """Return the edge labels of the ops a facet selector names, one per facet home."""

    from ...intervention.hooks import _resolve_facet_specs

    specs = _resolve_facet_specs(trace, selector)
    if not specs:
        raise _site_refusal(
            f"frozen= holds {selector!r}, which no module of this capture serves. Remedy: "
            "list a module's facets with trace.modules[name].facets.keys() and name one of "
            "them, or pass the module name or op label instead.",
            code="attribution_graph_site_not_found",
        )
    labels = []
    for facet, spec in specs:
        home = spec.home_label
        if spec.home_kind != "op" or spec.capability_class != "bijective_view" or home is None:
            raise _site_refusal(
                f"frozen= holds {selector!r}; its facet {facet!r} at "
                f"{spec.home_address!r} is not the whole output of one op "
                f"({spec.home_kind}, {spec.capability_class}), so holding it would hold more "
                "or less than the facet. Remedy: pass the op labels to hold instead, for "
                "example frozen=[str(op.label) for op in trace if op.func_name == 'softmax'].",
                code="attribution_graph_frozen_facet_unsupported",
            )
        if spec.pass_index is not None and ":" not in home:
            home = f"{home}:{spec.pass_index}"
        labels.append(resolve_site_edge(index, home, role="frozen").label)
    return labels
