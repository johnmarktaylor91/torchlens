"""Run facts and the replay journal a cooked Recording hands its Trace shell.

Both helpers read sealed cores only, so they sit outside the capture import
cycle; ``RecordingProjector.project`` (``projectors``) calls them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._multipass_cook import FoldedPasses
    from .session import CapturedRunCore

__all__ = ("_cook_facts", "_replay_capture_events")


def _replay_capture_events(cores: tuple[CapturedRunCore, ...], all_events: list[Any]) -> Any:
    """Return the first core's journal re-projected over the whole event stream."""

    selected = next(
        (
            (core, core.projection_facts.get("capture_events"))
            for core in cores
            if core.projection_facts.get("capture_events") is not None
        ),
        None,
    )
    if selected is None or selected[1] is None:
        return None
    source_core, capture_events = selected
    # The projection clone was made BEFORE the capture-end seal, so its
    # watermark normally arrives via the seal's dual-stamp; the sealed core's
    # own watermark is the authoritative fallback (reviewer note S-N2: either
    # transport alone suffices, both are kept so a future re-ordering of
    # snapshot/seal cannot silently strand the filter anchor).
    if (
        getattr(capture_events, "core_seal_watermark", None) is None
        and source_core.amendment_watermark is not None
    ):
        capture_events.core_seal_watermark = source_core.amendment_watermark
    capture_events = capture_events._copy_for_replay(projected_op_events=all_events)
    capture_events.raw_layer_counter = max((event.raw_index for event in all_events), default=0)
    return capture_events


def _cook_facts(
    cores: tuple[CapturedRunCore, ...], folded: FoldedPasses | None
) -> tuple[dict[str, Any], tuple[Any, ...], tuple[str, ...], tuple[str | None, ...]]:
    """Return the run facts and outputs the cook installs: the last pass's, merged when folded."""

    last_facts: dict[str, Any] = dict(cores[-1].projection_facts) if cores else {}
    if folded is None:
        return (
            last_facts,
            tuple(last_facts.get("output_tensors", ())),
            tuple(last_facts.get("output_tensor_addresses", ())),
            tuple(last_facts.get("output_labels", ())),
        )
    last_facts["call_site_classes"] = dict(folded.call_site_classes)
    last_facts["layer_counter"] = folded.layer_counter
    return (
        last_facts,
        tuple(folded.output_tensors),
        tuple(folded.output_tensor_addresses),
        tuple(folded.output_labels),
    )
