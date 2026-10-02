"""Span registry, region, session engine tests (torchnative W1.1-W1.3).

Includes the ONE-profiler-door dependency test the megaplan row mandates:
a second ``torch.profiler.profile`` construction site or a second grad_fn
node-hook stack anywhere in the package is RED.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import torchlens
from torchlens._registry.kernel import kernel_universe_rows
from torchlens.observability import (
    ProfilerSessionError,
    SpanError,
    SpanRegistry,
    region,
    session,
)
from torchlens.observability._session import active_session
from torchlens.observability._spans import MAX_LABEL_LENGTH, escape_label

_PACKAGE_ROOT = Path(torchlens.__file__).resolve().parent

#: The ONE sanctioned profiler construction site. F27 W2.1 burned down the
#: kernel_telemetry legacy seam: its private ATen activation now routes
#: through the session engine and the chrome-file join path is deleted.
#: Adding ANY row here requires a lane-report-visible justification.
_PROFILER_DOOR_ALLOWLIST = {
    "observability/_session.py": 1,
}


def _profiler_construction_sites() -> dict[str, int]:
    """AST census of torch.profiler.profile / autograd.profiler.profile calls."""

    sites: dict[str, int] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = ast.unparse(node.func)
            if dotted.endswith("profiler.profile"):
                count += 1
        if count:
            sites[str(path.relative_to(_PACKAGE_ROOT))] = count
    return sites


@pytest.mark.heavy
class TestOneProfilerDoor:
    """The dependency test: ONE activation knob, no second door (W1.3)."""

    def test_one_profiler_door(self) -> None:
        rows = kernel_universe_rows()
        assert rows.get("profiler_doors") == 1, (
            "The profiler_doors registry must hold EXACTLY the one session "
            "door. A second registration is a second activation knob."
        )
        sites = _profiler_construction_sites()
        assert sites == _PROFILER_DOOR_ALLOWLIST, (
            "New torch.profiler.profile construction site(s) found: "
            f"{sorted(set(sites) - set(_PROFILER_DOOR_ALLOWLIST))}. There is ONE "
            "profiler session engine (torchlens.observability.session); route "
            "through it instead of opening a second door."
        )

    #: Files besides backward.py itself that register a grad_fn node pre/posthook,
    #: each with a standing reason. Adding a row requires a lane-report-visible
    #: justification, same bar as _PROFILER_DOOR_ALLOWLIST above.
    _GRAD_FN_HOOK_SITE_EXEMPTIONS = {
        # backward.py's OWN per-fire timing helper (L9 backward residuals):
        # directly imported and called from backward.py
        # (`from ._fire_timing import _register_fire_timing_prehook`, invoked at
        # its one call site) -- split into its own file for the file-size
        # ratchet, not a second door. Its registration carries its own
        # try/except deliberately separated from the shipped coverage hook (an
        # optional measurement must never turn a complete-coverage node into a
        # gap), documented inline in _fire_timing.py.
        "backends/torch/_fire_timing.py",
        # The onebackward attribution engine's opt-in "frozen=" site masking
        # (FreezeHooks): hooks a SEPARATE, attribution-scoped backward() replay
        # the engine itself drives (never the capture's own observed backward
        # pass observe's bisector watches), installed and removed around that
        # one engine call. Not a parallel stack on the same pass.
        "attribution/onebackward/_frozen.py",
    }

    def test_no_parallel_grad_fn_hook_stack(self) -> None:
        """grad_fn node pre/posthook registration stays in backward.py's own door.

        observe's backward bisector and the F27 flip-2 marker sink must SHARE
        the shipped node-hook registration path -- a second stack would fire
        hooks twice and split pass-boundary cleanup. ``backward.py``'s own
        helper modules and the attribution engine's independently-scoped
        replay hooks are not that: see ``_GRAD_FN_HOOK_SITE_EXEMPTIONS``.
        """

        offenders: list[str] = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            relative = str(path.relative_to(_PACKAGE_ROOT))
            if relative == "backends/torch/backward.py" or relative in (
                self._GRAD_FN_HOOK_SITE_EXEMPTIONS
            ):
                continue
            text = path.read_text(encoding="utf-8")
            if "register_multi_grad_hook(" in text or ".register_prehook(" in text:
                offenders.append(relative)
        assert offenders == [], (
            f"grad_fn node-hook registration found outside backward.py: {offenders}"
        )


@pytest.mark.smoke
class TestSpanRegistry:
    """Entry registration, finally-pop, leak closure, coordinates."""

    def test_lifecycle_and_coordinates(self) -> None:
        registry = SpanRegistry(device="cpu", rank=0)
        span_id = registry.open("op::linear_1", altitude="op", owner="model")
        record = registry.close(span_id)
        assert record.end_ns is not None and record.end_ns >= record.start_ns
        assert record.clock_domain == "monotonic"
        assert record.device == "cpu" and record.rank == 0
        assert record.pid and record.tid
        assert not record.leaked

    def test_parent_nesting(self) -> None:
        registry = SpanRegistry()
        outer = registry.open("outer", altitude="op", owner="user_region")
        inner = registry.open("inner", altitude="op", owner="user_region")
        inner_record = registry.close(inner)
        registry.close(outer)
        assert inner_record.parent_span_id == outer

    def test_altitude_and_owner_vocabularies(self) -> None:
        registry = SpanRegistry()
        with pytest.raises(SpanError) as excinfo:
            registry.open("x", altitude="orbit", owner="model")
        assert excinfo.value.fields["code"] == "span_altitude_invalid"
        with pytest.raises(SpanError) as excinfo2:
            registry.open("x", altitude="op", owner="somebody")
        assert excinfo2.value.fields["code"] == "span_owner_invalid"
        # The mandatory internal bucket exists (TN-D11).
        span_id = registry.open("tl::bookkeeping", altitude="op", owner="torchlens_internal")
        registry.close(span_id)

    def test_double_close_refuses(self) -> None:
        registry = SpanRegistry()
        span_id = registry.open("x", altitude="op", owner="model")
        registry.close(span_id)
        with pytest.raises(SpanError) as excinfo:
            registry.close(span_id)
        assert excinfo.value.fields["code"] == "span_not_open"

    def test_leak_closure_discloses(self) -> None:
        registry = SpanRegistry()
        registry.open("crashed-body", altitude="grad_fn_fire", owner="model")
        leaked = registry.close_leaked()
        assert leaked == 1
        record = registry.snapshot()[-1]
        assert record.leaked
        assert registry.open_count == 0

    def test_label_escaping_and_bounds(self) -> None:
        assert escape_label("plain") == "plain"
        assert "\\n" in escape_label("evil\nlabel")
        bounded = escape_label("x" * 2000)
        assert len(bounded) <= MAX_LABEL_LENGTH
        assert bounded.endswith("...[truncated]")

    def test_scalar_metadata_refusal(self) -> None:
        registry = SpanRegistry()
        with pytest.raises(SpanError) as excinfo:
            registry.open("x", altitude="op", owner="model", metadata={"t": torch.ones(2)})
        assert excinfo.value.fields["code"] == "region_metadata_invalid"


@pytest.mark.smoke
class TestRegion:
    """tl.region semantics: ids, nesting, inertness, consumers (W1.2)."""

    def test_inert_without_a_consumer(self) -> None:
        with region("quiet", lr=0.1) as record:
            pass
        assert record.span_id is None
        assert not record.recorded_on_capture

    def test_occurrence_ids_increment(self) -> None:
        with region("epoch") as first:
            pass
        with region("epoch") as second:
            pass
        assert second.occurrence_index == first.occurrence_index + 1

    def test_parent_nesting(self) -> None:
        with region("outer"), region("inner") as inner:
            assert inner.parent_name == "outer"

    def test_non_scalar_metadata_refuses(self) -> None:
        with pytest.raises(SpanError) as excinfo, region("x", tensor=torch.ones(2)):
            pass
        assert excinfo.value.fields["code"] == "region_metadata_invalid"

    def test_region_records_span_under_session(self) -> None:
        with session(mode="owned") as live:
            with region("train_loop", lr=0.5) as record:
                pass
            assert record.span_id is not None
        assert live.result is not None
        span = next(s for s in live.result.spans if s.name == "train_loop")
        assert span.owner == "user_region"
        assert span.metadata["lr"] == 0.5
        assert span.metadata["occurrence"] == record.occurrence_index
        assert span.end_ns is not None


@pytest.mark.smoke
class TestSessionEngine:
    """Owned/borrowed lifecycles, nested refusal, restore-on-error (W1.3)."""

    def test_owned_lifecycle_produces_result(self) -> None:
        with session(mode="owned") as live:
            assert active_session() is live
            assert live.profiler is not None
        assert active_session() is None
        result = live.result
        assert result is not None
        assert result.mode == "owned"
        assert result.availability in ("empty", "partial")
        assert result.facts["kineto_join"] == "not_requested"

    def test_nested_owned_refuses_early(self) -> None:
        with session(mode="owned"):
            with pytest.raises(ProfilerSessionError) as excinfo, session(mode="owned"):
                pytest.fail("the nested body must never run")
            assert excinfo.value.fields["code"] == "profiler_session_nested"
        assert active_session() is None

    def test_borrowed_never_closes_the_callers_profiler(self) -> None:
        with torch.profiler.profile() as caller_owned:
            with session(mode="borrowed", profiler=caller_owned) as live, region("inside"):
                pass
            # The caller's profiler is still alive and steppable after our
            # session closed -- we never stepped or closed it.
            torch.ones(2) + 1
        assert live.result is not None
        assert live.result.mode == "borrowed"
        assert [s.name for s in live.result.spans] == ["inside"]

    def test_borrowed_requires_profiler(self) -> None:
        with pytest.raises(ProfilerSessionError) as excinfo:
            session(mode="borrowed")
        assert excinfo.value.fields["code"] == "profiler_session_invalid"

    def test_owned_rejects_a_passed_profiler(self) -> None:
        with pytest.raises(ProfilerSessionError):
            session(mode="owned", profiler=object())

    def test_mode_vocabulary(self) -> None:
        with pytest.raises(ProfilerSessionError):
            session(mode="psychic")

    def test_restore_on_error_closes_leaked_spans_and_slot(self) -> None:
        try:
            with session(mode="owned") as live:
                live.registry.open("about-to-crash", altitude="op", owner="model")
                raise RuntimeError("user code exploded")
        except RuntimeError:
            pass
        assert active_session() is None  # slot restored on the error path
        assert live.result is not None
        assert live.result.leaked_spans == 1
        leaked = [s for s in live.result.spans if s.leaked]
        assert len(leaked) == 1 and leaked[0].name == "about-to-crash"

    def test_session_reusable_after_error(self) -> None:
        try:
            with session(mode="owned"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        with session(mode="owned") as second:
            pass
        assert second.result is not None
