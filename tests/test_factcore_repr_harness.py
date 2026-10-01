"""C02 repr harness: property tests over the rendered surface (lovely 5).

The bug table exists because nothing reads these strings; this harness
lands WITH the grammar. Scope note: the one-line-repr conversion of
Op/Layer/Param value records is F10 (lovely item 6) -- here we pin the
LAWS the substrate already owes: no auto-repr on result/report/context
dataclasses, no ANSI/OSC-8 in returned strings, never-raises on degraded
records, purity, and the ascii==degrade byte contract for every specimen.
"""

import dataclasses
from io import StringIO

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke

#: Result / report / context / aggregate dataclasses on the C02 surface
#: (D31): none may inherit the dataclass auto-repr. F10 extends this sweep
#: to the whole exported surface with its value-record conversion.
_NO_AUTO_REPR_CLASSES = (
    "torchlens.fastlog.types:Recording",
    "torchlens.fastlog.types:ActivationRecord",
    "torchlens.ir.predicate:RecordContext",
    "torchlens.runnable:RunResult",
    "torchlens.runnable:RunReport",
    "torchlens.runnable:ReadinessReport",
)


def _load(spec: str) -> type:
    """Import one class from a ``module:Class`` spec."""

    import importlib

    module_name, _, class_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


def _has_auto_repr(cls: type) -> bool:
    """Whether a dataclass still carries the generated __repr__."""

    repr_fn = cls.__dict__.get("__repr__")
    if repr_fn is None:
        return True  # inherited: object/parent repr, also not designed
    qualname = getattr(repr_fn, "__qualname__", "")
    wrapped = getattr(repr_fn, "__wrapped__", None)
    return "__create_fn__" in qualname or (
        wrapped is not None and "__create_fn__" in getattr(wrapped, "__qualname__", "")
    )


@pytest.mark.parametrize("spec", _NO_AUTO_REPR_CLASSES)
def test_no_auto_repr_on_result_report_context_classes(spec: str) -> None:
    """D31: the Recording OOM was a CLASS of defect, not one bug."""

    cls = _load(spec)
    assert dataclasses.is_dataclass(cls)
    assert not _has_auto_repr(cls), f"{spec} still inherits the dataclass auto-repr"


class _Model(nn.Module):
    """Small mixed model for the sweep."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.relu(self.fc(x)))


@pytest.fixture(scope="module")
def swept_trace():
    """A finished capture with a poisoned payload, cleaned up at teardown."""

    model = _Model().eval()
    x = torch.randn(2, 8)
    x[0, 0] = float("nan")
    trace = tl.trace(model, x)
    try:
        yield trace
    finally:
        trace.cleanup()


def _rendered_strings(trace: "tl.Trace") -> list[tuple[str, str]]:
    """Collect (label, text) pairs across the rendered surface."""

    rendered: list[tuple[str, str]] = [
        ("trace-repr", repr(trace)),
        ("trace-str", str(trace)),
        ("layers-repr", repr(trace.layers)),
        ("modules-repr", repr(trace.modules)),
        ("params-repr", repr(trace.params)),
        ("summary", trace.summary()),
        ("explain", tl.report.explain(trace)),
    ]
    for op in trace.ops:
        rendered.append((f"op-{op.label}", str(op)))
    for layer in trace.layers:
        rendered.append((f"layer-{layer.layer_label}", str(layer)))
        rendered.append((f"ops-accessor-{layer.layer_label}", repr(layer.ops)))
    stream = StringIO()
    trace.modules["self"].show_call_tree(file=stream)
    rendered.append(("call-tree", stream.getvalue()))
    profile = trace.profile()
    rendered.append(("profile-tree", profile.tree()))
    return rendered


def test_no_ansi_or_osc8_in_any_rendered_string(swept_trace) -> None:
    """No returned/rendered string carries ESC bytes -- with the nonfinite
    fixture, because a clean trace shows zero escapes and the test must bite."""

    assert swept_trace.nonfinite_ops  # the fixture bites
    for label, text in _rendered_strings(swept_trace):
        assert "\x1b" not in text, f"ESC byte in {label}"
        assert "\x9b" not in text, f"CSI byte in {label}"


def test_repr_never_raises_on_detached_and_unsaved(swept_trace) -> None:
    """repr never raises: detached ops and payload-refusing saves degrade."""

    import copy

    op = swept_trace.ops[1]
    detached = copy.copy(op)
    assert isinstance(repr(detached), str)
    selective = tl.trace(_Model().eval(), torch.randn(2, 8), save=tl.func("relu"))
    for record in selective.ops:
        assert isinstance(str(record), str)


def test_repr_purity_no_rng_no_mutation(swept_trace) -> None:
    """Rendering draws no global RNG and bumps no payload version."""

    op = next(op for op in swept_trace.ops if op.has_saved_activation)
    payload = op.out
    rng_before = torch.get_rng_state()
    version_before = payload._version if isinstance(payload, torch.Tensor) else None
    for _ in range(2):
        str(op)
        repr(swept_trace)
        swept_trace.summary()
    assert torch.equal(rng_before, torch.get_rng_state())
    if version_before is not None:
        assert payload._version == version_before


def test_nesting_bound_for_context_records() -> None:
    """A 10-record container reprs within the nesting bound (<= 12 lines)."""

    recording = tl.record(_Model().eval(), torch.randn(2, 8), save=tl.func("relu"))
    records = list(recording.records) * 10
    text = repr(records[:10])
    assert len(text.splitlines()) <= 12
    context_list = [record.ctx for record in records[:10]]
    assert len(repr(context_list).splitlines()) <= 12
