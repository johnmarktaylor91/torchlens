"""W051-IO: write/read symmetry for ``metadata.pkl`` (AUD-CODE 2.20 / 3.11g).

``tl.save`` must never write metadata bytes that a default ``tl.load`` cannot
read back. Before this lane an arbitrary user object under
``Trace.annotations`` saved silently and made the WHOLE artifact unloadable
with a misleading "corrupt pickle" refusal, and a tensor logged during the
forward (``tl.observers.log_value``) could never be loaded because it still
carried its ``tl_*`` capture attributes.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io._portability_preflight import (
    preflight_metadata_portability,
    sanitize_annotation_tensors,
)
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.utils._torch_compat import HAS_SAFE_WEIGHTS_ONLY_LOAD

#: The preflight's own load-preflight dry run round-trips the metadata bytes through
#: the SAME default-deny unpickler a real tl.load() would use, so an embedded tensor
#: value (tl.observers.log_value, or a plain/Parameter tensor stashed in annotations)
#: trips the CVE-2025-32434 fail-closed gate on torch<2.6 before the mediated-allocation
#: or portability checks ever run -- a correct refusal of a genuinely unsupported
#: artifact shape there, not a bug (see test_io_mediated_allocation_bypass.py's and
#: test_w051_io_artifact_anchors.py's identical gate / d04aa2d1f).
_SKIP_BELOW_SAFE_WEIGHTS_ONLY_LOAD = pytest.mark.skipif(
    not HAS_SAFE_WEIGHTS_ONLY_LOAD,
    reason="embedded-tensor metadata (an annotation tensor logged via "
    "tl.observers.log_value, or a plain/Parameter tensor under annotations) is refused "
    "by the preflight's own load-preflight dry run on torch<2.6 (CVE-2025-32434); the "
    "round-trip is a torch>=2.6 feature",
)


class _UserObject:
    """A user class the default-deny loader can never construct."""


def _tiny_trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))


def _refusal(trace: tl.Trace, path: Path) -> TorchLensIOError:
    with pytest.raises(TorchLensIOError) as excinfo:
        tl.save(trace, path)
    assert not path.exists(), "a refused save must leave nothing behind"
    return excinfo.value


def test_user_object_under_user_namespace_refuses_before_write(tmp_path: Path) -> None:
    trace = _tiny_trace()
    trace.annotations.setdefault("user", {})["foo"] = _UserObject()
    exc = _refusal(trace, tmp_path / "a.tlspec")
    assert exc.fields["code"] == "annotation_value_unportable"
    assert exc.fields["field"] == "Trace.annotations['user']['foo']"
    assert "Nothing was written" in str(exc)


def test_user_object_at_top_level_annotation_key_refuses_naming_the_key(tmp_path: Path) -> None:
    trace = _tiny_trace()
    trace.annotations["foo"] = _UserObject()
    exc = _refusal(trace, tmp_path / "a.tlspec")
    assert exc.fields["code"] == "annotation_value_unportable"
    assert exc.fields["field"] == "Trace.annotations['foo']"


def test_user_object_in_op_annotation_refuses_naming_the_op(tmp_path: Path) -> None:
    trace = _tiny_trace()
    op = trace["linear_1_1"]
    op.annotations["foo"] = _UserObject()
    exc = _refusal(trace, tmp_path / "a.tlspec")
    assert exc.fields["code"] == "annotation_value_unportable"
    # The op's annotations mirror onto its persisted Layer record too; the
    # located path names whichever persisted record fails first, by label.
    assert exc.fields["field"].startswith("Trace.layer_")
    assert "linear_1_1" in exc.fields["field"]
    assert exc.fields["field"].endswith(".annotations['foo']")


@_SKIP_BELOW_SAFE_WEIGHTS_ONLY_LOAD
def test_log_value_tensor_round_trips(tmp_path: Path) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.lin(x)
            tl.observers.log_value("y_tensor", y)
            tl.observers.log_value("count", 3)
            return torch.relu(y)

    torch.manual_seed(0)
    trace = tl.trace(Model(), torch.randn(1, 4))
    live_value = trace.annotations["logged_values"]["y_tensor"]
    assert isinstance(live_value, torch.Tensor)
    path = tmp_path / "lv.tlspec"
    tl.save(trace, path)
    # The live trace keeps its session-time object; only the artifact copy is coerced.
    assert trace.annotations["logged_values"]["y_tensor"] is live_value
    loaded = tl.load(path)
    restored = loaded.annotations["logged_values"]["y_tensor"]
    assert type(restored) is torch.Tensor
    assert not restored.requires_grad
    torch.testing.assert_close(restored, live_value.detach())
    assert loaded.annotations["logged_values"]["count"] == 3


@_SKIP_BELOW_SAFE_WEIGHTS_ONLY_LOAD
def test_parameter_and_plain_tensor_annotations_round_trip(tmp_path: Path) -> None:
    trace = _tiny_trace()
    plain = torch.arange(3.0)
    trace.annotations.setdefault("user", {})["plain"] = plain
    trace.annotations["user"]["param"] = nn.Parameter(torch.ones(2))
    path = tmp_path / "p.tlspec"
    tl.save(trace, path)
    loaded = tl.load(path)
    torch.testing.assert_close(loaded.annotations["user"]["plain"], plain)
    restored = loaded.annotations["user"]["param"]
    assert type(restored) is torch.Tensor
    torch.testing.assert_close(restored, torch.ones(2))


@pytest.mark.smoke
def test_sanitize_rebuilds_containers_and_keeps_plain_tensors_by_identity() -> None:
    plain = torch.zeros(2)
    tagged = torch.ones(2)
    tagged.tl_marker = "session"  # type: ignore[attr-defined]
    grad = torch.ones(2, requires_grad=True)
    source = {"user": {"plain": plain, "tagged": tagged, "grad": grad, "n": 1}, "list": [tagged]}
    sanitized = sanitize_annotation_tensors(source)
    assert sanitized is not source and sanitized["user"] is not source["user"]
    assert sanitized["user"]["plain"] is plain
    assert sanitized["user"]["tagged"] is not tagged
    assert not hasattr(sanitized["user"]["tagged"], "tl_marker")
    assert not sanitized["user"]["grad"].requires_grad
    torch.testing.assert_close(sanitized["list"][0], tagged.detach())
    # The source tree is untouched (immutability by default).
    assert source["user"]["tagged"] is tagged and hasattr(tagged, "tl_marker")
    assert sanitized["user"]["n"] == 1


def test_preflight_returns_bytes_the_loader_reads() -> None:
    state = {"tlspec_version": 9, "annotations": {"user": {"k": [1, 2.0, "s", torch.Size([2])]}}}
    data = preflight_metadata_portability(
        state, unpickler_factory=_RenameAwareUnpickler, bundle_path="<memory>"
    )
    assert _RenameAwareUnpickler(io.BytesIO(data)).load() == state


@pytest.mark.smoke
def test_preflight_names_metadata_paths_outside_annotations() -> None:
    state = {"tlspec_version": 9, "some_field": [1, {"deep": _UserObject()}]}
    with pytest.raises(TorchLensIOError) as excinfo:
        preflight_metadata_portability(
            state, unpickler_factory=_RenameAwareUnpickler, bundle_path="<memory>"
        )
    assert excinfo.value.fields["code"] == "metadata_value_unportable"
    assert excinfo.value.fields["field"] == "Trace.some_field[1]['deep']"


@_SKIP_BELOW_SAFE_WEIGHTS_ONLY_LOAD
def test_streamed_capture_with_logged_tensor_round_trips(tmp_path: Path) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.lin(x)
            tl.observers.log_value("y_tensor", y)
            return torch.relu(y)

    torch.manual_seed(0)
    path = tmp_path / "stream.tlspec"
    trace = tl.trace(Model(), torch.randn(1, 4), storage=tl.to_disk(path))
    loaded = tl.load(path)
    restored = loaded.annotations["logged_values"]["y_tensor"]
    assert type(restored) is torch.Tensor
    torch.testing.assert_close(restored, trace.annotations["logged_values"]["y_tensor"].detach())


class _HostileTensor(torch.Tensor):
    """A tensor subclass whose ``detach`` raises: the sanitizer must not crash."""

    def detach(self) -> torch.Tensor:  # type: ignore[override]
        raise RuntimeError("hostile detach")


def test_hostile_tensor_subclass_passes_through_sanitizer_and_refuses_typed(
    tmp_path: Path,
) -> None:
    """The sanitizer's exotic-subclass arm returns the value unchanged; the preflight refuses.

    ``_plain_tensor`` coerces every shape TorchLens itself produces, but a
    subclass that cannot even be detached is left for the dry run to judge:
    the save still refuses typed instead of crashing inside the sanitizer or
    writing bytes the loader cannot read.
    """

    hostile = torch.zeros(2).as_subclass(_HostileTensor)
    sanitized = sanitize_annotation_tensors({"user": {"t": hostile}})
    assert sanitized["user"]["t"] is hostile

    trace = _tiny_trace()
    trace.annotations.setdefault("user", {})["t"] = hostile
    exc = _refusal(trace, tmp_path / "hostile.tlspec")
    assert exc.fields["code"] == "annotation_value_unportable"
    assert exc.fields["field"] == "Trace.annotations['user']['t']"


def test_locate_names_whole_state_when_every_part_is_portable_alone() -> None:
    """The culprit walk's fallback: no single part fails, so the whole state is named."""

    from torchlens._io._portability_preflight import _locate_unportable

    located, in_annotations = _locate_unportable(
        {"tlspec_version": 9, "annotations": {"k": [1, 2]}}, _RenameAwareUnpickler
    )
    assert located == "Trace (whole persisted state)"
    assert in_annotations is False
