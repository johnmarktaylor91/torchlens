"""Parse-time hardening for runnable descriptors (grind b3-l1, R10 supplement B)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable_load import (
    ContextFieldInvalidError,
    _parse_rng_profile,
    _validated_dtype_literal,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import ReadinessStatus


class TinyModel(nn.Module):
    """One-linear model for runnable-artifact tamper fixtures."""

    def __init__(self) -> None:
        """Initialize the single layer."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the recorded static path."""

        return torch.relu(self.linear(value))


@pytest.fixture
def runnable_manifest(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Save one runnable artifact and return its path, manifest path, and JSON."""

    captured = tl.trace(
        TinyModel(),
        torch.ones(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "tamper.tlspec"
    captured.save(path, level="runnable", include_weights=True)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return path, manifest_path, manifest


def test_edited_execution_context_fails_runtime_fingerprint(
    runnable_manifest: tuple[Path, Path, dict],
) -> None:
    """A hand-edited per-call grad_enabled is caught by the recomputed fingerprint.

    The fingerprint was previously written and parsed but never verified, so
    this exact edit parsed cleanly and replayed VERIFIED under the edited
    context (F-R10-B1).
    """

    path, manifest_path, manifest = runnable_manifest
    call = manifest["run"]["calls"][0]
    call["execution_context"]["grad_enabled"] = not call["execution_context"]["grad_enabled"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = tl.load(path)
    assert loaded.readiness.status is ReadinessStatus.UNAVAILABLE
    assert any("fingerprint" in diagnostic.message for diagnostic in loaded.readiness.diagnostics)


def test_edited_argument_names_fail_runtime_fingerprint(
    runnable_manifest: tuple[Path, Path, dict],
) -> None:
    """A hand-edited call arity/argument list refuses at parse."""

    path, manifest_path, manifest = runnable_manifest
    call = manifest["run"]["calls"][0]
    call["argument_names"] = [*call["argument_names"], "forged"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = tl.load(path)
    assert loaded.readiness.status is ReadinessStatus.UNAVAILABLE


def test_reserved_name_prefix_alias_group_refuses(
    runnable_manifest: tuple[Path, Path, dict],
) -> None:
    """A crafted 'name:'-prefixed alias_group is refused at parse (F-R10-B2).

    The staging loader keys unaliased slots by the synthetic fallback
    ``name:<state_dict_name>``; a crafted group in that namespace could share
    one tensor between two unrelated slots.
    """

    path, manifest_path, manifest = runnable_manifest
    tampered = False
    for slot in manifest["run"]["tensor_slots"]:
        binding = slot.get("state_binding")
        if binding is not None:
            binding["alias_group"] = "name:linear.bias"
            tampered = True
            break
    assert tampered
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = tl.load(path)
    assert loaded.readiness.status is ReadinessStatus.UNAVAILABLE


@pytest.mark.smoke
def test_parse_rng_profile_refuses_absent_or_coerced_fields() -> None:
    """The REQUIRED host-RNG profile is never defaulted or coerced (F-R10-B4)."""

    with pytest.raises(ContextFieldInvalidError):
        _parse_rng_profile(None)
    with pytest.raises(ContextFieldInvalidError):
        _parse_rng_profile("not-a-mapping")
    with pytest.raises(ContextFieldInvalidError):
        _parse_rng_profile({"capture_seed": None})
    with pytest.raises(ContextFieldInvalidError):
        _parse_rng_profile({"host_rng_consumed": "false", "capture_seed": None})
    with pytest.raises(ContextFieldInvalidError):
        _parse_rng_profile({"host_rng_consumed": False})
    with pytest.raises(ContextFieldInvalidError):
        _parse_rng_profile({"host_rng_consumed": False, "capture_seed": "7"})
    with pytest.raises(ContextFieldInvalidError):
        _parse_rng_profile({"host_rng_consumed": False, "capture_seed": True})

    profile = _parse_rng_profile({"host_rng_consumed": True, "capture_seed": 7})
    assert profile.host_rng_consumed is True
    assert profile.capture_seed == 7
    profile = _parse_rng_profile({"host_rng_consumed": False, "capture_seed": None})
    assert profile.host_rng_consumed is False
    assert profile.capture_seed is None


@pytest.mark.smoke
def test_validated_dtype_literal_returns_canonical_spelling() -> None:
    """A bare dtype spelling canonicalizes so the consumer compare binds (F-R10-B6)."""

    assert _validated_dtype_literal("f", "float32") == "torch.float32"
    assert _validated_dtype_literal("f", "torch.float32") == "torch.float32"
    with pytest.raises(ContextFieldInvalidError):
        _validated_dtype_literal("f", "not_a_dtype")


def test_encode_literal_exact_type_discipline() -> None:
    """Container subclasses refuse typed instead of laundering to builtins (F-R10-B3)."""

    from collections import OrderedDict, namedtuple

    from torchlens._io.runnable import _encode_literal, _UnsupportedLiteralError
    from torchlens._runnable_verification import _decode_literal

    # Exact builtins and the ratified torch.Size allowlist still encode.
    assert _decode_literal(_encode_literal([1, 2])) == [1, 2]
    assert _decode_literal(_encode_literal((1, 2))) == (1, 2)
    assert _decode_literal(_encode_literal({"k": 3})) == {"k": 3}
    assert _decode_literal(_encode_literal(torch.Size((2, 3)))) == (2, 3)

    Point = namedtuple("Point", ["x", "y"])
    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal(Point(1, 2))
    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal(OrderedDict([("k", 1)]))

    class MyList(list):
        """List subclass carrying semantic type identity."""

    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal(MyList([1]))


@pytest.mark.smoke
def test_encode_literal_bounds_nesting_below_every_decode_ceiling() -> None:
    """Deep nesting refuses typed at SAVE, never a RecursionError or an unloadable bundle (F-R10-B5)."""

    from torchlens._io.runnable import (
        _MAX_ENCODE_LITERAL_NESTING_DEPTH,
        _encode_literal,
        _UnsupportedLiteralError,
    )

    shallow: object = 1
    for _ in range(_MAX_ENCODE_LITERAL_NESTING_DEPTH):
        shallow = [shallow]
    _encode_literal(shallow)

    deep: object = 1
    for _ in range(_MAX_ENCODE_LITERAL_NESTING_DEPTH + 2):
        deep = [deep]
    with pytest.raises(_UnsupportedLiteralError, match="nesting"):
        _encode_literal(deep)

    bomb: object = 1
    for _ in range(1000):
        bomb = [bomb]
    with pytest.raises(_UnsupportedLiteralError, match="nesting"):
        _encode_literal(bomb)


def test_runnable_resave_refusal_is_honest_and_code_bearing(tmp_path: Path) -> None:
    """Loaded-runnable re-save never claims 'no state records' falsely (F-R10-2).

    A loaded artifact with embedded weights carries the full capture-time
    state_dict, so the state-universe ladder must consult it: the re-save now
    proceeds to the producer preflight and refuses with the TYPED diagnostics
    naming the genuinely missing session-time producer records. An artifact
    saved WITHOUT embedded weights keeps the universe refusal, now with an
    honest reason and a stable error code (fields were empty before).
    """

    from torchlens._io import TorchLensIOError
    from torchlens.errors import RunnablePreflightError

    captured = tl.trace(
        TinyModel(),
        torch.ones(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    with_weights = tmp_path / "with-weights.tlspec"
    captured.save(with_weights, level="runnable", include_weights=True)
    loaded = tl.load(with_weights)
    with pytest.raises(RunnablePreflightError) as preflight:
        tl.save(loaded, tmp_path / "resave.tlspec", level="runnable", include_weights=True)
    assert preflight.value.fields["code"] == "sparse_preflight_failed"
    assert preflight.value.fields["diagnostics"]

    recaptured = tl.trace(
        TinyModel(),
        torch.ones(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    without_weights = tmp_path / "no-weights.tlspec"
    recaptured.save(without_weights, level="runnable")
    loaded_bare = tl.load(without_weights)
    with pytest.raises(TorchLensIOError) as refusal:
        tl.save(loaded_bare, tmp_path / "resave2.tlspec", level="runnable")
    assert refusal.value.fields["code"] == "run_capability_unavailable"
    assert refusal.value.fields["detection_stage"] == "runnable_resave_state_universe"
