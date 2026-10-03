"""L7b entry-dark bridge battery (memo sec 5/8; request R-L7B-1).

Pins the declared late-bind posture's shippable half: mandatory bind-time
byte digests (never skipped, typed refusal on failure), declared-slot
geometry digests under the meta domain tag with the manifest sentinel
grammar, G2 domain separation at the bind boundary, validator REUSE through
the S1 surface on a real loaded runnable descriptor, and the dark gate
coupled to the still-refusing L7b capability rows.
"""

from __future__ import annotations

import re

import pytest
import torch
from _source_corpus import package_ast, package_files
from torch import nn

import torchlens as tl
from torchlens.capture._structure_only_bridge import (
    STRUCTURE_ONLY_BRIDGE_ENTRY_OPEN,
    compute_bound_state_digests,
    declared_slot_geometry_digest,
    validate_and_digest_bound_buffers,
)
from torchlens.capture.structure_only import (
    StructureOnlyCapabilityError,
    require_structure_only_capability,
)
from torchlens.errors.runnable import StateBindingError
from torchlens.options import CaptureOptions

_SENTINEL_GRAMMAR = re.compile(r"^unavailable:[A-Za-z_][A-Za-z0-9_]*$")


class _NonPersistentBufferModel(nn.Module):
    """One used non-persistent buffer, the late-bind posture's target slot."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0, 2.0, 2.0]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the non-persistent buffer so it enters declared state."""

        return x * self.scale


# ---------------------------------------------------------------------------
# Dark gate: the bridge is entry-dark and the rows still refuse
# ---------------------------------------------------------------------------


def test_bridge_is_entry_dark_and_rows_still_refuse() -> None:
    """The gate is False AND the L7b rows refuse — coupled in one pin so the
    gate can never flip without this test (and the row flips) moving too."""

    assert STRUCTURE_ONLY_BRIDGE_ENTRY_OPEN is False
    log = tl.trace(
        _NonPersistentBufferModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(structure_only=True),
    )
    for capability, code in (
        ("save_runnable", "structure_only_runnable_unsupported"),
        ("live_replay", "structure_only_replay_unsupported"),
    ):
        with pytest.raises(StructureOnlyCapabilityError) as excinfo:
            require_structure_only_capability(log, capability)
        assert excinfo.value.fields["code"] == code


def test_bridge_module_is_imported_nowhere_in_the_package() -> None:
    """Entry-dark means entry-dark: no torchlens module IMPORTS the bridge;
    it becomes reachable only through the L7b amendment implementation PR.
    (AST import scan — prose mentions, e.g. capability-row evidence cells,
    are legal.)"""

    import ast

    for path in package_files():
        if path.name == "_structure_only_bridge.py":
            continue
        tree = package_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "_structure_only_bridge" not in module, str(path)
                assert all("_structure_only_bridge" not in alias.name for alias in node.names), str(
                    path
                )
            elif isinstance(node, ast.Import):
                assert all("_structure_only_bridge" not in alias.name for alias in node.names), str(
                    path
                )


# ---------------------------------------------------------------------------
# Teaching refusals: the L7b rows name the boundary and the way forward
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_l7b_row_refusals_teach_the_boundary_and_the_remedy() -> None:
    """House rule: the L7b refusals name the flip boundary (declared
    late-bind posture) and what to do INSTEAD, not just the code."""

    log = tl.trace(
        _NonPersistentBufferModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(structure_only=True),
    )
    with pytest.raises(StructureOnlyCapabilityError) as runnable_exc:
        require_structure_only_capability(log, "save_runnable")
    message = str(runnable_exc.value)
    assert "late-bind" in message
    assert "intervention_ready=True" in message
    assert runnable_exc.value.fields["flip_event"] == "L7b amendment lands"
    with pytest.raises(StructureOnlyCapabilityError) as replay_exc:
        log.run(inputs=torch.randn(2, 3))
    assert "discharge_against" in str(replay_exc.value)


def test_entry_conflict_teaches_the_runnable_ready_flip_event() -> None:
    """The structure_only + intervention_ready entry conflict names its
    capability row and the L7b amendment as the flip event."""

    from torchlens._errors import StructureOnlyOptionConflictError

    with pytest.raises(StructureOnlyOptionConflictError) as excinfo:
        tl.trace(
            _NonPersistentBufferModel(),
            torch.randn(2, 3),
            capture=CaptureOptions(structure_only=True, intervention_ready=True),
        )
    message = str(excinfo.value)
    assert "runnable_ready_composition" in message
    assert "late-bind" in message
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"


# ---------------------------------------------------------------------------
# Bind-time digests: real byte digests, mandatory, never skipped
# ---------------------------------------------------------------------------


def test_bound_digests_route_through_the_one_content_authority() -> None:
    """Bind digests ARE tl.hash.content byte digests — one authority, and
    same-geometry different-bytes values digest differently."""

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 4.0])
    digests = compute_bound_state_digests({"a": a, "b": b})
    assert set(digests) == {"a", "b"}
    assert digests["a"] == tl.hash.content(a)
    assert digests["b"] == tl.hash.content(b)
    assert digests["a"] != digests["b"]


def test_bound_digest_refuses_meta_values_typed() -> None:
    """A meta value at bind time refuses typed: no bytes exist, and geometry
    digests belong to the declared slot, never the bound value (G2)."""

    with pytest.raises(StateBindingError) as excinfo:
        compute_bound_state_digests({"scale": torch.empty(3, device="meta")})
    assert excinfo.value.fields["code"] == "state_metadata_mismatch"
    assert excinfo.value.fields["state_dict_name"] == "scale"
    assert "meta" in str(excinfo.value)
    assert "remedy" in excinfo.value.fields


def test_bound_digest_failure_refuses_and_chains_never_skips(monkeypatch) -> None:
    """Digest failure is a typed refusal with the cause chained — the memo's
    'never a silent skip' arm."""

    def _boom(value):
        raise ValueError("bytes unreadable")

    monkeypatch.setattr(tl.hash, "content", _boom)
    with pytest.raises(StateBindingError) as excinfo:
        compute_bound_state_digests({"scale": torch.ones(3)})
    assert excinfo.value.fields["code"] == "state_metadata_mismatch"
    assert isinstance(excinfo.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# Declared-slot geometry digests: meta domain tag + sentinel grammar
# ---------------------------------------------------------------------------


def test_declared_digest_is_geometry_only_and_substrate_independent() -> None:
    """A real tensor, its meta twin, and a same-geometry different-values
    tensor all share ONE declared digest — geometry only, by construction."""

    real = torch.randn(4, 2)
    twin = torch.empty(4, 2, device="meta")
    other_values = torch.zeros(4, 2)
    assert (
        declared_slot_geometry_digest(real)
        == declared_slot_geometry_digest(twin)
        == declared_slot_geometry_digest(other_values)
    )
    assert declared_slot_geometry_digest(real) != declared_slot_geometry_digest(torch.randn(2, 4))


@pytest.mark.smoke
def test_declared_and_bound_digests_never_cross_domains() -> None:
    """G2 at the bind boundary: the geometry digest of a tensor never equals
    its byte digest — the meta domain tag separates them by construction."""

    value = torch.randn(3, 3)
    assert declared_slot_geometry_digest(value) != compute_bound_state_digests({"v": value})["v"]


def test_declared_digest_failure_degrades_to_the_sentinel_grammar(monkeypatch) -> None:
    """Genuine failure yields ``unavailable:<ExceptionName>`` — never None,
    never absent (manifest sentinel grammar)."""

    def _boom(value):
        raise RuntimeError("no geometry")

    monkeypatch.setattr(tl.hash, "content", _boom)
    sentinel = declared_slot_geometry_digest(torch.ones(2))
    assert sentinel == "unavailable:RuntimeError"
    assert _SENTINEL_GRAMMAR.match(sentinel)


# ---------------------------------------------------------------------------
# Validator REUSE on a real loaded descriptor (the S1-bound path)
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_runnable_trace(tmp_path):
    """One loaded runnable trace with a used non-persistent buffer slot."""

    model = _NonPersistentBufferModel().eval()
    trace = tl.trace(
        model,
        torch.tensor([1.0, 2.0, 3.0]),
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    path = tmp_path / "bridge.tlspec"
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tl.save(trace, str(path), level="runnable")
    return tl.load(str(path))


def test_late_bound_values_validate_through_the_s1_surface(loaded_runnable_trace) -> None:
    """The bridge binds through the EXISTING validator and every staged slot
    gets both digest families — present for every slot, never skipped."""

    bound = torch.tensor([5.0, 6.0, 7.0])
    binding = validate_and_digest_bound_buffers(
        loaded_runnable_trace.runnable_descriptor,
        {"scale": bound},
    )
    # The S1 validator stages by SLOT id (execution keying), one slot here.
    assert len(binding.staged) == 1
    (slot_key,) = binding.staged
    assert set(binding.bind_digests) == set(binding.staged)
    assert set(binding.declared_digests) == set(binding.staged)
    assert binding.bind_digests[slot_key] == tl.hash.content(binding.staged[slot_key])
    assert binding.bind_digests[slot_key] == tl.hash.content(bound)
    # G2 on the real path: the two families never cross.
    assert binding.bind_digests[slot_key] != binding.declared_digests[slot_key]


def test_late_bound_geometry_violations_refuse_via_the_s1_validator(
    loaded_runnable_trace,
) -> None:
    """Shape mismatches and missing keys refuse through the unchanged S1
    validator — the bridge adds digests, never a second validation layer."""

    descriptor = loaded_runnable_trace.runnable_descriptor
    with pytest.raises(StateBindingError):
        validate_and_digest_bound_buffers(descriptor, {"scale": torch.ones(7)})
    with pytest.raises(StateBindingError):
        validate_and_digest_bound_buffers(descriptor, {})
    with pytest.raises(StateBindingError):
        validate_and_digest_bound_buffers(
            descriptor, {"scale": torch.ones(3), "phantom": torch.ones(3)}
        )
