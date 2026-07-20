"""r55 CLASS 3 immunizer -- the op-execution allocation-bomb seam r53 free_1 missed.

r54 ``free_1`` (HIGH): a taken-path op whose output size is a literal integer
(``torch.arange(n)``/``zeros(n)``/...) let an attacker edit one plaintext
``manifest.json`` integer to drive an ~8 TB allocation on the default
``tl.load(path).run(inputs)`` path -- the r53 state-slot / arity gates did not sit
on the op-execution literal-argument path. W3's two lane-owned layers:

* a PARSE-time literal-int magnitude gate (closes the literal/slot int64
  asymmetry), and
* a run-prep RECORDED-OUTPUT-SLOT allocation preflight
  (``_preflight_run_allocation``, ``op_allocation_preflight``) that bounds each op
  output against the live memory budget -- per-slot, never a whole-graph sum, so a
  long trace with many small outputs is never over-refused.

W2's per-call ``FakeTensorMode`` projection preflight is the primary op-agnostic
layer and composes with the ``_preflight_run_allocation`` this suite exercises.
"""

from __future__ import annotations

import json
import resource
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io.runnable_load import _validate_literal_atom_value
from torchlens._runnable_state import _preflight_run_allocation, _recorded_output_slots
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.runnable import (
    LiteralAtomKind,
    RunnableErrorCode,
    StateSlotBinding,
    StateSlotRole,
    TensorSlotDescriptor,
    TensorSlotRole,
)

pytestmark = pytest.mark.smoke


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.fc(x), dim=1)


def _build(tmp_path: Path, name: str) -> Path:
    trace = tl.trace(_Tiny().eval(), torch.randn(2, 4), intervention_ready=True)
    bundle = tmp_path / name
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    return bundle


def _tamper(bundle: Path, mutate: Any) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


@contextmanager
def _rlimit_cap(extra: int = 1 << 30) -> Iterator[None]:
    """Cap address space so an un-refused bomb allocation raises instead of OOM-killing."""

    with open("/proc/self/status", encoding="ascii") as handle:
        vmsize_kb = next(int(line.split()[1]) for line in handle if line.startswith("VmSize"))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (vmsize_kb * 1024 + extra, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _output_slot(shape: tuple[int, ...], dtype: str = "torch.float32") -> TensorSlotDescriptor:
    return TensorSlotDescriptor(
        slot_id="s_out",
        role=TensorSlotRole.OUTPUT,
        use_sites=(),
        shape=shape,
        dtype=dtype,
        rank=len(shape),
        device_type="cpu",
        device_index=None,
        mutable=False,
        version_of=None,
        producer_slot_id=None,
        output_path=None,
        input_binding=None,
        state_binding=None,
    )


# --------------------------------------------------------------------------- #
# (a) parse-time literal magnitude gate (closes the literal/slot asymmetry)     #
# --------------------------------------------------------------------------- #


def test_literal_int_over_int64_refused_at_parse() -> None:
    """A literal int past the signed-64-bit ceiling is refused, typed."""

    from torchlens._io.runnable_load import DescriptorStructuralBoundError

    with pytest.raises(DescriptorStructuralBoundError):
        _validate_literal_atom_value(LiteralAtomKind.INT, 10**20)


def test_feasible_and_sub_int64_literals_accepted() -> None:
    """Ordinary and even large-but-int64 literals still parse (gate is structural)."""

    _validate_literal_atom_value(LiteralAtomKind.INT, 8)
    _validate_literal_atom_value(LiteralAtomKind.INT, -1)
    _validate_literal_atom_value(LiteralAtomKind.INT, 10**12)  # bounded later, not at parse


def test_literal_magnitude_bomb_degrades_to_analysis_only(tmp_path: Path) -> None:
    """An over-int64 literal in a real manifest degrades the load to analysis-only."""

    bundle = _build(tmp_path, "literal.tlspec")

    def _bomb(manifest: dict[str, Any]) -> None:
        for call in manifest["run"]["calls"]:
            for literal in call.get("literal_arguments", []):
                atom = literal.get("value", {})
                if atom.get("kind") == "int":
                    atom["value"] = 10**20
                    return

    _tamper(bundle, _bomb)
    loaded = tl.load(str(bundle))
    readiness = loaded.__dict__["_runnable_readiness"]
    from torchlens.runnable import ReadinessStatus

    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert RunnableErrorCode.STATE_SHAPE_MISMATCH in {d.code for d in readiness.diagnostics}


# --------------------------------------------------------------------------- #
# (b) recorded-output-slot preflight                                          #
# --------------------------------------------------------------------------- #


def test_recorded_output_slots_selects_op_produced_roles() -> None:
    """Only INTERMEDIATE/OUTPUT slots are treated as op-allocated outputs."""

    class _Desc:
        tensor_slots = (
            _output_slot((2, 3)),
            TensorSlotDescriptor(
                slot_id="s_param",
                role=TensorSlotRole.PARAMETER,
                use_sites=(),
                shape=(3, 4),
                dtype="torch.float32",
                rank=2,
                device_type="cpu",
                device_index=None,
                mutable=False,
                version_of=None,
                producer_slot_id=None,
                output_path=None,
                input_binding=None,
                state_binding=StateSlotBinding(
                    module_path="fc",
                    state_dict_name="fc.weight",
                    semantic_role=StateSlotRole.WEIGHT,
                    trainable=True,
                    persistent=True,
                    alias_group=None,
                ),
            ),
        )

    selected = _recorded_output_slots(_Desc())  # type: ignore[arg-type]
    assert [slot.role for slot in selected] == [TensorSlotRole.OUTPUT]


def test_output_slot_preflight_refuses_infeasible_slot() -> None:
    """A single op-output slot larger than the whole budget is refused, per-slot."""

    infeasible = _output_slot((10**9, 10**9))
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        _preflight_run_allocation((), (infeasible,))
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    assert int(caught.value.fields["required_bytes"]) > int(caught.value.fields["available_bytes"])


def test_output_slot_preflight_allows_feasible_and_many_small() -> None:
    """A feasible slot and MANY small slots pass (per-slot, no whole-graph sum)."""

    # A ~1.5 GiB single output slot is feasible on any normal host: must not refuse.
    _preflight_run_allocation((), (_output_slot((200_000_000,)),))
    # 5000 small slots -- their SUM is large but each is tiny: no over-trigger.
    small = tuple(_output_slot((1024, 1024)) for _ in range(5000))
    _preflight_run_allocation((), small)


def test_output_slot_shape_bomb_refused_at_run(tmp_path: Path) -> None:
    """A tampered op-output slot shape is refused at run-prep, before any allocation."""

    bundle = _build(tmp_path, "outbomb.tlspec")

    def _bomb(manifest: dict[str, Any]) -> None:
        for slot in manifest["run"]["tensor_slots"]:
            if slot.get("role") in {"output", "intermediate"}:
                slot["shape"] = [10**9, 10**9]
                slot["rank"] = 2
                return

    _tamper(bundle, _bomb)
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
    with _rlimit_cap(), pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=torch.randn(2, 4))
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"


# --------------------------------------------------------------------------- #
# (c) over-trigger guard                                                      #
# --------------------------------------------------------------------------- #


def test_untampered_bundle_runs_verified(tmp_path: Path) -> None:
    """The class is closed without over-triggering on a legitimate bundle."""

    bundle = _build(tmp_path, "clean.tlspec")
    loaded = tl.load(str(bundle))
    result = loaded.run(inputs=torch.randn(2, 4))
    assert result.report.path_faithfulness.value == "verified"
