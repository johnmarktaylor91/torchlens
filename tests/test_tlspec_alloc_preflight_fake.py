"""r55 C3 immunizer -- op-agnostic FakeTensorMode allocation preflight (W2 lane).

r54 ``free_1`` (HIGH) / ``sec_1`` (MED): a taken-path op whose output size is a
literal integer (``torch.zeros(n)``/``arange(n)``/``x.expand(n)``) let an attacker
edit one plaintext ``manifest.json`` integer to drive a multi-terabyte allocation
and OOM-kill the default ``tl.load(path).run(inputs)`` victim. W3's run-prep
recorded-output-slot bound catches the *self-consistent* tamper (literal AND
output slot both huge), but NOT the literal-only tamper (output slot left honest),
and per-arg ``.to("meta")`` cannot bound a *factory* op (no tensor operand).

W2's ``FakeTensorMode(allow_non_fake_inputs=True)`` per-call projection closes the
class op-agnostically: it projects the call's output bytes WITHOUT allocating, so a
projected over-budget request is refused BEFORE the real allocator runs. A
data-dependent op with no fake impl (``nonzero``/``unique`` ->
``DynamicOutputShapeException``) FAILS OPEN to the run-prep bound so a legitimate
such op is never over-refused (the r51 over-catch anti-pattern).

Also covers the defensive run-side ``_decode_literal`` nesting-depth guard (C4).
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
from torchlens._runnable_execution import (
    _MAX_DECODE_NESTING_DEPTH,
    _call_is_size_driving,
    _decode_literal,
    _fake_tensor_mode_class,
)
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.runnable import LiteralAtom, LiteralAtomKind, LiteralSequence, LiteralSequenceKind

pytestmark = pytest.mark.smoke

_CAPTURE = dict(intervention_ready=True)


class _Factory(nn.Module):
    """A factory op (``torch.zeros(n)``) whose ``numel`` is a literal -- no tensor operand."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(5)
        return x + pad.sum()


class _Repeat(nn.Module):
    """A MATERIALIZING shape-expanding op (``repeat``) whose factor is a literal."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide = x.repeat(3)
        return wide.sum() + x.sum()


class _Nonzero(nn.Module):
    """A data-dependent op (``nonzero``) with no fake impl -- must FAIL OPEN and run."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.nonzero(x > 0.0)
        return idx.float().sum() + x.sum()


def _build(tmp_path: Path, name: str, model: nn.Module, x: torch.Tensor) -> Path:
    trace = tl.trace(model.eval(), x, **_CAPTURE)
    bundle = tmp_path / name
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    return bundle


def _tamper(bundle: Path, mutate: Any) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


def _scale_first_int(target: int, replacement: int) -> Any:
    def _mutate(manifest: dict[str, Any]) -> None:
        for call in manifest["run"]["calls"]:
            for literal in call.get("literal_arguments", []):
                atom = literal.get("value", {})
                if atom.get("kind") == "int" and atom.get("value") == target:
                    atom["value"] = replacement
                    return

    return _mutate


@contextmanager
def _rlimit_cap(extra: int = 1 << 30) -> Iterator[None]:
    """Cap address space so an un-refused bomb raises MemoryError instead of OOM-killing."""

    with open("/proc/self/status", encoding="ascii") as handle:
        vmsize_kb = next(int(line.split()[1]) for line in handle if line.startswith("VmSize"))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (vmsize_kb * 1024 + extra, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


# --------------------------------------------------------------------------- #
# (a) the projection engine is available and closes the class                  #
# --------------------------------------------------------------------------- #


def test_fake_tensor_mode_available_and_projects_without_allocation() -> None:
    """The projection engine imports and projects a factory bomb without allocating."""

    mode_cls = _fake_tensor_mode_class()
    assert mode_cls is not None, "FakeTensorMode must be available on the supported torch range"
    with _rlimit_cap(), mode_cls(allow_non_fake_inputs=True):
        projected = torch.zeros(10**12)  # would be 4 TB if real -> refused by rlimit
    assert int(projected.numel()) == 10**12  # projected, never allocated


def test_factory_literal_bomb_refused_before_allocation(tmp_path: Path) -> None:
    """A literal-only ``torch.zeros`` bomb (output slot left honest) is refused, no alloc."""

    bundle = _build(tmp_path, "factory.tlspec", _Factory(), torch.randn(4))
    _tamper(bundle, _scale_first_int(5, 10**12))
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            loaded.run(inputs=torch.randn(4))
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    assert int(caught.value.fields["required_bytes"]) > int(caught.value.fields["available_bytes"])


def test_repeat_multiplier_bomb_refused(tmp_path: Path) -> None:
    """A MATERIALIZING ``repeat`` with a literal multiplier bomb is refused op-agnostically."""

    x = torch.randn(4)
    bundle = _build(tmp_path, "repeat.tlspec", _Repeat(), x)
    # repeat factor is literal 3; scale to 10**12 -> a 4e12-element materialization.
    _tamper(bundle, _scale_first_int(3, 10**12))
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"


# --------------------------------------------------------------------------- #
# (b) over-trigger guards -- legit ops must NOT be over-refused                 #
# --------------------------------------------------------------------------- #


def test_untampered_factory_model_runs_verified(tmp_path: Path) -> None:
    """A legitimate factory-using model runs VERIFIED (no over-trigger)."""

    bundle = _build(tmp_path, "clean_factory.tlspec", _Factory(), torch.randn(4))
    result = tl.load(str(bundle)).run(inputs=torch.randn(4))
    assert result.report.path_faithfulness.value == "verified"


def test_data_dependent_op_fails_open_and_runs(tmp_path: Path) -> None:
    """A legit ``nonzero`` (fake impl raises ``DynamicOutputShapeException``) FAILS OPEN."""

    x = torch.randn(8)
    bundle = _build(tmp_path, "nonzero.tlspec", _Nonzero(), x)
    result = tl.load(str(bundle)).run(inputs=x.clone())
    # fail-open: the projection raised, the run still executes and is faithful.
    assert result.report.path_faithfulness.value == "verified"


def test_size_driving_gate_requires_family_and_integer_literal() -> None:
    """The cheap gate fires only for a size-driving family call carrying an int literal."""

    class _Key:
        def __init__(self, namespace: str, qualname: str) -> None:
            self.namespace = namespace
            self.qualname = qualname

    class _Entry:
        def __init__(self, namespace: str, qualname: str) -> None:
            self.key = _Key(namespace, qualname)

    zeros = _Entry("torch", "zeros")
    matmul = _Entry("torch", "matmul")
    new = _Entry("torch.Tensor", "Tensor.new")
    # factory family + integer literal -> project
    assert _call_is_size_driving(zeros, [10**12], {})
    # factory family but NO integer literal (size from a tensor) -> skip
    assert not _call_is_size_driving(zeros, [], {})
    # non-family op even with an int literal -> skip (bounded by recorded-output layer)
    assert not _call_is_size_driving(matmul, [64], {})
    # size-gated Tensor.new with an int literal -> project
    assert _call_is_size_driving(new, [5], {})
    # None entry -> never project
    assert not _call_is_size_driving(None, [10**12], {})


# --------------------------------------------------------------------------- #
# (c) defensive run-side decode-depth guard (C4)                               #
# --------------------------------------------------------------------------- #


def test_decode_literal_depth_guard_raises_not_recursionerror() -> None:
    """An over-depth nested literal raises a typed ``ValueError``, never ``RecursionError``."""

    node: Any = LiteralAtom(kind=LiteralAtomKind.INT, value=1)
    for _ in range(_MAX_DECODE_NESTING_DEPTH + 5):
        node = LiteralSequence(kind=LiteralSequenceKind.TUPLE, items=(node,))
    with pytest.raises(ValueError, match="decode depth"):
        _decode_literal(node)


def test_decode_literal_normal_nesting_ok() -> None:
    """Ordinary shallow literal nesting decodes cleanly (guard never over-fires)."""

    node: Any = LiteralSequence(
        kind=LiteralSequenceKind.TUPLE,
        items=(
            LiteralAtom(kind=LiteralAtomKind.INT, value=2),
            LiteralAtom(kind=LiteralAtomKind.INT, value=3),
        ),
    )
    assert _decode_literal(node) == (2, 3)
