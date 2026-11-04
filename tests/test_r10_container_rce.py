"""Round-10 security regression: RCE via output-container reconstruction + member traversal.

FINDING 1 (CRITICAL RCE): the portable output ``ContainerSpec`` (torchlens-owned, so the
safe unpickler correctly admits it) carries ``type_module`` / ``type_qualname`` as plain
KEEP strings plus literal constructor args. Reconstruction did its OWN
``importlib.import_module(spec.type_module)`` + getattr-walk + ``container_type(*values)``
OUTSIDE both the r1-r9 unpickler/callable gates. A crafted bundle naming
``subprocess`` / ``Popen`` therefore executed arbitrary code on a plain
``tl.load()`` + ``trace.run()`` -- the documented flow. The fix
(``torchlens/_ir/container.py``) makes the container-type resolver default-deny: it never
imports an attacker-named module (resolves ONLY from ``sys.modules``) and returns a type
ONLY when it structurally matches the benign container kind recorded in the spec, so a
disallowed type is refused BEFORE any construction.

FINDING 2 (MEDIUM): ``bundle.json`` member paths were joined verbatim
(``bundle_path / relative_path``) without the anti-traversal guard applied to blob paths,
so an absolute / ``..`` member path escaped the bundle root. The fix
(``torchlens/_io/bundle.py``) routes member paths through a bundle-root containment guard.
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from _format3_envelope import redigest
from torch import nn

import torchlens as tl
from torchlens._io import ArtifactError
from torchlens._io.format3.values import ValueEncoder
from torchlens._ir.container import (
    ContainerReconstructionError,
    ContainerSpec,
    NamedField,
    _resolve_container_type,
    rebuild_container_from_spec,
)
from torchlens.errors.runnable import RunPreconditionError
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


class _TupleModel(nn.Module):
    """Return a plain tuple output used to build a tamperable runnable bundle."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        y = self.lin(x)
        return (y, y * 2)


class _Pair(NamedTuple):
    """Module-scope namedtuple output (resolvable, admissible)."""

    value: torch.Tensor
    meta: int


class _NamedTupleModel(nn.Module):
    """Return a genuine namedtuple output."""

    def forward(self, x: torch.Tensor) -> Any:
        return _Pair(x + 1, 7)


class _OrderedDictModel(nn.Module):
    """Return an ``OrderedDict`` output (faithfully reconstructable mapping subtype)."""

    def forward(self, x: torch.Tensor) -> Any:
        return OrderedDict([("value", x + 1), ("meta", 3)])


class _DictModel(nn.Module):
    """Return a plain three-tensor dict output (three leaves, no type reference)."""

    def forward(self, x: torch.Tensor) -> Any:
        return {"a": x + 1, "b": x + 2, "c": x + 3}


_CONTAINER_SPEC_TYPE = "torchlens._ir.container.ContainerSpec"


def _tampered_spec_cell(*, type_module: str, type_qualname: str, sentinel: Path) -> Any:
    """Return the typed-JSON cell of a ``ContainerSpec`` that names an arbitrary constructor."""

    spec = ContainerSpec(kind="tuple")
    child = ContainerSpec(kind="literal", literal_value=["/bin/sh", "-c", f"touch {sentinel}"])
    tampered = {
        "kind": "namedtuple",
        "type_module": type_module,
        "type_qualname": type_qualname,
        "fields": ("args",),
        "length": None,
        "keys": (),
        "child_specs": ((NamedField("args"), child),),
    }
    for key, value in tampered.items():
        object.__setattr__(spec, key, value)
    return ValueEncoder().encode(spec, "r10-tamper")


def _tampered_mapping_spec_cell(*, sentinel: Path) -> Any:
    """Return the typed-JSON cell of a three-slot ``dict`` spec whose type is ``subprocess.Popen``.

    The three keys are the argv of a shell command and each key holds one leaf slot, so the
    spec consumes exactly the three real output leaves: the leaf-count check passes and the
    mapping-subtype rebuild reaches ``container_type(items)``, which for ``Popen`` would run
    ``list(items)`` (the keys) as a command. Only the admissibility gate stands in between.
    """

    spec = ContainerSpec(kind="dict")
    tampered = {
        "type_module": "subprocess",
        "type_qualname": "Popen",
        "keys": ("/bin/sh", "-c", f"touch {sentinel}"),
        "child_specs": (),
    }
    for key, value in tampered.items():
        object.__setattr__(spec, key, value)
    return ValueEncoder().encode(spec, "r10-tamper-mapping")


def _replace_container_specs(node: Any, cell: Any) -> int:
    """Replace every ``ContainerSpec`` record cell under ``node`` in place; return the count."""

    replaced = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and value.get("type") == _CONTAINER_SPEC_TYPE:
                node[key] = copy.deepcopy(cell)
                replaced += 1
            else:
                replaced += _replace_container_specs(value, cell)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            if isinstance(value, dict) and value.get("type") == _CONTAINER_SPEC_TYPE:
                node[index] = copy.deepcopy(cell)
                replaced += 1
            else:
                replaced += _replace_container_specs(value, cell)
    return replaced


def _tamper_output_container_type(
    bundle: Path, *, type_module: str, type_qualname: str, sentinel: Path
) -> None:
    """Overwrite every saved ``ContainerSpec`` to name an arbitrary constructor.

    The saved format keeps container specs as typed-JSON records in the record
    tables (and the runnable descriptor when it carries one). Each rewritten file
    is re-listed in the manifest with its new digest, so the tampered spec reaches
    reconstruction instead of stopping at the file-table check.
    """

    cell = _tampered_spec_cell(
        type_module=type_module, type_qualname=type_qualname, sentinel=sentinel
    )
    _rewrite_container_spec_cells(bundle, cell)


# --------------------------------------------------------------------------- #
# FINDING 1 -- end-to-end RCE denial (the documented tl.load() + trace.run() flow).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("type_module", "type_qualname"),
    [("subprocess", "Popen"), ("os", "system")],
)
def test_tampered_container_type_denies_code_execution(
    type_module: str, type_qualname: str, tmp_path: Path
) -> None:
    """A crafted bundle naming a non-container constructor never executes code."""

    sentinel = tmp_path / "rce_proof.txt"
    bundle = tmp_path / "rce.tlspec"
    tl.trace(_TupleModel(), torch.randn(1, 4), capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    _tamper_output_container_type(
        bundle, type_module=type_module, type_qualname=type_qualname, sentinel=sentinel
    )

    loaded = tl.load(bundle)
    with pytest.raises(RunPreconditionError):
        loaded.run(inputs=torch.randn(1, 4))
    # The command must NEVER have run -- no sentinel, no side effect.
    assert not sentinel.exists()


def _rewrite_container_spec_cells(bundle: Path, cell: Any) -> None:
    """Replace every saved ``ContainerSpec`` cell with ``cell``, re-digesting each file."""

    total = 0
    for path in sorted(bundle.rglob("*.json")):
        relpath = path.relative_to(bundle).as_posix()
        if relpath == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        replaced = _replace_container_specs(payload, cell)
        if replaced:
            path.write_text(
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
            )
            redigest(bundle, relpath)
            total += replaced
    assert total, "the saved bundle carries no ContainerSpec to tamper"


def _chain_text(exc: BaseException) -> str:
    """Return the messages of ``exc`` and every exception it was raised from."""

    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(parts) < 16:
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def test_leaf_shaped_tampered_mapping_type_is_stopped_by_the_admissibility_gate(
    tmp_path: Path,
) -> None:
    """A tamper with exactly the real leaf count is stopped by the admissibility gate alone.

    The case above is also stopped by the leaf-count check, so it stays green when the
    gate is removed. Here the tampered spec declares one leaf slot per real output leaf,
    the leaf-count check passes, and the refusal must name the disallowed type.
    """

    import subprocess  # noqa: F401  (the resolver only reads modules already loaded)

    sentinel = tmp_path / "rce_proof_mapping.txt"
    bundle = tmp_path / "rce_mapping.tlspec"
    tl.trace(_DictModel(), torch.randn(1, 4), capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    _rewrite_container_spec_cells(bundle, _tampered_mapping_spec_cell(sentinel=sentinel))

    loaded = tl.load(bundle)
    with pytest.raises(RunPreconditionError) as caught:
        loaded.run(inputs=torch.randn(1, 4))
    chain = _chain_text(caught.value)
    assert "disallowed type subprocess.Popen" in chain
    assert "leaves supplied" not in chain
    assert not sentinel.exists()


# --------------------------------------------------------------------------- #
# FINDING 1 -- unit-level default-deny of the container-type resolver.
# --------------------------------------------------------------------------- #


def test_container_tripwire_is_a_registered_typed_refusal() -> None:
    """The default-deny tripwire is public, typed, and structured (R64-3).

    It escapes users raw from the documented ``Op.multi_output_type``
    property, so it must resolve from ``torchlens.errors``, carry the
    contract code ``container_spec_inadmissible`` plus a remedy, and keep
    its ``ValueError`` lineage for the runnable run wrapper. Pickle and
    deepcopy round-trips keep the structured payload (the R64 pickle-family
    defect class).
    """

    from torchlens import errors as tl_errors

    assert tl_errors.ContainerReconstructionError is ContainerReconstructionError

    spec = ContainerSpec(
        kind="namedtuple",
        fields=("args",),
        type_module="subprocess",
        type_qualname="Popen",
    )
    with pytest.raises(tl_errors.ContainerReconstructionError) as captured:
        _resolve_container_type(spec)
    assert isinstance(captured.value, ValueError)
    assert captured.value.fields["code"] == "container_spec_inadmissible"
    assert captured.value.fields["remedy"]
    for clone in (pickle.loads(pickle.dumps(captured.value)), copy.deepcopy(captured.value)):
        assert clone.fields == captured.value.fields


def test_resolver_never_imports_an_unloaded_attacker_module() -> None:
    """A container type naming an unloaded module is refused WITHOUT importing it."""

    victim = "http.client"  # stdlib, not imported by the test/torchlens import chain
    for name in list(sys.modules):
        if name == victim or name.startswith(f"{victim}."):
            del sys.modules[name]
    assert victim not in sys.modules

    spec = ContainerSpec(
        kind="namedtuple",
        fields=("x",),
        type_module=victim,
        type_qualname="HTTPConnection",
    )
    # Unloaded -> graceful ``None`` fallback (never imported to decide).
    assert _resolve_container_type(spec) is None
    assert victim not in sys.modules


def test_rebuild_dict_subtype_denies_disallowed_mapping_type() -> None:
    """The dict-subtype vector (``Popen(items)``) is refused too."""

    import subprocess  # noqa: F401

    spec = ContainerSpec(
        kind="dict",
        keys=("a",),
        type_module="subprocess",
        type_qualname="Popen",
    )
    with pytest.raises(ContainerReconstructionError):
        rebuild_container_from_spec(spec, [1])


# --------------------------------------------------------------------------- #
# FINDING 1 -- legit output containers still reconstruct + run + VERIFY.
# --------------------------------------------------------------------------- #


def test_legit_namedtuple_output_still_round_trips(tmp_path: Path) -> None:
    """A genuine namedtuple output reconstructs to its exact type through save/load/run."""

    x = torch.tensor([1.0, 2.0])
    bundle = tmp_path / "namedtuple.tlspec"
    tl.trace(_NamedTupleModel(), x, capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x)

    assert type(result.output) is _Pair
    assert result.output.meta == 7
    assert torch.equal(result.output.value, x + 1)


def test_legit_ordereddict_output_still_round_trips(tmp_path: Path) -> None:
    """An OrderedDict output reconstructs to the exact mapping subtype."""

    x = torch.tensor([1.0, 2.0])
    bundle = tmp_path / "ordered.tlspec"
    tl.trace(_OrderedDictModel(), x, capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x)

    assert type(result.output) is OrderedDict
    assert list(result.output.keys()) == ["value", "meta"]
    assert result.output["meta"] == 3


# --------------------------------------------------------------------------- #
# FINDING 2 -- member-path traversal guard.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("evil_path", ["/etc/passwd", "../../outside.tlspec"])
def test_bundle_loader_rejects_escaping_member_path(evil_path: str, tmp_path: Path) -> None:
    """A crafted ``bundle.json`` naming an escaping member path is rejected before load."""

    import json

    from torchlens._io.bundle_container import _load_unified_bundle_directory

    bundle_path = tmp_path / "bundle.tlspec"
    bundle_path.mkdir()
    bundle_json = bundle_path / "bundle.json"
    bundle_json.write_text(
        json.dumps({"members": [{"name": "a", "path": evil_path}], "baseline_name": None})
    )

    # The member-path guard must reject the traversal BEFORE any member is loaded,
    # so this never reaches (or reconstructs from) a path outside the bundle root.
    with pytest.raises(ArtifactError):
        _load_unified_bundle_directory(bundle_path, bundle_json)


# --------------------------------------------------------------------------- #
# Prior gate denials remain intact (defense-in-depth parity smoke).
# --------------------------------------------------------------------------- #
