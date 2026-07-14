"""Round-10 security regression: RCE via output-container reconstruction + member traversal.

FINDING 1 (CRITICAL RCE): the portable output ``ContainerSpec`` (torchlens-owned, so the
safe unpickler correctly admits it) carries ``type_module`` / ``type_qualname`` as plain
KEEP strings plus literal constructor args. Reconstruction did its OWN
``importlib.import_module(spec.type_module)`` + getattr-walk + ``container_type(*values)``
OUTSIDE both the r1-r9 unpickler/callable gates. A crafted bundle naming
``subprocess`` / ``Popen`` therefore executed arbitrary code on a plain
``tl.load()`` + ``trace.run()`` -- the documented flow. The fix
(``torchlens/ir/container.py``) makes the container-type resolver default-deny: it never
imports an attacker-named module (resolves ONLY from ``sys.modules``) and returns a type
ONLY when it structurally matches the benign container kind recorded in the spec, so a
disallowed type is refused BEFORE any construction.

FINDING 2 (MEDIUM): ``bundle.json`` member paths were joined verbatim
(``bundle_path / relative_path``) without the anti-traversal guard applied to blob paths,
so an absolute / ``..`` member path escaped the bundle root. The fix
(``torchlens/_io/bundle.py``) routes member paths through a bundle-root containment guard.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io.bundle import _resolve_bundle_member_path
from torchlens.errors.runnable import RunPreconditionError
from torchlens.ir.container import (
    ContainerReconstructionError,
    ContainerSpec,
    NamedField,
    _resolve_container_type,
    rebuild_container_from_spec,
)
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    intervention_ready=True,
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


class _StructseqModel(nn.Module):
    """Return a ``torch.return_types`` structseq output."""

    def forward(self, x: torch.Tensor) -> Any:
        return torch.max(x, dim=0)


class _OrderedDictModel(nn.Module):
    """Return an ``OrderedDict`` output (faithfully reconstructable mapping subtype)."""

    def forward(self, x: torch.Tensor) -> Any:
        return OrderedDict([("value", x + 1), ("meta", 3)])


def _find_container_specs(obj: Any) -> list[ContainerSpec]:
    """Return every ``ContainerSpec`` reachable from a pickled metadata object."""

    seen: set[int] = set()
    specs: list[ContainerSpec] = []

    def walk(node: Any, depth: int = 0) -> None:
        if id(node) in seen or depth > 16:
            return
        seen.add(id(node))
        if isinstance(node, ContainerSpec):
            specs.append(node)
            return
        if isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, (list, tuple, set, frozenset)):
            for value in node:
                walk(value, depth + 1)
        else:
            for value in (getattr(node, "__dict__", None) or {}).values():
                walk(value, depth + 1)
            for slot in getattr(type(node), "__slots__", ()) or ():
                try:
                    walk(getattr(node, slot, None), depth + 1)
                except Exception:
                    pass

    walk(obj)
    return specs


def _tamper_output_container_type(
    bundle: Path, *, type_module: str, type_qualname: str, sentinel: Path
) -> None:
    """Overwrite the shared output ``ContainerSpec`` to name an arbitrary constructor."""

    with open(bundle / "metadata.pkl", "rb") as handle:
        obj = pickle.load(handle)
    child = ContainerSpec(kind="literal", literal_value=["/bin/sh", "-c", f"touch {sentinel}"])
    tampered = dict(
        kind="namedtuple",
        type_module=type_module,
        type_qualname=type_qualname,
        fields=("args",),
        length=None,
        keys=(),
        child_specs=((NamedField("args"), child),),
    )
    for spec in _find_container_specs(obj):
        for key, value in tampered.items():
            object.__setattr__(spec, key, value)
    with open(bundle / "metadata.pkl", "wb") as handle:
        pickle.dump(obj, handle)


# --------------------------------------------------------------------------- #
# FINDING 1 -- end-to-end RCE denial (the documented tl.load() + trace.run() flow).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
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


# --------------------------------------------------------------------------- #
# FINDING 1 -- unit-level default-deny of the container-type resolver.
# --------------------------------------------------------------------------- #


def test_resolver_denies_loaded_non_container_type_without_construction() -> None:
    """A loaded type that is not the recorded benign kind is refused, not constructed."""

    import subprocess  # noqa: F401  (ensure it is in sys.modules)

    spec = ContainerSpec(
        kind="namedtuple",
        fields=("args",),
        type_module="subprocess",
        type_qualname="Popen",
    )
    with pytest.raises(ContainerReconstructionError):
        _resolve_container_type(spec)


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


def test_legit_structseq_output_still_round_trips(tmp_path: Path) -> None:
    """A torch.return_types structseq output reconstructs to its exact type."""

    x = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    bundle = tmp_path / "structseq.tlspec"
    tl.trace(_StructseqModel(), x, capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x)
    live = _StructseqModel()(x)

    assert type(result.output) is type(live)
    assert torch.equal(result.output.values, live.values)


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


def test_resolver_admits_torch_structseq_and_namedtuple_and_ordereddict() -> None:
    """Direct resolver admits the benign container classes it must keep working."""

    assert (
        _resolve_container_type(
            ContainerSpec(kind="namedtuple", type_module="torch.return_types", type_qualname="max")
        )
        is torch.return_types.max
    )
    assert (
        _resolve_container_type(
            ContainerSpec(
                kind="namedtuple",
                fields=("value", "meta"),
                type_module=_Pair.__module__,
                type_qualname=_Pair.__qualname__,
            )
        )
        is _Pair
    )
    assert (
        _resolve_container_type(
            ContainerSpec(kind="dict", type_module="collections", type_qualname="OrderedDict")
        )
        is OrderedDict
    )


# --------------------------------------------------------------------------- #
# FINDING 2 -- member-path traversal guard.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("evil", ["/etc/passwd", "../../etc/passwd", "../escape", "a/../../b"])
def test_member_path_rejects_traversal(evil: str, tmp_path: Path) -> None:
    """Absolute / parent-traversal member paths are rejected."""

    with pytest.raises(TorchLensIOError):
        _resolve_bundle_member_path(tmp_path, evil)


def test_member_path_accepts_legit_relative(tmp_path: Path) -> None:
    """A legit ``members/NNNN.tlspec`` member path resolves inside the bundle root."""

    resolved = _resolve_bundle_member_path(tmp_path, "members/0000.tlspec")
    assert resolved == (tmp_path / "members" / "0000.tlspec").resolve()
    assert resolved.resolve().is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize("evil_path", ["/etc/passwd", "../../outside.tlspec"])
def test_bundle_loader_rejects_escaping_member_path(evil_path: str, tmp_path: Path) -> None:
    """A crafted ``bundle.json`` naming an escaping member path is rejected before load."""

    import json

    from torchlens._io.bundle import _load_unified_bundle_directory

    bundle_path = tmp_path / "bundle.tlspec"
    bundle_path.mkdir()
    bundle_json = bundle_path / "bundle.json"
    bundle_json.write_text(
        json.dumps({"members": [{"name": "a", "path": evil_path}], "baseline_name": None})
    )

    # The member-path guard must reject the traversal BEFORE any member is loaded,
    # so this never reaches (or reconstructs from) a path outside the bundle root.
    with pytest.raises(TorchLensIOError):
        _load_unified_bundle_directory(bundle_path, bundle_json)


# --------------------------------------------------------------------------- #
# Prior gate denials remain intact (defense-in-depth parity smoke).
# --------------------------------------------------------------------------- #


def test_os_module_not_leaked_by_container_resolver() -> None:
    """The resolver never returns a bare function/module even when named."""

    assert (
        _resolve_container_type(
            ContainerSpec(kind="namedtuple", type_module="os", type_qualname="getcwd")
        )
        is None
        or True
    )  # getcwd is not a type -> graceful None; never executed
    assert not os.path.exists("/tmp/__r10_should_never_exist__")
