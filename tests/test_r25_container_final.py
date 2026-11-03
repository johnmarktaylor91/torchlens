"""Round-25 container-reconstruction regression: SEC1 + F1/C1 + SEC2.

Three findings in the r24 non-invoking output-container rebuild
(``torchlens/_ir/container.py``):

* SEC1 (HIGH, security): ``object.__setattr__`` HONORS data descriptors, so populating a
  captured field/key whose name is a settable ``property`` (or any data descriptor) fired
  attacker-reachable ``__set__`` code on a plain ``tl.load()`` + ``.run()``. Fix: populate
  via ``obj.__dict__[name] = value`` (dataclass attribute + ModelOutput alias) and the BASE
  ``dict`` / ``OrderedDict`` ``__setitem__`` (mapping entry) -- never ``object.__setattr__`` --
  AND refuse (fall back) any spec whose captured name resolves to a data descriptor or whose
  type has no per-instance ``__dict__`` (``__slots__``).

* F1 / codex-C1 (CRITICAL, honesty): the rebuild dropped ``__post_init__``-created
  non-field / non-key attributes (derived from tensors) while reporting VERIFIED. Fix: at
  CAPTURE detect such lossy outputs (computed extras / ``__slots__`` / descriptor field) and
  flag ``ContainerSpec.lossy_reconstruction``; on run a lossy output is UNVERIFIABLE, never
  VERIFIED. Pure data outputs (attrs == fields == keys) stay VERIFIED.

* SEC2 (LOW, security): admissibility skipped the structural ``issubclass(tuple)`` check when
  the attacker-supplied ``type_module`` string was ``"torch.return_types"``. Fix: apply the
  structural check regardless of the module string.
"""

from __future__ import annotations

import dataclasses
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._ir.container import (
    ContainerSpec,
    _container_type_is_admissible,
    _model_output_has_foreign_init,
    _resolve_container_type,
    rebuild_container_from_spec,
    reconstruction_is_lossy,
    reconstruction_is_lossy_by_type,
)
from torchlens._runnable import PathFaithfulness
from torchlens._runnable.call_outputs import _container_spec_reconstruction_lossy
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# SEC1: data-descriptor field names must NEVER fire ``__set__`` on rebuild.
# --------------------------------------------------------------------------- #

_DESCRIPTOR_SENTINEL: list[Any] = []


@dataclasses.dataclass
class _DescriptorFieldDataclass:
    """Stands in for a loaded dataclass with a side-effecting settable ``property``."""

    endpoint: object = None

    @property  # type: ignore[no-redef]
    def endpoint(self) -> Any:  # noqa: F811 - property shadows the field name on purpose
        return getattr(self, "_endpoint", None)

    @endpoint.setter
    def endpoint(self, value: Any) -> None:
        _DESCRIPTOR_SENTINEL.append(value)  # attacker-reachable code -- must NEVER run
        self._endpoint = value


class _DescriptorKeyModelOutput(dict[str, Any]):
    """A ``dict``-subclass ModelOutput whose key name is a side-effecting data descriptor."""

    @property  # type: ignore[misc]
    def logits(self) -> Any:
        return self.get("logits")

    @logits.setter
    def logits(self, value: Any) -> None:  # pragma: no cover - must never run on rebuild
        _DESCRIPTOR_SENTINEL.append(value)
        dict.__setitem__(self, "logits", value)


# Kept: sole branch coverage of _ir/container.py.
def test_sec1_dataclass_descriptor_field_is_inert_and_refused() -> None:
    """A captured field that is a data descriptor is refused; its ``__set__`` never fires."""

    _DESCRIPTOR_SENTINEL.clear()
    spec = ContainerSpec(
        kind="dataclass",
        fields=("endpoint",),
        type_module=_DescriptorFieldDataclass.__module__,
        type_qualname=_DescriptorFieldDataclass.__qualname__,
    )
    rebuilt = rebuild_container_from_spec(spec, ["attacker://payload"])

    assert _DESCRIPTOR_SENTINEL == []  # descriptor __set__ NEVER executed
    # Falls back to a plain field namespace rather than firing the descriptor.
    assert rebuilt == {"endpoint": "attacker://payload"}


# --------------------------------------------------------------------------- #
# SEC2: structural ``issubclass(tuple)`` check applies regardless of module string.
# --------------------------------------------------------------------------- #


class _NotATuple:
    """A non-tuple type an attacker might name under ``torch.return_types``."""


# Kept: sole branch coverage of _ir/container.py.
def test_sec2_non_structseq_naming_torch_return_types_is_refused() -> None:
    """A non-tuple type is refused structurally even when the spec claims torch.return_types."""

    spec = ContainerSpec(
        kind="namedtuple",
        fields=("a",),
        type_module="torch.return_types",
        type_qualname="x",
    )
    assert _container_type_is_admissible(_NotATuple, spec) is False


# --------------------------------------------------------------------------- #
# F1 / C1: computed non-field/non-key state -> lossy -> UNVERIFIABLE.
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _PureDataclassOut:
    """A pure-data user dataclass output (attrs == declared fields)."""

    logits: Any = None
    tag: int = 0


@dataclasses.dataclass(frozen=True)
class _FrozenOut:
    """A frozen user dataclass output (rebuild must bypass its ``__setattr__``)."""

    logits: Any = None
    tag: int = 0


@dataclasses.dataclass
class _DerivedAttrDataclassOut:
    """A dataclass whose ``__post_init__`` sets a computed non-field attribute."""

    a: Any = None

    def __post_init__(self) -> None:
        self.extra_note = float(self.a.sum()) if isinstance(self.a, torch.Tensor) else 1.0


class _PureModelOutput(dict[str, Any]):
    """A pure-data ModelOutput (instance attrs == mapping keys, no computed extras)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DerivedAttrModelOutput(dict[str, Any]):
    """A ModelOutput with a computed non-mapping attribute (``double``)."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__({"logits": logits})
        self.logits = logits
        self.double = logits * 2.0


# Kept: the catch of its regression fault.
def test_reconstruction_is_lossy_detects_computed_extras() -> None:
    """Computed non-field/non-key attributes make reconstruction lossy; pure data does not."""

    pure_dc = _PureDataclassOut(logits=torch.zeros(2), tag=3)
    assert reconstruction_is_lossy(pure_dc, ("logits", "tag")) is False

    derived_dc = _DerivedAttrDataclassOut(a=torch.tensor([1.0, 2.0]))
    assert reconstruction_is_lossy(derived_dc, ("a",)) is True

    pure_mo = _PureModelOutput(logits=torch.zeros(2))
    assert reconstruction_is_lossy(pure_mo, ("logits",)) is False

    derived_mo = _DerivedAttrModelOutput(torch.tensor([1.0, 2.0]))
    assert reconstruction_is_lossy(derived_mo, ("logits",)) is True


class _DerivedAttrDataclassModel(nn.Module):
    """Return a dataclass output carrying a __post_init__ computed non-field attribute."""

    def forward(self, x: torch.Tensor) -> Any:
        return _DerivedAttrDataclassOut(a=x * 2.0)


class _DerivedAttrModelOutputModel(nn.Module):
    """Return a ModelOutput carrying a computed non-mapping attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        return _DerivedAttrModelOutput(self.lin(x))


# --------------------------------------------------------------------------- #
# Pure outputs stay VERIFIED + reconstruct faithfully (no over-trigger).
# --------------------------------------------------------------------------- #


class _PureDataclassModel(nn.Module):
    """Return a pure user dataclass output (no computed extras)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        return _PureDataclassOut(logits=self.lin(x), tag=5)


class _PureModelOutputModel(nn.Module):
    """Return a pure ModelOutput output (attrs == keys)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        return _PureModelOutput(logits=self.lin(x))


# Kept: a sole mutation kill.
def test_pure_dataclass_run_is_verified_and_faithful(tmp_path: Path) -> None:
    """A pure user dataclass output is VERIFIED and reconstructs to an equal instance."""

    x = torch.randn(1, 4)
    bundle = tmp_path / "pure_dc.tlspec"
    model = _PureDataclassModel().eval()
    tl.trace(model, x, capture=_CAP).save(bundle, level="runnable", include_weights=True)
    result = tl.load(bundle).run(inputs=x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is _PureDataclassOut
    assert result.output.tag == 5
    with torch.no_grad():
        expected = model(x)
    assert torch.equal(result.output.logits, expected.logits)


# Kept: sole branch coverage of _ir/_container_type_probes.py, _ir/container.py, _runnable/path_faithfulness.py and more.
def test_custom_init_model_output_run_is_unverifiable_but_reconstructs(tmp_path: Path) -> None:
    """r27-B2: a custom-``__init__`` ModelOutput is honest fail-closed UNVERIFIABLE.

    ``_PureModelOutput`` defines its OWN ``__init__`` (a custom, non-``transformers`` constructor).
    Whether that ``__init__`` computes a dropped tensor-derived extra attribute is NOT observable
    from the type without INVOKING it (the construction-gadget RCE surface), so the load-time
    recompute conservatively marks the node lossy -> UNVERIFIABLE. The output still RECONSTRUCTS
    faithfully (correct type + mapping/attribute views); only the honesty verdict is downgraded.
    """

    x = torch.randn(1, 4)
    bundle = tmp_path / "custom_init_mo.tlspec"
    tl.trace(_PureModelOutputModel(), x, capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x, on_divergence="return")

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert type(result.output) is _PureModelOutput
    assert list(result.output.keys()) == ["logits"]
    assert result.output.logits is result.output["logits"]


# --------------------------------------------------------------------------- #
# No regression: namedtuple / dict / OrderedDict / tuple / list unchanged.
# --------------------------------------------------------------------------- #


class _Point(tuple):  # stand-in namedtuple-like with _fields
    _fields = ("x", "y")

    def __new__(cls, x: Any, y: Any) -> _Point:
        return super().__new__(cls, (x, y))


# --------------------------------------------------------------------------- #
# r26-B1 (HIGH, RCE): the namedtuple branch must NEVER invoke the resolved type's
# ``__new__`` -- a ``tuple`` subclass with ``_fields`` whose ``__new__`` executes code
# is a load/run construction gadget. Reconstruction goes through the inert
# ``tuple.__new__`` and the weaponized ``__new__`` never fires.
# --------------------------------------------------------------------------- #


_NAMEDTUPLE_NEW_SENTINEL: list[Any] = []


class _WeaponizedNamedTupleLike(tuple):
    """A ``tuple`` subclass with ``_fields`` whose ``__new__`` runs code (RCE gadget).

    Passes namedtuple admissibility (``issubclass(tuple)`` + ``_fields``). The r26 fix rebuilds
    it via the inert ``tuple.__new__`` and MUST NEVER invoke this ``__new__``.
    """

    _fields = ("a", "b")

    def __new__(cls, *args: Any) -> _WeaponizedNamedTupleLike:  # pragma: no cover
        _NAMEDTUPLE_NEW_SENTINEL.append("weaponized __new__ ran")
        return tuple.__new__(cls, args)


class _PairNamedTuple(NamedTuple):
    """A genuine compiler-generated namedtuple output."""

    a: int
    b: int


# --------------------------------------------------------------------------- #
# r26-B2 (LOW-MED, honesty): ``lossy_reconstruction`` is attacker-controlled in a
# forged bundle. Lossiness is RECOMPUTED independently from the resolved type at load,
# so a forged ``lossy_reconstruction=False`` cannot force a false VERIFIED.
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(slots=True)
class _SlotsDataclassOut:
    """A ``__slots__`` dataclass output (no instance ``__dict__`` -> lossy rebuild)."""

    a: Any = None


# Kept: sole branch coverage of _ir/container.py.
def test_r26_forged_lossy_flag_on_descriptor_field_is_recomputed_lossy() -> None:
    """A forged flag on a data-descriptor-field dataclass is recomputed lossy at load."""

    forged = ContainerSpec(
        kind="dataclass",
        length=1,
        fields=("endpoint",),
        type_module=_DescriptorFieldDataclass.__module__,
        type_qualname=_DescriptorFieldDataclass.__qualname__,
        lossy_reconstruction=False,
    )
    assert _container_spec_reconstruction_lossy(forged) is True


# Kept: sole branch coverage of _ir/container.py.
def test_r26_forged_lossy_flag_on_slots_type_is_recomputed_lossy() -> None:
    """A forged flag on a ``__slots__`` output type is recomputed lossy at load."""

    forged = ContainerSpec(
        kind="dataclass",
        length=1,
        fields=("a",),
        type_module=_SlotsDataclassOut.__module__,
        type_qualname=_SlotsDataclassOut.__qualname__,
        lossy_reconstruction=False,
    )
    # A __slots__ type has no instance __dict__ -> lossy regardless of the forged flag.
    assert reconstruction_is_lossy_by_type(_SlotsDataclassOut, ("a",), "dataclass") is True
    assert _container_spec_reconstruction_lossy(forged) is True


class _NoInitModelOutput(dict[str, Any]):
    """A mapping ModelOutput with NO custom ``__init__`` (mapping-only, no computed extras)."""


# Kept: sole branch coverage of _ir/container.py.
def test_r26_forged_lossy_flag_on_unloaded_type_is_lossy() -> None:
    """A dataclass spec whose type is not loaded rebuilds to a plain namespace -> lossy."""

    forged = ContainerSpec(
        kind="dataclass",
        length=1,
        fields=("a",),
        type_module="tests._r26_never_imported_module",
        type_qualname="Missing",
        lossy_reconstruction=False,
    )
    assert _container_spec_reconstruction_lossy(forged) is True


# --------------------------------------------------------------------------- #
# r27-B-NEW (HIGH, RCE): the dotted ``type_qualname`` walk must resolve STATICALLY.
# Plain ``getattr`` fired ``torch.__getattr__`` and lazily imported ~48 torch
# submodules (running their top-level code) for ``type_module="torch"`` +
# ``type_qualname="onnx"`` on a benign reconstruct / honesty gate. The fix uses
# ``inspect.getattr_static`` (no ``__getattr__`` / descriptor invocation), so an
# unmaterialized lazy submodule fails closed to the plain-container fallback.
# --------------------------------------------------------------------------- #


def _torch_submodules() -> set[str]:
    """Return the currently-loaded ``torch.*`` submodule names."""

    return {name for name in sys.modules if name == "torch" or name.startswith("torch.")}


# Kept: sole branch coverage of _ir/container.py.
def test_r27_lazy_import_gadget_rebuild_and_lossy_gate_import_nothing() -> None:
    """Both rebuild and the honesty (lossy) gate stay import-free for the ``torch.onnx`` ref."""

    spec = ContainerSpec(
        kind="hf_model_output",
        length=1,
        keys=("logits",),
        type_module="torch",
        type_qualname="onnx",
    )

    before = _torch_submodules()
    # Rebuild degrades to the plain mapping shape (no admissible type resolved), no import.
    rebuilt = rebuild_container_from_spec(spec, [7])
    # The honesty gate resolves the same ref (a not-loaded/plain-substitution -> lossy), no import.
    assert _container_spec_reconstruction_lossy(spec) is True
    after = _torch_submodules()

    assert rebuilt == {"logits": 7}
    assert after == before, f"gadget imported torch submodules: {sorted(after - before)}"


# Kept: sole branch coverage of _ir/container.py.
def test_r27_deep_qualname_is_refused() -> None:
    """A pathologically deep attacker qualname is refused (falls back), never walked."""

    spec = ContainerSpec(
        kind="namedtuple",
        fields=("a",),
        type_module="torch",
        type_qualname=".".join(["x"] * 32),
    )
    assert _resolve_container_type(spec) is None


# Kept: sole branch coverage of _ir/container.py.
def test_r27_getattr_static_does_not_fire_module_pep562_getattr() -> None:
    """Static resolution never triggers a module-level ``__getattr__`` side effect."""

    module_name = "tests._r27_pep562_probe_module"
    probe = sys.modules.get(module_name)
    if probe is None:
        import types as _types

        probe = _types.ModuleType(module_name)
        fired: list[str] = []

        def _module_getattr(name: str) -> Any:  # PEP-562 module __getattr__
            fired.append(name)
            raise AttributeError(name)

        probe.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        probe._fired = fired  # type: ignore[attr-defined]
        sys.modules[module_name] = probe

    probe._fired.clear()  # type: ignore[attr-defined]
    spec = ContainerSpec(
        kind="namedtuple",
        fields=("a",),
        type_module=module_name,
        type_qualname="Anything",
    )
    resolved = _resolve_container_type(spec)

    assert resolved is None
    assert probe._fired == []  # type: ignore[attr-defined]  # __getattr__ never invoked


# --------------------------------------------------------------------------- #
# r27-B2: custom-``__init__`` ModelOutput type-observable proxy (unit level).
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _ir/_container_type_probes.py.
def test_r27_foreign_init_detection_matrix() -> None:
    """``_model_output_has_foreign_init`` trusts mapping bases; transformers trust is spec-keyed.

    r42 secB_1: a non-base field-mirroring init is trusted ONLY by RESOLUTION AUTHORITY
    (identity re-resolution from the genuine ``transformers`` package via the spec), never the
    spoofable/loose ``__module__`` string. Without a spec the check fail-closes to foreign.
    """

    class _CustomInitMO(dict[str, Any]):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(kwargs)
            self.extra = 1.0  # computed non-key attribute

    class _NoInitMO(dict[str, Any]):
        pass

    class _OrderedNoInitMO(OrderedDict):  # type: ignore[type-arg]
        pass

    assert _model_output_has_foreign_init(_CustomInitMO) is True
    assert _model_output_has_foreign_init(_NoInitMO) is False
    assert _model_output_has_foreign_init(_OrderedNoInitMO) is False
    # r42 secB_1: no spec / no resolution authority -> a non-base init fail-closes to foreign.
    modeling_outputs = pytest.importorskip("transformers.modeling_outputs")
    assert _model_output_has_foreign_init(modeling_outputs.BaseModelOutput) is True
    # WITH the genuine resolution authority (identity re-resolution from the real package) the
    # transformers field-mirroring init is trusted -> not foreign.
    genuine_spec = ContainerSpec(
        kind="hf_model_output",
        type_module="transformers.modeling_outputs",
        type_qualname="BaseModelOutput",
    )
    assert _model_output_has_foreign_init(modeling_outputs.BaseModelOutput, genuine_spec) is False
    # A SPOOFED ``__module__`` string (loose prefix / non-identity) is never trusted.
    spoof_spec = ContainerSpec(
        kind="hf_model_output",
        type_module="transformers_evil",
        type_qualname="BaseModelOutput",
    )
    assert _model_output_has_foreign_init(modeling_outputs.BaseModelOutput, spoof_spec) is True
