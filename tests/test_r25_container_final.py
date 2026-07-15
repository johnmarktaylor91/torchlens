"""Round-25 container-reconstruction regression: SEC1 + F1/C1 + SEC2.

Three findings in the r24 non-invoking output-container rebuild
(``torchlens/ir/container.py``):

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
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.container import (
    ContainerSpec,
    _container_type_is_admissible,
    reconstruction_is_lossy,
    rebuild_container_from_spec,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAP = CaptureOptions(
    intervention_ready=True,
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


def test_sec1_model_output_descriptor_key_is_inert_and_refused() -> None:
    """A captured key that is a data descriptor is refused; its ``__set__`` never fires."""

    _DESCRIPTOR_SENTINEL.clear()
    spec = ContainerSpec(
        kind="hf_model_output",
        length=1,
        keys=("logits",),
        type_module=_DescriptorKeyModelOutput.__module__,
        type_qualname=_DescriptorKeyModelOutput.__qualname__,
    )
    rebuilt = rebuild_container_from_spec(spec, [7])

    assert _DESCRIPTOR_SENTINEL == []  # descriptor __set__ NEVER executed
    assert rebuilt == {"logits": 7}


def test_sec1_frozen_dataclass_populated_via_instance_dict() -> None:
    """A FROZEN dataclass rebuilds by writing directly into ``obj.__dict__``.

    Direct ``obj.__dict__`` population (not ``object.__setattr__``) is what bypasses both a
    frozen dataclass's ``__setattr__`` and any data descriptor. A faithful frozen rebuild is
    the observable proof the inert ``__dict__`` write path is used.
    """

    spec = ContainerSpec(
        kind="dataclass",
        fields=("logits", "tag"),
        type_module=_FrozenOut.__module__,
        type_qualname=_FrozenOut.__qualname__,
    )
    rebuilt = rebuild_container_from_spec(spec, [123, 5])

    assert type(rebuilt) is _FrozenOut
    assert rebuilt == _FrozenOut(logits=123, tag=5)
    assert vars(rebuilt) == {"logits": 123, "tag": 5}


def test_sec1_model_output_alias_and_mapping_both_populated() -> None:
    """A ModelOutput rebuild sets the mapping entry AND the ``__dict__`` attribute alias."""

    spec = ContainerSpec(
        kind="hf_model_output",
        length=1,
        keys=("logits",),
        type_module=_PureModelOutput.__module__,
        type_qualname=_PureModelOutput.__qualname__,
    )
    rebuilt = rebuild_container_from_spec(spec, [42])

    assert type(rebuilt) is _PureModelOutput
    assert rebuilt["logits"] == 42  # mapping entry via base dict.__setitem__
    assert vars(rebuilt) == {"logits": 42}  # attribute alias via obj.__dict__ write


# --------------------------------------------------------------------------- #
# SEC2: structural ``issubclass(tuple)`` check applies regardless of module string.
# --------------------------------------------------------------------------- #


class _NotATuple:
    """A non-tuple type an attacker might name under ``torch.return_types``."""


def test_sec2_non_structseq_naming_torch_return_types_is_refused() -> None:
    """A non-tuple type is refused structurally even when the spec claims torch.return_types."""

    spec = ContainerSpec(
        kind="namedtuple",
        fields=("a",),
        type_module="torch.return_types",
        type_qualname="x",
    )
    assert _container_type_is_admissible(_NotATuple, spec) is False


def test_sec2_real_structseq_still_admissible() -> None:
    """A genuine torch structseq (tuple subclass with n_fields) stays admissible."""

    structseq = torch.max(torch.randn(3, 3), dim=0)
    spec = ContainerSpec(
        kind="namedtuple",
        fields=("values", "indices"),
        type_module="torch.return_types",
        type_qualname="max",
    )
    assert _container_type_is_admissible(type(structseq), spec) is True


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


def test_reconstruction_is_lossy_ignores_none_extras() -> None:
    """A None-valued extra attribute (standard ModelOutput unset field) is NOT lossy."""

    mo = _PureModelOutput(logits=torch.zeros(2))
    mo.attentions = None  # a standard unset ModelOutput field kept as a None attribute
    assert reconstruction_is_lossy(mo, ("logits",)) is False


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


@pytest.mark.smoke
def test_derived_attr_dataclass_run_is_unverifiable(tmp_path: Path) -> None:
    """A dataclass with computed non-field state runs UNVERIFIABLE, never VERIFIED."""

    x = torch.tensor([2.0, 4.0, -1.0, 3.0])
    bundle = tmp_path / "derived_dc.tlspec"
    tl.trace(_DerivedAttrDataclassModel(), x.clone(), capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x.clone(), seed=0, on_divergence="return_diverged")

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert type(result.output) is _DerivedAttrDataclassOut
    # The dropped computed attribute is genuinely absent (why the run is UNVERIFIABLE).
    assert not hasattr(result.output, "extra_note")


@pytest.mark.smoke
def test_derived_attr_model_output_run_is_unverifiable(tmp_path: Path) -> None:
    """A ModelOutput with a computed non-mapping attribute runs UNVERIFIABLE."""

    x = torch.randn(1, 4)
    bundle = tmp_path / "derived_mo.tlspec"
    tl.trace(_DerivedAttrModelOutputModel(), x, capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x, seed=0, on_divergence="return_diverged")

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert type(result.output) is _DerivedAttrModelOutput
    assert not hasattr(result.output, "double")


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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_pure_model_output_run_is_verified_and_faithful(tmp_path: Path) -> None:
    """A pure ModelOutput output is VERIFIED with faithful mapping + attribute views."""

    x = torch.randn(1, 4)
    bundle = tmp_path / "pure_mo.tlspec"
    tl.trace(_PureModelOutputModel(), x, capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is _PureModelOutput
    assert list(result.output.keys()) == ["logits"]
    assert result.output.logits is result.output["logits"]


def test_real_transformers_model_output_not_lossy_and_faithful() -> None:
    """A real transformers ModelOutput (None fields excluded from keys) stays VERIFIED-eligible."""

    modeling_outputs = pytest.importorskip("transformers.modeling_outputs")
    output = modeling_outputs.BaseModelOutput(last_hidden_state=torch.zeros(2, 2))

    # None extra fields (hidden_states/attentions) must NOT over-trigger lossy.
    assert reconstruction_is_lossy(output, tuple(output.keys())) is False

    spec = ContainerSpec(
        kind="hf_model_output",
        length=1,
        keys=("last_hidden_state",),
        type_module=type(output).__module__,
        type_qualname=type(output).__qualname__,
    )
    rebuilt = rebuild_container_from_spec(spec, [torch.ones(2, 2)])
    assert type(rebuilt) is type(output)
    assert list(rebuilt.keys()) == ["last_hidden_state"]
    assert rebuilt.last_hidden_state is rebuilt["last_hidden_state"]


# --------------------------------------------------------------------------- #
# No regression: namedtuple / dict / OrderedDict / tuple / list unchanged.
# --------------------------------------------------------------------------- #


def test_no_regression_plain_containers_round_trip() -> None:
    """namedtuple / dict / OrderedDict / tuple / list reconstruct unchanged."""

    nt_spec = ContainerSpec(
        kind="namedtuple",
        length=2,
        fields=("x", "y"),
        type_module=_Point.__module__,
        type_qualname=_Point.__qualname__,
    )
    nt = rebuild_container_from_spec(nt_spec, [1, 2])
    assert type(nt) is _Point and nt == _Point(1, 2)

    dict_spec = ContainerSpec(kind="dict", length=2, keys=("a", "b"))
    assert rebuild_container_from_spec(dict_spec, [1, 2]) == {"a": 1, "b": 2}

    od_spec = ContainerSpec(
        kind="dict",
        length=2,
        keys=("a", "b"),
        type_module="collections",
        type_qualname="OrderedDict",
    )
    od = rebuild_container_from_spec(od_spec, [1, 2])
    assert type(od) is OrderedDict and list(od.items()) == [("a", 1), ("b", 2)]

    tup_spec = ContainerSpec(kind="tuple", length=2)
    assert rebuild_container_from_spec(tup_spec, [1, 2]) == (1, 2)

    list_spec = ContainerSpec(kind="list", length=2)
    assert rebuild_container_from_spec(list_spec, [1, 2]) == [1, 2]


class _Point(tuple):  # stand-in namedtuple-like with _fields
    _fields = ("x", "y")

    def __new__(cls, x: Any, y: Any) -> "_Point":
        return super().__new__(cls, (x, y))
