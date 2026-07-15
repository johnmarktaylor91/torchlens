"""Round-24 security regression: construction gadget via output-container reconstruction.

FINDING (R24-SEC1, MEDIUM -- construction gadget): the portable output ``ContainerSpec``
(torchlens-owned, so the safe unpickler admits it) carries ``type_module`` /
``type_qualname`` / ``fields`` / ``keys`` as attacker-influenceable data. The r10 hardening
tightened the ``namedtuple`` (inert ``tuple`` subclass with ``_fields``) and ``dict``
(2-type allowlist) branches, but the ``dataclass`` and ``hf_model_output`` branches were
admitted STRUCTURALLY (``dataclasses.is_dataclass`` / ``*ModelOutput``) and then rebuilt via
``container_type(**attacker_fields)`` -- invoking the resolved type's
``__init__`` / ``__post_init__``. On ``tl.load()`` + ``trace.run()``, a crafted bundle naming
ANY loaded dataclass / ModelOutput with a side-effecting construction therefore turned it
into a construction gadget with attacker-controlled kwargs.

The fix (``torchlens/ir/container.py``) reconstructs these two kinds WITHOUT invoking any
attacker constructor code: ``cls.__new__(cls)`` (verified to be an inert builtin allocator)
followed by inert per-field ``object.__setattr__`` (dataclass) or builtin-``dict`` mapping
population + ``object.__setattr__`` (ModelOutput). No ``__init__`` / ``__post_init__`` /
custom ``__new__`` runs. This reproduces the EXACT captured output (more faithful than
re-running ``__post_init__``) while executing no attacker code. A resolved type that
overrides ``__new__`` -- or a ``*ModelOutput`` that is not an ``OrderedDict`` subclass --
falls back to a plain namespace / mapping rather than being constructed.
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
    _resolve_container_type,
    rebuild_container_from_spec,
)
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# Module-scope types resolvable by (module, qualname) from ``sys.modules``.
# --------------------------------------------------------------------------- #

_GADGET_SENTINEL: Path | None = None


@dataclasses.dataclass
class _SideEffectingDataclass:
    """Stands in for any loaded dataclass whose construction has a side effect."""

    a: object = None

    def __post_init__(self) -> None:
        # An attacker-controlled construction would fire this. Reconstruction MUST NOT.
        if _GADGET_SENTINEL is not None:
            _GADGET_SENTINEL.write_text("post_init-ran")


@dataclasses.dataclass(frozen=True)
class _PlainOut:
    """A legit user dataclass model output (frozen, to exercise object.__setattr__)."""

    logits: Any = None
    tag: int = 0


class _ModelOutputBase(OrderedDict):
    """Minimal faithful mimic of ``transformers.utils.ModelOutput`` semantics."""

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            if value is not None:
                self[field.name] = value

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)


@dataclasses.dataclass
class _DemoModelOutput(_ModelOutputBase):
    """A legit HuggingFace-style ``ModelOutput`` model output."""

    logits: Any = None
    hidden: Any = None


_EVIL_POST_INIT_CALLS: list[str] = []


@dataclasses.dataclass
class _SideEffectingModelOutput(_ModelOutputBase):
    """A ``*ModelOutput`` whose ``__post_init__`` side-effects (must NEVER run on rebuild)."""

    a: Any = None

    def __post_init__(self) -> None:  # pragma: no cover - must never run
        _EVIL_POST_INIT_CALLS.append("post_init")
        super().__post_init__()


class _EvilNewList:
    """Records whether its overridden ``__new__`` ran (it must never be allocated)."""

    ran = False


@dataclasses.dataclass
class _EvilNewDataclass:
    """A dataclass that OVERRIDES ``__new__`` (must be refused, never allocated)."""

    a: object = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must never run
        _EvilNewList.ran = True
        return object.__new__(cls)


class _DataclassModel(nn.Module):
    """Return a legit user dataclass output."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        y = self.lin(x)
        return _PlainOut(logits=y, tag=5)


class _RunModelOutput(dict[str, Any]):
    """Non-dataclass HuggingFace-style ``ModelOutput`` (matches the supported run surface).

    A dataclass-decorated ModelOutput is classified ``dataclass`` (not ``hf_model_output``) by
    the runtime structure witness -- a pre-existing capture/witness ordering quirk unrelated to
    this fix -- so the supported end-to-end ModelOutput surface is a plain ``dict``/``OrderedDict``
    subclass, exactly like ``transformers`` outputs are consumed here. This ``dict`` subclass
    exercises the ``dict.__setitem__`` inert-population branch (the ``OrderedDict`` branch is
    covered by ``_DemoModelOutput`` at unit level).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _ModelOutputModel(nn.Module):
    """Return a legit HuggingFace-style ModelOutput output."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        y = self.lin(x)
        return _RunModelOutput(logits=y, hidden=y * 2)


# --------------------------------------------------------------------------- #
# The gadget is now INERT (no attacker __post_init__ runs), at unit level.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_dataclass_reconstruction_does_not_invoke_post_init(tmp_path: Path) -> None:
    """Rebuilding a dataclass output does NOT run its ``__post_init__`` side effect."""

    global _GADGET_SENTINEL
    sentinel = tmp_path / "dataclass_gadget.txt"
    _GADGET_SENTINEL = sentinel
    try:
        spec = ContainerSpec(
            kind="dataclass",
            fields=("a",),
            type_module=_SideEffectingDataclass.__module__,
            type_qualname=_SideEffectingDataclass.__qualname__,
        )
        rebuilt = rebuild_container_from_spec(spec, [1234])
    finally:
        _GADGET_SENTINEL = None

    # Reconstructed the exact captured output WITHOUT firing the constructor side effect.
    assert not sentinel.exists()
    assert type(rebuilt) is _SideEffectingDataclass
    assert rebuilt.a == 1234


@pytest.mark.smoke
def test_hf_model_output_reconstruction_does_not_invoke_post_init() -> None:
    """Rebuilding a ModelOutput output does NOT run its ``__post_init__`` side effect."""

    _EVIL_POST_INIT_CALLS.clear()
    spec = ContainerSpec(
        kind="hf_model_output",
        length=1,
        keys=("a",),
        type_module=_SideEffectingModelOutput.__module__,
        type_qualname=_SideEffectingModelOutput.__qualname__,
    )
    rebuilt = rebuild_container_from_spec(spec, [7])

    assert _EVIL_POST_INIT_CALLS == []  # __post_init__ never invoked
    assert type(rebuilt) is _SideEffectingModelOutput
    assert rebuilt["a"] == 7
    assert rebuilt.a == 7
    assert list(rebuilt.keys()) == ["a"]


def test_dataclass_with_overridden_new_falls_back_without_allocation() -> None:
    """A dataclass whose ``__new__`` is overridden is refused; the custom ``__new__`` never runs."""

    _EvilNewList.ran = False
    spec = ContainerSpec(
        kind="dataclass",
        fields=("a",),
        type_module=_EvilNewDataclass.__module__,
        type_qualname=_EvilNewDataclass.__qualname__,
    )
    rebuilt = rebuild_container_from_spec(spec, [99])

    # Falls back to a plain field namespace; the attacker ``__new__`` never executed.
    assert _EvilNewList.ran is False
    assert rebuilt == {"a": 99}


# --------------------------------------------------------------------------- #
# Legit dataclass / ModelOutput outputs still reconstruct faithfully.
# --------------------------------------------------------------------------- #


def test_dataclass_resolver_and_unit_reconstruction_faithful() -> None:
    """A legit user dataclass reconstructs to an equal instance without invoking __init__."""

    spec = ContainerSpec(
        kind="dataclass",
        fields=("logits", "tag"),
        type_module=_PlainOut.__module__,
        type_qualname=_PlainOut.__qualname__,
    )
    assert _resolve_container_type(spec) is _PlainOut
    rebuilt = rebuild_container_from_spec(spec, [123, 5])
    assert type(rebuilt) is _PlainOut
    assert rebuilt == _PlainOut(logits=123, tag=5)


def test_model_output_unit_reconstruction_faithful() -> None:
    """A legit ModelOutput reconstructs with faithful mapping view and attributes."""

    reference = _DemoModelOutput(logits=111, hidden=222)
    spec = ContainerSpec(
        kind="hf_model_output",
        length=2,
        keys=("logits", "hidden"),
        type_module=_DemoModelOutput.__module__,
        type_qualname=_DemoModelOutput.__qualname__,
    )
    rebuilt = rebuild_container_from_spec(spec, [111, 222])
    assert type(rebuilt) is _DemoModelOutput
    assert list(rebuilt.keys()) == ["logits", "hidden"]
    assert rebuilt.logits == 111 and rebuilt.hidden == 222
    assert rebuilt["logits"] == 111 and rebuilt["hidden"] == 222
    assert tuple(rebuilt.values()) == (111, 222)
    assert rebuilt == reference


def test_legit_dataclass_output_round_trips(tmp_path: Path) -> None:
    """A model returning a user dataclass round-trips through save/load/run faithfully."""

    x = torch.randn(1, 4)
    bundle = tmp_path / "dataclass.tlspec"
    tl.trace(_DataclassModel(), x, capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x)

    assert type(result.output) is _PlainOut
    assert result.output.tag == 5
    assert isinstance(result.output.logits, torch.Tensor)


def test_legit_model_output_round_trips(tmp_path: Path) -> None:
    """A model returning a HuggingFace-style ModelOutput round-trips faithfully."""

    x = torch.randn(1, 4)
    bundle = tmp_path / "modeloutput.tlspec"
    tl.trace(_ModelOutputModel(), x, capture=_CAP).save(
        bundle, level="runnable", include_weights=True
    )
    result = tl.load(bundle).run(inputs=x)

    assert type(result.output) is _RunModelOutput
    assert list(result.output.keys()) == ["logits", "hidden"]
    assert isinstance(result.output["logits"], torch.Tensor)
    assert isinstance(result.output.hidden, torch.Tensor)
    # Attribute and mapping views agree (faithful ModelOutput semantics).
    assert result.output.logits is result.output["logits"]
