"""r55 CLASS 2 immunizer -- no attacker instance state is read-then-called on load.

r54 ``sec_3`` (MED, latent RCE amplifier): ``metadata.pkl`` deserializes into the
``Trace`` state dict; an attacker who plants a top-level ``_internal_set`` key had
it read off the INSTANCE and invoked by ``rehydrate._assign_rehydrated_field`` with
attacker-controlled arguments. Two structural layers close the class:

* Layer 1 (primary): ``_assign_rehydrated_field`` resolves the protocol setter off
  the CLASS via ``inspect.getattr_static`` -- a planted instance key can never
  substitute for the real slotted-class method.
* Layer 2 (defense in depth): a NARROW ``refuse_callable_shadowing_state_keys``
  filter runs before every portable ``__dict__.update(state)`` and refuses any key
  shadowing a class-owned plain method (``save``/``run``/``draw``/``to_trace``/...),
  with zero legitimate collisions (the real fields are ``@property``/slot
  descriptors, deliberately NOT caught).

Plus a source-scan tripwire: no portable ``__setstate__`` updates ``__dict__``
without first calling the filter.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy
import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens._io.rehydrate import _assign_rehydrated_field
from torchlens._io.state_keys import (
    PortableStateKeyError,
    refuse_callable_shadowing_state_keys,
)
from torchlens.options import CaptureOptions


class _M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _build(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    cap = CaptureOptions(replayable=True, capture_container_structure=True, cache=False)
    trace = tl.trace(_M().eval(), torch.randn(2, 4), capture=cap)
    bundle = tmp_path / "base.tlspec"
    trace.save(str(bundle), level="runnable", include_weights=True)
    return bundle


# --------------------------------------------------------------------------- #
# (a) layer 1 -- planted instance _internal_set is NEVER read-then-called       #
# --------------------------------------------------------------------------- #


def test_assign_rehydrated_field_ignores_instance_setter() -> None:
    """``_assign_rehydrated_field`` never binds a setter off the attacker instance."""

    calls: list[tuple[str, object]] = []

    class _Victim:
        pass

    victim = _Victim()
    # Plant a callable in the instance namespace under the protocol name.
    victim._internal_set = lambda name, value: calls.append((name, value))  # type: ignore[attr-defined]
    _assign_rehydrated_field(victim, "field_a", 123)

    assert calls == [], "instance-planted _internal_set was invoked"
    assert victim.field_a == 123  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# (b) layer 2 -- narrow method-shadow filter (0 legit collisions, catches methods)
# --------------------------------------------------------------------------- #


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "method_name",
    [
        "save",
    ],
)
def test_state_key_shadowing_trace_method_is_refused(method_name: str) -> None:
    """A state key shadowing a real ``Trace`` method is refused, typed."""

    assert callable(getattr(Trace, method_name, None)), f"{method_name} must be a real method"
    with pytest.raises(PortableStateKeyError):
        refuse_callable_shadowing_state_keys(Trace, {method_name: object()})


def test_shadow_key_refused_on_session_pickle_restore() -> None:
    """Layer 2 stays armed OUTSIDE the governed window: a method-shadow key in
    plain session ``__setstate__`` state refuses through the narrow filter."""

    torch.manual_seed(0)
    trace = tl.trace(_M().eval(), torch.randn(2, 4))
    state = trace.__getstate__()
    state["save"] = numpy.dtype
    with pytest.raises(PortableStateKeyError):
        Trace.__new__(Trace).__setstate__(state)


# --------------------------------------------------------------------------- #
# (c) over-trigger guard + source-scan tripwire                               #
# --------------------------------------------------------------------------- #


def test_every_portable_setstate_guards_dict_update() -> None:
    """No portable ``__setstate__`` updates ``__dict__`` without the shadow filter.

    AST source-scan tripwire: a NEW portable ``__setstate__`` that calls
    ``self.__dict__.update(state)`` without first calling
    ``refuse_callable_shadowing_state_keys`` re-opens the sec_3 enabler class.
    """

    data_classes_dir = Path(tl.__file__).parent / "data_classes"
    offenders: list[str] = []
    for source_path in sorted(data_classes_dir.glob("*.py")):
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "__setstate__"):
                continue
            body_src = ast.get_source_segment(source_path.read_text(), node) or ""
            if "__dict__.update(state)" not in body_src:
                continue
            if "refuse_callable_shadowing_state_keys" not in body_src:
                offenders.append(f"{source_path.name}::{node.name}")
    assert not offenders, (
        f"portable __setstate__ updates __dict__ without the callable-shadow filter: {offenders}"
    )
