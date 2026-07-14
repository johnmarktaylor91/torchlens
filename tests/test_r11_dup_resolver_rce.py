"""Round-11 security regression: duplicate container-type resolver RCE.

r10 hardened the container-type resolver in ``torchlens/ir/container.py`` into a strict
default-deny, never-import gate. But a PARALLEL resolver survived in
``torchlens/data_classes/op.py``: the public inspection property ``Op.multi_output_type``
called its own ``importlib.import_module(container_spec.type_module)`` on the
attacker-controlled, portable ``ContainerSpec`` strings (they ride through the safe
unpickler untouched -- it gates GLOBALS, not string field VALUES). Reading that documented
property on an untrusted loaded ``.tlspec`` therefore imported the attacker-named module,
executing its top-level code = arbitrary-code execution -- even though ``trace.run()`` was
already guarded through the ``ir/container.py`` path.

The fix consolidates BOTH sinks onto the SINGLE default-deny resolver
``torchlens.ir.container.resolve_container_type``: it never imports a bundle-controlled
module name (resolves only from ``sys.modules``), decides admissibility by name against the
recorded container kind, and returns a graceful qualified-name string only when the type is
not resolvable without importing. A legit ``torch.max`` / ``torch.topk`` structseq (or
``OrderedDict`` / HF ``ModelOutput``) container still resolves correctly and round-trips.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess  # noqa: F401 -- ensure it is in sys.modules for the tamper test
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.container import (
    ContainerReconstructionError,
    resolve_container_type,
)
from torchlens.ir import container as _ir_container


class _MaxModel(nn.Module):
    """Return a genuine ``torch.return_types.max`` multi-output container."""

    def forward(self, x: torch.Tensor) -> Any:
        r = torch.max(x, dim=1)
        return r.values + r.indices.float()


def _multi_output_op(trace: tl.Trace) -> Any:
    """Return the first op that came from a multi-output container."""

    for op in trace.ops:
        if getattr(op, "in_multi_output", False):
            return op
    raise AssertionError("no multi-output op captured")


def _trace_max_model() -> tl.Trace:
    return tl.trace(
        _MaxModel(),
        torch.randn(2, 4),
        save=tl.func("max"),
        intervention_ready=True,
    )


def test_op_py_has_no_second_import_module_resolver() -> None:
    """Lock: op.py must not re-introduce its own ``importlib.import_module`` resolver.

    The whole class of bug is a SECOND resolver drifting from the hardened one; guard
    against regression by asserting the vulnerable primitive is absent from the module.
    """

    op_src = Path(_ir_container.__file__).parent.parent / "data_classes" / "op.py"
    text = op_src.read_text()
    assert "import_module" not in text, (
        "op.py must route container-type resolution through the single hardened "
        "ir.container.resolve_container_type resolver, never its own import_module."
    )


def test_shared_resolver_is_the_ir_container_one() -> None:
    """The consolidated resolver is exactly ir.container's private default-deny gate."""

    assert resolve_container_type is _ir_container._resolve_container_type


def test_legit_multi_output_type_resolves_live_and_after_load() -> None:
    """A real ``torch.max`` structseq resolves to ``torch.return_types.max`` and round-trips."""

    trace = _trace_max_model()
    live_type = _multi_output_op(trace).multi_output_type
    assert live_type is torch.return_types.max

    bundle = os.path.join(tempfile.mkdtemp(), "m.tlspec")
    tl.save(trace, bundle, level="runnable")
    loaded = tl.load(bundle)
    loaded_type = _multi_output_op(loaded).multi_output_type
    assert loaded_type is torch.return_types.max


def test_multi_output_type_never_imports_attacker_module() -> None:
    """Tampered ``type_module`` (module NOT loaded) imports nothing; degrades to a string.

    This is the direct property-read RCE from the repro: an attacker-named module that is
    importable on the victim path must NOT be imported when the documented public property
    is read. It degrades to the historical qualified-name string instead.
    """

    evil = "evil_pwn_module_r11_xyz"
    assert evil not in sys.modules
    trace = _trace_max_model()
    op = _multi_output_op(trace)
    tampered = dataclasses.replace(op.container_spec, type_module=evil, type_qualname="max")
    op.container_spec = tampered

    resolved = op.multi_output_type

    assert evil not in sys.modules, "reading multi_output_type imported an attacker module"
    assert resolved == f"{evil}.max"
    assert not isinstance(resolved, type)


def test_multi_output_type_denies_loaded_but_inadmissible_type() -> None:
    """Tampered spec naming an already-loaded but wrong type is refused, not constructed.

    ``subprocess.Popen`` is loaded in-process yet is not a benign ``namedtuple`` container
    type; the default-deny resolver must raise ``ContainerReconstructionError`` BEFORE any
    construction so a crafted spec cannot execute ``subprocess.Popen(...)``.
    """

    trace = _trace_max_model()
    op = _multi_output_op(trace)
    tampered = dataclasses.replace(
        op.container_spec, type_module="subprocess", type_qualname="Popen"
    )
    op.container_spec = tampered

    with pytest.raises(ContainerReconstructionError):
        _ = op.multi_output_type


def test_end_to_end_tampered_bundle_property_read_is_safe() -> None:
    """Full flow: tamper the saved bundle's metadata, load, read property -> no code exec."""

    sentinel = os.path.join(tempfile.mkdtemp(), "r11_pwned_sentinel")
    if os.path.exists(sentinel):
        os.remove(sentinel)
    evil = "evil_pwn_module_xx"  # 18 chars == len("torch.return_types")
    assert len(evil) == len("torch.return_types")
    drop = tempfile.mkdtemp()
    with open(os.path.join(drop, evil + ".py"), "w") as handle:
        handle.write(f"import os\nopen({sentinel!r}, 'w').write('pwned')\n")
    sys.path.insert(0, drop)
    try:
        trace = _trace_max_model()
        bundle = os.path.join(tempfile.mkdtemp(), "m.tlspec")
        tl.save(trace, bundle, level="runnable")

        pkl = os.path.join(bundle, "metadata.pkl")
        raw = open(pkl, "rb").read()
        raw = raw.replace(b"torch.return_types", evil.encode())
        open(pkl, "wb").write(raw)

        loaded = tl.load(bundle)
        _ = _multi_output_op(loaded).multi_output_type

        assert evil not in sys.modules
        assert not os.path.exists(sentinel), "untrusted bundle executed attacker code"
    finally:
        sys.path.remove(drop)
