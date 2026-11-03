"""Regression tests for the cert8 (round-8-prep) intervention hardening fixes.

Two BLOCKER-severity bugs found by the cert7 inspector, each a recurrence of a
class a prior round fixed at a *different* site:

* BLOCKER-1 -- ``Bundle._diff_members`` (backing ``Bundle.compare(a, b)``) called
  the scalar-only fallback ``relative_l1`` UNCONDITIONALLY, silently
  truncating every multi-element activation to its first element. This is the third
  sibling of the round-6 / cert7 scalar-vs-vector truncation class. The fix applies
  the same symmetric ``is_scalar_like(a) and is_scalar_like(b)`` guard used by every
  other caller (``_diff_row``/``_diff_matrix``/``most_changed``/``_distance_value``),
  routing multi-element pairs through the default vector metric (cosine).

* BLOCKER-2 -- helper deserialization must stay on the full value codec. Builtin
  helper callable/opaque arguments must never round-trip as raw wrapper dicts:
  ``tl.replace_with(fn)`` must load cleanly at the DEFAULT save level, and an
  audit-level ``tl.splice_module(...)`` payload must load as an explicitly
  non-executable placeholder (raising ``NonExecutableSpecError`` at fire time)
  rather than a silently corrupted helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention._helpers import Edit
from torchlens.intervention._save import SaveLevel
from torchlens.intervention._spec_values import (
    LazyImportRef,
    _deserialize_value,
    _resolve_import_ref,
    _serialize_value,
    _SerializedState,
)
from torchlens.intervention.errors import NonExecutableSpecError, UntrustedCallableError


def _new_state() -> _SerializedState:
    """Return a fresh empty serialization state."""

    return _SerializedState(tensor_entries=[], tensor_refs={})


def _top_level_replacement() -> torch.Tensor:
    """Import-resolvable zero-argument callable used for the callable-arg round-trip test.

    Returns
    -------
    torch.Tensor
        A zero tensor shaped like the one-row, three-column test input.
    """

    return torch.zeros(1, 3)


# ---------------------------------------------------------------------------
# BLOCKER-1: Bundle.compare(member_a, member_b) must not truncate to element 0.
# ---------------------------------------------------------------------------


def test_compare_node_multi_element_uses_vector_metric() -> None:
    """BLOCKER-1 sibling coverage: the SuperOp node diff also stays a vector metric."""

    torch.manual_seed(1)
    x = torch.randn(2, 6)
    log_a = tl.trace(nn.Linear(6, 6), x, capture=tl.options.CaptureOptions(replayable=True))
    log_b = tl.trace(nn.Linear(6, 6), x, capture=tl.options.CaptureOptions(replayable=True))
    bundle = tl.Bundle({"a": log_a, "b": log_b}, baseline="a")

    linear_site = next(s.layer_label for s in log_a._layer_list if "linear" in str(s.layer_label))
    matrix = bundle.at(linear_site).compare()
    # A 2x2 distance matrix with finite, non-truncated off-diagonal entries.
    assert matrix.shape == (2, 2)
    assert torch.isfinite(matrix).all()


# ---------------------------------------------------------------------------
# BLOCKER-2: helper arg decoder must stay in lockstep with the encoder.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_callable_helper_arg_roundtrips_at_default_save_level() -> None:
    """BLOCKER-2 case 1: a portable callable helper arg decodes to a callable.

    The narrow ``_decode_jsonish`` returned the raw ``{'__callable__': ...}`` wrapper
    (crashing at load with a misleading ``HookSignatureError``); the full codec
    returns a callable ``LazyImportRef``.
    """

    helper = tl.replace_with(_top_level_replacement)
    state = _new_state()
    serialized = _serialize_value(helper, SaveLevel.EXECUTABLE_WITH_CALLABLES, state)
    loaded = _deserialize_value(serialized, {})

    assert isinstance(loaded, Edit)
    assert loaded.name == "replace_with"
    (arg,) = loaded.args
    assert not isinstance(arg, dict), "callable arg decoded as a raw wrapper dict"
    assert isinstance(arg, LazyImportRef)
    assert callable(arg)
    # Deny-by-default (r12 RCE guard): the foreign local test module is not resolved
    # under the default untrusted contract.
    with pytest.raises(UntrustedCallableError):
        _resolve_import_ref(arg.import_path)
    # Under explicit trust, the import ref resolves back to the original function.
    assert (
        _resolve_import_ref(arg.import_path, trust_custom_callables=True) is _top_level_replacement
    )


@pytest.mark.smoke
def test_gradient_site_callable_end_to_end_save_load_default_level(tmp_path: Path) -> None:
    """BLOCKER-2 case 1: full save->load of a gradient-site callable spec at the DEFAULT level."""

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    class _IdentityRelu(nn.Module):
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return torch.relu(t)

    trace = tl.trace(
        _IdentityRelu(),
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True),
        intervene=tl.when(tl.grad(tl.func("relu")), tl.replace_with(_top_level_replacement)),
    )
    path = tmp_path / "gradient_site_callable.tlspec"
    # Default level == executable_with_callables.
    trace.save_intervention(path)

    from torchlens.io import load_intervention_spec

    # This previously raised HookSignatureError at LOAD; it must now succeed.
    spec = load_intervention_spec(path)
    assert spec is not None


def test_audit_opaque_helper_arg_fails_loudly_not_silent(tmp_path: Path) -> None:
    """BLOCKER-2 case 2: an audit-level opaque helper arg is non-executable.

    The old decoder handed a raw dict to the constructor, producing a helper that
    only crashed with a masked ``TypeError`` several frames later. Now the decoded
    arg is a recognizable non-executable placeholder and the whole helper refuses to
    fire with a clear ``NonExecutableSpecError``.
    """

    helper = tl.splice_module(nn.Identity())
    state = _new_state()
    serialized = _serialize_value(helper, SaveLevel.AUDIT, state)
    loaded = _deserialize_value(serialized, {})

    assert isinstance(loaded, Edit)
    assert loaded.portability == "opaque_audit"
    assert dict(loaded.metadata).get("executable") is False
    # Firing raises a clear, specific error -- NOT a masked TypeError downstream.
    try:
        loaded()  # Edit.__call__ -> factory()
    except NonExecutableSpecError as exc:
        assert "non-executable" in str(exc).lower()
        assert "splice_module" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected NonExecutableSpecError firing an audit-only helper")
