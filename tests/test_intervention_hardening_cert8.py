"""Regression tests for the cert8 (round-8-prep) intervention hardening fixes.

Two BLOCKER-severity bugs found by the cert7 inspector, each a recurrence of a
class a prior round fixed at a *different* site:

* BLOCKER-1 -- ``Bundle._diff_members`` (backing ``Bundle.diff_pair(a, b)``) called
  the scalar-only fallback ``relative_l1_scalar`` UNCONDITIONALLY, silently
  truncating every multi-element activation to its first element. This is the third
  sibling of the round-6 / cert7 scalar-vs-vector truncation class. The fix applies
  the same symmetric ``is_scalar_like(a) and is_scalar_like(b)`` guard used by every
  other caller (``_diff_row``/``_diff_matrix``/``most_changed``/``_distance_value``),
  routing multi-element pairs through the default vector metric (cosine).

* BLOCKER-2 -- ``helpers.py::_decode_jsonish`` was a strictly narrower decoder than
  ``save.py::_serialize_value``'s encoder (it understood only ``__tensor_ref__``),
  so a builtin helper's callable/opaque argument round-tripped as a raw wrapper
  dict. This broke ``tl.bwd_hook(fn)`` save/load at the DEFAULT save level and let an
  audit-level ``tl.splice_module(...)`` spec load a corrupted arg that only crashed
  later. The fix routes helper args/kwargs through the SAME full codec and returns an
  explicitly non-executable placeholder (raising ``NonExecutableSpecError`` at fire
  time) when a decoded arg is a non-executable ``opaque_audit`` placeholder.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.bundle import _distance_value
from torchlens.intervention import helpers as _helpers_mod
from torchlens.intervention._metrics import (
    cosine_distance,
    is_scalar_like,
    relative_l1_scalar,
)
from torchlens.intervention.errors import NonExecutableSpecError
from torchlens.intervention.helpers import HelperSpec, helper_from_serialized
from torchlens.intervention.save import (
    LazyImportRef,
    SaveLevel,
    _SerializedState,
    _deserialize_value,
    _resolve_import_ref,
    _serialize_value,
)


def _new_state() -> _SerializedState:
    """Return a fresh empty serialization state."""

    return _SerializedState(tensor_entries=[], tensor_refs={})


def _top_level_bwd_hook(grad: torch.Tensor, *, hook: object) -> torch.Tensor:
    """Import-resolvable backward hook used for the bwd_hook round-trip test.

    Parameters
    ----------
    grad:
        Incoming gradient tensor.
    hook:
        Hook context (unused).

    Returns
    -------
    torch.Tensor
        A zeroed gradient.
    """

    del hook
    return torch.zeros_like(grad)


# ---------------------------------------------------------------------------
# BLOCKER-1: Bundle.diff_pair(member_a, member_b) must not truncate to element 0.
# ---------------------------------------------------------------------------


def test_diff_pair_members_multi_element_uses_vector_metric_no_truncation() -> None:
    """BLOCKER-1: per-site scores are a genuine vector metric, not first-element."""

    torch.manual_seed(0)
    x = torch.randn(2, 8)
    log_a = tl.trace(nn.Linear(8, 8), x, intervention_ready=True)
    log_b = tl.trace(nn.Linear(8, 8), x, intervention_ready=True)
    bundle = tl.bundle({"a": log_a, "b": log_b}, baseline="a")

    rows = dict(bundle.diff_pair("a", "b"))
    assert rows, "diff_pair returned no rows"

    # The differing linear site: score must equal the real cosine distance over the
    # whole tensor, and must NOT equal the old first-element-only relative_l1 formula.
    linear_site = next(label for label in rows if "linear" in label)
    left_out = log_a[linear_site].out
    right_out = log_b[linear_site].out
    assert left_out.numel() > 1  # genuinely multi-element -- truncation would matter

    expected = float(cosine_distance(left_out, right_out).detach().item())
    assert abs(rows[linear_site] - expected) < 1e-5

    first_element_only = float(relative_l1_scalar(left_out[:1], right_out[:1]).detach().item())
    # For a non-degenerate site the aggregate and the first-element score differ;
    # the OLD bug returned the latter.
    assert abs(rows[linear_site] - first_element_only) > 1e-4


def test_diff_pair_node_multi_element_uses_vector_metric() -> None:
    """BLOCKER-1 sibling coverage: the SuperOp node diff also stays a vector metric."""

    torch.manual_seed(1)
    x = torch.randn(2, 6)
    log_a = tl.trace(nn.Linear(6, 6), x, intervention_ready=True)
    log_b = tl.trace(nn.Linear(6, 6), x, intervention_ready=True)
    bundle = tl.bundle({"a": log_a, "b": log_b}, baseline="a")

    linear_site = next(s.layer_label for s in log_a.layer_list if "linear" in str(s.layer_label))
    matrix = bundle.node(linear_site).diff_pair()
    # A 2x2 distance matrix with finite, non-truncated off-diagonal entries.
    assert matrix.shape == (2, 2)
    assert torch.isfinite(matrix).all()


def test_no_unguarded_relative_l1_scalar_call_site_remains() -> None:
    """BLOCKER-1: EVERY relative_l1_scalar call site is scalar-guarded.

    Source-level tripwire: catches a FUTURE un-swept sibling. Any call to
    ``relative_l1_scalar(`` in the multi-trace scope must sit inside an
    ``is_scalar_like``-gated ternary (or be the definition/import itself).
    """

    import importlib

    bundle_mod = importlib.import_module("torchlens.bundle")
    super_base_mod = importlib.import_module("torchlens.intervention._super._base")

    call_re = re.compile(r"\brelative_l1_scalar\s*\(")
    for module in (bundle_mod, super_base_mod):
        source_path = Path(str(module.__file__))
        lines = source_path.read_text().splitlines()
        for idx, line in enumerate(lines):
            if not call_re.search(line):
                continue
            # The guard is on the same or an immediately adjacent line
            # (``relative_l1_scalar(...)\n if is_scalar_like(a) and is_scalar_like(b)``).
            window = "\n".join(lines[idx : idx + 3])
            assert "is_scalar_like" in window, (
                f"UNGUARDED relative_l1_scalar call at {source_path}:{idx + 1} -- "
                "every call site must gate on is_scalar_like (scalar/vector truncation "
                "class, 4th recurrence)"
            )


def test_relative_l1_scalar_itself_rejects_numel_mismatch() -> None:
    """BLOCKER-1 defense-in-depth: the scalar fallback refuses to truncate."""

    try:
        relative_l1_scalar(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0]))
    except ValueError as exc:
        assert "equal element counts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("relative_l1_scalar must reject mismatched numel")


def test_distance_value_multi_element_matches_cosine() -> None:
    """BLOCKER-1 sibling: _distance_value routes multi-element through cosine."""

    a = torch.randn(16)
    b = torch.randn(16)
    assert not (is_scalar_like(a) and is_scalar_like(b))
    got = _distance_value(a, b, "cosine")
    expected = float(cosine_distance(a, b).detach().item())
    assert abs(got - expected) < 1e-6


# ---------------------------------------------------------------------------
# BLOCKER-2: helper arg decoder must stay in lockstep with the encoder.
# ---------------------------------------------------------------------------


def test_bwd_hook_callable_arg_roundtrips_at_default_save_level() -> None:
    """BLOCKER-2 case 1: a portable callable helper arg decodes to a callable.

    The narrow ``_decode_jsonish`` returned the raw ``{'__callable__': ...}`` wrapper
    (crashing at load with a misleading ``HookSignatureError``); the full codec
    returns a callable ``LazyImportRef``.
    """

    helper = tl.bwd_hook(_top_level_bwd_hook)
    state = _new_state()
    serialized = _serialize_value(helper, SaveLevel.EXECUTABLE_WITH_CALLABLES, state)
    loaded = _deserialize_value(serialized, {})

    assert isinstance(loaded, HelperSpec)
    assert loaded.name == "bwd_hook"
    (arg,) = loaded.args
    assert not isinstance(arg, dict), "callable arg decoded as a raw wrapper dict"
    assert isinstance(arg, LazyImportRef)
    assert callable(arg)
    # The import ref resolves back to the original function.
    assert _resolve_import_ref(arg.import_path) is _top_level_bwd_hook


def test_bwd_hook_end_to_end_save_load_default_level(tmp_path: Path) -> None:
    """BLOCKER-2 case 1: full save->load of a bwd_hook spec at the DEFAULT level."""

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    class _IdentityRelu(nn.Module):
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return torch.relu(t)

    trace = tl.trace(
        _IdentityRelu(),
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True),
        intervene=tl.when(tl.func("relu"), tl.bwd_hook(_top_level_bwd_hook)),
    )
    path = tmp_path / "bwd_hook.tlspec"
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

    assert isinstance(loaded, HelperSpec)
    assert loaded.portability == "opaque_audit"
    assert dict(loaded.metadata).get("executable") is False
    # Firing raises a clear, specific error -- NOT a masked TypeError downstream.
    try:
        loaded()  # HelperSpec.__call__ -> factory()
    except NonExecutableSpecError as exc:
        assert "non-executable" in str(exc).lower()
        assert "splice_module" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected NonExecutableSpecError firing an audit-only helper")


def test_helper_from_serialized_uses_full_codec_for_opaque_args() -> None:
    """BLOCKER-2: helper_from_serialized detects a decoded opaque_audit placeholder."""

    # A builtin helper payload whose single arg is an opaque_audit callable wrapper.
    payload = {
        "portability": "builtin",
        "name": "splice_module",
        "kind": "forward",
        "direction": "forward",
        "args": [{"__callable__": {"portability": "opaque_audit", "repr": "<opaque>"}}],
        "kwargs": {},
    }
    result = helper_from_serialized(
        payload,
        tensor_loader=lambda tid: torch.zeros(1),
        import_resolver=_resolve_import_ref,
        value_decoder=lambda item: _deserialize_value(item, {}),
    )
    assert isinstance(result, HelperSpec)
    assert result.portability == "opaque_audit"
    try:
        result()
    except NonExecutableSpecError:
        pass
    else:  # pragma: no cover
        raise AssertionError("non-executable placeholder must refuse to fire")


def test_decode_gap_would_have_returned_raw_dict() -> None:
    """BLOCKER-2 characterization: the narrow decoder leaks non-tensor wrappers.

    Pins WHY the full codec is required -- the legacy ``_decode_jsonish`` returns a
    ``__callable__`` wrapper unchanged (as a raw dict), which is exactly the silent
    corruption the fix eliminates.
    """

    wrapper = {"__callable__": {"portability": "import_ref", "import_path": "x:y"}}
    leaked = _helpers_mod._decode_jsonish(wrapper, lambda tid: torch.zeros(1))
    assert leaked == wrapper  # raw dict leaks through -- must never reach a constructor
