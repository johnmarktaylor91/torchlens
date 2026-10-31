"""Regression tests for the r18i utils/options hardening group.

Covers:
* M15 ``resolve_runnable_torch_alias`` target-existence capability probe
  (runnable tripwire; strengthened, mutation-proof).
* M16 ``run_dynamo_explain`` signature-based convention dispatch (no
  swallowed internal ``TypeError`` / double side effect).
* M17 ``_non_torch_array_summary`` rejects callable ``.shape``/``.dtype``.
* M20 ``copy_arg_tree`` cycle safety.
* M3 ``suppress_mutate_warnings`` context-manager state restoration.
* F9 ``CaptureOptions.from_values`` validation parity with ``__init__``.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict, defaultdict
from typing import Any

import pytest
import torch

from torchlens._utils import _torch_compat
from torchlens._utils.arg_handling import copy_arg_tree
from torchlens._utils.display import _non_torch_array_summary
from torchlens.options import (
    CaptureOptions,
    suppress_mutate_warnings,
)

# --------------------------------------------------------------------------- #
# M15 -- runnable torch alias target-existence probe
# --------------------------------------------------------------------------- #


def test_m15_helper_reports_target_presence() -> None:
    """The capability probe resolves live targets and rejects absent ones."""

    assert _torch_compat._runtime_alias_target_exists("torch.nn.functional", "linear") is True
    assert _torch_compat._runtime_alias_target_exists("torch.special", "gammaln") is True
    assert _torch_compat._runtime_alias_target_exists("torch", "add") is True
    assert _torch_compat._runtime_alias_target_exists("torch.Tensor", "add") is True
    assert _torch_compat._runtime_alias_target_exists("torch.special", "nonexistent_fn___") is False
    assert _torch_compat._runtime_alias_target_exists("numpy", "linear") is False


def test_m15_dead_target_alias_skipped() -> None:
    """An unbounded/legacy version key must not manufacture a dead alias.

    Mutation-killing test: reverting the target-existence gate (letting the
    resolver always emit the matched entry) makes this assertion fail, because
    ``torch.special.nonexistent_fn___`` does not exist in the runtime.
    """

    # recorded_version=None bypasses the version bounds (documented legacy
    # behavior); only the target-existence gate stops a dead alias here.
    assert (
        _torch_compat.resolve_runnable_torch_alias(
            "torch._C._special.special_nonexistent_fn___", None
        )
        is None
    )
    assert (
        _torch_compat.resolve_runnable_torch_alias("torch._C._fft.fft_nonexistent___", "not-a-ver")
        is None
    )


def test_m15_live_target_alias_still_resolves() -> None:
    """Real private->public redirections with live targets must keep firing."""

    assert _torch_compat.resolve_runnable_torch_alias(
        "torch._C._special.special_gammaln", None
    ) == ("torch.special", "gammaln", "private_to_public:_C._special.special_*->torch.special.*")
    # A present internal builtin is still intentionally routed to its public
    # wrapper (target exists) -- redirection is not blocked by the gate.
    assert _torch_compat.resolve_runnable_torch_alias("torch._C._nn.linear", "2.5.0") == (
        "torch.nn.functional",
        "linear",
        "private_to_public:_C._nn.linear->torch.nn.functional.linear",
    )


# --------------------------------------------------------------------------- #
# M16 -- dynamo explain convention dispatch
# --------------------------------------------------------------------------- #


def test_m16_internal_typeerror_propagates_without_double_call(monkeypatch: Any) -> None:
    """A modern-signature internal ``TypeError`` propagates on a single call."""

    calls: list[Any] = []

    def fake_explain(f: Any, *extra_args: Any, **extra_kwargs: Any) -> Any:
        calls.append((extra_args, extra_kwargs))
        raise TypeError("internal error inside explain body")

    monkeypatch.setattr(_torch_compat, "get_dynamo_explain", lambda: fake_explain)
    with pytest.raises(TypeError, match="internal error inside explain body"):
        _torch_compat.run_dynamo_explain(object(), (1, 2), {"k": 3})
    assert len(calls) == 1  # no swallow-and-retry double invocation


def test_m16_modern_convention_calls_returned_callable(monkeypatch: Any) -> None:
    """Modern ``explain(f)`` returning a callable is invoked with args/kwargs."""

    inner_calls: list[Any] = []

    def fake_explain(f: Any, *extra_args: Any, **extra_kwargs: Any) -> Any:
        if extra_args or extra_kwargs:  # pragma: no cover - defensive
            raise AssertionError("modern path must not pass args to explain(f)")

        def inner(*args: Any, **kwargs: Any) -> str:
            inner_calls.append((args, kwargs))
            return "explanation"

        return inner

    monkeypatch.setattr(_torch_compat, "get_dynamo_explain", lambda: fake_explain)
    result = _torch_compat.run_dynamo_explain(object(), (7,), {"k": 8})
    assert result == "explanation"
    assert inner_calls == [((7,), {"k": 8})]


def test_m16_legacy_signature_routes_positional_args(monkeypatch: Any) -> None:
    """A legacy signature that cannot bind ``explain(f)`` uses the args call."""

    seen: list[Any] = []

    def legacy_explain(f: Any, example_inputs: Any) -> str:
        seen.append(example_inputs)
        return "legacy"

    assert "example_inputs" in inspect.signature(legacy_explain).parameters
    monkeypatch.setattr(_torch_compat, "get_dynamo_explain", lambda: legacy_explain)
    result = _torch_compat.run_dynamo_explain(object(), (99,), {})
    assert result == "legacy"
    assert seen == [99]


# --------------------------------------------------------------------------- #
# M17 -- non-torch array summary rejects callables
# --------------------------------------------------------------------------- #


def test_m17_callable_shape_and_dtype_rejected() -> None:
    """Bound-method ``.shape``/``.dtype`` never leak into the summary text."""

    class MethodArray:
        def shape(self) -> tuple[int, ...]:  # noqa: D401 - test double
            return (2, 3)

        def dtype(self) -> str:  # noqa: D401 - test double
            return "float32"

    summary = _non_torch_array_summary(MethodArray())
    assert "bound method" not in summary
    assert summary == "Tensor[?] unknown dtype"


def test_m17_value_shape_and_dtype_preserved() -> None:
    """Genuine value attributes still render as before."""

    class ValueArray:
        shape = (4, 5)
        dtype = "int64"

    assert _non_torch_array_summary(ValueArray()) == "Tensor[4, 5] int64"


# --------------------------------------------------------------------------- #
# M20 -- copy_arg_tree cycle safety
# --------------------------------------------------------------------------- #


def test_m20_self_referential_list_terminates() -> None:
    """A directly self-referential list copies without a RecursionError."""

    original: list[Any] = [1]
    original.append(original)
    copied = copy_arg_tree(original)
    assert copied[0] == 1
    assert copied[1] is copied  # cycle reproduced in the copy
    assert copied is not original


def test_m20_mutual_dict_list_cycle_terminates() -> None:
    """A cycle through mixed mutable containers terminates and is reproduced."""

    d: dict[str, Any] = {}
    lst: list[Any] = [d]
    d["back"] = lst
    copied = copy_arg_tree(d)
    assert copied["back"][0] is copied
    assert isinstance(copied["back"], list)


def test_m20_containers_and_tensors_still_copied() -> None:
    """Non-cyclic structures keep their historical clone/recurse behavior."""

    t = torch.ones(2, 2)
    payload = OrderedDict(a=[t, (t, 1)], b=defaultdict(list, {"x": [t]}))
    copied = copy_arg_tree(payload)
    assert isinstance(copied, OrderedDict)
    assert isinstance(copied["b"], defaultdict)
    assert copied["b"].default_factory is list
    cloned = copied["a"][0]
    assert isinstance(cloned, torch.Tensor)
    assert cloned is not t
    assert torch.equal(cloned, t)
    # Tensors are cloned per occurrence (not memoized/aliased).
    assert copied["a"][0] is not copied["a"][1][0]


# --------------------------------------------------------------------------- #
# M3 -- suppress_mutate_warnings context state restoration
# --------------------------------------------------------------------------- #


def test_m3_context_restores_prior_state() -> None:
    """``with suppress_mutate_warnings():`` restores the pre-with state."""

    suppress_mutate_warnings(False)
    try:
        assert suppress_mutate_warnings.is_suppressed is False
        with suppress_mutate_warnings():
            assert suppress_mutate_warnings.is_suppressed is True
        assert suppress_mutate_warnings.is_suppressed is False  # no leak
    finally:
        suppress_mutate_warnings(False)


def test_m3_context_restores_prior_true_state_and_nests() -> None:
    """Nesting restores each level's prior state without leaking."""

    suppress_mutate_warnings(True)
    try:
        assert suppress_mutate_warnings.is_suppressed is True
        with suppress_mutate_warnings():
            assert suppress_mutate_warnings.is_suppressed is True
            with suppress_mutate_warnings():
                assert suppress_mutate_warnings.is_suppressed is True
            assert suppress_mutate_warnings.is_suppressed is True
        assert suppress_mutate_warnings.is_suppressed is True  # restored to prior True
    finally:
        suppress_mutate_warnings(False)


def test_m3_bare_call_toggle_and_identity_preserved() -> None:
    """The bare session toggle and ``returned is controller`` contract hold."""

    suppress_mutate_warnings(False)
    try:
        returned = suppress_mutate_warnings(True)
        assert returned is suppress_mutate_warnings
        assert suppress_mutate_warnings.is_suppressed is True
    finally:
        suppress_mutate_warnings(False)
    assert suppress_mutate_warnings.is_suppressed is False


def test_m3_context_suppresses_warning_body() -> None:
    """The suppressed block behaves like the historical usage."""

    suppress_mutate_warnings(False)
    try:
        with suppress_mutate_warnings():
            assert suppress_mutate_warnings.is_suppressed is True
    finally:
        suppress_mutate_warnings(False)


# --------------------------------------------------------------------------- #
# F9 -- CaptureOptions.from_values validation parity
# --------------------------------------------------------------------------- #


def _capture_values(**overrides: Any) -> dict[str, Any]:
    """Return default capture field values with overrides applied."""

    values = CaptureOptions().as_dict()
    values.update(overrides)
    return values


def test_f9_from_values_rejects_bad_jax_control_flow() -> None:
    """``from_values`` rejects an invalid ``jax_control_flow`` like ``__init__``.

    Mutation-killing test: removing the ``_validate_capture_values`` call from
    ``from_values`` makes this pass silently (no raise), failing the test.
    """

    with pytest.raises(ValueError, match="jax_control_flow"):
        CaptureOptions.from_values(_capture_values(jax_control_flow="garbage"), frozenset())


def test_f9_from_values_rejects_nonpositive_unroll() -> None:
    """``from_values`` rejects a non-positive unroll bound like ``__init__``."""

    with pytest.raises(ValueError, match="jax_max_control_flow_unroll"):
        CaptureOptions.from_values(_capture_values(jax_max_control_flow_unroll=-5), frozenset())


def test_f9_from_values_accepts_valid_values() -> None:
    """Valid resolved values still construct successfully."""

    opts = CaptureOptions.from_values(
        _capture_values(jax_control_flow="region", jax_max_control_flow_unroll=8),
        frozenset({"jax_control_flow"}),
    )
    assert opts.jax_control_flow == "region"
    assert opts.jax_max_control_flow_unroll == 8


def test_f9_flat_merge_path_validates() -> None:
    """The flat-kwarg capture merge path routes through the same validation."""

    with pytest.raises((ValueError, TypeError)):
        CaptureOptions(jax_control_flow="garbage")
    # The flat merge uses from_values under the hood; a bad value must not
    # silently construct an invalid CaptureOptions.
    with pytest.raises(ValueError):
        CaptureOptions.from_values(_capture_values(jax_control_flow="garbage"), frozenset())
