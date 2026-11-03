"""Round-22 arg-position hardening gates.

Guards the class fixed in round 22: under-specified ``FUNC_ARG_SPECS`` entries
dropping real tensor parents (F1 ``lu_solve``, F2 ``cosine_similarity``, F4
``ctc_loss``, F5 ``searchsorted``), Tier-2 dynamic-cache poisoning making capture
depend on trace order (F3), and the completeness witness being disarmed by a
same-named-but-narrower ATen packet (F6, ``torch.tensor``).

The headline gate is the schema-vs-spec completeness sweep: every static spec's
tensor coverage must match its ATen schemas' input-tensor slots, so the whole
under-specified-spec bug class stays closed structurally, not just per instance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
import torchlens._state as _state
from torchlens._capture.arg_positions import (
    DYNAMIC_SPEC_UNCACHEABLE,
    FUNC_ARG_SPECS,
    ArgSpec,
    _iter_aten_packet_names,
    _normalize_func_name,
    _schema_arg_is_parent_candidate,
    _schema_tensor_arg_kind,
)
from torchlens._constants import get_orig_torch_funcs
from torchlens.backends.torch._ops_predicate_events import _extract_arg_tensors_and_params

_PROVENANCE_MATCH = "no graph/source provenance"


# ---------------------------------------------------------------------------
# Headline gate: schema-vs-spec completeness sweep (systemic guard for F1/F2/F4/F5)
# ---------------------------------------------------------------------------


def _spec_schema_tensor_slot_violations(
    spec_table: dict[str, ArgSpec],
) -> list[str]:
    """Return every schema tensor slot not covered by its static spec.

    Parameters
    ----------
    spec_table:
        Arg-spec table to audit (normally ``FUNC_ARG_SPECS``).

    Returns
    -------
    list[str]
        One entry per uncovered parent-candidate tensor slot, formatted as
        ``"<normalized_name>: <packet>.<overload> arg<i> (<name>) ..."``.
    """

    violations: list[str] = []
    audited: set[str] = set()
    for namespace_name, func_name in get_orig_torch_funcs(include_torchvision=False):
        normalized_name = _normalize_func_name(func_name.strip("_"))
        spec = spec_table.get(normalized_name)
        if spec is None:
            continue
        position_set = set(spec.positions)
        sequence_set = set(spec.sequence_positions)
        kwarg_set = {_normalize_func_name(str(name)) for name in spec.tensor_kwargs}
        for packet_name in _iter_aten_packet_names(namespace_name, func_name):
            packet = getattr(torch.ops.aten, packet_name, None)
            if packet is None:
                continue
            for overload_name in packet.overloads():
                schema = getattr(getattr(packet, overload_name, None), "_schema", None)
                if schema is None:
                    continue
                for index, schema_arg in enumerate(getattr(schema, "arguments", ()) or ()):
                    if not _schema_arg_is_parent_candidate(schema_arg):
                        continue
                    tensor_kind = _schema_tensor_arg_kind(schema_arg)
                    if tensor_kind is None:
                        continue
                    key = f"{normalized_name}:{packet_name}.{overload_name}:{index}"
                    if key in audited:
                        continue
                    audited.add(key)
                    # A "single" tensor slot must be a covered position; a
                    # "sequence" slot may be covered by sequence_positions OR
                    # positions (position extraction handles shallow sequences).
                    position_covered = index in position_set or (
                        tensor_kind == "sequence" and index in sequence_set
                    )
                    arg_name = getattr(schema_arg, "name", None)
                    name_covered = (
                        isinstance(arg_name, str) and _normalize_func_name(arg_name) in kwarg_set
                    )
                    if not (position_covered and name_covered):
                        violations.append(
                            f"{normalized_name}: {packet_name}.{overload_name or 'default'} "
                            f"arg{index} ({arg_name}) kind={tensor_kind} "
                            f"position_covered={position_covered} name_covered={name_covered}"
                        )
    return violations


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/arg_positions.py.
def test_completeness_sweep_detects_narrowed_spec() -> None:
    """Mutation guard: the sweep reports a spec narrowed back to pre-fix shape.

    Re-narrows ``ctcloss`` to the round-22 F4 under-specified input/target spec
    in a COPY of the table and asserts the sweep helper catches the dropped
    tensor-length slots. Proves the sweep actually kills the bug class it
    guards, without touching live state.
    """

    mutated = dict(FUNC_ARG_SPECS)
    mutated["ctcloss"] = ArgSpec(positions=(0, 1), tensor_kwargs=("input", "target"))

    violations = _spec_schema_tensor_slot_violations(mutated)

    assert any(v.startswith("ctcloss:") and "arg2" in v for v in violations)
    assert any(v.startswith("ctcloss:") and "arg3" in v for v in violations)


# ---------------------------------------------------------------------------
# F3: Tier-2 dynamic-cache poisoning
# ---------------------------------------------------------------------------


def _pop_dynamic(name: str) -> None:
    """Remove one dynamic-cache entry so a test starts from a cold cache."""

    _state._dynamic_arg_specs.pop(name, None)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/arg_positions.py.
def test_dynamic_cache_extraction_is_call_order_independent() -> None:
    """A scalar-RHS first observation must not drop a later tensor-RHS operand.

    Exercises the real Tier-2 path through ``_extract_arg_tensors_and_params``
    with a name absent from ``FUNC_ARG_SPECS``, in both observation orders, plus
    the kwarg spelling. Pre-fix the first call froze ``positions=(0,)`` in the
    process-global cache and the second call silently dropped its tensor RHS.
    """

    name = "zzargposhardeningorderprobe"
    lhs, rhs = torch.randn(3), torch.randn(3)
    try:
        _pop_dynamic(name)
        first, _ = _extract_arg_tensors_and_params(name, (lhs, 2.0), {})
        assert [id(t) for t in first] == [id(lhs)]
        second, _ = _extract_arg_tensors_and_params(name, (lhs, rhs), {})
        assert {id(t) for t in second} == {id(lhs), id(rhs)}

        # Reverse order must give identical coverage (order independence).
        _pop_dynamic(name)
        wide, _ = _extract_arg_tensors_and_params(name, (lhs, rhs), {})
        assert {id(t) for t in wide} == {id(lhs), id(rhs)}
        narrow, _ = _extract_arg_tensors_and_params(name, (lhs, 2.0), {})
        assert [id(t) for t in narrow] == [id(lhs)]

        # Kwarg slot appearing only in a later call (pack_padded_sequence-style
        # optional tensor lengths) must also be picked up.
        _pop_dynamic(name)
        _extract_arg_tensors_and_params(name, (lhs,), {"lengths": [2, 1]})
        with_kwarg, _ = _extract_arg_tensors_and_params(name, (lhs,), {"lengths": rhs})
        assert {id(t) for t in with_kwarg} == {id(lhs), id(rhs)}
    finally:
        _pop_dynamic(name)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/arg_positions.py, backends/torch/_ops_predicate_events.py, campaign mutant kills.
def test_dynamic_cache_unrepresentable_call_never_caches_lossy_spec() -> None:
    """Tensors beyond ArgSpec's representable shapes force a re-crawl every call.

    A dict-nested tensor is found by the BFS crawl but cannot be expressed in an
    ``ArgSpec``; caching the lossy spec would silently drop it from every call
    after the first. The name must be marked uncacheable and keep re-crawling.
    """

    name = "zzargposhardeningdeepprobe"
    base, nested = torch.randn(3), torch.randn(3)
    try:
        _pop_dynamic(name)
        first, _ = _extract_arg_tensors_and_params(name, (base, {"deep": nested}), {})
        assert {id(t) for t in first} == {id(base), id(nested)}
        assert _state._dynamic_arg_specs.get(name) is DYNAMIC_SPEC_UNCACHEABLE
        second, _ = _extract_arg_tensors_and_params(name, (base, {"deep": nested}), {})
        assert {id(t) for t in second} == {id(base), id(nested)}
    finally:
        _pop_dynamic(name)


# ---------------------------------------------------------------------------
# F1/F2/F4/F5: per-op parent-presence pins
# ---------------------------------------------------------------------------


class _CosineSimilarity(nn.Module):
    """``F.cosine_similarity`` in positional or all-kwarg spelling."""

    def __init__(self, use_kwargs: bool) -> None:
        """Store the spelling toggle.

        Parameters
        ----------
        use_kwargs:
            Whether to pass both operands by keyword.
        """

        super().__init__()
        self.use_kwargs = use_kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return cosine similarity between two input-derived operands.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar reduction of the similarity.
        """

        first = x + 1.0
        second = x * 2.0
        if self.use_kwargs:
            return F.cosine_similarity(x1=first, x2=second).sum()
        return F.cosine_similarity(first, second).sum()


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/flops.py.
def test_cosine_similarity_kwarg_spelling_parity() -> None:
    """Kwarg and positional spellings record the SAME parent set.

    Round-22 F2: the kwarg spelling dropped ALL parents (spec knew only the
    loss-style ``input``/``target`` names), recording a parentless internal
    source and severing the output's input ancestry.
    """

    x = torch.randn(4, 8)
    positional = tl.trace(_CosineSimilarity(use_kwargs=False).eval(), x)
    kwarg = tl.trace(_CosineSimilarity(use_kwargs=True).eval(), x)

    positional_op = next(o for o in positional.ops if o.type == "cosinesimilarity")
    kwarg_op = next(o for o in kwarg.ops if o.type == "cosinesimilarity")

    assert len(positional_op.parents) == 2
    assert len(kwarg_op.parents) == 2
    assert kwarg_op.has_input_ancestor
    assert kwarg_op.unattributed_tensor_args == ()


# ---------------------------------------------------------------------------
# F6: torch.tensor(existing_tensor) -- parent edge + witness authority
# ---------------------------------------------------------------------------
