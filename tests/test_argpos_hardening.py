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

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
import torchlens._state as _state
from torchlens._constants import get_orig_torch_funcs
from torchlens.backends.torch.ops import (
    _extract_arg_tensors_and_params,
)
from torchlens.capture.arg_positions import (
    DYNAMIC_SPEC_UNCACHEABLE,
    FUNC_ARG_SPECS,
    ArgSpec,
    _ensure_schema_tensor_position_corrections,
    _iter_aten_packet_names,
    _normalize_func_name,
    _schema_arg_is_parent_candidate,
    _schema_tensor_arg_kind,
)

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


def test_static_specs_cover_all_aten_schema_tensor_slots() -> None:
    """Every static spec covers every ATen-schema input-tensor slot.

    This is the systemic tripwire behind round-22 F1/F2/F4/F5: each of those
    bugs was a hand-grouped spec silently missing a schema-typed tensor operand
    (``lu_solve`` pos 2, ``cosine_similarity`` x1/x2 kwargs, ``ctc_loss``
    lengths, ``searchsorted`` sorter). The schema-correction pass (armed lazily
    by ``wrap_torch()`` before any torch op record can be built) widens EVERY
    spec from schema, so this sweep must be clean; a new violation means a spec
    was narrowed or a correction regressed -- fix the spec, never ledger the
    violation away.
    """

    # Audit the table exactly as torch capture reads it: corrections armed.
    _ensure_schema_tensor_position_corrections()
    violations = _spec_schema_tensor_slot_violations(FUNC_ARG_SPECS)

    assert violations == [], "\n".join(violations)


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


class _ModScalarThenTensor(nn.Module):
    """``x % scalar`` before ``a % b`` inside one forward (round-22 F3a)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a mod chain whose second ``%`` has a tensor RHS.

        Parameters
        ----------
        x:
            Strictly positive input tensor.

        Returns
        -------
        torch.Tensor
            Combined mod results.
        """

        scalar_first = x % 2.0
        return (x * 3.0) % (x + 1.0) + scalar_first.sum()


class _ModTensorOnly(nn.Module):
    """A single ``a % b`` with tensor RHS (round-22 F3b fresh-trace probe)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(x * 3) % (x + 1)``.

        Parameters
        ----------
        x:
            Strictly positive input tensor.

        Returns
        -------
        torch.Tensor
            Elementwise modulo of the two derived tensors.
        """

        return (x * 3.0) % (x + 1.0)


def test_mod_tensor_rhs_parent_is_trace_order_independent() -> None:
    """``%`` keeps its tensor RHS parent regardless of prior scalar-RHS traces.

    Round-22 F3: ``Tensor.__mod__`` (the public ``%`` operator) was missing from
    ``FUNC_ARG_SPECS`` (ledger-mislabeled as an internal helper), so a first
    ``x % scalar`` froze ``positions=(0,)`` in the process-global dynamic cache
    and every later ``a % b`` -- same trace AND fresh traces -- dropped the
    tensor RHS parent. Pins both the same-trace and the fresh-trace directions.
    """

    x = torch.abs(torch.randn(4)) + 2.0

    trace = tl.trace(_ModScalarThenTensor().eval(), x)
    mods = [op for op in trace.ops if op.type == "mod"]
    assert len(mods) == 2
    tensor_mod = mods[1]
    assert len(tensor_mod.parents) == 2
    assert set(tensor_mod.parent_arg_positions["args"]) == {0, 1}
    assert tensor_mod.unattributed_tensor_args == ()

    fresh = tl.trace(_ModTensorOnly().eval(), x)
    fresh_mod = next(op for op in fresh.ops if op.type == "mod")
    assert len(fresh_mod.parents) == 2
    assert set(fresh_mod.parent_arg_positions["args"]) == {0, 1}
    assert fresh_mod.unattributed_tensor_args == ()


# ---------------------------------------------------------------------------
# F1/F2/F4/F5: per-op parent-presence pins
# ---------------------------------------------------------------------------


class _LuSolveLinalg(nn.Module):
    """``torch.linalg.lu_solve`` with an input-derived RHS at position 2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Solve ``A Z = B`` where every operand is input-derived.

        Parameters
        ----------
        x:
            Square input tensor.

        Returns
        -------
        torch.Tensor
            Scalar reduction of the solve result.
        """

        matrix = (x @ x.transpose(-1, -2)) + 3.0 * torch.eye(3)
        lu, pivots = torch.linalg.lu_factor(matrix)
        rhs = (x + 1.0)[:, :2]
        return torch.linalg.lu_solve(lu, pivots, rhs).sum()


class _LuSolveLegacy(nn.Module):
    """Legacy ``torch.lu_solve`` with input-derived pivots at position 2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Solve through the legacy reversed-argument spelling.

        Parameters
        ----------
        x:
            Square input tensor.

        Returns
        -------
        torch.Tensor
            Scalar reduction of the solve result.
        """

        matrix = (x @ x.transpose(-1, -2)) + 3.0 * torch.eye(3)
        lu, pivots = torch.linalg.lu_factor(matrix)
        rhs = (x + 1.0)[:, :2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.lu_solve(rhs, lu, pivots).sum()


class _LuSolveBufferAmplification(nn.Module):
    """The F1 amplification: only the RHS carries input ancestry."""

    def __init__(self) -> None:
        """Register the constant coefficient matrix as a buffer."""

        super().__init__()
        self.register_buffer("coeffs", torch.eye(3) * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Solve with a buffer-derived matrix and an input-derived RHS.

        Parameters
        ----------
        x:
            Input tensor shaped (3, k).

        Returns
        -------
        torch.Tensor
            Solve result; its ONLY input ancestry flows through the RHS.
        """

        lu, pivots = torch.linalg.lu_factor(self.coeffs)
        return torch.linalg.lu_solve(lu, pivots, x + 1.0)


def test_lu_solve_third_operand_is_parent_both_spellings() -> None:
    """Both ``lu_solve`` spellings keep their position-2 tensor operand."""

    for model in (_LuSolveLinalg(), _LuSolveLegacy()):
        trace = tl.trace(model.eval(), torch.randn(3, 3))
        op = next(o for o in trace.ops if o.type == "lusolve")
        assert 2 in op.parent_arg_positions["args"], model
        assert len(op.parents) == 3, model
        assert op.unattributed_tensor_args == (), model


def test_lu_solve_input_ancestry_survives_buffer_matrix() -> None:
    """The solve op stays input-connected when ONLY the RHS is input-derived.

    Pre-fix this model's output reported ``has_input_ancestor=False`` -- a total
    input disconnect breaking ancestry, RF, and subgraph reasoning through the
    solve.
    """

    trace = tl.trace(_LuSolveBufferAmplification().eval(), torch.randn(3, 2))
    op = next(o for o in trace.ops if o.type == "lusolve")

    assert op.has_input_ancestor
    assert all(trace[label].has_input_ancestor for label in trace.output_layers)


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


class _CtcLoss(nn.Module):
    """``F.ctc_loss`` with input-derived tensor lengths."""

    def __init__(self, use_kwargs: bool) -> None:
        """Store the spelling toggle.

        Parameters
        ----------
        use_kwargs:
            Whether to pass the lengths by keyword.
        """

        super().__init__()
        self.use_kwargs = use_kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CTC loss with tensor lengths derived from the input.

        Parameters
        ----------
        x:
            Log-probability source tensor shaped (T, N, C).

        Returns
        -------
        torch.Tensor
            CTC loss value.
        """

        log_probs = F.log_softmax(x, dim=-1)
        targets = torch.ones(2, 5, dtype=torch.long)
        input_lengths = (x.abs().sum(dim=(0, 2)) * 0 + 10).to(torch.long)
        target_lengths = (x.abs().sum(dim=(0, 2)) * 0 + 5).to(torch.long)
        if self.use_kwargs:
            return F.ctc_loss(
                log_probs,
                targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def test_ctc_loss_tensor_lengths_are_parents() -> None:
    """Tensor ``input_lengths``/``target_lengths`` are parents in both spellings.

    Round-22 F4: the lengths mask/normalize the loss (value-affecting) and are
    commonly input-derived (attention-mask sums) in speech models.
    """

    x = torch.randn(10, 2, 6)
    for use_kwargs in (False, True):
        trace = tl.trace(_CtcLoss(use_kwargs=use_kwargs).eval(), x)
        op = next(o for o in trace.ops if o.type == "ctcloss")
        assert len(op.parents) == 4, f"use_kwargs={use_kwargs}"
        assert op.unattributed_tensor_args == (), f"use_kwargs={use_kwargs}"
        if not use_kwargs:
            assert {0, 1, 2, 3} <= set(op.parent_arg_positions["args"])


class _SearchSortedSorter(nn.Module):
    """``torch.searchsorted`` with an input-derived ``sorter=`` kwarg."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Search with an explicit sorter permutation.

        Parameters
        ----------
        x:
            1-D input tensor.

        Returns
        -------
        torch.Tensor
            Scalar reduction of the found indices.
        """

        sequence = x * 2.0
        values = x + 1.0
        sorter = torch.argsort(sequence)
        return torch.searchsorted(sequence, values, sorter=sorter).sum()


def test_searchsorted_sorter_kwarg_is_parent() -> None:
    """The value-affecting ``sorter=`` kwarg becomes a parent (round-22 F5)."""

    trace = tl.trace(_SearchSortedSorter().eval(), torch.randn(8))
    op = next(o for o in trace.ops if o.type == "searchsorted")

    assert "sorter" in op.parent_arg_positions["kwargs"]
    assert len(op.parents) == 3
    assert op.unattributed_tensor_args == ()


# ---------------------------------------------------------------------------
# F6: torch.tensor(existing_tensor) -- parent edge + witness authority
# ---------------------------------------------------------------------------


class _TensorOfTensor(nn.Module):
    """``torch.tensor(existing_tensor)``: legal, warned, value-copying."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebuild a tensor from an input-derived scalar tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Value flowing through the ``torch.tensor`` copy.
        """

        source = x.sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch's clone().detach() advisory
            copied = torch.tensor(source)
        return copied + 0.0


class _TensorScalarFactory(nn.Module):
    """Plain ``torch.tensor(5.0)`` factory call -- no data lineage."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a freshly constructed constant to the input.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Input plus constant.
        """

        return x + torch.tensor(5.0)


def test_torch_tensor_of_tensor_records_parent() -> None:
    """``torch.tensor(t)`` records ``t`` as a parent (data-lineage edge).

    Round-22 F6: this was the only fully SILENT drop found -- no parent AND no
    unattributed marker, because the narrower scalar-only ``aten::tensor``
    packet disarmed the witness. The contract: the parent is recorded OR the
    witness fires; both absent is never acceptable.
    """

    trace = tl.trace(_TensorOfTensor().eval(), torch.randn(3))
    op = next(o for o in trace.ops if o.type == "tensor")

    assert len(op.parents) == 1
    assert 0 in op.parent_arg_positions["args"]
    assert op.has_input_ancestor
    assert op.unattributed_tensor_args == ()


def test_torch_tensor_scalar_factory_stays_clean() -> None:
    """``torch.tensor(5.0)`` keeps zero parents and zero witness markers.

    The no-false-fire direction of the F6 fix: a plain scalar factory call has
    no tensor operand, so neither the widened spec nor the strengthened witness
    may invent an edge or a warning for it.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(_TensorScalarFactory().eval(), torch.randn(3))
    op = next(o for o in trace.ops if o.type == "tensor")

    assert op.parents == ()
    assert op.unattributed_tensor_args == ()
    assert not [w for w in caught if _PROVENANCE_MATCH in str(w.message)]


def test_runtime_tensor_control_slots_become_parents() -> None:
    """Round-31 H2: runtime tensors at schema int/Scalar control slots are parents.

    Supersedes the deleted ATen-schema operand classifier pins: the extraction
    coverage guard now records EVERY shallow runtime tensor slot as a data
    parent (``roll`` shifts, ``softmax`` dim, factory size dims), so there is
    no "schema-confirmed non-operand slot" left to suppress and the witness
    stays armed everywhere.
    """

    fallback_tensors, _ = _extract_arg_tensors_and_params(
        "roll", (torch.ones(3), torch.tensor(1)), {}
    )
    assert len(fallback_tensors) == 2

    covered_tensors, _ = _extract_arg_tensors_and_params("roll", (torch.ones(3), 1), {})
    assert len(covered_tensors) == 1


def test_spec_regression_is_healed_by_runtime_coverage_guard() -> None:
    """Mutation proof: a re-narrowed static spec can no longer drop the edge.

    Reinstalls the pre-fix under-specified factory spec (no positions) for
    ``torch.tensor``. Round-31 H2 made Tier-1 static specs subject to the
    runtime coverage guard, so the live tensor argument the narrow spec would
    have dropped is extracted through the BFS fallback anyway: the parent edge
    is recorded, no witness marker appears, and the F6 silent-drop hole cannot
    even be REACHED through a spec regression. Restores the real spec after.
    """

    real_spec = FUNC_ARG_SPECS["tensor"]
    try:
        FUNC_ARG_SPECS["tensor"] = ArgSpec()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trace = tl.trace(_TensorOfTensor().eval(), torch.randn(3))
        op = next(o for o in trace.ops if o.type == "tensor")
        assert len(op.parents) == 1
        assert op.unattributed_tensor_args == ()
        assert not [w for w in caught if _PROVENANCE_MATCH in str(w.message)]
    finally:
        FUNC_ARG_SPECS["tensor"] = real_spec
