"""r29 capture/validation hardening pins (round-32 re-attack closures).

F1 (HIGH/SECURITY): the r28 storage-rebind ancestry barrier (91fff81c) failed
open for the LAYOUT-fact family. A storage-SWAPPING ``t.data = rhs`` receiver's
own label is correctly refused by the barrier (rung 1 of
``_resolve_layout_rooting_labels``), but the fall-through dispatch-origin ledger
rung is identity-keyed and still held the PRE-rebind entry (pure ``state:``
leaf basis), which the layout ladder read as a positive state-rooted signal --
recording NO layout witness. A loaded artifact run on a same-VALUES /
different-LAYOUT input then replayed the frozen branch ``VERIFIED`` /
``poisoned=False`` while missing oracle 1 by ~96
(``runnable_tlspec_contract.md`` section 11 violation). Closed at the source,
twice over:

* ``_operand_leaf_origins`` resolves a receiver whose CURRENT label is a
  storage-rebind BARRIER label to ``unknown`` (explicit taint) instead of the
  stale ledger entry; the taint propagates through every downstream
  registration, so layout reads on PRODUCTS of the rebound tensor fail closed
  too.
* ``_resolve_layout_rooting_labels`` no longer trusts an empty (pure-state /
  literal) ledger basis for a receiver whose OWN labeled ancestry was TAINTED
  -- the identity-keyed entry may predate the taint event -- and returns
  ``None`` (caller fails closed) instead of the empty set the caller maps to
  "record nothing".

Pointer-PRESERVING rebinds never register a barrier, so the honest r85
siblings stay ``VERIFIED`` (pinned below); the r79/r81/r85 value belts are
untouched.

F3a: a dropped PARAMETER edge was invisible to the r28 per-slot identity
witness (``tensor_session_parent_labels`` returns ``()`` for ``nn.Parameter``,
the param rung of ``_tensor_has_known_provenance`` exempts it, and the value
sweep only indexes op candidates). Closed by a parameter rung in the capture
witness: a ``nn.Parameter`` arg slot must resolve to a recorded
``parent_params`` address, else it is stamped ``dropped_edge_tensor_args``.

F3b: the witness is stamped at CAPTURE, so a POSTPROCESS-stage edge drop with
a trivial payload was invisible. Closed by a post-pipeline invariant: every
capture-witnessed (slot -> producer) pair must survive into the final
``parent_arg_positions`` or be accounted for by a recorded graph rewrite.

F3c: ``recorded_parent_labels`` was a label SET, so a slot PERMUTATION between
value-identical producers passed -- corrupting the runnable call recipe.
Closed by per-slot witness granularity.

F5: the foreach sibling-slot exemption was CANDIDATE-keyed, so honest in-place
``_foreach_*_`` captures false-FAILED validation whenever a zipped member's
producer had a value-identical twin anywhere in the trace. Narrowed to
SLOT-keyed (foreach-only, exact tuple slot).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.backends.torch.ops as tlops
from torchlens.errors import PathDivergenceError, RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


# --------------------------------------------------------------------------- #
# F1 -- storage-swap ``.data=`` rebind + LAYOUT read must never false-VERIFY.
# --------------------------------------------------------------------------- #


class _SwapRebindLayoutRead(nn.Module):
    """Storage-SWAPPING ``.data=`` rebind whose receiver steers a layout branch."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.ones(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b * 1.0
        y.data = (x * 1.0).detach()  # storage SWAP -> ancestry barrier registered
        k = 2.0 if y.is_contiguous() else 5.0  # LAYOUT read on the rebound receiver
        return x.sum() * k


class _SwapRebindDownstreamLayoutRead(nn.Module):
    """Layout read on a PRODUCT of the rebound receiver (propagated staleness)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.ones(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b * 1.0
        y.data = (x * 1.0).detach()
        z = y * 1.0  # downstream product inherits the rebound value DAG
        k = 2.0 if z.is_contiguous() else 5.0
        return x.sum() * k


class _PointerPreservingRebindLayoutRead(nn.Module):
    """Honest pointer-PRESERVING ``.data=`` rebind (no barrier) + layout read."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.ones(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b * 1.0
        y.data = y.view(2, 4, 8, 8)  # same storage: never a barrier
        k = 2.0 if y.is_contiguous() else 5.0
        return x.sum() * k


def _layout_twin(x: torch.Tensor) -> torch.Tensor:
    """Return a same-VALUES / different-LAYOUT (channels_last) twin of ``x``."""

    twin = x.clone().to(memory_format=torch.channels_last)
    assert torch.equal(x, twin) and not twin.is_contiguous()
    return twin


def _save_and_run(
    model: nn.Module, capture_input: torch.Tensor, run_input: torch.Tensor, tmp_path: Path
) -> Any:
    """Capture, runnable-save, load, and run; save refusal is a valid fail-closed form."""

    path = tmp_path / "art.tlspec"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, capture_input, capture=_CAPTURE)
        trace.save(path, level="runnable", include_weights=True)
        return tl.load(path).run(inputs=run_input.clone())


def _assert_layout_launder_closed(model_factory: Any, tmp_path: Path) -> None:
    """Assert the layout-twin replay of a swap-rebind vehicle FAILS CLOSED.

    Accepted fail-closed forms: save refusal, ``PathDivergenceError``, or a
    non-``VERIFIED`` poisoned run. A ``VERIFIED``/unpoisoned verdict is the
    reopened launder regardless of the numeric outcome (the frozen branch
    disagrees with oracle 1 by ~96 on this vehicle).
    """

    torch.manual_seed(0)
    capture_input = torch.randn(2, 4, 8, 8)
    twin = _layout_twin(capture_input)
    oracle = model_factory().eval()(twin.clone())
    try:
        result = _save_and_run(model_factory().eval(), capture_input, twin, tmp_path)
    except (RunnablePreflightError, PathDivergenceError):
        return  # fail-closed at save or at divergence: both honest
    max_diff = (result.output - oracle).abs().max().item()
    assert not (
        result.report.path_faithfulness is PathFaithfulness.VERIFIED and not result.report.poisoned
    ), f"FALSE VERIFIED reopened: layout twin missed oracle 1 by {max_diff}"
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.poisoned


def test_f1_swap_rebind_layout_read_ceils(tmp_path: Path) -> None:
    """The sec1 vehicle: swap-rebind receiver layout read ceils on a layout twin."""

    _assert_layout_launder_closed(_SwapRebindLayoutRead, tmp_path)


def test_f1_swap_rebind_downstream_layout_read_ceils(tmp_path: Path) -> None:
    """The propagated vehicle: layout read on a PRODUCT of the rebound tensor ceils."""

    _assert_layout_launder_closed(_SwapRebindDownstreamLayoutRead, tmp_path)


def test_f1_pointer_preserving_rebind_stays_verified(tmp_path: Path) -> None:
    """r85 sibling guard: an honest same-storage rebind is NOT over-ceiled."""

    torch.manual_seed(0)
    capture_input = torch.randn(2, 4, 8, 8)
    result = _save_and_run(
        _PointerPreservingRebindLayoutRead().eval(), capture_input, capture_input, tmp_path
    )
    oracle = _PointerPreservingRebindLayoutRead().eval()(capture_input.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert (result.output - oracle).abs().max().item() < 1e-6


# --------------------------------------------------------------------------- #
# F3a -- a dropped PARAMETER edge must be witnessed.
# --------------------------------------------------------------------------- #


class _LinearModel(nn.Module):
    """One linear layer whose weight/bias params feed the wrapped call."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


class _ConvModel(nn.Module):
    """One conv layer whose weight/bias params feed the wrapped call."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).relu()


@pytest.mark.parametrize(
    "model_cls,func_name,x",
    [
        (_LinearModel, "linear", torch.randn(4, 5)),
        (_ConvModel, "conv2d", torch.randn(1, 2, 6, 6)),
    ],
    ids=["linear", "conv2d"],
)
@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments:UserWarning")
def test_f3a_dropped_param_edge_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[nn.Module],
    func_name: str,
    x: torch.Tensor,
) -> None:
    """An injected param-detection failure must be caught by the witness.

    Emptying the ``arg_parameters`` list ``_build_param_fields`` receives makes
    the whole pipeline behave exactly as if capture had failed to detect the
    call's parameters (``num_params == 0``, empty ``parent_params``). The
    session-validated Parameters still sit in the live args, so the r29 F3a
    parameter rung must stamp them as dropped edges and public validation must
    fail.
    """

    original = tlops._build_param_fields
    state = {"dropped": 0}

    def patched(self: Any, fields_dict: dict, arg_parameters: list) -> dict:
        """Empty the detected-parameter list for the targeted call."""

        if fields_dict.get("func_name") == func_name and arg_parameters:
            state["dropped"] += len(arg_parameters)
            arg_parameters = []
        return original(self, fields_dict, arg_parameters)

    monkeypatch.setattr(tlops, "_build_param_fields", patched)
    torch.manual_seed(0)
    result = tl.validate(model_cls().eval(), x, scope="forward")
    assert state["dropped"] > 0, "injection did not drop any parameter -- inconclusive"
    assert not bool(result)


@pytest.mark.parametrize(
    "model_cls,func_name,x",
    [
        (_LinearModel, "linear", torch.randn(4, 5)),
        (_ConvModel, "conv2d", torch.randn(1, 2, 6, 6)),
    ],
    ids=["linear", "conv2d"],
)
def test_f3a_honest_param_capture_stays_clean(
    model_cls: type[nn.Module], func_name: str, x: torch.Tensor
) -> None:
    """The param rung never fires on an honest capture (no false positive)."""

    torch.manual_seed(0)
    trace = tl.trace(model_cls().eval(), x)
    op = next(o for o in trace.ops if o.func_name == func_name)
    assert tuple(getattr(op, "dropped_edge_tensor_args", ()) or ()) == ()
    assert op.num_params > 0
    assert bool(tl.validate(model_cls().eval(), x, scope="forward"))


# --------------------------------------------------------------------------- #
# F3c -- a slot PERMUTATION between value-identical producers must fail.
# --------------------------------------------------------------------------- #


class _TwinSub(nn.Module):
    """Non-commutative op between value-identical, distinct producers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(x)
        b = torch.relu(x + 0)
        y = a - b
        return y + a.sum() * 0.001 + b.sum() * 0.002


@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments:UserWarning")
def test_f3c_slot_permutation_fails_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swapping two recorded slots between value-identical producers must fail.

    ``recorded_parent_labels`` was a label SET, so a permuted
    ``parent_arg_positions`` -- which corrupts the runnable call recipe for a
    non-commutative op -- validated silently. The r29 F3c per-slot identity
    witness must stamp both slots and fail public validation.
    """

    original = tlops._build_graph_relationship_fields
    state = {"permuted": 0}

    def patched(
        self: Any,
        fields_dict: dict,
        parent_layer_labels: list,
        parent_layer_entries: list,
        args: tuple,
        kwargs: dict,
        out_orig: Any,
    ) -> Any:
        """Swap the two recorded positional parent slots on the target op."""

        result = original(
            self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
        )
        if fields_dict.get("func_name") == "__sub__":
            pos = fields_dict["parent_arg_positions"]["args"]
            if 0 in pos and 1 in pos and pos[0] != pos[1]:
                pos[0], pos[1] = pos[1], pos[0]
                state["permuted"] += 1
        return result

    monkeypatch.setattr(tlops, "_build_graph_relationship_fields", patched)
    torch.manual_seed(0)
    x = torch.randn(4)
    trace = tl.trace(_TwinSub().eval(), x)
    sub_op = next(o for o in trace.ops if o.func_name == "__sub__")
    assert state["permuted"] > 0, "injection did not permute any slot -- inconclusive"
    assert tuple(sub_op.dropped_edge_tensor_args or ()) != ()
    assert not bool(tl.validate(_TwinSub().eval(), x, scope="forward"))


def test_f3c_honest_twin_capture_stays_clean() -> None:
    """Value-identical twin producers with correct slots never fire the witness."""

    torch.manual_seed(0)
    x = torch.randn(4)
    trace = tl.trace(_TwinSub().eval(), x)
    sub_op = next(o for o in trace.ops if o.func_name == "__sub__")
    assert tuple(getattr(sub_op, "dropped_edge_tensor_args", ()) or ()) == ()
    assert bool(tl.validate(_TwinSub().eval(), x, scope="forward"))


# --------------------------------------------------------------------------- #
# F3b -- a POSTPROCESS-stage edge drop (trivial payload) must be caught.
# --------------------------------------------------------------------------- #


class _MulTrivialProducer(nn.Module):
    """Consumer of an all-ones producer -- the value sweep's blind class."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x * 0 + 1
        return x.relu() * w + w.sum()


def test_f3b_postprocess_edge_drop_fails_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    """An edge dropped AFTER the capture witness must fail metadata invariants.

    The witness (``dropped_edge_tensor_args``) is stamped at capture, so a
    symmetric edge removal after the postprocess pipeline -- ``parents``,
    ``parent_arg_positions``, and the producer's ``children`` all edited
    together -- left validation True whenever the payload was trivial. The r29
    F3b ``capture_edge_survival`` invariant reconciles the final graph against
    the sealed capture-time edge truth and must raise.
    """

    import torchlens.postprocess as pp
    from torchlens.validation.invariants import (
        MetadataInvariantError,
        check_metadata_invariants,
    )

    original = pp.postprocess
    state = {"dropped": 0}

    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Drop the trivial-valued parent edge after the full pipeline runs."""

        result = original(self, *args, **kwargs)
        target = next(
            (o for o in self.layer_list if o.func_name == "__mul__" and len(o.parents) >= 2),
            None,
        )
        if target is None:
            return result
        victims = [p for p in target.parents if str(p).startswith("add")]
        if not victims:
            return result
        victim = victims[0]
        target.parents = [p for p in target.parents if p != victim]
        for domain in ("args", "kwargs"):
            positions = target.parent_arg_positions.get(domain, {})
            for key in [k for k, v in positions.items() if v == victim]:
                del positions[key]
        producer = next((o for o in self.layer_list if o.layer_label == victim), None)
        if producer is not None:
            producer.children = [c for c in producer.children if c != target.layer_label]
        state["dropped"] += 1
        return result

    monkeypatch.setattr(pp, "postprocess", patched)
    torch.manual_seed(0)
    trace = tl.trace(
        _MulTrivialProducer().eval(),
        torch.randn(4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    assert state["dropped"] > 0, "injection did not drop any edge -- inconclusive"
    # Two independent tripwires catch this plant: the capture-sealed
    # edge-survival witness (the original pin) and, since the R08/R75
    # multiplicity witness landed, edge_use_parent_arg_consistency -- the
    # post-freeze drop leaves the CSR occurrence in place while the
    # parents/parent_arg_positions roots lose it, so the count cross-check
    # fires first in dispatch order. Either name proves the drop is caught;
    # the capture-sealed witness staying independently armed is pinned by
    # the CSR-scrubbed plants in test_oracle_independence.py.
    with pytest.raises(
        MetadataInvariantError,
        match="capture_edge_survival|edge_use_parent_arg_consistency",
    ):
        check_metadata_invariants(trace)


def test_f3b_honest_capture_passes_invariants() -> None:
    """The capture-edge-survival invariant never fires on an honest capture."""

    from torchlens.validation.invariants import check_metadata_invariants

    torch.manual_seed(0)
    trace = tl.trace(
        _MulTrivialProducer().eval(),
        torch.randn(4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    assert check_metadata_invariants(trace)


# --------------------------------------------------------------------------- #
# r33 F-1 -- capture_edge_survival must reconcile PER-SLOT, not by label-set
# membership. Two postprocess-stage corruptions passed the set reduction: a
# slot PERMUTATION between value-identical surviving producers (P1) and an
# argpos entry DROP with parents intact (P2).
# --------------------------------------------------------------------------- #


def _final_twin_sub_trace() -> Any:
    """Return a postprocessed _TwinSub trace plus its final ``__sub__`` op."""

    torch.manual_seed(0)
    trace = tl.trace(
        _TwinSub().eval(),
        torch.randn(4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    sub_op = next(o for o in trace.ops if o.func_name == "__sub__")
    return trace, sub_op


def test_f1_final_slot_permutation_fails_invariants() -> None:
    """P1: permuting final argpos slots between value-identical survivors raises.

    Both producers survive and both labels stay present on the op, so the old
    label-SET reduction passed; the sealed per-record triplets pin each label
    to its exact slot.
    """

    from torchlens.validation.invariants import (
        MetadataInvariantError,
        check_metadata_invariants,
    )

    trace, sub_op = _final_twin_sub_trace()
    positions = sub_op.parent_arg_positions["args"]
    slots = sorted(positions, key=str)
    assert len(slots) >= 2, "twin consumer lost its two positional slots -- inconclusive"
    first, second = slots[0], slots[1]
    assert positions[first] != positions[second]
    positions[first], positions[second] = positions[second], positions[first]
    with pytest.raises(MetadataInvariantError, match="capture_edge_survival"):
        check_metadata_invariants(trace)


def test_f1_final_argpos_entry_drop_fails_invariants() -> None:
    """P2: dropping one final argpos entry with parents intact raises.

    The producer label survives via ``parents`` (and the producer itself
    survives), so the old set reduction passed while the user-facing slot
    metadata silently lost an edge.
    """

    from torchlens.validation.invariants import (
        MetadataInvariantError,
        check_metadata_invariants,
    )

    trace, sub_op = _final_twin_sub_trace()
    positions = sub_op.parent_arg_positions["args"]
    assert len(positions) >= 2, "twin consumer lost its two positional slots -- inconclusive"
    victim_slot = sorted(positions, key=str)[0]
    victim_label = positions[victim_slot]
    del positions[victim_slot]
    assert victim_label in sub_op.parents, "parents no longer hold the label -- inconclusive"
    with pytest.raises(MetadataInvariantError, match="capture_edge_survival"):
        check_metadata_invariants(trace)


# --------------------------------------------------------------------------- #
# F5 -- honest in-place foreach captures must validate.
# --------------------------------------------------------------------------- #


class _ForeachAddInplace(nn.Module):
    """Honest in-place ``_foreach_add_`` whose members' producers have twins."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.relu().clone()
        b = x.sigmoid().clone()
        torch._foreach_add_([a, b], 1.0)
        return a + b


class _ForeachMulThree(nn.Module):
    """Three-member in-place ``_foreach_mul_``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.relu().clone()
        b = x.sigmoid().clone()
        c = x.tanh().clone()
        torch._foreach_mul_([a, b, c], 2.0)
        return a + b + c


class _ForeachOptimStyleChain(nn.Module):
    """Fused-optimizer-style ``_foreach_mul_`` then ``_foreach_add_`` chain."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        members = [x.relu().clone(), x.sigmoid().clone()]
        torch._foreach_mul_(members, 0.9)
        torch._foreach_add_(members, 0.1)
        return members[0] + members[1]


class _ForeachCopyInplace(nn.Module):
    """In-place ``_foreach_copy_`` (structural destination, tested sources)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.relu().clone()
        b = x.sigmoid().clone()
        torch._foreach_copy_([a, b], [x.tanh().clone(), x.cos().clone()])
        return a + b


@pytest.mark.parametrize(
    "model_cls",
    [_ForeachAddInplace, _ForeachMulThree, _ForeachOptimStyleChain, _ForeachCopyInplace],
    ids=["add-2", "mul-3", "optim-chain", "copy"],
)
def test_f5_honest_inplace_foreach_validates(model_cls: type[nn.Module]) -> None:
    """Honest in-place foreach captures must not false-fail validation.

    Every member producer is a ``clone`` -- guaranteeing a value-identical twin
    op elsewhere in the trace, the exact trigger of the candidate-keyed
    exemption's false failure (r29 F5). The sibling-owned-slot exemption is
    slot-keyed, so these validate while a genuinely dropped zipped edge (no
    sibling attributes the slot) still fails -- pinned by
    ``test_m3_dropped_zipped_edge_still_fails_validation``.
    """

    torch.manual_seed(0)
    x = torch.randn(4)
    assert bool(tl.validate(model_cls().eval(), x, scope="forward"))
