"""Forward validation keeps a saved argument identical to its parent's saved output by reference.

Validation captures with ``save_args=True``. Inside that one capture (first
attempt, ``output_device="same"``), a retained tensor argument whose recorded
parent op already saved the bit-identical value is saved as a detached alias of
that saved output instead of a second copy. These tests pin, on the CPU:

* a fan-out model holds fewer distinct saved-argument storages and bytes than
  the copy-only path, and reaches the same verdict;
* a planted wrong-parent swap to a producer with different values keeps the
  copy at that slot and still fails;
* an in-place write after the parent was saved keeps the copy, with the
  copy-only verdict, and still fails when its per-child version record is
  dropped (a planted capture bug);
* replay never writes through an alias (versions and bytes are unchanged after
  validation, with in-place ops consuming aliased arguments), and a write into
  an alias during the capture makes validation rerun with every argument copied,
  also when the write skips the version counter (``.data``, a raw storage write);
* view-context receivers, conj and neg bits and tensor subclasses are never
  aliased, and an exception while aliasing falls back to the copy;
* with grad disabled, where the parent's saved output is still a pending
  clone-on-write alias of the live tensor, the argument's alias is registered
  under the same key: a no-grad fan-out holds fewer saved-argument storages, an
  in-place write after the save copies the pre-write bytes out of both, the
  storage-alias label scan never sees the alias and the graph is unchanged, the
  planted negatives still fail, and a write into the alias reruns validation,
  also when it skips the version counter.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

import pytest
import torch
from torch import nn

import torchlens._user_funcs as user_funcs
import torchlens.backends.torch._ops_emission
import torchlens.backends.torch._ops_exhaustive
import torchlens.backends.torch._ops_predicate_events
import torchlens.backends.torch._ops_shared_fields
import torchlens.backends.torch.wrappers
from torchlens import _state, _user_public_impls, _validation_argument_references as arg_references
from torchlens.backends.torch import _ops_saved_arg_copies as saved_arg_copies
from torchlens.validation._diagnostics import get_validation_diagnostics

# These tests assert that payloads defer, which no capture does once Dynamo is loaded.
pytestmark = pytest.mark.usefixtures("dynamo_free_process")


def _validate(
    model: nn.Module, x: torch.Tensor, observe: Callable[[Any], None], *, copy_only: bool = False
) -> bool:
    """Validate ``model`` on ``x``; ``copy_only`` runs the path with every argument copied."""

    validate: Callable[..., Any] = _user_public_impls._validate_forward_pass_torch
    if copy_only:
        validate = validate.__wrapped__  # type: ignore[attr-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate(model, x, seed=0, _trace_observer=observe))


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield the tensor leaves of a saved-argument tree."""

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _shares_parent_output(value: torch.Tensor, parent: Any) -> bool:
    """Whether a saved argument sits on its parent's saved-output storage."""

    out = getattr(parent, "out", None)
    return isinstance(out, torch.Tensor) and _ptr(value) == _ptr(out)


def _ops_by_label(trace: Any) -> dict[str, Any]:
    """Index a validation trace's ops by every label spelling they carry."""

    index: dict[str, Any] = {}
    for op in trace._layer_list:
        for attr in ("label", "layer_label", "_label_raw"):
            label = getattr(op, attr, None)
            if isinstance(label, str):
                index[label] = op
    return index


def _slot_value(op: Any, domain: str, location: Any) -> Any:
    """Read the saved argument at one parent slot of ``op``."""

    call = op.call
    container = list(call.saved_args or ()) if domain == "args" else dict(call.saved_kwargs or {})
    path = location if isinstance(location, tuple) else (location,)
    value: Any = container
    for key in path:
        value = value[key]
    return value


def _parent_slots(trace: Any) -> list[tuple[Any, Any, torch.Tensor]]:
    """Return ``(op, parent op, saved argument)`` for every saved tensor at a parent slot."""

    by_label = _ops_by_label(trace)
    slots = []
    for op in trace._layer_list:
        positions = op.call.parent_arg_positions or {}
        for domain in ("args", "kwargs"):
            for location, label in (positions.get(domain) or {}).items():
                parent = by_label.get(label)
                try:
                    value = _slot_value(op, domain, location)
                except (IndexError, KeyError, TypeError):
                    continue
                if parent is not None and isinstance(value, torch.Tensor):
                    slots.append((op, parent, value))
    return slots


def _saved_arg_only_storage(trace: Any, model: nn.Module) -> tuple[int, int]:
    """Count distinct saved-argument storages, and their bytes, that no saved output holds."""

    held = {_ptr(op.out) for op in trace._layer_list if isinstance(op.out, torch.Tensor)}
    held |= {_ptr(p) for p in model.parameters()}
    storages: dict[int, int] = {}
    for op in trace._layer_list:
        for tensor in _iter_tensors((op.call.saved_args, op.call.saved_kwargs)):
            ptr = _ptr(tensor)
            if ptr and ptr not in held:
                storages[ptr] = tensor.untyped_storage().nbytes()
    return len(storages), sum(storages.values())


class _FanOut(nn.Module):
    """One activation consumed by several ops, a flat list and an in-place receiver."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(16, 16)
        self.proj = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.lin(x))
        doubled = hidden * 2.0
        shifted = hidden + 1.0
        joined = torch.cat([doubled, shifted, hidden], dim=1)
        mixed = hidden @ self.proj.weight
        clipped = mixed.clamp(-1.0, 1.0)
        clipped.mul_(0.5)
        return joined.sum(dim=1, keepdim=True) + clipped.tanh().sum(dim=1, keepdim=True)


class _Different(nn.Module):
    """Two producers with different values; the mul consumes the first."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = torch.relu(x)
        second = torch.relu(-x)
        return first * 3.0 + second


class _UntracedWrite(nn.Module):
    """Writes an activation in place, unseen by the capture, after it was saved."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.lin(x))
        with _state.pause_logging(), torch.no_grad():
            hidden.add_(1.0)
        return hidden * 2.0


def test_fan_out_holds_fewer_saved_argument_bytes_with_the_same_verdict() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    counts: dict[str, tuple[int, int]] = {}
    aliased: list[int] = []

    def observe_copy(trace: Any) -> None:
        counts["copy"] = _saved_arg_only_storage(trace, model)

    def observe_alias(trace: Any) -> None:
        counts["alias"] = _saved_arg_only_storage(trace, model)
        aliased.append(
            sum(_shares_parent_output(value, parent) for _, parent, value in _parent_slots(trace))
        )

    torch.manual_seed(0)
    model = _FanOut()
    copy_verdict = _validate(model, x, observe_copy, copy_only=True)
    alias_verdict = _validate(model, x, observe_alias)
    assert copy_verdict is True
    assert alias_verdict is copy_verdict
    assert aliased and aliased[0] >= 5, aliased
    copy_storages, copy_bytes = counts["copy"]
    alias_storages, alias_bytes = counts["alias"]
    assert alias_storages < copy_storages, counts
    assert alias_bytes < copy_bytes, counts


def test_wrong_parent_swap_to_different_values_keeps_the_copy_and_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torchlens.backends.torch._ops_predicate_events._build_graph_relationship_fields
    state = {"hits": 0}

    def patched(
        self: Any,
        fields_dict: dict[str, Any],
        parent_layer_labels: Any,
        parent_layer_entries: Any,
        args: Any,
        kwargs: Any,
        out_orig: Any,
    ) -> Any:
        """Record the mul's parent as the second relu, whose values differ."""

        result = original(
            self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
        )
        if fields_dict.get("func_name") != "__mul__":
            return result
        live = self.capture_events.live_index.by_raw_label
        replacements = [label for label in live if label.startswith("relu_2")]
        if not replacements:
            return result
        parents = fields_dict.get("parents") or []
        for index, label in enumerate(list(parents)):
            if label.startswith("relu_1"):
                parents[index] = replacements[0]
                state["hits"] += 1
        positions = fields_dict.get("parent_arg_positions") or {}
        for domain in ("args", "kwargs"):
            for key, label in list((positions.get(domain) or {}).items()):
                if label.startswith("relu_1"):
                    positions[domain][key] = replacements[0]
        return result

    for module in (
        torchlens.backends.torch._ops_predicate_events,
        torchlens.backends.torch._ops_exhaustive,
        torchlens.backends.torch._ops_shared_fields,
    ):
        monkeypatch.setattr(module, "_build_graph_relationship_fields", patched)
    shared: list[bool] = []

    def observe(trace: Any) -> None:
        for op, parent, value in _parent_slots(trace):
            if op.func_name == "__mul__":
                shared.append(_shares_parent_output(value, parent))

    torch.manual_seed(0)
    x = torch.randn(6)
    alias_verdict = _validate(_Different(), x, observe)
    assert state["hits"] > 0, "injection never swapped a parent: inconclusive"
    assert alias_verdict is False
    assert shared and not any(shared), shared
    assert _validate(_Different(), x, lambda trace: None, copy_only=True) is False


def _mul_slot_shares_parent(shared: list[bool]) -> Callable[[Any], None]:
    """Observer recording whether the mul's saved argument sits on its parent's output."""

    def observe(trace: Any) -> None:
        for op, parent, value in _parent_slots(trace):
            if op.func_name == "__mul__":
                shared.append(_shares_parent_output(value, parent))

    return observe


def test_an_in_place_write_after_save_is_kept_as_a_copy_with_the_same_verdict() -> None:
    torch.manual_seed(0)
    model = _UntracedWrite()
    x = torch.randn(3, 4)
    shared: list[bool] = []
    copy_verdict = _validate(model, x, lambda trace: None, copy_only=True)
    assert _validate(model, x, _mul_slot_shares_parent(shared)) is copy_verdict
    assert shared and not any(shared), shared


def test_an_in_place_write_after_save_with_its_version_record_dropped_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped: list[str] = []

    def drop_version(self: Any, parent_label: str, *args: Any, **kwargs: Any) -> None:
        """Planted capture bug: never record the value a child consumed after a write."""

        dropped.append(parent_label)

    monkeypatch.setattr(
        torchlens.backends.torch._ops_shared_fields,
        "_get_parent_output_version_snapshot",
        drop_version,
    )
    torch.manual_seed(0)
    model = _UntracedWrite()
    x = torch.randn(3, 4)
    shared: list[bool] = []
    assert _validate(model, x, lambda trace: None, copy_only=True) is False
    assert _validate(model, x, _mul_slot_shares_parent(shared)) is False
    assert dropped, "the planted bug never ran: inconclusive"
    assert shared and not any(shared), shared


def test_replay_never_writes_through_an_alias_to_the_parents_saved_output() -> None:
    torch.manual_seed(0)
    model = _FanOut()
    x = torch.randn(8, 16)
    copy_values: dict[tuple[str, str], torch.Tensor] = {}
    checked: list[tuple[str, str]] = []
    funcs: set[str] = set()

    def observe_copy(trace: Any) -> None:
        for op, parent, value in _parent_slots(trace):
            copy_values[(op.layer_label, parent.layer_label)] = value.detach().clone()

    def observe_alias(trace: Any) -> None:
        for op, parent, value in _parent_slots(trace):
            if not _shares_parent_output(value, parent):
                continue
            key = (op.layer_label, parent.layer_label)
            assert torch.equal(value.detach(), copy_values[key]), key
            assert torch.equal(parent.out.detach(), copy_values[key]), key
            checked.append(key)
            funcs.add(op.func_name)

    assert _validate(model, x, observe_copy, copy_only=True) is True
    assert _validate(model, x, observe_alias) is True
    assert "mul_" in funcs, funcs
    assert len(checked) >= 5, checked


def test_a_write_into_an_alias_during_capture_reruns_with_every_argument_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = arg_references.ParameterArgumentReferences.value_alias
    written: list[str] = []

    def write_first_alias(
        self: arg_references.ParameterArgumentReferences,
        tensor: torch.Tensor,
        saved_value: Any,
        parent_label: str,
    ) -> torch.Tensor | None:
        alias = original(self, tensor, saved_value, parent_label)
        if alias is not None and not written:
            with torch.no_grad():
                alias.add_(1.0)
            written.append(parent_label)
        return alias

    monkeypatch.setattr(
        arg_references.ParameterArgumentReferences, "value_alias", write_first_alias
    )
    diagnostics: list[Any] = []

    def observe(trace: Any) -> None:
        diagnostics.extend(get_validation_diagnostics(trace))

    torch.manual_seed(0)
    model = _FanOut()
    x = torch.randn(8, 16)
    verdict = _validate(model, x, observe)
    assert written, "no alias was taken: inconclusive"
    assert verdict is True
    fallbacks = [d for d in diagnostics if d.check_name == arg_references.FALLBACK_CHECK]
    assert len(fallbacks) == 1, diagnostics
    assert "saved output" in fallbacks[0].extra["reason"]


class _NoGradChain(nn.Module):
    """An integer chain (its saved outputs stay clone-on-write) written after a child read it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = x.argmax(dim=1)
        parity = index.remainder(2)
        shifted = parity + 1
        parity.mul_(3)
        return x.sum(dim=1) * shifted + parity


class _StorageContextReceiver(nn.Module):
    """A storage-context receiver whose parent saved the identical bytes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x * 1.0
        scaled.as_strided_((2, 2), (4, 1))
        return scaled + 1.0


def _saved_slot_values(trace: Any, func_name: str) -> list[tuple[torch.Tensor, bool]]:
    """Return each saved tensor at a parent slot of ``func_name`` ops, and whether it is aliased."""

    return [
        (value.detach().clone(), _shares_parent_output(value, parent))
        for op, parent, value in _parent_slots(trace)
        if op.func_name == func_name
    ]


def _slot_values_by_path(
    model_type: type[nn.Module], func_name: str, shape: tuple[int, ...]
) -> tuple[bool, dict[str, list[tuple[torch.Tensor, bool]]]]:
    """Validate on the copy-only path and the alias path; return the verdict and slot values."""

    torch.manual_seed(0)
    x = torch.randn(*shape)
    seen: dict[str, list[tuple[torch.Tensor, bool]]] = {}

    def observe(key: str) -> Callable[[Any], None]:
        def record(trace: Any) -> None:
            seen[key] = _saved_slot_values(trace, func_name)

        return record

    copy_verdict = _validate(model_type(), x, observe("copy"), copy_only=True)
    assert _validate(model_type(), x, observe("alias")) is copy_verdict
    assert seen["alias"], seen
    assert len(seen["alias"]) == len(seen["copy"])
    for (alias_value, _), (copy_value, _) in zip(seen["alias"], seen["copy"], strict=True):
        assert torch.equal(alias_value, copy_value)
    return copy_verdict, seen


def test_a_storage_context_receiver_keeps_its_copy_with_the_same_verdict() -> None:
    _, seen = _slot_values_by_path(_StorageContextReceiver, "as_strided_", (4, 4))
    assert not any(aliased for _, aliased in seen["alias"]), seen["alias"]


def test_a_pending_parent_read_before_an_in_place_write_keeps_its_pre_write_bytes() -> None:
    _, seen = _slot_values_by_path(_NoGradChain, "__add__", (3, 5))
    # ``parity + 1`` read parity before ``parity.mul_(3)``: its alias was copied out with
    # the pre-write bytes. The final add's slots were never written after, so stay aliased.
    (first, first_aliased), *rest = seen["alias"]
    assert not first_aliased
    assert torch.equal(first, seen["alias"][-1][0] // 3)
    assert rest and all(aliased for _, aliased in rest), seen["alias"]


def test_a_grad_alias_is_a_non_leaf_that_takes_the_clones_in_place_write() -> None:
    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    source = torch.randn(4, requires_grad=True)
    consumed = source * 1.0
    saved = consumed.clone()
    with _state.pause_logging():
        alias = guard.value_alias(consumed.clone(), saved, "p")
    assert alias is not None
    assert alias.requires_grad and not alias.is_leaf
    assert alias.untyped_storage().data_ptr() == saved.untyped_storage().data_ptr()
    assert guard.first_change() is None
    alias.add_(1.0)
    assert guard.first_change() is not None


def test_the_guard_refuses_unequal_bytes_signed_zero_and_partial_storage() -> None:
    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    saved = torch.tensor([0.0, 1.0, float("nan")])
    with _state.pause_logging():
        assert guard.value_alias(saved.clone(), saved, "p") is not None
        assert guard.value_alias(torch.tensor([-0.0, 1.0, float("nan")]), saved, "p") is None
        assert guard.value_alias(torch.tensor([0.0, 2.0, float("nan")]), saved, "p") is None
        assert guard.value_alias(saved[:2].clone(), saved[:2], "p") is None
        assert guard.value_alias(saved.clone().double(), saved, "p") is None
    with _state.pause_logging():
        detached = guard.value_alias(saved.clone(), saved, "p")
    assert detached is not None and not detached.requires_grad
    off = arg_references.ParameterArgumentReferences({}, alias_saved_values=False)
    with _state.pause_logging():
        assert off.value_alias(saved.clone(), saved, "p") is None
    assert guard.first_change() is None


# ---------------------------------------------------------------------------
# Parents saved with grad disabled: their saved output is still a pending
# clone-on-write alias of the live tensor, so the argument's alias is
# registered under the same clone-on-write key.
# ---------------------------------------------------------------------------


class _NoGradFanOut(nn.Module):
    """A no-grad fan-out; ``write_after`` scales the shared activation in place afterwards."""

    def __init__(self, write_after: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(16, 16)
        self.write_after = write_after

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = torch.relu(self.lin(x))
            doubled = hidden * 2.0
            shifted = hidden + 1.0
            joined = torch.cat([doubled, shifted, hidden], dim=1)
            if self.write_after:
                hidden.mul_(0.5)
            return joined.sum(dim=1, keepdim=True) + hidden.sum(dim=1, keepdim=True)


def _count_pending_aliases(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Record each argument alias registered under its parent's clone-on-write key."""

    original = arg_references.ParameterArgumentReferences._pending_value_alias
    taken: list[torch.Tensor] = []

    def counted(self: Any, tensor: Any, saved_value: Any, parent_label: str, *rest: Any) -> Any:
        alias = original(self, tensor, saved_value, parent_label, *rest)
        if alias is not None:
            taken.append(alias)
        return alias

    monkeypatch.setattr(arg_references.ParameterArgumentReferences, "_pending_value_alias", counted)
    return taken


def _graph(trace: Any) -> list[tuple[Any, ...]]:
    """Every op's label, function, parents and children, in capture order."""

    return [
        (
            op.layer_label,
            op.func_name,
            tuple(getattr(op, "parents", ()) or ()),
            tuple(getattr(op, "children", ()) or ()),
        )
        for op in trace._layer_list
    ]


def test_a_no_grad_fan_out_holds_fewer_saved_argument_storages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taken = _count_pending_aliases(monkeypatch)
    counts: dict[str, tuple[int, int]] = {}
    graphs: dict[str, list[tuple[Any, ...]]] = {}

    def observe(key: str) -> Callable[[Any], None]:
        def record(trace: Any) -> None:
            counts[key] = _saved_arg_only_storage(trace, model)
            graphs[key] = _graph(trace)

        return record

    torch.manual_seed(0)
    model = _NoGradFanOut(write_after=False)
    x = torch.randn(8, 16)
    copy_verdict = _validate(model, x, observe("copy"), copy_only=True)
    assert not taken
    alias_verdict = _validate(model, x, observe("alias"))
    assert copy_verdict is True
    assert alias_verdict is copy_verdict
    assert len(taken) >= 4, taken
    assert counts["alias"][0] < counts["copy"][0], counts
    assert counts["alias"][1] < counts["copy"][1], counts
    assert graphs["alias"] == graphs["copy"]


def test_an_in_place_write_after_save_copies_the_pre_write_bytes_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taken = _count_pending_aliases(monkeypatch)
    values: dict[str, dict[tuple[str, str], torch.Tensor]] = {}
    graphs: dict[str, list[tuple[Any, ...]]] = {}
    original_scan = torchlens.backends.torch._ops_emission.overlapping_labeled_aliases
    scans: list[list[Any]] = []

    def scan(mutated: torch.Tensor, exclude_label: str | None) -> list[tuple[torch.Tensor, str]]:
        """Record what the storage-alias label scan finds for each in-place write."""

        found = original_scan(mutated, exclude_label)
        scans.append([alias for alias, _ in found])
        return found

    monkeypatch.setattr(torchlens.backends.torch._ops_emission, "overlapping_labeled_aliases", scan)

    diagnostics: list[Any] = []
    copied_out: list[bool] = []

    def observe(key: str) -> Callable[[Any], None]:
        def record(trace: Any) -> None:
            graphs[key] = _graph(trace)
            values[key] = {
                (op.layer_label, parent.layer_label): value.detach().clone()
                for op, parent, value in _parent_slots(trace)
            }
            if key == "alias":
                diagnostics.extend(get_validation_diagnostics(trace) or ())
                # The in-place receiver itself is saved after the write copied the payload
                # out, so it takes the exclusive-copy path; the three readers were registered.
                copied_out.extend(
                    not _shares_parent_output(value, parent)
                    for op, parent, value in _parent_slots(trace)
                    if parent.layer_label.startswith("relu") and op.func_name != "mul_"
                )

        return record

    torch.manual_seed(0)
    model = _NoGradFanOut(write_after=True)
    x = torch.randn(8, 16)
    copy_verdict = _validate(model, x, observe("copy"), copy_only=True)
    alias_verdict = _validate(model, x, observe("alias"))
    assert copy_verdict is True
    assert alias_verdict is copy_verdict
    assert len(taken) >= 4, taken
    assert scans, "the in-place write never ran the storage-alias scan: inconclusive"
    for found in scans:
        assert not any(alias is registered for alias in found for registered in taken)
    assert graphs["alias"] == graphs["copy"]
    assert values["alias"].keys() == values["copy"].keys()
    for key, copied in values["copy"].items():
        assert torch.equal(values["alias"][key], copied), key
    relu = torch.relu(model.lin(x)).detach()
    readers = [key for key in values["alias"] if key[1].startswith("relu")]
    assert readers, values["alias"].keys()
    for key in readers:
        assert torch.equal(values["alias"][key], relu), key
    # Copied out by the write, the aliases still match the digest of their pre-write bytes:
    # validation's first attempt stands, with no rerun.
    assert len(copied_out) == 3 and all(copied_out), copied_out
    assert not [d for d in diagnostics if d.check_name == arg_references.FALLBACK_CHECK]


def test_a_host_write_into_a_registered_alias_is_caught_by_its_digest() -> None:
    real_runner = user_funcs._run_model_and_save_specified_outs
    written: list[int] = []
    diagnostics: list[Any] = []

    def runner(*args: Any, **kwargs: Any) -> Any:
        trace = real_runner(*args, **kwargs)
        if kwargs.get("save_args"):
            for _, parent, value in _parent_slots(trace):
                if parent.layer_label.startswith("relu") and _shares_parent_output(value, parent):
                    before = value._version
                    value.detach().numpy()[...] += 1.0  # a host write: no dispatch, no version
                    assert value._version == before
                    written.append(before)
                    break
        return trace

    torch.manual_seed(0)
    model = _NoGradFanOut(write_after=False)
    x = torch.randn(8, 16)
    with mock.patch.object(user_funcs, "_run_model_and_save_specified_outs", runner):
        verdict = _validate(model, x, lambda t: diagnostics.extend(get_validation_diagnostics(t)))
    assert written, "no registered alias still shared the live storage: inconclusive"
    assert verdict is True
    fallbacks = [d for d in diagnostics if d.check_name == arg_references.FALLBACK_CHECK]
    assert len(fallbacks) == 1, diagnostics
    assert "without a version change" in fallbacks[0].extra["reason"]


@pytest.mark.parametrize("planted", ["wrong-parent", "dropped-version-record"])
def test_planted_capture_bugs_still_fail_with_grad_disabled(
    monkeypatch: pytest.MonkeyPatch, planted: str
) -> None:
    hits: list[str] = []
    if planted == "wrong-parent":
        _plant_wrong_parent(monkeypatch, hits)
        model: nn.Module = _NoGradWrapper(_Different())
        x = torch.randn(6)
    else:

        def drop_version(self: Any, parent_label: str, *args: Any, **kwargs: Any) -> None:
            hits.append(parent_label)

        monkeypatch.setattr(
            torchlens.backends.torch._ops_shared_fields,
            "_get_parent_output_version_snapshot",
            drop_version,
        )
        model = _NoGradWrapper(_UntracedWrite())
        x = torch.randn(3, 4)
    torch.manual_seed(0)
    assert _validate(model, x, lambda trace: None, copy_only=True) is False
    assert _validate(model, x, lambda trace: None) is False
    assert hits, "the planted bug never ran: inconclusive"


class _NoGradWrapper(nn.Module):
    """Run a wrapped module's forward with grad disabled."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.inner(x)


def _plant_wrong_parent(monkeypatch: pytest.MonkeyPatch, hits: list[str]) -> None:
    """Record the mul's relu parent as the other relu, whose values differ."""

    original = torchlens.backends.torch._ops_predicate_events._build_graph_relationship_fields

    def patched(self: Any, fields_dict: dict[str, Any], *args: Any) -> Any:
        result = original(self, fields_dict, *args)
        if fields_dict.get("func_name") != "__mul__":
            return result
        live = self.capture_events.live_index.by_raw_label
        replacements = [label for label in live if label.startswith("relu_2")]
        positions = fields_dict.get("parent_arg_positions") or {}
        for domain in ("args", "kwargs"):
            for key, label in list((positions.get(domain) or {}).items()):
                if replacements and label.startswith("relu_1"):
                    positions[domain][key] = replacements[0]
                    hits.append(label)
        parents = fields_dict.get("parents") or []
        for index, label in enumerate(list(parents)):
            if replacements and label.startswith("relu_1"):
                parents[index] = replacements[0]
        return result

    for module in (
        torchlens.backends.torch._ops_predicate_events,
        torchlens.backends.torch._ops_exhaustive,
        torchlens.backends.torch._ops_shared_fields,
    ):
        monkeypatch.setattr(module, "_build_graph_relationship_fields", patched)


def test_a_write_into_a_registered_alias_reruns_with_every_argument_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = arg_references.ParameterArgumentReferences._pending_value_alias
    written: list[str] = []

    def write_first(self: Any, tensor: Any, saved_value: Any, parent_label: str, *rest: Any) -> Any:
        alias = original(self, tensor, saved_value, parent_label, *rest)
        if alias is not None and not written:
            alias.add_(1.0)
            written.append(parent_label)
        return alias

    monkeypatch.setattr(
        arg_references.ParameterArgumentReferences, "_pending_value_alias", write_first
    )
    diagnostics: list[Any] = []
    torch.manual_seed(0)
    model = _NoGradFanOut(write_after=False)
    verdict = _validate(
        model, torch.randn(8, 16), lambda t: diagnostics.extend(get_validation_diagnostics(t))
    )
    assert written, "no alias was registered: inconclusive"
    assert verdict is True
    fallbacks = [d for d in diagnostics if d.check_name == arg_references.FALLBACK_CHECK]
    assert len(fallbacks) == 1, diagnostics
    assert "saved output" in fallbacks[0].extra["reason"]


def test_the_registered_alias_check_reads_the_registry_state() -> None:
    from torchlens._utils import tensor_utils

    live = torch.randn(5)
    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    with _state.pause_logging():
        tensor_utils.arm_deferred_payload_window(frozenset())
        try:
            payload = tensor_utils._try_defer_payload_alias(live)
            assert payload is not None
            alias = guard.value_alias(live.clone(), payload, "p")
        finally:
            tensor_utils.disarm_deferred_payload_window()
    assert alias is not None and guard.is_value_alias(alias)
    assert alias.untyped_storage().data_ptr() == live.untyped_storage().data_ptr()
    assert guard.first_change() is None
    before = live.clone()
    tensor_utils.materialize_deferred_for_call([live])
    live.mul_(2.0)
    assert alias.untyped_storage().data_ptr() != live.untyped_storage().data_ptr()
    assert torch.equal(alias, before) and torch.equal(payload, before)
    assert guard.first_change() is None
    alias.add_(1.0)
    assert guard.first_change() is not None


# ---------------------------------------------------------------------------
# Writes that skip the version counter, refusal pins, fail-closed alias
# ---------------------------------------------------------------------------


class _ArgmaxOfInput(nn.Module):
    """A replay-less parent (the input) read by a consumer blind to a uniform shift."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=1)


def _write_through_data(value: torch.Tensor) -> None:
    value.data.add_(1.0)


def _write_into_storage(value: torch.Tensor) -> None:
    with torch.no_grad():
        raw = torch.empty(0, dtype=value.dtype, device=value.device)
        raw.set_(value.untyped_storage(), value.storage_offset(), value.shape, value.stride())
        raw.add_(1.0)


def _validate_with_corrupted_argmax_input(
    write: Callable[[torch.Tensor], None], *, copy_only: bool = False
) -> tuple[bool, list[Any]]:
    """Validate ``_ArgmaxOfInput``, writing into the argmax's saved input before replay."""

    real_runner = user_funcs._run_model_and_save_specified_outs
    diagnostics: list[Any] = []

    def runner(*args: Any, **kwargs: Any) -> Any:
        trace = real_runner(*args, **kwargs)
        if kwargs.get("save_args"):
            argmax = next(op for op in trace._layer_list if op.func_name == "argmax")
            write(argmax.call.saved_args[0])
        return trace

    def observe(trace: Any) -> None:
        diagnostics.extend(get_validation_diagnostics(trace) or ())

    torch.manual_seed(0)
    x = torch.randn(4, 6, requires_grad=True)
    with mock.patch.object(user_funcs, "_run_model_and_save_specified_outs", runner):
        verdict = _validate(_ArgmaxOfInput(), x, observe, copy_only=copy_only)
    return verdict, diagnostics


@pytest.mark.parametrize(
    "write", [_write_through_data, _write_into_storage], ids=["data", "storage"]
)
def test_a_write_that_skips_the_version_counter_still_fails(
    write: Callable[[torch.Tensor], None],
) -> None:
    shared: list[bool] = []

    def observe(trace: Any) -> None:
        shared.extend(
            _shares_parent_output(value, parent)
            for op, parent, value in _parent_slots(trace)
            if op.func_name == "argmax"
        )

    torch.manual_seed(0)
    x = torch.randn(4, 6, requires_grad=True)
    assert _validate(_ArgmaxOfInput(), x, observe) is True
    assert shared == [True], "the argmax slot was not aliased: inconclusive"
    assert _validate_with_corrupted_argmax_input(write, copy_only=True)[0] is False
    verdict, diagnostics = _validate_with_corrupted_argmax_input(write)
    assert verdict is False
    fallbacks = [d for d in diagnostics if d.check_name == arg_references.FALLBACK_CHECK]
    assert len(fallbacks) == 1, diagnostics
    assert "without a version change" in fallbacks[0].extra["reason"]


class _Sub(torch.Tensor):
    """A tensor subclass: never aliased, whichever side it is on."""


def test_the_guard_refuses_tensor_subclasses() -> None:
    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    saved = torch.arange(6.0)
    with _state.pause_logging():
        assert guard.value_alias(saved.clone().as_subclass(_Sub), saved, "p") is None
        assert guard.value_alias(saved.clone(), saved.clone().as_subclass(_Sub), "p") is None
        assert guard.value_alias(saved.clone(), saved, "p") is not None


def _conj_case() -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.complex(torch.arange(4.0), torch.arange(4.0) + 1.0)
    return base.conj(), base.clone()


def _neg_case() -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(4.0) + 1.0
    return torch._neg_view(base), base.clone()


@pytest.mark.parametrize("case", [_conj_case, _neg_case], ids=["conj", "neg"])
def test_the_guard_refuses_conj_and_neg_bits_even_when_the_raw_bytes_match(
    case: Callable[[], tuple[torch.Tensor, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    tensor, saved = case()
    assert tensor.is_conj() or tensor.is_neg()
    # The physical bytes are the same; the logical values differ by the lazy bit.
    monkeypatch.setattr(arg_references, "_same_bits", lambda tensor, saved_value: True)
    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    with _state.pause_logging():
        assert guard.value_alias(tensor, saved, "p") is None
        assert guard.value_alias(tensor.resolve_conj().resolve_neg(), saved, "p") is not None


def test_a_view_context_receiver_is_never_a_saved_value_alias() -> None:
    base = torch.arange(8.0)
    receiver = base[2:]
    payload = receiver.clone()
    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    with _state.pause_logging():
        copied = saved_arg_copies._parameter_alias(
            guard,
            receiver,
            True,
            receiver_copy=saved_arg_copies._copy_receiver_with_view_context,
            parent=("p", payload),
        )
    assert copied.untyped_storage().data_ptr() != payload.untyped_storage().data_ptr()
    assert not guard.is_value_alias(copied)
    assert torch.equal(copied, payload)


def test_a_view_context_receiver_on_a_parameter_is_copied_not_aliased() -> None:
    model = nn.Linear(4, 4)
    receiver = model.weight[1:]
    guard = arg_references.ParameterArgumentReferences(
        arg_references._parameter_owners(model), alias_saved_values=True
    )
    token = arg_references._ACTIVE.set(guard)
    try:
        copied = saved_arg_copies._copy_saved_arg_tree(
            receiver, True, receiver_copy=saved_arg_copies._copy_receiver_with_view_context
        )
    finally:
        arg_references._ACTIVE.reset(token)
    weight_ptr = model.weight.untyped_storage().data_ptr()
    assert copied.untyped_storage().data_ptr() != weight_ptr
    assert torch.equal(copied, receiver)


def test_an_exception_while_aliasing_falls_back_to_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raised: list[int] = []

    def boom(*args: Any) -> bool:
        raised.append(1)
        raise RuntimeError("unexpected tensor")

    monkeypatch.setattr(arg_references, "_value_alias_eligible", boom)
    shared: list[bool] = []

    def observe(trace: Any) -> None:
        shared.extend(
            _shares_parent_output(value, parent) for _, parent, value in _parent_slots(trace)
        )

    torch.manual_seed(0)
    model = _FanOut()
    x = torch.randn(8, 16)
    copy_verdict = _validate(model, x, lambda trace: None, copy_only=True)
    assert _validate(model, x, observe) is copy_verdict is True
    assert raised, "the planted exception never ran: inconclusive"
    assert shared and not any(shared), shared


def _registered_alias(
    guard: arg_references.ParameterArgumentReferences, live: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Defer ``live``'s payload, then ask ``guard`` for an argument alias of it."""

    from torchlens._utils import tensor_utils

    with _state.pause_logging():
        tensor_utils.arm_deferred_payload_window(frozenset())
        try:
            payload = tensor_utils._try_defer_payload_alias(live)
            assert payload is not None
            alias = guard.value_alias(live.clone(), payload, "p")
        finally:
            tensor_utils.disarm_deferred_payload_window()
    return alias, payload


def test_a_write_the_registry_never_sees_is_named_by_the_version_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torchlens._utils import tensor_utils

    live = torch.randn(5)
    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    alias, _ = _registered_alias(guard, live)
    assert alias is not None and guard.first_change() is None
    # An in-place write no wrapper announces: the clone-on-write interception is skipped,
    # so the alias stays on the live storage and its bytes change with a version bump.
    monkeypatch.setattr(tensor_utils, "_DEFER_BUSY", True)
    live.add_(1.0)
    monkeypatch.setattr(tensor_utils, "_DEFER_BUSY", False)
    assert alias.untyped_storage().data_ptr() == live.untyped_storage().data_ptr()
    change = guard.first_change()
    assert change is not None and "without interception" in change, change


def test_a_host_write_into_a_registered_alias_moves_its_digest_not_its_version() -> None:
    live = torch.randn(5)
    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    alias, _ = _registered_alias(guard, live)
    assert alias is not None and guard.first_digest_change() is None
    live.detach().numpy()[0] += 1.0
    assert guard.first_change() is None
    change = guard.first_digest_change()
    assert change is not None and "without a version change" in change, change


def test_the_pending_alias_refusals_fall_back_to_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torchlens._utils import tensor_utils

    guard = arg_references.ParameterArgumentReferences({}, alias_saved_values=True)
    live = torch.randn(5)
    alias, payload = _registered_alias(guard, live)
    assert alias is not None, "the unrefused case must alias: inconclusive"
    with _state.pause_logging():
        # Window unarmed: the registry refuses to mint, so the argument is copied.
        assert guard.value_alias(live.clone(), payload, "p") is None
        # A copy that would require grad cannot be a detached registry alias.
        tensor_utils.arm_deferred_payload_window(frozenset())
        try:
            with torch.enable_grad():
                assert guard.value_alias(live.clone().requires_grad_(), payload, "p") is None
            # Kept on another device (an offloading output_device): refused.
            offload = arg_references.ParameterArgumentReferences(
                {}, None, "meta", alias_saved_values=True
            )
            assert offload.value_alias(live.clone(), payload, "p") is None
        finally:
            tensor_utils.disarm_deferred_payload_window()
    # Storage that cannot be identified refuses before anything is minted.
    monkeypatch.setattr(tensor_utils, "_deferred_storage_key", lambda tensor: None)
    with _state.pause_logging():
        assert guard.value_alias(live.clone(), payload, "p") is None
