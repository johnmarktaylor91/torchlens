"""The output alias checks that read each storage pointer once, against the per-pair checks.

``aliasing`` reads each tensor's storage pointer at most once per call and compares
integers. The answer and the function-call ids drawn must stay exactly those of the
per-pair checks it replaced (``op.func_call_id`` persists into runnable descriptors).
The reference functions below are the checks as they stood before the change: the
plain bare-output check and the generic per-pair loop over :func:`aliasing._tensors_alias`.
The direct tests compare answers and id draws case by case; the capture tests patch the
references in and require the persisted ``func_call_id`` sequence of a fixed capture to be
byte-identical.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._ir.intervention import FunctionEventInput
from torchlens._ir.semantics import BackendSemantics
from torchlens._utils._torch_symbols import tensor_read
from torchlens.backends.torch import aliasing

_FIELDS = {
    "grad_fn_handle": None,
    "grad_fn_class_name": "AddBackward0",
    "autograd_memory": 0,
    "num_autograd_tensors": 2,
    "bytes_delta_at_call": None,
    "bytes_peak_at_call": None,
}


# --- references: the checks as they stood before the pointer-once change ----------------


def _ref_tensors_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    if id(left) == id(right):
        return True
    logging_enabled = _state.active_capture()[1]
    try:
        with _state.pause_logging():
            if logging_enabled:
                _state.next_func_call_id()
            left_ptr = tensor_read(left, "untyped_storage")(left).data_ptr()
            if logging_enabled:
                _state.next_func_call_id()
            right_ptr = tensor_read(right, "untyped_storage")(right).data_ptr()
        return bool(left_ptr == right_ptr)
    except RuntimeError:
        return False


def _ref_aliased_positions(raw_output: object, input_tensors: Any) -> tuple[object, ...]:
    def outputs() -> Any:
        for _, tensor in aliasing._iter_tensor_positions((), raw_output):
            yield tensor

    return tuple(
        position
        for position, tensor in input_tensors
        if any(_ref_tensors_alias(out, tensor) for out in outputs())
    )


def _ref_bare_output_aliases(raw_output: torch.Tensor, positions: Any) -> tuple[object, ...]:
    aliased: list[object] = []
    logging_enabled = _state.active_capture()[1]
    output_ptr: Any = None
    no_pointer = object()
    with _state.pause_logging():
        for position, tensor in positions:
            if tensor is raw_output:
                aliased.append(position)
                continue
            if logging_enabled:
                _state.next_func_call_id()
            if output_ptr is None:
                try:
                    output_ptr = tensor_read(raw_output, "untyped_storage")(raw_output).data_ptr()
                except RuntimeError:
                    output_ptr = no_pointer
            if output_ptr is no_pointer:
                continue
            if logging_enabled:
                _state.next_func_call_id()
            try:
                input_ptr = tensor_read(tensor, "untyped_storage")(tensor).data_ptr()
            except RuntimeError:
                continue
            if input_ptr == output_ptr:
                aliased.append(position)
    return tuple(aliased)


def _ref_plain(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], raw_output: torch.Tensor, fields: Any
) -> BackendSemantics:
    positions: list[tuple[object, torch.Tensor]] = []
    for index, value in enumerate(args):
        positions.extend(aliasing._iter_tensor_positions(index, value))
    for key, value in kwargs.items():
        positions.extend(aliasing._iter_tensor_positions(key, value))
    return BackendSemantics(
        backend_grad_handle=fields["grad_fn_handle"],
        grad_fn_class_name=fields["grad_fn_class_name"],
        autograd_memory=fields["autograd_memory"],
        num_autograd_tensors=fields["num_autograd_tensors"],
        mutated_input_positions=(),
        aliased_output_inputs=_ref_bare_output_aliases(raw_output, positions) if positions else (),
        unknown_aliasing=False,
        bytes_delta_at_call=fields["bytes_delta_at_call"],
        bytes_peak_at_call=fields["bytes_peak_at_call"],
    )


def _ref_contract(full: bool) -> Callable[..., BackendSemantics]:
    def contract(func_event_input: FunctionEventInput, **facts: Any) -> BackendSemantics:
        inputs = aliasing._flatten_input_tensors(func_event_input.args, func_event_input.kwargs)
        mutated: tuple[object, ...] = ()
        if full:
            copies = aliasing._flatten_copied_input_tensors(
                func_event_input.arg_copies, func_event_input.kwarg_copies
            )
            mutated = tuple(
                position
                for position, tensor in inputs
                if aliasing._position_was_mutated(position, tensor, copies)
            )
        return BackendSemantics(
            mutated_input_positions=mutated,
            aliased_output_inputs=_ref_aliased_positions(func_event_input.raw_output, inputs),
            unknown_aliasing=False,
            **facts,
        )

    return contract


# --- direct cases -------------------------------------------------------------------------


def _sparse() -> torch.Tensor:
    return torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]), (3,))


def _inputs_and_outputs() -> list[tuple[tuple[Any, ...], dict[str, Any], Any]]:
    base = torch.arange(12.0).reshape(3, 4)
    other = torch.ones(3, 4)
    view = base.view(-1)
    meta = torch.empty(3, device="meta")
    param = nn.Parameter(torch.ones(2, 2))
    empty_a, empty_b = torch.empty(0), torch.empty(0)
    sparse = _sparse()
    return [
        ((base, other), {}, base + other),
        ((base, 2), {}, base.t()),
        ((base,), {}, base),
        (([base, other], 1), {"alpha": 2, "src": view}, view),
        (({"a": base, "b": (other, base)},), {}, view[2:]),
        ((meta, base), {}, meta.view(3)),
        ((), {}, base.clone()),
        ((base, base), {"out": base}, base),
        ((base, base, other), {}, base.view(4, 3)),
        ((param, base), {}, param.view(4)),
        ((param, param), {}, param * 2),
        ((empty_a, empty_b), {}, empty_a.view(0)),
        ((sparse, base), {}, base.view(-1)),
        ((base, sparse), {}, sparse.coalesce()),
        ((sparse,), {}, sparse),
        ((1.5, None, "x", True), {"k": 3}, base.clone()),
    ]


def _multi_outputs() -> list[tuple[tuple[Any, ...], dict[str, Any], Any]]:
    base = torch.arange(12.0).reshape(3, 4)
    other = torch.ones(3, 4)
    sparse = _sparse()
    return [
        ((base,), {}, base.split(2, dim=1)),
        ((base, other), {}, (other.clone(), base[0])),
        ((base, other), {}, [base.clone(), {"o": other.view(-1)}]),
        ((base, base), {}, (base, base.t())),
        ((other,), {}, (sparse, other.view(-1))),
        ((sparse, other), {}, (sparse, other.clone())),
        ((base,), {}, ()),
        ((), {}, (base,)),
        ((base, 3), {"dim": 0}, (1, base[1:], None)),
    ]


def _event(args: tuple[Any, ...], kwargs: dict[str, Any], out: Any) -> FunctionEventInput:
    return FunctionEventInput(
        func=torch.add,
        func_name="add",
        func_qualname=None,
        args=args,
        kwargs=kwargs,
        raw_output=out,
        arg_copies=args,
        kwarg_copies=kwargs,
        module_stack=(),
        is_bottom_level_func=True,
        func_call_id=0,
        expected_output_count=1,
    )


def _draws(monkeypatch: pytest.MonkeyPatch, logging_enabled: bool, call: Callable[[], Any]) -> Any:
    """Return ``call()``'s answer and the number of function-call ids it drew."""

    with monkeypatch.context() as patch:
        patch.setattr(_state, "_logging_enabled", logging_enabled)
        start = _state.next_func_call_id()
        answer = call()
        end = _state.next_func_call_id()
    return answer, end - start - 1


_FACTS = {
    "backend_grad_handle": None,
    "grad_fn_class_name": "AddBackward0",
    "autograd_memory": 0,
    "num_autograd_tensors": 2,
    "bytes_delta_at_call": None,
    "bytes_peak_at_call": None,
}


@pytest.mark.parametrize("case", range(len(_inputs_and_outputs())))
@pytest.mark.parametrize("logging_enabled", [False, True])
def test_plain_check_matches_the_reference(
    monkeypatch: pytest.MonkeyPatch, case: int, logging_enabled: bool
) -> None:
    """The bare-output check gives the reference answer and draws the same ids."""

    args, kwargs, out = _inputs_and_outputs()[case]
    ref = _draws(monkeypatch, logging_enabled, lambda: _ref_plain(args, kwargs, out, _FIELDS))
    new = _draws(
        monkeypatch,
        logging_enabled,
        lambda: aliasing.plain_output_alias_semantics(args, kwargs, out, _FIELDS),
    )
    assert new == ref


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("logging_enabled", [False, True])
def test_generic_contracts_match_the_reference(
    monkeypatch: pytest.MonkeyPatch, full: bool, multi: bool, logging_enabled: bool
) -> None:
    """Both generic contracts give the reference answers and draw the same ids, per case."""

    cases = _multi_outputs() if multi else _inputs_and_outputs()
    contract = (
        aliasing.detect_torch_alias_contract
        if full
        else aliasing.detect_torch_output_alias_contract
    )
    reference = _ref_contract(full)
    for args, kwargs, out in cases:
        event = _event(args, kwargs, out)
        ref = _draws(monkeypatch, logging_enabled, lambda: reference(event, **_FACTS))
        new = _draws(monkeypatch, logging_enabled, lambda: contract(event, **_FACTS))
        assert new == ref, (args, kwargs, out)


def test_each_pointer_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One storage read per distinct tensor in the generic check, one per position in the plain one."""

    base = torch.arange(12.0).reshape(3, 4)
    other = torch.ones(3, 4)
    reads: list[int] = []
    real = tensor_read

    def counting(tensor: Any, name: str) -> Any:
        resolved = real(tensor, name)

        def read(value: Any) -> Any:
            reads.append(id(value))
            return resolved(value)

        return read

    monkeypatch.setattr(aliasing, "tensor_read", counting)
    outputs = (base.view(-1), other.clone(), base[1])
    event = _event((other, base, other), {}, outputs)
    aliasing.detect_torch_output_alias_contract(event, **_FACTS)
    assert sorted(reads) == sorted({id(t) for t in (other, base, *outputs)})
    reads.clear()
    out = base.view(-1)
    aliasing.plain_output_alias_semantics((other, base), {"x": other}, out, _FIELDS)
    assert len(reads) == 4  # the output once, each of the three input positions once


# --- the persisted func_call_id sequence of a fixed capture ---------------------------------


class _Fixed(nn.Module):
    """Views, in-place ops, repeated inputs, parameters and multi-output calls."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)
        self.norm = nn.LayerNorm(6)
        self.register_buffer("count", torch.zeros(6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        v = y.view(-1, 3)
        z = v.t().contiguous()
        a, b = torch.chunk(y, 2, dim=1)
        w = torch.cat([a, b, a], dim=1)
        w = w + w
        w.add_(1.0)
        self.count.add_(1.0)
        parts = torch.split(w, 2, dim=1)
        s = torch.stack([parts[0], parts[1], parts[0]], dim=0).sum(0)
        r = torch.relu(self.norm(w)) * x
        r = r.transpose(0, 1).transpose(0, 1)
        return r.sum() + s.mean() + z.mean() + v[0].sum()


def _patch_references(patch: pytest.MonkeyPatch) -> None:
    references = {
        "plain_output_alias_semantics": _ref_plain,
        "detect_torch_output_alias_contract": _ref_contract(False),
        "detect_torch_alias_contract": _ref_contract(True),
    }
    for name, reference in references.items():
        current = getattr(aliasing, name)
        for module in list(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith("torchlens"):
                continue
            if vars(module).get(name) is current:
                patch.setattr(module, name, reference)


def _sequence(trace: Any) -> bytes:
    """The persisted ``func_call_id`` of every op, in order, as JSON bytes."""

    return json.dumps([op.func_call_id for op in trace.ops]).encode()


def _aliases(trace: Any) -> bytes:
    """Each op event's call id and alias answer, in order, as JSON bytes."""

    rows = [
        [event.function.func_call_id, repr(event.backend_semantics.aliased_output_inputs)]
        for event in trace._capture_events.op_events
    ]
    return json.dumps(rows).encode()


def _capture(tmp_path: Path, tag: str) -> tuple[bytes, bytes, bytes]:
    torch.manual_seed(0)
    model = _Fixed()
    trace = tl.trace(model, torch.randn(4, 6), capture=tl.options.CaptureOptions(seed=0))
    path = tmp_path / f"{tag}.tlspec"
    trace.save(str(path))
    return _sequence(trace), _sequence(tl.load(str(path))), _aliases(trace)


def test_persisted_func_call_ids_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fixed capture persists the same func_call_id sequence with the references patched in."""

    head = _capture(tmp_path, "head")
    with monkeypatch.context() as patch:
        _patch_references(patch)
        reference = _capture(tmp_path, "reference")
    assert head[0] == reference[0]
    assert head[1] == reference[1]
    assert head[2] == reference[2]
    ids = json.loads(head[0])
    assert len(ids) > 20 and all(isinstance(i, int) for i in ids)
    assert max(ids) - min(ids) + 1 > len(set(ids))  # the alias checks drew ids between ops
    assert any(row[1] != "()" for row in json.loads(head[2]))
