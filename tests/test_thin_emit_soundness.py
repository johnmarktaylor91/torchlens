"""The thin record path against today's emit, both paths captured and every public field compared.

``_thin_emit`` logs an admitted plain call (one bare tensor output in a plain
session) without today's draft, freeze and view. Each case here captures the same
seeded forward twice, with the thin path on and with it forced off
(``set_thin_emit_enabled(False)``: every call takes today's emit, the reference),
and compares the field differential's whole surface (labels, call ids,
version-counter verdicts, alias semantics, buffer records, reprs and tables), the
model state both forwards leave, and the fast-lane admission count. The deferred
cases change what a call read before any row fills (inputs mutated in place, the
containers it was given changed, inputs deleted, bases of its views mutated), with
the default capture and with saved arguments, since a thin row resolves its argument
template and its function config at the fill (``_thin_defer``). The direct tests pin
the folded admission check, the deferred resolution and the layering contract.
"""

from __future__ import annotations

import ast
import gc
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
import torchlens._postprocess as postprocess_module
from tests.test_plain_emit_soundness import (
    _Body,
    _Buffers,
    _Chain,
    _Cond,
    _Loop,
    _Nested,
    _Raises,
    _x,
)
from torchlens._capture.salient_args import extract_salient_args
from torchlens._ir import _row_journal
from torchlens._postprocess._step_views import MIN_COLUMN_OPS
from torchlens.backends.torch import (
    _call_plan,
    _ops_exhaustive,
    _thin_defer,
    _thin_emit,
    _thin_record,
    _thin_settle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from field_diff_lib.cases import SEED  # noqa: E402
from field_diff_lib.compare import diff  # noqa: E402
from field_diff_lib.mask import Mask  # noqa: E402
from field_diff_lib.surface import snapshot_trace  # noqa: E402

Build = Callable[[], nn.Module]


def _dump(trace: Any, model: nn.Module | None = None) -> dict[str, Any]:
    """Return the trace's canonical field-differential dump, process-local names removed.

    With ``model``, the module scope entries of the code-context cache keys
    (``("module", id(module))``, ``_call_site.module_id_scope_entry``) name the
    module by its address in ``model`` instead of its ``id``. Each side of a
    comparison builds its own model, so the ids differ by construction; the
    mask's ``address`` rule hides only integers of twelve digits or more, which
    covers Linux heap addresses but not macOS ones (about ten digits). An id
    that is not one of ``model``'s modules is left as it is and still compared.

    For the same reason the storage address inside each ``storage_key`` (a
    ``(device, data_ptr, nbytes)`` tuple, ``buffer_writes.storage_key``) is
    replaced by the order in which the dump first names it, so two keys on one
    storage still read equal and keys on different storages still differ.
    """

    dump, _canon = snapshot_trace(trace, [(str(REPO_ROOT), "<repo>")], Mask(enabled=True))
    dump["trace"]["event_stream"]["fields"].pop("run_nonce", None)
    text = json.dumps(dump, default=repr, sort_keys=True).replace(trace.name, "<trace>")
    loaded = _numbered_storage_addresses(json.loads(text), {})
    if model is None:
        return loaded
    names = {id(module): name or "<root>" for name, module in model.named_modules()}
    return _with_named_cache_modules(loaded, names)


def _numbered_storage_addresses(value: Any, seen: dict[int, int]) -> Any:
    """Replace the storage address of every canonical ``storage_key`` tuple by its first-seen order."""

    if isinstance(value, dict):
        out = {}
        for key, entry in value.items():
            parts = (
                entry.get("__tuple__") if key == "storage_key" and isinstance(entry, dict) else None
            )
            if isinstance(parts, list) and len(parts) == 3 and isinstance(parts[1], int):
                number = seen.setdefault(parts[1], len(seen))
                out[key] = {**entry, "__tuple__": [parts[0], f"<storage {number}>", parts[2]]}
            else:
                out[key] = _numbered_storage_addresses(entry, seen)
        return out
    if isinstance(value, list):
        return [_numbered_storage_addresses(entry, seen) for entry in value]
    return value


def _with_named_cache_modules(value: Any, names: dict[int, str]) -> Any:
    """Return ``value`` with the module entries under ``_code_context_cache`` named."""

    if isinstance(value, dict):
        return {
            key: _named_module_entries(entry, names)
            if key == "_code_context_cache"
            else _with_named_cache_modules(entry, names)
            for key, entry in value.items()
        }
    if isinstance(value, list):
        return [_with_named_cache_modules(entry, names) for entry in value]
    return value


def _named_module_entries(value: Any, names: dict[int, str]) -> Any:
    """Replace each canonical ``("module", id)`` pair whose id is a known module by its name."""

    if isinstance(value, dict):
        pair = value.get("__tuple__")
        if (
            isinstance(pair, list)
            and len(pair) == 2
            and pair[0] == "module"
            and isinstance(pair[1], int)
            and pair[1] in names
        ):
            return {"__tuple__": ["module", f"<module {names[pair[1]]}>"]}
        return {key: _named_module_entries(entry, names) for key, entry in value.items()}
    if isinstance(value, list):
        return [_named_module_entries(entry, names) for entry in value]
    return value


@pytest.fixture
def thin(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """Count thin hits and fast-lane commits; restore the process's thin setting after.

    Thin hits are counted from a profile hook on the bound thin emit's returns,
    not by wrapping it: a wrapper defined here is user code to the call-site
    walk, so it would add a source frame to thin rows only.
    """

    counts = {"thin": 0, "fast": 0}
    thin_code = _ops_exhaustive.bound_thin_emit().__code__
    previous_profile = sys.getprofile()

    def count_thin_returns(frame: Any, event: str, arg: Any) -> None:
        if event == "return" and frame.f_code is thin_code:
            counts["thin"] += bool(arg)

    real_fast = _call_plan.commit_fast_call

    def counted_fast(*args: Any) -> Any:
        counts["fast"] += 1
        return real_fast(*args)

    monkeypatch.setattr(_call_plan, "commit_fast_call", counted_fast)
    previous = _thin_emit.thin_emit_enabled()
    sys.setprofile(count_thin_returns)
    try:
        yield counts
    finally:
        sys.setprofile(previous_profile)
        _thin_emit.set_thin_emit_enabled(previous)


def _capture(build: Build, x: torch.Tensor, on: bool, **kwargs: Any) -> tuple[Any, nn.Module]:
    """Trace a freshly built, seeded model with the thin path on or off."""

    _thin_emit.set_thin_emit_enabled(on)
    torch.manual_seed(SEED)
    model = build()
    capture = kwargs.pop("capture", None) or tl.options.CaptureOptions(seed=SEED)
    return tl.trace(model, x.clone(), capture=capture, **kwargs), model


def _check(counts: dict[str, int], build: Build, x: torch.Tensor, **kwargs: Any) -> tuple[int, Any]:
    """Return the thin hits and the thin trace, after checking both paths agree."""

    _capture(build, x, True, **dict(kwargs))  # warm the process-wide caches
    counts["thin"] = counts["fast"] = 0
    on, on_model = _capture(build, x, True, **dict(kwargs))
    hits, fast_on = counts["thin"], counts["fast"]
    on_dump = _dump(on, on_model)
    counts["thin"] = counts["fast"] = 0
    off, off_model = _capture(build, x, False, **dict(kwargs))
    assert counts["thin"] == 0
    assert counts["fast"] == fast_on, "admission counts differ between the paths"
    assert diff(_dump(off, off_model), on_dump) == []
    on_state, off_state = on_model.state_dict(), off_model.state_dict()
    assert on_state.keys() == off_state.keys()
    for key, value in on_state.items():
        assert torch.equal(value, off_state[key]), key
    return hits, on


def _body(fn: Callable[[nn.Module, torch.Tensor], Any]) -> Build:
    return lambda: _Body(fn)


def test_inplace_ops(thin: dict[str, int]) -> None:
    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = m.lin(x) * 2
        y.relu_()
        y.add_(1)
        y += x
        return y * m.weight

    hits, trace = _check(thin, _body(body), _x())
    assert hits > 0
    assert any(op.is_inplace for op in trace.ops)


def test_aliasing_views(thin: dict[str, int]) -> None:
    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x + 1
        v = y.view(-1)
        t = y.t()
        v.mul_(2)
        d = y.detach()
        return t.sum() + v[1:4].sum() + d.mean() + y.expand(2, 2, 3).sum()

    hits, trace = _check(thin, _body(body), _x())
    assert hits > 0
    assert [op for op in trace.ops if op.func_name in ("view", "t")]


def test_tensor_mutated_after_its_op(thin: dict[str, int]) -> None:
    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        a = x * 2
        b = a + 1
        a.add_(5)
        c = a * b
        a.zero_()
        return c + a

    hits, _trace = _check(thin, _body(body), _x())
    assert hits > 0


def test_nested_modules_and_repeated_calls(thin: dict[str, int]) -> None:
    hits, trace = _check(thin, _Nested, _x())
    assert hits > 0
    assert any(len(op.modules) > 1 for op in trace.ops if op.modules)


def test_loops(thin: dict[str, int]) -> None:
    hits, _trace = _check(thin, _Loop, _x())
    assert hits > 0


def test_loopcell100(thin: dict[str, int]) -> None:
    """The gate model with a hundred passes of one cell."""

    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks import capture_core_cost as cost

    model, x = cost.build("loopcell100")
    hits, trace = _check(thin, lambda: model, x)
    assert hits > 100
    assert len(trace.ops) >= hits


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_conditionals(thin: dict[str, int], sign: float) -> None:
    hits, _trace = _check(thin, lambda: _Cond(), _x().abs() * sign + sign)
    assert hits > 0


def test_buffers(thin: dict[str, int]) -> None:
    hits, _trace = _check(thin, _Buffers, _x())
    assert hits > 0


def test_backward(thin: dict[str, int]) -> None:
    """A gradient-saving session keeps alias contracts: the thin path declines every call."""

    capture = tl.options.CaptureOptions(seed=SEED, save_grads="all")
    hits, trace = _check(thin, _Nested, _x(), capture=capture)
    assert hits == 0
    assert trace.ops


def test_backward_after_a_thin_capture(thin: dict[str, int]) -> None:
    """Gradients through a forward the thin path logged match today's emit's."""

    grads = []
    for on in (True, False):
        _thin_emit.set_thin_emit_enabled(on)
        torch.manual_seed(SEED)
        model = _Nested()
        x = _x().requires_grad_(True)
        tl.trace(model, x, capture=tl.options.CaptureOptions(seed=SEED))
        model(x).sum().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for left, right in zip(*grads, strict=True):
        assert torch.equal(left, right)


def test_interventions(thin: dict[str, int]) -> None:
    intervene = tl.when(tl.func("relu"), tl.zero_ablate())
    _hits, trace = _check(thin, _Nested, _x(), intervene=intervene)
    assert any(op.func_name == "relu" for op in trace.ops)


def test_halt(thin: dict[str, int]) -> None:
    def halt_on_relu(ctx: Any) -> bool:
        return ctx.kind == "op" and ctx.func_name == "relu"

    _hits, trace = _check(thin, _Nested, _x(), stop_after=halt_on_relu)
    assert trace.is_stopped


def test_exception_mid_forward(thin: dict[str, int]) -> None:
    """A forward that raises after some ops raises the same error on both paths."""

    errors = []
    for on in (True, False):
        with pytest.raises(Exception) as caught:
            _capture(_Raises, _x(), on)
        errors.append((type(caught.value).__name__, str(caught.value)))
    assert errors[0] == errors[1]
    assert "planted failure mid-forward" in errors[0][1]
    hits, _trace = _check(thin, _Nested, _x())
    assert hits > 0


@pytest.mark.parametrize("length", [1, 2, MIN_COLUMN_OPS - 1, MIN_COLUMN_OPS, 400])
def test_both_sides_of_min_column_ops(thin: dict[str, int], length: int) -> None:
    """Below and at the column postprocess threshold, and well past it."""

    hits, trace = _check(thin, lambda: _Chain(length), _x())
    assert hits == length
    assert len(trace.ops) >= length


def test_save_and_load(thin: dict[str, int], tmp_path: Path) -> None:
    """The saved trace of the thin path loads to the same fields as today's emit's."""

    dumps = []
    for on in (True, False):
        trace, _model = _capture(_Nested, _x(), on)
        path = tmp_path / f"thin-{on}.tlspec"
        trace.save(str(path))
        dumps.append(_dump(tl.load(str(path))))
    assert diff(dumps[1], dumps[0]) == []


# Deferred families: each case changes what a call read before any row fills.
_SAVE_ARGS = pytest.mark.parametrize("save_args", [False, True], ids=["default", "save_args"])

_EXTERNAL = torch.linspace(-1.0, 1.0, 3)
"""A tensor made outside the capture: an unlabeled argument the template copies."""


def _options(save_args: bool) -> dict[str, Any]:
    return {"capture": tl.options.CaptureOptions(save_args=save_args, seed=SEED)}


@_SAVE_ARGS
def test_inputs_mutated_in_place_after_their_calls(thin: dict[str, int], save_args: bool) -> None:
    """In-place writes on a call's inputs after it (and on the model input) leave its row as today's."""

    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        a = x * 2
        s = torch.softmax(a, dim=1)
        p = F.pad(a, [1, 1])
        c = torch.clamp(a, min=-1.0, max=1.0)
        e = a * _EXTERNAL.clone()
        a.mul_(3)
        x.add_(1)
        s.zero_()
        return s.sum() + p.sum() + c.sum() + e.sum() + a.sum() + x.sum()

    hits, _trace = _check(thin, _body(body), _x(), **_options(save_args))
    assert hits > 0


@_SAVE_ARGS
def test_argument_containers_changed_after_their_calls(
    thin: dict[str, int], save_args: bool
) -> None:
    """Lists and dicts the call was given, changed after it, leave its literals and config as today's."""

    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        pad = [1, 2]
        parts = [x * 2, x + 1]
        dims = {"dim": 0}
        p = F.pad(x, pad)
        c = torch.cat(parts, **dims)
        st = torch.stack(parts, dim=1)
        r = torch.sum(c, dim=[0, 1])
        pad[0] = 5
        pad.append(7)
        parts[0] = x
        parts.clear()
        dims["dim"] = 1
        return p.sum() + c.sum() + st.sum() + r + F.pad(x, pad[:2]).sum()

    hits, _trace = _check(thin, _body(body), _x(), **_options(save_args))
    assert hits > 0


@_SAVE_ARGS
def test_inputs_deleted_before_settle(thin: dict[str, int], save_args: bool) -> None:
    """Inputs deleted (and collected) before any row fills leave the rows as today's."""

    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        a = x * 2
        b = torch.softmax(a, dim=0)
        c = F.layer_norm(a, (3,))
        t = torch.tensor([1.0, 2.0, 3.0])
        d = b * t
        k = _EXTERNAL.clone()
        e = torch.clamp(c, min=-0.5) + k
        del a, t, k
        gc.collect()
        return (b + c + d + e).sum()

    hits, _trace = _check(thin, _body(body), _x(), **_options(save_args))
    assert hits > 0


@_SAVE_ARGS
def test_views_of_mutated_bases(thin: dict[str, int], save_args: bool) -> None:
    """Views read by a call, whose bases (or the views) are mutated after it, leave its row as today's."""

    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        a = x + 1
        v = a[0]
        w = torch.softmax(v, dim=0)
        t = a.t()
        s = torch.sum(t, dim=1)
        q = F.pad(v, [1, 0])
        a.mul_(2)
        v.add_(1)
        t.clamp_(max=2.0)
        u = a.view(-1).clamp(min=0.0)
        return w.sum() + s.sum() + q.sum() + u.sum() + v.sum()

    hits, _trace = _check(thin, _body(body), _x(), **_options(save_args))
    assert hits > 0


def test_thin_rows_resolve_the_deferred_families_at_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thin row holds its template and argument-reading config unresolved until it fills."""

    seen: list[tuple[bool, bool, bool]] = []
    real_fill = _thin_record._ThinRow._fill

    def spy(row: Any) -> None:
        fields = row.fields
        seen.append(
            (_thin_defer.DEFERRED in fields, "args_template" in fields, "func_config" in fields)
        )
        real_fill(row)
        assert _thin_defer.DEFERRED not in fields
        assert {"args_template", "kwargs_template", "func_id", "func_config"} <= fields.keys()

    monkeypatch.setattr(_thin_record._ThinRow, "_fill", spy)
    tl.trace(nn.Sequential(nn.Linear(3, 3), nn.Softmax(dim=1), nn.ReLU()), torch.randn(2, 3))
    assert seen and all(deferred and not template for deferred, template, _ in seen)
    assert any(not config for _, _, config in seen), "no argument-reading config was deferred"
    assert any(config for _, _, config in seen), "a config with no extractor was deferred"


def test_set_function_config_matches_the_eager_extraction() -> None:
    """Deferred, the config equals the eager one; a refused record resolves at the call."""

    x = torch.randn(2, 3)
    pad = [1, 2]
    call = ("pad", "pad", (x, pad), {})
    eager: dict[str, Any] = {}
    _thin_defer.set_function_config(eager, call, False)
    deferred: dict[str, Any] = {}
    _thin_defer.set_function_config(deferred, call, True)
    assert "func_config" not in deferred
    pad.append(9)
    x.resize_(4, 4)
    _thin_defer.resolve_deferred(deferred)
    assert deferred == eager and _thin_defer.DEFERRED not in deferred

    class Holder(set):  # a container the recorder does not rebuild, holding a tensor
        pass

    refused: dict[str, Any] = {}
    odd = ("softmax", "softmax", (torch.randn(2),), {"dim": 0, "holder": Holder({torch.ones(1)})})
    _thin_defer.set_function_config(refused, odd, True)
    assert _thin_defer.DEFERRED not in refused
    assert refused["func_config"] == extract_salient_args(*odd) == {"dim": 0}


def _at_ingest(monkeypatch: pytest.MonkeyPatch, hook: Callable[[], None]) -> None:
    """Run ``hook`` once per capture, before the hand-off to ingest builds anything."""

    real_hand_off = postprocess_module.hand_off_capture_rows

    def hand_off(*args: Any) -> Any:
        hook()
        return real_hand_off(*args)

    monkeypatch.setattr(postprocess_module, "hand_off_capture_rows", hand_off)


def test_thin_rows_build_no_record_before_ingest_and_each_once(
    thin: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thin row builds its record only after capture, and each record once."""

    builds = [0]
    real_build = _row_journal._record_from_row

    def counting_build(handle: Any) -> Any:
        builds[0] += 1
        return real_build(handle)

    seen: list[int] = []
    monkeypatch.setattr(_row_journal, "_record_from_row", counting_build)
    _at_ingest(monkeypatch, lambda: seen.append(builds[0]))
    _thin_emit.set_thin_emit_enabled(True)
    trace = tl.trace(_Nested(), _x())
    try:
        assert thin["thin"] > 0
        assert seen == [0]
        built = builds[0]
        events = trace.op_events
        assert builds[0] == built
        assert built == len(events)
    finally:
        trace.cleanup()


def test_requires_grad_flip_before_the_record_build(
    thin: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping params and outputs before ingest reaches thin rows no more than today's rows."""

    outputs: list[torch.Tensor] = []
    counted_fast = _call_plan.commit_fast_call

    def keeping_fast(*args: Any) -> Any:
        outputs.append(args[2][6])  # the call's output; both paths pass through here
        return counted_fast(*args)

    monkeypatch.setattr(_call_plan, "commit_fast_call", keeping_fast)
    models: list[nn.Module] = []

    def flip() -> None:
        for param in models[-1].parameters():
            param.requires_grad_(False)
        for tensor in outputs:
            tensor.detach_()

    _at_ingest(monkeypatch, flip)
    dumps = []
    for on in (True, False):
        outputs.clear()
        _thin_emit.set_thin_emit_enabled(on)
        torch.manual_seed(SEED)
        models.append(nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)))
        x = torch.randn(2, 3)
        trace = tl.trace(models[-1], x, capture=tl.options.CaptureOptions(seed=SEED))
        dumps.append(_dump(trace, models[-1]))
        assert outputs  # every fast commit passes here, served thin or not
    assert diff(dumps[1], dumps[0]) == []


def test_session_facts_decline(thin: dict[str, int]) -> None:
    """Each plain-session fact on its own sends every call to today's emit."""

    class Facts:
        track_device_memory = False
        _options_aten_ops = "off"
        _options_on_nonfinite = None

    never = lambda _trace: False  # noqa: E731
    assert _thin_settle.plain_session(Facts(), never)
    for name, value in (
        ("track_device_memory", True),
        ("_predicate_save_options", object()),
        ("_options_aten_ops", "values"),
        ("track_nonfinite", True),
        ("_echo_session", object()),
        ("_options_on_nonfinite", "raise"),
    ):
        facts = Facts()
        setattr(facts, name, value)
        assert not _thin_settle.plain_session(facts, never), name
    assert not _thin_settle.plain_session(Facts(), lambda _trace: True)


def test_thin_modules_import_nothing_from_the_capture_cycle() -> None:
    """The ``_thin_*`` modules import only modules outside the capture import cycle."""

    allowed = {
        "..._capture._recorded_tensor",
        "..._capture.plan",
        "..._capture.salient_args",
        "..._ir._row_journal",
        "..._utils.display",
        "..._utils._internal_read",
        "..._utils.env_flags",
        "._arg_template_resolve",
        "._ops_freeze_refs",
        "._ops_row_journal",
        "._thin_defer",
        "._thin_record",
        "._thin_settle",
        ".aliasing",
    }
    package = REPO_ROOT / "torchlens" / "backends" / "torch"
    for name in ("_thin_defer.py", "_thin_emit.py", "_thin_record.py", "_thin_settle.py"):
        tree = ast.parse((package / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                module = "." * node.level + (node.module or "")
                in_type_block = any(
                    isinstance(parent, ast.If) and node in ast.walk(parent)
                    for parent in tree.body
                    if isinstance(parent, ast.If)
                )
                assert in_type_block or module in allowed, (name, module)


def test_fill_settle_fields_takes_the_builders_and_keeps_call_values() -> None:
    """The settle fill writes today's descriptors and defaults, never over a call-time value."""

    from torchlens.backends.torch._ops_descriptive_fields import _set_descriptive_fields

    fields: dict[str, Any] = {
        "func": torch.relu,
        "func_name": "relu",
        "non_tensor_pos_args": [1],
        "non_tensor_kwargs": {"a": 2},
        "out": "saved",
        "container_path": (0,),
    }
    _thin_settle.fill_settle_fields(fields, _set_descriptive_fields)
    expected: dict[str, Any] = {}
    _set_descriptive_fields(expected, torch.relu, "relu")
    assert {name: fields[name] for name in expected} == expected
    assert fields["func_non_tensor_args"] == [1, 2]
    assert fields["out"] == "saved" and fields["container_path"] == (0,)
    assert fields["step_index"] is None and fields["_construction_done"] is False
    assert "fire_results" not in fields


def test_thin_rows_fill_the_settle_families_on_first_need(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thin row holds no settle family until a read needs one; the fill runs once per row."""

    from torchlens.backends.torch import _thin_record
    from torchlens.backends.torch._ops_row_journal import _PendingRow

    seen: list[tuple[int, bool, bool, bool]] = []
    real_fill = _thin_record._ThinRow._fill

    def spy(row: Any) -> None:
        fields = row.fields
        seen.append(
            (
                id(row),
                "func_qualname" in fields,
                "step_index" in fields,
                "func_non_tensor_args" in fields,
            )
        )
        real_fill(row)
        assert type(row) is _PendingRow
        assert type(row.device) is str and type(row.handle_id) is str

    monkeypatch.setattr(_thin_record._ThinRow, "_fill", spy)
    tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Tanh()), torch.randn(2, 3))
    assert seen and all(entry[1:] == (False, False, False) for entry in seen)
    assert len({entry[0] for entry in seen}) == len(seen)
