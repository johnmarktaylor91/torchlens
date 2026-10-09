"""The per-op live intervention check is skipped only when no rule can fire at an op door.

On the torch backend an op-time site never carries ``output_of_module_calls`` (that field is
joined at module exit), so a ``tl.module(...)`` WHERE term, or an ``|`` / ``&`` composite of
them, fires only at the module-boundary door. ``apply_live_hooks_to_outputs`` returns early
when the ``intervene=`` spec and every active hook-plan entry are module-exit-only; every
other operand keeps the per-op path.

These tests pin three things: the eligibility predicates (exact and conservative), equality of
every observable output against the per-op path forced on (fire records, saved payloads, op
labels, metadata invariants), and the saving itself as deterministic per-op call counts.
"""

from __future__ import annotations

import gc
import hashlib
import json
import warnings
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _ops_interventions, ops as torch_ops
from torchlens.intervention import runtime as intervention_runtime
from torchlens.intervention.hooks import NormalizedHookEntry
from torchlens.intervention.selectors import CompositeSelector
from torchlens.intervention.spec import InterventionSpec

_MAGNITUDE = 2.0
_VOLATILE_KEYS = ("time", "duration", "elapsed", "_at", "timestamp", "memory", "pid", "id(")


class _Block(nn.Module):
    """A linear layer followed by ReLU, so module and op scopes differ."""

    def __init__(self) -> None:
        """Build the block."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``relu(linear(x))``."""

        return torch.relu(self.linear(x))


class _Net(nn.Module):
    """``block1`` runs twice, ``block2`` once; the model output is a later op."""

    def __init__(self) -> None:
        """Build the blocks and the head."""

        super().__init__()
        self.block1 = _Block()
        self.block2 = _Block()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run block1 twice, block2 once, then the head."""

        hidden = self.block1(x)
        hidden = self.block1(hidden * 0.5)
        hidden = self.block2(hidden + 0.25)
        return self.head(hidden)


def _model() -> _Net:
    """Return a freshly seeded model in eval mode."""

    torch.manual_seed(0)
    return _Net().eval()


def _input() -> torch.Tensor:
    """Return a fixed input batch."""

    generator = torch.Generator().manual_seed(1)
    return torch.randn(2, 4, generator=generator)


def _direction() -> torch.Tensor:
    """Return a fixed steering direction."""

    return torch.tensor([1.0, -0.5, 0.25, 0.0])


def _steer() -> Any:
    """Return the steering helper used across the tests."""

    return tl.steer(_direction(), magnitude=_MAGNITUDE, feature_axis=-1)


# spec name -> (spec factory, whether the per-op check is skipped, whether the spec fires).
# The two mixes are controls that keep the per-op path; ``module_and_func`` never fires (an op
# door never matches ``tl.module`` and a boundary never matches ``tl.func``), so it pins the
# zero-match warning instead.
_SPECS: dict[str, tuple[Callable[[], Any], bool, bool]] = {
    "steer": (lambda: tl.when(tl.module("block1"), _steer()), True, True),
    "zero_ablate": (lambda: tl.when(tl.module("block2"), tl.zero_ablate()), True, True),
    "two_module_union": (
        lambda: tl.when(tl.module("block1") | tl.module("block2"), tl.scale(0.5)),
        True,
        True,
    ),
    "module_and_func": (
        lambda: tl.when(tl.module("block1") & tl.func("relu"), _steer()),
        False,
        False,
    ),
    "module_or_func": (
        lambda: tl.when(tl.module("block2") | tl.func("relu"), tl.scale(0.5)),
        False,
        True,
    ),
    "func_steer": (lambda: tl.when(tl.func("relu"), _steer()), False, True),
}


def _thash(value: Any) -> str:
    """Return a bit-exact fingerprint of a tensor, or a type tag for anything else."""

    if not isinstance(value, torch.Tensor):
        return f"<{type(value).__name__}>"
    flat = value.detach().cpu().contiguous().reshape(-1)
    digest = hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()
    return f"{digest[:24]}{tuple(value.shape)}{value.dtype}"


def _scrub(obj: Any) -> Any:
    """Drop timing and identity fields from an agent JSON dump."""

    if isinstance(obj, dict):
        return {
            key: _scrub(value)
            for key, value in sorted(obj.items())
            if not any(token in str(key).lower() for token in _VOLATILE_KEYS)
        }
    if isinstance(obj, list):
        return [_scrub(value) for value in obj]
    return obj


def _fire_row(record: Any) -> tuple[Any, ...]:
    """Return the comparable fields of one fire record (everything but the timestamp)."""

    return (
        record.target_label,
        record.call_label,
        record.func_call_id,
        tuple(record.container_path),
        record.engine,
        record.site_label,
        record.timing,
        record.direction,
        record.helper_name,
        record.seed,
    )


@contextmanager
def _fire_spy() -> Iterator[list[tuple[bool, int]]]:
    """Record ``(module_boundary, n_fires)`` for every live-hook visit that fired.

    Yields
    ------
    list[tuple[bool, int]]
        One row per ``_apply_live_hooks`` call that returned fire results.
    """

    rows: list[tuple[bool, int]] = []
    original = intervention_runtime._apply_live_hooks

    def spy(*args: Any, **kwargs: Any) -> Any:
        """Forward to the real hook runner and note its fires."""

        hooked, fire_results = original(*args, **kwargs)
        if fire_results:
            site = kwargs.get("site")
            rows.append((bool(getattr(site, "_tl_module_boundary", False)), len(fire_results)))
        return hooked, fire_results

    intervention_runtime._apply_live_hooks = spy
    try:
        yield rows
    finally:
        intervention_runtime._apply_live_hooks = original


@contextmanager
def _legacy_door(force: bool) -> Iterator[None]:
    """Force the per-op check on by making both door predicates answer True.

    ``raising=False`` semantics: on a base without the predicates the per-op path is the only
    path, so forcing is a no-op there.
    """

    names = ("_intervene_reaches_op_door", "_hook_plan_reaches_op_door")
    saved = {name: torch_ops.__dict__.get(name) for name in names}
    if force:
        for name in names:
            setattr(torch_ops, name, lambda *_args, **_kwargs: True)
    try:
        yield
    finally:
        if force:
            for name, value in saved.items():
                if value is None:
                    delattr(torch_ops, name)
                else:
                    setattr(torch_ops, name, value)


def _invariants_outcome(trace: Any) -> str:
    """Return the metadata-invariant verdict of a trace as a comparable string."""

    try:
        return f"ok:{bool(trace.check_metadata_invariants())}"
    except Exception as exc:  # the verdict, not the exception object, is compared
        return f"raised:{type(exc).__name__}:{exc}"


def _trace_fingerprint(spec_name: str, *, force_legacy: bool) -> dict[str, Any]:
    """Capture with ``tl.trace(intervene=)`` and fingerprint every observable output."""

    spec = _SPECS[spec_name][0]()
    with (
        _legacy_door(force_legacy),
        _fire_spy() as fires,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        trace = tl.trace(_model(), _input(), intervene=spec)
    saved: dict[str, str] = {}
    for op in trace.ops:
        try:
            saved[str(op.label)] = _thash(op.out)
        except Exception as exc:  # unsaved payloads refuse typed
            saved[str(op.label)] = f"<refused {type(exc).__name__}>"
    dump = json.dumps(_scrub(trace.to_agent_json(max_ops=None)), sort_keys=True, default=str)
    return {
        "labels": [str(op.label) for op in trace.ops],
        "saved": saved,
        "fire_records": {
            str(op.label): [_fire_row(record) for record in op.interventions]
            for op in trace.ops
            if op.interventions
        },
        "fires": fires,
        "output": _thash(trace[trace.output_layers[0]].out),
        "agent_json_sha": hashlib.sha256(dump.encode()).hexdigest(),
        "invariants": _invariants_outcome(trace),
        "warnings": [(w.category.__name__, str(w.message)) for w in caught],
    }


def _record_fingerprint(spec_name: str, *, force_legacy: bool) -> dict[str, Any]:
    """Capture with ``tl.record(intervene=)`` and fingerprint every observable output."""

    spec = _SPECS[spec_name][0]()
    with (
        _legacy_door(force_legacy),
        _fire_spy() as fires,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        output, recording = tl.record(
            _model(), _input(), save=lambda ctx: True, intervene=spec, return_output=True
        )
    return {
        "records": [
            (str(r.ctx.label), str(r.ctx.layer_type), _thash(r.ram_payload))
            for r in recording.records
        ],
        "n_ops": recording.n_ops,
        "status": recording.status,
        "fires": fires,
        "output": _thash(output),
        "warnings": [(w.category.__name__, str(w.message)) for w in caught],
    }


# ---------------------------------------------------------------------------
# 1. Eligibility predicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selector", "exit_only"),
    [
        pytest.param(tl.module("block1"), True, id="module"),
        pytest.param(tl.module("block1:2"), True, id="module_pass_label"),
        pytest.param(tl.module("block1") | tl.module("block2"), True, id="module_or_module"),
        pytest.param(tl.module("block1") & tl.module("block2"), True, id="module_and_module"),
        pytest.param(
            (tl.module("block1") | tl.module("block2")) & tl.module("head"),
            True,
            id="nested_module_composite",
        ),
        pytest.param(~tl.module("block1"), False, id="negation"),
        pytest.param(tl.func("relu"), False, id="func"),
        pytest.param(tl.in_module("block1"), False, id="in_module"),
        pytest.param(tl.where(lambda site: True), False, id="callable"),
        pytest.param(tl.module("block1") & tl.func("relu"), False, id="module_and_func"),
        pytest.param(tl.module("block1") | tl.func("relu"), False, id="module_or_func"),
        pytest.param(tl.label("relu_1_3"), False, id="label"),
        pytest.param(CompositeSelector("or", ()), False, id="empty_or"),
        pytest.param(CompositeSelector("and", ()), False, id="empty_and"),
        pytest.param(
            CompositeSelector("or", (tl.module("block1"), CompositeSelector("and", ()))),
            False,
            id="composite_with_empty_child",
        ),
        pytest.param("block1", False, id="bare_string"),
    ],
)
def test_module_exit_only_selector_classification(selector: Any, exit_only: bool) -> None:
    """Only pure ``tl.module`` leaves under ``|`` / ``&`` are module-exit-only."""

    assert _ops_interventions._is_module_exit_only(selector) is exit_only


@pytest.mark.parametrize(
    ("intervene", "reaches"),
    [
        pytest.param(tl.when(tl.module("block1"), tl.zero_ablate()), False, id="module_rule"),
        pytest.param(
            tl.when(tl.module("block1") | tl.module("block2"), tl.zero_ablate()),
            False,
            id="union_rule",
        ),
        pytest.param(
            tl.when(tl.module("block1"), tl.zero_ablate()).merge(
                tl.when(tl.module("block2"), tl.scale(0.5))
            ),
            False,
            id="two_module_rules",
        ),
        pytest.param(
            tl.when(tl.module("block1"), tl.zero_ablate()).merge(
                tl.when(tl.func("relu"), tl.scale(0.5))
            ),
            True,
            id="module_rule_plus_func_rule",
        ),
        pytest.param(tl.when(tl.func("relu"), tl.zero_ablate()), True, id="func_rule"),
        pytest.param(
            tl.when(tl.module("block1") & tl.func("relu"), tl.zero_ablate()),
            True,
            id="module_and_func_rule",
        ),
        pytest.param(InterventionSpec(rules=()), True, id="empty_rules"),
        pytest.param(lambda ctx: None, True, id="opaque_callable"),
    ],
)
def test_intervene_operand_door_classification(intervene: Any, reaches: bool) -> None:
    """Only an ``InterventionSpec`` whose every rule is module-exit-only skips the op door."""

    options = SimpleNamespace(intervene=intervene)
    assert _ops_interventions._intervene_reaches_op_door(options) is reaches


def _entry(
    site_target: Any,
    *,
    metadata: dict[str, Any] | None = None,
    helper_spec: Any = None,
) -> NormalizedHookEntry:
    """Build one normalized hook-plan entry with an identity callable."""

    return NormalizedHookEntry(
        site_target=site_target,
        normalized_callable=lambda out, *, hook: out,
        helper_spec=helper_spec,
        metadata=dict(metadata or {}),
    )


_INPUT_SPLICE = SimpleNamespace(name="splice_module", metadata={"input": "in"})


@pytest.mark.parametrize(
    ("entry", "reaches"),
    [
        pytest.param(_entry(tl.module("block1")), False, id="module_post_hook"),
        pytest.param(
            _entry(tl.module("block1") | tl.module("block2")), False, id="module_union_post_hook"
        ),
        pytest.param(_entry(tl.func("relu")), True, id="func_post_hook"),
        pytest.param(_entry(tl.module("block1") & tl.func("relu")), True, id="module_and_func"),
        pytest.param(_entry(tl.label("relu_1_3")), True, id="label_post_hook"),
        pytest.param(_entry(~tl.module("block1")), True, id="negated_module"),
        pytest.param(
            _entry(tl.func("relu"), metadata={"direction": "backward"}),
            False,
            id="backward_entry",
        ),
        pytest.param(
            _entry(tl.func("relu"), metadata={"timing": "pre"}), False, id="pre_hook_entry"
        ),
        pytest.param(
            _entry(tl.in_module("block1"), helper_spec=_INPUT_SPLICE),
            False,
            id="input_splice_plain_in_module",
        ),
        pytest.param(
            _entry(tl.module("block1"), helper_spec=_INPUT_SPLICE),
            False,
            id="input_splice_plain_module",
        ),
        pytest.param(
            _entry(tl.in_module("block1") & tl.func("relu"), helper_spec=_INPUT_SPLICE),
            True,
            id="input_splice_non_plain",
        ),
        pytest.param(
            SimpleNamespace(site_target=tl.module("block1"), metadata={}),
            True,
            id="not_a_normalized_entry",
        ),
    ],
)
def test_hook_entry_door_classification(entry: Any, reaches: bool) -> None:
    """A hook-plan entry skips the op door only where the op-door loop would skip it."""

    assert _ops_interventions._hook_entry_reaches_op_door(entry) is reaches


def test_hook_plan_reaches_op_door_when_any_entry_does() -> None:
    """One op-capable entry keeps the whole plan on the per-op path."""

    module_only = [_entry(tl.module("block1")), _entry(tl.module("block2"))]
    assert _ops_interventions._hook_plan_reaches_op_door(module_only) is False
    assert (
        _ops_interventions._hook_plan_reaches_op_door([*module_only, _entry(tl.func("relu"))])
        is True
    )


# ---------------------------------------------------------------------------
# 2. Equality with the per-op path forced on
# ---------------------------------------------------------------------------


@pytest.mark.smoke_cells("test_skip_matches_legacy_path[trace-steer]")
@pytest.mark.parametrize("spec_name", list(_SPECS))
@pytest.mark.parametrize("entry", ["trace", "record"])
def test_skip_matches_legacy_path(entry: str, spec_name: str) -> None:
    """Every observable output equals the per-op path's, skipped or not."""

    fingerprint = _trace_fingerprint if entry == "trace" else _record_fingerprint
    skipped = fingerprint(spec_name, force_legacy=False)
    legacy = fingerprint(spec_name, force_legacy=True)
    assert skipped == legacy
    fires = _SPECS[spec_name][2]
    assert bool(skipped["fires"]) is fires
    if not fires and entry == "trace":
        assert any("matched zero sites" in text for _cat, text in skipped["warnings"])


def test_validation_verdicts_unchanged_by_the_door() -> None:
    """``tl.validate`` forward and saved scopes give the same verdicts with the door forced on."""

    verdicts = {}
    for force in (False, True):
        with _legacy_door(force):
            verdicts[force] = (
                tl.validate(_model(), _input(), scope="forward"),
                tl.validate(_model(), _input(), scope="saved"),
            )
    assert verdicts[False] == verdicts[True] == (True, True)


# ---------------------------------------------------------------------------
# 3. Deterministic per-op call counts
# ---------------------------------------------------------------------------

_COUNTED = ("make_live_site_proxy", "_build_shared_fields_dict", "_evaluate_intervene_op")


@contextmanager
def _call_counter() -> Iterator[dict[str, int]]:
    """Count calls to the per-op hot-path functions as bound in the torch ops module.

    Yields
    ------
    dict[str, int]
        Calls per function name.
    """

    counts = dict.fromkeys(_COUNTED, 0)
    originals = {name: getattr(torch_ops, name) for name in _COUNTED}

    def counting(name: str) -> Callable[..., Any]:
        """Wrap one function with a counter."""

        original = originals[name]

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Count, then forward."""

            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for name in _COUNTED:
        setattr(torch_ops, name, counting(name))
    try:
        yield counts
    finally:
        for name, original in originals.items():
            setattr(torch_ops, name, original)


def _trace_counts(spec_name: str | None) -> tuple[dict[str, int], int]:
    """Return call counts and the logged-op count of one ``tl.trace`` capture."""

    kwargs = {} if spec_name is None else {"intervene": _SPECS[spec_name][0]()}
    with _call_counter() as counts:
        trace = tl.trace(_model(), _input(), **kwargs)
    return counts, len(trace.ops)


def _record_counts(spec_name: str) -> dict[str, int]:
    """Return call counts of one ``tl.record`` capture."""

    with _call_counter() as counts:
        tl.record(_model(), _input(), save=tl.module("block1"), intervene=_SPECS[spec_name][0]())
    return counts


# Per-op path counts, measured on the base before the skip (2.36.1 line). There the module-only
# steer cost make_live_site_proxy 9, _build_shared_fields_dict 61 on tl.trace and
# _evaluate_intervene_op 9 on tl.record; the controls keep the per-op path, call for call.
_FUNC_STEER_TRACE_COUNTS = {
    "make_live_site_proxy": 9,
    "_build_shared_fields_dict": 45,
    "_evaluate_intervene_op": 9,
}
_FUNC_STEER_RECORD_COUNTS = {
    "make_live_site_proxy": 3,
    "_build_shared_fields_dict": 0,
    "_evaluate_intervene_op": 9,
}
# With the skip: shared fields are built by emission only, once per logged op.
_SKIP_STEER_TRACE_COUNTS = {
    "make_live_site_proxy": 0,
    "_build_shared_fields_dict": 9,
    "_evaluate_intervene_op": 0,
}


@pytest.mark.smoke
def test_module_only_spec_builds_no_per_op_site() -> None:
    """A module-only steer builds each op's shared fields at most once and no op-door site."""

    steered, n_ops = _trace_counts("steer")
    assert steered == _SKIP_STEER_TRACE_COUNTS
    assert steered["_build_shared_fields_dict"] <= n_ops
    assert _record_counts("steer") == dict.fromkeys(_COUNTED, 0)
    assert _trace_counts("func_steer")[0] == _FUNC_STEER_TRACE_COUNTS
    assert _record_counts("func_steer") == _FUNC_STEER_RECORD_COUNTS


# ---------------------------------------------------------------------------
# 4. The intervene-operand memo
# ---------------------------------------------------------------------------


def test_op_door_memo_is_identity_keyed_bounded_and_weak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal specs do not share an entry, the memo stays bounded, and it keeps nothing alive."""

    memo: dict[int, Any] = {}
    monkeypatch.setattr(_ops_interventions, "_OP_DOOR_MEMO", memo)
    limit = _ops_interventions._OP_DOOR_MEMO_LIMIT

    def spec() -> Any:
        """Build a fresh module-only spec."""

        return tl.when(tl.module("block1"), tl.zero_ablate())

    first, second = spec(), spec()
    for operand in (first, second, first, second):
        assert (
            _ops_interventions._intervene_reaches_op_door(SimpleNamespace(intervene=operand))
            is False
        )
    assert set(memo) == {id(first), id(second)}

    alive = [spec() for _ in range(limit + 5)]
    for operand in alive:
        _ops_interventions._intervene_reaches_op_door(SimpleNamespace(intervene=operand))
        assert len(memo) <= limit

    doomed = spec()
    ref = weakref.ref(doomed)
    _ops_interventions._intervene_reaches_op_door(SimpleNamespace(intervene=doomed))
    del doomed
    gc.collect()
    assert ref() is None, "the memo must not keep an intervene operand alive"


# ---------------------------------------------------------------------------
# 5. The predicate door on tl.record
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_record_module_steer_fires_once_per_module_call() -> None:
    """``tl.record`` fires a module-only steer once per call, at the boundary, like a plain hook."""

    with _fire_spy() as fires:
        output, recording = tl.record(
            _model(),
            _input(),
            save=tl.module("block1"),
            intervene=_SPECS["steer"][0](),
            return_output=True,
        )
    assert fires == [(True, 1), (True, 1)]  # block1 runs twice; no op-door fire

    # The same steer as a plain torch forward hook, run eagerly: its replaced outputs are the
    # oracle for the saved payloads, bit for bit.
    model = _model()
    shift = _direction() * _MAGNITUDE
    hooked_outputs: list[torch.Tensor] = []

    def plain_steer(_module: nn.Module, _inputs: Any, out: torch.Tensor) -> torch.Tensor:
        """Add the steering shift and keep the replaced output."""

        replaced = out + shift
        hooked_outputs.append(replaced)
        return replaced

    handle = model.block1.register_forward_hook(plain_steer)
    try:
        with torch.no_grad():
            eager_output = model(_input())
    finally:
        handle.remove()
    assert torch.equal(output, eager_output)
    assert [_thash(r.ram_payload) for r in recording.records] == [
        _thash(value) for value in hooked_outputs
    ]
