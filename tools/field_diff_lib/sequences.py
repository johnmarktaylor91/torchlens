"""Behavioural sequences: outcomes that depend on when a field is first read.

A field dump taken after everything is computed cannot see an error that
depends on the order of reads, on a mutation made before a first read, or on
what is still referenced. Each sequence here performs a fixed series of
steps and records, per step and in order, the canonical result or the raised
error (type, code, message) and every warning. The position of a step in the
log is its timing. The log is compared between two commits by the field
differential, and ``tests/test_capture_core_sequences.py`` asserts the
invariants that hold at one commit.
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import os
import pickle
import tempfile
import warnings
import weakref
from collections import Counter
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .canonical import Canonicalizer, error_record
from .cases import SEED, build_model
from .compare import diff, field_key
from .mask import Mask
from .seq_models import ARG_MODEL_SOURCE, SOURCE_MODEL_TEXT
from .surface import (
    REFERENCED_CLASSES,
    Reader,
    field_order,
    public_names,
    snapshot_trace,
    storage_attrs,
    trace_families,
)

#: Field families by name fragment, checked in order; the first match wins.
FAMILY_FRAGMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source", ("source", "code", "call_stack", "func_call", "line", "file")),
    ("arguments", ("arg", "kwarg")),
    ("module_context", ("module",)),
    ("relations", ("parent", "child", "ancestor", "descendant", "sibling", "spouse", "dist")),
    ("labels", ("label", "name", "index", "num")),
    ("tensor_stats", ("shape", "dtype", "memory", "size", "bytes", "out", "grad")),
)
FAMILIES = tuple(name for name, _ in FAMILY_FRAGMENTS) + ("other",)


def family_of(field: str) -> str:
    """Return the family of a field name."""

    for family, fragments in FAMILY_FRAGMENTS:
        if any(fragment in field for fragment in fragments):
            return family
    return "other"


class Log:
    """The ordered outcome log of one sequence.

    Parameters
    ----------
    roots:
        Path prefixes replaced inside strings.
    """

    def __init__(self, roots: list[tuple[str, str]]) -> None:
        """Start an empty log."""

        self.roots = roots
        self.steps: list[dict[str, Any]] = []
        self._canon = Canonicalizer(roots, REFERENCED_CLASSES)

    def step(self, name: str, action: Callable[[], Any]) -> Any:
        """Run one step; record its result or error and its warnings; return the raw result."""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = action()
                outcome = self._canon.value(value)
            except Exception as error:  # noqa: BLE001 - a raising step is an outcome
                value, outcome = None, error_record(error, self._canon.string)
        self.steps.append(
            {
                "step": name,
                "outcome": outcome,
                "warnings": [
                    [w.category.__name__, self._canon.string(str(w.message))[:200]] for w in caught
                ],
            }
        )
        return value

    def note(self, name: str, value: Any) -> None:
        """Record an already-canonical value as a step."""

        self.steps.append({"step": name, "outcome": value, "warnings": []})

    def outcome(self, name: str) -> Any:
        """Return the recorded outcome of the step called ``name``."""

        for step in self.steps:
            if step["step"] == name:
                return step["outcome"]
        raise KeyError(name)


def fresh(tl: Any, model_name: str = "inplace_buffers", **options: Any) -> tuple[Any, Any, Any]:
    """Build a model under the fixed seed and capture it; return ``(model, input, trace)``."""

    model, model_input = build_model(model_name)
    capture = tl.options.CaptureOptions(seed=SEED, **options)
    return model, model_input, tl.trace(model, model_input, capture=capture)


def snap(trace: Any, roots: list[tuple[str, str]]) -> dict[str, Any]:
    """Dump one trace under the fixed mask."""

    return snapshot_trace(trace, roots)[0]


def paths(base: Any, head: Any) -> set[str]:
    """Return the set of paths at which two dumps differ."""

    return {path for path, _, _ in diff(base, head)}


def summary(found: set[str]) -> dict[str, int]:
    """Count differences per ``family/field``."""

    return dict(sorted(Counter(field_key(path) for path in found).items()))


def read_fields(trace: Any, wanted: Callable[[str], bool]) -> int:
    """Read every public field whose name ``wanted`` accepts, on every record; return the count."""

    count = 0
    records = [trace]
    for family in trace_families(trace).values():
        if isinstance(family, dict):
            records.extend(family.values())
    for record in records:
        storage = storage_attrs(type(record).__name__)
        for name in public_names(type(record)):
            if name in ("receptive_field", "projective_field") or not wanted(name):
                continue
            count += 1
            try:
                getattr(record, storage.get(name, name))
            except Exception:  # noqa: BLE001, S112 - only the read order matters here
                continue
    return count


def seq_read_orders(tl: Any, log: Log, scratch: str) -> None:
    """Sequence 1: each family read first, last, and after another family; the dump must not move."""

    reference = snap(fresh(tl)[2], log.roots)
    control = paths(reference, snap(fresh(tl)[2], log.roots))
    log.note("control.varies_between_two_fresh_traces", summary(control))
    for position, family in enumerate(FAMILIES):
        other = FAMILIES[(position + 1) % len(FAMILIES)]
        orders: dict[str, list[Callable[[str], bool]]] = {
            "first": [lambda name, f=family: family_of(name) == f],
            "last": [
                lambda name, f=family: family_of(name) != f,
                lambda name, f=family: family_of(name) == f,
            ],
            "after_other": [
                lambda name, o=other: family_of(name) == o,
                lambda name, f=family: family_of(name) == f,
            ],
        }
        for order, passes in orders.items():
            trace = fresh(tl)[2]
            for wanted in passes:
                read_fields(trace, wanted)
            moved = paths(reference, snap(trace, log.roots)) - control
            log.note(f"{family}.{order}.moved", summary(moved))


def _checks(log: Log, prefix: str, trace: Any, op_label: str, layer_label: str) -> None:
    """Read the overridden, annotated and deleted fields of one op, one layer and the trace."""

    log.step(f"{prefix}.op.func_name", lambda: trace.ops[op_label].func_name)
    log.step(f"{prefix}.op.annotations", lambda: trace.ops[op_label].annotations)
    log.step(f"{prefix}.op.deleted_shape", lambda: trace.ops[op_label].shape)
    log.step(f"{prefix}.layer.func_name", lambda: trace.layers[layer_label].func_name)
    log.step(f"{prefix}.layer.deleted_shape", lambda: trace.layers[layer_label].shape)
    log.step(f"{prefix}.trace.name", lambda: trace.name)


def _mutate(log: Log, trace: Any, op_label: str, layer_label: str) -> None:
    """Override, annotate and delete on one op, one layer and the trace; hold nothing after."""

    op = trace.ops[op_label]
    layer = trace.layers[layer_label]
    log.step("op.override", lambda: setattr(op, "func_name", "H3_OVERRIDE"))
    log.step("layer.override", lambda: setattr(layer, "func_name", "H3_LAYER_OVERRIDE"))
    log.step("trace.override", lambda: setattr(trace, "name", "h3_renamed"))
    log.step("trace.annotate", lambda: trace.annotate(op_label, data={"h3": 1}) and None)
    log.step("op.delete", lambda: delattr(op, "shape"))
    log.step("op.delete_again", lambda: delattr(op, "shape"))
    log.step("op.read_deleted", lambda: op.shape)
    log.step("layer.delete", lambda: delattr(layer, "shape"))
    log.step("layer.delete_again", lambda: delattr(layer, "shape"))
    log.step("layer.read_deleted", lambda: layer.shape)


def _new_attributes(tl: Any, log: Log, scratch: str) -> None:
    """Write a name that is not a field on an op, a layer and the trace, then save."""

    trace = fresh(tl)[2]
    op_label = list(trace.ops.keys())[1]
    layer_label = str(trace.ops[op_label].layer_label)
    for name, take in (
        ("op", lambda: trace.ops[op_label]),
        ("layer", lambda: trace.layers[layer_label]),
        ("trace", lambda: trace),
    ):
        log.step(f"new.{name}.set", lambda take=take: setattr(take(), "h3_note", "h3 note"))
        log.step(f"new.{name}.read", lambda take=take: take().h3_note)
    gc.collect()
    log.step("new.op.read_other_door", lambda: trace[op_label].h3_note)
    log.step("new.layer.read_other_door", lambda: trace[layer_label].h3_note)
    fork = log.step("new.fork", trace.fork)
    if fork is not None:
        log.step("new.fork.layer.read", lambda: fork.layers[layer_label].h3_note)
        log.step("new.fork.trace.read", lambda: fork.h3_note)
    path = os.path.join(tempfile.mkdtemp(dir=scratch), "new_attributes.tlspec")
    log.step("new.save", lambda: trace.save(path) and None)


def seq_mutate_before_read(tl: Any, log: Log, scratch: str) -> None:
    """Sequence 2: override, annotate and delete before a first read; then other doors and copies."""

    trace = fresh(tl)[2]
    keys = list(trace.ops.keys())
    op_label = keys[1]
    layer_label = str(trace.ops[keys[3]].layer_label)
    log.note("labels", [op_label, layer_label])
    _mutate(log, trace, op_label, layer_label)
    log.note("read_rest", snap(trace, log.roots))
    gc.collect()
    _checks(log, "other_door", trace, op_label, layer_label)
    log.step("plain_door.op.func_name", lambda: trace[op_label].func_name)
    log.step("plain_door.layer.func_name", lambda: trace[layer_label].func_name)
    log.step(
        "list_door.op.func_name",
        lambda: next(o for o in trace._layer_list if o.label == op_label).func_name,  # noqa: SLF001
    )
    fork = log.step("fork", trace.fork)
    if fork is not None:
        _checks(log, "fork", fork, op_label, layer_label)
    path = os.path.join(tempfile.mkdtemp(dir=scratch), "mutated.tlspec")
    log.step("save", lambda: trace.save(path) and None)
    loaded = log.step("load", lambda: tl.load(path))
    if loaded is not None:
        _checks(log, "loaded", loaded, op_label, layer_label)
    pickled = log.step("pickle", lambda: pickle.loads(pickle.dumps(trace)))  # noqa: S301
    if pickled is not None:
        _checks(log, "pickled", pickled, op_label, layer_label)
    copied = log.step("deepcopy", lambda: copy.deepcopy(trace))
    if copied is not None:
        _checks(log, "deepcopied", copied, op_label, layer_label)
    _new_attributes(tl, log, scratch)


def _source_model(scratch: str, name: str) -> tuple[str, nn.Module]:
    """Write the source-sequence model to its own file and import it."""

    folder = tempfile.mkdtemp(dir=scratch)
    path = os.path.join(folder, f"{name}.py")
    with open(path, "w") as handle:
        handle.write(SOURCE_MODEL_TEXT)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(SEED)
    return path, module.SourceModel().eval()


def _source_fields(trace: Any, canon: Canonicalizer) -> dict[str, Any]:
    """Read the source-family fields of every op."""

    names = [name for name in field_order("Op") if family_of(name) == "source"]
    storage = storage_attrs("Op")
    mask = Mask()
    result: dict[str, Any] = {}
    for op in trace._layer_list:  # noqa: SLF001
        for name in names:
            try:
                value = mask.apply("Op", name, canon.value(getattr(op, storage.get(name, name))))
            except Exception as error:  # noqa: BLE001 - a raising read is an outcome
                value = error_record(error, canon.string)
            result[f"{op.label}.{name}"] = value
    return result


def seq_source_changes(tl: Any, log: Log, scratch: str) -> None:
    """Sequence 3: edit or delete the model's source file between capture and the first source read."""

    canon = Canonicalizer(log.roots, REFERENCED_CLASSES)
    for variant in ("unchanged", "edited", "deleted"):
        path, model = _source_model(scratch, f"h3_source_{variant}")
        capture = tl.options.CaptureOptions(seed=SEED, save_source_context=True)
        trace = log.step(
            f"{variant}.capture", lambda: tl.trace(model, torch.ones(2, 4), capture=capture)
        )
        if trace is None:
            continue
        if variant == "edited":
            with open(path, "w") as handle:
                handle.write("# two lines inserted after capture\n# by the sequence\n")
                handle.write(SOURCE_MODEL_TEXT.replace("torch.relu", "torch.tanh"))
        elif variant == "deleted":
            os.remove(path)
        log.step(f"{variant}.source_fields", lambda: _source_fields(trace, canon))
        log.step(f"{variant}.repr", lambda: canon.string(repr(trace)))


def _argument_fields(trace: Any, canon: Canonicalizer) -> dict[str, Any]:
    """Read the argument-family fields of every op and module call."""

    mask = Mask()
    result: dict[str, Any] = {}
    families = trace_families(trace)
    for family in ("ops", "module_calls"):
        records = families.get(family)
        if not isinstance(records, dict):
            continue
        for key, record in records.items():
            kind = type(record).__name__
            storage = storage_attrs(kind)
            for name in field_order(kind):
                if family_of(name) != "arguments":
                    continue
                try:
                    raw = canon.value(getattr(record, storage.get(name, name)))
                    value = mask.apply(kind, name, raw)
                except Exception as error:  # noqa: BLE001 - a raising read is an outcome
                    value = error_record(error, canon.string)
                result[f"{family}[{key}].{name}"] = value
    return result


def seq_arguments(tl: Any, log: Log, scratch: str) -> None:
    """Sequence 4: mutate a list, a dictionary and a tensor after the call, then read the arguments."""

    namespace: dict[str, Any] = {"torch": torch, "nn": nn}
    exec(compile(ARG_MODEL_SOURCE, "<h3-arg-model>", "exec"), namespace)  # noqa: S102
    for save_args in (False, True):
        tag = f"save_args={save_args}"
        torch.manual_seed(SEED)
        model = namespace["ArgModel"]().eval()
        model_input = torch.ones(2, 4)
        capture = tl.options.CaptureOptions(seed=SEED, save_args=save_args)
        trace = log.step(f"{tag}.capture", lambda: tl.trace(model, model_input, capture=capture))
        if trace is None:
            continue
        with torch.no_grad():
            model_input.zero_()
        model.held_list.clear()
        model.held_dict.clear()
        canon = Canonicalizer(log.roots, REFERENCED_CLASSES)
        log.step(f"{tag}.arguments", lambda: _argument_fields(trace, canon))


def seq_forks(tl: Any, log: Log, scratch: str) -> None:
    """Sequence 5: fork before any read and after every read; edit; replace a module; rerun."""

    snaps: dict[str, Any] = {}
    for timing in ("cold", "warm"):
        model, model_input, trace = fresh(tl, "two_relu")
        if timing == "warm":
            snap(trace, log.roots)
        fork = log.step(f"{timing}.fork", trace.fork)
        edited = log.step(f"{timing}.fork_for_edit", trace.fork)
        log.step(f"{timing}.edit", lambda: edited.do(tl.func("relu"), tl.zero_ablate()) and None)
        replaced = log.step(f"{timing}.fork_for_module", trace.fork)
        log.step(
            f"{timing}.replace_module",
            lambda: replaced.do(tl.module("fc"), tl.zero_ablate()) and None,
        )
        rerun = log.step(f"{timing}.rerun", lambda: trace.run(model, model_input))
        for name, item in (
            ("fork", fork),
            ("edited", edited),
            ("replaced", replaced),
            ("rerun", rerun),
            ("parent", trace),
        ):
            if item is not None:
                snaps[f"{timing}.{name}"] = snap(item, log.roots)
        for name in ("fork", "edited", "replaced", "rerun"):
            if f"{timing}.{name}" in snaps:
                found = paths(snaps[f"{timing}.parent"], snaps[f"{timing}.{name}"])
                log.note(f"{timing}.parent_vs_{name}", summary(found))
    for name in ("fork", "edited", "replaced", "rerun", "parent"):
        if f"cold.{name}" in snaps and f"warm.{name}" in snaps:
            found = paths(snaps[f"cold.{name}"], snaps[f"warm.{name}"])
            log.note(f"cold_vs_warm.{name}", summary(found))


def seq_serialisation(tl: Any, log: Log, scratch: str) -> None:
    """Sequence 6: save, pickle and deepcopy a cold trace and a fully read one; load and compare."""

    snaps: dict[str, Any] = {}
    for timing in ("cold", "warm"):
        trace = fresh(tl)[2]
        if timing == "warm":
            snap(trace, log.roots)
        path = os.path.join(tempfile.mkdtemp(dir=scratch), f"{timing}.tlspec")
        log.step(f"{timing}.save", lambda: trace.save(path) and None)
        copies = {
            "loaded": log.step(f"{timing}.load", lambda: tl.load(path)),
            "pickled": log.step(f"{timing}.pickle", lambda: pickle.loads(pickle.dumps(trace))),  # noqa: S301
            "deepcopied": log.step(f"{timing}.deepcopy", lambda: copy.deepcopy(trace)),
        }
        snaps[f"{timing}.live"] = snap(trace, log.roots)
        for name, item in copies.items():
            if item is None:
                continue
            snaps[f"{timing}.{name}"] = snap(item, log.roots)
            found = paths(snaps[f"{timing}.live"], snaps[f"{timing}.{name}"])
            log.note(f"{timing}.live_vs_{name}", summary(found))
    control = paths(snaps["cold.live"], snaps["warm.live"])
    log.note("cold_vs_warm.live", summary(control))
    for name in ("loaded", "pickled", "deepcopied"):
        if f"cold.{name}" in snaps and f"warm.{name}" in snaps:
            found = paths(snaps[f"cold.{name}"], snaps[f"warm.{name}"]) - control
            log.note(f"cold_vs_warm.{name}", summary(found))


def _bulk(trace: Any, roots: list[tuple[str, str]]) -> dict[str, Any]:
    """Run the readers that touch everything and return their results under the fixed mask."""

    canon = Canonicalizer(roots, REFERENCED_CLASSES)
    mask = Mask()
    reader = Reader(canon, mask)

    def table(read: Callable[[], Any], site: str) -> Any:
        try:
            return reader.table(read(), site)
        except Exception as error:  # noqa: BLE001 - a refusing reader is an outcome
            return error_record(error, canon.string)

    result: dict[str, Any] = {
        "trace.to_pandas": table(trace.to_pandas, "trace.to_pandas"),
        "ops.to_pandas": table(lambda: trace.ops.to_pandas(), "ops.to_pandas"),
        "layers.to_pandas": table(lambda: trace.layers.to_pandas(), "layers.to_pandas"),
        "modules.to_pandas": table(lambda: trace.modules.to_pandas(), "modules.to_pandas"),
        "repr": reader.text(trace, repr, "trace.__repr__"),
        "str": reader.text(trace, str, "trace.__str__"),
    }
    facets: dict[str, Any] = {}
    for op in trace._layer_list:  # noqa: SLF001
        try:
            facets[str(op.label)] = mask.apply("Op", "facets", canon.value(op.facets))
        except Exception as error:  # noqa: BLE001 - a refusing facet read is an outcome
            facets[str(op.label)] = error_record(error, canon.string)
    result["facets"] = facets
    return result


def seq_facets_bulk(tl: Any, log: Log, scratch: str) -> None:
    """Sequence 7: facets and bulk readers before and after single-field reads."""

    reference = snap(fresh(tl)[2], log.roots)
    control = paths(reference, snap(fresh(tl)[2], log.roots))
    bulk_first = fresh(tl)[2]
    bulk_before = _bulk(bulk_first, log.roots)
    fields_after_bulk = snap(bulk_first, log.roots)
    fields_first = fresh(tl)[2]
    fields_before_bulk = snap(fields_first, log.roots)
    bulk_after = _bulk(fields_first, log.roots)
    log.note("control.varies_between_two_fresh_traces", summary(control))
    log.note("bulk.before_vs_after_field_reads", sorted(paths(bulk_before, bulk_after))[:50])
    log.note(
        "fields.before_vs_after_bulk",
        summary(paths(fields_before_bulk, fields_after_bulk) - control),
    )
    log.note(
        "bulk.refusals",
        {key: value for key, value in bulk_before.items() if "__raises__" in value},
    )
    log.note("bulk.results", bulk_before)


#: Singular names of the record families, as the release sequence reports them.
RECORD_NAMES = {
    "ops": "op",
    "layers": "layer",
    "modules": "module",
    "module_calls": "module_call",
    "params": "param",
    "buffers": "buffer",
    "grad_fns": "grad_fn",
    "backward_passes": "backward_pass",
}


def sample_records(trace: Any, prefix: str = "") -> dict[str, Any]:
    """Take the trace and one record of every family through its door."""

    taken: dict[str, Any] = {f"{prefix}trace": trace}
    for family, records in trace_families(trace).items():
        if isinstance(records, dict) and records:
            taken[prefix + RECORD_NAMES[family]] = list(records.values())[len(records) // 2]
    return taken


def _weak(objects: dict[str, Any]) -> dict[str, Any]:
    """Return a weak reference per named object; ``None`` where the type allows none."""

    refs: dict[str, Any] = {}
    for name, item in objects.items():
        try:
            refs[name] = weakref.ref(item)
        except TypeError:
            refs[name] = None
    return refs


def _dead(refs: dict[str, Any]) -> set[str]:
    """Return the names whose object is gone."""

    return {name for name, ref in refs.items() if ref is not None and ref() is None}


def _freed(make: Callable[[], Any], act: Callable[[Any], tuple[dict[str, Any], Any]]) -> Any:
    """Drop a trace, then what was kept from it, with the collector off; name what died when.

    ``act`` returns the named objects to watch and whatever stays referenced
    after the trace itself is dropped.
    """

    trace = make()
    watched, kept = act(trace)
    refs = _weak(watched)
    del watched
    gc.collect()
    gc.disable()
    try:
        del trace
        first = _dead(refs)
        del kept
        second = _dead(refs)
    finally:
        gc.enable()
    gc.collect()
    third = _dead(refs)
    return {
        "freed_when_trace_dropped": sorted(first),
        "freed_when_kept_dropped": sorted(second - first),
        "freed_only_by_collector": sorted(third - second),
        "never_freed": sorted(set(refs) - third - {n for n, r in refs.items() if r is None}),
        "no_weak_reference": sorted(name for name, ref in refs.items() if ref is None),
    }


Scenario = tuple[Callable[[], Any], Callable[[Any], tuple[dict[str, Any], Any]]]


def release_scenarios(tl: Any, scratch: str) -> dict[str, Scenario]:
    """Return the release scenarios: how the trace is made, what is watched and what is kept."""

    def plain() -> Any:
        return fresh(tl)[2]

    def backward() -> Any:
        trace = fresh(tl, "two_relu", backward_ready=True, save_grads="all")[2]
        trace.backward(trace.output.sum())
        return trace

    def every_door(trace: Any) -> list[Any]:
        label = list(trace.ops.keys())[1]
        families = [f for f in trace_families(trace).values() if isinstance(f, dict)]
        return [trace.ops[label], trace[label], [list(f.values()) for f in families]]

    def after(read: Callable[[Any], Any]) -> Callable[[Any], tuple[dict[str, Any], Any]]:
        def act(trace: Any) -> tuple[dict[str, Any], Any]:
            read(trace)
            return sample_records(trace), None

        return act

    def keeping(take: Callable[[Any], Any]) -> Callable[[Any], tuple[dict[str, Any], Any]]:
        return lambda trace: (sample_records(trace), take(trace))

    def facets(trace: Any) -> tuple[dict[str, Any], Any]:
        views = [op.facets for op in trace._layer_list]  # noqa: SLF001
        return {**sample_records(trace), "facet_view": views[len(views) // 2]}, None

    def derived(build: Callable[[Any], Any], keep: bool) -> Any:
        def act(trace: Any) -> tuple[dict[str, Any], Any]:
            other = build(trace)
            watched = {**sample_records(trace), **sample_records(other, "derived_")}
            return watched, other if keep else None

        return act

    def round_trip(trace: Any) -> Any:
        path = os.path.join(tempfile.mkdtemp(dir=scratch), "release.tlspec")
        trace.save(path)
        return tl.load(path)

    return {
        "no_record_taken": (plain, lambda trace: ({"trace": trace}, None)),
        "records_taken_and_dropped": (plain, after(lambda trace: None)),
        "op_held": (plain, keeping(lambda trace: trace.ops[list(trace.ops.keys())[1]])),
        "layer_held": (plain, keeping(lambda trace: list(trace.layers.values())[1])),
        "modules_held": (
            plain,
            keeping(lambda t: [list(t.modules.values()), list(t.module_calls.values())]),
        ),
        "params_and_buffers_held": (
            plain,
            keeping(lambda t: [list(t.params.values()), list(t.buffers.values())]),
        ),
        "every_door_held": (plain, keeping(every_door)),
        "every_door_held_reversed": (plain, keeping(lambda t: list(reversed(every_door(t))))),
        "all_fields_read": (plain, after(lambda trace: read_fields(trace, bool))),
        "facets_read": (plain, facets),
        "to_pandas_read": (plain, after(lambda trace: trace.to_pandas())),
        "repr_read": (plain, after(repr)),
        "fork_held": (plain, derived(lambda trace: trace.fork(), True)),
        "fork_dropped": (plain, derived(lambda trace: trace.fork(), False)),
        "pickle_copy_dropped": (
            plain,
            derived(lambda trace: pickle.loads(pickle.dumps(trace)), False),  # noqa: S301
        ),
        "loaded_copy_dropped": (plain, derived(round_trip, False)),
        "backward_capture": (backward, after(lambda trace: None)),
        "backward_capture_all_fields_read": (backward, after(lambda t: read_fields(t, bool))),
    }


def seq_release(tl: Any, log: Log, scratch: str) -> None:
    """Sequence 8: which objects reference counting alone frees, per scenario, by name."""

    for name, (make, act) in release_scenarios(tl, scratch).items():
        log.step(name, lambda make=make, act=act: _freed(make, act))
    model, model_input = build_model("two_relu")
    log.step("chained_read", lambda: tl.trace(model, model_input)["relu_1_2:1"].out)


#: The sequences, in the order of the plan.
SEQUENCES: dict[str, Callable[[Any, Log, str], None]] = {
    "1_read_orders": seq_read_orders,
    "2_mutate_before_read": seq_mutate_before_read,
    "3_source_changes": seq_source_changes,
    "4_arguments": seq_arguments,
    "5_forks": seq_forks,
    "6_serialisation": seq_serialisation,
    "7_facets_bulk": seq_facets_bulk,
    "8_release": seq_release,
}


def run_sequences(
    scratch: str, roots: list[tuple[str, str]], names: set[str] | None = None
) -> dict[str, Any]:
    """Run the selected sequences and return ``name -> ordered step log``.

    Parameters
    ----------
    scratch:
        Folder for the files the sequences write.
    roots:
        Path prefixes replaced inside strings.
    names:
        Sequence names to run; all when omitted.

    Returns
    -------
    dict[str, Any]
        Per sequence, ``{"steps": [...]}``, or ``{"sequence_raises": ...}``
        with the steps recorded before the failure.
    """

    import torchlens as tl

    results: dict[str, Any] = {}
    for name, sequence in SEQUENCES.items():
        if names is not None and name not in names:
            continue
        log = Log(roots)
        try:
            sequence(tl, log, scratch)
        except Exception as error:  # noqa: BLE001 - a failing sequence is recorded whole
            results[name] = {"steps": log.steps, "sequence_raises": error_record(error)}
            continue
        results[name] = {"steps": log.steps}
    return results
