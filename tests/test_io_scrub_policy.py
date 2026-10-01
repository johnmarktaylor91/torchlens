"""Completeness lint for portable scrub policy coverage."""

from __future__ import annotations

import functools
import inspect
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens._io.scrub import _RAW_IMAGE_SENTINEL, _RAW_INPUT_IMAGE_BYTES_LIMIT
from torchlens.data_classes._state_adapter import state_items
from torchlens.data_classes.buffer import Buffer
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.op import Op
from torchlens.data_classes.param import Param
from torchlens.data_classes.trace import Trace


class _TinyIOModel(nn.Module):
    """Small model covering every target log class."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny test model."""
        return torch.relu(self.linear(self.bn(x)))


class _PixelOutput:
    """Small image-processor return object."""

    def __init__(self, pixel_values: torch.Tensor) -> None:
        """Store processed image pixels.

        Parameters
        ----------
        pixel_values:
            Processed image tensor.
        """

        self.pixel_values = pixel_values


class _TinyImageInputModel(nn.Module):
    """Small model that accepts auto-coerced PIL image input."""

    def __init__(self) -> None:
        """Initialize the convolution."""

        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def image_processor(self, image: Any, *, return_tensors: str) -> _PixelOutput:
        """Convert a PIL image into a tensor batch.

        Parameters
        ----------
        image:
            Raw image input.
        return_tensors:
            Requested tensor backend.

        Returns
        -------
        _PixelOutput
            Processed pixel tensor.
        """

        del image, return_tensors
        return _PixelOutput(torch.ones(1, 3, 8, 8))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the tiny image model."""

        return self.conv(pixel_values).mean()


def _build_live_log() -> Trace:
    """Create one canonical live ``Trace`` for completeness checks."""

    torch.manual_seed(0)
    model = _TinyIOModel()
    x = torch.randn(2, 4)
    return trace_fn(
        model,
        x,
        capture=tl.options.CaptureOptions(
            layers_to_save="all",
            save_arg_values=True,
            save_rng_states=True,
            save_code_context=True,
            random_seed=0,
        ),
    )


@pytest.mark.parametrize("include_source", [True, False])
def test_backward_source_fields_are_in_the_privacy_belt(include_source: bool) -> None:
    """B8-21: GradFn backward source path/docstring are relativized or dropped.

    Fail-before: the source-privacy belt covered class/init/forward only, so
    ``backward_source_file`` (an absolute path) and ``backward_docstring`` persisted
    verbatim for Python-inspectable custom autograd Functions.
    """

    from torchlens._io.payload_codec import get_payload_codec
    from torchlens._io.scrub import _apply_source_metadata_policy, _ScrubOptions

    scrubbed_state = {
        "class_docstring": "cls doc",
        "backward_source_file": "/home/someone/secret/model.py",
        "backward_docstring": "sensitive backward doc",
    }
    options = _ScrubOptions(
        include_outs=False,
        include_grads=False,
        include_saved_args=False,
        include_rng_states=False,
        include_source=include_source,
        payload_codec=get_payload_codec("torch"),
    )
    _apply_source_metadata_policy(scrubbed_state, options)

    if include_source:
        assert scrubbed_state["backward_source_file"] == "model.py"
        assert "/home/" not in (scrubbed_state["backward_source_file"] or "")
    else:
        assert scrubbed_state["backward_source_file"] is None
        assert scrubbed_state["backward_docstring"] is None


def test_partial_activation_transform_repr_does_not_leak_bound_values(tmp_path: Path) -> None:
    """B8-20: a ``functools.partial`` transform's bound arg values are not persisted.

    Fail-before: ``_activation_transform_repr = repr(fn)`` embedded a partial's bound
    argument VALUES verbatim into ``metadata.pkl`` at every save level; a probe
    recovered a planted token.
    """

    import functools

    secret = "SENSITIVE-ACTIVATION-TOKEN"

    def _identity(t: torch.Tensor, token: str | None = None) -> torch.Tensor:
        return t

    transform = functools.partial(_identity, token=secret)
    trace = trace_fn(
        _TinyIOModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
        save=tl.options.SaveOptions(activation_transform=transform),
    )
    spec = tmp_path / "partial.tlspec"
    tl.save(trace, str(spec))
    for artifact in spec.rglob("*"):
        if artifact.is_file():
            assert secret.encode() not in artifact.read_bytes(), (
                f"partial bound value leaked into {artifact.name}"
            )
    saved_repr = pickle.loads((spec / "metadata.pkl").read_bytes())["_activation_transform_repr"]
    assert "<scrubbed>" in saved_repr
    assert secret not in saved_repr


def test_transform_repr_redacts_heap_addresses() -> None:
    """b3-opus (B8-20 completion): a callable repr's heap address is not persisted.

    A callable using the default object/function repr (a lambda, or a callable
    instance) embeds a live ``0x<hex>`` heap address, which is non-deterministic
    (breaks byte-identical artifacts) and an ASLR-layout leak. The scrub redacts
    it, directly and through a ``functools.partial`` wrapper.
    """

    import functools

    from torchlens.data_classes.trace import _scrubbed_transform_repr

    class _CallableWithDefaultRepr:
        def __call__(self, t: torch.Tensor) -> torch.Tensor:
            return t

    instance = _CallableWithDefaultRepr()
    assert "0x" in repr(instance), "test precondition: default repr has a heap address"

    direct = _scrubbed_transform_repr(instance)
    assert direct is not None
    assert "0x<scrubbed>" in direct
    assert "0x" not in direct.replace("0x<scrubbed>", "")

    wrapped = _scrubbed_transform_repr(functools.partial(instance))
    assert wrapped is not None
    assert "0x" not in wrapped.replace("0x<scrubbed>", "")


def test_portable_state_specs_cover_every_live_attribute() -> None:
    """Each target class must map every live attribute to a scrub policy."""

    live_log = _build_live_log()
    instances = {
        Trace: live_log,
        Op: next(layer for layer in live_log.layer_list if type(layer) is Op),
        Layer: next(iter(live_log.layer_logs.values())),
        Module: next(iter(live_log.modules)),
        ModuleCall: next(iter(live_log.modules._pass_dict.values())),
        Param: next(iter(live_log.param_logs)),
        Buffer: next(iter(live_log.buffers)),
        FuncCallLocation: next(
            frame
            for layer in live_log.layer_list
            for frame in layer.code_context
            if layer.code_context
        ),
    }

    missing_by_class = {}
    for cls, instance in instances.items():
        missing = sorted(
            {field_name for field_name, _ in state_items(instance)} - set(cls.PORTABLE_STATE_SPEC)
        )
        if missing:
            missing_by_class[cls.__name__] = missing

    assert missing_by_class == {}


def _all_record_instances(live_log: Trace) -> list[Any]:
    """Return every record instance a public read could poke state onto."""

    return [
        live_log,
        *live_log.layer_list,
        *live_log.layer_logs.values(),
        *live_log.modules,
        *live_log.modules._pass_dict.values(),
        *live_log.param_logs,
        *live_log.buffers,
    ]


def _sweep_public_accessors(record: Any) -> None:
    """Read every public attribute on ``record``, ignoring raising accessors.

    A raising accessor cannot have handed the user a value, but it may still
    have partially populated a cache before raising, so the sweep never skips
    a name preemptively.
    """

    for name in sorted(set(dir(type(record)))):
        if name.startswith("_"):
            continue
        try:
            getattr(record, name)
        # An accessor that raises is not this sweep's subject (hence the suppressions
        # below). The sweep exists to prove that READING the public surface leaves the trace
        # saveable; logging every unreadable accessor would bury that signal, and a
        # raising accessor cannot populate the live state this test inspects.
        except Exception:  # noqa: BLE001, S112
            continue


def test_public_accessor_reads_never_poison_save(tmp_path: Path) -> None:
    """A read-only public accessor sweep must leave the trace saveable.

    Fail-before: ``ModuleCall.facets`` / ``Module.facets`` cached a FacetView
    in ``__dict__["_facets_cache"]`` with no PORTABLE_STATE_SPEC row, so one
    documented read made every later ``tl.save`` refuse with an error naming
    an internal cache the user never touched. The sibling completeness test
    above checks live state straight after capture, which is exactly why the
    lazily-populated caches slipped past it: they appear only after a read.
    """

    live_log = _build_live_log()
    records = _all_record_instances(live_log)
    for record in records:
        _sweep_public_accessors(record)

    undeclared_by_class: dict[str, list[str]] = {}
    for record in records:
        spec = getattr(type(record), "PORTABLE_STATE_SPEC", None)
        if spec is None:
            continue
        undeclared = sorted(
            field_name
            for field_name, _ in state_items(record)
            if field_name not in spec
            and not isinstance(
                inspect.getattr_static(type(record), field_name, None),
                functools.cached_property,
            )
        )
        if undeclared:
            undeclared_by_class.setdefault(type(record).__name__, undeclared)
    assert undeclared_by_class == {}, (
        "public accessor reads populated live state with no scrub policy; "
        f"declare each field (usually FieldPolicy.DROP): {undeclared_by_class}"
    )

    tl.save(live_log, str(tmp_path / "after_sweep.tlspec"))


def test_intervention_ready_accessor_sweep_still_saves(tmp_path: Path) -> None:
    """The sweep also holds for intervention-ready captures (edge family armed)."""

    log = trace_fn(
        _TinyIOModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    _ = log.edges
    for record in _all_record_instances(log):
        _sweep_public_accessors(record)
    tl.save(log, str(tmp_path / "after_armed_sweep.tlspec"))


# --- Public METHOD sweep (v2 of the accessor tripwire) ----------------------
#
# The attribute sweep above closed the lazily-caching PROPERTY class, but
# ``draw()`` is a METHOD -- it populated ``Trace._last_encoding_state`` with no
# scrub policy, so "visualize then save" failed while the attribute sweep
# stayed green. The method sweep closes the sibling class: call every
# read-only presentation/analysis method a user would plausibly run before
# saving, then assert the trace still saves.
#
# TOTALITY: every public method on every record class must appear in exactly
# one of the two tables below (sweep or exclusion). A new public method fails
# ``test_public_method_sweep_tables_are_total`` until it is classified -- an
# untested method silently omitted is exactly how the draw() gap recurred.

#: Public Trace methods deliberately NOT swept, each with the reason. A
#: reason must name why the call is unsafe or out of scope for a read-only
#: pre-save sweep; "we forgot" is not representable in this table.
_TRACE_METHOD_EXCLUSIONS: dict[str, str] = {
    "annotate": "user annotation writer (mutates trace annotations by contract)",
    "with_annotations": "user annotation writer (mutating fluent spelling)",
    "add_node_overlay": "mutates the overlay table consumed by draw",
    "attach_hooks": "mutates the live model's hook registry",
    "clear_hooks": "mutates the live model's hook registry",
    "detach_hooks": "mutates the live model's hook registry",
    "disarm_triggers": "mutates armed trigger state",
    "backward": "autograd execution; writes gradient state",
    "log_backward": "autograd execution; writes gradient state",
    "recording_backward": "autograd execution; writes gradient state",
    "cleanup": "destructive teardown of the trace",
    "release_param_refs": "destructive; nulls live parameter refs",
    "do": "intervention editor (mutates fork state)",
    "set": "intervention editor (mutates fork state)",
    "remove": "removal scrub (mutates the graph)",
    "push": "replay/push engine; execution, not presentation",
    "push_from": "replay/push engine; execution, not presentation",
    "save_new_outs": "fast re-capture engine; execution, not presentation",
    "replace_state_from": "state mutator (cross-trace state transplant)",
    "append_state_from": "state mutator (cross-trace state transplant)",
    "run": "sparse/live execution engine",
    "preview_fastlog": "fastlog execution",
    "fork": "constructs a new trace; the sweep audits reads on the original",
    "load_state_dict": "stages state onto the trace",
    "save": "the save boundary itself is the sweep's assertion, not a subject",
    "save_intervention": "save boundary (intervention spec artifact)",
    "validate_forward_pass": "full replay validation; execution-tier compute",
    "discharge_against": "requires a second real capture as input",
    "render_dagua_graph": "optional external dagua renderer dependency",
    "to_dagua_graph": "optional external dagua renderer dependency",
}

_OP_METHOD_EXCLUSIONS: dict[str, str] = {
    "attach_hooks": "mutates the live model's hook registry",
    "do": "intervention editor (mutates fork state)",
    "set": "intervention editor (mutates fork state)",
    "log_tensor_grad": "capture-internal gradient writer",
    "save_activation": "capture-internal payload writer",
}

_LAYER_METHOD_EXCLUSIONS: dict[str, str] = {
    "attach_hooks": "mutates the live model's hook registry",
    "do": "intervention editor (mutates fork state)",
    "set": "intervention editor (mutates fork state)",
}

_PARAM_METHOD_EXCLUSIONS: dict[str, str] = {
    "release_param_ref": "destructive; nulls the live param ref",
}

_MODULE_METHOD_EXCLUSIONS: dict[str, str] = {}
_MODULE_CALL_METHOD_EXCLUSIONS: dict[str, str] = {}
_BUFFER_METHOD_EXCLUSIONS: dict[str, str] = {}


def _trace_method_sweep(tmp_path: Path, sink: Any) -> dict[str, Any]:
    """Return caller-per-method for every swept public ``Trace`` method."""

    out = tmp_path / "method_sweep"
    out.mkdir(exist_ok=True)
    return {
        "summary": lambda t: t.summary(),
        "profile": lambda t: t.profile(),
        "to_pandas": lambda t: t.to_pandas(),
        # Plain draw AND an encoding-channel draw: the channel path is the
        # one that populated `_last_encoding_state` (fail-before case).
        "draw": lambda t: (
            t.draw(vis_outpath=str(out / "graph")),
            t.draw(vis_outpath=str(out / "graph_encoded"), color_by="time"),
        ),
        "collapse_plan": lambda t: t.collapse_plan(),
        "collapse_schedule": lambda t: t.collapse_schedule(),
        "collapse_order": lambda t: t.collapse_order(),
        "receptive_fields": lambda t: t.receptive_fields(),
        "projective_fields": lambda t: t.projective_fields(),
        "show_call_tree": lambda t: t.show_call_tree(file=sink),
        "walk_calls": lambda t: list(t.walk_calls()),
        "find_layers": lambda t: t.find_layers("relu"),
        "audit": lambda t: t.audit(),
        "check_metadata_invariants": lambda t: t.check_metadata_invariants(),
        "find_nan": lambda t: t.find_nan(),
        "first_nonfinite": lambda t: t.first_nonfinite(),
        "last_run_records": lambda t: t.last_run_records(),
        "decode_output": lambda t: t.decode_output(),
        "output_table": lambda t: t.output_table(),
        "reconstruct_output": lambda t: t.reconstruct_output(),
        "reconstruct_container": lambda t: t.reconstruct_container(),
        "visualization_field_audit": lambda t: t.visualization_field_audit(),
        "attention_blocks": lambda t: list(t.attention_blocks()),
        "modules_with_facet": lambda t: list(t.modules_with_facet("query")),
        "activations_by_pass": lambda t: t.activations_by_pass(1),
        "activations_by_address": lambda t: t.activations_by_address("linear"),
        "activation_by_raw_label": lambda t: t.activation_by_raw_label(t.layer_list[0].raw_label),
        # Post-tour sprint additions. All five are pure reads, so they are SWEPT
        # rather than excluded: "analysis populated a cache with no persistence
        # policy" is precisely the draw()-then-save() poison this sweep exists to
        # catch, and a presenter that builds a sub-DAG view is a prime candidate.
        "bill_of_materials": lambda t: t.bill_of_materials(),
        "sites_table": lambda t: t.sites_table(),
        # C02: a pure read over retained payloads (typed row states, no
        # persistence side effects) -- swept for the same reason as
        # sites_table/bill_of_materials.
        "stats_table": lambda t: t.stats_table(),
        "to_agent_json": lambda t: t.to_agent_json(),
        "between": lambda t: t.between(t.input_ops[0].label, t.output_ops[0].label),
        "subgraph": lambda t: t.subgraph(tl.func("relu")),
        "find_sites": lambda t: t.find_sites("relu"),
        "resolve_sites": lambda t: t.resolve_sites("relu"),
        "stack": lambda t: t.stack(tl.func("relu")),
        "show": lambda t: t.show(method="graph", vis_outpath=str(out / "shown")),
        "draw_backward": lambda t: t.draw_backward(vis_outpath=str(out / "bwd")),
        "draw_combined": lambda t: t.draw_combined(vis_outpath=str(out / "cmb")),
        "animate_ops": lambda t: t.animate_ops(t.layer_list[0].layer_label),
    }


def _op_method_sweep(tmp_path: Path, sink: Any) -> dict[str, Any]:
    """Return caller-per-method for every swept public ``Op`` method."""

    return {
        "copy": lambda o: o.copy(),
        "get_children": lambda o: o.get_children(),
        "get_parents": lambda o: o.get_parents(),
        "grad_for": lambda o: o.grad_for(bwd=0),
        "materialize_grad": lambda o: o.materialize_grad(),
        "materialize_out": lambda o: o.materialize_out(),
        "show": lambda o: o.show(),
        "to_pandas": lambda o: o.to_pandas(),
    }


def _layer_method_sweep(tmp_path: Path, sink: Any) -> dict[str, Any]:
    """Return caller-per-method for every swept public ``Layer`` method."""

    return {
        "get_children": lambda x: x.get_children(),
        "get_parents": lambda x: x.get_parents(),
        "show": lambda x: x.show(),
        "to_pandas": lambda x: x.to_pandas(),
    }


def _module_method_sweep(tmp_path: Path, sink: Any) -> dict[str, Any]:
    """Return caller-per-method for every swept public ``Module`` method."""

    out = tmp_path / "method_sweep"
    out.mkdir(exist_ok=True)
    return {
        "draw": lambda m: m.draw(vis_outpath=str(out / f"module_{id(m)}")),
        "show_call_tree": lambda m: m.show_call_tree(file=sink),
        "to_pandas": lambda m: m.to_pandas(),
        "walk_descendants": lambda m: list(m.walk_descendants()),
    }


def _module_call_method_sweep(tmp_path: Path, sink: Any) -> dict[str, Any]:
    """Return caller-per-method for every swept public ``ModuleCall`` method."""

    return {
        "show_call_tree": lambda m: m.show_call_tree(file=sink),
        "to_pandas": lambda m: m.to_pandas(),
        "walk_descendants": lambda m: list(m.walk_descendants()),
    }


def _param_method_sweep(tmp_path: Path, sink: Any) -> dict[str, Any]:
    """Return caller-per-method for every swept public ``Param`` method."""

    return {"to_pandas": lambda p: p.to_pandas()}


def _buffer_method_sweep(tmp_path: Path, sink: Any) -> dict[str, Any]:
    """Return caller-per-method for every swept public ``Buffer`` method."""

    return {
        "to_pandas": lambda b: b.to_pandas(),
        "value_after": lambda b: b.value_after(1),
        "value_at": lambda b: b.value_at(1),
    }


#: (class, sweep factory, exclusion table, instance picker) per record class.
_METHOD_SWEEP_PLAN: list[tuple[type, Any, dict[str, str], Any]] = [
    (Trace, _trace_method_sweep, _TRACE_METHOD_EXCLUSIONS, lambda log: [log]),
    (
        Op,
        _op_method_sweep,
        _OP_METHOD_EXCLUSIONS,
        lambda log: [layer for layer in log.layer_list if type(layer) is Op],
    ),
    (
        Layer,
        _layer_method_sweep,
        _LAYER_METHOD_EXCLUSIONS,
        lambda log: list(log.layer_logs.values()),
    ),
    (Module, _module_method_sweep, _MODULE_METHOD_EXCLUSIONS, lambda log: list(log.modules)),
    (
        ModuleCall,
        _module_call_method_sweep,
        _MODULE_CALL_METHOD_EXCLUSIONS,
        lambda log: list(log.modules._pass_dict.values()),
    ),
    (Param, _param_method_sweep, _PARAM_METHOD_EXCLUSIONS, lambda log: list(log.param_logs)),
    (Buffer, _buffer_method_sweep, _BUFFER_METHOD_EXCLUSIONS, lambda log: list(log.buffers)),
]

#: Methods that must run WITHOUT raising on the canonical tiny capture. The
#: long tail is best-effort (a typed refusal on a plain tiny trace -- e.g.
#: draw_backward without a backward pass -- is fine, and a raising method may
#: still have populated state before raising, which is exactly what the sweep
#: exists to catch). This floor keeps the sweep from silently degrading into
#: all-TypeErrors if a signature changes.
_METHOD_SWEEP_MUST_SUCCEED: dict[str, frozenset[str]] = {
    "Trace": frozenset(
        {
            "summary",
            "profile",
            "to_pandas",
            "draw",
            "collapse_plan",
            "collapse_schedule",
            "collapse_order",
            "receptive_fields",
            "projective_fields",
            "show_call_tree",
            "walk_calls",
            "find_layers",
            "audit",
            "check_metadata_invariants",
        }
    ),
    "Op": frozenset({"get_children", "get_parents", "to_pandas", "show"}),
    "Layer": frozenset({"get_children", "get_parents", "to_pandas", "show"}),
    "Module": frozenset({"show_call_tree", "to_pandas", "walk_descendants"}),
    "ModuleCall": frozenset({"show_call_tree", "to_pandas", "walk_descendants"}),
    "Param": frozenset({"to_pandas"}),
    "Buffer": frozenset({"to_pandas"}),
}


def _public_methods(cls: type) -> set[str]:
    """Return the public instance-method names on ``cls``.

    Properties and ``functools.cached_property`` descriptors belong to the
    attribute sweep above; classmethod/staticmethod constructors are not
    instance reads and stay out of both sweeps.
    """

    names: set[str] = set()
    for name in dir(cls):
        if name.startswith("_"):
            continue
        static = inspect.getattr_static(cls, name, None)
        if isinstance(static, (property, functools.cached_property, classmethod, staticmethod)):
            continue
        if callable(static):
            names.add(name)
    return names


@pytest.mark.smoke
def test_public_method_sweep_tables_are_total(tmp_path: Path) -> None:
    """Every public method is either swept or excluded with a reason -- exactly.

    Both directions are exact: a new public method must be classified before
    it ships, and a retired method's row must be deleted. An untested method
    silently omitted is how the draw() poison recurred after the attribute
    sweep landed.
    """

    problems: dict[str, dict[str, list[str]]] = {}
    for cls, sweep_factory, exclusions, _picker in _METHOD_SWEEP_PLAN:
        swept = set(sweep_factory(tmp_path, None))
        public = _public_methods(cls)
        unclassified = sorted(public - swept - set(exclusions))
        stale = sorted((swept | set(exclusions)) - public)
        overlap = sorted(swept & set(exclusions))
        entry: dict[str, list[str]] = {}
        if unclassified:
            entry["unclassified"] = unclassified
        if stale:
            entry["stale"] = stale
        if overlap:
            entry["both_swept_and_excluded"] = overlap
        if entry:
            problems[cls.__name__] = entry
    assert problems == {}, (
        "public-method sweep tables are not total; classify each method as "
        f"swept or excluded-with-reason: {problems}"
    )


@pytest.mark.smoke
def test_public_method_reads_never_poison_save(tmp_path: Path) -> None:
    """Read-only presentation/analysis methods must leave the trace saveable.

    Fail-before: ``Trace.draw()`` stored its encoding diagnostic on
    ``_last_encoding_state`` with no scrub policy, so the flagship
    visualize-then-save workflow refused with an error naming an internal
    field the user never touched. The attribute sweep above could not see it:
    ``draw()`` is a method, not a property.
    """

    import io

    live_log = _build_live_log()
    sink = io.StringIO()
    raised_must_succeed: dict[str, str] = {}

    for cls, sweep_factory, _exclusions, picker in _METHOD_SWEEP_PLAN:
        callers = sweep_factory(tmp_path, sink)
        must_succeed = _METHOD_SWEEP_MUST_SUCCEED[cls.__name__]
        for record in picker(live_log):
            for name, caller in callers.items():
                try:
                    caller(record)
                except Exception as exc:  # noqa: BLE001 - refusals are not the subject
                    if name in must_succeed:
                        raised_must_succeed[f"{cls.__name__}.{name}"] = repr(exc)

    assert raised_must_succeed == {}, (
        "headline presentation methods raised on the canonical tiny capture "
        f"(sweep coverage is degrading): {raised_must_succeed}"
    )

    from torchlens._io.scrub import _is_runtime_only_trace_field

    undeclared_by_class: dict[str, list[str]] = {}
    for record in _all_record_instances(live_log):
        spec = getattr(type(record), "PORTABLE_STATE_SPEC", None)
        if spec is None:
            continue
        undeclared = sorted(
            field_name
            for field_name, _ in state_items(record)
            if field_name not in spec
            and not (isinstance(record, Trace) and _is_runtime_only_trace_field(field_name))
            and not isinstance(
                inspect.getattr_static(type(record), field_name, None),
                functools.cached_property,
            )
        )
        if undeclared:
            undeclared_by_class.setdefault(type(record).__name__, undeclared)
    assert undeclared_by_class == {}, (
        "public method calls populated live state with no scrub policy; "
        f"declare each field (usually FieldPolicy.DROP): {undeclared_by_class}"
    )

    tl.save(live_log, str(tmp_path / "after_method_sweep.tlspec"))
    pickle.dumps(live_log)


@pytest.mark.smoke
def test_draw_then_save_and_pickle_regression(tmp_path: Path) -> None:
    """Focused fail-before pin: draw() must not poison tl.save or pickle.

    ``draw()`` stored the encoding diagnostic on ``_last_encoding_state``
    (ledgered as "scrub-declared runtime-only" but enrolled nowhere the scrub
    consults), so one draw made every later ``tl.save`` refuse. The callable
    variant pins the pickle side: a ``color_by=`` lambda rode the diagnostic
    into ``__dict__``, so ``pickle.dumps`` crashed on the raw user callable
    while ``tl.save`` succeeded on the same trace (the R10-7 class).
    """

    log = trace_fn(_TinyIOModel(), torch.randn(2, 4))
    log.draw(vis_outpath=str(tmp_path / "graph"), color_by=lambda node: 1.0)
    tl.save(log, str(tmp_path / "after_draw.tlspec"))
    pickle.dumps(log)


@pytest.mark.smoke
def test_ledgered_undeclared_trace_field_refusal_teaches(tmp_path: Path) -> None:
    """The completeness refusal quotes the external-write ledger row.

    ``_last_encoding_state`` was ledgered in TRACE_EXTERNAL_WRITE_EXEMPTIONS
    as "scrub-declared runtime-only" while no scrub declaration existed -- the
    ledger documents the write, it is not a policy. When a ledgered field with
    no declaration reaches the save boundary, the refusal must name the writer
    (via the ledger reason) and state the remedy, not just an internal field
    name the user never touched.
    """

    from torchlens._io import TorchLensIOError
    from torchlens._io.scrub import _is_runtime_only_trace_field
    from torchlens.data_classes._trace_components import TRACE_EXTERNAL_WRITE_EXEMPTIONS

    field_name = next(
        name
        for name in TRACE_EXTERNAL_WRITE_EXEMPTIONS
        if name not in Trace.PORTABLE_STATE_SPEC and not _is_runtime_only_trace_field(name)
    )
    log = trace_fn(_TinyIOModel(), torch.randn(2, 4))
    setattr(log, field_name, "still-live-at-save-time")
    with pytest.raises(TorchLensIOError) as excinfo:
        tl.save(log, str(tmp_path / "refused.tlspec"))
    message = str(excinfo.value)
    assert field_name in message
    assert "TRACE_EXTERNAL_WRITE_EXEMPTIONS" in message
    assert TRACE_EXTERNAL_WRITE_EXEMPTIONS[field_name] in message
    assert "not a scrub policy" in message


@pytest.mark.smoke
@pytest.mark.parametrize("raw_policy", [True, "small"], ids=["true", "small"])
def test_r69_sparse_runnable_save_always_drops_raw_fields(tmp_path: Path, raw_policy) -> None:
    """r69 E: effective DROP wins before the Trace raw-value special case.

    Sparse runnable saves scrub ``raw_input``/``raw_output`` to ``None`` regardless
    of the ordinary ``save_raw_input``/``save_raw_output`` policy (``True`` or
    ``"small"``), for nested tensor/string containers included -- no raw blob or
    tensor may enter the value-free sparse core (the payload assertion stays the
    unchanged final tripwire, pinned by the genuine-stray negatives in
    test_tlspec_runnable_param_conditional.py).
    """

    import warnings

    from torchlens.options import CaptureOptions

    class _NestedStr(nn.Module):
        def forward(self, x: torch.Tensor, cfg: dict) -> torch.Tensor:
            if cfg["mode"] == "fast":
                return x * 2.0
            return x + 100.0

    trace = trace_fn(
        _NestedStr(),
        [torch.randn(3), {"mode": "fast"}],
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
            save_raw_input=raw_policy,
            save_raw_output=raw_policy,
        ),
    )
    path = tmp_path / f"sparse_raw_{raw_policy}.tlspec"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable")
    with (path / "metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    assert metadata["raw_input"] is None
    assert metadata["raw_output"] is None
    loaded = tl.load(path)
    assert loaded.raw_input is None
    assert loaded.raw_output is None


def test_r69_ordinary_analysis_save_retains_raw_values(tmp_path: Path) -> None:
    """Ordinary analysis saves keep their bounded raw-value behavior (no over-drop)."""

    class _NestedStr(nn.Module):
        def forward(self, x: torch.Tensor, cfg: dict) -> torch.Tensor:
            return x * 2.0

    trace = trace_fn(
        _NestedStr(),
        [torch.randn(3), {"mode": "fast"}],
        capture=tl.options.CaptureOptions(layers_to_save="none", save_raw_input="small"),
    )
    path = tmp_path / "analysis_raw.tlspec"
    trace.save(path)
    with (path / "metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    raw = metadata["raw_input"]
    assert isinstance(raw, list) and len(raw) == 2
    assert torch.is_tensor(raw[0])
    assert raw[1] == {"mode": "fast"}


def test_small_raw_input_pil_round_trips_bounded_image(tmp_path: Path) -> None:
    """PIL raw input should survive ``save_raw_input='small'`` as a bounded image."""

    from PIL import Image as pil_image

    image = pil_image.new("RGB", (512, 300), color=(10, 120, 200))
    trace = trace_fn(
        _TinyImageInputModel(),
        image,
        capture=tl.options.CaptureOptions(layers_to_save="none"),
    )
    path = tmp_path / "pil_raw_input.tlspec"

    trace.save(path)
    with (path / "metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    loaded = tl.load(path)

    raw_image_record = metadata["raw_input"]
    assert raw_image_record[_RAW_IMAGE_SENTINEL] is True
    assert isinstance(raw_image_record["data"], bytes)
    assert len(raw_image_record["data"]) <= _RAW_INPUT_IMAGE_BYTES_LIMIT
    assert loaded.raw_input is not None
    assert loaded.raw_input.size[0] <= 256
    assert loaded.raw_input.size[1] <= 256
