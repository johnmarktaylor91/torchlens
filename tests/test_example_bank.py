"""The example bank (``torchlens/_docs/example_bank/``) runs, is indexed and covers the surface.

The bank holds small snippets agents copy: one task each, grouped by object, with the
printed output as a ``# Output:`` comment. ``tools/docs/example_bank.py`` parses and runs
them. This file checks:

* every example parses, has a unique id and a known fixture, spells in its code each name
  it claims to cover, has at most 15 code lines and shows output other than ``...``;
* ``INDEX.md`` is current;
* every example runs after its fixture and prints its ``# Output:`` comment (one cell per
  example; the language-model ones are ``heavy``, the toy ones fit the unmarked budget);
* every public name the bank owes an example (root ``tl.*`` names and the members of the
  core objects) has one, except the names in :data:`EXCLUDED`, each with a reason. The
  exclusion list is checked both ways, so an entry whose name gains an example or leaves
  the surface fails too.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "docs"))

import example_bank  # noqa: E402

#: Shared exclusion reasons (one per library finding that hides several names).
_REASON_1 = (
    "finding F-4: None even on a custom autograd.Function node (double_back_1_2, "
    "is_custom True), because the source is read from the generated backward node class "
    "(type(grad_fn)), which has no forward/backward/__init__ source, not from the user's "
    "Function"
)
_REASON_2 = (
    "set only by the optional JAX capture backend; the bank fixtures are torch models and"
    " the bank environment has no JAX"
)

#: Owed public names the bank deliberately has no example for: ``name -> reason``.
#: "finding F-n" marks a member that misbehaves on a normal call; fixing it means writing
#: its example and deleting the entry here (the test checks both directions).
EXCLUDED: dict[str, str] = {
    **dict.fromkeys(
        [
            "GradFn.backward_docstring",
            "GradFn.backward_source_line",
            "GradFn.backward_source_location",
            "GradFn.forward_docstring",
            "GradFn.forward_signature",
            "GradFn.forward_source_line",
            "GradFn.forward_source_location",
            "GradFn.init_docstring",
            "GradFn.init_signature",
            "GradFn.init_source_line",
            "GradFn.init_source_location",
        ],
        _REASON_1,
    ),
    "GradFn.cls": (
        "finding F-1: GradFn.cls is always None after a backward, built-in and custom nodes alike"
    ),
    "GradFn.differentiates": (
        "finding F-5: never written (no differentiates= writer in the package), so it is "
        "None even when op_label is set (addmm_back_1_3 pairs linear_2_3)"
    ),
    "Layer.control_flow.terminal_bool_for": (
        "finding F-4: None on the branch fixture gt layer while its single op reports "
        "(cond_gt_1_3, 0)"
    ),
    "Module.backward_duration": (
        "finding F-6: hard-coded to return None, even after a backward where "
        "GradFn.backward_duration holds measured times; the glossary says None only when "
        "backward timing is not available"
    ),
    "Module.cls": (
        "finding F-1: Module.cls is always None, even while the model object is alive; "
        "the glossary describes it as the live class"
    ),
    "Module.total_backward_duration": (
        "finding F-6: hard-coded to return None, even after a backward with measured GradFn timings"
    ),
    "ModuleCall.class_name": (
        "finding F-2: ModuleCall.class_name is the empty string on every non-root call "
        "(fc1:1, cell:2); only the root call gives the class name"
    ),
    "ModuleCall.class_qualname": (
        "finding F-2: ModuleCall.class_qualname is the empty string on every non-root "
        "call; only the root call gives it"
    ),
    "ModuleCall.cls": "finding F-1: ModuleCall.cls is always None, even while the model object is alive",
    "Op.control_flow.is_in_conditional_body": (
        "finding F-3: on the branch fixture it is False for the then-arm relu and True "
        "for the if-test sum, the reverse of Layer.control_flow.is_in_conditional_body and the "
        "glossary meaning"
    ),
    "Param.device_ref": (
        "finding F-3: Param.device_ref is never written anywhere in the package, so it is"
        " always None"
    ),
    "Param.module_cls": (
        "finding F-1: Param.module_cls is always None, even while the owning module is alive"
    ),
    "Trace.backend": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, x).backend);"
        " declared only as a class annotation, never set on the Trace"
    ),
    "Trace.backend_runtime_config": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).backend_runtime_config); declared only as a class annotation, never set on "
        "the Trace"
    ),
    "Trace.backend_runtime_version": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).backend_runtime_version); declared only as a class annotation, never set on "
        "the Trace"
    ),
    "Trace.backward_ready": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).backward_ready); declared only as a class annotation, never set on the Trace"
    ),
    "Trace.backward_root_grad_fn_object_ids": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).backward_root_grad_fn_object_ids); declared only as a class annotation, never"
        " set on the Trace"
    ),
    "Trace.capture_end_time": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).capture_end_time); declared only as a class annotation, never set on the "
        "Trace"
    ),
    "Trace.capture_start_time": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).capture_start_time); declared only as a class annotation, never set on the "
        "Trace"
    ),
    "Trace.chunked_forward": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).chunked_forward); declared only as a class annotation, never set on the Trace"
    ),
    "Trace.code_context": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).code_context); declared only as a class annotation, never set on the Trace"
    ),
    "Trace.distributed_scope": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).distributed_scope); declared only as a class annotation, never set on the "
        "Trace"
    ),
    "Trace.inference_only": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).inference_only); declared only as a class annotation, never set on the Trace"
    ),
    "Trace.input_object_id": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).input_object_id); declared only as a class annotation, never set on the Trace"
    ),
    "Trace.input_preprocessor": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).input_preprocessor); declared only as a class annotation, never set on the "
        "Trace"
    ),
    "Trace.input_signature_hash": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).input_signature_hash); declared only as a class annotation, never set on the "
        "Trace"
    ),
    "Trace.input_structure": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).input_structure); declared only as a class annotation, never set on the Trace"
    ),
    **dict.fromkeys(
        [
            "Trace.jax_capture_index_to_final_op_label",
            "Trace.jax_closed_jaxpr",
            "Trace.jax_equation_captures",
            "Trace.jax_inlined_call_primitives",
            "Trace.jax_outvar_key_to_capture_index",
            "Trace.jax_static_argnums",
        ],
        _REASON_2,
    ),
    "Trace.model_object_id": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).model_object_id); declared only as a class annotation, never set on the Trace"
    ),
    "Trace.module_identity_mode": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).module_identity_mode); declared only as a class annotation, never set on the "
        "Trace"
    ),
    "Trace.op_equivalence_classes": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).op_equivalence_classes); declared only as a class annotation, never set on "
        "the Trace"
    ),
    "Trace.output_postprocessor": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).output_postprocessor); declared only as a class annotation, never set on the "
        "Trace"
    ),
    "Trace.param_source": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).param_source); declared only as a class annotation, never set on the Trace"
    ),
    "Trace.save_arg_templates": (
        "finding F-1: AttributeError on a plain toy capture (tl.trace(model, "
        "x).save_arg_templates); declared only as a class annotation, never set on the "
        "Trace"
    ),
    "tl.callbacks": (
        "its only content (tl.callbacks.lightning) builds Lightning callback classes and "
        "needs the optional lightning package, which the bank test environment does not "
        "install, so an example would only ever be skipped"
    ),
}

_EXAMPLES = example_bank.load_examples()
_FIXTURES = example_bank.load_fixtures()
#: Fixtures whose examples cost a heavy-tier budget (a transformer capture).
_HEAVY_FIXTURES = frozenset({"lm"})


def _cells() -> list[Any]:
    """One parameter per example, heavy-marked when its fixture is heavy."""

    return [
        pytest.param(
            example,
            id=example.id,
            marks=[pytest.mark.heavy] if example.fixture in _HEAVY_FIXTURES else [],
        )
        for example in _EXAMPLES
    ]


def test_examples_parse_with_unique_ids_and_known_fixtures() -> None:
    """Ids are unique, fixtures exist, kinds are known and wrong examples pair with a right."""

    ids = [example.id for example in _EXAMPLES]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})
    assert {example.fixture for example in _EXAMPLES} <= set(_FIXTURES)
    assert {example.kind for example in _EXAMPLES} <= {"example", "wrong", "right"}
    wrong = {e.id.removesuffix("-wrong") for e in _EXAMPLES if e.kind == "wrong"}
    right = {e.id.removesuffix("-right") for e in _EXAMPLES if e.kind == "right"}
    assert wrong == right, f"unpaired trap examples: {sorted(wrong ^ right)}"
    assert all(example.shows for example in _EXAMPLES)


def test_examples_spell_the_names_they_cover() -> None:
    """A covered name appears in the code, the code is short and the output shows something."""

    problems = [p for example in _EXAMPLES for p in example_bank.cover_problems(example)]
    assert not problems, "\n".join(problems)


def test_index_is_current() -> None:
    """INDEX.md matches the examples; regenerate with ``example_bank.py index``."""

    rendered = example_bank.render_index(_EXAMPLES)
    assert example_bank.INDEX.read_text(encoding="utf-8") == rendered


@pytest.mark.parametrize("example", _cells())
def test_example_runs_and_prints_its_output(example: Any, tmp_path: Path) -> None:
    """The example runs after its fixture and prints what its output comment shows."""

    result = example_bank.run_example(example, _FIXTURES, tmp_path)
    if result.status == "skipped":
        pytest.skip(f"needs {result.detail}")
    assert result.status == "ok", result.detail


@functools.lru_cache(maxsize=1)
def _inventory() -> tuple[dict[str, Any], ...]:
    """Build the lockstep inventory once per process."""

    return tuple(example_bank._inventory_rows())


@pytest.mark.heavy
def test_every_owed_public_name_has_an_example() -> None:
    """Root names and core-object members each have an example or an excluded reason."""

    rows = list(_inventory())
    missing = [
        f"{name} ({kind})"
        for name, kind, ids in example_bank.coverage_rows(_EXAMPLES, rows)
        if not ids and name not in EXCLUDED
    ]
    assert not missing, f"{len(missing)} owed names without an example:\n" + "\n".join(missing)


@pytest.mark.heavy
def test_exclusions_are_current_and_covers_are_real_names() -> None:
    """Excluded names are owed and uncovered; every covered name is on the surface."""

    rows = list(_inventory())
    table = {name: ids for name, _, ids in example_bank.coverage_rows(_EXAMPLES, rows)}
    stale = [name for name in EXCLUDED if name not in table or table[name]]
    assert not stale, f"excluded names now covered or gone: {stale}"
    known = example_bank.known_names(rows)
    unknown = sorted({n for e in _EXAMPLES for n in e.covers if n not in known})
    assert not unknown, f"covered names not on the public surface: {unknown}"
