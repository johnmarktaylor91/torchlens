"""Receiver-aware rename codemod for attribute and keyword names shared by several types.

Some decided renames touch a name that several unrelated classes or callables also use
(``layers``, ``layout``, ``x=``). A text rename would hit them all, so this codemod types
each site's receiver with a static pass over the tree and renames only the sites whose
receiver is one of the renamed owners:

* ``self`` inside the owner class, parameter and variable annotations, constructor and
  factory calls (via the return annotations of every function in ``torchlens/``),
  ``isinstance`` guards, field chains, ``for`` targets over typed containers and pytest
  fixtures with an annotated or inferable return;
* python fences in Markdown pages and code cells of notebooks are parsed the same way,
  one pseudo-module per page or notebook;
* qualified ``Owner.old`` spellings in prose, comments and docstrings are renamed.

Every other site is either "definite other" (the receiver type is known and is not an
owner) or "undecided". Undecided sites are settled by the per-site ``OVERRIDES`` table,
keyed on path, stripped line text and occurrence, so a re-run on a merged tree
reproduces the hand review. The codemod is idempotent: a second run changes nothing.

Usage::

    python tools/codemods/r2_ambiguous.py            # dry run, prints counts
    python tools/codemods/r2_ambiguous.py --apply    # write the edits
    python tools/codemods/r2_ambiguous.py --report out.json  # every site with its verdict
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREES = ("torchlens", "tests", "docs", "examples", "notebooks")
INDEX_TREES = ("torchlens", "tests")


@dataclass(frozen=True)
class MemberRow:
    """A class member renamed on its owner classes only."""

    owners: tuple[str, ...]
    old: str
    new: str
    # Also rename the matching constructor keyword (dataclass fields and init params).
    keyword: bool = True


@dataclass(frozen=True)
class ParamRow:
    """A keyword renamed on named callables only (``Class.method`` or ``function``)."""

    callables: tuple[str, ...]
    old: str
    new: str


# The member rows take subclasses with them (the owner check walks the bases).
ROWS: tuple[MemberRow | ParamRow, ...] = (
    MemberRow(("CausalReceipt",), "layer", "module_address"),
    MemberRow(("GqaInfo",), "n_kv_heads", "num_key_value_heads"),
    MemberRow(("HistogramPoint",), "edges", "bin_edges"),
    MemberRow(("HistogramResult",), "count_total", "numel"),
    MemberRow(("MergedTrace",), "release", "cleanup", keyword=False),
    MemberRow(("PredictionTrajectory",), "layers", "module_addresses"),
    MemberRow(("ReceptiveField", "ReceptiveFieldView"), "layout", "axis_roles"),
    MemberRow(("ReceptiveFieldView",), "gradient", "probe_grad", keyword=False),
    MemberRow(("Resolution",), "transform", "input_transform"),
    MemberRow(("SrpSizing",), "n_components", "num_components"),
    MemberRow(("SrpSizing",), "n_samples", "num_samples"),
    MemberRow(("TensorSpec",), "of", "from_tensor", keyword=False),
)

# Filled by the row table below (keyword rows); kept separate for readability.
PARAM_ROWS: tuple[ParamRow, ...] = ()

# Per-site hand decisions: (path, stripped line text, old token, occurrence) -> verdict.
# ``occurrence`` counts matches of ``old`` as a whole word on that line, from 0.
OVERRIDES: dict[tuple[str, str, str, int], str] = {
    (
        "docs/acknowledgments.md",
        "expectations: `input_size=`, `depth=`, the returned statistics object,",
        "input_size",
        0,
    ): "skip",
    (
        "docs/agent-reference/architecture.md",
        "# show(gradient=True) recomputes the gradient WITHOUT retain_graph and frees the",
        "gradient",
        0,
    ): "apply",
    (
        "docs/agent-reference/architecture.md",
        "persisted envelope; `tl.summary(meta_model, input_size=...)` and",
        "input_size",
        0,
    ): "apply",
    (
        "docs/agent-reference/architecture.md",
        "render share one resolver -- real input XOR `input_size=` (flat tuple /",
        "input_size",
        0,
    ): "apply",
    (
        "docs/agent-reference/architecture.md",
        "rf_gradient = armed_op.receptive_field.gradient(armed_unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "docs/agent-reference/architecture.md",
        "when underivable. One-call input precedence: args XOR `input_size=` XOR",
        "input_size",
        0,
    ): "apply",
    (
        "docs/agent-reference/common-patterns.md",
        "# show(gradient=True) recomputes the gradient WITHOUT retain_graph and frees the",
        "gradient",
        0,
    ): "apply",
    (
        "docs/agent-reference/common-patterns.md",
        "gradient = armed_op.receptive_field.gradient(armed_unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "docs/agent-reference/current-2-x-surface.md",
        "- EDGE SUBSTITUTION (L6 stage 3; DOCUMENTED-UNSTABLE): `trace.graph.edges` is the",
        "edges",
        -1,
    ): "skip",
    (
        "docs/agent-reference/current-2-x-surface.md",
        "a real input (gold rung) XOR `input_size=` (declared shape, seed-0 local synthesis,",
        "input_size",
        0,
    ): "apply",
    (
        "docs/agent-reference/current-2-x-surface.md",
        "input_size=...)` is the rung-4 auto-selecting facade (bare summary with",
        "input_size",
        0,
    ): "apply",
    (
        "docs/agent-reference/current-2-x-surface.md",
        "precedence args XOR `input_size=` XOR zero-input (reuses",
        "input_size",
        0,
    ): "apply",
    (
        "docs/agent-reference/current-2-x-surface.md",
        'retain=, engine="live_hook", model=, x=)` is the ONE candidate engine',
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "- The capture doors, `fork.do` (replay), `do(model=, x=)` (rerun), `attach_hooks` + `run` and",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `add(scalar)` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `add(tensor)` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `add_noise` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `at_positions` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `clamp` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `compose` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `hook callable` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `mean_ablate` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `mean_fill(reference)` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `mean_fill` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `mean_from` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `patch_from` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `permute_batch` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `project_onto` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `project_out` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `replace_with` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `resample_rows_from` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `scale` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `scramble_elements` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `set_affine_coordinates` | raises:affine_source_needs_subspace | raises:affine_source_needs_subspace |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `set_direction_mean` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `splice_module` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `steer` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `subspace_ablate(resample)` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `subspace_ablate` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `subspace_interchange` | reaches | reaches |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `swap_with` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/gradient_contract.md",
        "| do(model=, x=) (rerun) | `zero_ablate` | cut | raises:edit_detached_from_graph |",
        "x",
        0,
    ): "apply",
    (
        "docs/intervention_api.md",
        'intervene on a FRESH execution (`do(..., model=..., x=..., intervention=InterventionOptions(engine="rerun"))`',
        "x",
        0,
    ): "apply",
    (
        "docs/intervention_explainers.md",
        "`do()` picks its engine from its arguments: with neither `model=` nor `x=` it pushes over the saved",
        "x",
        0,
    ): "apply",
    (
        "docs/intervention_explainers.md",
        "| The same question when the edit may change control flow, randomness or module state | `trace.fork().do(site, action, model=model, x=x)`, or `trace.run(model, x)` | yes | the fork, re-executed under the edit |",
        "x",
        0,
    ): "apply",
    (
        "docs/migration/from_torchinfo.md",
        "tl.summary(model, input_size=(1, 3, 224, 224))  # the batch dimension is explicit",
        "input_size",
        0,
    ): "apply",
    (
        "docs/migration/from_torchinfo.md",
        "| `summary(model, input_size=(1, 3, 224, 224))` | `tl.summary(model, input_size=(1, 3, 224, 224))` |",
        "input_size",
        0,
    ): "skip",
    (
        "docs/migration/from_torchinfo.md",
        "| `summary(model, input_size=(1, 3, 224, 224))` | `tl.summary(model, input_size=(1, 3, 224, 224))` |",
        "input_size",
        1,
    ): "apply",
    (
        "docs/migration/from_torchview.md",
        "| `draw_graph(model, input_size=(1, 3, 32, 32))` | `tl.trace(model, torch.randn(1, 3, 32, 32)).draw()` | TorchLens runs a real input; build one of the size you would pass. |",
        "input_size",
        0,
    ): "skip",
    (
        "docs/quickstart.md",
        "**Rung 2 (`input_size=`).** One flat positive-integer tuple is one input",
        "input_size",
        0,
    ): "apply",
    (
        "docs/quickstart.md",
        "The rungs are exclusive by law: a real input XOR `input_size=` XOR nothing.",
        "input_size",
        0,
    ): "apply",
    (
        "docs/quickstart.md",
        "next command, in order: pass the real call; pass `input_size=` (including",
        "input_size",
        0,
    ): "apply",
    (
        "docs/receptive_projective_fields.md",
        "# .gradient() and .check() FREE the armed autograd graph by default, so pass",
        "gradient",
        -1,
    ): "apply",
    (
        "docs/receptive_projective_fields.md",
        "Use a complete index for `.gradient()`, `.check()`, and `.show()`; `.at()` alone takes the",
        "gradient",
        -1,
    ): "apply",
    (
        "docs/receptive_projective_fields.md",
        "`ReceptiveFieldView.at()` is the cheap geometric query. `.gradient()` probes exactly one output",
        "gradient",
        -1,
    ): "apply",
    (
        "docs/receptive_projective_fields.md",
        "empirical = rf.gradient(unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "docs/receptive_projective_fields.md",
        "explicit because they consume autograd work. Both `.gradient()` and `.check()` FREE the armed",
        "gradient",
        -1,
    ): "apply",
    (
        "docs/receptive_projective_fields.md",
        "tripwire, and `.show()` renders an input overlay; with `gradient=True`, it also overlays the",
        "gradient",
        0,
    ): "apply",
    (
        "docs/receptive_projective_fields.md",
        "| `DATA_DEPENDENT` | Routing depends on data; use `.gradient()`. |",
        "gradient",
        -1,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        '| `bridge_module_site_unresolved` | `tl.bridge.pytorch_grad_cam.layer`/`cam` or `tl.bridge.captum.layer` was given a site that is not the output of any module of the source model (an input placeholder, a free function op), so there is no module to hook (`InvalidArgumentError`) | Pass a module address such as `"layer4"`, or the label of an op whose tensor a module returns |',
        "layer",
        -1,
    ): "skip",
    (
        "docs/reference/error_refusal_contract.md",
        "| `edge_provenance_unavailable` | `trace.graph.edges` / EDGE-kind resolution on a capture without edge provenance (`intervention_ready` not armed; spelling DOCUMENTED-UNSTABLE) | Re-capture with `CaptureOptions(intervention_ready=True)` |",
        "edges",
        -1,
    ): "skip",
    (
        "docs/reference/error_refusal_contract.md",
        "| `input_dtype_ambiguous` | No static model fact settles a dtype/value recipe for an `input_size=` slot; TorchLens never discovers dtypes by running failed forwards (`InvalidArgumentError`) | Pass `torchlens.types.InputSpec(shape, dtype=..., low=..., high=...)` for the slot, or a real input |",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        '| `input_inference_failed` | The zero-argument rung could not infer a runnable input; the bounded probe diary rides `exc.fields["attempts"]` (`ShapeInferenceError`) | Pass the real forward call; or `input_size=` (mapping form for multi-input models); or run `torchlens.debug.guess_input_shape` with hints — for HF models a real prompt string is the better answer |',
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `input_ladder_backend_unsupported` | `input_size=`/zero-argument resolution requested with an explicit non-torch backend (`InvalidArgumentError`) | Pass a real input for non-torch backends |",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `input_rung_conflict` | A real input and `input_size=` were both supplied; the input ladder's rungs are exclusive by law (`ArgumentConflictError`) | Pass the real input alone, or drop it and keep `input_size=` |",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `input_rung_conflict` | A real input and `input_size=` were both supplied; the input ladder's rungs are exclusive by law (`ArgumentConflictError`) | Pass the real input alone, or drop it and keep `input_size=` |",
        "input_size",
        1,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `input_rung_requires_module_root` | `input_size=`/inferred-input capture with a non-`nn.Module` root (F41): declared and inferred rungs synthesize inputs from a module's geometry, which a bound-method root does not expose | Pass a real input (the gold rung), e.g. `tl.trace(model.generate, input_ids)` |",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `input_size_invalid` | `input_size=` grammar violation: empty, zero/negative/symbolic/bool dimension, or not a shape tuple / sequence of shapes / keyword-to-shape mapping (`InvalidArgumentError`) | Use one flat positive-int tuple, a sequence of shape tuples, or a mapping of forward keyword names to shapes; `torchlens.types.InputSpec` pins dtype/value recipes |",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `input_size_unknown_binding` | `input_size=` mapping binds a keyword the model's forward does not accept (`InvalidArgumentError`) | Bind one of the forward's known keyword names (listed in the refusal) |",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `preprocessing_audit_mismatch` | The strict preprocessing audit found fields where the applied pipeline DISAGREES with the authority's declared configuration, named per field with both values (tvscope B3) | Apply the authority's own transform (`resolution.transform`) or correct the mismatched fields; rerun with `strict=False` for the full report without raising |",
        "transform",
        -1,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `site_sweep_inputs_missing` | `sweep_sites` on the live_hook lane runs a REAL forward per candidate and needs `model=` and `x=` (F03 item 6) | Pass both, or use `engine='replay'` |",
        "x",
        0,
    ): "apply",
    (
        "docs/reference/error_refusal_contract.md",
        "| `weightsfree_facade_input_evidence_required` | summary() on a meta-built model with no input evidence: the zero-argument inference rung runs real probing forwards a weights-free model cannot serve | Pass input_size= (declared shapes) or explicit meta inputs (torch.zeros(..., device='meta')) |",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/experiment_ledger.md",
        'retain=, engine="live_hook", model=, x=)` runs ONE candidate engine (the',
        "x",
        0,
    ): "apply",
    (
        "docs/reference/glossary.md",
        '**tl.summary("<hub model id>", input_size=...)** — *unstable — no deprecation shim owed*',
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/glossary.md",
        "**trace.graph.edges / edge substitution** — *unstable — no deprecation shim owed;",
        "edges",
        -1,
    ): "skip",
    (
        "docs/reference/glossary.md",
        ": `trace.graph.edges` returns the dataflow edge family (one `EdgeUseRecord` per",
        "edges",
        -1,
    ): "skip",
    (
        "docs/reference/glossary.md",
        ": `trace.layers.compute`, `.recurrent`, `.saved_grad`, `.internal_source` and `.internal_sink`",
        "layers",
        -1,
    ): "skip",
    (
        "docs/reference/glossary.md",
        "`input_size=` or meta inputs. Requires `transformers`.",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/glossary.md",
        "filter, so `len(trace.layers.compute)` counts the compute Layers.",
        "layers",
        -1,
    ): "skip",
    (
        "docs/reference/glossary.md",
        "| `RunningStatsResult.count_total` | field of `RunningStatsResult` |",
        "count_total",
        -1,
    ): "skip",
    (
        "docs/reference/stats.md",
        "print(label, spines[label].result().count_total, spines[label].result().mean)",
        "count_total",
        -1,
    ): "skip",
    (
        "docs/reference/summary.md",
        "input_size=None,          # synthetic input shape (XOR real input_args)",
        "input_size",
        0,
    ): "apply",
    (
        "docs/reference/transforms.md",
        "across k — run `T.srp_fidelity_probe(pilot_batch, n_components=(...))`",
        "n_components",
        0,
    ): "apply",
    (
        "docs/reference/tviz.md",
        'q_grid, layer="model.layers.0.self_attn", negative_control=neg,',
        "layers",
        -1,
    ): "skip",
    (
        "docs/reference/tviz.md",
        'v_grid, layer="model.layers.0.self_attn", negative_control=neg,',
        "layers",
        -1,
    ): "skip",
    (
        "docs/reference/warning_contract.md",
        "| `nongold_raw_value_read` | First raw value read (`Layer.out`) on a trace whose input values were synthesized (`input_size=` or inferred rung): the numbers are real measurements of a forward over RANDOM data, not of your data; fires once per trace object, and internal readers pass an ack so the warning always names a user read site (quickstart memo D7 -- the handoff case a passive persisted record cannot reach) | Re-capture with a real input for value-level claims, or filter the warning by its code to acknowledge |",
        "input_size",
        0,
    ): "apply",
    ("docs/semantic_io.md", 'source="project.transform",', "transform", -1): "skip",
    (
        "notebooks/audit/14_huggingface_guarded.ipynb",
        'print(f"n_heads={_vit_n_heads}  d_head={_vit_d_head}")',
        "n_heads",
        0,
    ): "skip",
    (
        "notebooks/audit/17_receptive_projective_fields.ipynb",
        "# show(gradient=True) recomputes the gradient WITHOUT retain_graph and frees",
        "gradient",
        0,
    ): "apply",
    (
        "notebooks/audit/17_receptive_projective_fields.ipynb",
        "# would return no gradient at rf.gradient/check time):",
        "gradient",
        -1,
    ): "apply",
    (
        "notebooks/audit/17_receptive_projective_fields.ipynb",
        '**Surfaces covered:** `tl.receptive_field`, `Op.receptive_field`, `Op.projective_field`, `.at()`, `.gradient()`, `.check()`, `.show()`, `Trace.receptive_fields()`, `Trace.projective_fields()`, `cross_validate()`, `verify()`, `node_spec()`, `source=`, `direction=`, `target=`, and `tl.validate(..., scope="receptive_field")`.',
        "gradient",
        -1,
    ): "apply",
    (
        "notebooks/audit/17_receptive_projective_fields.ipynb",
        "`.at()` takes windowed-grid coordinates. `.gradient()` and `.check()` take a complete output-element index, including batch.",
        "gradient",
        -1,
    ): "apply",
    (
        "notebooks/audit/17_receptive_projective_fields.ipynb",
        "gradient = rf.gradient(unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    ("tests/classics_corpus/models/rs_CV1_29.py", "n_heads=8,", "n_heads", 0): "skip",
    (
        "tests/classics_corpus/models/rs_L1142_stargan_vc2.py",
        "def __init__(self, input_size=(36, 512), conv_dim=64, repeat_num=5, num_speakers=70):",
        "input_size",
        0,
    ): "skip",
    (
        "tests/classics_corpus/models/rs_L1280_2_danse.py",
        "mu_batch, vars_batch = self.rnn.forward(x=Yi_batch)",
        "x",
        0,
    ): "skip",
    ("tests/classics_corpus/models/rs_L130_4.py", "n_heads=4,", "n_heads", 0): "skip",
    (
        "tests/classics_corpus/models/rs_L130_4.py",
        "self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/classics_corpus/models/rs_L1869_1_kdeep.py",
        "# (`summary(model, input_size=(16, 24, 24, 24))` in pytorch/model_test.py).",
        "input_size",
        0,
    ): "skip",
    (
        "tests/classics_corpus/models/rs_L564_biotools.py",
        "# hits the class default `c_in=4, d_model=288, n_heads=8, n_cells=2034,",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/classics_corpus/models/rs_L564_biotools.py",
        "# hits the class defaults (c_in=4, d_model=288, n_heads=8, n_cells=2034). We use",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/classics_corpus/models/rs_L564_biotools.py",
        "def __init__(self, d_models=288, n_heads=8):",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/classics_corpus/models/rs_L564_biotools.py",
        "self, seq_len=1344, seq_len_up_bound=1344, c_in=4, d_model=288, n_heads=8, n_cells=2034",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/classics_corpus/models/rs_RM4b_3.py",
        "# hard-swish(x) is bounded below by ~-0.375 (at x=-1.5) and grows",
        "x",
        1,
    ): "skip",
    ("tests/example_models.py", "def __init__(self, dim=32, n_heads=4):", "n_heads", 0): "skip",
    (
        "tests/example_models.py",
        "def __init__(self, dim=32, n_heads=4, max_seq=32):",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/example_models.py",
        "def __init__(self, dim=32, n_heads=4, max_seq=64):",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/example_models.py",
        "def __init__(self, dim=32, n_heads=4, n_kv_heads=2):",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/example_models.py",
        "def __init__(self, in_dim=8, hidden=16, n_heads=2, out_dim=4):",
        "n_heads",
        0,
    ): "skip",
    (
        "tests/real_model/r0/test_realism_randinit_axes.py",
        '"""A keyword-only call on a plain module reaches ``forward(x=...)`` once.',
        "x",
        0,
    ): "skip",
    (
        "tests/test_docs_executable_contracts.py",
        "gradient = target.receptive_field.gradient(unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_docs_snippets.py",
        "#: unexecuted and rotted silently -- the show(gradient=True)/RF-ordering rot",
        "gradient",
        0,
    ): "apply",
    (
        "tests/test_explorer_watch_gates.py",
        'assert spine.count_total == 2 * (2 * 4), "both calls\' populations folded exactly"',
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_explorer_watch_gates.py",
        "assert spine.count_total == direct.numel()",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_explorer_watch_op_tier.py",
        "assert spine is not None and spine.count_total == 2 * 4  # one head slice",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_explorer_watch_op_tier.py",
        "assert spine is not None and spine.count_total > 0",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_explorer_watch_op_tier.py",
        "assert spine.count_total == 4 * 8  # relu output numel, exactly",
        "count_total",
        -1,
    ): "skip",
    ("tests/test_glossary_conformance_b2c.py", '".gradient()",', "gradient", -1): "apply",
    (
        "tests/test_gradient_contract_matrix.py",
        '"do(model=, x=) (rerun)": (_door_rerun, _plain),',
        "x",
        0,
    ): "apply",
    ("tests/test_merged_engine.py", "merged.release()", "release", -1): "apply",
    ("tests/test_name_lockstep_codemod.py", '"    return f(x=x)\\n"', "x", 0): "skip",
    ("tests/test_name_lockstep_codemod.py", '"def helper(y, x=2):\\n"', "x", 0): "skip",
    (
        "tests/test_name_lockstep_codemod.py",
        '"docs/page.md": "Call `f(x=1)`.\\n\\n```python\\nx = 3\\nf(\\n    x=x,\\n)\\n```\\n",',
        "x",
        0,
    ): "skip",
    (
        "tests/test_name_lockstep_codemod.py",
        '"docs/page.md": "Call `f(x=1)`.\\n\\n```python\\nx = 3\\nf(\\n    x=x,\\n)\\n```\\n",',
        "x",
        1,
    ): "skip",
    (
        "tests/test_obs_substrate_artifact.py",
        "assert row.spine is not None and row.spine.count_total == 3",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_obs_substrate_artifact.py",
        "assert spine is not None and spine.count_total == 6",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_obs_substrate_artifact.py",
        "b.observations[0].spine.count_total  # type: ignore[union-attr]",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_obs_substrate_collector.py",
        "assert activation.spine.count_total == 3 * 4 * 16",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_obs_substrate_kernels.py",
        "assert result.count_total == 0",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_obs_substrate_kernels.py",
        "assert result.count_total == 1000",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_obs_substrate_kernels.py",
        "assert result.count_total == 6",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_obs_substrate_kernels.py",
        "assert result.count_total == 7",
        "count_total",
        -1,
    ): "apply",
    (
        "tests/test_obs_substrate_schema.py",
        "assert merged.spine.count_total == 2 * a.spine.count_total",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_quickstart_ladder.py",
        '"""A real input plus input_size= is a typed conflict, never a guess."""',
        "input_size",
        0,
    ): "apply",
    (
        "tests/test_quickstart_ladder.py",
        '"""input_size= traces persist the disclosure on the KEEP field."""',
        "input_size",
        0,
    ): "apply",
    (
        "tests/test_quickstart_realmodel.py",
        '"""Keyword-mapping input_size= succeeds with per-input dtype recipes."""',
        "input_size",
        0,
    ): "apply",
    (
        "tests/test_r19b_rf_gradient_verify.py",
        'projective = log["conv2d_1_1:1"].projective_field.gradient(',
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_r19b_rf_gradient_verify.py",
        "result = trace.input_ops[0].projective_field.gradient((1, 0, 0, 2), target=trace.output_ops[0])",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_r19b_rf_gradient_verify.py",
        "result = trace.output_ops[0].receptive_field.gradient((0, 1, 0, 2), input_op=trace.input_ops[0])",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_repgeom_validation_edges.py",
        'with pytest.raises(ValueError, match="n_components"):',
        "n_components",
        0,
    ): "apply",
    (
        "tests/test_report_lane_summary_hub.py",
        '"""U04.4: ``tl.summary("<hub name>", input_size=...)`` builds on the meta path.',
        "input_size",
        0,
    ): "apply",
    (
        "tests/test_rf_crossval.py",
        'result = _op(trace, "conv2d").receptive_field.gradient((0, 0, 2, 2), input_op=_input_op(trace))',
        "gradient",
        -1,
    ): "apply",
    ("tests/test_rf_entities.py", "assert view.layout is descriptor.layout", "layout", -1): "apply",
    (
        "tests/test_rf_entities.py",
        "gradient = view.gradient((0, 0, 2, 2), input_op=input_op, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_entities.py",
        "unavailable_target.receptive_field.gradient((0, 0, 2, 2))",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_ext_integration.py",
        'assert projective.layout.axis_kinds == ("pointwise",)',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_ext_integration.py",
        'assert receptive.layout.axis_kinds == ("pointwise",)',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_ext_integration.py",
        "loaded_source.projective_field.gradient((0, 0, 3, 3), target=loaded_target)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_ext_integration.py",
        "result = source.projective_field.gradient(  # type: ignore[union-attr]",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_ext_integration.py",
        "source.projective_field.gradient((0,), target=target)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_ext_integration.py",
        "target.receptive_field.gradient((0,))",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_ext_projective_desc.py",
        'assert descriptor.layout.axis_kinds == ("pointwise", "pointwise", "full", "full")',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometric.py",
        "assert all(label == target.label for label in descriptor.layout.provenance)",
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometric.py",
        'assert descriptor.layout.axis_kinds == ("pointwise", "pointwise", "full", "full")',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometric.py",
        'assert descriptor.layout.axis_kinds == ("unknown",) * 4',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometric.py",
        "assert descriptor.layout.axis_kinds == expected_kinds",
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometric.py",
        'assert descriptor.layout.source == "derived"',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometric.py",
        "assert descriptor.layout.windowed_axes == expected_windowed",
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometry_hardening.py",
        "assert target.receptive_field.gradient(unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometry_hardening.py",
        "fresh_target.receptive_field.gradient(fresh_unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometry_hardening.py",
        "results = target.receptive_field.gradient((0, 0, 2, 2))",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_geometry_hardening.py",
        'with pytest.raises(ReceptiveFieldError, match="use .gradient"):',
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_gradient.py",
        "fork_target.receptive_field.gradient((0, 0, 2, 2), retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_gradient.py",
        "gradient = target.receptive_field.gradient(",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_gradient.py",
        "source_gradient = source_target.receptive_field.gradient(",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_integration.py",
        'assert descriptor.layout.axis_kinds == ("pointwise",)',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_integration.py",
        "target.receptive_field.gradient((0,))",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_rules.py",
        'assert descriptor.layout.axis_kinds == ("full", "pointwise", "full", "full")',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_rules.py",
        'assert descriptor.layout.axis_kinds == ("pointwise", "full", "windowed", "windowed")',
        "layout",
        -1,
    ): "apply",
    (
        "tests/test_rf_viz.py",
        '"""R81: show(gradient=True) must plumb retain_graph to the empirical probe.',
        "gradient",
        0,
    ): "apply",
    ("tests/test_rf_viz.py", "def gradient(", "gradient", 0): "apply",
    (
        "tests/test_rf_viz.py",
        "op.receptive_field.gradient(unit, retain_graph=False)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_viz.py",
        "op.receptive_field.gradient(unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_rf_viz.py",
        'repeated show(gradient=True) -- raised a raw RuntimeError ("backward',
        "gradient",
        0,
    ): "apply",
    (
        "tests/test_rng_inert_reachability.py",
        'exec("def generated(x=None):\\n    return x", synthetic_globals)',
        "x",
        0,
    ): "skip",
    (
        "tests/test_selection_algebra.py",
        "gradient_by_role = armed_op.receptive_field.gradient(unit, retain_graph=True)",
        "gradient",
        -1,
    ): "apply",
    (
        "tests/test_stats_save_stats.py",
        "assert tl.stats.site_stats(trace)[label].result().count_total > 0",
        "count_total",
        -1,
    ): "skip",
    (
        "tests/test_summary_rebuild_config.py",
        '"""input_size= builds a seeded synthetic input and says so."""',
        "input_size",
        0,
    ): "apply",
    (
        "tests/test_summary_rebuild_config.py",
        'assert "input_size=" in str(excinfo.value.fields["remedy"])',
        "input_size",
        0,
    ): "apply",
    (
        "tests/test_summary_rebuild_config.py",
        'assert "input_size=(2, 8)" in str(report)',
        "input_size",
        0,
    ): "apply",
    (
        "tests/test_themes_families_budget.py",
        'assert channel.transform == "rank"',
        "transform",
        -1,
    ): "skip",
    (
        "tests/test_tlspec_runnable_scalar_escape.py",
        "x * s`` captured with ``x=[2, 2]`` bakes ``s=2``; run with ``x=[10, 10]`` a live",
        "x",
        1,
    ): "skip",
    (
        "tests/test_tlspec_runnable_scalar_escape.py",
        "x * s`` captured with ``x=[2, 2]`` bakes ``s=2``; run with ``x=[10, 10]`` a live",
        "x",
        2,
    ): "skip",
    (
        "tests/test_trace_autoroute_kwarg_forwarding.py",
        "# runs with input_size=None (the ladder re-enters trace() with a real",
        "input_size",
        0,
    ): "apply",
    (
        "tests/test_track_sinks_fakes.py",
        'assert stored["edges"] == list(point.edges)',
        "edges",
        -1,
    ): "apply",
    (
        "tests/test_track_sinks_mlflow_comet_real.py",
        'assert stored["edges"] == list(point.edges)',
        "edges",
        -1,
    ): "apply",
    ("tests/test_transforms_lib_srp.py", '{"n_components": 0},', "n_components", 0): "apply",
    (
        "tests/test_transforms_lib_srp.py",
        '{"n_components": 16, "construction": "gaussian"},',
        "n_components",
        0,
    ): "apply",
    (
        "tests/test_transforms_lib_srp.py",
        '{"n_components": 16, "density": "high"},',
        "n_components",
        0,
    ): "apply",
    (
        "tests/test_transforms_lib_srp.py",
        '{"n_components": 16, "density": 1.5},',
        "n_components",
        0,
    ): "apply",
    (
        "tests/test_transforms_lib_srp.py",
        '{"n_components": 16, "mode": "rows"},',
        "n_components",
        0,
    ): "apply",
    (
        "tests/test_transforms_lib_srp.py",
        '{"n_components": 16, "seed": 1.5},',
        "n_components",
        0,
    ): "apply",
    (
        "tests/test_transforms_lib_srp.py",
        '{"n_components": 16, "share": "by_vibes"},',
        "n_components",
        0,
    ): "apply",
    ("tests/test_transforms_lib_srp.py", '{"n_components": None},', "n_components", 0): "apply",
    (
        "tests/test_tviz_realmodel.py",
        "== model.model.layers[0].self_attn.v_proj.out_features",
        "layers",
        -1,
    ): "skip",
    (
        "tests/test_tviz_realmodel.py",
        "assert view.gqa.n_kv_heads == config.num_key_value_heads",
        "n_kv_heads",
        -1,
    ): "apply",
    (
        "torchlens/_data_classes/_trace_intervention.py",
        "\"do(..., engine='rerun') requires both model= and x=. \"",
        "x",
        0,
    ): "apply",
    (
        "torchlens/_data_classes/_trace_intervention.py",
        "\"do(engine='auto') needs both model= and x= for rerun, or neither for \"",
        "x",
        0,
    ): "apply",
    (
        "torchlens/_data_classes/_trace_intervention.py",
        '"engine (or set_only); for rerun, pass the modified input as x=."',
        "x",
        0,
    ): "apply",
    (
        "torchlens/_data_classes/_trace_intervention.py",
        "raise EngineDispatchError(\"do(..., engine='rerun') requires model= and x=.\")",
        "x",
        0,
    ): "apply",
    (
        "torchlens/_data_classes/_trace_validation.py",
        "\"hooks (do(..., engine='rerun', model=..., x=...)) or capture the new \"",
        "x",
        0,
    ): "apply",
    (
        "torchlens/_data_classes/_trace_viz.py",
        '"input_size": input_size,',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_data_classes/op.py",
        "for fid, name in enumerate(store.layout.names):",
        "layout",
        -1,
    ): "skip",
    (
        "torchlens/_extraction/reader.py",
        "row's ``keys[key].layout`` (absent = dense) must agree. A mixed artifact",
        "layout",
        -1,
    ): "skip",
    (
        "torchlens/_observability/_render.py",
        'f\'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observability/_render.py",
        'f\'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white"/>\\n\'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observability/_render.py",
        'f\'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observability/_render.py",
        'f\'<text x="{x:.2f}" y="{y:.2f}" font-family="monospace" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observe/_svg.py",
        'f\'<rect x="0" y="0" width="{_WIDTH}" height="{_HEIGHT}" fill="#ffffff"/>\',',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observe/_svg.py",
        'f\'<rect x="{legend_x}" y="{axis_y + 24}" width="10" height="10" fill="{fill}"/>\'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observe/_svg.py",
        'f\'<rect x="{x:.2f}" y="{y:.2f}" width="{column_width:.2f}" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observe/_svg.py",
        'f\'<text x="{_MARGIN_LEFT}" y="18" font-family="monospace" font-size="13">\'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observe/_svg.py",
        'f\'<text x="{_MARGIN_LEFT}" y="{axis_y + 16}" font-family="monospace" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observe/_svg.py",
        'f\'<text x="{_MARGIN_LEFT}" y="{axis_y + 50}" font-family="monospace" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observe/_svg.py",
        'f\'<text x="{_MARGIN_LEFT}" y="{axis_y + 64}" font-family="monospace" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_observe/_svg.py",
        'f\'<text x="{legend_x + 14}" y="{axis_y + 33}" font-family="monospace" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/_quickstart/__init__.py",
        "- rung 2: ``input_size=`` -- your shape, synthesized values, disclosed;",
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_quickstart/_ladder.py",
        'f"input_size= and zero-argument input resolution are torch-only; "',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_quickstart/_plan.py",
        '"Got BOTH a real input and input_size=. The input ladder\'s rungs are "',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_quickstart/_plan.py",
        '"exclusive: a real input (best), input_size= (your shape, synthesized "',
        "input_size",
        0,
    ): "apply",
    ("torchlens/_quickstart/_plan.py", '"input and keep input_size="', "input_size", 0): "apply",
    (
        "torchlens/_quickstart/_plan.py",
        '"pass the real input alone (drop input_size=), or drop the real "',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_quickstart/_plan.py",
        "Code ``input_rung_conflict`` when a real input and ``input_size=``",
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_quickstart/_plan.py",
        "The ladder's law (memo D2): a real input XOR ``input_size=`` XOR nothing.",
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_quickstart/_provenance.py",
        'return f"input: synthesized from input_size= ({shapes}); values are random, disclosed"',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_quickstart/_render.py",
        "``input_kwargs`` unset to use ``input_size=`` (rung 2) or inference",
        "input_size",
        0,
    ): "skip",
    (
        "torchlens/_quickstart/_render.py",
        "render(model, input_size=(1, 3, 224, 224))  # your shape, random values; disclosed",
        "input_size",
        0,
    ): "skip",
    (
        "torchlens/_quickstart/_resolve.py",
        '"(2) pass input_size= with your shape -- a flat tuple, a sequence of "',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_quickstart/_resolve.py",
        "\"answer than input_size= -- tl.<verb>(model, 'The quick brown fox') \"",
        "input_size",
        0,
    ): "apply",
    ("torchlens/_runnable_execution.py", "torch.layout,", "layout", -1): "skip",
    (
        "torchlens/_user_public_impls.py",
        '"""Render the caller\'s ``input_size=`` spelling compactly for disclosure."""',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_user_public_impls.py",
        '"pass input_size= (declared shapes; e.g. input_size=(1, 8)) "',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_user_public_impls.py",
        '"pass input_size= (declared shapes; e.g. input_size=(1, 8)) "',
        "input_size",
        1,
    ): "apply",
    (
        "torchlens/_user_public_impls.py",
        "``input_args`` XOR ``input_size=`` XOR nothing. ``input_size=``",
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_user_public_impls.py",
        "``input_args`` XOR ``input_size=`` XOR nothing. ``input_size=``",
        "input_size",
        1,
    ): "apply",
    (
        "torchlens/_user_public_impls.py",
        "``input_size=`` for a declared shape with synthesized values; omit",
        "input_size",
        0,
    ): "apply",
    ("torchlens/_user_public_impls.py", 'argument="input_size",', "input_size", 0): "apply",
    (
        "torchlens/_user_public_impls.py",
        "needs ``input_size=`` or meta inputs. Requires ``transformers``; the",
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_user_public_impls.py",
        'return f"input_size={_format_input_size(input_size)}, {recipe_part}{seed_part}"',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/_utils/arg_handling.py",
        '``tl.trace(model, (), {"x": t})`` reaches ``model(x=t)``.',
        "x",
        1,
    ): "skip",
    (
        "torchlens/_utils/tensor_utils.py",
        "if tensor_a.layout != tensor_b.layout:",
        "layout",
        -1,
    ): "skip",
    (
        "torchlens/_utils/tensor_utils.py",
        "if tensor_a.layout == torch.sparse_coo:",
        "layout",
        -1,
    ): "skip",
    ("torchlens/_utils/tensor_utils.py", "if tensor_a.layout in (", "layout", -1): "skip",
    (
        "torchlens/_utils/tensor_utils.py",
        "if tensor_a.layout in (torch.sparse_csr, torch.sparse_bsr):",
        "layout",
        -1,
    ): "skip",
    ("torchlens/_utils/tensor_utils.py", "tensor_a.layout == torch.strided", "layout", -1): "skip",
    ("torchlens/backends/jax/jaxpr.py", 'f"x={xs_index}:scan_read",', "x", 0): "skip",
    (
        "torchlens/backends/jax/jaxpr.py",
        'outvars=(f"{path[-1]}/iter={logical_index}/x={xs_index}",),',
        "x",
        0,
    ): "skip",
    (
        "torchlens/backends/torch/bound_root.py",
        '"input_size=/inferred-input capture requires an nn.Module "',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/errors/__init__.py",
        '"torchlens.visualization._rank_layout_internal.layout",',
        "layout",
        -1,
    ): "skip",
    (
        "torchlens/experiment/_engine.py",
        '"candidate and needs model= and x=; pass both, or use "',
        "x",
        0,
    ): "apply",
    (
        "torchlens/export/__init__.py",
        "f\"<title>{title}</title><rect x='-65' y='-28' width='130' height='56' rx='6' />\"",
        "x",
        0,
    ): "skip",
    (
        "torchlens/export/_report.py",
        "\"function apply(){i.style.transform='translate('+tx+'px,'+ty+'px) scale('+s+')';\"",
        "transform",
        -1,
    ): "skip",
    (
        "torchlens/intervention/rerun.py",
        '"run(..., x=None) cannot recover the original input",',
        "x",
        0,
    ): "apply",
    (
        "torchlens/intervention/rerun.py",
        "and candidate.layout is torch.strided",
        "layout",
        -1,
    ): "skip",
    ("torchlens/intervention/rerun.py", 'argument="x",', "x", 0): "apply",
    ("torchlens/intervention/rerun.py", "value.layout is torch.strided", "layout", -1): "skip",
    (
        "torchlens/options.py",
        '(``"encoder.layer.4"``, at that module\'s exit boundary) or function',
        "layer",
        -1,
    ): "skip",
    ("torchlens/options.py", "from ._data_classes.layer import Layer", "layer", -1): "skip",
    (
        "torchlens/preprocessing/_attached.py",
        "from open_clip.transform import PreprocessCfg, image_transform_v2",
        "transform",
        -1,
    ): "skip",
    (
        "torchlens/preprocessing/_audit.py",
        '"apply the authority\'s own transform (resolution.transform) or "',
        "transform",
        -1,
    ): "apply",
    (
        "torchlens/preprocessing/_authorities.py",
        "f\"timm data config: input_size={authority.get('input_size')} \"",
        "input_size",
        0,
    ): "skip",
    (
        "torchlens/receptive_field/AGENTS.md",
        "`gradient()`, and `show()`; `show(gradient=True)` recomputes WITHOUT retain_graph",
        "gradient",
        1,
    ): "apply",
    (
        "torchlens/receptive_field/_gradient_forward.py",
        "support, None if descriptor is None else descriptor.layout",
        "layout",
        -1,
    ): "apply",
    (
        "torchlens/receptive_field/_types.py",
        'raise ReceptiveFieldError("Geometric axes are unavailable; use .gradient() instead.")',
        "gradient",
        -1,
    ): "apply",
    (
        "torchlens/receptive_field/_types.py",
        'raise ReceptiveFieldError("No windowed axes are available; use .gradient() instead.")',
        "gradient",
        -1,
    ): "apply",
    (
        "torchlens/receptive_field/_view.py",
        "Coordinates must be non-negative and in range; unlike ``.gradient()``",
        "gradient",
        -1,
    ): "apply",
    (
        "torchlens/receptive_field/_view.py",
        'raise ReceptiveFieldError("Geometric axes are unavailable; use .gradient() instead.")',
        "gradient",
        -1,
    ): "apply",
    (
        "torchlens/receptive_field/_view.py",
        "repeated ``show(gradient=True)`` or later ``gradient()`` calls over",
        "gradient",
        0,
    ): "apply",
    (
        "torchlens/receptive_field/_viz.py",
        'raise ReceptiveFieldError("Per-unit geometry is unavailable; use .gradient() instead.")',
        "gradient",
        -1,
    ): "apply",
    (
        "torchlens/receptive_field/_viz.py",
        'raise ReceptiveFieldError("The derived layout is ambiguous; use .gradient() instead.")',
        "gradient",
        -1,
    ): "apply",
    (
        "torchlens/receptive_field/_viz.py",
        'raise ReceptiveFieldError("gradient=True requires an explicit complete output unit.")',
        "gradient",
        0,
    ): "apply",
    ("torchlens/repgeom/_geometry.py", '"n_components": n_components,', "n_components", 0): "skip",
    (
        "torchlens/repgeom/_geometry.py",
        'raise ValueError("n_components must be at least 1.")',
        "n_components",
        0,
    ): "apply",
    (
        "torchlens/report/_summary_hub.py",
        '``tl.summary("gpt2", input_size=(1, 8))`` fetches the model\'s CONFIG only,',
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/transformer/_receipts.py",
        "ax.set_yticklabels([receipt.layer for receipt in rows], fontsize=_TEXT_PT)",
        "layer",
        -1,
    ): "apply",
    (
        "torchlens/transformer/_render_attention.py",
        'ax.set_ylabel(view.layer.split(".")[-1] or view.layer, fontsize=5, rotation=0)',
        "layer",
        -1,
    ): "skip",
    ("torchlens/transforms/_helpers.py", '"n_components": int(k),', "n_components", 0): "skip",
    (
        "torchlens/transforms/_helpers.py",
        "Converts to ``int`` (feeds ``srp(n_components=...)`` directly); the",
        "n_components",
        0,
    ): "apply",
    ("torchlens/transforms/_srp.py", '"n_components": n_components,', "n_components", 0): "skip",
    (
        "torchlens/transforms/_srp.py",
        '``srp(n_components=REQUIRED, seed=0, density="auto",',
        "n_components",
        0,
    ): "apply",
    (
        "torchlens/transforms/_srp.py",
        'f"srp(n_components={n_components!r}) needs a positive integer output "',
        "n_components",
        0,
    ): "apply",
    (
        "torchlens/user_funcs.py",
        "it and pass ``input_size=``, or omit both to infer -- synthesized",
        "input_size",
        0,
    ): "apply",
    (
        "torchlens/visualization/_render_flow.py",
        'f\'<text text-anchor="middle" x="{text_x:.2f}" y="{text_y:.2f}" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/visualization/code_panel.py",
        '\'<rect x="0" y="0" width="100%" height="100%" fill="white"/>\'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/visualization/code_panel.py",
        'f\'<svg x="0" y="{left_y:.2f}" width="{left.width:.2f}" \'',
        "x",
        0,
    ): "skip",
    (
        "torchlens/visualization/code_panel.py",
        'f\'<svg x="{left.width + gap:.2f}" y="{right_y:.2f}" \'',
        "x",
        0,
    ): "skip",
    # Integrator B (batch B1): sites the merged round-2 tree introduced. ``trace.layers`` is the
    # Trace accessor, not PredictionTrajectory.layers; ``.layout`` ends a module path string.
    # Occurrence -1 is the key the static pass computes for these unparsed lines.
    (
        "docs/reference/glossary.md",
        ": Every accessor (`trace.layers`, `trace.ops`, `trace.modules`, `trace.params`, ...) counts",
        "layers",
        -1,
    ): "skip",
    (
        "torchlens/errors/__init__.py",
        '"torchlens._visualization._rank_layout_internal.layout",',
        "layout",
        -1,
    ): "skip",
}

WORD = r"(?<![\w.]){}(?!\w)"


# --------------------------------------------------------------------------- type refs


@dataclass(frozen=True)
class T:
    """A static type: ``inst`` an instance of ``name``, ``cls`` the class object,
    ``iter`` a container of ``name``, ``other`` a known type that is no indexed class."""

    kind: str
    name: str


OTHER = T("other", "?")
CONTAINERS = {
    "tuple",
    "list",
    "Sequence",
    "Iterable",
    "Iterator",
    "set",
    "frozenset",
    "Collection",
    "Generator",
    "MutableSequence",
}
UNKNOWN_ANN = {"Any", "object", "_Any", "Self", "TypeVar"}


def ann_type(node: ast.expr | None, classes: dict[str, list[ClassInfo]]) -> T | None:
    """Return the type an annotation names, or ``None`` when it is open."""

    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return ann_type(ast.parse(node.value, mode="eval").body, classes)
        except SyntaxError:
            return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        parts = [p for p in (node.left, node.right) if not _is_none(p)]
        types = {ann_type(p, classes) for p in parts}
        return types.pop() if len(types) == 1 else None
    if isinstance(node, ast.Subscript):
        base = _terminal(node.value)
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base == "Optional":
            return ann_type(args[0], classes)
        if base == "Union":
            types = {ann_type(a, classes) for a in args if not _is_none(a)}
            return types.pop() if len(types) == 1 else None
        if base in CONTAINERS and args:
            inner = ann_type(args[0], classes)
            if inner is not None and inner.kind == "inst":
                return T("iter", inner.name)
            return OTHER
        if base in ("type", "Type") and args:
            inner = ann_type(args[0], classes)
            return T("cls", inner.name) if inner and inner.kind == "inst" else OTHER
        return OTHER if base not in UNKNOWN_ANN else None
    name = _terminal(node)
    if name is None or name in UNKNOWN_ANN:
        return None
    if name in classes:
        return T("inst", name)
    return OTHER


def _is_none(node: ast.expr) -> bool:
    return (isinstance(node, ast.Constant) and node.value is None) or (
        isinstance(node, ast.Name) and node.id == "None"
    )


def _terminal(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


# --------------------------------------------------------------------------- index


@dataclass
class ClassInfo:
    name: str
    path: str
    bases: list[str]
    fields: dict[str, ast.expr | None] = field(default_factory=dict)
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = field(default_factory=dict)


@dataclass
class Index:
    classes: dict[str, list[ClassInfo]] = field(default_factory=lambda: defaultdict(list))
    functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = field(
        default_factory=lambda: defaultdict(list)
    )
    fixtures: dict[str, T | None] = field(default_factory=dict)

    _ancestors: dict[str, frozenset[str]] = field(default_factory=dict)
    _producers: dict[str, frozenset[str]] = field(default_factory=dict)

    def ancestors(self, name: str) -> frozenset[str]:
        """``name`` and every base reachable from it, by simple class name."""

        cached = self._ancestors.get(name)
        if cached is not None:
            return cached
        out, stack = {name}, [name]
        while stack:
            for info in self.classes.get(stack.pop(), []):
                for base in info.bases:
                    if base not in out:
                        out.add(base)
                        stack.append(base)
        self._ancestors[name] = frozenset(out)
        return self._ancestors[name]

    def producers(self, owners: tuple[str, ...]) -> frozenset[str]:
        """Identifiers that can hand out an owner instance: the owner classes, their
        subclasses, and every function, method, field, property or fixture typed to one."""

        key = "/".join(owners)
        cached = self._producers.get(key)
        if cached is not None:
            return cached
        kinds = {n for n in self.classes if self.ancestors(n) & set(owners)}
        names = set(kinds)

        def hits(node: ast.expr | None) -> bool:
            typ = ann_type(node, self.classes)
            return typ is not None and typ.kind in ("inst", "iter", "cls") and typ.name in kinds

        for infos in self.classes.values():
            for info in infos:
                names.update(a for a, ann in info.fields.items() if hits(ann))
                names.update(m for m, fn in info.methods.items() if hits(fn.returns))
        for fname, fns in self.functions.items():
            if any(hits(fn.returns) for fn in fns):
                names.add(fname)
        for fname, typ in self.fixtures.items():
            if typ is not None and typ.name in kinds:
                names.add(fname)
        self._producers[key] = frozenset(names)
        return self._producers[key]

    def member_ann(self, cls: str, attr: str) -> T | None:
        for anc in self.ancestors(cls):
            for info in self.classes.get(anc, []):
                if attr in info.fields:
                    return ann_type(info.fields[attr], self.classes)
                if attr in info.methods:
                    fn = info.methods[attr]
                    if _is_property(fn):
                        return ann_type(fn.returns, self.classes)
        return None

    def method_return(self, cls: str, meth: str) -> T | None:
        for anc in self.ancestors(cls):
            for info in self.classes.get(anc, []):
                if meth in info.methods:
                    fn = info.methods[meth]
                    ret = fn.returns
                    if _terminal(ret) == "Self" if ret is not None else False:
                        return T("inst", cls)
                    return ann_type(ret, self.classes)
        return None

    def function_return(self, name: str) -> T | None:
        types = {ann_type(fn.returns, self.classes) for fn in self.functions.get(name, [])}
        return types.pop() if len(types) == 1 else None


def _is_property(fn: ast.AST) -> bool:
    return any(
        _terminal(d) in ("property", "cached_property") for d in getattr(fn, "decorator_list", [])
    )


def _is_fixture(fn: ast.AST) -> bool:
    for dec in getattr(fn, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        if _terminal(target) == "fixture":
            return True
    return False


def build_index(modules: dict[str, ast.Module]) -> Index:
    index = Index()
    for path, tree in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info = ClassInfo(node.name, path, [b for b in map(_terminal, node.bases) if b])
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        info.fields[stmt.target.id] = stmt.annotation
                    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        info.methods[stmt.name] = stmt
                index.classes[node.name].append(info)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                index.functions[node.name].append(node)
    for path, tree in modules.items():
        if not path.startswith("tests/"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_fixture(node):
                index.fixtures[node.name] = _fixture_type(node, index)
    return index


def _fixture_type(fn: ast.FunctionDef | ast.AsyncFunctionDef, index: Index) -> T | None:
    declared = ann_type(fn.returns, index.classes)
    if declared is not None:
        return declared
    scope = Scope(index, fn, None)
    types = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.Return, ast.Yield)) and node.value is not None:
            types.add(scope.infer(node.value))
    return types.pop() if len(types) == 1 else None


# --------------------------------------------------------------------------- scopes


class Scope:
    """Flow-insensitive variable types for one function or module body."""

    def __init__(
        self, index: Index, node: ast.AST, cls: str | None, parent: Scope | None = None
    ) -> None:
        self.index = index
        self.cls = cls
        self.parent = parent
        self.env: dict[str, T | None] = {}
        self.guards: list[tuple[ast.AST, str, T]] = []
        self._bind(node)

    def lookup(self, name: str) -> T | None:
        if name in self.env:
            return self.env[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        if name in self.index.classes:
            return T("cls", name)
        return None

    def _set(self, name: str, value: T | None) -> None:
        if name in self.env and self.env[name] != value:
            self.env[name] = None  # conflicting bindings: open
        else:
            self.env[name] = value

    def _bind(self, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            self._bind_args(node)
        for child in _own_nodes(node):
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                self._set(child.target.id, ann_type(child.annotation, self.index.classes))
            elif isinstance(child, ast.Assign):
                value = self.infer(child.value)
                for target in child.targets:
                    self._bind_target(target, value)
            elif isinstance(child, (ast.For, ast.AsyncFor, ast.comprehension)):
                src = self.infer(child.iter)
                elem = T("inst", src.name) if src and src.kind == "iter" else None
                if src == OTHER:
                    elem = None
                self._bind_target(child.target, elem)
            elif isinstance(child, ast.withitem) and child.optional_vars is not None:
                self._bind_target(child.optional_vars, self.infer(child.context_expr))
            elif isinstance(child, ast.NamedExpr):
                self._set(child.target.id, self.infer(child.value))
            elif isinstance(child, ast.If):
                self._guard(child)
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    if alias.asname and alias.name in self.index.classes:
                        self.env[alias.asname] = T("cls", alias.name)

    def _bind_args(self, fn: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        args = fn.args
        every = args.posonlyargs + args.args + args.kwonlyargs
        for pos, arg in enumerate(every):
            declared = ann_type(arg.annotation, self.index.classes) if arg.annotation else None
            if pos == 0 and self.cls and arg.arg in ("self", "cls"):
                is_cls = arg.arg == "cls"
                declared = T("cls" if is_cls else "inst", self.cls)
            elif declared is None and arg.annotation is None and arg.arg in self.index.fixtures:
                declared = self.index.fixtures[arg.arg]
            self.env[arg.arg] = declared
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                self.env[extra.arg] = OTHER

    def _bind_target(self, target: ast.expr, value: T | None) -> None:
        if isinstance(target, ast.Name):
            self._set(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_target(elt, None)

    def _guard(self, node: ast.If) -> None:
        test = node.test
        if (
            isinstance(test, ast.Call)
            and _terminal(test.func) == "isinstance"
            and len(test.args) == 2
            and isinstance(test.args[0], ast.Name)
        ):
            name = _terminal(test.args[1])
            if name in self.index.classes:
                self.guards.append((node, test.args[0].id, T("inst", name)))

    def infer(self, expr: ast.expr) -> T | None:  # noqa: PLR0911, PLR0912 -- one case per expression kind
        if isinstance(expr, ast.Name):
            return self.lookup(expr.id)
        if isinstance(expr, ast.Constant):
            return OTHER
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp)):
            elts = getattr(expr, "elts", None)
            if elts:
                kinds = {self.infer(e) for e in elts}
                one = kinds.pop() if len(kinds) == 1 else None
                if one is not None and one.kind == "inst":
                    return T("iter", one.name)
            return None
        if isinstance(expr, (ast.Dict, ast.DictComp, ast.JoinedStr, ast.Compare, ast.BoolOp)):
            return OTHER
        if isinstance(expr, ast.Await):
            return self.infer(expr.value)
        if isinstance(expr, ast.IfExp):
            a, b = self.infer(expr.body), self.infer(expr.orelse)
            return a if a == b else None
        if isinstance(expr, ast.Subscript):
            base = self.infer(expr.value)
            if base is not None and base.kind == "iter" and not isinstance(expr.slice, ast.Slice):
                return T("inst", base.name)
            return None
        if isinstance(expr, ast.Attribute):
            base = self.infer(expr.value)
            if base is not None and base.kind == "inst":
                return self.index.member_ann(base.name, expr.attr)
            return None
        if isinstance(expr, ast.Call):
            return self._infer_call(expr)
        return None

    def _infer_call(self, call: ast.Call) -> T | None:
        func = call.func
        name = _terminal(func)
        if name is None:
            return None
        if name == "replace" and call.args:
            return self.infer(call.args[0])
        if isinstance(func, ast.Attribute):
            base = self.infer(func.value)
            if base is not None and base.kind == "cls":
                ret = self.index.method_return(base.name, name)
                return ret if ret is not None else None
            if base is not None and base.kind == "inst":
                return self.index.method_return(base.name, name)
            if base is not None:
                return None
            if not (isinstance(func.value, ast.Name) and not _is_bound(self, func.value.id)):
                return None  # a method of an unknown receiver, not a module function
        callee = self.lookup(name) if isinstance(func, ast.Name) else None
        if callee is not None and callee.kind == "cls":
            return T("inst", callee.name)
        if name in self.index.classes and (isinstance(func, ast.Attribute) or callee is None):
            return T("inst", name)
        return self.index.function_return(name)

    def type_at(self, name: str, node: ast.AST, parents: dict[ast.AST, ast.AST]) -> T | None:
        for guard, var, typ in self.guards:
            if var == name and _inside(node, guard.body, parents):
                return typ
        return self.lookup(name)


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Walk ``node`` without entering nested function, lambda or class bodies."""

    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(child))


def _inside(node: ast.AST, body: list[ast.stmt], parents: dict[ast.AST, ast.AST]) -> bool:
    targets = {id(s) for s in body}
    cur: ast.AST | None = node
    while cur is not None:
        if id(cur) in targets:
            return True
        cur = parents.get(cur)
    return False


# --------------------------------------------------------------------------- sources


@dataclass
class Source:
    """One parseable unit: a python file, or the code of a page or notebook."""

    path: str
    code: str
    # pseudo line (1-based) -> (location key, file line text, column shift in characters)
    lines: dict[int, tuple[object, str, int]]


FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*(python|py)\s*$")


def md_source(path: str, text: str) -> Source | None:
    """Join the python fences of a page into one pseudo-module."""

    out: list[str] = []
    lines: dict[int, tuple[object, str, int]] = {}
    raw = text.split("\n")
    i = 0
    while i < len(raw):
        match = FENCE.match(raw[i])
        if not match:
            i += 1
            continue
        indent, ticks = len(match.group(1)), match.group(2)
        body: list[tuple[int, str]] = []
        i += 1
        while i < len(raw) and not raw[i].strip().startswith(ticks):
            body.append((i, raw[i]))
            i += 1
        i += 1
        chunk = [line[indent:] if line[:indent].strip() == "" else line for _, line in body]
        try:
            ast.parse("\n".join(chunk))
        except SyntaxError:
            continue
        for (lineno, line), code_line in zip(body, chunk, strict=True):
            out.append(code_line)
            lines[len(out)] = (lineno, line, len(line) - len(code_line))
    return Source(path, "\n".join(out), lines) if out else None


def nb_source(path: str, nb: dict) -> Source | None:
    out: list[str] = []
    lines: dict[int, tuple[object, str, int]] = {}
    for c_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        cell_lines = "".join(src if isinstance(src, list) else [src]).split("\n")
        if any(line.lstrip().startswith(("%", "!")) for line in cell_lines):
            cell_lines = [
                "" if line.lstrip().startswith(("%", "!")) else line for line in cell_lines
            ]
        try:
            ast.parse("\n".join(cell_lines))
        except SyntaxError:
            continue
        for l_idx, line in enumerate(cell_lines):
            out.append(line)
            lines[len(out)] = ((c_idx, l_idx), line, 0)
    return Source(path, "\n".join(out), lines) if out else None


def py_source(path: str, text: str) -> Source:
    raw = text.split("\n")
    return Source(path, text, {n + 1: (n, line, 0) for n, line in enumerate(raw)})


# --------------------------------------------------------------------------- sites


@dataclass
class Site:
    path: str
    loc: object  # file line index (0-based) or (cell, line) for notebooks
    line: str
    col: int  # character column in ``line``
    old: str
    new: str
    row: str
    verdict: str  # apply | other | undecided
    why: str

    def key(self) -> tuple[str, str, str, int]:
        pattern = re.compile(WORD.format(re.escape(self.old)))
        starts = [m.start() for m in pattern.finditer(self.line)]
        occurrence = starts.index(self.col) if self.col in starts else -1
        return (self.path, self.line.strip(), self.old, occurrence)


def _char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


class Collector:
    """Walk one source and classify every site of every row."""

    def __init__(
        self, index: Index, src: Source, targets: dict[str, list[ParamTarget]], tree: ast.Module
    ) -> None:
        self.index = index
        self.src = src
        self.targets = targets
        self.sites: list[Site] = []
        self.tree = tree
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node

    # position helpers -------------------------------------------------------
    def _at(
        self, lineno: int, byte_col: int, row: str, old: str, new: str, verdict: str, why: str
    ) -> None:
        loc, line, shift = self.src.lines[lineno]
        col = _char_col(line[shift:], byte_col) + shift
        if line[col : col + len(old)] != old:
            found = re.search(WORD.format(re.escape(old)), line[col:])
            if not found:
                return
            col += found.start()
        self.sites.append(Site(self.src.path, loc, line, col, old, new, row, verdict, why))

    def _attr_pos(self, node: ast.Attribute) -> tuple[int, int]:
        assert node.end_lineno is not None and node.end_col_offset is not None
        _, line, shift = self.src.lines[node.end_lineno]
        end_bytes = node.end_col_offset
        return node.end_lineno, end_bytes - len(node.attr.encode("utf-8"))

    # scopes -----------------------------------------------------------------
    def run(self) -> list[Site]:
        self._visit(self.tree, Scope(self.index, self.tree, None), None)
        return self.sites

    def _visit(self, node: ast.AST, scope: Scope, cls: ClassInfo | str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._class_def(child, scope)
                self._visit(child, scope, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                inner = Scope(self.index, child, cls if isinstance(cls, str) else None, scope)
                if not isinstance(child, ast.Lambda):
                    self._func_def(child, inner, cls if isinstance(cls, str) else None)
                self._visit(child, inner, None)
            else:
                self._node(child, scope)
                self._visit(child, scope, cls)

    # member and keyword sites -----------------------------------------------
    def _owner_verdict(self, typ: T | None, owners: tuple[str, ...]) -> tuple[str, str]:
        if typ is None:
            return "undecided", "receiver type unknown"
        if typ.kind in ("inst", "cls"):
            ancestors = self.index.ancestors(typ.name)
            if ancestors & set(owners):
                return "apply", f"receiver {typ.name}"
            owner_ancestors = set().union(*(self.index.ancestors(o) for o in owners))
            if typ.name in owner_ancestors:
                return "undecided", f"receiver {typ.name} is a base of the owner"
            return "other", f"receiver {typ.name}"
        return "other", f"receiver {typ.kind}"

    def _scope_type(self, expr: ast.expr, scope: Scope) -> T | None:
        if isinstance(expr, ast.Name):
            return scope.type_at(expr.id, expr, self.parents)
        return scope.infer(expr)

    def _node(self, node: ast.AST, scope: Scope) -> None:
        if isinstance(node, ast.Attribute):
            for row in MEMBER_BY_OLD.get(node.attr, []):
                verdict, why = self._owner_verdict(self._scope_type(node.value, scope), row.owners)
                line, col = self._attr_pos(node)
                self._at(line, col, row_id(row), row.old, row.new, verdict, why)
        elif isinstance(node, ast.Call):
            self._call(node, scope)

    def _call(self, call: ast.Call, scope: Scope) -> None:
        callee_t = self._callee_class(call, scope)
        for kw in call.keywords:
            if kw.arg is None:
                continue
            for row in MEMBER_BY_OLD.get(kw.arg, []):
                if not row.keyword:
                    continue
                if callee_t is None:
                    verdict, why = "undecided", "callee unknown"
                else:
                    verdict, why = self._owner_verdict(callee_t, row.owners)
                self._at(kw.lineno, kw.col_offset, row_id(row), row.old, row.new, verdict, why)
            for row in PARAM_BY_OLD.get(kw.arg, []):
                verdict, why = self._param_call_verdict(call, scope, row)
                self._at(kw.lineno, kw.col_offset, row_id(row), row.old, row.new, verdict, why)
            value = kw.value
            if kw.arg == "argument" and isinstance(value, ast.Constant):
                for row in PARAM_BY_OLD.get(str(value.value), []):
                    self._at(
                        value.lineno,
                        value.col_offset + 1,
                        row_id(row),
                        row.old,
                        row.new,
                        "undecided",
                        "argument= may name a renamed keyword",
                    )
        name = _terminal(call.func)
        if name in ("getattr", "hasattr", "setattr") and len(call.args) >= 2:
            arg = call.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                for row in MEMBER_BY_OLD.get(arg.value, []):
                    verdict, why = self._owner_verdict(
                        self._scope_type(call.args[0], scope), row.owners
                    )
                    self._at(
                        arg.lineno, arg.col_offset + 1, row_id(row), row.old, row.new, verdict, why
                    )

    def _callee_class(self, call: ast.Call, scope: Scope) -> T | None:
        """The class a call constructs or ``replace``s; ``OTHER`` for any other callee."""

        name = _terminal(call.func)
        if name == "replace" and call.args:
            return self._scope_type(call.args[0], scope)
        if isinstance(call.func, ast.Name):
            local = scope.lookup(call.func.id)
            if local is not None and local.kind == "cls":
                return local
            if local is None and _is_bound(scope, call.func.id):
                return None  # a local of unknown type might hold a class
            return OTHER
        if isinstance(call.func, ast.Attribute):
            if name in self.index.classes:
                return T("cls", name)
            base = self._scope_type(call.func.value, scope)
            if base is not None and base.kind == "cls":
                ret = self.index.method_return(base.name, name)
                return ret if ret is not None else OTHER  # alternate constructor
            return OTHER
        return None

    def _param_call_verdict(self, call: ast.Call, scope: Scope, row: ParamRow) -> tuple[str, str]:
        name = _terminal(call.func)
        targets = self.targets.get(row_id(row), [])
        hits = [t for t in targets if t.func == name or (t.func == "__init__" and t.cls == name)]
        if isinstance(call.func, ast.Attribute):
            base = self._scope_type(call.func.value, scope)
            if base is not None and base.kind in ("inst", "cls"):
                ancestors = self.index.ancestors(base.name)
                if any(t.cls in ancestors for t in hits if t.func == name):
                    return "apply", f"method of {base.name}"
                if base.kind == "cls" and any(t.cls == name and t.func == "__init__" for t in hits):
                    return "apply", f"constructs {name}"
                return "other", f"receiver {base.name}"
        if not hits:
            return "other", f"callee {name} is not a renamed callable"
        rivals = [
            fn
            for fn in ALL_DEFS.get(name, [])
            if row.old in _arg_names(fn) and not any(t.node is fn for t in targets)
        ]
        if rivals:
            return "undecided", f"{name} also defined elsewhere with {row.old}="
        return "apply", f"callee {name}"

    # definitions --------------------------------------------------------------
    def _class_def(self, node: ast.ClassDef, scope: Scope) -> None:
        ancestors = self.index.ancestors(node.name)
        for row in MEMBER_ROWS:
            if not ancestors & set(row.owners):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == row.old:
                        tgt = stmt.target
                        self._at(
                            tgt.lineno,
                            tgt.col_offset,
                            row_id(row),
                            row.old,
                            row.new,
                            "apply",
                            "field definition",
                        )
                elif (
                    isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and stmt.name == row.old
                ):
                    self._def_name(stmt, row_id(row), row.old, row.new)
            self._docstring(node, row_id(row), row.old, row.new)

    def _def_name(self, fn: ast.AST, row: str, old: str, new: str) -> None:
        _, line, shift = self.src.lines[fn.lineno]
        found = re.search(r"\bdef\s+" + re.escape(old) + r"\b", line)
        if found:
            col = found.end() - len(old)
            self.sites.append(
                Site(
                    self.src.path,
                    self.src.lines[fn.lineno][0],
                    line,
                    col,
                    old,
                    new,
                    row,
                    "apply",
                    "method definition",
                )
            )

    def _docstring(self, node: ast.AST, row: str, old: str, new: str) -> None:
        body = getattr(node, "body", [])
        if not body or not isinstance(body[0], ast.Expr):
            return
        doc = body[0].value
        if not (isinstance(doc, ast.Constant) and isinstance(doc.value, str)):
            return
        entry = re.compile(r"^\s*(?:\w+\s*/\s*)*(" + re.escape(old) + r")(?:\s*/\s*\w+)*\s*:")
        literal = re.compile(r"``(" + re.escape(old) + r")``")
        for lineno in range(doc.lineno, (doc.end_lineno or doc.lineno) + 1):
            loc, line, _ = self.src.lines[lineno]
            for pattern in (entry, literal):
                for m in pattern.finditer(line):
                    self.sites.append(
                        Site(
                            self.src.path,
                            loc,
                            line,
                            m.start(1),
                            old,
                            new,
                            row,
                            "apply",
                            "docstring entry",
                        )
                    )

    def _func_def(
        self, fn: ast.FunctionDef | ast.AsyncFunctionDef, scope: Scope, cls: str | None
    ) -> None:
        for row in PARAM_ROWS:
            targets = self.targets.get(row_id(row), [])
            if not any(t.node is fn for t in targets):
                continue
            if row.new in _bound_names(fn):
                self._at(
                    fn.lineno,
                    fn.col_offset,
                    row_id(row),
                    "def",
                    "def",
                    "undecided",
                    f"{row.new} already bound in {fn.name}",
                )
                continue
            for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
                if arg.arg == row.old:
                    self._at(
                        arg.lineno,
                        arg.col_offset,
                        row_id(row),
                        row.old,
                        row.new,
                        "apply",
                        "parameter",
                    )
            for name in _free_uses(fn, row.old):
                self._at(
                    name.lineno,
                    name.col_offset,
                    row_id(row),
                    row.old,
                    row.new,
                    "apply",
                    "parameter use",
                )
            for const in ast.walk(fn):
                if isinstance(const, ast.Constant) and const.value == row.old:
                    verdict, why = self._name_string_verdict(const)
                    if verdict:
                        self._at(
                            const.lineno,
                            const.col_offset + 1,
                            row_id(row),
                            row.old,
                            row.new,
                            verdict,
                            why,
                        )
            self._docstring(fn, row_id(row), row.old, row.new)

    def _name_string_verdict(self, const: ast.Constant) -> tuple[str, str]:
        """Classify a string equal to a renamed parameter inside its own function.

        ``locals()`` snapshots are trimmed by name (``kwargs.pop("x")``, a ``for`` over a
        tuple of names) and must follow the rename. A dict key could be a payload or spec
        field that keeps its spelling, so it goes to hand review. Anything else is ignored.
        """

        parent = self.parents.get(const)
        if isinstance(parent, ast.Call) and _terminal(parent.func) == "pop":
            return "apply", "locals() key popped by name"
        if isinstance(parent, (ast.Tuple, ast.List)) and isinstance(
            self.parents.get(parent), (ast.For, ast.comprehension)
        ):
            return "apply", "locals() key in a name loop"
        if isinstance(parent, ast.keyword) and parent.arg == "argument":
            return "apply", "argument= names the parameter"
        if isinstance(parent, ast.Dict) and const in parent.keys:
            return "undecided", "dict key spelled like the parameter"
        return "", ""


def _is_bound(scope: Scope | None, name: str) -> bool:
    while scope is not None:
        if name in scope.env:
            return True
        scope = scope.parent
    return False


def _arg_names(fn: ast.AST) -> set[str]:
    args = getattr(fn, "args", None)
    if args is None:
        return set()
    return {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}


def _bound_names(fn: ast.AST) -> set[str]:
    names = set(_arg_names(fn))
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _free_uses(fn: ast.AST, name: str) -> Iterator[ast.Name]:
    """Name nodes in ``fn`` that refer to its own parameter ``name``."""

    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if name in _arg_names(node) or any(
                isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
                for n in ast.walk(node)
            ):
                continue  # a nested scope rebinding the name keeps its own
            if not isinstance(node, ast.Lambda):
                stack.extend(node.decorator_list)
                stack.extend(node.body)
            else:
                stack.append(node.body)
            continue
        comprehensions = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        if isinstance(node, comprehensions) and any(
            isinstance(n, ast.Name) and n.id == name
            for g in node.generators
            for n in ast.walk(g.target)
        ):
            continue  # the comprehension rebinds the name
        if isinstance(node, ast.Name) and node.id == name:
            yield node
        stack.extend(ast.iter_child_nodes(node))


# --------------------------------------------------------------------------- rows


@dataclass
class ParamTarget:
    cls: str | None
    func: str
    node: ast.AST
    path: str


def row_id(row: MemberRow | ParamRow) -> str:
    if isinstance(row, MemberRow):
        return f"member:{'/'.join(row.owners)}.{row.old}"
    return f"param:{row.old}"


MEMBER_ROWS: list[MemberRow] = []
MEMBER_BY_OLD: dict[str, list[MemberRow]] = defaultdict(list)
PARAM_BY_OLD: dict[str, list[ParamRow]] = defaultdict(list)
ALL_DEFS: dict[str, list[ast.AST]] = defaultdict(list)


ROW_BY_ID: dict[str, MemberRow | ParamRow] = {}


def _register_rows() -> None:
    for row in ROWS + PARAM_ROWS:
        ROW_BY_ID[row_id(row)] = row
        if isinstance(row, MemberRow):
            MEMBER_ROWS.append(row)
            MEMBER_BY_OLD[row.old].append(row)
        else:
            PARAM_BY_OLD[row.old].append(row)


def find_targets(modules: dict[str, ast.Module]) -> dict[str, list[ParamTarget]]:
    """Locate each keyword row's callables in ``torchlens/``."""

    defs: dict[tuple[str | None, str], list[tuple[ast.AST, str]]] = defaultdict(list)
    for path, tree in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ALL_DEFS[node.name].append(node)
            elif isinstance(node, ast.ClassDef) and path.startswith("torchlens/"):
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defs[(node.name, stmt.name)].append((stmt, path))
        if path.startswith("torchlens/"):
            for stmt in tree.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs[(None, stmt.name)].append((stmt, path))
    found: dict[str, list[ParamTarget]] = defaultdict(list)
    for row in PARAM_ROWS:
        for spec in row.callables:
            cls, _, func = spec.rpartition(".")
            for node, path in defs.get((cls or None, func), []):
                if row.old in _arg_names(node):
                    found[row_id(row)].append(ParamTarget(cls or None, func, node, path))
    return found


# Keyword rows: the callables are ``Class.method``, ``Class.__init__`` or module-level names.
# Left out on purpose (see the lane notes): ``Bundle.run`` (restructured to ``inputs`` by the
# bundle redesign), ``Trace.run`` (already takes both ``x`` and ``inputs``; the run-door
# redesign settles it), and ``edges=`` on graph-edge callables such as ``causal_audit`` (the
# decided row covers histogram bin edges only, which the HistogramPoint member row renames).
PARAM_ROWS = (
    ParamRow(
        (
            "Extractor.forward",
            "Layer.do",
            "Op.do",
            "TraceInterventionMixin.do",
            "dump",
            "sweep_sites",
            "extract",
            "run",
            "list_sites",
            "pluck",
            "graph_breaks",
            "sweep",
        ),
        "x",
        "inputs",
    ),
    ParamRow(("CometSink.__init__",), "experiment", "comet_experiment"),
    ParamRow(("MergedTrace.__init__",), "handles", "rank_handles"),
    ParamRow(("summary", "trace", "TraceVisualizationMixin.summary"), "input_size", "input_shape"),
    ParamRow(
        ("classical_mds", "PCA.__init__", "srp", "srp_fidelity_probe"),
        "n_components",
        "num_components",
    ),
    ParamRow(("FacetSpec.heads",), "n_heads", "num_heads"),
    ParamRow(("GqaInfo.__init__",), "n_query_heads", "num_query_heads"),
    ParamRow(
        ("Trace.__init__", "FuncCallLocation.__init__"), "num_context_lines", "source_context_lines"
    ),
    ParamRow(("AttentionView.crop_to",), "queries", "query_slice"),
    ParamRow(("ReceptiveFieldView.show", "show", "node_spec"), "gradient", "grad"),
)


# --------------------------------------------------------------------------- driver


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *TREES], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.split("\n")
    keep = (".py", ".md", ".ipynb", ".rst")
    return sorted(p for p in out if p.endswith(keep) and not p.startswith("tools/codemods/"))


def load_sources(paths: Iterable[str]) -> tuple[list[Source], dict[str, str], dict[str, dict]]:
    sources: list[Source] = []
    texts: dict[str, str] = {}
    notebooks: dict[str, dict] = {}
    for path in paths:
        text = (ROOT / path).read_text(encoding="utf-8")
        texts[path] = text
        src: Source | None
        if path.endswith(".py"):
            src = py_source(path, text)
        elif path.endswith(".ipynb"):
            nb = json.loads(text)
            notebooks[path] = nb
            src = nb_source(path, nb)
        else:
            src = md_source(path, text)
        if src is not None:
            sources.append(src)
    return sources, texts, notebooks


def prose_sites(path: str, text: str, nb: dict | None) -> Iterator[Site]:
    """Qualified ``Owner.old`` spellings anywhere, including comments and prose."""

    for row in MEMBER_ROWS:
        if row.old not in text:
            continue
        for owner in row.owners:
            if owner not in text:
                continue
            pattern = re.compile(rf"\b{owner}\.({re.escape(row.old)})\b")
            for loc, line in _text_lines(text, nb):
                for m in pattern.finditer(line):
                    yield Site(
                        path,
                        loc,
                        line,
                        m.start(1),
                        row.old,
                        row.new,
                        row_id(row),
                        "apply",
                        "qualified spelling",
                    )


def _text_lines(text: str, nb: dict | None) -> Iterator[tuple[object, str]]:
    if nb is None:
        yield from enumerate(text.split("\n"))
        return
    for c_idx, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", [])
        for l_idx, line in enumerate("".join(src if isinstance(src, list) else [src]).split("\n")):
            yield (c_idx, l_idx), line


def leftover_sites(
    path: str, text: str, nb: dict | None, seen: set[tuple[object, int, str]]
) -> Iterator[Site]:
    """Textual spellings no parsed site covered (``.old`` or ``old=``), for review."""

    for row in MEMBER_ROWS:
        if row.old not in text:
            continue
        pattern = re.compile(
            rf"\.({re.escape(row.old)})\b(?!\s*\()"
            if row.keyword
            else rf"\.({re.escape(row.old)})\b"
        )
        yield from _left(path, text, nb, seen, pattern, row)
    for row in PARAM_ROWS:
        if row.old + "=" not in text:
            continue
        yield from _left(
            path, text, nb, seen, re.compile(rf"(?<![\w.])({re.escape(row.old)})=(?!=)"), row
        )


def _left(path, text, nb, seen, pattern, row) -> Iterator[Site]:  # noqa: ANN001
    for loc, line in _text_lines(text, nb):
        for m in pattern.finditer(line):
            if (loc, m.start(1), row.old) not in seen:
                yield Site(
                    path,
                    loc,
                    line,
                    m.start(1),
                    row.old,
                    row.new,
                    row_id(row),
                    "undecided",
                    "not parsed",
                )


def classify(
    paths: list[str],
) -> tuple[list[Site], dict[str, str], dict[str, dict], dict[str, list[ParamTarget]]]:
    _register_rows()
    sources, texts, notebooks = load_sources(paths)
    modules = {}
    for src in sources:
        try:
            modules[src.path] = ast.parse(src.code)
        except SyntaxError:
            continue
    index = build_index({p: m for p, m in modules.items() if p.startswith(INDEX_TREES)})
    targets = find_targets(modules)
    sites: list[Site] = []
    tokens = {row.old for row in ROWS + PARAM_ROWS}
    target_paths = {t.path for ts in targets.values() for t in ts}
    for src in sources:
        if src.path not in modules:
            continue
        if src.path not in target_paths and not any(tok in src.code for tok in tokens):
            continue
        try:
            sites.extend(Collector(index, src, targets, modules[src.path]).run())
        except (KeyError, RecursionError) as exc:  # a mapping gap: report the file, never guess
            print(f"warning: {src.path}: {exc!r}", file=sys.stderr)
    unique: dict[tuple[str, object, int, str], Site] = {}
    for site in sites:
        prior = unique.get((site.path, site.loc, site.col, site.old))
        if prior is None or (site.verdict == "apply" and prior.verdict != "apply"):
            unique[(site.path, site.loc, site.col, site.old)] = site
    for path, text in texts.items():
        if not any(tok in text for tok in tokens):
            continue
        nb = notebooks.get(path)
        for site in prose_sites(path, text, nb):
            prior = unique.get((path, site.loc, site.col, site.old))
            if prior is None or prior.verdict != "apply":
                unique[(path, site.loc, site.col, site.old)] = site
        seen = {(loc, col, old) for (p, loc, col, old) in unique if p == path}
        for site in leftover_sites(path, text, nb, seen):
            unique.setdefault((path, site.loc, site.col, site.old), site)
    words: dict[str, set[str]] = {}
    for site in unique.values():
        if site.verdict != "undecided" or site.why.startswith(("parameter", "def")):
            continue
        row = ROW_BY_ID[site.row]
        if site.path not in words:
            words[site.path] = set(re.findall(r"[A-Za-z_]\w*", texts[site.path]))
        if isinstance(row, MemberRow):
            reach = index.producers(row.owners)
        else:
            reach = {
                t.func if t.func != "__init__" else (t.cls or "") for t in targets.get(site.row, [])
            }
        if not reach & words[site.path]:
            site.verdict, site.why = "other", f"no producer of the owner in file ({site.why})"
    final = []
    for site in unique.values():
        decided = OVERRIDES.get(site.key()) if site.verdict == "undecided" else None
        if decided is not None:
            site.verdict, site.why = decided, f"override ({site.why})"
        final.append(site)
    final.extend(listed_sites({s.key() for s in final}, texts, notebooks))
    return final, texts, notebooks, targets


def listed_sites(
    known: set[tuple[str, str, str, int]], texts: dict[str, str], notebooks: dict[str, dict]
) -> Iterator[Site]:
    """Hand-listed ``apply`` overrides the static pass cannot see (``**kwargs`` dicts,
    test doubles standing in for an owner): renamed where the listed line still stands."""

    for key, verdict in OVERRIDES.items():
        path, text, old, occurrence = key
        if verdict != "apply" or key in known or occurrence < 0:
            continue
        if path not in texts:
            continue  # the file is gone or outside the scanned trees
        row = next((r for r in ROWS + PARAM_ROWS if r.old == old), None)
        if row is None:
            continue
        pattern = re.compile(WORD.format(re.escape(old)))
        for loc, line in _text_lines(texts[path], notebooks.get(path)):
            if line.strip() != text:
                continue
            starts = [m.start() for m in pattern.finditer(line)]
            if occurrence < len(starts):
                site = Site(
                    path,
                    loc,
                    line,
                    starts[occurrence],
                    old,
                    row.new,
                    row_id(row),
                    "apply",
                    "listed site",
                )
                yield site


def apply_edits(sites: list[Site], texts: dict[str, str], notebooks: dict[str, dict]) -> int:
    by_path: dict[str, list[Site]] = defaultdict(list)
    for site in sites:
        if site.verdict == "apply" and site.old != site.new:
            by_path[site.path].append(site)
    changed = 0
    for path, edits in by_path.items():
        nb = notebooks.get(path)
        if nb is None:
            lines = texts[path].split("\n")
            for site in sorted(edits, key=lambda s: (s.loc, s.col), reverse=True):
                lines[site.loc] = _swap(lines[site.loc], site)
            new_text = "\n".join(lines)
        else:
            new_text = _apply_nb(texts[path], nb, edits)
        if new_text != texts[path]:
            (ROOT / path).write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def _swap(line: str, site: Site) -> str:
    if line[site.col : site.col + len(site.old)] != site.old:
        raise RuntimeError(f"stale site {site.path}:{site.loc}:{site.col} {site.old!r}")
    return line[: site.col] + site.new + line[site.col + len(site.old) :]


def _apply_nb(text: str, nb: dict, edits: list[Site]) -> str:
    indent = 1 if text.startswith('{\n "') else 2
    if json.dumps(nb, indent=indent, ensure_ascii=False) + "\n" != text:
        raise RuntimeError("notebook does not round-trip its JSON formatting; edit by hand")
    by_cell: dict[int, list[Site]] = defaultdict(list)
    for site in edits:
        by_cell[site.loc[0]].append(site)  # type: ignore[index]
    for c_idx, cell_edits in by_cell.items():
        cell = nb["cells"][c_idx]
        src = cell["source"]
        lines = "".join(src if isinstance(src, list) else [src]).split("\n")
        for site in sorted(cell_edits, key=lambda s: (s.loc, s.col), reverse=True):
            l_idx = site.loc[1]  # type: ignore[index]
            lines[l_idx] = _swap(lines[l_idx], site)
        joined = "\n".join(lines)
        cell["source"] = [ln + "\n" for ln in joined.split("\n")[:-1]] + [joined.split("\n")[-1]]
        if cell["source"] and cell["source"][-1] == "":
            cell["source"].pop()
        if not isinstance(src, list):
            cell["source"] = joined
    return json.dumps(nb, indent=indent, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the edits")
    parser.add_argument("--report", type=Path, help="write every site with its verdict as JSON")
    parser.add_argument("--targets", action="store_true", help="list each keyword row's callables")
    parser.add_argument("paths", nargs="*", help="limit to these tracked files")
    args = parser.parse_args(argv)
    paths = tracked_files()
    if args.paths:
        paths = [p for p in paths if p in set(args.paths)]
    sites, texts, notebooks, targets = classify(paths)
    if args.targets:
        for row in PARAM_ROWS:
            for t in targets.get(row_id(row), []):
                print(f"{row_id(row)}\t{t.path}:{t.node.lineno}\t{t.cls or ''}.{t.func}")
    counts = Counter(s.verdict for s in sites if s.old == s.new or s.old != s.new)
    per_row: dict[str, Counter[str]] = defaultdict(Counter)
    for s in sites:
        per_row[s.row][s.verdict] += 1
    for row, c in sorted(per_row.items()):
        print(
            f"{row}\tapply={c['apply']}\tother={c['other']}\tundecided={c['undecided']}\tskip={c['skip']}"
        )
    if args.report:
        args.report.write_text(
            json.dumps(
                [
                    {
                        "row": s.row,
                        "path": s.path,
                        "loc": s.loc,
                        "col": s.col,
                        "old": s.old,
                        "new": s.new,
                        "verdict": s.verdict,
                        "why": s.why,
                        "line": s.line,
                        "key": list(s.key()),
                    }
                    for s in sorted(sites, key=lambda s: (s.row, s.path, str(s.loc), s.col))
                ],
                indent=1,
            )
        )
    changed = apply_edits(sites, texts, notebooks) if args.apply else 0
    print(
        f"applied={counts['apply']} skipped-definite-other={counts['other'] + counts['skip']} "
        f"undecided={counts['undecided']} files-changed={changed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
