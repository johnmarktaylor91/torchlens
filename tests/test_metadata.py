"""Comprehensive metadata field testing for Trace and Op.

Uses small/fast models from example_models.py to verify that all key metadata
fields are populated correctly across different model types.
"""

import linecache
import pickle

import example_models
import pytest
import torch
import torch.nn as nn

import torchlens
import torchlens as tl
from torchlens import trace as trace_fn
from torchlens._capture.flops import (
    compute_forward_flops,
)
from torchlens._data_classes import FuncCallLocation

# =============================================================================
# Trace fields
# =============================================================================


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups_capture.py.
def test_general_info_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.source.class_name, str)
    assert len(mh.source.class_name) > 0
    assert mh._tracing_finished is True
    assert isinstance(mh._num_ops, int)
    assert mh._num_ops > 0


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_layer_accessors.py.
def test_model_structure_recurrent(input_2d: torch.Tensor) -> None:
    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    assert mh.graph.is_recurrent is True
    assert len(mh.layers.recurrent) > 0
    assert all(layer.num_passes > 1 for layer in mh.layers.recurrent)
    assert [layer.layer_label for layer in mh.layers.recurrent] == [
        layer.layer_label for layer in mh.layers if layer.num_passes > 1
    ]
    first_recurrent_layer = mh.layers.recurrent[0]
    assert mh.layers.recurrent[first_recurrent_layer.layer_label] is first_recurrent_layer
    assert mh.layers.recurrent[f"{first_recurrent_layer.layer_label}:1"] is first_recurrent_layer


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups.py.
def test_tensor_info_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(len(mh), int)
    assert len(mh) > 0
    assert isinstance(mh.cost.total_activation_memory, (int, float))
    assert mh.cost.total_activation_memory > 0
    assert isinstance(mh.cost.total_activation_memory, torchlens.types.Bytes)
    assert len(str(mh.cost.total_activation_memory)) > 0


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_forward_helpers.py, _data_classes/_trace_groups.py.
def test_measure_python_peak_memory_opt_in_reports_positive_peak(small_input):
    """The opt-in tracemalloc probe reads positive even for a tiny model.

    ``CaptureOptions(measure_python_peak_memory=True)`` folds the stdlib
    tracemalloc Python-allocation peak into ``forward_peak_memory`` via ``max()``,
    which stays reliably positive where the host RSS delta rounds to zero. Only
    the measurement changes: the captured graph must be identical to the default
    path.
    """

    model = example_models.SimpleFF()
    baseline = trace_fn(model, small_input)
    measured = trace_fn(
        model,
        small_input,
        capture=torchlens.options.CaptureOptions(
            measure_python_peak_memory=True, measure_host_peak_memory=True
        ),
    )
    assert measured.measure_python_peak_memory is True
    assert isinstance(measured.cost.forward_peak_memory, torchlens.types.Bytes)
    assert int(measured.cost.forward_peak_memory) > 0
    assert measured.provenance.forward_memory_backend in {"cpu", "cuda", "mps"}
    assert [op.layer_label for op in measured.ops] == [op.layer_label for op in baseline.ops]


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/module.py.
def test_module_info_fields(small_input):
    model = example_models.NestedModules()
    mh = trace_fn(model, small_input)
    # Module info is now accessed via structured Module objects
    assert len(mh.modules) > 1
    root = mh.modules["self"]
    assert root.address == "self"
    assert len(root.layers) > 0
    # Submodules should have valid addresses and class names
    for ml in mh.modules:
        if ml.address != "self":
            assert isinstance(ml.class_name, str)
            assert ml.address_parent in mh.modules


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups.py, _data_classes/trace.py.
def test_time_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.provenance.duration, float)
    assert mh.provenance.duration > 0
    assert isinstance(mh.provenance.setup_duration, float)
    assert isinstance(mh.cost.forward_duration, float)
    assert isinstance(mh.provenance.cleanup_duration, float)
    assert isinstance(mh.cost.overhead_duration, float)


# =============================================================================
# Op fields
# =============================================================================


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/layer.py.
def test_input_layer_properties(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    input_entry = None
    for label in mh.layer_labels:
        e = mh[label]
        if e.is_input:
            input_entry = e
            break
    assert input_entry is not None
    assert input_entry.is_input is True
    assert input_entry.is_output is False
    assert input_entry.has_parents is False
    assert input_entry.has_children is True


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/layer.py.
def test_output_layer_properties(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    output_entry = None
    for label in mh.layer_labels:
        e = mh[label]
        if e.is_output:
            output_entry = e
            break
    assert output_entry is not None
    assert output_entry.is_output is True
    assert output_entry.has_children is False


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_labels.py, _data_classes/_layer_views.py, _data_classes/_op_relations.py ....


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
def test_module_fields(small_input):
    model = example_models.NestedModules()
    mh = trace_fn(model, small_input)
    found = False
    for label in mh.layer_labels:
        entry = mh.layers[label]
        if entry.in_submodule:
            assert isinstance(entry.module, str)
            assert isinstance(entry.module_call_depth, int)
            assert entry.module_call_depth > 0
            found = True
            break
    assert found, "NestedModules should have layers inside submodules"


def test_layer_siblings_tolerate_orphan_relation_labels():
    """Layer aggregates mirror the op-level orphan tolerance (r3 R05-N1).

    ``Op.siblings``/``Op.co_parents`` deliberately resolve relation labels
    through the ``orphans`` fallback on ``keep_orphans=True`` traces and may
    APPEND orphan labels to their result; the Layer aggregates (``siblings``,
    ``co_parents``, ``children``, ``parents``, and the per-pass dict views)
    previously did a bare ``trace[label]`` lookup on those same labels and
    raised from public read-only properties — the ``children`` instance even
    fired INSIDE ``Op.siblings`` whenever a relation label resolved to a
    Layer, outside its own fallback. No public capture constructs the state
    today (orphans are component-disjoint by the step-3 bidirectional flood),
    so the state is injected surgically: a mainline parent's relation list
    gains an orphan child label, exactly the shape the op-level fallback
    exists for.
    """

    class _WithOrphans(nn.Module):
        def forward(self, x):
            internal = torch.ones(3)
            _a = internal * 2
            _b = internal + 3
            return x * 5

    mh = trace_fn(
        model=_WithOrphans(),
        inputs=torch.randn(2, 3),
        capture=torchlens.options.CaptureOptions(keep_orphans=True),
    )
    orphan_labels = list(mh.graph.orphans.keys())
    assert orphan_labels, "probe model must produce orphans"
    orphan_sibling = next(k for k in orphan_labels if mh.graph.orphans[k].siblings)

    mainline = next(
        mh.layers[label]
        for label in mh.layer_labels
        if not mh.layers[label].is_input and mh.layers[label].parents
    )
    parent_op = list(mh.layers[mainline.parents[0]].ops.values())[0]
    # Direct assignment is the supported mutation spelling on Op records; the
    # list normalizes to the immutable view type.
    parent_op.children = list(parent_op.children) + [orphan_sibling]

    op0 = list(mainline.ops.values())[0]
    assert orphan_sibling in op0.siblings  # op level tolerates and appends
    aggregated = mainline.siblings  # RED before the fix: uncaught lookup error
    assert mh.graph.orphans[orphan_sibling].layer_label in aggregated
    assert isinstance(mainline.co_parents, list)

    # b3 R05-N2: the 62aba742 tolerance stopped at label aggregates -- the
    # OBJECT-resolving surfaces of the same relation family crashed on the
    # fix's own output (`parent_layer.children` succeeded while
    # `parent_layer.get_children()` raised "not found"). They now resolve
    # orphan labels through the same fallback.
    parent_layer = mh.layers[mainline.parents[0]]
    layer_children = parent_layer.get_children()
    assert orphan_sibling in {getattr(c, "layer_label", None) for c in layer_children} or any(
        getattr(c, "label", None) == orphan_sibling for c in layer_children
    )
    op_children = parent_op.get_children()
    assert any(
        getattr(c, "label", "").startswith(orphan_sibling.split(":")[0]) for c in op_children
    )
    assert isinstance(parent_op.get_parents(), list)
    assert isinstance(parent_layer.get_parents(), list)


# =============================================================================
# Recurrent metadata
# =============================================================================


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
def test_recurrent_layer_ops_total(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    found = False
    for label in mh.layer_labels:
        entry = mh.layers[label]
        if entry.num_passes > 1:
            found = True
            break
    assert found, "Recurrent model should have layers with ops_total > 1"


# =============================================================================
# Trace access patterns
# =============================================================================


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_views.py, _data_classes/trace.py.
def test_exhaustive_saved_layer_count_uses_finalized_layer_list() -> None:
    """Exhaustive capture must report its saved unique-layer count after Step 11."""

    trace = trace_fn(
        nn.Sequential(nn.Linear(2, 2), nn.ReLU()),
        torch.ones(1, 2),
        save="all",
    )
    expected = len(
        {
            op.layer_label
            for op in trace._layer_list
            if op.has_saved_activation and not getattr(op, "is_orphan", False)
        }
    )

    assert len(trace.saved_layers) == expected
    assert len(trace.saved_layers) == len(trace.saved_ops)


# =============================================================================
# Function args saving
# =============================================================================


# =============================================================================
# Activation transform
# =============================================================================


# =============================================================================
# FLOPs: forward formula correctness
# =============================================================================


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/flops.py.
@pytest.mark.parametrize(
    "a_shape,b_shape,output_shape,expected",
    [
        ((3,), (3, 5), (5,), 30),
    ],
)
def test_flops_matmul_vector_operands(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    expected: int,
) -> None:
    """Matmul accounts for unit axes removed from vector-operand outputs."""

    a = torch.randn(a_shape)
    b = torch.randn(b_shape)

    result = compute_forward_flops("matmul", output_shape, [], (a, b), {})

    assert result == expected


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/flops.py.
def test_flops_none_func_name():
    result = compute_forward_flops(None, (3, 4), [], (), {})
    assert result is None


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/flops.py.
def test_flops_none_output_shape_elementwise():
    result = compute_forward_flops("add", None, [], (), {})
    assert result is None


# =============================================================================
# FLOPs: backward estimation
# =============================================================================


# =============================================================================
# FLOPs: integration tests
# =============================================================================


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups.py, _data_classes/trace.py.
def test_flops_total_positive():
    """total_flops_forward should be positive for a model with compute."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    x = torch.randn(4, 10)
    mh = trace_fn(model, x)
    assert mh.cost.total_flops_forward > 0
    assert mh.cost.total_flops_backward > 0
    assert mh.cost.total_flops == mh.cost.total_flops_forward + mh.cost.total_flops_backward


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups.py, _data_classes/trace.py, report/_trace_facts.py.
def test_flops_by_type():
    """flops_by_op_type returns a dict with expected keys."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    x = torch.randn(4, 10)
    mh = trace_fn(model, x)
    fbt = mh.cost.flops_by_op_type()
    assert isinstance(fbt, dict)
    assert len(fbt) > 0
    for _layer_type, info in fbt.items():
        assert "forward" in info
        assert "backward" in info
        assert "count" in info


def test_flops_conv_transpose_uses_input_work_basis() -> None:
    """Grouped transposed convolution counts input scatter MACs exactly."""

    model = nn.ConvTranspose2d(
        6,
        4,
        kernel_size=3,
        stride=2,
        padding=1,
        groups=2,
        bias=False,
    )
    trace = trace_fn(model, torch.randn(1, 6, 8, 8))
    operation = next(entry for entry in trace._layer_list if entry.func_name == "conv_transpose2d")

    assert operation.shape == (1, 4, 15, 15)
    assert operation.flops_forward == 2 * 6 * 8 * 8 * (4 // 2) * 3 * 3


# =============================================================================
# Module training mode tracking (issue #52)
# =============================================================================


# Kept by shrink-r4 (per-id subset rule): sole coverage of _quantities.py.
@pytest.mark.slow
def test_flops_resnet18_range():
    """ResNet-18 ~ 1.8 GFLOPs. Check we're in the right ballpark."""
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    mh = trace_fn(model, x)
    gflops = mh.cost.total_flops_forward / 1e9
    assert 1.0 < gflops < 5.0, f"ResNet-18 FLOPs = {gflops:.2f}G, expected ~1.8G"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/flops.py.
def test_flops_einsum_matmul():
    """einsum with matmul-like subscripts should compute FLOPs."""
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    output_shape = (3, 5)
    result = compute_forward_flops("einsum", output_shape, [], ("ij,jk->ik", a, b), {})
    assert result == 2 * 3 * 4 * 5  # 120


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/projectors.py, _runnable/input_sites.py.
def test_refresh_shape_change_emits_single_aggregated_warning() -> None:
    """A multi-layer shape-change refresh emits ONE aggregated warning (B8-36).

    Per-layer warnings with the label interpolated into the message defeated
    Python's warning dedup: hundreds of warnings per refresh on a real CNN, and
    ``-W error`` aborted the refresh at layer one. The refresh now aggregates to
    one warning naming the changed-layer count.
    """

    import re
    import warnings as warnings_module

    class _TwoStage(nn.Module):
        """Two linear stages so several layers change shape at once."""

        def __init__(self) -> None:
            """Initialize two chained projections."""

            super().__init__()
            self.linear1 = nn.Linear(4, 8)
            self.linear2 = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply linear -> relu -> linear."""

            return self.linear2(torch.relu(self.linear1(x)))

    model = _TwoStage()
    trace = trace_fn(model, torch.randn(2, 4))

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        trace.run(inputs=torch.randn(8, 4))

    shape_warnings = [w for w in caught if "Tensor shape changed" in str(w.message)]
    assert len(shape_warnings) == 1
    message = str(shape_warnings[0].message)
    match = re.search(r"Tensor shape changed for (\d+) layer", message)
    assert match is not None
    assert int(match.group(1)) >= 2


# =============================================================================
# FuncCallLocation tests
# =============================================================================


def _get_code_context_with_flag(small_input, save_source_context: bool):
    """Helper: return a non-input layer's code_context for either source-loading mode."""
    model = example_models.SimpleFF()
    mh = trace_fn(
        model,
        small_input,
        capture=tl.options.CaptureOptions(save_source_context=save_source_context),
    )
    for label in mh.layer_labels:
        entry = mh[label]
        if not entry.is_input:
            return entry.code_context, mh
    raise RuntimeError("No non-input layer found")


def _get_code_context(small_input):
    """Helper: run a model and return the code_context from a non-input layer."""
    return _get_code_context_with_flag(small_input, save_source_context=True)


# --- Class structure ---


# --- Content correctness ---


# --- Dunders ---


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/func_call_location.py.
def test_repr_contains_key_info(small_input):
    stack, _ = _get_code_context(small_input)
    # Find a frame with code context
    loc = None
    for entry in stack:
        if entry.code_context is not None:
            loc = entry
            break
    assert loc is not None
    r = repr(loc)
    assert loc.file in r
    assert str(loc.line_number) in r
    assert loc.func_name in r
    assert "--->" in r


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/func_call_location.py.
def test_repr_source_unavailable():
    loc = FuncCallLocation(
        file="test.py",
        line_number=1,
        func_name="test",
        func_signature=None,
        func_docstring=None,
        call_line="",
        code_context=None,
        source_context="None",
        code_context_labeled="",
        source_context_lines=0,
    )
    r = repr(loc)
    assert "source unavailable" in r


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/func_call_location.py.
def test_func_call_location_no_source_state_with_save_code_context_off(small_input, monkeypatch):
    """Without save_source_context, frames store no text and read it lazily on access."""
    stack, mh = _get_code_context_with_flag(small_input, save_source_context=False)
    assert isinstance(stack, list)
    assert len(stack) > 0
    captured_entries = [
        entry for entry in mh._layer_list if not entry.is_input and entry.func_name != "none"
    ]
    assert all(len(entry.code_context) > 0 for entry in captured_entries)
    for entry in captured_entries:
        for frame in entry.code_context:
            assert frame.file is not None
            assert frame.line_number is not None
            assert frame.code_firstlineno is not None
            assert frame.source_loading_enabled is False
            # Stored state: no text kept at capture.
            assert frame._code_context is None
            assert frame._call_line == ""

    loc = stack[-1]
    assert loc._source_loaded is True
    assert loc._frame_func_obj is None
    assert loc.func_signature is None
    assert loc.func_docstring is None

    accessed_files = []
    original_getlines = linecache.getlines

    def _tracking_getlines(*args, **kwargs):
        accessed_files.append(args[0])
        return original_getlines(*args, **kwargs)

    with monkeypatch.context() as local_patch:
        local_patch.setattr(linecache, "getlines", _tracking_getlines)
        expected = original_getlines(loc.file)[loc.line_number - 1].strip()
        assert loc.call_line == expected != ""
        assert loc.code_context is not None
        assert len(loc) == len(loc.code_context) == loc.num_context_lines == 15
        assert loc[0] == loc.code_context[0]
        assert "  --->  " in loc.code_context_labeled
        assert "code: source unavailable" not in repr(loc)
        assert loc.file in accessed_files

    # Never persisted: the stored no-source state is what a pickle carries.
    mh_roundtrip = pickle.loads(pickle.dumps(mh))
    roundtrip_stack = next(
        entry.code_context for entry in mh_roundtrip._layer_list if not entry.is_input
    )
    assert len(roundtrip_stack) > 0
    assert roundtrip_stack[0].source_loading_enabled is False
    assert roundtrip_stack[0]._source_context == "None"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/func_call_location.py.
def test_getitem_returns_context_line(small_input):
    stack, _ = _get_code_context(small_input)
    loc = None
    for entry in stack:
        if entry.code_context is not None and len(entry.code_context) > 0:
            loc = entry
            break
    assert loc is not None
    assert loc[0] == loc.code_context[0]


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/func_call_location.py.
def test_len_matches_code_context(small_input):
    stack, _ = _get_code_context(small_input)
    for loc in stack:
        if loc.code_context is not None:
            assert len(loc) == len(loc.code_context) == loc.num_context_lines


# --- Context lines parameter ---


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups_capture.py.
def test_num_context_lines_stored_on_trace(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input, capture=tl.options.CaptureOptions(source_context_lines=5))
    assert mh.options.source_context_lines == 5


# --- Labeled code context ---


# =============================================================================
# Meta-validation: corrupt outs and verify detection
# =============================================================================


@pytest.fixture
def valid_mh_and_ground_truth():
    """Return a valid (Trace, ground_truth_tensors) pair for SimpleFF."""
    model = example_models.SimpleFF()
    x = torch.rand(2, 3, 32, 32)
    mh = trace_fn(
        model,
        x,
        capture=tl.options.CaptureOptions(save_args=True),
        save="all",
    )
    ground_truth = [mh[label].out.clone() for label in mh.output_layers]
    return mh, ground_truth


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/core.py.
def test_wrong_shape_outs(valid_mh_and_ground_truth):
    """Replacing out with a wrong-shaped tensor should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    output_label = mh.output_layers[0]
    mh.layers[output_label].ops[0].out = torch.randn(1, 1)
    assert mh.validate_forward_pass(ground_truth) is False


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_corrupt_saved_args(valid_mh_and_ground_truth):
    """Modifying saved function arguments (saved_args) should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    # Find a non-input layer that has saved_args with tensors
    for label in mh.layer_labels:
        entry = mh.layers[label].ops[0]
        if entry.is_input:
            continue
        if entry.saved_args and any(isinstance(a, torch.Tensor) for a in entry.saved_args):
            for i, arg in enumerate(entry.saved_args):
                if isinstance(arg, torch.Tensor):
                    corrupted_args = list(entry.saved_args)
                    corrupted_args[i] = torch.randn_like(arg)
                    entry.saved_args = tuple(corrupted_args)
                    assert mh.validate_forward_pass(ground_truth) is False
                    return
    pytest.fail(
        "No layer with tensor saved_args found. The fixture traces with "
        "save_arg_values enabled on a model with tensor-consuming ops, so an "
        "empty saved_args surface is a capture regression, not an environment "
        "limit (hardened from a silent fall-through skip, R79 skip audit "
        "2026-08-15)."
    )


# =============================================================================
# Conditional Branch Detection (Bug #88 fix + THEN labeling)
# =============================================================================


class TestConditionalBranchDetection:
    """Tests for conditional branch detection: backward-only IF flood + THEN detection."""

    # --- Shared helpers ---

    @staticmethod
    def _log(model, x, save_source_context=False):
        return trace_fn(
            model, x, capture=tl.options.CaptureOptions(save_source_context=save_source_context)
        )

    @staticmethod
    def _cond_input():
        """Negative input so ConditionalBranching takes the else branch."""
        return -torch.ones(2, 3, 32, 32)

    @staticmethod
    def _pos_input():
        """Positive input so ConditionalBranching takes the if branch."""
        return torch.ones(2, 3, 32, 32)

    # --- Core detection tests ---

    # Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/layer.py.
    def test_if_branch_detected(self):
        """ConditionalBranching has is_in_conditional_body=True nodes."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._cond_input())
        found = any(mh[label].is_in_conditional_body for label in mh.layer_labels)
        assert found, "Should detect conditional branch nodes"

    # --- False positive tests (Bug #88) ---

    # --- Edge/model log tests ---

    # --- Post-validation tests ---

    # --- Complex scenario tests ---

    # Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/control_flow.py.
    def test_nested_conditions(self):
        """ConditionalNested has conditional branching at both levels."""
        model = example_models.ConditionalNested()
        x = torch.rand(2, 3, 32, 32)
        mh = self._log(model, x)
        assert mh.graph.has_conditional_branching is True

    # --- Visualization integration tests ---

    # Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_render_leaf.py.
    @pytest.mark.parametrize(
        ("stem", "view", "message"),
        [("cond_rolled_test", "rolled", "Rolled graph should name its IF test")],
        ids=["rolled_graph_conditional_edges"],
    )
    def test_if_label_in_visualization(self, stem, view, message):
        """Rendered graph names the IF test node; rolled view keeps the THEN arm labels."""
        import os
        import tempfile

        from torchlens._visualization import show_model_graph

        model = example_models.ConditionalBranching()
        x = self._cond_input()
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, stem)
            show_model_graph(
                model,
                x,
                rolled=(view == "rolled"),
                visualization=tl.options.VisualizationOptions(
                    container_path=outpath, file_format="dot"
                ),
            )
            dot_file = outpath + ".dot"
            if os.path.exists(dot_file):
                with open(dot_file) as f:
                    dot_content = f.read()
                assert "IF" in dot_content, message

    # --- Rolled graph tests ---

    # --- Fallback tests (without source context) ---
