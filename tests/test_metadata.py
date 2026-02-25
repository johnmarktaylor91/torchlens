"""Comprehensive metadata field testing for ModelHistory and TensorLogEntry.

Uses small/fast models from example_models.py to verify that all key metadata
fields are populated correctly across different model types.
"""

import torch

import example_models
from torchlens import log_forward_pass


# =============================================================================
# TestModelHistoryFields
# =============================================================================


class TestModelHistoryFields:
    """Tests for ModelHistory-level metadata fields."""

    def test_general_info_fields(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.model_name, str)
        assert len(mh.model_name) > 0
        assert mh._pass_finished is True
        assert isinstance(mh.num_operations, int)
        assert mh.num_operations > 0

    def test_model_structure_non_recurrent(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert mh.model_is_recurrent is False
        assert mh.model_is_branching is False

    def test_model_structure_branching(self, small_input):
        model = example_models.SimpleBranching()
        mh = log_forward_pass(model, small_input)
        assert mh.model_is_branching is True

    def test_model_structure_recurrent(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        assert mh.model_is_recurrent is True

    def test_model_structure_conditional(self):
        model = example_models.ConditionalBranching()
        model_input = -torch.ones(6, 3, 224, 224)
        mh = log_forward_pass(model, model_input)
        assert mh.model_has_conditional_branching is True

    def test_layer_tracking_fields(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.layer_list, list)
        assert len(mh.layer_list) > 0
        assert isinstance(mh.layer_labels, list)
        assert len(mh.layer_labels) > 0
        assert isinstance(mh.layer_dict_main_keys, dict)
        assert isinstance(mh.layer_dict_all_keys, dict)

    def test_input_output_layers(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.input_layers, list)
        assert len(mh.input_layers) >= 1
        assert isinstance(mh.output_layers, list)
        assert len(mh.output_layers) >= 1

    def test_buffer_layers(self):
        model = example_models.BufferModel()
        model_input = torch.rand(12, 12)
        mh = log_forward_pass(model, model_input)
        assert isinstance(mh.buffer_layers, list)
        assert len(mh.buffer_layers) > 0

    def test_tensor_info_fields(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.num_tensors_total, int)
        assert mh.num_tensors_total > 0
        assert isinstance(mh.tensor_fsize_total, (int, float))
        assert mh.tensor_fsize_total > 0
        assert isinstance(mh.tensor_fsize_total_nice, str)
        assert len(mh.tensor_fsize_total_nice) > 0

    def test_param_info_fields(self, small_input):
        model = example_models.BatchNormModel()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.total_param_tensors, int)
        assert isinstance(mh.total_params, int)
        assert mh.total_params > 0

    def test_module_info_fields(self, small_input):
        model = example_models.NestedModules()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.module_addresses, list)
        assert len(mh.module_addresses) > 0
        assert isinstance(mh.module_types, dict)
        assert isinstance(mh.module_passes, list)
        assert isinstance(mh.module_children, dict)

    def test_time_fields(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.elapsed_time_total, float)
        assert mh.elapsed_time_total > 0
        assert isinstance(mh.elapsed_time_setup, float)
        assert isinstance(mh.elapsed_time_forward_pass, float)
        assert isinstance(mh.elapsed_time_cleanup, float)
        assert isinstance(mh.elapsed_time_torchlens_logging, float)

    def test_multi_input_layers(self):
        model = example_models.MultiInputs()
        inputs = [
            torch.rand(6, 3, 224, 224),
            torch.rand(6, 3, 224, 224),
            torch.rand(6, 3, 224, 224),
        ]
        mh = log_forward_pass(model, inputs)
        assert len(mh.input_layers) == 3

    def test_multi_output_layers(self, small_input):
        model = example_models.MultiOutputs()
        mh = log_forward_pass(model, small_input)
        assert len(mh.output_layers) >= 2

    def test_internally_initialized_layers(self, small_input):
        model = example_models.SimpleInternallyGenerated()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.internally_initialized_layers, list)
        assert len(mh.internally_initialized_layers) > 0

    def test_equivalent_operations(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        assert isinstance(mh.equivalent_operations, dict)
        # Recurrent model should have equivalent operations
        assert len(mh.equivalent_operations) > 0


# =============================================================================
# TestTensorLogEntryFields
# =============================================================================


class TestTensorLogEntryFields:
    """Tests for TensorLogEntry-level metadata fields."""

    def test_label_fields(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        entry = mh[0]
        assert isinstance(entry.layer_label, str)
        assert isinstance(entry.layer_type, str)
        assert isinstance(entry.pass_num, int)
        assert isinstance(entry.lookup_keys, list)
        assert len(entry.lookup_keys) > 0

    def test_input_layer_properties(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        # Find an input layer
        input_entry = None
        for label in mh.layer_labels:
            e = mh[label]
            if e.is_input_layer:
                input_entry = e
                break
        assert input_entry is not None
        assert input_entry.is_input_layer is True
        assert input_entry.is_output_layer is False
        assert input_entry.has_parents is False
        assert input_entry.has_children is True

    def test_output_layer_properties(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        # Find an output layer
        output_entry = None
        for label in mh.layer_labels:
            e = mh[label]
            if e.is_output_layer:
                output_entry = e
                break
        assert output_entry is not None
        assert output_entry.is_output_layer is True
        assert output_entry.has_children is False

    def test_tensor_info_fields(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        entry = mh[0]
        assert isinstance(entry.tensor_shape, (tuple, torch.Size))
        assert isinstance(entry.tensor_dtype, torch.dtype)
        assert isinstance(entry.tensor_fsize, (int, float))
        assert entry.tensor_contents is not None

    def test_function_call_fields(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        # Find a non-input layer (which has a function applied)
        non_input = None
        for label in mh.layer_labels:
            e = mh[label]
            if not e.is_input_layer:
                non_input = e
                break
        assert non_input is not None
        assert isinstance(non_input.func_applied_name, str)
        assert len(non_input.func_applied_name) > 0
        assert isinstance(non_input.func_time_elapsed, float)

    def test_graph_relationships(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        for label in mh.layer_labels:
            entry = mh[label]
            assert isinstance(entry.parent_layers, list)
            assert isinstance(entry.child_layers, list)
            # Verify referential integrity
            for parent_label in entry.parent_layers:
                parent = mh[parent_label]
                assert label in parent.child_layers
            for child_label in entry.child_layers:
                child = mh[child_label]
                assert label in child.parent_layers

    def test_inplace_function_flag(self, small_input):
        model = example_models.InPlaceFuncs()
        mh = log_forward_pass(model, small_input)
        # Verify the field exists and is a bool on all entries
        for label in mh.layer_labels:
            entry = mh[label]
            assert isinstance(entry.function_is_inplace, bool)
        # The inplace relu should be tracked as relu_ (trailing underscore)
        found_relu_ = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.func_applied_name and entry.func_applied_name.endswith("_"):
                found_relu_ = True
                break
        assert found_relu_, "InPlaceFuncs should have relu_ (inplace function name)"

    def test_param_fields_with_params(self, small_input):
        model = example_models.BatchNormModel()
        mh = log_forward_pass(model, small_input)
        # Find a layer computed with params (batchnorm has weight/bias)
        found = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.computed_with_params:
                assert entry.num_params_total > 0
                found = True
                break
        assert found, "BatchNormModel should have layers computed with params"

    def test_param_fields_without_params(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        # All layers in SimpleFF should have no params (just add/mul)
        for label in mh.layer_labels:
            entry = mh[label]
            assert entry.computed_with_params is False

    def test_module_fields(self, small_input):
        model = example_models.NestedModules()
        mh = log_forward_pass(model, small_input)
        # Find a layer inside a submodule
        found = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.is_computed_inside_submodule:
                assert isinstance(entry.containing_module_origin, str)
                assert isinstance(entry.module_nesting_depth, int)
                assert entry.module_nesting_depth > 0
                found = True
                break
        assert found, "NestedModules should have layers inside submodules"

    def test_buffer_layer_fields(self):
        model = example_models.BufferModel()
        model_input = torch.rand(12, 12)
        mh = log_forward_pass(model, model_input)
        # Find a buffer layer
        found = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.is_buffer_layer:
                assert isinstance(entry.buffer_address, str)
                found = True
                break
        assert found, "BufferModel should have buffer layers"

    def test_internally_initialized_fields(self, small_input):
        model = example_models.SimpleInternallyGenerated()
        mh = log_forward_pass(model, small_input)
        # Find an internally initialized layer
        found = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.initialized_inside_model:
                found = True
                break
        assert found, "SimpleInternallyGenerated should have internally init layers"

    def test_sibling_spouse_fields(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        for label in mh.layer_labels:
            entry = mh[label]
            assert isinstance(entry.sibling_layers, list)
            assert isinstance(entry.spouse_layers, list)

    def test_conditional_fields(self):
        model = example_models.ConditionalBranching()
        model_input = -torch.ones(6, 3, 224, 224)
        mh = log_forward_pass(model, model_input)
        # At least one layer should be in a conditional branch
        found = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.in_cond_branch:
                found = True
                break
        assert found, "ConditionalBranching should have layers in cond branches"

    def test_distances_with_flag(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input, mark_input_output_distances=True)
        for label in mh.layer_labels:
            entry = mh[label]
            assert entry.min_distance_from_input is not None
            assert entry.max_distance_from_input is not None
            assert entry.min_distance_from_output is not None
            assert entry.max_distance_from_output is not None
            assert isinstance(entry.min_distance_from_input, int)


# =============================================================================
# TestRecurrentMetadata
# =============================================================================


class TestRecurrentMetadata:
    """Tests for recurrent model specific metadata."""

    def test_pass_numbers(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        # Some layers should have pass_num > 1
        found_multi_pass = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.pass_num > 1:
                found_multi_pass = True
                break
        assert found_multi_pass, "Recurrent model should have layers with pass_num > 1"

    def test_layer_passes_total(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        found = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.layer_passes_total > 1:
                found = True
                break
        assert found, "Recurrent model should have layers with passes_total > 1"

    def test_same_layer_operations(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        # At least one layer should have same_layer_operations linking across passes
        found = False
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.same_layer_operations and len(entry.same_layer_operations) > 0:
                found = True
                break
        assert found, "Recurrent model should have same_layer_operations"

    def test_rolled_layer_list(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        assert isinstance(mh.layer_list_rolled, list)
        # Rolled should be shorter than unrolled for recurrent model
        assert len(mh.layer_list_rolled) < len(mh.layer_list)


# =============================================================================
# TestModelHistoryAccess
# =============================================================================


class TestModelHistoryAccess:
    """Tests for ModelHistory access patterns."""

    def test_len(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert len(mh) == len(mh.layer_labels)

    def test_getitem_by_positive_index(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        entry = mh[0]
        assert entry.layer_label == mh.layer_labels[0]

    def test_getitem_by_negative_index(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        entry = mh[-1]
        assert entry.layer_label == mh.layer_labels[-1]

    def test_getitem_by_label(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        label = mh.layer_labels[0]
        entry = mh[label]
        assert entry.layer_label == label

    def test_iter(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        items = list(mh)
        assert len(items) == len(mh.layer_labels)

    def test_layer_labels_properties(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.layer_labels, list)
        assert all(isinstance(lbl, str) for lbl in mh.layer_labels)
        assert isinstance(mh.layer_labels_no_pass, list)
        assert all(isinstance(lbl, str) for lbl in mh.layer_labels_no_pass)
        assert isinstance(mh.layer_labels_w_pass, list)
        assert all(isinstance(lbl, str) for lbl in mh.layer_labels_w_pass)


# =============================================================================
# TestFunctionArgsSaving
# =============================================================================


class TestFunctionArgsSaving:
    """Tests for function args saving feature."""

    def test_creation_args_populated(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input, save_function_args=True)
        assert mh.save_function_args is True
        # At least one non-input layer should have creation_args
        found = False
        for label in mh.layer_labels:
            entry = mh[label]
            if not entry.is_input_layer and entry.creation_args is not None:
                found = True
                break
        assert found, "save_function_args=True should populate creation_args"

    def test_creation_args_not_populated(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input)
        assert mh.save_function_args is False


# =============================================================================
# TestActivationPostfunc
# =============================================================================


class TestActivationPostfunc:
    """Tests for activation postfunc feature."""

    def test_postfunc_applied(self, small_input):
        model = example_models.SimpleFF()
        mh = log_forward_pass(model, small_input, activation_postfunc=torch.mean)
        # All saved tensors should be scalar (mean reduces to scalar)
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.tensor_contents is not None:
                assert entry.tensor_contents.dim() == 0, (
                    f"Layer {label} should be scalar after torch.mean postfunc"
                )
