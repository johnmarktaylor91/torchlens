# This is for validating all the different model types. Have it both validate and spit out the visual for checking.
# Let the default input size be 3x3x224x224.

from os.path import join as opj
import pytest
import torch
import torchvision
import example_models

from torchlens import log_forward_pass, show_model_graph, validate_saved_activations


@pytest.fixture
def default_input():
    return torch.rand(6, 3, 224, 224)


def test_model_simple_ff(default_input):
    model = example_models.SimpleFF()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'simple_ff'))


def test_model_inplace_funcs(default_input):
    model = example_models.InPlaceFuncs()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'inplace_funcs'))


def test_model_simple_internally_generated(default_input):
    model = example_models.SimpleInternallyGenerated()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'simple_internally_generated'))


def test_model_new_tensor_inside(default_input):
    model = example_models.NewTensorInside()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'new_tensor_inside'))


def test_model_new_tensor_from_numpy(default_input):
    model = example_models.TensorFromNumpy()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'new_tensor_from_numpy'))


def test_model_simple_random(default_input):
    model = example_models.SimpleRandom()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'simple_random'))


def test_dropout_model_real_train(default_input):
    model = example_models.DropoutModelReal()
    model.train()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'dropout_model_real_train'))


def test_dropout_model_real_eval(default_input):
    model = example_models.DropoutModelReal()
    model.eval()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'dropout_model_real_eval'))


def test_dropout_model_dummy_zero_train(default_input):
    model = example_models.DropoutModelDummyZero()
    model.train()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'dropout_model_dummyzero_train'))


def test_dropout_model_dummy_zero_eval(default_input):
    model = example_models.DropoutModelDummyZero()
    model.eval()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'dropout_model_dummyzero_eval'))


def test_batchnorm_train(default_input):
    model = example_models.BatchNormModel()
    model.train()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'batchnorm_model_train'))


def test_batchnorm_eval(default_input):
    model = example_models.BatchNormModel()
    model.eval()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'batchnorm_model_eval'))


def test_concat_tensors(default_input):
    model = example_models.ConcatTensors()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'concat_tensors_model'))


def test_split_tensor(default_input):
    model = example_models.SplitTensor()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'split_tensors_model'))


def test_identity_model(default_input):
    model = example_models.IdentityModel()
    validate_saved_activations(model, default_input)
    show_model_graph(model,
                     default_input,
                     vis_opt='unrolled',
                     vis_outpath=opj('visualization_outputs', 'identity_model'))
