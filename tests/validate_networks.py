# This is a batch script for validating the different networks for sample inputs. There will be a stack of
# sample inputs for each model.

import cornet
import os
from os.path import join as opj
import torch
import torchvision
import torchvision.transforms as transforms
import visualpriors

from torchlens.user_funcs import validate_saved_activations, get_model_activations

# TODO: make this nicer; nice interface for logging the results of this, noting the image size,
# auto-checking if it's passed yet, and so on; like a YAML or something? Maybe specify both models and model classes?

# Assemble the models and associated inputs to test.

models_and_images = []  # list of tuples (model, inputs_to_test)


def write_line_to_txt_if_new(out_path, line):
    """Write a line to a .txt file at out_path if that line is not already in the file; creates the file if it
    doesn't already exist.

    Args:
        out_path: Path to the file
        line: Line to write.

    Returns:
        Nothing, but writes the line
    """
    if not os.path.exists(out_path):
        with open(out_path, 'w') as f:
            f.write(line)
            f.write('\n')
    else:
        with open(out_path, 'r') as f:
            lines = f.readlines()
        if line not in lines:
            with open(out_path, 'a') as f:
                f.write(line)
                f.write('\n')


# Images:

image_size = (6, 3, 224, 224)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize(image_size[2:])])

cifar = torchvision.datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)

image_inputs = {'zeros': torch.zeros(image_size),
                'ones': torch.ones(image_size),
                'rand1': torch.rand(image_size),
                'cifar1': torch.stack([cifar[0][0], cifar[1][0]])}

torchvision_model_names = torchvision.models.list_models(torchvision.models)
out_dir = opj('/home/jtaylor/projects/torchlens/local_jmt/example_visuals')

models_passed = ['alexnet', 'convnext_base', 'convnext_large', 'vit_b_16', 'vit_b_32', 'maxvit_t', 'vit_l_16',
                 'vit_l_32', 'swin_b', 'swin_s', 'swint_t']

# NOTE: vit_h_14 is huge and kills the RAM. Try it later
transformer_models = ['vit_b_16', 'vit_b_32', 'maxvit_t', 'vit_l_16', 'vit_l_32',
                      'swin_b', 'swin_s', 'swint_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t']

failed_model_inputs = []
for model_name in transformer_models:
    if model_name in models_passed:
        continue
    try:
        model = getattr(torchvision.models, model_name)()
        del model
    except:
        print(f"{model_name} is not a model, skipping...")
        continue
    print(f"Testing {model_name}")
    for input_name, input_tensor in image_inputs.items():
        model = getattr(torchvision.models, model_name)()
        if model_name == 'inception_v3':
            image_size = (299, 299)
            input_tensor = transforms.Resize(image_size)(input_tensor)
        model_history = get_model_activations(model, input_tensor, vis_opt='rolled',
                                              vis_outpath=opj(out_dir, model_name))
        saved_activations_are_valid = validate_saved_activations(model, input_tensor,
                                                                 min_proportion_consequential_layers=.98,
                                                                 random_seed=None, verbose=True)
        if saved_activations_are_valid:
            msg = 'passed'
        else:
            msg = 'failed'
            failed_model_inputs.append((model_name, input_name))
        for param in model.parameters():
            del param
        del model
        print(f"\t{input_name}: {msg}")
        write_line_to_txt_if_new(opj(out_dir, 'validation_results.txt'), f"{model_name}: {input_name}: {msg}")

print(f"**********DONE**********\n These models/inputs failed: \n {failed_model_inputs}")

# Now the cornet models:

cornet_model_names = [
    'cornet_s',
    'cornet_z'
]

failed_model_inputs = []
for model_name in cornet_model_names:
    try:
        model = getattr(cornet, model_name)()
        del model
    except:
        print(f"{model_name} is not a model, skipping...")
        continue
    print(f"Testing {model_name}")
    for input_name, input_tensor in image_inputs.items():
        model = getattr(cornet, model_name)()
        model_history = get_model_activations(model, input_tensor, vis_opt='rolled',
                                              vis_outpath=opj(out_dir, model_name))
        saved_activations_are_valid = validate_saved_activations(model, input_tensor,
                                                                 min_proportion_consequential_layers=.9,
                                                                 random_seed=None, verbose=False)
        if saved_activations_are_valid:
            msg = 'passed'
        else:
            msg = 'failed'
            failed_model_inputs.append((model_name, input_name))
        for param in model.parameters():
            del param
        del model
        print(f"\t{input_name}: {msg}")

# Now the taskonomy network:

taskonomy_model = visualpriors.taskonomy_network.TaskonomyNetwork()
for input_name, input_tensor in image_inputs.items():
    model = taskonomy_model
    model_history = get_model_activations(model, input_tensor, vis_opt='rolled',
                                          vis_outpath=opj(out_dir, model_name))
    saved_activations_are_valid = validate_saved_activations(model, input_tensor,
                                                             min_proportion_consequential_layers=.9, random_seed=None,
                                                             verbose=True)
    if saved_activations_are_valid:
        msg = 'passed'
    else:
        msg = 'failed'
        failed_model_inputs.append((model_name, input_name))
    for param in model.parameters():
        del param
    del model
    print(f"\t{input_name}: {msg}")
