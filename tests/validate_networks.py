# This is a batch script for validating the different networks for sample inputs. There will be a stack of
# sample inputs for each model.

import cornet
import torch
import torchvision
import torchvision.transforms as transforms
import visualpriors

from torchlens.user_funcs import validate_saved_activations

# Assemble the models and associated inputs to test.

models_and_images = []  # list of tuples (model, inputs_to_test)

# Images:

image_size = (6, 3, 256, 256)

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

# Assemble torchvision models:

torchvision_model_names = [
    'alexnet',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'googlenet',
    'inception_v3',
    'mnasnet0_5',
    'mnasnet0_75',
    'mnasnet1_0',
    'mnasnet1_3',
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    'resnet101',
    'resnet152',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnext101_32x8d',
    'resnext50_32x4d',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
    'squeezenet1_0',
    'squeezenet1_1',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    'wide_resnet101_2',
    'wide_resnet50_2'
]

failed_model_inputs = []
for model_name in torchvision_model_names:
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
