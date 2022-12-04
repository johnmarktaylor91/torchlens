# <img src="resources/logo.png" width=15% height=15%> Torchlens

Torchlens is a package for doing exactly two things:

1) Easily extracting the activations from every single intermediate operation in a PyTorch model—no
   modifications needed—in one line of code. "Every" means every.
2) Understanding the model's computational structure via an intuitive automatic visualization and extensive metadata
   about the network's computational graph.

```python
model_history = tl.get_model_activations(simple_recurrent, x,
                                         which_layers='all',
                                         vis_opt='rolled')
print(model_history['linear_1_1:2'].tensor_contents)

'''
tensor([[-0.0690, -1.3957, -0.3231, -0.1980,  0.7197],
        [-0.1083, -1.5051, -0.2570, -0.2024,  0.8248],
        [ 0.1031, -1.4315, -0.5999, -0.4017,  0.7580],
        [-0.0396, -1.3813, -0.3523, -0.2008,  0.6654],
        [ 0.0980, -1.4073, -0.5934, -0.3866,  0.7371],
        [-0.1106, -1.2909, -0.3393, -0.2439,  0.7345]])
'''
```

<img src="resources/simple_recurrent.png" width=30% height=30%>

## Installation

To install Torchlens, first install graphviz if you haven't already (required to generate the network visualizations),
and then install Torchlens from GitHub using pip:

```bash
sudo apt install graphviz
pip install git+https://github.com/johnmarktaylor91/torchlens
```

Torchlens is compatible with versions 1.8.0+ of PyTorch.

## How-To Guide

Below is a quick demo of how to use it; for an interactive demonstration, see
the [CoLab walkthrough](https://colab.research.google.com/drive/1ORJLGZPifvdsVPFqq1LYT3t5hV560SoW?usp=sharing).

The main function of Torchlens is **get_model_activations**: when called on a model and input, it runs a
forward pass on the model and returns a ModelHistory object containing the intermediate layer activations and
accompanying metadata, along with a visual representation of every operation that occurred during the forward pass:

```python
import torch
import torchvision
import torchlens as tl

alexnet = torchvision.models.alexnet()
x = torch.rand(1, 3, 224, 224)
model_history = tl.get_model_activations(alexnet, x, which_layers='all', vis_opt='unrolled')
print(model_history)

'''
Log of AlexNet forward pass:
	Model structure: purely feedforward, without branching; 23 total modules.
	24 tensors (4.8 MB) computed in forward pass; 24 tensors (4.8 MB) saved.
	16 parameter operations (61100840 params total; 248.7 MB).
	Random seed: 3210097511
	Time elapsed: 0.288s
	Module Hierarchy:
		features:
		    features.0, features.1, features.2, features.3, features.4, features.5, features.6, features.7, 
		    features.8, features.9, features.10, features.11, features.12
		avgpool
		classifier:
		    classifier.0, classifier.1, classifier.2, classifier.3, classifier.4, classifier.5, classifier.6
	Layers:
		0: input_1_0 
		1: conv2d_1_1 
		2: relu_1_2 
		3: maxpool2d_1_3 
		4: conv2d_2_4 
		5: relu_2_5 
		6: maxpool2d_2_6 
		7: conv2d_3_7 
		8: relu_3_8 
		9: conv2d_4_9 
		10: relu_4_10 
		11: conv2d_5_11 
		12: relu_5_12 
		13: maxpool2d_3_13 
		14: adaptiveavgpool2d_1_14 
		15: flatten_1_15 
		16: dropout_1_16 
		17: linear_1_17 
		18: relu_6_18 
		19: dropout_2_19 
		20: linear_2_20 
		21: relu_7_21 
		22: linear_3_22 
		23: output_1_23 
'''
```

<img src="resources/alexnet.png" width=30% height=30%>

You can pull out information about a given layer, including its activations and helpful metadata, by indexing
the ModelHistory object in any of these equivalent ways:

1) the name of a layer (with the convention that 'conv2d_3_7' is the 3rd convolutional layer, and the 7th layer overall)
2) the name of a module (e.g., 'features' or 'classifier.3') for which that layer is an output, or
3) the ordinal position of the layer (e.g., 2 for the 2nd layer, -5 for the fifth-to-last; inputs and outputs count as
   layers here).

To quickly figure out these names, you can look at the graph visualization, or at the output of printing the
ModelHistory object (both shown above). Here are some examples of how to pull out information about a
particular layer, and also how to pull out the actual activations from that layer:

```python
print(model_history['conv2d_3_7'])  # pulling out layer by its name 
# The following commented lines pull out the same layer:
# model_history['conv2d_3_7:1'] colon indicates the pass of a layer (here just one)
# model_history['features.6'] can grab a layer by the module for which it is an output
# model_history[7] the 7th layer overall
# model_history[-17] the 17th-to-last layer
'''
Layer conv2d_3_7, operation 8/24:
	Output tensor: shape=(1, 384, 13, 13), dype=torch.float32, size=253.5 KB
		tensor([[ 0.0503, -0.1089, -0.1210, -0.1034, -0.1254],
        [ 0.0789, -0.0752, -0.0581, -0.0372, -0.0181],
        [ 0.0949, -0.0780, -0.0401, -0.0209, -0.0095],
        [ 0.0929, -0.0353, -0.0220, -0.0324, -0.0295],
        [ 0.1100, -0.0337, -0.0330, -0.0479, -0.0235]])...
	Params: Computed from params with shape (384,), (384, 192, 3, 3); 663936 params total (2.5 MB)
	Parent Layers: maxpool2d_2_6
	Child Layers: relu_3_8
	Function: conv2d (gradfunc=ConvolutionBackward0) 
	Computed inside module: features.6
	Time elapsed:  5.670E-04s
	Output of modules: features.6
	Output of bottom-level module: features.6
	Lookup keys: -17, 7, conv2d_3_7, conv2d_3_7:1, features.6, features.6:1
'''

# You can pull out the actual output activations from a layer with the tensor_contents field: 
print(model_history['conv2d_3_7'].tensor_contents)

'''
tensor([[[[-0.0867, -0.0787, -0.0817,  ..., -0.0820, -0.0655, -0.0195],
          [-0.1213, -0.1130, -0.1386,  ..., -0.1331, -0.1118, -0.0520],
          [-0.0959, -0.0973, -0.1078,  ..., -0.1103, -0.1091, -0.0760],
          ...,
          [-0.0906, -0.1146, -0.1308,  ..., -0.1076, -0.1129, -0.0689],
          [-0.1017, -0.1256, -0.1100,  ..., -0.1160, -0.1035, -0.0801],
          [-0.1006, -0.0941, -0.1204,  ..., -0.1146, -0.1065, -0.0631]]...
'''
```

If you do not wish to save the activations for all layers (e.g., to save memory), you can specify which layers to save
with the
which_layers argument when calling **get_model_activations**; you can either indicate layers in the same way
as indexing them above, or by passing in a desired substring for filtering the layers (e.g., 'conv'
will pull out all conv layers):

```python
# Pull out conv2d_3_7, the output of the 'features' module, the fifth-to-last layer, and all linear (i.e., fc) layers:
model_history = tl.get_model_activations(alexnet, x, vis_opt='unrolled',
                                         which_layers=['conv2d_3_7', 'features', -5, 'linear'])
print(model_history.layer_labels)
'''
['conv2d_3_7', 'maxpool2d_3_13', 'linear_1_17', 'dropout_2_19', 'linear_2_20', 'linear_3_22']
'''
```

The main function of torchlens is **get_model_activations**; the remaining functions are:

1) **get_model_structure**, to retrieve all model metadata without saving any activations (e.g., to figure out which
   layers
   you wish to save)
2) **show_model_graph**, which visualizes the model graph without saving any activations,
3) **validate_model_activations**, which runs a procedure to check that the activations are correct: specifically,
   it runs a forward pass and saves all intermediate activations, re-runs the forward pass from each intermediate
   layer, and checks that the resulting output matches the ground-truth output. It also checks that swapping in
   random nonsense activations instead of the saved activations generates the wrong output. **If this function ever
   returns False (i.e., the saved activations are wrong), please contact me via email (johnmarkedwardtaylor@gmail.com)
   or on this GitHub page with a description of the
   problem, and I will update torchlens to fix the problem.**

And that's it. Torchlens remains in active development, and the goal is for it to work with any PyTorch model
whatosever without exception. As of the time of this writing, it has not been tested with transformers (coming soon),
but reliably works with typical feedforward and recurrent neural networks.

## Acknowledgments

The development of Torchlens benefitted greatly from discussions with Nikolaus Kriegeskorte, George Alvarez,
Alfredo Canziani, Tal Golan, and the Visual Inference Lab at Columbia University. All network
visualizations were created with graphviz; logo created with DALL-E 2.

## Citing Torchlens

The plan is for Torchlens to be described in a preprint shortly and later as a published journal article; as these
become available, Torchlens can be cited by citing them.

## Contact

As Torchlens is still in active development, I would love your feedback. Please contact johnmarkedwardtaylor@gmail.com,
or post on the issues or discussion page for this GitHub repository, if you have any questions, comments, or
suggestions.
