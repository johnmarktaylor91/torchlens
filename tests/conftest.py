import os
from os.path import join as opj

import pytest
import torch

# Deterministic seeding
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)

# Output directories — anchored to tests/ so they don't pollute project root
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_OUTPUTS_DIR = opj(TESTS_DIR, "test_outputs")
VIS_OUTPUT_DIR = TEST_OUTPUTS_DIR

sub_dirs = [
    "cornet",
    "graph-neural-networks",
    "language-models",
    "multimodal-models",
    "taskonomy",
    "timm",
    "torchaudio",
    "torchvision-main",
    "torchvision-detection",
    "torchvision-opticflow",
    "torchvision-segmentation",
    "torchvision-video",
    "torchvision-quantize",
    "toy-networks",
    "aesthetic_test_models",
]

os.makedirs(TEST_OUTPUTS_DIR, exist_ok=True)
for sub_dir in sub_dirs:
    os.makedirs(opj(VIS_OUTPUT_DIR, sub_dir), exist_ok=True)


# Fixtures


@pytest.fixture
def default_input1():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input2():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input3():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input4():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def zeros_input():
    return torch.zeros(6, 3, 224, 224)


@pytest.fixture
def ones_input():
    return torch.ones(6, 3, 224, 224)


@pytest.fixture
def vector_input():
    return torch.rand(5)


@pytest.fixture
def input_2d():
    return torch.rand(5, 5)


@pytest.fixture
def input_complex():
    return (torch.complex(torch.rand(3, 3), torch.rand(3, 3)),)


@pytest.fixture
def small_input():
    return torch.rand(2, 3, 32, 32)
