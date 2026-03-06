import os
from os.path import join as opj

import pytest
import torch

from torchlens import _state

# Deterministic seeding
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)

# Output directories — anchored to tests/ so they don't pollute project root
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_OUTPUTS_DIR = opj(TESTS_DIR, "test_outputs")
REPORTS_DIR = opj(TEST_OUTPUTS_DIR, "reports")
VIS_OUTPUT_DIR = opj(TEST_OUTPUTS_DIR, "visualizations")

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
    "generative-models",
    "nlp-models",
    "time-series",
    "point-cloud",
    "super-resolution",
]

os.makedirs(REPORTS_DIR, exist_ok=True)
for sub_dir in sub_dirs:
    os.makedirs(opj(VIS_OUTPUT_DIR, sub_dir), exist_ok=True)


# ---------------------------------------------------------------------------
# Coverage: auto-generate text report when pytest --cov is used
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Enable usage stats collection for ArgSpec coverage analysis."""
    _state._collect_usage_stats = True
    _state._function_call_counts.clear()
    _state._function_call_models.clear()


def pytest_collection_modifyitems(config, items):
    """Move ArgSpec coverage test to run last so it sees all accumulated stats."""
    coverage_tests = []
    other_tests = []
    for item in items:
        if "test_arg_positions" in item.nodeid:
            coverage_tests.append(item)
        else:
            other_tests.append(item)
    items[:] = other_tests + coverage_tests


def pytest_sessionfinish(session, exitstatus):
    """Write a coverage text report to test_outputs/ if coverage data exists."""
    _state._collect_usage_stats = False
    try:
        from coverage import Coverage

        cov = Coverage()
        cov.load()
        report_path = opj(REPORTS_DIR, "coverage_report.txt")
        with open(report_path, "w") as f:
            cov.report(file=f, show_missing=True, skip_empty=True)
        html_dir = opj(REPORTS_DIR, "coverage_html")
        cov.html_report(directory=html_dir, skip_empty=True)
    except Exception:
        pass  # No coverage data or coverage not installed — skip silently


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


@pytest.fixture
def seq_input():
    """(seq_len, batch, embed_dim) for transformer models."""
    return torch.rand(10, 2, 16)


@pytest.fixture
def token_input():
    """Integer tokens for embedding models."""
    return torch.randint(0, 100, (2, 10))


@pytest.fixture
def input_3d():
    """Volumetric input for Conv3d models."""
    return torch.rand(1, 1, 4, 4, 4)


@pytest.fixture
def input_1d_seq():
    """1D sequence input for Conv1d models."""
    return torch.rand(2, 3, 16)
