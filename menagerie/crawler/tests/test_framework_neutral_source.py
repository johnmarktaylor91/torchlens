"""Round-4 framework-neutral implementation-source regression coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.proposal import ProposalValidationError, validate_author_proposal
from menagerie.crawler.tests.test_sandbox_award_and_source_inventory import _add_archive_source
from menagerie.crawler.tests.test_slice_d_proposal_author import _ground_proposal, _make_r4


def _adapter_code() -> str:
    """Return a minimal typed R4 adapter used by proposal fixtures.

    Returns
    -------
    str
        Complete staged adapter source.
    """

    return (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> "
        "tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )


@pytest.mark.parametrize(
    "source_code",
    [
        (
            "import flax.linen as nn\n"
            "import jax\n"
            "import jax.numpy as jnp\n\n"
            "class ExampleNetArchitecture(nn.Module):\n"
            "    @nn.compact\n"
            "    def __call__(self, value):\n"
            "        scanned, _ = jax.lax.scan(custom_step, value, value)\n"
            "        return jnp.einsum('...d,df->...f', scanned, custom_weights())\n"
        ),
        (
            "import paddle\n\n"
            "class ExampleNetArchitecture(CustomPaddleBase):\n"
            "    def forward(self, value):\n"
            "        mixed = custom_paddle_stage(value)\n"
            "        return paddle.add(mixed, value)\n"
        ),
        (
            "import torch\n\n"
            "def example_net_architecture(value, weights):\n"
            "    mixed = custom_channel_mix(value, weights)\n"
            "    return torch.einsum('bcd,ce->bed', mixed, weights)\n"
        ),
    ],
    ids=("jax-flax", "paddle", "custom-functional-pytorch"),
)
def test_framework_neutral_implementation_bytes_refuse_r4(tmp_path: Path, source_code: str) -> None:
    """JAX/Flax, Paddle, and custom-functional model sources all block R4.

    Parameters
    ----------
    tmp_path:
        Isolated model and CAS directory.
    source_code:
        Exact framework-specific upstream implementation bytes.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    _add_archive_source(
        manifest,
        tmp_path / "implementation.zip",
        {"upstream/src/example_net.py": source_code},
    )

    with pytest.raises(ProposalValidationError, match="source code is available"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


def test_irrelevant_code_archive_still_permits_r4(tmp_path: Path) -> None:
    """Metrics and plotting files do not masquerade as model implementations."""

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    _add_archive_source(
        manifest,
        tmp_path / "paper-materials.zip",
        {
            "supplement/metrics.py": (
                "def example_net_accuracy(expected, observed):\n"
                "    return (expected == observed).mean()\n"
            ),
            "supplement/plotting.c": "void plot_example_net_metrics(void) { return; }\n",
        },
    )

    report = validate_author_proposal(
        proposal,
        allowed_model_dir=tmp_path,
        source_manifest=manifest,
    )

    assert report.rung.value == "R4_REIMPLEMENT"


def test_split_registry_to_model_symbol_refuses_r4(tmp_path: Path) -> None:
    """Identity-bearing config linked to a separate executable model blocks R4."""

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    _add_archive_source(
        manifest,
        tmp_path / "split-implementation.zip",
        {
            "configs/model.py": (
                "from src.net import Net\n\nMODEL_REGISTRY = {'ExampleNet': Net}\n"
            ),
            "src/net.py": (
                "class Net:\n"
                "    def forward(self, value):\n"
                "        hidden = self.encoder(value)\n"
                "        return self.decoder(hidden)\n"
            ),
        },
    )

    with pytest.raises(ProposalValidationError, match="source code is available"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


def test_unrelated_linked_generic_helper_does_not_block_r4(tmp_path: Path) -> None:
    """A generic forward helper with only a prose identity mention is not an implementation."""

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    _add_archive_source(
        manifest,
        tmp_path / "generic-helper.zip",
        {
            "utils/helper.py": (
                "# Used by the ExampleNet documentation build.\n"
                "class Helper:\n"
                "    def forward(self, value):\n"
                "        return normalize(value)\n"
            )
        },
    )

    report = validate_author_proposal(
        proposal,
        allowed_model_dir=tmp_path,
        source_manifest=manifest,
    )
    assert report.rung.value == "R4_REIMPLEMENT"


def test_large_notebook_is_streamed_and_structurally_inspected(tmp_path: Path) -> None:
    """A model notebook above the former 8 MiB cap still blocks source-free R4."""

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["x" * (8 * 1024**2 + 1)]},
            {
                "cell_type": "code",
                "source": [
                    "class ExampleNet:\n",
                    "    def forward(self, value):\n",
                    "        hidden = self.encoder(value)\n",
                    "        return self.decoder(hidden)\n",
                ],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _add_archive_source(
        manifest,
        tmp_path / "large-notebook.zip",
        {"notebooks/example_net.ipynb": json.dumps(notebook)},
    )

    with pytest.raises(ProposalValidationError, match="source code is available"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


@pytest.mark.parametrize("missing_proof", ["negative-attempt", "bounded-report"])
def test_r4_requires_explicit_bounded_negative_proof(tmp_path: Path, missing_proof: str) -> None:
    """R4 fails unless higher-rung absence and a bounded search are explicit.

    Parameters
    ----------
    tmp_path:
        Isolated model and CAS directory.
    missing_proof:
        Negative-proof component removed from the otherwise valid fixture.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    resolution = proposal["proposed_facts"]["source_resolution"]
    if missing_proof == "negative-attempt":
        resolution["attempted_rungs"][1]["result"] = "not-reached"
    else:
        resolution["search_report"]["queries"] = []

    with pytest.raises(ProposalValidationError, match="explicit negative proof|bounded search"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )
