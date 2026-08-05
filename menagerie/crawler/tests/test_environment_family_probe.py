"""Head20 family coverage pins for the routed ``core`` environment intent.

The head20 population routes its four dominant families -- timm,
base-pytorch-compact, segmentation-models-pytorch, and transformers -- to the
``core`` intent (no ``routing._PACKAGE_INTENTS`` marker, not legacy, not an
exact repository). Rung 7 died before the environment lane could prove any of
them, and the 2026-08-04 pilot rung proved the failure mode is real: ``core``
gained ``segmentation-models-pytorch`` hours after the run's solve, the stale
residue kept winning, and every smp model died at the probes.

Two layers pin the coverage here:

* An always-on registry pin: the ``core`` declaration and probe contract must
  keep naming every family's entry distribution and import canary. This is the
  cheap tripwire for the "family library silently dropped from the intent"
  class -- it fails at spec-edit time, not ninety minutes into a rung.
* An opt-in materialized probe: with ``MENAGERIE_CORE_ENV_PREFIX`` naming an
  operator-created core prefix, one tiny random-init forward runs per family
  inside that prefix's interpreter. A prefix that lacks a family's entry
  library FAILS (never skips): the variable is an explicit claim that the
  prefix realizes the core intent, and a skip would read as green.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.package_namespace import (
    dependency_spec_name,
    inventory_row_provides_distribution,
)

CORE_PREFIX_VARIABLE = "MENAGERIE_CORE_ENV_PREFIX"

#: Head20 family -> (entry distributions, entry import module).
HEAD20_FAMILY_ENTRIES: dict[str, tuple[tuple[str, ...], str]] = {
    "timm": (("timm",), "timm"),
    "base-pytorch-compact": (("torch", "torchvision"), "torchvision"),
    "smp": (("segmentation-models-pytorch",), "segmentation_models_pytorch"),
    "transformers": (("transformers",), "transformers"),
}

_FORWARD_PROBE_SOURCE = r"""
import json
import traceback

import torch

torch.set_num_threads(2)
torch.manual_seed(0)
results = {}


def run(family, fn):
    try:
        shape = fn()
        results[family] = {"status": "STANDS", "output_shape": list(shape)}
    except Exception as exc:
        results[family] = {
            "status": "FAILS",
            "cause": f"{type(exc).__name__}: {exc}",
            "trace_tail": traceback.format_exc().splitlines()[-3:],
        }


def probe_timm():
    import timm

    model = timm.create_model("resnet18", pretrained=False).eval()
    with torch.no_grad():
        return model(torch.randn(1, 3, 64, 64)).shape


def probe_base_pytorch_compact():
    import torchvision.models as tvm

    model = tvm.resnet18(weights=None).eval()
    with torch.no_grad():
        return model(torch.randn(1, 3, 64, 64)).shape


def probe_smp():
    import segmentation_models_pytorch as smp

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None).eval()
    with torch.no_grad():
        return model(torch.randn(1, 3, 64, 64)).shape


def probe_transformers():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        return model(input_ids=torch.randint(0, 128, (1, 8))).logits.shape


run("timm", probe_timm)
run("base-pytorch-compact", probe_base_pytorch_compact)
run("smp", probe_smp)
run("transformers", probe_transformers)
print(json.dumps(results))
"""


@pytest.mark.smoke
def test_core_intent_declares_every_head20_family_entry_distribution() -> None:
    """The core declaration provides each family's entry libraries by name.

    The check runs through the same namespace bridge the coverage and refusal
    code read (``pytorch`` provides ``torch``), so it can only disagree with a
    runtime refusal when the spec really stopped declaring a family.
    """

    registry = load_environment_registry()
    declared = [
        dependency_spec_name(spec) for spec in registry.intents["core"].dependencies
    ]
    for family, (distributions, _module) in HEAD20_FAMILY_ENTRIES.items():
        for distribution in distributions:
            assert any(
                inventory_row_provides_distribution(name, distribution)
                for name in declared
                if name
            ), (
                f"head20 family {family!r} needs distribution {distribution!r}, and no "
                "declared core dependency provides it; models routed to core would "
                "defer with needs-environment-coverage"
            )


@pytest.mark.smoke
def test_core_probe_contract_imports_every_head20_family_entry_module() -> None:
    """The core probe canaries import each family's entry module."""

    registry = load_environment_registry()
    imports = set(registry.intents["core"].probes.imports)
    for family, (_distributions, module) in HEAD20_FAMILY_ENTRIES.items():
        assert module in imports, (
            f"head20 family {family!r} entry module {module!r} has no import canary "
            "in the core probe contract; a broken install would pass the probes"
        )


@pytest.mark.heavy
def test_materialized_core_prefix_runs_one_forward_per_head20_family() -> None:
    """Each head20 family imports and completes a tiny forward in the real prefix.

    Opt-in: set ``MENAGERIE_CORE_ENV_PREFIX`` to an operator-created core
    prefix. A mispointed prefix fails loudly rather than skipping, because the
    variable asserts the prefix realizes the core intent.
    """

    value = os.environ.get(CORE_PREFIX_VARIABLE)
    if not value:
        pytest.skip(f"{CORE_PREFIX_VARIABLE} is unset; materialized probe not requested")
    prefix = Path(value).resolve()
    interpreter = prefix / "bin" / "python"
    assert (prefix / "conda-meta").is_dir() and interpreter.is_file(), (
        f"{CORE_PREFIX_VARIABLE} does not name a materialized conda prefix: {prefix}"
    )
    completed = subprocess.run(
        [str(interpreter), "-c", _FORWARD_PROBE_SOURCE],
        capture_output=True,
        text=True,
        timeout=600,
        env={
            "HOME": os.environ.get("HOME", str(prefix)),
            "PATH": f"{prefix / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
        },
        check=False,
    )
    assert completed.returncode == 0, (
        f"family forward probe crashed in {prefix}:\n{completed.stderr[-2000:]}"
    )
    results = json.loads(completed.stdout.splitlines()[-1])
    assert set(results) == set(HEAD20_FAMILY_ENTRIES)
    failed = {
        family: outcome for family, outcome in results.items() if outcome["status"] != "STANDS"
    }
    assert not failed, f"head20 families failed their forwards: {json.dumps(failed, indent=1)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
