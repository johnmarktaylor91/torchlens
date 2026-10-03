"""Skip-surface audit (R79): make every skip in tests/ honest and auditable.

Three enforcement layers:

1. **importorskip ledger.** Every ``pytest.importorskip`` target in tests/ must
   appear in ``IMPORTORSKIP_LEDGER`` with an availability tier, so a skip is
   always attributable to a *known* class of absence:

   - ``"test-extra"``: the module ships with the declared ``[test]`` extra (or
     is a core torchlens dependency), so a full dev/CI install HAS it and a
     skip indicates install breakage, never a legitimate environment.
   - ``"optional-preview"``: the module belongs to a backend/bridge/appliance
     extra (``jax``, ``mlx``, ``tf``, ``tabular``, ``notebook``, ...) and is
     legitimately absent outside that extra's environment.
   - ``"unavailable-ok"``: no declared extra covers it; the ledger note says
     why the absence is acceptable (torch build probes, unreleased deps,
     research-model deps, py-version backports).

   A new importorskip target must be ledgered consciously (the inventory test
   fails otherwise -- red-capable by construction), and when the environment
   claims the full ``[test]`` extra every ``"test-extra"`` target must resolve.

2. **Unconditional-skip ledger.** An AST scan flags every ``pytest.skip`` call
   with no conditional ancestor, every bare ``pytest.mark.skip`` decorator, and
   every ``pytestmark`` carrying a bare skip. Each hit must be ledgered in
   ``UNCONDITIONAL_SKIP_LEDGER`` with a dated justification: no silent dead
   tests.

3. **``python -O`` leg self-verification.** The ``requires_assertions`` marker
   exists so assert-based audits skip (not silently pass) under ``-O``; no CI
   workflow runs that leg, so this file proves the contract in-suite with one
   targeted subprocess: marked tests SKIP under ``-O`` while a plain sentinel
   test executes.

Both scanners prove red-capability against planted offenders in ``tmp_path``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

# NOTE: no module-level smoke pytestmark -- the -O subprocess test below is
# `slow`, and the tier markers are additive/disjoint (tests/test_marker_lint.py).
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

TEST_EXTRA = "test-extra"
OPTIONAL_PREVIEW = "optional-preview"
UNAVAILABLE_OK = "unavailable-ok"
VALID_TIERS = frozenset({TEST_EXTRA, OPTIONAL_PREVIEW, UNAVAILABLE_OK})

# Importing this sentinel is the environment's claim to carry the full [test]
# extra; a partial install then can no longer masquerade as full coverage.
# MUST be a [test]-extra-only package: the latest-torch-canary leg installs
# "transformers timm" directly (unpinned, alongside the [test] extra) without
# ever installing the rest of [test], so "timm" (the prior sentinel) falsely
# claimed full coverage there (round-2 CI triage, 2026-10-01). torch_geometric
# is declared only inside [test] and is not installed standalone by any
# workflow or job script.
FULL_TEST_EXTRA_SENTINEL = "torch_geometric"

# Every pytest.importorskip target in tests/ -> (tier, why that tier).
# Keep sorted; the inventory test enforces exact set equality.
IMPORTORSKIP_LEDGER: dict[str, tuple[str, str]] = {
    "IPython": (OPTIONAL_PREVIEW, "notebook extra"),
    "treescope": (
        OPTIONAL_PREVIEW,
        "treescope bridge extra (F16); absent-treescope is itself a supported "
        "environment cell, so the bridge tests importorskip while the "
        "cards/report tests run everywhere",
    ),
    "treescope._internal.arrayviz_impl": (
        OPTIONAL_PREVIEW,
        "port-equivalence leg (F16): compares the truncation port against the "
        "installed upstream implementation cell-for-cell",
    ),
    "treescope.external.torch_support": (
        OPTIONAL_PREVIEW,
        "port-equivalence leg (F16): the slice-then-convert adapter comparison",
    ),
    "PIL": (TEST_EXTRA, "pillow (also a core torchlens dependency)"),
    "brainscore_core": (OPTIONAL_PREVIEW, "neuro extra (brain-score dist)"),
    "brainscore_vision": (
        OPTIONAL_PREVIEW,
        "neuro extra; the ActivationsExtractorHelper adapter contract. The wheel "
        "requires Python >= 3.11 and this venv is 3.10, so the interface is pinned "
        "against the real 2.3.22 wheel SOURCE and the live-install test is "
        "importorskip-gated until a 3.11 env exists (2026-08-19)",
    ),
    "bitsandbytes": (
        OPTIONAL_PREVIEW,
        "deploy extra (requested in the packaging-request ledger, F37 "
        "2026-08-31): 8/4-bit quantization dep for the deployment-envelope "
        "suite (test_deploy_env_quantized.py); executes on the D02 "
        "GPU-cluster C-DEPLOY leg (MEMO section 9), where the quantization "
        "deps are provisioned",
    ),
    "cairosvg": (
        UNAVAILABLE_OK,
        "undeclared SVG-render inspection helper; extras-gap candidate reported 2026-08-15",
    ),
    "captum.attr": (OPTIONAL_PREVIEW, "captum extra"),
    "cornet": (
        UNAVAILABLE_OK,
        "CORnet research package, installed from GitHub, no maintained PyPI dist",
    ),
    "dacite": (
        UNAVAILABLE_OK,
        "model-explorer export-bridge demo dependency, no declared extra",
    ),
    "dagua": (UNAVAILABLE_OK, "unreleased in-development layout engine"),
    "e3nn.o3": (
        UNAVAILABLE_OK,
        "research-model dependency for real-world coverage, deliberately undeclared",
    ),
    "equinox": (OPTIONAL_PREVIEW, "jax extra"),
    "scipy": (
        OPTIONAL_PREVIEW,
        "extraction-export extra (MAT exporter); requested in "
        "the packaging-request ledger (F18, extract memo D15)",
    ),
    "fvcore.nn": (
        UNAVAILABLE_OK,
        "supplementary external FLOP-counter cross-oracle (A07 numbers truth); "
        "in no extra -- the closed-form and gpt2/bert pins are the blocking "
        "oracles, fvcore corroborates when present",
    ),
    "fitz": (
        UNAVAILABLE_OK,
        "PyMuPDF PDF-render inspection helper, undeclared; extras-gap candidate reported 2026-08-15",
    ),
    "flax.nnx": (OPTIONAL_PREVIEW, "jax extra"),
    "git": (
        UNAVAILABLE_OK,
        "gitpython, a python-semantic-release dependency; release-environment-only "
        "(installed from the hash-locked release-requirements.txt by the lint "
        "release-defenses job)",
    ),
    "google.protobuf.json_format": (
        UNAVAILABLE_OK,
        "protobuf JSON parser for the netron-export acceptance contract; ships "
        "with the undeclared onnx contract dependency",
    ),
    "graphviz": (TEST_EXTRA, "core torchlens dependency (dist 'graphviz')"),
    "jax": (OPTIONAL_PREVIEW, "jax extra"),
    "jax.experimental.shard_map": (OPTIONAL_PREVIEW, "jax extra"),
    "jax.lax": (OPTIONAL_PREVIEW, "jax extra"),
    "jax.numpy": (OPTIONAL_PREVIEW, "jax extra"),
    "jax.random": (OPTIONAL_PREVIEW, "jax extra"),
    "jupyter_client": (OPTIONAL_PREVIEW, "notebook extra"),
    "jsonschema": (
        UNAVAILABLE_OK,
        "dev-tooling dependency (pre-commit config validation); the agent schema "
        "lockstep's full Draft-2020-12 validation leg runs where it is installed, "
        "the dep-free structural checks run everywhere (F29)",
    ),
    "keras": (OPTIONAL_PREVIEW, "tf extra (ships with tensorflow>=2.16)"),
    "lightning": (TEST_EXTRA, "lightning"),
    "lit_nlp.api.model": (
        OPTIONAL_PREVIEW,
        "lit extra (lit-nlp>=1.3,<1.4); gates the real-dependency LIT bridge suite "
        "run by the nightly real-LIT leg (F31 packaging request)",
    ),
    "matplotlib": (
        UNAVAILABLE_OK,
        "undeclared viz-test dependency; extras-gap candidate reported 2026-08-15",
    ),
    "matplotlib.pyplot": (
        UNAVAILABLE_OK,
        "undeclared viz-test dependency; extras-gap candidate reported 2026-08-15",
    ),
    "mcp": (OPTIONAL_PREVIEW, "mcp extra (mcp>=2.0); gates the stdio server only"),
    "mcp.server": (OPTIONAL_PREVIEW, "mcp extra (mcp>=2.0); gates the stdio server only"),
    "mlx": (OPTIONAL_PREVIEW, "mlx extra"),
    "mlx.core": (OPTIONAL_PREVIEW, "mlx extra"),
    "mlx.nn": (OPTIONAL_PREVIEW, "mlx extra"),
    "model_explorer": (
        UNAVAILABLE_OK,
        "export-bridge target with no declared extra; extras-gap candidate reported 2026-08-15",
    ),
    "netron": (
        UNAVAILABLE_OK,
        "netron viewer package: the serve/widget one-liner and the "
        "vendor-execution parser harness (tests/test_netron_export_vendor.py); "
        "user extra netron>=9.2,<10 + CI pin netron==9.2.2 requested via "
        "the packaging-request ledger (F14 2026-08-28)",
    ),
    "onnx": (
        UNAVAILABLE_OK,
        "netron-export acceptance-contract dependency (strict ModelProto JSON "
        "parse + check_model); test-extra declaration requested via "
        "the packaging-request ledger (F14 2026-08-28)",
    ),
    "paddle": (OPTIONAL_PREVIEW, "paddle extra (paddlepaddle dist)"),
    "playwright.sync_api": (
        UNAVAILABLE_OK,
        "headless-Chromium driver for the netron T4 browser smoke "
        "(tests/test_netron_export_browser.py); CI installs it only in the "
        "path-filtered export job requested via the packaging-request ledger "
        "(F14 2026-08-28)",
    ),
    "pandas": (OPTIONAL_PREVIEW, "tabular extra"),
    "pandas.api.types": (OPTIONAL_PREVIEW, "tabular extra"),
    "peft": (
        OPTIONAL_PREVIEW,
        "deploy extra (requested in the packaging-request ledger, F37 "
        "2026-08-31): LoRA/adapter dep for the deployment-envelope suite "
        "(test_deploy_env_peft.py); executes on the D02 GPU-cluster "
        "C-DEPLOY leg (MEMO section 9), where the adapter deps are "
        "provisioned",
    ),
    "pennylane": (
        UNAVAILABLE_OK,
        "quantum-ML research-model dependency, deliberately undeclared",
    ),
    "pyarrow": (OPTIONAL_PREVIEW, "tabular extra"),
    "pyarrow.parquet": (OPTIONAL_PREVIEW, "tabular extra"),
    "pydot": (TEST_EXTRA, "pydot"),
    "pyg_lib": (
        UNAVAILABLE_OK,
        "torch_geometric's optional compiled extension (DimeNet's radius_graph); "
        "deliberately undeclared, no pure-torch fallback",
    ),
    "pytorch_lightning": (TEST_EXTRA, "ships inside the 'lightning' distribution"),
    "rsatoolbox": (OPTIONAL_PREVIEW, "neuro extra"),
    "sae_lens": (OPTIONAL_PREVIEW, "sae extra"),
    "semantic_release": (
        UNAVAILABLE_OK,
        "python-semantic-release, deliberately in no dev extra; release-environment-"
        "only (installed from the hash-locked release-requirements.txt by the lint "
        "release-defenses job, which executes tests/test_no_major_parser.py)",
    ),
    "clearml": (
        UNAVAILABLE_OK,
        "T-RELAY-C fidelity-pin target; relay-test extra requested in "
        "the packaging-request ledger (F26 2026-08-28)",
    ),
    "sentence_transformers": (OPTIONAL_PREVIEW, "compat-shims extra"),
    "tensorboard": (
        UNAVAILABLE_OK,
        "export-bridge target with no declared extra",
    ),
    "tensorflow": (OPTIONAL_PREVIEW, "tf extra"),
    "timm": (TEST_EXTRA, "timm"),
    "tinygrad": (OPTIONAL_PREVIEW, "tinygrad extra (py>=3.11 only)"),
    "tinygrad.nn": (OPTIONAL_PREVIEW, "tinygrad extra (py>=3.11 only)"),
    "tomli": (
        UNAVAILABLE_OK,
        "py<3.11 tomllib backport, only conditionally needed; extras-gap candidate "
        "reported 2026-08-15 (tomli; python_version<'3.11' belongs in [test])",
    ),
    "torch._dynamo.trace_rules": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required); "
        "private Dynamo module, absent/relocated on some torch builds",
    ),
    "torch._subclasses.fake_tensor": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required)",
    ),
    "torch.ao.quantization": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required)",
    ),
    "torch.distributed": (
        UNAVAILABLE_OK,
        "torch build capability probe; absent on some torch builds",
    ),
    "torch.distributed.tensor": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required)",
    ),
    "torch.distributed.tensor.parallel": (
        UNAVAILABLE_OK,
        "torch build/version capability probe (core torch is required)",
    ),
    "torch.nn.attention.bias": (
        UNAVAILABLE_OK,
        "torch version capability probe (newer-torch namespace)",
    ),
    "torch_geometric": (TEST_EXTRA, "torch_geometric"),
    "torch_geometric.nn": (TEST_EXTRA, "torch_geometric"),
    "torchaudio": (
        UNAVAILABLE_OK,
        "deliberately NOT in the test extra (pyproject.toml): torchaudio's last "
        "release (2.11.0) is built only for torch 2.11 and fails to load "
        "(undefined symbol: torch_library_impl) against every other declared "
        "torch; the model tests importorskip it by design",
    ),
    "torchvision": (TEST_EXTRA, "torchvision"),
    "torchvision.models": (TEST_EXTRA, "torchvision"),
    "torchvision.models.resnet": (TEST_EXTRA, "torchvision"),
    "torchvision.models.segmentation": (TEST_EXTRA, "torchvision"),
    "torchvision.ops": (TEST_EXTRA, "torchvision"),
    "torchvision.transforms": (TEST_EXTRA, "torchvision"),
    "transformer_lens": (
        UNAVAILABLE_OK,
        "bridge integration without a declared extra; extras-gap candidate reported 2026-08-15",
    ),
    "transformers": (TEST_EXTRA, "transformers"),
    "transformers.modeling_outputs": (TEST_EXTRA, "transformers"),
    "transformers.models.qwen3_moe": (
        TEST_EXTRA,
        "transformers; submodule gate because Qwen3-MoE ships only in newer "
        "transformers releases, so an older pinned install skips honestly",
    ),
    # F33 weightsfree: the accelerate on-ramp row (init_empty_weights) runs
    # where accelerate is installed; the [test]-extra request is filed in
    # the packaging-request ledger and flips this to TEST_EXTRA when merged.
    "accelerate": (
        UNAVAILABLE_OK,
        "weightsfree on-ramp target; [test]-extra request filed (F33)",
    ),
    "visualpriors": (TEST_EXTRA, "visualpriors"),
    "wandb": (OPTIONAL_PREVIEW, "wandb extra"),
    "xarray": (
        UNAVAILABLE_OK,
        "export-bridge target with no declared extra; extras-gap candidate reported 2026-08-15",
    ),
}

# importorskip call sites whose target is computed, not a string literal.
# Each must be consciously allowlisted here (relative posix path from tests/).
DYNAMIC_IMPORTORSKIP_SITES = frozenset(
    {
        "test_backend_registry.py",  # importorskips the registry's own dependency map
    }
)

# Every unconditional skip in tests/ (scanner key -> dated justification).
UNCONDITIONAL_SKIP_LEDGER: dict[str, str] = {
    "test_io_integration.py::test_data_parallel_and_ddp_streaming_case_is_explicitly_skipped": (
        "[2026-08-15] deliberate placeholder: torchlens is single-process by design, "
        "so no DataParallel/DDP streaming capture exists to test; the entry documents "
        "the absent coverage explicitly instead of vanishing from the suite"
    ),
}

# NOTE (b10 R79-4 round 3): ``ast.Try`` is deliberately ABSENT. A ``try``
# BODY always executes, so ``try: pytest.skip(...)`` laundered an
# unconditional skip as "conditional"; only a genuinely branchy ancestor
# counts (an except handler runs iff its try body raised).
_CONDITIONAL_ANCESTORS: tuple[type, ...] = tuple(
    node_type
    for node_type in (
        ast.If,
        ast.IfExp,
        ast.ExceptHandler,
        ast.While,
        ast.For,
        getattr(ast, "Match", None),
    )
    if node_type is not None
)


def warm_scan_caches() -> None:
    """Pre-fill the per-file text/AST caches OUTSIDE any test's charged window.

    Called from the root conftest's collection hook (uncharged time): the
    whole-tree scans below otherwise charge their one-time ~5-7s parse cost to
    whichever audit test runs first under randomized ordering.
    """

    for path in _iter_test_files(TESTS_DIR):
        _parse_with_parents(str(path))


def _iter_test_files(root: Path) -> list[Path]:
    """Every python file under ``root``, excluding bytecode caches."""

    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@cache
def _read_text(path_str: str) -> str:
    """Cached raw source read (needle pre-filter input)."""

    return Path(path_str).read_text()


@cache
def _parse_with_parents(path_str: str) -> ast.Module:
    """Parse a file and annotate every node with its ``_tl_parent``.

    Only files that pass a raw-text needle check ever reach this parse, which
    keeps the scanners inside the smoke duration budget.
    """

    tree = ast.parse(_read_text(path_str), filename=path_str)
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._tl_parent = node  # type: ignore[attr-defined]
    return tree


def collect_importorskip_targets(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Return (literal target -> sites, dynamic-call site files) under ``root``."""

    literal: dict[str, list[str]] = {}
    dynamic: list[str] = []
    for path in _iter_test_files(root):
        if "importorskip" not in _read_text(str(path)):
            continue
        rel = path.relative_to(root).as_posix()
        tree = _parse_with_parents(str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name != "importorskip":
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                literal.setdefault(first.value, []).append(f"{rel}:{node.lineno}")
            else:
                dynamic.append(rel)
    return literal, dynamic


def _is_bare_mark_skip(node: ast.expr) -> bool:
    """Whether ``node`` is ``pytest.mark.skip`` or ``pytest.mark.skip(...)``."""

    target = node.func if isinstance(node, ast.Call) else node
    try:
        rendered = ast.unparse(target)
    except Exception:  # pragma: no cover - unparse never fails on parsed source
        return False
    return rendered.endswith("mark.skip")


def collect_unconditional_skips(root: Path) -> dict[str, str]:
    """Scan ``root`` for skips that fire unconditionally.

    Detects three shapes:

    - a ``pytest.skip(...)`` call statement with no conditional ancestor
      (``if``/``try``/``while``/``for``/``match``) inside its enclosing scope;
    - a bare ``pytest.mark.skip`` / ``pytest.mark.skip(...)`` decorator on a
      function or class (``skipif`` never matches);
    - a module-level ``pytestmark`` assignment carrying a bare skip mark.

    Returns scanner keys (``relpath::qualname``) -> a short site description.
    """

    findings: dict[str, str] = {}
    needles = ("pytest.skip", "mark.skip", "pytestmark")
    for path in _iter_test_files(root):
        text = _read_text(str(path))
        if not any(needle in text for needle in needles):
            continue
        rel = path.relative_to(root).as_posix()
        tree = _parse_with_parents(str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    if _is_bare_mark_skip(dec):
                        findings[f"{rel}::{node.name}"] = (
                            f"bare mark.skip decorator at line {dec.lineno}"
                        )
            elif isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "pytestmark" not in targets:
                    continue
                marks = (
                    node.value.elts
                    if isinstance(node.value, (ast.List, ast.Tuple))
                    else [node.value]
                )
                if any(_is_bare_mark_skip(mark) for mark in marks):
                    findings[f"{rel}::<pytestmark>"] = (
                        f"bare mark.skip in pytestmark at line {node.lineno}"
                    )
            elif isinstance(node, ast.Call):
                try:
                    func_name = ast.unparse(node.func)
                except Exception:  # pragma: no cover
                    continue
                if func_name != "pytest.skip":
                    continue
                ancestor = getattr(node, "_tl_parent", None)
                conditional = False
                scope = "<module>"
                while ancestor is not None:
                    if isinstance(ancestor, _CONDITIONAL_ANCESTORS):
                        conditional = True
                    if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        scope = ancestor.name
                        break
                    ancestor = getattr(ancestor, "_tl_parent", None)
                if not conditional:
                    findings[f"{rel}::{scope}"] = (
                        f"unconditional pytest.skip call at line {node.lineno}"
                    )
    return findings


# ---------------------------------------------------------------------------
# 1. importorskip inventory vs ledger
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_ledger_tiers_are_valid() -> None:
    """Every ledger row uses a closed tier vocabulary and a non-empty note."""

    for target, (tier, note) in IMPORTORSKIP_LEDGER.items():
        assert tier in VALID_TIERS, f"{target}: unknown tier {tier!r}"
        assert note.strip(), f"{target}: empty ledger note"
    sentinel_tier, _ = IMPORTORSKIP_LEDGER[FULL_TEST_EXTRA_SENTINEL]
    assert sentinel_tier == TEST_EXTRA, "the full-extra sentinel must itself be test-extra"


@pytest.mark.smoke
def test_importorskip_inventory_matches_ledger() -> None:
    """Every importorskip target is ledgered; every ledger row is still used."""

    literal, _ = collect_importorskip_targets(TESTS_DIR)
    inventory = set(literal)
    ledgered = set(IMPORTORSKIP_LEDGER)
    unledgered = inventory - ledgered
    stale = ledgered - inventory
    assert not unledgered, (
        "New pytest.importorskip targets must be ledgered consciously in "
        "IMPORTORSKIP_LEDGER (tests/test_skip_audit.py) with an availability tier:\n  "
        + "\n  ".join(f"{t} (e.g. {literal[t][0]})" for t in sorted(unledgered))
    )
    assert not stale, (
        "Ledger rows with no remaining importorskip site (delete them):\n  "
        + "\n  ".join(sorted(stale))
    )


@pytest.mark.smoke
def test_dynamic_importorskip_sites_are_allowlisted() -> None:
    """Non-literal importorskip calls stay confined to the known dynamic sites."""

    _, dynamic = collect_importorskip_targets(TESTS_DIR)
    unexpected = set(dynamic) - DYNAMIC_IMPORTORSKIP_SITES
    assert not unexpected, (
        "importorskip with a computed target evades the ledger; allowlist the site "
        "consciously in DYNAMIC_IMPORTORSKIP_SITES:\n  " + "\n  ".join(sorted(unexpected))
    )
    missing = DYNAMIC_IMPORTORSKIP_SITES - set(dynamic)
    assert not missing, (
        "Allowlisted dynamic importorskip sites no longer exist (delete them):\n  "
        + "\n  ".join(sorted(missing))
    )


@pytest.mark.smoke
def test_importorskip_scanner_is_red_capable(tmp_path: Path) -> None:
    """The inventory scanner catches a planted unledgered target and dynamic site."""

    planted = tmp_path / "test_planted_offender.py"
    planted.write_text(
        "import pytest\n"
        'pytest.importorskip("planted_unledgered_module_xyz")\n'
        "name = 'computed'\n"
        "pytest.importorskip(name)\n"
    )
    literal, dynamic = collect_importorskip_targets(tmp_path)
    assert "planted_unledgered_module_xyz" in literal
    assert dynamic == ["test_planted_offender.py"]


@pytest.mark.smoke
def test_test_extra_targets_import_in_full_env() -> None:
    """When the env claims the full [test] extra, every test-extra target resolves.

    The claim is the sentinel dep importing; a partial install (sentinel absent)
    legitimately skips, but can then never masquerade as full-extra coverage.
    Resolution is checked on each target's top-level module via ``find_spec``
    (cheap; no heavyweight imports), which is what importorskip availability
    hinges on.
    """

    if importlib.util.find_spec(FULL_TEST_EXTRA_SENTINEL) is None:
        pytest.skip(
            f"environment does not claim the full [test] extra "
            f"(sentinel {FULL_TEST_EXTRA_SENTINEL!r} is absent)"
        )
    missing = []
    for target, (tier, _note) in sorted(IMPORTORSKIP_LEDGER.items()):
        if tier != TEST_EXTRA:
            continue
        top_level = target.split(".", 1)[0]
        if importlib.util.find_spec(top_level) is None:
            missing.append(target)
    assert not missing, (
        "The environment claims the full [test] extra (sentinel imports) but these "
        "test-extra importorskip targets do not resolve -- their tests are silently "
        "skipping on what should be a fully-provisioned install:\n  " + "\n  ".join(missing)
    )


# ---------------------------------------------------------------------------
# 2. unconditional-skip ledger
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_no_unledgered_unconditional_skips() -> None:
    """Every unconditional skip in tests/ is a consciously ledgered placeholder."""

    findings = collect_unconditional_skips(TESTS_DIR)
    unledgered = set(findings) - set(UNCONDITIONAL_SKIP_LEDGER)
    stale = set(UNCONDITIONAL_SKIP_LEDGER) - set(findings)
    assert not unledgered, (
        "Unconditional skips make tests silently dead; either realize the test, "
        "delete it, or ledger it as a dated deliberate placeholder in "
        "UNCONDITIONAL_SKIP_LEDGER (tests/test_skip_audit.py):\n  "
        + "\n  ".join(f"{k}: {findings[k]}" for k in sorted(unledgered))
    )
    assert not stale, (
        "Ledgered unconditional skips no longer exist (delete the rows):\n  "
        + "\n  ".join(sorted(stale))
    )


@pytest.mark.smoke
def test_unconditional_skip_scanner_is_red_capable(tmp_path: Path) -> None:
    """The scanner catches all three planted offender shapes and no decoys."""

    planted = tmp_path / "test_planted_skips.py"
    planted.write_text(
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.mark.skip(reason='planted decorator offender')\n"
        "def test_decorated():\n"
        "    pass\n"
        "\n"
        "\n"
        "def test_body_skip():\n"
        "    pytest.skip('planted body offender')\n"
        "\n"
        "\n"
        "def test_conditional_decoy():\n"
        "    if False:\n"
        "        pytest.skip('conditional; must NOT be flagged')\n"
        "\n"
        "\n"
        "@pytest.mark.skipif(True, reason='skipif decoy; must NOT be flagged')\n"
        "def test_skipif_decoy():\n"
        "    pass\n"
        "\n"
        "\n"
        "def test_try_laundered_skip():\n"
        "    try:\n"
        "        pytest.skip('try BODY always runs; MUST be flagged (R79-4)')\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
        "\n"
        "def test_handler_skip_decoy():\n"
        "    try:\n"
        "        _probe()\n"
        "    except ImportError:\n"
        "        pytest.skip('handler runs iff the body raised; must NOT be flagged')\n"
    )
    marked = tmp_path / "test_planted_pytestmark.py"
    marked.write_text(
        "import pytest\n"
        "pytestmark = [pytest.mark.smoke, pytest.mark.skip(reason='planted module offender')]\n"
    )
    findings = collect_unconditional_skips(tmp_path)
    assert set(findings) == {
        "test_planted_skips.py::test_decorated",
        "test_planted_skips.py::test_body_skip",
        "test_planted_skips.py::test_try_laundered_skip",
        "test_planted_pytestmark.py::<pytestmark>",
    }


# ---------------------------------------------------------------------------
# 2b. skipif audit: repo-unsatisfiable conditions (b10 R79-1 round 3)
# ---------------------------------------------------------------------------
#
# ``skipif`` was explicitly OUT of the unconditional-skip scanner's scope, and
# that is where the real always-skips hid: ten tests gated on (a) a gitignored
# local artifact that cannot exist in any fresh clone or CI runner, and (b) an
# env var nothing in the repo ever sets. A skip that fires EVERYWHERE is a
# deleted test wearing a disguise — each such site must be ledgered, and the
# ledger is two-way: when the condition becomes satisfiable (the artifact is
# committed / the env var gains a repo-side setter) the row goes stale and
# this audit demands its removal.

#: scanner key (``relpath::qualname``) -> dated justification for a skipif
#: whose condition is UNSATISFIABLE in every repo-defined environment.
REPO_UNSATISFIABLE_SKIPIF_LEDGER: dict[str, str] = {}


def _module_path_literal_constants(tree: ast.AST) -> dict[str, str]:
    """Map module-level names to repo-relative paths built from `/` literals.

    Matches the ``NAME = <base> / "seg" / "seg"`` idiom: the string-literal
    segments are joined; the (dynamic) base is ignored, since the repo-relative
    tail is what decides trackability.
    """

    constants: dict[str, str] = {}
    for node in tree.body if hasattr(tree, "body") else []:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        segments: list[str] = []
        value: ast.expr = node.value
        while isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
            if isinstance(value.right, ast.Constant) and isinstance(value.right.value, str):
                segments.append(value.right.value)
            value = value.left
        if segments:
            constants[target.id] = "/".join(reversed(segments))
    return constants


def _module_env_gate_constants(tree: ast.AST) -> dict[str, str]:
    """Map module-level names to the env var they gate on.

    Matches ``NAME = os.environ.get("X")`` optionally wrapped in ``bool(...)``.
    """

    constants: dict[str, str] = {}
    for node in tree.body if hasattr(tree, "body") else []:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value: ast.expr = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "bool"
            and value.args
        ):
            value = value.args[0]
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in {"get", "getenv"}
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
        ):
            constants[target.id] = value.args[0].value
    return constants


def _skipif_gate(condition: ast.expr) -> tuple[str, str] | None:
    """Classify a skipif condition into an auditable gate shape.

    Returns ``("artifact", NAME)`` for ``not NAME.exists()``, ``("env", NAME)``
    for ``not NAME``, or ``None`` for any other shape (out of scope).
    """

    if not (isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not)):
        return None
    operand = condition.operand
    if (
        isinstance(operand, ast.Call)
        and isinstance(operand.func, ast.Attribute)
        and operand.func.attr == "exists"
        and isinstance(operand.func.value, ast.Name)
    ):
        return ("artifact", operand.func.value.id)
    if isinstance(operand, ast.Name):
        return ("env", operand.id)
    return None


@cache
def _tracked_repo_files() -> frozenset[str]:
    """Return git-tracked repo-relative paths (empty outside a git checkout)."""

    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return frozenset(completed.stdout.splitlines())


@cache
def _repo_ignored(tail: str) -> bool:
    """Return whether ``tail`` is gitignored (never present in a fresh clone).

    A gitignored artifact existing locally must not rescue its gate from the
    repo-unsatisfiable classification: the audit's verdict otherwise flips
    between the owner's boxes (private ``.research/`` notes present) and every
    fresh clone/CI runner (fw3settle: two ledgered csv-export rows read
    "stale" only on checkouts that happened to carry the private schema doc).
    """

    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", tail],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


@cache
def _repo_config_corpus() -> str:
    """Concatenate the repo's tracked config/tooling text for env-var searches.

    An env var counts as REPO-SET when any tracked workflow, script, tool, or
    config mentions it; only vars mentioned NOWHERE outside their defining
    test module classify as repo-unsatisfiable gates.
    """

    chunks: list[str] = []
    for rel in sorted(_tracked_repo_files()):
        if rel.startswith((".github/", "scripts/", "tools/", "benchmarks/")) or rel in (
            "pyproject.toml",
            ".pre-commit-config.yaml",
        ):
            path = REPO_ROOT / rel
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(chunks)


def collect_repo_unsatisfiable_skipifs(root: Path, repo_root: Path) -> dict[str, str]:
    """Scan ``root`` for skipif sites whose condition no repo environment meets.

    Two proven classes (b10 R79-1):

    - **artifact gates**: ``not NAME.exists()`` where NAME's string-literal
      path tail is neither git-tracked nor present on disk — impossible in a
      fresh clone or CI runner;
    - **env gates**: ``not NAME`` where NAME wraps ``os.environ.get("X")`` and
      ``X`` is mentioned in no tracked workflow/script/tool/config.

    Returns scanner keys (``relpath::qualname``) -> site description.
    """

    findings: dict[str, str] = {}
    for path in _iter_test_files(root):
        text = _read_text(str(path))
        if "skipif" not in text:
            continue
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(text, filename=str(path))
        path_constants = _module_path_literal_constants(tree)
        env_constants = _module_env_gate_constants(tree)

        def classify(
            condition: ast.expr,
            key: str,
            lineno: int,
            # Per-file maps bound as defaults: B023 hygiene (also enforced by
            # the ruff deferred-debt ratchet this same wave landed).
            path_constants: dict[str, str] = path_constants,
            env_constants: dict[str, str] = env_constants,
        ) -> None:
            gate = _skipif_gate(condition)
            if gate is None:
                return
            kind, name = gate
            if kind == "artifact" and name in path_constants:
                tail = path_constants[name]
                if tail not in _tracked_repo_files() and (
                    _repo_ignored(tail) or not (repo_root / tail).exists()
                ):
                    findings[key] = f"line {lineno}: gated on untracked, absent artifact {tail!r}"
            elif kind == "env" and name in env_constants:
                var = env_constants[name]
                if var not in _repo_config_corpus():
                    findings[key] = f"line {lineno}: gated on env var {var!r} set by no repo config"

        def classify_mark(mark: ast.expr, key: str) -> None:
            if (
                isinstance(mark, ast.Call)
                and isinstance(mark.func, ast.Attribute)
                and mark.func.attr == "skipif"
                and mark.args
            ):
                classify(mark.args[0], key, mark.lineno)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    classify_mark(dec, f"{rel}::{node.name}")
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                marks = (
                    node.value.elts
                    if isinstance(node.value, (ast.List, ast.Tuple))
                    else [node.value]
                )
                for mark in marks:
                    classify_mark(mark, f"{rel}::<pytestmark>")
    return findings


@pytest.mark.smoke
def test_no_unledgered_repo_unsatisfiable_skipifs() -> None:
    """Every provably-always-firing skipif is ledgered; no ledger row is stale."""

    findings = collect_repo_unsatisfiable_skipifs(TESTS_DIR, REPO_ROOT)
    unledgered = set(findings) - set(REPO_UNSATISFIABLE_SKIPIF_LEDGER)
    stale = set(REPO_UNSATISFIABLE_SKIPIF_LEDGER) - set(findings)
    assert not unledgered and not stale, (
        "repo-unsatisfiable skipif drift (a skip that fires everywhere is a "
        "deleted test wearing a disguise — b10 R79-1). Ledger new sites with a "
        "dated justification in REPO_UNSATISFIABLE_SKIPIF_LEDGER; delete rows "
        "whose condition became satisfiable.\n"
        f"  unledgered: {sorted(unledgered)}\n"
        f"  stale: {sorted(stale)}\n"
        f"  details: { {key: findings[key] for key in sorted(unledgered)} }"
    )


def test_repo_unsatisfiable_skipif_scanner_is_red_capable(tmp_path: Path) -> None:
    """Planted artifact/env gates are caught; satisfiable decoys are not."""

    planted = tmp_path / "test_planted_skipifs.py"
    planted.write_text(
        "import os\n"
        "import pytest\n"
        "from pathlib import Path\n"
        "\n"
        "MISSING = Path(__file__).parents[1] / 'no_such_dir' / 'no_such.db'\n"
        "TRACKED = Path(__file__).parents[1] / 'pyproject.toml'\n"
        "HAS_PHANTOM = bool(os.environ.get('TORCHLENS_PHANTOM_NEVER_SET_VAR'))\n"
        "HAS_REAL = bool(os.environ.get('CI'))\n"
        "\n"
        "\n"
        "@pytest.mark.skipif(not MISSING.exists(), reason='planted artifact gate')\n"
        "def test_artifact_gated():\n"
        "    pass\n"
        "\n"
        "\n"
        "@pytest.mark.skipif(not TRACKED.exists(), reason='tracked decoy; not flagged')\n"
        "def test_tracked_decoy():\n"
        "    pass\n"
        "\n"
        "\n"
        "@pytest.mark.skipif(not HAS_PHANTOM, reason='planted env gate')\n"
        "def test_env_gated():\n"
        "    pass\n"
        "\n"
        "\n"
        "@pytest.mark.skipif(not HAS_REAL, reason='repo-set decoy; not flagged')\n"
        "def test_repo_set_decoy():\n"
        "    pass\n"
    )
    findings = collect_repo_unsatisfiable_skipifs(tmp_path, REPO_ROOT)
    assert set(findings) == {
        "test_planted_skipifs.py::test_artifact_gated",
        "test_planted_skipifs.py::test_env_gated",
    }


# ---------------------------------------------------------------------------
# 2b. Device-gated skipifs (r7 R79-1): CUDA skips that fire on EVERY leg
# ---------------------------------------------------------------------------

#: Tests gated on ``torch.cuda.is_available()`` execute on NO CI leg (every
#: workflow installs ``+cpu`` torch) and not on the owner's GPU box either
#: (documented cu130-wheel / driver mismatch) — the exact "skip that fires
#: everywhere" class the sibling ledgers govern, previously falling into
#: none of them. Every row is DARK COVERAGE disclosed until a CUDA leg
#: exists; deleting a row requires the site to gain an executing environment,
#: never just deleting the test.
_CUDA_DARK = (
    "[2026-08-16] CUDA-gated: all CI legs install +cpu torch and the one GPU "
    "box has the cu130-wheel/driver mismatch; dark until a device leg exists"
)

DEVICE_GATED_SKIPIF_LEDGER: dict[str, str] = {
    "backend_parity/test_b10_torch_characterization.py::test_paramless_model_cuda_inputs_stay_on_cuda": (
        "[2026-08-19] device-preservation pin for the H200 finding (a device-less "
        "model's CUDA inputs silently computed on CPU); dark on CPU-only CI, "
        "executed on the Fellows-cluster CUDA leg"
    ),
    "test_explorer_watch_gates.py::test_module_tier_overhead_gates": (
        "[2026-08-29] F25 explorer D25 overhead gate: the measured per-step "
        "watch-overhead A/B rows need two gpt2 replicas training on CUDA (a "
        "CPU run measures a different regime and would publish a false "
        "ratio); dark on CPU-only CI and this box's cu-wheel/driver "
        "mismatch, executed on the Fellows-cluster CUDA leg -- the pinned "
        "docs/_watch_perf_numbers.md rows are its recorded evidence"
    ),
    "test_hash_determinism.py::test_graph_shape_hash_matches_between_cpu_and_cuda": _CUDA_DARK,
    "test_kernel_telemetry.py::test_real_cuda_cupti_correlation_matrix": (
        "[2026-08-17] L3 telemetry OPTIONAL_INTEGRATION: the exact Kineto CUDA/CUPTI "
        "correlation matrix is NOT-RUN-DISCLOSED on CPU-only CI and this no-NVIDIA host; "
        "synthetic correlation graphs remain an always-executed honesty tripwire"
    ),
    "test_param_as_input.py::test_cross_device_parameter_input_matches_plain_tensor_path_if_cuda_available": _CUDA_DARK,
    "test_perf_bundle.py::test_cuda_path_still_runs_when_available": _CUDA_DARK,
    "test_robustness_pr2.py::test_cuda_channels_last_safe_copy": _CUDA_DARK,
    "test_snoop_real_model.py::test_cuda_metadata_echo_is_zero_sync": (
        "[2026-08-28] lane F28 snoop memo test 4: metadata echo under "
        "torch.cuda.set_sync_debug_mode('error') proves the zero-device-sync "
        "claim; dark on CPU-only CI, executed on the Fellows-cluster CUDA leg"
    ),
    "test_robustness_pr2.py::test_cuda_forward_pass_still_logs": _CUDA_DARK,
    "test_runnable_r36_regressions.py::TestCudaStagingAndReadiness": _CUDA_DARK,
    "test_tlspec_runnable_r35_attestation_lattice.py::test_r35_device_diverged_run_is_never_attested": _CUDA_DARK,
    "test_tlspec_runnable_r35_exact_semantics.py::test_r35_seeded_run_restores_produced_only_cuda_rng": _CUDA_DARK,
    "test_tlspec_runnable_r65_state_metadata_parity.py::test_r65_cuda_is_shared_read_stays_verified": _CUDA_DARK,
    "test_tlspec_runnable_r65_state_metadata_parity.py::test_r65_cuda_staged_state_satisfies_full_signature": _CUDA_DARK,
    "test_tlspec_runnable_r65_state_metadata_parity.py::test_r65_pinned_read_refuses_when_pinned": _CUDA_DARK,
    "test_tlspec_runnable_r65_torch_rng.py::test_cuda_default_get_offset_is_ceiled": _CUDA_DARK,
    "test_tlspec_runnable_r65_torch_rng.py::test_r67_cuda_default_get_offset_ceilings_every_run": _CUDA_DARK,
    "test_tlspec_runnable_r67_storage_metadata.py::test_r67_pinned_state_read_refuses_via_observation": _CUDA_DARK,
    "test_transport_idiom.py::test_cross_device_channels_last_single_host_copy": _CUDA_DARK,
}


def collect_device_gated_skipifs(root: Path) -> dict[str, str]:
    """Scan ``root`` for skipif sites gated on ``torch.cuda.is_available()``.

    Matches the canonical decorator/pytestmark shape
    ``pytest.mark.skipif(not torch.cuda.is_available(), ...)``. Returns
    scanner keys (``relpath::qualname``) -> site description.
    """

    def _is_cuda_gate(condition: ast.expr) -> bool:
        if not (isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not)):
            return False
        operand = condition.operand
        return (
            isinstance(operand, ast.Call)
            and isinstance(operand.func, ast.Attribute)
            and operand.func.attr == "is_available"
            and isinstance(operand.func.value, ast.Attribute)
            and operand.func.value.attr == "cuda"
            and isinstance(operand.func.value.value, ast.Name)
            and operand.func.value.value.id == "torch"
        )

    findings: dict[str, str] = {}
    for path in _iter_test_files(root):
        text = _read_text(str(path))
        if "is_available" not in text:
            continue
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(text, filename=str(path))

        def classify_mark(mark: ast.expr, key: str) -> None:
            if (
                isinstance(mark, ast.Call)
                and isinstance(mark.func, ast.Attribute)
                and mark.func.attr == "skipif"
                and mark.args
                and _is_cuda_gate(mark.args[0])
            ):
                findings[key] = f"line {mark.lineno}: gated on torch.cuda.is_available()"

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    classify_mark(dec, f"{rel}::{node.name}")
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                marks = (
                    node.value.elts
                    if isinstance(node.value, (ast.List, ast.Tuple))
                    else [node.value]
                )
                for mark in marks:
                    classify_mark(mark, f"{rel}::<pytestmark>")
    return findings


@pytest.mark.smoke
def test_no_unledgered_device_gated_skipifs() -> None:
    """Every CUDA-gated skip is ledgered as dark coverage; no row is stale."""

    findings = collect_device_gated_skipifs(TESTS_DIR)
    unledgered = set(findings) - set(DEVICE_GATED_SKIPIF_LEDGER)
    stale = set(DEVICE_GATED_SKIPIF_LEDGER) - set(findings)
    assert not unledgered and not stale, (
        "device-gated skipif drift (r7 R79-1: these fire on every CI leg and "
        "the owner's GPU box alike, so each is dark coverage that must be "
        "DISCLOSED). Ledger new sites with a dated justification in "
        "DEVICE_GATED_SKIPIF_LEDGER; delete rows only when the site gains an "
        "executing environment.\n"
        f"  unledgered: {sorted(unledgered)}\n"
        f"  stale: {sorted(stale)}"
    )


def test_device_gated_skipif_scanner_is_red_capable(tmp_path: Path) -> None:
    """A planted CUDA gate is caught; a satisfiable device decoy is not."""

    planted = tmp_path / "test_planted_cuda_skipifs.py"
    planted.write_text(
        "import pytest\n"
        "import torch\n"
        "\n"
        "\n"
        "@pytest.mark.skipif(not torch.cuda.is_available(), reason='planted')\n"
        "def test_cuda_gated():\n"
        "    pass\n"
        "\n"
        "\n"
        "@pytest.mark.skipif(not torch.backends.mkldnn.is_available(), reason='decoy')\n"
        "def test_other_backend_decoy():\n"
        "    pass\n"
    )
    findings = collect_device_gated_skipifs(tmp_path)
    assert set(findings) == {"test_planted_cuda_skipifs.py::test_cuda_gated"}


# ---------------------------------------------------------------------------
# 2c. tripwire-guard classification of unavailable-ok targets (skip-audit lane)
# ---------------------------------------------------------------------------
#
# The ledger tiers classify DEPENDENCY AVAILABILITY; they say nothing about
# what the guarded test protects. That is where the label-geometry incident
# hid (2026-08-16): ``cairosvg`` sat truthfully in ``unavailable-ok`` while
# the test it guards -- the 16-model rendered label-geometry gate -- silently
# disarmed on every dev box, and 4 models drifted into hard violations behind
# a green summary. Same shape as the core.hooksPath incident: a check that
# looks armed and is not.
#
# So every ``unavailable-ok`` target must now be consciously classified:
#
# - ``TRIPWIRE_GUARD_TARGETS``: the dep's absence disarms a correctness or
#   honesty GATE (rendered-output audits, byte/pixel honesty layers). On any
#   box claiming the full ``[test]`` extra (the same sentinel claim the
#   test-extra enforcement uses) these must RESOLVE -- a missing dep breaks
#   the build instead of quietly disarming the check.
# - ``OPTIONAL_INTEGRATION_TARGETS``: genuinely-optional integration coverage
#   whose absence costs breadth, never gate integrity; one-line reason each.
#
# ``optional-preview`` targets are out of the classification's domain: that
# tier is by construction backend/bridge extras with their own preview envs
# (and the weekly workflow's executed-floor gate arms the bridge families).

#: unavailable-ok target -> the tripwire gate its absence silently disarms.
TRIPWIRE_GUARD_TARGETS: dict[str, str] = {
    "cairosvg": (
        "tests/test_label_geometry.py (16-model rendered label-geometry gate) and "
        "tests/test_bundle_diff_renderer.py's pixel-exoneration layer (its byte-drift "
        "check degraded to a SKIP once before -- b10 R78-4)"
    ),
    "fitz": (
        "tests/test_render_bugs.py::test_large_composed_pdf_contains_visible_graph_region "
        "(composed-PDF visible-graph-region honesty gate)"
    ),
    "matplotlib": (
        "whole-file guards on tests/test_viz_display_behaviors.py and tests/test_reprs.py "
        "(viz display/repr behavior checks disarm as entire files)"
    ),
    "matplotlib.pyplot": (
        "per-test guards in tests/test_node_plots.py (tensor-display rendering checks)"
    ),
    "onnx": (
        "tests/test_exports.py::test_netron_export_is_valid_onnx_modelproto_json "
        "plus the whole tests/test_netron_export_contract.py / _rolled.py T1 tier "
        "(strict parse + check_model(full_check=True); without it the "
        "'artifact opens in Netron' claim reverts to unverified)"
    ),
    "google.protobuf.json_format": (
        "strict protobuf JSON parse inside the netron-export acceptance gate "
        "(same tests as the onnx target)"
    ),
}

#: unavailable-ok target -> why its absence is breadth loss, not gate loss.
OPTIONAL_INTEGRATION_TARGETS: dict[str, str] = {
    "netron": (
        "tests/test_netron_export_vendor.py (the executed netron 9.2.2 parser "
        "harness) and the serve round-trip in tests/test_netron_export_serve.py; "
        "netron is its OWN declared extra (not part of [test]) and the dedicated "
        "nightly.yml netron-vendor job installs it and independently attests the "
        "executed floor for both files, so the full-[test]-extra box never needs "
        "it -- absence there costs nothing beyond the transcribed-sniffer canary"
    ),
    "playwright.sync_api": (
        "tests/test_netron_export_browser.py (T4 semantic smoke); Playwright has "
        "no declared extra and installs nowhere in CI -- the browser smoke stays "
        "queued per the gate-law ruling in nightly.yml's netron-vendor job "
        "comment, so its absence is the documented current state, not a silent "
        "regression"
    ),
    "clearml": (
        "T-RELAY-C relay-fidelity pin (F26); absence costs the vendor pin "
        "only -- the dep-free relay-law halves (detection + the G6 histogram "
        "refusal) run unconditionally in test_trackers_sinks_delivery.py"
    ),
    "cornet": "research-model real-world coverage; GitHub-only distribution",
    "jsonschema": (
        "the agent schema lockstep's full Draft-2020-12 validation leg (F29); "
        "the dep-free structural checks (index/id/title, registry parity) run "
        "everywhere, so absence costs validation breadth only"
    ),
    "dacite": "model-explorer export-bridge demo dependency",
    "dagua": "unreleased in-development layout engine",
    "e3nn.o3": "research-model real-world coverage, deliberately undeclared",
    "fvcore.nn": (
        "supplementary FLOP-counter cross-oracle (A07); the closed-form and "
        "gpt2/bert pins are the blocking oracles, so absence costs breadth only"
    ),
    "git": "release-environment-only (hash-locked release-defenses job installs it)",
    "model_explorer": "export-bridge integration target with no declared extra",
    "pennylane": "quantum-ML research-model coverage, deliberately undeclared",
    "pyg_lib": "torch_geometric compiled extension for test_dimenet, deliberately undeclared",
    "semantic_release": "release-environment-only (release-defenses job)",
    "tensorboard": "export-bridge integration target with no declared extra",
    "tomli": "py<3.11 tomllib backport, only conditionally needed",
    "transformer_lens": "bridge integration without a declared extra",
    "accelerate": "weightsfree on-ramp target; [test]-extra request filed (F33)",
    "xarray": "export-bridge integration target with no declared extra",
    "torch._dynamo.trace_rules": "torch build/version capability probe",
    "torch._subclasses.fake_tensor": "torch build/version capability probe",
    "torch.ao.quantization": "torch build/version capability probe",
    "torch.distributed": "torch build capability probe",
    "torch.distributed.tensor": "torch build/version capability probe",
    "torch.distributed.tensor.parallel": "torch build/version capability probe",
    "torch.nn.attention.bias": "torch version capability probe",
    "torchaudio": (
        "deliberately undeclared in [test] (undefined-symbol load failure against "
        "every torch except its own pinned 2.11.0); the model tests importorskip "
        "it, costing breadth only"
    ),
}


def _unresolved_tripwire_targets(targets: dict[str, str]) -> list[str]:
    """Tripwire targets whose top-level module does not resolve right now."""

    return sorted(
        target for target in targets if importlib.util.find_spec(target.split(".", 1)[0]) is None
    )


@pytest.mark.smoke
def test_unavailable_ok_targets_are_all_classified() -> None:
    """Every unavailable-ok target is consciously tripwire XOR optional.

    A new ``unavailable-ok`` ledger row must land with a classification, so
    "the dep is legitimately absent" can never again silently answer the
    different question "is the guarded test a gate?".
    """

    unavailable_ok = {
        target for target, (tier, _note) in IMPORTORSKIP_LEDGER.items() if tier == UNAVAILABLE_OK
    }
    tripwire = set(TRIPWIRE_GUARD_TARGETS)
    optional = set(OPTIONAL_INTEGRATION_TARGETS)
    overlap = tripwire & optional
    unclassified = unavailable_ok - tripwire - optional
    stale = (tripwire | optional) - unavailable_ok
    assert not overlap, f"targets classified both ways: {sorted(overlap)}"
    assert not unclassified, (
        "unavailable-ok importorskip targets must be consciously classified as "
        "TRIPWIRE_GUARD_TARGETS (absence disarms a correctness/honesty gate) or "
        "OPTIONAL_INTEGRATION_TARGETS (breadth-only) in tests/test_skip_audit.py:\n  "
        + "\n  ".join(sorted(unclassified))
    )
    assert not stale, (
        "classified targets no longer ledgered unavailable-ok (move or delete "
        "the rows):\n  " + "\n  ".join(sorted(stale))
    )
    for target, reason in (*TRIPWIRE_GUARD_TARGETS.items(), *OPTIONAL_INTEGRATION_TARGETS.items()):
        assert reason.strip(), f"{target}: empty classification reason"


@pytest.mark.smoke
def test_tripwire_guard_deps_resolve_in_full_env() -> None:
    """A full-[test]-extra box may not silently disarm a tripwire gate.

    The claim is the same sentinel the test-extra enforcement uses: a partial
    install legitimately skips, but a provisioned dev/CI box with a tripwire
    dep absent means a correctness gate is skipping while every summary reads
    green -- that must be LOUD. Remedy: install the named dep (additive), or
    consciously reclassify the target with its gate's owner.
    """

    if importlib.util.find_spec(FULL_TEST_EXTRA_SENTINEL) is None:
        pytest.skip(
            f"environment does not claim the full [test] extra "
            f"(sentinel {FULL_TEST_EXTRA_SENTINEL!r} is absent)"
        )
    missing = _unresolved_tripwire_targets(TRIPWIRE_GUARD_TARGETS)
    assert not missing, (
        "TRIPWIRE GATES ARE SILENTLY SKIPPING on this full-[test]-extra "
        "environment -- each absent dep below disarms a correctness/honesty "
        "gate that then reads green in every summary. Install the dep or "
        "consciously reclassify it in tests/test_skip_audit.py:\n"
        + "\n".join(f"  {target}: disarms {TRIPWIRE_GUARD_TARGETS[target]}" for target in missing)
    )


@pytest.mark.smoke
def test_tripwire_resolution_check_is_red_capable() -> None:
    """A planted unresolvable tripwire target is reported missing."""

    planted = {
        "torchlens_planted_missing_dep_xyz": "planted gate (must be reported)",
        "pytest": "resolvable decoy (must NOT be reported)",
    }
    assert _unresolved_tripwire_targets(planted) == ["torchlens_planted_missing_dep_xyz"]


# ---------------------------------------------------------------------------
# 3. requires_assertions / python -O leg self-verification
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_o_leg_sentinel_executes() -> None:
    """Trivial sentinel proving the ``-O`` subprocess really executes tests.

    Deliberately assert-free (``-O`` strips assert statements): failure is an
    explicit ``pytest.fail``.
    """

    if (1, 2)[0] != 1:  # pragma: no cover - arithmetic sanity sentinel
        pytest.fail("sentinel arithmetic failed")


@pytest.mark.slow
def test_requires_assertions_marker_engages_under_python_O(tmp_path: Path) -> None:
    """``requires_assertions`` tests SKIP (never pass/fail) under ``python -O``.

    No CI workflow runs a ``-O`` leg, so without this subprocess probe the
    marker and its conftest hook would rot unverified. One targeted run under
    ``-O`` proves both directions: the marked tests are collected and SKIPPED
    with the documented reason, while an ordinary test executes and passes on
    the same interpreter. Red-capable end to end: removing the conftest hook
    makes the marked assert-based tests FAIL under ``-O`` (stripped asserts
    never raise), and removing the marker breaks the exact skip count.
    """

    marked_nodes = [
        "tests/test_postprocess_contract_arming.py::test_unknown_step_contract_is_rejected",
        "tests/test_postprocess_contract_arming.py::test_undeclared_write_is_rejected",
    ]
    plain_node = "tests/test_skip_audit.py::test_o_leg_sentinel_executes"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(tmp_path / "oleg-bt"),
            *marked_nodes,
            plain_node,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"-O leg subprocess failed:\n{output}"
    assert "1 passed" in output, f"plain sentinel did not execute under -O:\n{output}"
    assert "2 skipped" in output, (
        f"expected exactly the two requires_assertions tests to skip under -O:\n{output}"
    )
    assert "requires assertions" in output, (
        f"skip reason does not name the requires_assertions contract:\n{output}"
    )
