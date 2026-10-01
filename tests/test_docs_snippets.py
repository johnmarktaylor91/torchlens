"""Executable checks for documentation snippets.

Two gates live here:

* the per-block gate over the P2 ``docs/`` pages (``DOC_FILES``), and
* the canonical-page gate that EXECUTES every non-sketch Python fence in
  ``README.md``, ``CLAUDE.md``, and ``AGENTS.md`` top to bottom, statement by
  statement, in one shared namespace per page.

The canonical-page ambient contract is deliberately tiny: the harness injects
only the names the pages' prose treats as application-supplied -- ``model`` and
``x`` (a small fully-convolutional demo model), plus ``tf_model``/``tf_x`` when
the TensorFlow preview dependency is installed. Everything else must be defined
by the documentation code itself; a fence that references an undefined name is
a documentation bug and fails this gate. Blocks whose first line is a comment
containing "API sketch" are compile-gated instead of executed.
"""

from __future__ import annotations

import ast
import importlib.util
import linecache
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

#: Executed pages, grouped by docs section into one parametrized family each.
#: The duration-budget lint bounds every 5s-tier family at
#: ``max(2 x per-test budget, per-cell allowance x cells)`` (tests/conftest.py),
#: and ONE family over every page measured 13.2s across 66 cells against that
#: 12s bound once the FW2 closure sweep added the 2026-08-28 pages (the
#: doctor(), resnet50 summary, and wandb cells alone charge ~7.7s). The split
#: is by page group, never by cell cost, so per-cell tiers stay honest: only
#: the cold ``torch.compile`` fence is heavy-class by measurement.
DOC_FILE_GROUPS: dict[str, tuple[str, ...]] = {
    "guides": (
        "performance.md",
        "for-ai-agents.md",
        # FW2 closure sweep: the 2026-08-28 pages whose fences are
        # self-contained programs (spelling-only sketches on them are tagged
        # ```text).
        "migration/coming_from_torchsnooper.md",
        "migration/from_hooks.md",
        "native-torch.md",
        "recipes/lrp_epsilon_litmus.md",
    ),
    "reference_tooling": (
        "reference/debug.md",
        "reference/export.md",
    ),
    "reference_views": (
        "reference/attribution.md",
        "reference/collapse.md",
        "reference/lenses.md",
        "reference/summary.md",
    ),
}
DOC_FILES = tuple(name for group in DOC_FILE_GROUPS.values() for name in group)
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def _docs_dir() -> Path:
    """Return the documentation directory.

    Returns
    -------
    Path
        Absolute path to ``docs``.
    """

    return Path(__file__).resolve().parents[1] / "docs"


def _iter_python_blocks(file_names: tuple[str, ...] = DOC_FILES) -> list[tuple[str, int, str]]:
    """Collect Python code fences from the P2 documentation pages.

    Parameters
    ----------
    file_names:
        Markdown page names under ``docs`` to scan (a ``DOC_FILE_GROUPS`` group,
        or every executed page by default).

    Returns
    -------
    list[tuple[str, int, str]]
        Tuples of ``(file_name, block_index, code)``.
    """

    blocks: list[tuple[str, int, str]] = []
    for file_name in file_names:
        text = (_docs_dir() / file_name).read_text(encoding="utf-8")
        for block_index, match in enumerate(BLOCK_RE.finditer(text), start=1):
            blocks.append((file_name, block_index, match.group("code")))
    return blocks


def _doc_block_params(group: str) -> list[object]:
    """Wrap one page group's blocks in params; compile-bearing blocks are HEAVY.

    r7 R41/R72 fresh-env gate finding: the ``torch.compile`` fence in
    reference/debug.md charges ~15s CPU in a COLD environment (first-run
    dynamo/inductor compilation; the dev box's warm cache hid it), blowing
    the 5s-tier budget the moment enforcement became always-on. Cold-cache
    compilation cost is heavy-class by measurement, per-cell.
    """

    params: list[object] = []
    for file_name, block_index, code in _iter_python_blocks(DOC_FILE_GROUPS[group]):
        marks = (
            (pytest.mark.heavy,) if ("torch.compile" in code or "frames_compiled" in code) else ()
        )
        params.append(pytest.param(file_name, block_index, code, marks=marks))
    return params


def _run_doc_block(file_name: str, block_index: int, code: str, tmp_path: Path) -> None:
    """Run one Python code fence from the new docs pages.

    Parameters
    ----------
    file_name:
        Markdown file name under ``docs``.
    block_index:
        One-based code-block index within the file.
    code:
        Python code fence body.
    tmp_path:
        Temporary directory supplied by pytest.
    """

    synthetic_filename = f"{file_name}:python-block-{block_index}"
    linecache.cache[synthetic_filename] = (
        len(code),
        None,
        [f"{line}\n" for line in code.splitlines()],
        synthetic_filename,
    )
    namespace: dict[str, Any] = {
        "__file__": synthetic_filename,
        "__name__": f"docs_snippet_{Path(file_name).stem}_{block_index}",
        "DOCS_TMPDIR": str(tmp_path),
    }
    try:
        exec(compile(code, synthetic_filename, "exec"), namespace)
    except (ImportError, ModuleNotFoundError) as exc:
        # A doc example may demonstrate an OPTIONAL-dependency feature (e.g. the xarray
        # export) that is not installed in every test env (CI installs only core deps).
        # The example is still correct; skip when its dependency is absent rather than
        # fail. Users who want that feature install the extra.
        pytest.skip(f"doc snippet requires an unavailable optional dependency: {exc}")


@pytest.mark.parametrize(
    ("file_name", "block_index", "code"),
    _doc_block_params("guides"),
    ids=lambda value: str(value),
)
def test_doc_python_blocks_run_guides(
    file_name: str, block_index: int, code: str, tmp_path: Path
) -> None:
    """Run one fence from the guide, migration, and recipe pages."""

    _run_doc_block(file_name, block_index, code, tmp_path)


@pytest.mark.parametrize(
    ("file_name", "block_index", "code"),
    _doc_block_params("reference_tooling"),
    ids=lambda value: str(value),
)
def test_doc_python_blocks_run_reference_tooling(
    file_name: str, block_index: int, code: str, tmp_path: Path
) -> None:
    """Run one fence from the debug and export reference pages."""

    _run_doc_block(file_name, block_index, code, tmp_path)


@pytest.mark.parametrize(
    ("file_name", "block_index", "code"),
    _doc_block_params("reference_views"),
    ids=lambda value: str(value),
)
def test_doc_python_blocks_run_reference_views(
    file_name: str, block_index: int, code: str, tmp_path: Path
) -> None:
    """Run one fence from the attribution, collapse, lenses, and summary pages."""

    _run_doc_block(file_name, block_index, code, tmp_path)


CANONICAL_PAGES = ("README.md", "CLAUDE.md", "AGENTS.md")
_SKETCH_MARKER_RE = re.compile(r"^\s*#.*\bAPI sketch\b", re.IGNORECASE)
_OPTIONAL_AMBIENT_NAMES = frozenset({"tf_model", "tf_x"})


class _DocsEncoder(nn.Module):
    """Conv+ReLU encoder giving the canonical pages a real ``encoder`` module."""

    def __init__(self) -> None:
        """Initialize one padded convolution."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the ReLU-activated convolution of ``value``.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return torch.relu(self.conv(value))


class _DocsModel(nn.Module):
    """Fully-convolutional demo model satisfying the canonical-page ambient contract.

    The pages index ``relu_1_2``, query receptive/projective geometry at spatial
    position ``(10, 10)``, and select ``tl.in_module("encoder")``, so the model
    keeps a windowed (convolutional) path from input to output, an ``encoder``
    submodule, and spatial extents larger than the queried coordinates.
    """

    def __init__(self) -> None:
        """Initialize the encoder and a convolutional head."""

        super().__init__()
        self.encoder = _DocsEncoder()
        self.head = nn.Conv2d(4, 2, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the head applied to the encoded input.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        return self.head(self.encoder(value))


def _optional_tf_ambient() -> dict[str, Any]:
    """Build the TensorFlow-preview ambient names when the dependency is present.

    Returns
    -------
    dict[str, Any]
        ``tf_model``/``tf_x`` bindings, or an empty mapping when the TensorFlow
        preview prerequisites are unavailable.
    """

    if importlib.util.find_spec("tensorflow") is None:
        return {}
    try:
        import keras
        import tensorflow as tf
    except Exception:  # pragma: no cover - partial/broken optional install
        return {}
    if keras.backend.backend() != "tensorflow":  # pragma: no cover - env-specific
        return {}
    tf_model = keras.Sequential([keras.layers.Dense(3, activation="relu")])
    return {"tf_model": tf_model, "tf_x": tf.ones((2, 4))}


def _canonical_ambient() -> dict[str, Any]:
    """Return the shared execution namespace for one canonical page.

    Returns
    -------
    dict[str, Any]
        The declared ambient contract: ``model``/``x`` plus optional
        TensorFlow-preview names. Nothing else is injected.
    """

    torch.manual_seed(0)
    namespace: dict[str, Any] = {
        "model": _DocsModel().eval(),
        "x": torch.randn(1, 3, 16, 16),
    }
    namespace.update(_optional_tf_ambient())
    return namespace


@pytest.mark.heavy
@pytest.mark.parametrize("page_name", CANONICAL_PAGES)
def test_canonical_page_python_blocks_execute(
    page_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute every non-sketch Python fence of one canonical page as written.

    Fences run cumulatively in one shared namespace, statement by statement.
    The only tolerated non-executions are (a) fences whose first line carries an
    explicit ``API sketch`` comment marker (compile-gated) and (b) statements
    that read a declared OPTIONAL ambient name (``tf_model``/``tf_x``) on a box
    without the TensorFlow preview dependency. Any other failure is a
    documentation bug or a harness gap and fails the gate.

    Parameters
    ----------
    page_name:
        Repo-root markdown page name.
    tmp_path:
        Working directory for artifacts the snippets write (drawings, bundles).
    monkeypatch:
        Used to isolate the snippet working directory.
    """

    monkeypatch.chdir(tmp_path)
    page = _docs_dir().parent / page_name
    blocks = BLOCK_RE.findall(page.read_text(encoding="utf-8"))
    assert blocks, f"{page_name} has no python fences"
    namespace = _canonical_ambient()
    executed_statements = 0
    for block_index, code in enumerate(blocks, start=1):
        source_name = f"{page_name}:python-block-{block_index}"
        if _SKETCH_MARKER_RE.match(code.splitlines()[0]):
            compile(code, source_name, "exec")
            continue
        for node in ast.parse(code, filename=source_name).body:
            loads = {
                name.id
                for name in ast.walk(node)
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
            }
            missing_optional = {
                name for name in loads & _OPTIONAL_AMBIENT_NAMES if name not in namespace
            }
            if missing_optional:
                continue
            statement = ast.Module(body=[node], type_ignores=[])
            try:
                exec(compile(statement, source_name, "exec"), namespace)
            except (ImportError, ModuleNotFoundError) as exc:
                pytest.skip(f"{source_name} requires an unavailable optional dependency: {exc}")
            executed_statements += 1
    assert executed_statements > 0, f"{page_name}: nothing executed"


def test_collapse_reference_gallery_exists_and_is_regenerable() -> None:
    """Keep visual-reference links and the render-script manifest aligned."""

    from scripts.render_collapse_reference import DEFAULT_OUT_DIR, IMAGE_NAMES

    page = (_docs_dir() / "reference" / "collapse.md").read_text(encoding="utf-8")
    linked_names = tuple(re.findall(r"\.\./images/collapse/([^)]*\.svg)", page))
    assert linked_names == IMAGE_NAMES
    missing = [name for name in IMAGE_NAMES if not (DEFAULT_OUT_DIR / name).is_file()]
    assert not missing, f"Regenerate with scripts/render_collapse_reference.py: {missing}"


#: r7 R81 (opus b2 MED): pages with python fences OUTSIDE the executed set,
#: each with a reason. The executed gate covered 9 of 33 fence-bearing pages
#: with NO drift gate, so a new page (or a new fence on an old one) shipped
#: unexecuted and rotted silently -- the show(gradient=True)/RF-ordering rot
#: class. SHRINK-ONLY: move pages into DOC_FILES as they gain execution
#: coverage; a NEW page must either join DOC_FILES or take a reasoned row
#: here in the same change.
DOC_FENCE_EXEMPT: dict[str, str] = {
    "reference/stats.md": (
        "executed end-to-end by tests/test_unhide_surfaces.py::"
        "test_stats_doc_python_fences_execute (needs the ambient demo model)"
    ),
    "backends.md": "preview-backend snippets need tf/jax/mlx/tinygrad/paddle runtimes",
    "backward.md": "pending execution coverage (queued: backward capture snippets)",
    "buffers.md": "pending execution coverage",
    "containers.md": "pending execution coverage",
    "facets.md": "pending execution coverage (semantic recipes; heavier model deps)",
    "intervention_api.md": "pending execution coverage",
    "intervention_explainers.md": "pending execution coverage",
    "method_x_model_compatibility.md": "compatibility matrix stubs, not runnable programs",
    "migration/from_captum.md": "needs captum installed (weekly bridge leg env only)",
    "migration/from_fx.md": "pending execution coverage",
    "migration/from_nnsight.md": "needs nnsight (heavyweight, deliberately dark)",
    "migration/from_pyvene.md": "needs pyvene (not a declared extra)",
    "migration/from_thingsvision.md": "needs thingsvision (not a declared extra)",
    "reference/neuro.md": (
        "needs rsatoolbox (the neuro extra); every snippet's behavior is "
        "executed by tests/test_neuro_pkg_* at BOTH rsatoolbox versions"
    ),
    "migration/from_torchextractor.md": "needs torchextractor (bridge shim demo)",
    "migration/from_transformerlens.md": "needs transformer_lens (heavyweight)",
    "migration/v2.0_api_changes.md": "contains deliberate 'Before:' v1 blocks that must NOT run",
    "neuroai/byo_alignment.md": (
        "the file-round-trip blocks assume a completed real-checkpoint extraction and a "
        "user-owned learned transform; the offline mechanics (extract -> manifest -> "
        "torchlens-free reload) execute in tests/test_tvscope_export_acceptance.py"
    ),
    "neuroai/journey.md": (
        "journey cells name real checkpoints (ResNet50 V2 weights, CORnet-S via torch.hub) "
        "for the docs/nightly execution tier; the same mechanics execute offline against "
        "config-built fixtures in tests/test_tvscope_*.py"
    ),
    "neuroai/loaders.md": (
        "loader rows fetch real checkpoints (HF, timm pretrained, open_clip) by design; "
        "the resolver/adapter mechanics execute offline in "
        "tests/test_tvscope_preprocessing.py"
    ),
    "rank_layout.md": "pending execution coverage (graphviz layout demo)",
    "reference/checks_kit.md": (
        "executed top-to-bottom in one shared namespace by "
        "tests/test_checks_kit_docs.py (the fences build on each other, so the "
        "per-block gate's fresh-namespace model does not fit)"
    ),
    "receptive_projective_fields.md": "pending execution coverage",
    "skeleton_change_recipe.md": (
        "the fences are ONE sequential program (build models -> author/save recipe -> "
        "edit skeleton -> re-align), not independently runnable blocks; "
        "tests/test_skeleton_recipe_docs.py executes them top to bottom in one "
        "namespace on the real transformers GPT-2 classes and asserts the printed "
        "verdicts and diffs"
    ),
    "reference/hash.md": "pending execution coverage",
    "reference/episode_capture.md": (
        "the fence is a real usage example but is not self-contained (it references an "
        "ambient stepped `model`/`prompt_ids` the canonical contract does not provide) "
        "and episode capture is DIAGNOSTIC-TIER cost by the doc's own measurements "
        "(N=20 is 79 s on gpt2-124M CPU), so executing it would buy a slow test and no "
        "coverage the L2 episode suite does not already carry"
    ),
    "reference/limitations.md": "illustrative failure-mode fragments, not runnable programs",
    "reference/predicate_runtime.md": (
        "the fence is a Protocol DECLARATION sketch (`...` bodies, and `Protocol`/"
        "`RecordContext` deliberately unimported) -- there is no program to run; the "
        "shipped protocol itself is pinned by the predicate-registry suite"
    ),
    "reference/runnable_tlspec_contract.md": "contract fragments reference artifacts not in-repo",
    "semantic_io.md": "pending execution coverage (autoroute/facet demos)",
    "speed_optimized_defaults.md": "pending execution coverage",
    "visibility.md": "pending execution coverage",
    # -- FW2 closure sweep (pages added 2026-08-28 without a row) --------------
    "guides/offline_report.md": (
        "the one-line spelling over an ambient `log`; the same call executes with a "
        "real trace in reference/export.md (DOC_FILES) and tests/test_treescope_cards_report.py"
    ),
    "guides/treescope_cards.md": (
        "spelling sheet over an ambient `op` with a `...` scope body (no program); the "
        "bridge's register/status/display/disabled/unregister doors execute in "
        "tests/test_treescope_cards_bridge.py"
    ),
    "memory_debugging.md": (
        "torch-native recipes, not TorchLens API: a CUDA-only allocator-history sketch "
        "(refuses on CPU hosts, dumps a snapshot to cwd) and a process-global gc "
        "callback install (warn_tensor_cycles) that must not run in the shared test process"
    ),
    "monitor_training.md": (
        "training-loop sketches over an ambient HF-style model/optimizer/loader "
        "(`model(batch).loss`); the collector mechanics execute in "
        "tests/test_obs_substrate_*.py and tests/test_explorer_watch_*.py"
    ),
    "quickstart.md": (
        "the input ladder over ambient models incl. pretrained torchvision/HF checkpoint "
        "downloads (resnet18 IMAGENET1K_V1, an `lm`, a `clip`); the resolver executes "
        "offline in tests/test_quickstart_*.py"
    ),
    "reference/device_attribution.md": (
        "executed top-to-bottom in one shared namespace by "
        "tests/test_report_family_docs.py::test_every_docs_code_block_executes (the "
        "blocks build on one capture, so the per-block fresh-namespace model does not fit)"
    ),
    "reference/lit_bridge.md": (
        "needs lit_nlp (not in the core env) and a pretrained HF SST-2 checkpoint "
        "download, and ends in a blocking dev_server.serve(); the bridge executes in "
        "tests/test_lit_bridge_*.py"
    ),
    "reference/model_explorer.md": (
        "spelling sheet over an ambient trace whose serve() line needs model_explorer "
        "(not in the core env) and blocks; the export executes in reference/export.md "
        "(DOC_FILES) and tests/test_modelexplorer_*.py"
    ),
    "reference/netron_export.md": (
        "spelling sheet over an ambient trace whose open=True line serves a browser tab; "
        "the export executes in reference/export.md (DOC_FILES) and "
        "tests/test_netron_export_*.py"
    ),
    "reference/onebackward_reads.md": (
        "the read over an ambient language model (`seed(index=(0, -1, 464))` is a vocab "
        "position); the seed/read/top_k mechanics execute in tests/test_onebackward_*.py"
    ),
    "reference/trackers.md": (
        "an ambient AMP training loop (build_model_and_optimizer/loader/scaler) against a "
        "TensorBoard sink (tensorboard not in the core env); the sinks execute in "
        "tests/test_trackers_sinks_*.py"
    ),
    "reference/transforms.md": (
        "an extraction to disk over an ambient model/stimuli; the transform chain "
        "executes in tests/test_transforms_lib_*.py"
    ),
    "reference/tviz.md": (
        "real usage over an ambient HF model/tokenizer/attribution result plus an sklearn "
        "NMF recipe; the picture families execute in tests/test_tviz_*.py"
    ),
    "reference/weightsfree_capture.md": (
        "fetches a gated Llama-3.1-8B config from the Hub; the meta-substrate admission "
        "and parity gates execute offline in tests/test_weightsfree_*.py"
    ),
}


def test_every_fence_bearing_doc_page_is_executed_or_reason_exempt() -> None:
    """The executed-docs set is closed under new pages, both directions."""

    docs = _docs_dir()
    fence_pages = {
        page.relative_to(docs).as_posix()
        for page in sorted(docs.rglob("*.md"))
        if BLOCK_RE.search(page.read_text(encoding="utf-8"))
    }
    unaccounted = fence_pages - set(DOC_FILES) - set(DOC_FENCE_EXEMPT)
    assert not unaccounted, (
        "docs page(s) carry python fences but are neither executed by this "
        f"suite nor reason-exempt: {sorted(unaccounted)} -- add to DOC_FILES "
        "(preferred) or write a reasoned DOC_FENCE_EXEMPT row"
    )
    stale = (set(DOC_FENCE_EXEMPT) | set(DOC_FILES)) - fence_pages
    # Canonical non-docs pages (README/CLAUDE/AGENTS) are executed separately.
    stale -= {"performance.md", "for-ai-agents.md"}
    assert not stale.intersection(DOC_FENCE_EXEMPT), (
        f"stale DOC_FENCE_EXEMPT row(s): {sorted(stale & set(DOC_FENCE_EXEMPT))} "
        "-- the page lost its fences or moved; delete the row"
    )
