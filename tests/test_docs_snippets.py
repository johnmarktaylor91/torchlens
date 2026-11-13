"""Executable checks for documentation snippets.

Two gates live here:

* the per-block gate over the P2 ``docs/`` pages (``DOC_FILES``), and
* the canonical-page gate that EXECUTES every non-sketch Python fence in
  ``README.md`` plus AGENTS.md's former "Common Patterns" and "Architecture" code
  (``CLAUDE.md`` only imports AGENTS.md) top to bottom, statement by statement, in one
  shared namespace per page. The 2026-10-01 docs move ("docs: move agent reference
  material out of startup instructions") relocated those two fences out of AGENTS.md
  into ``docs/internal/agent-reference/common-patterns.md`` and
  ``docs/internal/agent-reference/architecture.md``, which this gate now reads directly;
  AGENTS.md itself carries no fences any more (just "Required reading" pointers).

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
from types import ModuleType
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
        "guides/fast_paths.md",
        "for-ai-agents.md",
        # FW2 closure sweep: the 2026-08-28 pages whose fences are
        # self-contained programs (spelling-only sketches on them are tagged
        # ```text).
        "native-torch.md",
        "field-guide/lrp_epsilon_litmus.md",
        "guides/grouping.md",
        # tl.preprocessing / tl.postprocessing / tl.facets registration walkthrough.
        "guides/model_conveniences.md",
        # The internals guide: one self-contained fence (loop grouping, a replayed edit).
        "how-it-works.md",
        # Backend protocol: the reference backend's tutorial (graduation G5).
        "backends/torch.md",
        # Agent-reference pages whose fences are self-contained programs.
        "internal/agent-reference/current-2-x-surface.md",
        "internal/agent-reference/feature-inspection.md",
    ),
    "reference_tooling": (
        "reference/debug.md",
        "reference/export.md",
        "reference/harness_bundles.md",
    ),
    "reference_views": (
        "reference/attribution.md",
        "reference/collapse.md",
        "reference/lenses.md",
        "reference/mechinterp.md",
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


#: TorchLens entry points that need an optional extra the docs demonstrate, with
#: the module each one imports lazily. A doc fence that calls one is skipped by
#: feature detect when the module is absent; a new extra-gated entry point in the
#: docs must be added here, or its fence fails loudly.
_OPTIONAL_EXTRA_CALLS: dict[str, str] = {
    "tl.export.wandb": "wandb",
    "tl.export.xarray": "xarray",
}


def _missing_extra_modules(code: str) -> list[str]:
    """Return modules the code needs through an extra-gated TorchLens call."""

    return sorted(
        module
        for call, module in _OPTIONAL_EXTRA_CALLS.items()
        if call in code and importlib.util.find_spec(module) is None
    )


def _missing_imported_modules(tree: ast.AST) -> list[str]:
    """Return the root modules the code imports that are not installed.

    A doc example may demonstrate an OPTIONAL-dependency feature that is not
    installed in every test env. This feature detect runs BEFORE execution, so a
    missing extra is a visible skip with its module named, while any ImportError
    raised during execution (a broken import path, a torchlens module that moved)
    fails loudly instead of skipping.
    """

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    return sorted(root for root in roots if importlib.util.find_spec(root) is None)


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
    compiled = compile(code, synthetic_filename, "exec")
    missing = _missing_imported_modules(ast.parse(code)) + _missing_extra_modules(code)
    if missing:
        pytest.skip(f"doc snippet imports uninstalled optional dependencies: {missing}")
    from tests.support.capability_routes import capability_route

    # A fence needing a torch surface this build lacks (one-backward reads without
    # GradientEdge) stops at its named refusal; any other exception fails here.
    with capability_route():
        exec(compiled, namespace)


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


#: AGENTS.md's own fences moved during the 2026-10-01 docs move (see the module
#: docstring); these two pages carry the content "AGENTS.md" used to name here.
CANONICAL_PAGES = (
    "README.md",
    "docs/index.md",
    "docs/internal/agent-reference/common-patterns.md",
    "docs/internal/agent-reference/architecture.md",
)
_SKETCH_MARKER_RE = re.compile(r"^\s*#.*\bAPI sketch\b", re.IGNORECASE)

#: README fences are advertisements: one fence may mix setups or lean on names it
#: never defines. The README's setup lives here, in the test, never in README
#: markers. ``README_PRELUDE`` supplies the imports the fences elide (the opening
#: ViT splash fence runs before any fence imports ``tl`` or the ViT class; the loop
#: fence uses ``torch`` and ``nn``; the raw-PyTorch comparisons use
#: ``AutoImageProcessor`` and ``Image``; the GoogLeNet fence uses ``googlenet``).
#: ``README_FIXTURES`` defines the inputs the fences name but never build: the
#: raw-PyTorch comparisons' GPT-2 tokenizer ``tok``, preprocessed ResNet input
#: ``image`` and ImageNet label list ``cats``, the RDM fence's three photos
#: ``cat1``, ``cat2`` and ``car``, and the Extract fence's second GPT-2 input ``new_text``.
#: That fence's mixture of experts ``moe`` and its ``tokens`` are the learning path's
#: ``tiny_moe()`` (``notebooks/learn/learn_models.py``, loaded by ``_readme_setup``), the
#: model of the notebook the README links for it. Every photo (``cat.jpg`` included) is a plain
#: synthetic image the test writes; the splash's printed labels and probabilities
#: are gated on the real, checksum-pinned image by tests/test_quickstart_readme_gate.py.
README_PRELUDE = (
    "import torch\n"
    "from torch import nn\n"
    "import torchlens as tl\n"
    "from PIL import Image\n"
    "from torchvision.models import ResNet18_Weights, googlenet\n"
    "from transformers import AutoImageProcessor, AutoTokenizer, ViTForImageClassification\n"
)
README_FIXTURES = (
    'tok = AutoTokenizer.from_pretrained("gpt2")\n'
    'image = ResNet18_Weights.IMAGENET1K_V1.transforms()(Image.open("cat.jpg")).unsqueeze(0)\n'
    'cats = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]\n'
    "cat1, cat2, car = (\n"
    '    Image.new("RGB", (224, 224), color) for color in ((140, 110, 80), (90, 90, 90), (200, 30, 30))\n'
    ")\n"
    'new_text = "The capital of Germany is"\n'
)
#: README comparison fences written for another library. Each leans on that
#: library's own model wrapper, which the README never builds (``vit.trace(...)``
#: and ``model.trace(...)`` need nnsight's wrapper, ``model.run_with_cache`` a
#: TransformerLens HookedTransformer), so, like an ``API sketch`` fence, each is
#: compiled, not run. A row is (a line unique to the fence, the library it is
#: written for); ``test_readme_foreign_fences_are_live_and_foreign`` keeps every
#: row matched to exactly one fence that never calls TorchLens.
README_FOREIGN_FENCES: tuple[tuple[str, str], ...] = (
    ('with vit.trace(Image.open("cat.jpg")):', "nnsight"),
    ("net = NNsight(resnet)", "nnsight"),
    ("_, france_cache = model.run_with_cache(france)", "transformer_lens"),
    ('with model.trace("The capital of France is"):', "nnsight"),
)
#: README statements that need a model the page's shared namespace does not hold
#: (the ResNet-18 line in the GPT-2 selection fence). Each is compiled, not run,
#: and names the test that executes the same API spelling for real:
#: (statement as written, covering test file, spelling the covering test must use).
README_ILLUSTRATIVE_STATEMENTS: tuple[tuple[str, str, str], ...] = (
    (
        'trace["conv1"].projective_field.of(units, within=["layer1", "layer2"])',
        "tests/test_field_view_of_resnet.py",
        'trace["conv1"].projective_field.of(',
    ),
)
#: README statements the page test runs right after a fixed torch seed:
#: (statement as written, seed). The loop fence's pass lookup and branch outcome
#: (``trace["relu_1_5:2"]``, ``conditionals[0].outcome``) exist only when its random
#: input takes the branch; seed 2 before ``loop = Loop()`` is the run the README's
#: loop pictures and comments come from (``x.sum() < 0``, branch taken).
README_SEEDED_STATEMENTS: tuple[tuple[str, int], ...] = (("loop = Loop()", 2),)
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


def _readme_setup(namespace: dict[str, Any], workdir: Path) -> None:
    """Prepare the README's shared namespace and working directory (see README_PRELUDE)."""

    # The fences need Pillow for their photos, torchvision for the ResNet and GoogLeNet
    # fences and transformers for the GPT-2 and ViT models, the same conditions the
    # fences' own imports have.
    required = ("PIL", "torchvision", "transformers")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"README.md imports uninstalled optional dependencies: {missing}")
    from PIL import Image

    Image.new("RGB", (256, 256), (140, 110, 80)).save(workdir / "cat.jpg")
    exec(compile(README_PRELUDE, "README.md:prelude", "exec"), namespace)
    exec(compile(README_FIXTURES, "README.md:fixtures", "exec"), namespace)
    namespace["moe"], namespace["tokens"] = _learn_models().tiny_moe()


def _learn_models() -> ModuleType:
    """Load ``notebooks/learn/learn_models.py`` from its file, without touching ``sys.path``."""

    path = _docs_dir().parent / "notebooks" / "learn" / "learn_models.py"
    spec = importlib.util.spec_from_file_location("_readme_learn_models", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_readme_foreign_fence(page_name: str, code: str) -> bool:
    """Return whether one README fence is in README_FOREIGN_FENCES."""

    return page_name == "README.md" and any(
        marker in _code_lines(code) for marker, _ in README_FOREIGN_FENCES
    )


def _code_lines(code: str) -> set[str]:
    """Return a fence's lines with indentation and trailing comments stripped."""

    return {line.split("#")[0].strip() for line in code.splitlines()}


def _is_readme_illustrative(page_name: str, node: ast.stmt) -> bool:
    """Return whether one README statement is in README_ILLUSTRATIVE_STATEMENTS."""

    return page_name == "README.md" and any(
        ast.unparse(node) == ast.unparse(ast.parse(statement).body[0])
        for statement, _, _ in README_ILLUSTRATIVE_STATEMENTS
    )


def _readme_seed(page_name: str, node: ast.stmt) -> int | None:
    """Return the torch seed README_SEEDED_STATEMENTS sets before one README statement."""

    if page_name != "README.md":
        return None
    for statement, seed in README_SEEDED_STATEMENTS:
        if ast.unparse(node) == ast.unparse(ast.parse(statement).body[0]):
            return seed
    return None


@pytest.mark.heavy
@pytest.mark.parametrize(
    "page_name",
    # The README runs real pretrained GPT-2 and ViT fences (a 144-run patching sweep included);
    # the docs home page runs the README's pretrained GPT-2 fence.
    [
        pytest.param(p, marks=pytest.mark.slow) if p in {"README.md", "docs/index.md"} else p
        for p in CANONICAL_PAGES
    ],
)
def test_canonical_page_python_blocks_execute(
    page_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute every non-sketch Python fence of one canonical page as written.

    Fences run cumulatively in one shared namespace, statement by statement.
    The only tolerated non-executions are (a) fences whose first line carries an
    explicit ``API sketch`` comment marker (compile-gated), (b) statements
    that read a declared OPTIONAL ambient name (``tf_model``/``tf_x``) on a box
    without the TensorFlow preview dependency and (c) the README statements in
    ``README_ILLUSTRATIVE_STATEMENTS`` (compiled; each names the test that runs
    its spelling). Any other failure is a documentation bug or a harness gap
    and fails the gate.

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
    if page_name == "README.md":
        _readme_setup(namespace, tmp_path)
    executed_statements = 0
    for block_index, code in enumerate(blocks, start=1):
        source_name = f"{page_name}:python-block-{block_index}"
        # Register the fence's text as a notebook cell is registered (and as the guide
        # runner above does), so capture reads its source the way it reads a reader's
        # file or cell: the README loop fence's ``if`` becomes a recorded conditional.
        monkeypatch.setitem(
            linecache.cache,
            source_name,
            (len(code), None, [f"{line}\n" for line in code.splitlines()], source_name),
        )
        if _SKETCH_MARKER_RE.match(code.splitlines()[0]) or _is_readme_foreign_fence(
            page_name, code
        ):
            compile(code, source_name, "exec")
            continue
        for node in ast.parse(code, filename=source_name).body:
            if _is_readme_illustrative(page_name, node):
                continue
            seed = _readme_seed(page_name, node)
            if seed is not None:
                torch.manual_seed(seed)
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
            missing = _missing_imported_modules(statement) + _missing_extra_modules(
                ast.unparse(statement)
            )
            if missing:
                pytest.skip(f"{source_name} imports uninstalled optional dependencies: {missing}")
            exec(compile(statement, source_name, "exec"), namespace)
            executed_statements += 1
    assert executed_statements > 0, f"{page_name}: nothing executed"


def test_readme_foreign_fences_are_live_and_foreign() -> None:
    """Every README fence the page test only compiles is still there and never calls TorchLens.

    A row whose marker line left the README is stale; a row matching two fences, or
    a fence that uses ``tl.``, would hide TorchLens code from the page test.
    """

    fences = BLOCK_RE.findall((_docs_dir().parent / "README.md").read_text(encoding="utf-8"))
    for marker, library in README_FOREIGN_FENCES:
        matched = [code for code in fences if marker in _code_lines(code)]
        assert len(matched) == 1, (
            f"README_FOREIGN_FENCES row {marker!r} ({library}) matches {len(matched)} fences, not 1"
        )
        assert "tl." not in matched[0] and "torchlens" not in matched[0], (
            f"the {library} fence marked by {marker!r} calls TorchLens: run it, do not exempt it"
        )


def test_readme_seeded_statements_are_live() -> None:
    """Every README_SEEDED_STATEMENTS row still names exactly one README statement."""

    statements = [
        ast.unparse(node)
        for code in BLOCK_RE.findall((_docs_dir().parent / "README.md").read_text(encoding="utf-8"))
        for node in ast.parse(code).body
    ]
    for statement, _seed in README_SEEDED_STATEMENTS:
        wanted = ast.unparse(ast.parse(statement).body[0])
        assert statements.count(wanted) == 1, (
            f"README_SEEDED_STATEMENTS row {statement!r} matches {statements.count(wanted)} statements"
        )


def test_readme_illustrative_statements_are_live_and_covered() -> None:
    """Every README statement the page test skips is still in the README and covered.

    A row whose statement left the README is stale; a row whose covering test
    no longer exercises the same API spelling would leave that spelling unrun.
    """

    root = _docs_dir().parent
    readme_statements = {
        ast.unparse(node)
        for code in BLOCK_RE.findall((root / "README.md").read_text(encoding="utf-8"))
        for node in ast.parse(code).body
    }
    for statement, covering_test, spelling in README_ILLUSTRATIVE_STATEMENTS:
        assert ast.unparse(ast.parse(statement).body[0]) in readme_statements, (
            f"stale README_ILLUSTRATIVE_STATEMENTS row: {statement!r} is no longer in README.md"
        )
        assert spelling in statement, f"{statement!r} does not use its spelling {spelling!r}"
        covering = root / covering_test
        assert covering.is_file(), f"{covering_test} no longer exists"
        assert spelling in covering.read_text(encoding="utf-8"), (
            f"{covering_test} no longer runs {spelling!r}"
        )


#: Peer libraries the nightly peer leg installs beside TorchLens (the recipe
#: pages' tools plus the optional-extra fences). A peer that patches torch at
#: import time AFTER the first capture (nnsight re-wraps ``torch.Tensor.backward``
#: with ``functools.wraps`` around whatever is installed) once made
#: ``unwrap_torch`` raise its burial warning under the suite's
#: warnings-as-errors filter before the decorated flag flipped, so every later
#: snippet logged only module boundaries: 27 reds that appeared only with the
#: peers present. ``test_docs_snippets_pass_set_matches_with_peer_imported`` is
#: the bare-vs-peers parity guard: each installed peer runs the failing shape in
#: a subprocess under the same filters this suite runs with.
PEER_MODULES: tuple[str, ...] = (
    "cornet",
    "dictionary_learning",
    "himalaya",
    "lxt",
    "matplotlib",
    "netrep",
    "nnsight",
    "open_clip",
    "quantus",
    "sae_lens",
    "sparsify",
    "timm",
    "torchcam",
    "torchvision",
    "transformers",
    "tuned_lens",
    "vit_prisma",
    "wandb",
    "weightwatcher",
    "xarray",
)

_PEER_PARITY_PROBE = r"""
import importlib
import sys
import warnings

warnings.filterwarnings("error", category=UserWarning, module=r"torchlens(\.|$)")
warnings.filterwarnings("error", category=DeprecationWarning, module=r"torchlens(\.|$)")

import torch
from torch import nn

import torchlens as tl

model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
x = torch.ones(2, 4)
bare = [op.func_name for op in tl.trace(model, x)]
assert bare[1:-1] == ["linear", "relu", "linear"], bare

importlib.import_module(sys.argv[1])

# backends/torch.md fence 1: validate unwraps to pristine torch and re-wraps.
assert tl.validate(model, x, scope="forward", validate_metadata=False) is True
after = [op.func_name for op in tl.trace(model, x)]
assert after == bare, (bare, after)
assert tl.validate(model, x, scope="forward", validate_metadata=False) is True
# The peer's own patch (if any) still runs outside a capture.
model(x).sum().backward()
print("PARITY_OK")
"""


@pytest.mark.heavy
@pytest.mark.parametrize("peer", PEER_MODULES)
def test_docs_snippets_pass_set_matches_with_peer_imported(peer: str) -> None:
    """The snippet suite's pass set is the same with ``peer`` imported mid-process.

    Runs in a subprocess so the peer's import-time torch patches land on a
    process that has already captured once, the order the combined nightly
    session produces, under this suite's warnings-as-errors filters.
    """

    import subprocess
    import sys

    if importlib.util.find_spec(peer) is None:
        pytest.skip(f"peer library {peer} is not installed")
    completed = subprocess.run(
        [sys.executable, "-c", _PEER_PARITY_PROBE, peer],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0 and "PARITY_OK" in completed.stdout, (
        f"{peer}: rc={completed.returncode}\n{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}"
    )


def test_collapse_reference_gallery_exists_and_is_regenerable() -> None:
    """Keep visual-reference links and the render-script manifest aligned."""

    from scripts.render_collapse_reference import DEFAULT_OUT_DIR, IMAGE_NAMES

    page = (_docs_dir() / "reference" / "collapse.md").read_text(encoding="utf-8")
    linked_names = tuple(re.findall(r"\.\./images/collapse/([^)]*\.svg)", page))
    assert linked_names == IMAGE_NAMES
    missing = [name for name in IMAGE_NAMES if not (DEFAULT_OUT_DIR / name).is_file()]
    assert not missing, f"Regenerate with scripts/render_collapse_reference.py: {missing}"


#: Reason shared by the coming-from pages whose every fence is headed and run.
_MIGRATION_PAGES_EXECUTED = (
    "executed fence by fence by tests/test_migration_pages_executed.py (the TorchLens "
    "fences always; a third-party fence where that package is installed)"
)
#: Reason shared by the coming-from pages run by tests/test_migrations.py.
_MIGRATIONS_TEST_EXECUTED = (
    "executed fence by fence by tests/test_migrations.py (the TorchLens fences always; a "
    "third-party fence where that package is installed)"
)
#: Reason shared by the teaching pages whose every fence is headed and run.
_TEACHING_PAGES_EXECUTED = (
    "executed fence by fence by tests/test_teaching_pages_executed.py, each against its "
    "declared result"
)

#: Reason shared by the recipe pages whose every fence is headed and run.
_RECIPE_PAGES_EXECUTED = (
    "executed fence by fence by tests/test_recipe_pages_executed.py (a TorchLens-only "
    "page always; a third-party page where that package is installed, and always in "
    "the peer-recipe leg)"
)

#: Reason shared by the launch docs pages (Field guide, strip, comparison).
_LAUNCH_PAGES_EXECUTED = (
    "executed by tests/test_launch_docs_pages.py (a Field guide page as one session in "
    "order; each strip fence alone, skipped where its declared package is absent)"
)

#: r7 R81 (opus b2 MED): pages with python fences OUTSIDE the executed set,
#: each with a reason. The executed gate covered 9 of 33 fence-bearing pages
#: with NO drift gate, so a new page (or a new fence on an old one) shipped
#: unexecuted and rotted silently -- the show(grad=True)/RF-ordering rot
#: class. SHRINK-ONLY: move pages into DOC_FILES as they gain execution
#: coverage; a NEW page must either join DOC_FILES or take a reasoned row
#: here in the same change.
DOC_FENCE_EXEMPT: dict[str, str] = {
    # 2026-10-01 docs move ("docs: move agent reference material out of startup
    # instructions"): these two pages carry the fences the canonical AGENTS.md gate
    # used to execute directly; CANONICAL_PAGES now reads them by this path, so they
    # are covered there, not by the per-block DOC_FILES mechanism.
    "internal/agent-reference/common-patterns.md": (
        "executed end-to-end by test_canonical_page_python_blocks_execute "
        "(moved from AGENTS.md's Common Patterns section)"
    ),
    "internal/agent-reference/architecture.md": (
        "executed end-to-end by test_canonical_page_python_blocks_execute "
        "(moved from AGENTS.md's Architecture section's "
        "'Common unified capture patterns' fence)"
    ),
    # Same docs move relocated torchlens/AGENTS.md's "Current 2.x Surface" prose
    # (never itself part of CANONICAL_PAGES or DOC_FILES, so never executed) into
    # docs/, where this lint sees it for the first time. The fences are illustrative
    # sketches over backends/inputs the canonical ambient does not provide
    # (paddle_model/paddle_x, raw_text/text_to_tensor, block1/block2 submodules).
    "internal/agent-reference/package/public-surface.md": (
        "illustrative surface-listing fragments (paddle/raw_text/block1-block2 "
        "ambient names undefined), not runnable programs"
    ),
    "internal/agent-reference/package/architecture-overview.md": (
        "illustrative surface-listing fragments (paddle/raw_text/block1-block2 "
        "ambient names undefined), not runnable programs"
    ),
    "reference/stats.md": (
        "executed end-to-end by tests/test_unhide_surfaces.py::"
        "test_stats_doc_python_fences_execute (needs the ambient demo model)"
    ),
    "reference/selection_completeness.md": (
        "executed end-to-end by tests/test_selection_completeness_examples.py::"
        "test_page_examples_run_as_written (Section 5's four fences are one script run in "
        "order; each later fence reads the first fence's trace and helpers)"
    ),
    "backends.md": "preview-backend snippets need tf/jax/mlx/tinygrad/paddle runtimes",
    "bridges.md": (
        "bridge fences share the first fence's resnet18 trace and need optional "
        "third-party runtimes (captum, grad-cam, rsatoolbox, brain-score) plus a "
        "network call (from_hub); claims covered by tests/test_bridge_real_captum.py, "
        "tests/test_b2_bridge_captum.py and tests/test_b2_bridge_hub.py"
    ),
    "backends/tf.md": (
        "tutorial fence executed by tests/backends/test_tf_docs_page.py under the "
        "tf_backend marker (needs the tensorflow runtime)"
    ),
    "backends/jax.md": (
        "tutorial fence executed by tests/test_jax_docs_page.py under the backend_jax "
        "marker (needs the jax runtime)"
    ),
    "backends/mlx.md": (
        "tutorial fences executed by tests/test_backend_docs_lockstep.py under the "
        "backend_mlx marker (needs the mlx runtime: macOS, or Linux with "
        "TORCHLENS_RUN_UNSTABLE_MLX_TESTS=1)"
    ),
    "backends/paddle.md": (
        "tutorial fences executed by tests/test_paddle_docs_tutorial.py and "
        "tests/test_backend_docs_lockstep.py under the backend_paddle marker "
        "(needs the paddle runtime)"
    ),
    "backends/tinygrad.md": (
        "tutorial fences executed by tests/test_backend_docs_lockstep.py under the "
        "backend_tinygrad marker (needs the tinygrad runtime)"
    ),
    "backward.md": "pending execution coverage (queued: backward capture snippets)",
    "buffers.md": "pending execution coverage",
    "containers.md": "pending execution coverage",
    "facets.md": "pending execution coverage (semantic recipes; heavier model deps)",
    "intervention_api.md": "pending execution coverage",
    "method_x_model_compatibility.md": "compatibility matrix stubs, not runnable programs",
    "migration/from_captum.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/from_fx.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/from_hooks.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/from_torchinfo.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/coming_from_torchsnooper.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/cross_engine.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/from_nnsight.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/from_thingsvision.md": _MIGRATIONS_TEST_EXECUTED,
    "reference/neuro.md": (
        "needs rsatoolbox (the neuro extra); every snippet's behavior is "
        "executed by tests/test_neuro_pkg_* at BOTH rsatoolbox versions"
    ),
    "migration/from_torchextractor.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/from_transformerlens.md": _MIGRATIONS_TEST_EXECUTED,
    "migration/v2.0_api_changes.md": "contains deliberate 'Before:' v1 blocks that must NOT run",
    "migration/from_bertviz_ecco.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_hf_kwargs.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_intermediate_layer_getter.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_model_zoos.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_surgeon_pytorch.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_timm_features_only.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_torchview.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_torchvision_feature_extraction.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_torchviz_hiddenlayer.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_transformers_interpret.md": _MIGRATION_PAGES_EXECUTED,
    "migration/multi_gpu_extraction.md": _MIGRATION_PAGES_EXECUTED,
    "migration/penzai.md": _MIGRATION_PAGES_EXECUTED,
    "migration/saelens.md": _MIGRATION_PAGES_EXECUTED,
    "migration/lovely_stats_line.md": _MIGRATION_PAGES_EXECUTED,
    "migration/inseq.md": _MIGRATION_PAGES_EXECUTED,
    "migration/repe.md": _MIGRATION_PAGES_EXECUTED,
    "migration/autocircuit.md": _MIGRATION_PAGES_EXECUTED,
    "migration/baukit.md": _MIGRATION_PAGES_EXECUTED,
    "migration/from_flax_tabulate.md": _MIGRATION_PAGES_EXECUTED,
    "field-guide/acdc_ioi.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/lrp_transformers_lxt.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/seeded_subsample.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/training_ribbon.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/tcav.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/linear_probes.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/craft_concepts.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/das_gpt2.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/entropy_steering.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/generation_step_attribution.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/generation_attention.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/tensor_images.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/trainable_backbone.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/tuned_lens.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/jacobian_lens.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/conditional_steering.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/weightwatcher.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/torchcam.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/quantus.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/sae_trainers.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/prisma_clip_sae.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/sae_crosscoder_diff.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/sae_latent_export.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/pretrained_saes.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/feature_visualisation.md": _RECIPE_PAGES_EXECUTED,
    "field-guide/dissection.md": _RECIPE_PAGES_EXECUTED,
    "neuroai/netrep.md": _RECIPE_PAGES_EXECUTED,
    "neuroai/himalaya.md": _RECIPE_PAGES_EXECUTED,
    "neuroai/task_rnn_circuits.md": _RECIPE_PAGES_EXECUTED,
    "neuroai/protein_language_model.md": _RECIPE_PAGES_EXECUTED,
    "gallery/causal_tracing.md": _RECIPE_PAGES_EXECUTED,
    "gallery/cnn_fields.md": _RECIPE_PAGES_EXECUTED,
    "gallery/neuro_rdms.md": _RECIPE_PAGES_EXECUTED,
    "gallery/ioi_circuit.md": _RECIPE_PAGES_EXECUTED,
    "gallery/bundles_harness.md": _RECIPE_PAGES_EXECUTED,
    "gallery/jacobian_lens.md": _RECIPE_PAGES_EXECUTED,
    "neuroai/byo_alignment.md": (
        "the file-round-trip blocks assume a completed real-checkpoint extraction and a "
        "user-owned learned transform; the offline mechanics (extract -> manifest -> "
        "torchlens-free reload) execute in tests/test_tvscope_export_acceptance.py"
    ),
    "neuroai/journey.md": (
        "journey cells name real checkpoints (ResNet50 V2 weights) for the docs/nightly "
        "execution tier; the same mechanics execute offline against config-built fixtures "
        "in tests/test_tvscope_*.py, and the CORnet-S showpiece (pinned checkout, direct "
        "import) runs in tests/test_recipe_pages_executed.py"
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
        "TensorBoard sink (tensorboard not in the core env), and a Lightning Trainer "
        "(lightning not in the core env); the sinks execute in "
        "tests/test_trackers_sinks_*.py and SummaryCallback in tests/test_callbacks_lightning.py"
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
    # Teaching pages and the pyvene migration pages (teaching-pass rows): every fence
    # on them is headed, and the named test runs every TorchLens fence.
    "intervention_explainers.md": _TEACHING_PAGES_EXECUTED,
    "stochastic_edits.md": _TEACHING_PAGES_EXECUTED,
    "guides/kv_cache.md": _TEACHING_PAGES_EXECUTED,
    "sae_training_handoff.md": _TEACHING_PAGES_EXECUTED,
    "gradient_contract.md": _TEACHING_PAGES_EXECUTED,
    "migration/from_pyvene.md": _MIGRATIONS_TEST_EXECUTED,
    "field-guide/running_examples.md": _LAUNCH_PAGES_EXECUTED,
    "field-guide/see_and_inspect.md": _LAUNCH_PAGES_EXECUTED,
    "field-guide/extract_features.md": _LAUNCH_PAGES_EXECUTED,
    "field-guide/intervene.md": _LAUNCH_PAGES_EXECUTED,
    "gallery/any_architecture.md": _LAUNCH_PAGES_EXECUTED,
    "migration/comparison.md": _LAUNCH_PAGES_EXECUTED,
    # Conditional attribution reads the model's source file (tokenize.open), and
    # this suite execs fences from memory, so an executed copy records no conditional.
    "conditional_attribution.md": (
        "the conditional indexer needs the model's source on disk; the same "
        "early-return shape and its fired_arm_kind and outcome asserts are "
        "pinned in tests/test_conditional_fallthrough_outcome.py"
    ),
    "index.md": (
        "executed end-to-end by test_canonical_page_python_blocks_execute[docs/index.md] "
        "(slow: the README's pretrained GPT-2 fence)"
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


#: Fences heavy-class by measurement (beyond the cold ``torch.compile`` fences),
#: keyed by page and a line unique to the fence, since block indexes shift as
#: pages grow. Measured single-threaded and serial on the gate worker at a
#: 2.1-2.2x slowdown factor, each charged 10.0-10.7 s (about 4.7 s at 1x) against
#: the reference-views family's 23-24 s aggregate 5s-tier budget there, while
#: every other fence in the family stayed under 6.2 s (under 0.7 s for all but two).
MEASURED_HEAVY_FENCES: frozenset[tuple[str, str]] = frozenset(
    {
        # The component-graph example patches a circuit into every receiver of a
        # two-layer GPT-2 in one patch_edges call (the same fence is the heavy
        # tests/test_circ1_component_graph.py::test_reference_doc_example_runs).
        ("reference/mechinterp.md", "# A circuit runs as one patch_edges call"),
        # edge_scores on a one-layer GPT-2, then a causal_audit of its six
        # largest edges with random draws.
        ("reference/attribution.md", "largest = sorted(claim.rows()"),
    }
)


def _is_measured_heavy_fence(file_name: str, code: str) -> bool:
    """Return whether one docs fence is in ``MEASURED_HEAVY_FENCES``."""

    return any(page == file_name and line in code for page, line in MEASURED_HEAVY_FENCES)


def _classify_measured_heavy_fences(*tests: Any) -> None:
    """Mark each family's measured-heavy fences HEAVY at collection.

    The page-group families build their params when they are decorated; this
    adds the ``heavy`` mark to the measured fences' params in place, so the
    mark selects (``-m "not heavy"`` deselects them) and resolves the 20s tier,
    which takes them out of their family's aggregate 5s-tier budget.
    """

    for test in tests:
        for mark in test.pytestmark:
            if mark.name != "parametrize":
                continue
            params = mark.args[1]
            params[:] = [
                pytest.param(*param.values, marks=(*param.marks, pytest.mark.heavy), id=param.id)
                if _is_measured_heavy_fence(param.values[0], param.values[2])
                else param
                for param in params
            ]


_classify_measured_heavy_fences(
    test_doc_python_blocks_run_guides,
    test_doc_python_blocks_run_reference_tooling,
    test_doc_python_blocks_run_reference_views,
)


def test_every_measured_heavy_fence_names_exactly_one_executed_fence() -> None:
    """A measured-heavy row that stops matching would leave its fence unclassified."""

    blocks = _iter_python_blocks()
    for page, line in sorted(MEASURED_HEAVY_FENCES):
        matches = [index for name, index, code in blocks if name == page and line in code]
        assert len(matches) == 1, f"{page}: {line!r} matches fences {matches}, expected one"
