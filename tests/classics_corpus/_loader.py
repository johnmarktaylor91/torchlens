"""Load the classics corpus manifest and build its vendored models.

The corpus is a coverage-chosen sample of hand-built, trace-verified "classics"
(historical and unusual architectures written directly in PyTorch). Each model
file under ``models/`` is vendored byte-identical to its source and pinned by
sha256 in ``manifest.json``; see ``README.md`` beside this file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

CORPUS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
MODELS_DIR = CORPUS_DIR / "models"
MANIFEST_SCHEMA = "torchlens.classics_corpus.v1"
TIERS = ("smoke", "comprehensive")
BUILD_SEED = 0
INPUT_SEED = 1
VALIDATION_SEED = 0


@dataclass(frozen=True)
class CorpusEntry:
    """One model in the classics corpus.

    Attributes
    ----------
    id:
        Unique entry name (the classic's catalog name).
    module:
        Stem of the vendored file under ``models/``.
    build:
        Name of the zero-argument model constructor in that file.
    example_input:
        Name of the zero-argument input factory in that file.
    tier:
        ``"smoke"`` or ``"comprehensive"``.
    era:
        Catalog era code (E1 oldest to E7 newest).
    year:
        Publication year recorded by the source, possibly empty.
    features:
        Coverage features the census observed on this entry's trace.
    why:
        Why the selection kept this entry.
    """

    id: str
    module: str
    build: str
    example_input: str
    tier: str
    era: str
    year: str
    features: tuple[str, ...]
    why: str


@cache
def load_manifest() -> dict[str, Any]:
    """Return the parsed corpus manifest.

    Returns
    -------
    dict[str, Any]
        The manifest document.
    """

    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def corpus_entries(tier: str | None = None) -> tuple[CorpusEntry, ...]:
    """Return the corpus entries, optionally restricted to one tier.

    Parameters
    ----------
    tier:
        ``"smoke"``, ``"comprehensive"``, or ``None`` for every entry.

    Returns
    -------
    tuple[CorpusEntry, ...]
        Entries in manifest order.
    """

    entries = []
    for row in load_manifest()["entries"]:
        if tier is not None and row["tier"] != tier:
            continue
        entries.append(
            CorpusEntry(
                id=row["id"],
                module=row["module"],
                build=row["build"],
                example_input=row["example_input"],
                tier=row["tier"],
                era=row["era"],
                year=row["year"],
                features=tuple(row["features"]),
                why=row["why"],
            )
        )
    return tuple(entries)


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file.

    Parameters
    ----------
    path:
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model_module(module: str) -> ModuleType:
    """Import one vendored model file as a real module.

    A real module (registered in ``sys.modules``) keeps ``inspect.getsource``
    working for the model classes, as it does for any user model.

    Parameters
    ----------
    module:
        Stem of the file under ``models/``.

    Returns
    -------
    ModuleType
        The imported module.
    """

    name = f"torchlens_classics_corpus.{module}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODELS_DIR / f"{module}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load classics corpus module {module!r}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    try:
        spec.loader.exec_module(loaded)
    except BaseException:
        del sys.modules[name]
        raise
    return loaded


def build_entry(entry: CorpusEntry) -> tuple[torch.nn.Module, Any]:
    """Build an entry's model (in eval mode) and its example input, seeded.

    Parameters
    ----------
    entry:
        Corpus entry to build.

    Returns
    -------
    tuple[torch.nn.Module, Any]
        The model and the input passed to ``tl.trace`` as ``input_args``.
    """

    module = load_model_module(entry.module)
    torch.manual_seed(BUILD_SEED)
    model = getattr(module, entry.build)()
    model.eval()
    torch.manual_seed(INPUT_SEED)
    inputs = getattr(module, entry.example_input)()
    return model, inputs
