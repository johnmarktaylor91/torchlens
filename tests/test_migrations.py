"""Executable checks for functional migration examples."""

from __future__ import annotations

import ast
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import linecache
import re
from pathlib import Path
from typing import Any

import pytest
import torch


MIGRATION_FILES = (
    "from_nnsight.md",
    "from_transformerlens.md",
    "from_captum.md",
    "from_thingsvision.md",
    "from_pyvene.md",
    "from_torchextractor.md",
    "from_fx.md",
)

OPTIONAL_IMPORTS = {
    "captum": "captum",
    "fx": "torch.fx",
    "nnsight": "nnsight",
    "pyvene": "pyvene",
    "thingsvision": "thingsvision",
    "torchextractor": "torchextractor",
    "torchlens": "torchlens",
    "transformer_lens": "transformer_lens",
}

HEADER_RE = re.compile(r"# migration-test: tool=(?P<tool>[a-zA-Z0-9_]+) expected=(?P<expected>.+)")
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def _migration_dir() -> Path:
    """Return the migration documentation directory.

    Returns
    -------
    Path
        Absolute path to ``docs/migration``.
    """

    return Path(__file__).resolve().parents[1] / "docs" / "migration"


def _iter_examples() -> list[tuple[str, str, Any, str]]:
    """Collect executable migration examples from Markdown code blocks.

    Returns
    -------
    list[tuple[str, str, Any, str]]
        Tuples of ``(file_name, tool, expected, code)``.
    """

    examples: list[tuple[str, str, Any, str]] = []
    for file_name in MIGRATION_FILES:
        text = (_migration_dir() / file_name).read_text(encoding="utf-8")
        for match in BLOCK_RE.finditer(text):
            code = match.group("code")
            first_line = code.splitlines()[0]
            header = HEADER_RE.match(first_line)
            if header is None:
                continue
            expected = ast.literal_eval(header.group("expected"))
            examples.append((file_name, header.group("tool"), expected, code))
    return examples


def _skip_incompatible_optional_example(file_name: str, tool: str) -> None:
    """Skip optional examples whose installed dependency is known incompatible.

    Parameters
    ----------
    file_name:
        Migration Markdown file name.
    tool:
        Tool key from the migration-test header.
    """
    if file_name == "from_nnsight.md" and tool == "nnsight":
        try:
            installed = version("nnsight")
        except PackageNotFoundError:
            return
        major, minor, *_ = installed.split(".")
        if int(major) > 0 or int(minor) >= 7:
            pytest.skip("nnsight>=0.7 changed LanguageModel trace input semantics")
    if file_name == "from_transformerlens.md" and tool == "transformer_lens":
        pytest.skip(
            "transformer_lens pretrained tiny-stories-1M cache values vary by installed weights/version"
        )


@pytest.mark.parametrize(("file_name", "tool", "expected", "code"), _iter_examples())
def test_migration_example_runs(file_name: str, tool: str, expected: Any, code: str) -> None:
    """Run one migration code block and verify its declared output."""

    module_name = OPTIONAL_IMPORTS[tool]
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"{file_name} requires optional dependency {module_name!r}")
    _skip_incompatible_optional_example(file_name, tool)

    synthetic_filename = f"{file_name}:{tool}"
    linecache.cache[synthetic_filename] = (
        len(code),
        None,
        [f"{line}\n" for line in code.splitlines()],
        synthetic_filename,
    )
    namespace: dict[str, Any] = {
        "__file__": synthetic_filename,
        "__name__": f"migration_example_{tool}",
    }
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        exec(compile(code, synthetic_filename, "exec"), namespace)
    finally:
        torch.use_deterministic_algorithms(deterministic_enabled)
    assert namespace["RESULT"] == expected


def test_each_migration_file_has_old_and_torchlens_examples() -> None:
    """Every Phase 15 migration file should contain both sides of the migration."""

    by_file: dict[str, set[str]] = {file_name: set() for file_name in MIGRATION_FILES}
    for file_name, tool, _, _ in _iter_examples():
        by_file[file_name].add(tool)

    for file_name, tools in by_file.items():
        assert "torchlens" in tools
        assert len(tools - {"torchlens"}) == 1
