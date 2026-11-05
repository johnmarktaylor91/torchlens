"""The README first-screen subprocess golden gate (F17 B16; memo D15/D16).

The block a newcomer actually copies (the FIRST ```python fence in
README.md) runs in a fresh subprocess with every warning recorded, twice:

- ZERO warnings (deprecations included) -- the gate that guards the first
  screen forever;
- the two runs' stdout (taken after the pretrained weights are cached, since
  a first download can announce itself on stdout) must be byte-identical after
  normalizing EXACTLY the declared volatile-field list (today:
  ``capture_timestamp`` and the summary's host-RSS ``forward peak`` token) --
  a NEW volatile field FAILS this gate rather than being silently normalized;
- pinned structural facts (the real pretrained model the block names, eval
  mode, and every ``# torch.Size([...])`` comment in the block evaluated
  against the run) so a content regression cannot hide behind determinism.

The screen-one input rule (D15) is a predicate: this screen prints only
structure, shapes, and counts; any future screen printing a value/label/decode
must pin its real input instead.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The asserted-exact volatile-field normalizer list (memo D16). Adding a
#: line pattern here is a reviewed golden change, never a quiet fix.
_VOLATILE_LINE_PATTERNS = (
    re.compile(r"^\s*capture_timestamp: .*$", re.MULTILINE),
    # The summary's Memory line prints ``forward peak <N> MB (<meaning>)``: on
    # CPU that is the host RSS growth, documented as a real runtime measurement
    # that is never portable (it read 78.9 MB and 80 MB on two cold runs of the
    # same block -- AUD-CODE 3.10). Only the peak TOKEN is volatile; the
    # activation byte counts on the same line stay pinned.
    re.compile(r"forward peak \d+(?:\.\d+)? [KMGT]?i?B \((?:[^()]|\([^()]*\))*\)"),
)

_RUNNER = """
import json, re, sys, warnings

block = sys.stdin.read()
captured = []
import io, contextlib
stdout = io.StringIO()
namespace = {"__name__": "__main__"}
shape_comment = re.compile(r"^(\\S.*?)\\s+#\\s*(torch\\.Size\\(\\[[^\\]]*\\]\\))")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with contextlib.redirect_stdout(stdout):
        exec(compile(block, "README-first-screen", "exec"), namespace)
        import torch
        models = sorted(
            {type(v).__name__ for v in namespace.values() if isinstance(v, torch.nn.Module)}
        )
        shapes = [
            [m.group(1).strip(), m.group(2), str(eval(m.group(1), namespace))]
            for m in map(shape_comment.match, block.splitlines())
            if m
        ]
    captured = [f"{type(w.message).__name__}: {w.message}" for w in caught]
sys.stdout.write(
    json.dumps(
        {"warnings": captured, "stdout": stdout.getvalue(), "models": models, "shapes": shapes}
    )
)
"""


def _first_readme_python_block() -> str:
    """Extract the first ```python fence from the repo README (the screen)."""

    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"```python\n(.*?)```", readme, re.DOTALL)
    assert match, "README.md lost its first python block"
    return match.group(1)


def _run_screen_one(tmp_path: Path, run_id: int) -> dict:
    """Execute the README block in a cold subprocess; return its payload."""

    workdir = tmp_path / f"run{run_id}"
    workdir.mkdir()
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER],
        input=_first_readme_python_block(),
        capture_output=True,
        text=True,
        cwd=workdir,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(_REPO_ROOT),
        },
        timeout=560,
        check=False,
    )
    assert completed.returncode == 0, (
        f"README first screen failed ({completed.returncode}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout)


def _warm_pretrained_weights_cache(block: str) -> None:
    """Download the block's pretrained weights before the compared runs.

    On a cold cache the block's first run downloads its weights, and a download
    can announce itself on stdout. That reflects the cache's state, not the
    screen's output, so the gate fills the cache in-process first; both compared
    runs stay cold subprocesses. Hugging Face ``from_pretrained`` loads and
    torchvision ``weights=`` builders are both warmed.

    Parameters
    ----------
    block:
        The README first-screen source.
    """

    hf_loads = re.findall(r"(\w+)\.from_pretrained\(\"([^\"]+)\"\)", block)
    if hf_loads:
        transformers = pytest.importorskip("transformers")
        for cls_name, repo_id in hf_loads:
            getattr(transformers, cls_name).from_pretrained(repo_id)
    tv_loads = re.findall(r"models\.(\w+)\(weights=\"(\w+)\"\)", block)
    if tv_loads:
        models = pytest.importorskip("torchvision.models")
        for builder, weights in tv_loads:
            models.get_model_weights(builder)[weights].get_state_dict(progress=False)


def _normalize(text: str) -> str:
    """Blank exactly the declared volatile lines, nothing else."""

    for pattern in _VOLATILE_LINE_PATTERNS:
        text = pattern.sub("<volatile>", text)
    return text


def test_readme_first_screen_zero_warnings_and_reproducible(tmp_path: Path) -> None:
    """The gate: zero warnings, reproducible modulo the declared normalizer."""

    _warm_pretrained_weights_cache(_first_readme_python_block())
    first = _run_screen_one(tmp_path, 1)
    second = _run_screen_one(tmp_path, 2)
    assert first["warnings"] == [], (
        "the README first screen must emit ZERO warnings; got:\n" + "\n".join(first["warnings"])
    )
    normalized_first = _normalize(first["stdout"])
    normalized_second = _normalize(second["stdout"])
    assert normalized_first == normalized_second, (
        "README first-screen output is not reproducible after normalizing the "
        "DECLARED volatile fields (capture_timestamp, the host-RSS forward-peak "
        "token). A new volatile field "
        "fails this gate deliberately -- extend _VOLATILE_LINE_PATTERNS only "
        "as a reviewed change.\n"
        + "\n".join(
            line
            for pair in zip(
                normalized_first.splitlines(), normalized_second.splitlines(), strict=False
            )
            if pair[0] != pair[1]
            for line in pair
        )
    )
    assert "GPT2LMHeadModel" in first["models"], "the screen lost the model identity"
    assert first["shapes"], "the first screen lost its torch.Size comments"
    for expression, claimed, measured in first["shapes"]:
        assert measured == claimed, (
            f"README first screen claims {expression} is {claimed}; the run measured {measured}"
        )


def test_readme_screen_one_prints_structure_only(tmp_path: Path) -> None:
    """D15 predicate: screen one prints no decoded values or labels.

    The block may print shapes, counts, tables, and names; a decode/label
    print would require the checksum-pinned real image instead.
    """

    block = _first_readme_python_block()
    assert ".eval()" in block, "screen one must pin eval mode on the setup line"
    assert re.search(r"\.from_pretrained\(\"gpt2\"\)", block), (
        "screen one uses real pretrained weights"
    )
    assert "decode" not in block and "output_table" not in block, (
        "value-level claims moved onto screen one: use the pinned real image"
    )
