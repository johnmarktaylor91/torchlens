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
  against the run) so a content regression cannot hide behind determinism;
- every printed table the block claims in comments (``# rank label prob``
  rows under an ``output_table(...)`` line) evaluated against the run.

README fences elide their imports, so the gate supplies exactly the declared
``_SCREEN_PRELUDE`` and nothing else. The screen-one input rule (D15) is a
predicate: a screen that prints only structure, shapes, and counts may run on
any input, but a screen printing a value/label/decode must pin its real input.
Today's screen prints the ViT's top-3 labels and probabilities, so it runs on
the README's own ``cat.jpg``, fetched once and pinned by size and sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
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

#: The imports README fences elide; the gate adds these and nothing else.
_SCREEN_PRELUDE = "import torchlens as tl\nfrom transformers import ViTForImageClassification\n"

#: D15: the screen prints labels and probabilities, so it runs on its real input,
#: the README's ``cat.jpg``: Wikimedia Commons "European tabby cat looking away"
#: (CC0, by Gugalcrom123), pinned by size and sha256.
_SCREEN_IMAGE_NAME = "cat.jpg"
_SCREEN_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/4/43/European_tabby_cat_looking_away.jpg"
)
_SCREEN_IMAGE_BYTES = 1_652_492
_SCREEN_IMAGE_SHA256 = (
    "7f6a70ce301d8c2260fd81c440bc135262f989d9b4191eda5b4f9eb8b1f1dc8e"  # pragma: allowlist secret
)

#: One claimed row of a printed table: ``#   <rank>   <label>   <prob>``.
_TABLE_ROW = re.compile(r"^#\s+(\d+)\s+(.+?)\s+(\d+\.\d+)\s*$")

_RUNNER = """
import json, re, sys, warnings

payload = json.loads(sys.stdin.read())
block = payload["block"]
captured = []
import io, contextlib
stdout = io.StringIO()
namespace = {"__name__": "__main__"}
shape_comment = re.compile(r"^(\\S.*?)\\s+#\\s*(torch\\.Size\\(\\[[^\\]]*\\]\\))")
table_line = re.compile(r"^(\\S.*\\.output_table\\(.*\\))\\s*$")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with contextlib.redirect_stdout(stdout):
        exec(compile(payload["prelude"], "README-prelude", "exec"), namespace)
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
        tables = [
            [
                m.group(1).strip(),
                [
                    [int(rank), str(label), float(prob)]
                    for rank, label, prob in eval(m.group(1), namespace)[
                        ["rank", "label", "prob"]
                    ].itertuples(index=False)
                ],
            ]
            for m in map(table_line.match, block.splitlines())
            if m
        ]
    captured = [f"{type(w.message).__name__}: {w.message}" for w in caught]
sys.stdout.write(
    json.dumps(
        {
            "warnings": captured,
            "stdout": stdout.getvalue(),
            "models": models,
            "shapes": shapes,
            "tables": tables,
        }
    )
)
"""


def _first_readme_python_block() -> str:
    """Extract the first ```python fence from the repo README (the screen)."""

    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"```python\n(.*?)```", readme, re.DOTALL)
    assert match, "README.md lost its first python block"
    return match.group(1)


def _pinned_screen_image() -> bytes:
    """Return the README's ``cat.jpg``, fetched once into the user cache and checksum-pinned.

    Returns
    -------
    bytes
        The image file, verified against ``_SCREEN_IMAGE_BYTES`` and ``_SCREEN_IMAGE_SHA256``.
    """

    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cached = cache_root / "torchlens-tests" / _SCREEN_IMAGE_SHA256[:16] / _SCREEN_IMAGE_NAME
    if not cached.exists():
        request = urllib.request.Request(
            _SCREEN_IMAGE_URL, headers={"User-Agent": "torchlens-tests (README first-screen gate)"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - pinned https URL
            fetched = response.read()
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(fetched)
    data = cached.read_bytes()
    assert len(data) == _SCREEN_IMAGE_BYTES, f"{cached}: {len(data)} bytes, not the pinned image"
    assert hashlib.sha256(data).hexdigest() == _SCREEN_IMAGE_SHA256, (
        f"{cached} is not the pinned README image (sha256 mismatch); delete it to refetch"
    )
    return data


def _claimed_tables(block: str) -> list[tuple[str, list[tuple[int, str, str]]]]:
    """Return each ``output_table(...)`` line of the block with the rows its comments claim.

    Parameters
    ----------
    block:
        The README first-screen source.

    Returns
    -------
    list[tuple[str, list[tuple[int, str, str]]]]
        ``(expression, [(rank, label, printed prob), ...])`` per table line, in order.
    """

    lines = block.splitlines()
    claimed = []
    for index, line in enumerate(lines):
        if not re.match(r"^\S.*\.output_table\(.*\)\s*$", line):
            continue
        rows = []
        for comment in lines[index + 1 :]:
            if not comment.startswith("#"):
                break
            row = _TABLE_ROW.match(comment)
            if row:
                rows.append((int(row.group(1)), row.group(2), row.group(3)))
        claimed.append((line.strip(), rows))
    return claimed


def _run_screen_one(tmp_path: Path, run_id: int) -> dict:
    """Execute the README block in a cold subprocess; return its payload."""

    workdir = tmp_path / f"run{run_id}"
    workdir.mkdir()
    (workdir / _SCREEN_IMAGE_NAME).write_bytes(_pinned_screen_image())
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER],
        input=json.dumps({"prelude": _SCREEN_PRELUDE, "block": _first_readme_python_block()}),
        capture_output=True,
        text=True,
        cwd=workdir,
        env={
            **os.environ,
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
    torchvision ``weights=`` builders are both warmed. For a Hugging Face image
    model, ``tl.trace`` on a PIL image also resolves the model's own image
    processor (``preprocessor_config.json``), so that file is cached here too:
    the compared runs inherit this environment, and test modules that set
    ``HF_HUB_OFFLINE`` at import time can leave them offline, where a file the
    warm-up skipped is a hard failure rather than a download.

    Parameters
    ----------
    block:
        The README first-screen source.
    """

    hf_loads = re.findall(r"(\w+)\.from_pretrained\(\"([^\"]+)\"\)", block)
    if hf_loads:
        transformers = pytest.importorskip("transformers")
        for cls_name, repo_id in hf_loads:
            model = getattr(transformers, cls_name).from_pretrained(repo_id)
            if getattr(model, "main_input_name", None) == "pixel_values":
                transformers.AutoImageProcessor.from_pretrained(repo_id)
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
    assert "ViTForImageClassification" in first["models"], "the screen lost the model identity"
    assert first["shapes"], "the first screen lost its torch.Size comments"
    for expression, claimed, measured in first["shapes"]:
        assert measured == claimed, (
            f"README first screen claims {expression} is {claimed}; the run measured {measured}"
        )
    claimed_tables = _claimed_tables(_first_readme_python_block())
    assert [expression for expression, _ in claimed_tables] == [
        expression for expression, _ in first["tables"]
    ], "the runner and the gate disagree on the screen's output_table lines"
    for (expression, claimed_rows), (_, measured_rows) in zip(
        claimed_tables, first["tables"], strict=True
    ):
        measured = [
            (rank, label, f"{prob:.{len(claimed[2].split('.')[1])}f}")
            for (rank, label, prob), claimed in zip(measured_rows, claimed_rows, strict=False)
        ]
        assert measured == claimed_rows, (
            f"README first screen claims {expression} prints {claimed_rows}; "
            f"the run printed {measured_rows[: len(claimed_rows)]}"
        )


def test_readme_screen_one_pins_its_real_inputs() -> None:
    """D15 predicate: screen one runs real weights, and any value it prints is pinned.

    The block may print shapes, counts, and names on any input; a printed
    label/probability/decode must run on the checksum-pinned real image and be
    claimed in rows the gate compares against the run.
    """

    block = _first_readme_python_block()
    assert ".eval()" in block, "screen one must pin eval mode on the setup line"
    assert re.search(r"\.from_pretrained\(\"google/vit-base-patch16-224\"\)", block), (
        "screen one uses real pretrained weights"
    )
    assert "decode" not in block, "a decode on screen one needs its own pinned-value check"
    if "output_table" in block:
        assert f'"{_SCREEN_IMAGE_NAME}"' in block, (
            "screen one prints labels and probabilities: it must read the pinned real image"
        )
        tables = _claimed_tables(block)
        assert tables and all(rows for _, rows in tables), (
            "every output_table line on screen one must claim its printed rows in comments, "
            "so the gate can compare them with the run"
        )
