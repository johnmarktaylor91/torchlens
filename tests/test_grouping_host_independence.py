"""Layer grouping does not depend on how the host process imported TorchLens.

The call-site walk (``introspection._get_code_context``) must skip TorchLens's own
frames whatever spelling the import system gave their filenames. A ``sys.path``
entry of ``"."`` (or ``"/repo/."``) leaves ``/./`` inside every TorchLens filename;
a raw prefix comparison against the normalised package directory then failed,
236 internal frames leaked into Exphormer's chains, and the recursion collapse on
the wrapper's repeated code object merged the in-projection and out-projection
``linear`` of one ``MultiheadAttention`` into "one layer, two passes". Grouping
differed between plain scripts and pytest for nine classics-corpus entries, and a
census run in such a process mis-tagged them recurrent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

_PACKAGE_DIR = Path(tl.__file__).resolve().parent
# Spelled the way ``torchlens._utils._package_frames`` spells it, computed here so the
# behavioural tests below import (and fail) on a package without that module.
_PACKAGE_PREFIX = os.path.normcase(str(_PACKAGE_DIR)) + os.sep


def _normalized(path: str) -> str:
    """``path`` spelled the way :data:`_PACKAGE_PREFIX` is."""

    return os.path.normcase(os.path.abspath(path))


class OneAttention(nn.Module):
    """One MultiheadAttention call and a head: no loop anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(x, x, x, need_weights=False)[0]
        return self.head(h[:, 0])


def _build() -> tuple[nn.Module, torch.Tensor]:
    """The model and a fixed input under a fixed seed."""

    torch.manual_seed(0)
    return OneAttention(), torch.randn(1, 5, 8)


# The same model for the subprocess. Torch's ``multi_head_attention_forward`` (itself a
# wrapped torch function) calls the wrapped ``linear`` from two places, so the
# wrapper's code object sits twice on the stack of each inner ``linear`` op: the
# shape a leaked internal frame merges.
_MODEL_SOURCE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, DOTTED_ROOT)
    import torch
    from torch import nn


    class OneAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
            self.head = nn.Linear(8, 3)

        def forward(self, x):
            h = self.attn(x, x, x, need_weights=False)[0]
            return self.head(h[:, 0])


    def build():
        torch.manual_seed(0)
        return OneAttention(), torch.randn(1, 5, 8)
    """
)

_SUBPROCESS_SOURCE = _MODEL_SOURCE + textwrap.dedent(
    """
        import json, os, sys
        import torchlens as tl

        model, x = build()
        trace = tl.trace(model, x)
        seen = {}
        internal = []
        for op in trace.layer_list:
            if op.is_input or op.is_output or op.is_buffer:
                continue
            seen.setdefault(op.layer_label, (str(op.func_name), int(op.num_passes or 1)))
        for op in trace.ops:
            for location in op.code_context or ():
                if os.path.normcase(os.path.abspath(str(location.file))).startswith(PREFIX):
                    internal.append([op.label, str(location.file), location.func_name])
        print(json.dumps({
            "torchlens_file": tl.__file__,
            "is_recurrent": bool(trace.is_recurrent),
            "shape": sorted(seen.values()),
            "internal_context_frames": internal,
        }))
        """
)


def _layer_shape(trace: Any) -> list[tuple[str, int]]:
    """Sorted (func_name, num_passes) per layer, inputs/outputs/buffers excluded."""

    seen: dict[str, tuple[str, int]] = {}
    for op in trace.layer_list:
        if op.is_input or op.is_output or op.is_buffer:
            continue
        seen.setdefault(op.layer_label, (str(op.func_name), int(op.num_passes or 1)))
    return sorted(seen.values())


def test_is_torchlens_code_ignores_filename_spelling() -> None:
    """The frame filter answers by location, not by how the path is spelled."""

    from torchlens._utils._package_frames import (
        TORCHLENS_PACKAGE_PREFIX,
        is_torchlens_code,
        normalized_filename,
    )

    assert TORCHLENS_PACKAGE_PREFIX == _PACKAGE_PREFIX
    code = is_torchlens_code.__code__
    real = Path(code.co_filename).resolve()
    assert normalized_filename(str(real)).startswith(TORCHLENS_PACKAGE_PREFIX)
    assert is_torchlens_code(code)
    dotted = os.path.join(str(real.parent.parent.parent), ".", "torchlens", "_utils", real.name)
    assert "." + os.sep in dotted
    assert is_torchlens_code(code.replace(co_filename=dotted))
    assert is_torchlens_code(code.replace(co_filename=os.path.relpath(str(real))))
    sibling = str(real.parent.parent.parent / "torchlens_contrib" / "model.py")
    assert not is_torchlens_code(code.replace(co_filename=sibling))
    assert not is_torchlens_code(code.replace(co_filename=__file__))
    assert not is_torchlens_code(code.replace(co_filename="<string>"))
    # Code objects compare equal across filenames; verdicts must not.
    inside = compile("x = 1", str(real), "exec")
    outside = compile("x = 1", __file__, "exec")
    assert inside == outside
    assert is_torchlens_code(inside)
    assert not is_torchlens_code(outside)


def test_no_torchlens_frame_in_any_code_context() -> None:
    """Every recorded frame of an op lies outside the package, in-process."""

    model, x = _build()
    trace = tl.trace(model, x)
    leaked = [
        (op.label, str(location.file))
        for op in trace.ops
        for location in op.code_context or ()
        if _normalized(str(location.file)).startswith(_PACKAGE_PREFIX)
    ]
    assert leaked == []
    shape = Counter(_layer_shape(trace))
    # The two ``linear`` calls inside one attention (in-projection, out-projection)
    # are two places in torch's code: two layers, never passes of one.
    assert shape[("linear", 1)] >= 3, shape
    assert not trace.is_recurrent


@pytest.mark.heavy
def test_grouping_equal_when_torchlens_imported_through_dotted_path(tmp_path: Path) -> None:
    """A process that imported TorchLens through ``<root>/.`` groups like this one."""

    model, x = _build()
    here = tl.trace(model, x)

    root = _PACKAGE_DIR.parent
    script = tmp_path / "capture_dotted.py"
    dotted_root = str(root) + os.sep + "."
    script.write_text(
        _SUBPROCESS_SOURCE.replace("DOTTED_ROOT", repr(dotted_root)).replace(
            "PREFIX", repr(_PACKAGE_PREFIX)
        ),
        encoding="utf-8",
    )
    # No inherited PYTHONPATH: a normalised repo root there would win the import.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(tmp_path),
        env=env,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    there = json.loads(proc.stdout.strip().splitlines()[-1])
    if os.sep + "." + os.sep not in there["torchlens_file"]:
        pytest.skip(
            "this interpreter normalised the dotted sys.path entry; the leak cannot be "
            f"staged here ({there['torchlens_file']})"
        )
    assert there["internal_context_frames"] == []
    assert [tuple(item) for item in there["shape"]] == _layer_shape(here)
    assert there["is_recurrent"] == here.is_recurrent
