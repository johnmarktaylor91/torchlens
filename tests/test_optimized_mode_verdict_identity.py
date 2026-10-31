"""Package-wide ``python -O`` verdict-identity gate (r7 b7-sol R24).

97 shipped ``assert`` statements survive in ``torchlens/`` (shrink-only
ceiling), several on verdict-adjacent paths (``validation/backward.py``
pre-comparator guards, postprocess invariants). ``-O`` strips every one of
them, and until now nothing proved that validation VERDICTS are identical in
optimized mode -- the matrix stop condition explicitly requires an
optimized-mode verdict-equivalence proof, not just the audit's armed-under--O
refusal.

The gate runs one ``-O`` child that must reproduce three verdict classes
byte-for-byte with normal mode:

* a clean forward validation stays ``True``;
* a clean backward validation stays ``True`` (the pre-comparator asserts at
  ``validation/backward.py`` are stripped here -- the verdict must not
  change);
* a planted spurious bool edge (frozen replay callable) STILL settles
  ``failed``/``perturbation_insensitive`` -- the tripwire direction: an
  assert-stripped check failing OPEN would bless corruption under ``-O``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.heavy

_CHILD = """
import sys
if sys.flags.optimize < 1:
    print("NOT_OPTIMIZED")
    sys.exit(2)
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation.core import (
    _check_whether_func_on_saved_parents_yields_saved_tensor,
)

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
x = torch.randn(2, 4)
print("FORWARD_VERDICT", tl.validate(model, x, scope="forward", seed=0))
print("BACKWARD_VERDICT", tl.validate(model, x, scope="backward", seed=0))


class Gate(nn.Module):
    def forward(self, z):
        return (z > 1.0e30).float()


trace = tl.trace(
    Gate(),
    torch.randn(3, 4),
    capture=CaptureOptions(layers_to_save="all", save_args=True, seed=0),
)
op = [o for o in trace.layer_list if o.func_name == "__gt__"][0]
saved = op.out.detach().clone()
object.__setattr__(op, "func", lambda *a, **k: saved.clone())
result = _check_whether_func_on_saved_parents_yields_saved_tensor(
    trace, op.label, perturb=True, layers_to_perturb=[op.parents[0]]
)
print("SPURIOUS_DECISION", result.decision, result.reason)
"""


def test_optimized_mode_verdicts_are_identical_and_tripwires_still_fire() -> None:
    """The three verdict classes must not change under ``python -O``."""

    completed = subprocess.run(
        [sys.executable, "-O", "-c", _CHILD],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = {
        line.split(" ", 1)[0]: line.split(" ", 1)[1]
        for line in completed.stdout.splitlines()
        if " " in line
    }
    assert lines.get("FORWARD_VERDICT") == "True", lines
    assert lines.get("BACKWARD_VERDICT") == "True", lines
    assert lines.get("SPURIOUS_DECISION") == "failed perturbation_insensitive", (
        "the perturbation tripwire changed its verdict under -O -- an "
        f"assert-stripped check is failing open: {lines}"
    )
