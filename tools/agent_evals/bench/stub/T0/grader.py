"""Deterministic grader for the harness stub: exact equality with a hook oracle."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--work", required=True)
parser.add_argument("--heldout", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--arm-python", default=sys.executable)
args = parser.parse_args()
work = Path(args.work)
sys.path.insert(0, str(Path(__file__).resolve().parent / "public"))
from model import build_model, make_inputs  # noqa: E402


def oracle(seed: int) -> torch.Tensor:
    model = build_model()
    with torch.no_grad():
        return model.fc2(model.act(model.fc1(make_inputs(seed))))


def compare(path: Path, seed: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    got = torch.load(path, weights_only=True)
    want = oracle(seed)
    if not isinstance(got, torch.Tensor) or got.shape != want.shape:
        return False, f"shape {getattr(got, 'shape', type(got).__name__)}"
    return bool(torch.equal(got, want)), f"max abs diff {float((got - want).abs().max()):.3g}"


bars = []
ok, value = compare(work / "outputs.pt", 0)
bars.append({"name": "visible seed exact", "passed": ok, "value": value, "threshold": "exact"})
for seed in json.loads(Path(args.heldout, "seeds.json").read_text())["seeds"]:
    target = work / f".heldout-{seed}.pt"
    proc = subprocess.run(
        [args.arm_python, "solution.py", "--seed", str(seed), "--out", str(target)],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    ok, value = compare(target, seed) if proc.returncode == 0 else (False, proc.stderr[-300:])
    bars.append(
        {"name": f"held-out seed {seed} exact", "passed": ok, "value": value, "threshold": "exact"}
    )
Path(args.out).write_text(json.dumps({"passed": all(b["passed"] for b in bars), "bars": bars}))
