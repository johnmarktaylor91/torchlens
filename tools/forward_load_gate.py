"""Forward-load CI gate: the previous release's wheel loads the new goldens.

Ecosystem MEMO 3.4 / G8: on every persistence train, install the PREVIOUS
release's torchlens wheel and analysis-load the new train's golden artifacts
under it. This is the only mechanism that catches a same-stamp drift break
(two writers sharing a tlspec stamp with different persisted grammars --
the released v2.33.0/v2.34.1 pair), which damages the BACKWARD window the
project actually promises. There is NO forward-compatibility promise: the
green condition per golden is

- artifacts at or below the previous release's schema ceiling must
  ANALYSIS-LOAD cleanly, and
- artifacts above its ceiling must refuse TYPED (a TorchLens error class,
  never an arbitrary crash and never a silent wrong load).

The per-PR cheap variant (no wheels, no network) is the writer-contract
lockstep + alias-or-fail suite in tests/test_tlspec_envelope_contract.py.

Usage (CI wiring lives in the release workflow; see
sprint/packaging_requests.tsv for the P05 request):

    python tools/forward_load_gate.py --previous 2.34.1 \
        --corpus tests/release_goldens/genuine_release_artifacts.tar.gz

The previous release installs with ``pip install --no-deps --target`` so the
multi-GB dependency stack (torch) is reused from the running environment;
the probe subprocess runs OUTSIDE the repo root so the dev tree cannot
shadow the installed wheel.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

#: Per-golden expectation classes.
LOAD = "load"
TYPED_REFUSAL_OK = "load_or_typed_refusal"

#: Verdicts the probe subprocess reports.
LOADED = "loaded"
REFUSED_TYPED = "refused_typed"
CRASHED = "crashed"

_PROBE_SOURCE = r"""
import json, sys, warnings
results = {}
corpus_root = sys.argv[1]
names = json.loads(sys.argv[2])
import torchlens as tl
results["reader_version"] = getattr(tl, "__version__", "?")
for name in names:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = tl.load(corpus_root + "/" + name)
        results[name] = {"verdict": "loaded", "n_ops": len(trace)}
    except Exception as exc:
        typed = type(exc).__module__.startswith("torchlens")
        results[name] = {
            "verdict": "refused_typed" if typed else "crashed",
            "error_type": type(exc).__name__,
            "code": (getattr(exc, "fields", None) or {}).get("code"),
            "message": str(exc)[:200],
        }
print(json.dumps(results))
"""


def classify(verdict: str, expectation: str) -> bool:
    """Return whether one golden's verdict satisfies its expectation."""

    if expectation == LOAD:
        return verdict == LOADED
    if expectation == TYPED_REFUSAL_OK:
        return verdict in (LOADED, REFUSED_TYPED)
    raise ValueError(f"unknown expectation {expectation!r}")


def expectations_for(previous_ceiling: int | None) -> dict[str, str]:
    """Expected outcome per corpus golden under the previous release.

    Without a known ceiling for the previous reader, every golden gets the
    conservative bar: load OR refuse typed -- an untyped crash or a silent
    wrong load is always red.
    """

    names = (
        "art_v2.33.0_portable",
        "art_v2.34.1_portable",
        "art_main_portable",
    )
    return dict.fromkeys(names, TYPED_REFUSAL_OK)


def install_previous(version: str, target: Path) -> None:
    """Install the previous release's wheel (no deps) into ``target``."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            f"torchlens=={version}",
        ],
        check=True,
    )


def run_probe(wheel_dir: Path, corpus_root: Path, names: list[str]) -> dict:
    """Load each golden under the installed previous release, out of tree."""

    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(wheel_dir)
    with tempfile.TemporaryDirectory() as neutral_cwd:
        probe = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE, str(corpus_root), json.dumps(names)],
            env=env,
            cwd=neutral_cwd,
            capture_output=True,
            text=True,
            timeout=1200,
        )
    if probe.returncode != 0:
        raise RuntimeError(
            f"probe interpreter died (rc={probe.returncode}):\n{probe.stderr[-2000:]}"
        )
    return json.loads(probe.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, help="previous release version, e.g. 2.34.1")
    parser.add_argument(
        "--corpus",
        default="tests/release_goldens/genuine_release_artifacts.tar.gz",
        help="golden corpus tarball",
    )
    args = parser.parse_args()

    expectations = expectations_for(None)
    names = sorted(expectations)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        with tarfile.open(args.corpus, "r:gz") as tar:
            tar.extractall(corpus_root)
        wheel_dir = tmp_path / "previous_wheel"
        install_previous(args.previous, wheel_dir)
        results = run_probe(wheel_dir, corpus_root, names)

    print(f"previous reader: torchlens {results.pop('reader_version', '?')}")
    failures = []
    for name in names:
        row = results.get(name, {"verdict": CRASHED, "error_type": "missing-result"})
        ok = classify(row["verdict"], expectations[name])
        print(
            f"  {'PASS' if ok else 'FAIL'} {name}: {row['verdict']}"
            + (f" [{row.get('error_type')}/{row.get('code')}]" if row["verdict"] != LOADED else "")
        )
        if not ok:
            failures.append(name)
    if failures:
        print(f"FORWARD-LOAD GATE RED: {failures}")
        return 1
    print("FORWARD-LOAD GATE GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
