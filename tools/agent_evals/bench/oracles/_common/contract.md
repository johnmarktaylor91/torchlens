# Solution contract (shared by every task)

A solution is a directory holding `solve.py`. The harness runs it as

    python solve.py --inputs INPUTS_DIR --outputs OUTPUTS_DIR

with the solution directory as the working directory. `INPUTS_DIR` holds the
task's inputs exactly as described in the task's `spec.md`. The script must
create `OUTPUTS_DIR` if needed and write the files the spec names, in the
formats the spec names (`.npz` written with `numpy.savez`, `.json` written with
`json.dump`). It must not read anything outside `INPUTS_DIR`, its own
directory, the Python environment and the model cache, and it must not write
anywhere but `OUTPUTS_DIR`.

The same script is run a second time on a held-out inputs directory the
solution never saw, when the spec says so. A solution therefore has to compute
from the inputs it is given, never from values typed into the script.

Determinism: solutions run on CPU in float32 unless the spec says otherwise.
Every random choice uses the seeds named in the spec.

Grading: `python grader.py --task-dir TASK --outputs OUT [--heldout-outputs HOUT] [--cache DIR]`
prints one JSON object on its last stdout line:

    {"task": "...", "passed": true|false,
     "bars": [{"name": "...", "passed": true|false, "value": ..., "threshold": ..., "note": "..."}]}

A missing or malformed output file fails every bar that depends on it. The
grader never calls a judge model; every bar is a deterministic computation.

Isolation: the runner copies the inputs and the solution into a fresh temporary
workspace outside the task tree before running, so held-out inputs, the grader
and its oracle cache are not reachable by a relative path. The harness that runs
subject agents must add real filesystem isolation (a container or a sandbox
user) on top of this.

Toolkit use: in every arm a solution may use plain PyTorch for any part of the
task; the library arms are not required to route everything through the
library. A solution may write a free-form `approach.txt` into its outputs saying
which parts used the library; the harness records it and no grader reads it.
