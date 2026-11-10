"""C8, the fast-lane share of function calls, counted per op commit.

C8 is counted over function-call commits only: an op commit made inside a
wrapped torch callable that is neither a source record nor a host
constructor. Source records (the model input, a buffer registered on first use
inside a wrapper) and host constructors (``torch.tensor``, ``arange``) are left
out of both sides of the share and reported as their own counts, as are
commits made outside any wrapper (synthesized boundary ops).

A commit is on the fast lane when one of the fast lane's entry functions
(``_call_plan.FAST_LANE_COUNTER_NAMES``) is on the stack between the commit and
its nearest wrapper frame. A multi-output call is one fast-lane call and one
commit per output, so the share counts each output.
"""

from __future__ import annotations

from typing import Any

#: ``wrapped_func`` frames live in this file.
WRAPPER_FILE = "backends/torch/wrappers.py"
#: The fast lane's entry functions live in this file.
FAST_LANE_FILE = "backends/torch/_call_plan.py"
#: Source-record entry functions (``sources.py``); a commit under one is a source record.
SOURCE_FILE = "backends/torch/sources.py"
SOURCE_FUNCS = frozenset({"log_source_tensor", "log_source_tensor_exhaustive"})
#: Stack frames walked from a commit before giving up on finding a wrapper.
MAX_DEPTH = 200


class CommitCensus:
    """Classify each op commit of a trace for C8 (see the module docstring).

    Parameters
    ----------
    fast_names
        The fast lane's entry function names, as the product declares them.
    """

    def __init__(self, fast_names: tuple[str, ...]) -> None:
        self.fast_names = frozenset(fast_names)
        self.function_calls = 0
        self.fast = 0
        self.source_records = 0
        self.host_constructors = 0
        self.unwrapped = 0

    def on_commit(self, frame: Any) -> None:
        """Count one ``commit_op`` call, given its frame (a ``count_phases`` call hook)."""

        fast = False
        source = False
        caller = frame.f_back
        depth = 0
        while caller is not None and depth < MAX_DEPTH:
            code = caller.f_code
            name = code.co_name
            filename = code.co_filename.replace("\\", "/")
            if name == "wrapped_func" and filename.endswith(WRAPPER_FILE):
                self._count_wrapped(caller, fast, source)
                return
            if name in self.fast_names and filename.endswith(FAST_LANE_FILE):
                fast = True
            elif name in SOURCE_FUNCS and filename.endswith(SOURCE_FILE):
                source = True
            caller = caller.f_back
            depth += 1
        if source:
            self.source_records += 1
        else:
            self.unwrapped += 1

    def _count_wrapped(self, wrapper: Any, fast: bool, source: bool) -> None:
        if source:
            self.source_records += 1
        elif wrapper.f_locals.get("is_host_constructor"):
            self.host_constructors += 1
        else:
            self.function_calls += 1
            self.fast += fast

    def report(self) -> dict[str, Any]:
        """Return the C8 counts and the share over function-call commits."""

        return {
            "function_call_commits": self.function_calls,
            "fast_lane_commits": self.fast,
            "source_record_commits": self.source_records,
            "host_constructor_commits": self.host_constructors,
            "unwrapped_commits": self.unwrapped,
            "fast_lane_share": self.fast / self.function_calls if self.function_calls else 0.0,
        }
