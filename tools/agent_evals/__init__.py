"""Agent-from-docs evaluation harness for TorchLens.

An agent receives only TorchLens's shipped documentation and a short task, writes one script,
and this harness executes the script and grades it with deterministic checks. Offline mode runs
the reference solutions (written from the docs alone) in the test suite, so docs and API drift
surfaces with no model call. Live mode runs a pluggable agent command and is never run in CI.
See ``README.md`` beside this file.
"""
