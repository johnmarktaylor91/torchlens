"""Read the example bank: tested snippets for agents, keyed by public name and by task.

The bank ships in the wheel as Markdown under ``torchlens/_docs/example_bank/``, so it
always matches the installed version. Each group file holds examples; each example is a
``### <id>`` heading, a one-line "what this shows", a short field list and one Python
block whose trailing ``# Output:`` comment lines are the printed output::

    ### trace-read-activation
    Capture a forward pass and read one op's saved output.

    - covers: `tl.trace`, `Trace.ops`, `Op.out`
    - tasks: capture
    - fixture: toy

    ```python
    trace = tl.trace(model, x)
    print(tuple(trace.ops["relu_1_2"].out.shape))
    # Output:
    # (3, 8)
    ```

Fields: ``covers`` (public names the example uses), ``tasks`` (task-card ids and task
words), ``fixture`` (a setup block from the bank's README, ``none`` when self-contained),
and the optional ``kind`` (``wrong`` or ``right`` for a trap pair), ``raises`` (the error
class a ``wrong`` example ends in) and ``requires`` (an optional module the example
imports). The test suite runs every example and checks its output.

Torch-free: ``torchlens api-map --name`` reads it for the detail record's example.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

#: Package-relative folder of the bank.
BANK_DIR = ("_docs", "example_bank")
#: Group files that hold no examples.
_NON_GROUP_FILES = frozenset({"README.md", "INDEX.md"})
#: Line that starts the expected-output comment block inside a code block.
OUTPUT_MARKER = "# Output:"

_HEADING = re.compile(r"^### ([a-z0-9][a-z0-9-]*)\s*$", re.MULTILINE)
_FIELD = re.compile(r"^- (covers|tasks|fixture|kind|raises|requires): (.+)$", re.MULTILINE)
_FENCE = re.compile(r"^```python\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
_NAME = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class BankExample:
    """One tested snippet of the example bank.

    Attributes
    ----------
    id:
        Stable slug, unique across the bank.
    group:
        Group file stem (``trace``, ``layer``, ...).
    shows:
        One line: what the example shows.
    covers:
        Public names the example uses (``tl.trace``, ``Trace.ops``).
    tasks:
        Task-card ids and task words the example serves.
    fixture:
        Setup block the example assumes, ``none`` when self-contained.
    kind:
        ``example``, or ``wrong`` / ``right`` for a trap pair.
    raises:
        Error class a ``wrong`` example ends in, else ``None``.
    requires:
        Optional module the example needs, else ``None``.
    code:
        The snippet without its output comment.
    expected:
        Printed output the snippet produces (empty when it prints nothing).
    """

    id: str
    group: str
    shows: str
    covers: tuple[str, ...]
    tasks: tuple[str, ...]
    fixture: str
    kind: str
    raises: str | None
    requires: str | None
    code: str
    expected: str

    def text(self) -> str:
        """Return the snippet with its output comment, as the bank shows it.

        Returns
        -------
        str
            The code block body.
        """

        if not self.expected:
            return self.code
        lines = [f"# {line}".rstrip() for line in self.expected.splitlines()]
        return "\n".join([self.code, OUTPUT_MARKER, *lines])


def split_output(block: str) -> tuple[str, str]:
    """Split a code block into the code and its expected output.

    Parameters
    ----------
    block:
        Body of a ``python`` fence.

    Returns
    -------
    tuple[str, str]
        The code and the output text (``""`` when the block has no output comment).
    """

    lines = block.rstrip("\n").splitlines()
    if OUTPUT_MARKER not in lines:
        return "\n".join(lines), ""
    cut = len(lines) - 1 - lines[::-1].index(OUTPUT_MARKER)
    output = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines[cut + 1 :]]
    return "\n".join(lines[:cut]).rstrip("\n"), "\n".join(output)


def _names(value: str) -> tuple[str, ...]:
    """Return the backticked names of a field value, else its comma-separated words."""

    found = _NAME.findall(value)
    if found:
        return tuple(found)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_group(group: str, text: str) -> list[BankExample]:
    """Parse one group file into examples.

    Parameters
    ----------
    group:
        Group name (the file stem).
    text:
        File text.

    Returns
    -------
    list[BankExample]
        Examples in file order.

    Raises
    ------
    ValueError
        When an example has no code block or no ``covers`` field.
    """

    heads = list(_HEADING.finditer(text))
    examples: list[BankExample] = []
    for index, head in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
        body = text[head.end() : end]
        fields = {key: value.strip() for key, value in _FIELD.findall(body)}
        fence = _FENCE.search(body)
        if fence is None or "covers" not in fields:
            raise ValueError(f"example {head.group(1)!r} in {group}.md needs covers and code")
        prose = body[: fence.start()].strip().splitlines()
        code, expected = split_output(fence.group(1))
        examples.append(
            BankExample(
                id=head.group(1),
                group=group,
                shows=prose[0].strip() if prose else "",
                covers=_names(fields["covers"]),
                tasks=_names(fields.get("tasks", "")),
                fixture=fields.get("fixture", "none"),
                kind=fields.get("kind", "example"),
                raises=fields.get("raises", "").strip("`") or None,
                requires=fields.get("requires", "").strip("`") or None,
                code=code,
                expected=expected,
            )
        )
    return examples


def _bank_files() -> list[tuple[str, str]]:
    """Return ``(name, text)`` for every Markdown file of the shipped bank."""

    root = resources.files("torchlens").joinpath(*BANK_DIR)
    return sorted(
        (entry.name, entry.read_text(encoding="utf-8"))
        for entry in root.iterdir()
        if entry.name.endswith(".md")
    )


@lru_cache(maxsize=1)
def load_bank() -> tuple[BankExample, ...]:
    """Return every example of the shipped bank.

    Returns
    -------
    tuple[BankExample, ...]
        Examples ordered by group file, then file order.
    """

    examples: list[BankExample] = []
    for name, text in _bank_files():
        if name not in _NON_GROUP_FILES:
            examples.extend(parse_group(name[: -len(".md")], text))
    return tuple(examples)


@lru_cache(maxsize=1)
def load_fixtures() -> dict[str, str]:
    """Return the setup code of every fixture the bank's README defines.

    Returns
    -------
    dict[str, str]
        Fixture name to setup code.
    """

    readme = dict(_bank_files())["README.md"]
    fixtures: dict[str, str] = {}
    for match in re.finditer(r"^### fixture: ([a-z0-9-]+)\s*$", readme, re.MULTILINE):
        fence = _FENCE.search(readme, match.end())
        if fence is not None:
            fixtures[match.group(1)] = fence.group(1).rstrip("\n")
    return fixtures


def examples_for(name: str) -> list[BankExample]:
    """Return the bank examples that cover one public name.

    Parameters
    ----------
    name:
        Public name as the bank spells it (``tl.trace``, ``Trace.fork``).

    Returns
    -------
    list[BankExample]
        Matching examples, ``right`` and plain examples before ``wrong`` ones.
    """

    found = [example for example in load_bank() if name in example.covers]
    return sorted(found, key=lambda example: example.kind == "wrong")
