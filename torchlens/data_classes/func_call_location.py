from typing import List, Optional, Union


class FuncCallLocation:
    """A location in source code where a function call occurred."""

    def __init__(
        self,
        file: str,
        line_number: int,
        func_name: str,
        func_signature: Optional[str],
        func_docstring: Optional[str],
        call_line: str,
        code_context: Optional[List[str]],
        code_context_str: str,
        code_context_labeled: str,
        num_context_lines: int,
    ):
        self.file = file
        self.line_number = line_number
        self.func_name = func_name
        self.func_signature = func_signature
        self.func_docstring = func_docstring
        self.call_line = call_line
        self.code_context = code_context
        self.code_context_str = code_context_str
        self.code_context_labeled = code_context_labeled
        self.num_context_lines = num_context_lines

    def __repr__(self) -> str:
        lines = [
            "FuncCallLocation:",
            f"  file: {self.file}",
            f"  line: {self.line_number}",
            f"  function: {self.func_name}",
        ]
        if self.code_context is not None:
            lines.append("  code:")
            lines.append(self.code_context_labeled)
        else:
            lines.append("  code: source unavailable")
        return "\n".join(lines)

    def __getitem__(self, i: Union[int, slice]) -> Union[str, List[str]]:
        return self.code_context[i]

    def __len__(self) -> int:
        if self.code_context is None:
            return 0
        return len(self.code_context)
