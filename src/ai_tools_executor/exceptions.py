"""Custom exceptions for the AI Tools Executor package.

Provides a structured exception hierarchy so errors can be caught
at the right granularity and formatted into the standard
Input / Error / Expected / Hint format the agent expects.
"""

from __future__ import annotations


class ToolExecutorError(Exception):
    """Base exception for all AI Tools Executor errors."""

    def __init__(
        self,
        message: str,
        *,
        input_text: str | None = None,
        expected: str | None = None,
        hint: str | None = None,
    ) -> None:
        self.input_text = input_text
        self.expected = expected
        self.hint = hint
        super().__init__(message)

    def format(self) -> str:
        """Return the structured error block the agent receives."""
        lines: list[str] = []
        lines.append(f"{self.__class__.__name__}: {self}")
        if self.input_text is not None:
            lines.append(f"  Input:    {self.input_text}")
        lines.append(f"  Error:    {self}")
        if self.expected is not None:
            lines.append(f"  Expected: {self.expected}")
        if self.hint is not None:
            lines.append(f"  Hint:     {self.hint}")
        return "\n".join(lines)


class ToolNotFoundError(ToolExecutorError):
    """Raised when a tool name does not exist in the registry."""


class ToolAlreadyRegisteredError(ToolExecutorError):
    """Raised when attempting to register a tool whose name is already taken."""


class ParseError(ToolExecutorError):
    """Raised when ``ast.parse()`` fails on the agent's input string."""


class ValidationError(ToolExecutorError):
    """Raised when AST-level validation fails.

    Examples: missing required param, unknown param name, wrong type.
    """


class ExecutionError(ToolExecutorError):
    """Raised when the actual tool function raises at runtime."""
