"""Structured result models for tool execution.

Uses frozen dataclasses — zero external dependencies, typed attribute
access, and easy conversion to dict/JSON when needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class CallStatus(str, Enum):
    """Outcome status of a single tool call."""

    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Result of a single tool invocation inside ``execute()``.

    Attributes
    ----------
    tool:
        Name of the tool that was called.
    status:
        ``"ok"`` on success, ``"error"`` on failure.
    result:
        The return value of the tool function (present when
        ``status == "ok"``).
    error:
        Structured error string (present when ``status == "error"``).
    """

    tool: str
    status: CallStatus
    result: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Convenience check: ``True`` when the call succeeded."""
        return self.status is CallStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (e.g. for JSON transport)."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict())
