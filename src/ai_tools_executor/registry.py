"""Thread-safe tool registry.

Stores all :class:`ToolInfo` objects and provides lookup / search /
hot-reload capabilities.  A module-level default instance is created on
first access via :func:`get_default_registry`.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from ai_tools_executor.exceptions import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
)

if TYPE_CHECKING:
    from ai_tools_executor.decorator import ToolInfo

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central store for registered tools.

    All mutations are guarded by a :class:`threading.Lock` so the
    registry is safe to share across threads.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}
        self._lock = threading.Lock()

    # ── Mutation ──────────────────────────────────────────────────────

    def register(self, tool_info: ToolInfo) -> None:
        """Add a tool to the registry.

        Raises :class:`ToolAlreadyRegisteredError` if a tool with the
        same name is already registered.
        """
        with self._lock:
            if tool_info.name in self._tools:
                raise ToolAlreadyRegisteredError(
                    f"Tool '{tool_info.name}' is already registered",
                    input_text=tool_info.name,
                )
            self._tools[tool_info.name] = tool_info
            logger.debug(
                "Registered tool %r (category=%r)", tool_info.name, tool_info.category,
            )

    def unregister(self, name: str) -> None:
        """Remove a tool by name (hot-reload support).

        Raises :class:`ToolNotFoundError` if the tool does not exist.
        """
        with self._lock:
            if name not in self._tools:
                raise ToolNotFoundError(
                    f"Tool '{name}' is not registered",
                    input_text=name,
                )
            del self._tools[name]
            logger.debug("Unregistered tool %r", name)

    def clear(self) -> None:
        """Remove all tools.  Primarily useful in tests."""
        with self._lock:
            count = len(self._tools)
            self._tools.clear()
            logger.debug("Cleared registry (%d tool(s) removed)", count)

    # ── Lookup ────────────────────────────────────────────────────────

    def get(self, name: str) -> ToolInfo:
        """Return the :class:`ToolInfo` for *name*.

        Raises :class:`ToolNotFoundError` if the name is unknown.
        """
        with self._lock:
            try:
                return self._tools[name]
            except KeyError:
                available = ", ".join(sorted(self._tools)) or "(none)"
                raise ToolNotFoundError(
                    f"Tool '{name}' not found",
                    input_text=name,
                    hint=f"Available tools: {available}",
                ) from None

    def list_all(self) -> list[ToolInfo]:
        """Return a snapshot of all registered tools."""
        with self._lock:
            return list(self._tools.values())

    def has(self, name: str) -> bool:
        """Check whether *name* is registered without raising."""
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={len(self)}>"


# ── Module-level default instance ─────────────────────────────────────

_default_registry: ToolRegistry | None = None
_default_lock = threading.Lock()


def get_default_registry() -> ToolRegistry:
    """Return (and lazily create) the process-wide default registry."""
    global _default_registry  # noqa: PLW0603
    if _default_registry is None:
        with _default_lock:
            if _default_registry is None:
                _default_registry = ToolRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Replace the default registry with a fresh instance.

    Intended for test isolation — never call this in production.
    """
    global _default_registry  # noqa: PLW0603
    with _default_lock:
        _default_registry = ToolRegistry()
