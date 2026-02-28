"""Tool Executor — the main public interface.

Exposes the three meta-tools that an AI agent interacts with:

* :meth:`ToolExecutor.search_tools` — discover available tools
* :meth:`ToolExecutor.execute` — parse, validate, and run tool calls
* :meth:`ToolExecutor.describe_tool` — get detailed docs for one tool
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

from ai_tools_executor.exceptions import (
    ExecutionError,
    ToolExecutorError,
    ToolNotFoundError,
)
from ai_tools_executor.parser import ParsedCall, parse_calls
from ai_tools_executor.registry import ToolRegistry, get_default_registry
from ai_tools_executor.search import (
    KeywordSearchStrategy,
    SearchStrategy,
    format_search_results,
)


class ToolExecutor:
    """Orchestrator for the 3 meta-tool interface.

    Parameters
    ----------
    registry:
        Override the process-wide default :class:`ToolRegistry`.
    search_strategy:
        Override the default :class:`KeywordSearchStrategy`.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        search_strategy: SearchStrategy | None = None,
    ) -> None:
        self.registry = registry or get_default_registry()
        self.search_strategy = search_strategy or KeywordSearchStrategy()

    # ── Meta-tool 1: search_tools ─────────────────────────────────────

    def search_tools(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> str:
        """Discover tools matching a natural-language *query*.

        Returns compact function signatures with one-line descriptions,
        ready to be consumed by the agent.
        """
        all_tools = self.registry.list_all()
        matched = self.search_strategy.search(
            query, all_tools, max_results=max_results,
        )
        return format_search_results(matched, query)

    # ── Meta-tool 2: execute ──────────────────────────────────────────

    def execute(self, calls: str) -> list[dict[str, Any]]:
        """Parse, validate, and run one or more tool calls.

        Parameters
        ----------
        calls:
            A string containing either a single Python function call
            (``'fn(a=1)'``) or a list (``'[fn1(), fn2()]'``).

        Returns
        -------
        list[dict]:
            One result dict per call, each with keys:

            - ``tool`` — the tool name
            - ``status`` — ``"ok"`` or ``"error"``
            - ``result`` — the return value (on success)
            - ``error`` — formatted error string (on failure)
        """
        # Phase 1: parse & validate all calls
        try:
            parsed: list[ParsedCall] = parse_calls(calls, self.registry)
        except ToolExecutorError as exc:
            return [{"tool": "unknown", "status": "error", "error": exc.format()}]

        # Phase 2: execute each call independently (partial failure)
        results: list[dict[str, Any]] = []
        for pc in parsed:
            results.append(self._run_single(pc, calls))

        return results

    async def execute_async(self, calls: str) -> list[dict[str, Any]]:
        """Async version of :meth:`execute`.

        Independent calls are run concurrently via
        :func:`asyncio.gather`.
        """
        try:
            parsed: list[ParsedCall] = parse_calls(calls, self.registry)
        except ToolExecutorError as exc:
            return [{"tool": "unknown", "status": "error", "error": exc.format()}]

        tasks = [
            asyncio.to_thread(self._run_single, pc, calls) for pc in parsed
        ]
        return list(await asyncio.gather(*tasks))

    # ── Meta-tool 3: describe_tool ────────────────────────────────────

    def describe_tool(self, name: str) -> str:
        """Return the full signature + complete docstring for *name*.

        This is the "deep-dive" meta-tool — only called when the agent
        needs more context than ``search_tools`` provides.
        """
        try:
            tool_info = self.registry.get(name)
        except ToolNotFoundError as exc:
            return exc.format()
        return tool_info.full_description()

    # ── Internals ─────────────────────────────────────────────────────

    def _run_single(
        self,
        pc: ParsedCall,
        raw_input: str,
    ) -> dict[str, Any]:
        """Execute a single :class:`ParsedCall` and return a result dict."""
        try:
            result = pc.tool_info.func(**pc.kwargs)
            return {"tool": pc.name, "status": "ok", "result": result}
        except Exception as exc:  # noqa: BLE001
            err = ExecutionError(
                str(exc),
                input_text=raw_input,
                expected=pc.tool_info.short_summary(),
                hint=_extract_hint(exc),
            )
            return {"tool": pc.name, "status": "error", "error": err.format()}


def _extract_hint(exc: Exception) -> str | None:
    """Try to produce a helpful hint from the exception.

    Returns ``None`` when no actionable suggestion can be made.
    """
    # Pydantic ValidationError
    if type(exc).__name__ == "ValidationError" and hasattr(exc, "errors"):
        try:
            msgs = [
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in exc.errors()  # type: ignore[union-attr]
            ]
            return "Validation issues: " + "; ".join(msgs)
        except Exception:  # noqa: BLE001
            pass
    return None
