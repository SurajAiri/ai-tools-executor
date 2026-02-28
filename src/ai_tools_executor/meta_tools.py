"""Meta-tools — schema definitions and dispatch handler.

Provides:

* :func:`get_meta_tools_schema` — auto-generate the 3 meta-tool JSON
  definitions for LLM integrations (OpenAI, LiteLLM, etc.).
* :func:`handle_tool_call` — dispatch a meta-tool call to the
  :class:`~ai_tools_executor.executor.ToolExecutor` and return a string
  ready to be sent back to the LLM.

Usage::

    from ai_tools_executor import ToolExecutor, get_meta_tools_schema, handle_tool_call

    tools = get_meta_tools_schema()          # pass to LiteLLM / OpenAI
    executor = ToolExecutor()

    # Inside your agent loop:
    result_str = handle_tool_call(executor, "search_tools", {"query": "stock"})
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_tools_executor.executor import ToolExecutor


# ─── Schema ──────────────────────────────────────────────────────────


def get_meta_tools_schema() -> list[dict]:
    """Return the 3 meta-tool definitions in OpenAI-compatible format.

    Each entry follows the ``{"type": "function", "function": {...}}``
    structure expected by OpenAI, LiteLLM, and compatible providers.

    Returns
    -------
    list[dict]
        A list of 3 tool-schema dicts for ``search_tools``,
        ``execute``, and ``describe_tool``.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": (
                    "Search for available tools by describing what "
                    "capability you need. Returns function signatures."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural language description of the "
                                "capability you need."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute",
                "description": (
                    "Execute one or more tool calls using "
                    "Python function-call syntax."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calls": {
                            "type": "string",
                            "description": (
                                "A Python list of function calls to execute. "
                                "Always pass a list, even for a single tool. "
                                'Example: "[get_stock_price(symbol=\'GOOG\'), '
                                "get_weather(city='London')]\""
                            ),
                        },
                    },
                    "required": ["calls"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "describe_tool",
                "description": (
                    "Get detailed documentation and examples for a "
                    "specific tool. Use when you need more detail."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the tool to describe.",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
    ]


# ─── Handler ─────────────────────────────────────────────────────────


def handle_tool_call(
    executor: ToolExecutor,
    tool_name: str,
    arguments: dict,
) -> str:
    """Dispatch a meta-tool call to *executor* and return a string result.

    This is the single routing function developers need in their agent
    loop — no manual ``if/elif`` chain required.

    Parameters
    ----------
    executor:
        A :class:`~ai_tools_executor.executor.ToolExecutor` instance.
    tool_name:
        One of ``"search_tools"``, ``"execute"``, or ``"describe_tool"``.
    arguments:
        The arguments dict parsed from the LLM's tool-call response.

    Returns
    -------
    str
        A string (often JSON) ready to be sent back to the LLM as the
        tool-call result.

    Examples
    --------
    >>> handle_tool_call(executor, "search_tools", {"query": "stock price"})
    >>> handle_tool_call(
    ...     executor, "execute",
    ...     {"calls": "get_stock_price(symbol='GOOG')"},
    ... )
    >>> handle_tool_call(
    ...     executor, "describe_tool", {"name": "get_stock_price"},
    ... )
    """
    if tool_name == "search_tools":
        return executor.search_tools(arguments["query"])
    if tool_name == "execute":
        results = executor.execute(arguments["calls"])
        return json.dumps(
            [r.to_dict() for r in results],
            indent=2,
        )
    if tool_name == "describe_tool":
        return executor.describe_tool(arguments["name"])
    return json.dumps({"error": f"Unknown meta-tool: {tool_name}"})
