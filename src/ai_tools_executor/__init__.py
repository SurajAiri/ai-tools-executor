"""AI Tools Executor — public API.

An executor layer that sits between AI agents and tools.
The agent gets only 3 meta-tools: ``search_tools``, ``execute``,
and ``describe_tool``.

Quick start::

    from ai_tools_executor import tool, ToolExecutor

    @tool(description="Add two numbers.", category="math", tags=["add"])
    def add(a: int, b: int) -> int:
        \"\"\"Add *a* and *b*.\"\"\"
        return a + b

    executor = ToolExecutor()
    print(executor.search_tools("add numbers"))
    print(executor.execute("add(a=1, b=2)"))
"""

from ai_tools_executor.decorator import ParameterInfo, ToolInfo, tool
from ai_tools_executor.exceptions import (
    ExecutionError,
    ParseError,
    ToolAlreadyRegisteredError,
    ToolExecutorError,
    ToolNotFoundError,
    ValidationError,
)
from ai_tools_executor.executor import ToolExecutor
from ai_tools_executor.registry import (
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
)
from ai_tools_executor.search import KeywordSearchStrategy, SearchStrategy

__all__ = [
    # Core
    "tool",
    "ToolExecutor",
    "ToolRegistry",
    # Models
    "ToolInfo",
    "ParameterInfo",
    # Search
    "SearchStrategy",
    "KeywordSearchStrategy",
    # Registry helpers
    "get_default_registry",
    "reset_default_registry",
    # Exceptions
    "ToolExecutorError",
    "ToolNotFoundError",
    "ToolAlreadyRegisteredError",
    "ParseError",
    "ValidationError",
    "ExecutionError",
]

__version__ = "0.1.0"