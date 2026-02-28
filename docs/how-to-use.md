# How to Use AI Tools Executor

A comprehensive guide for developers integrating **ai-tools-executor** into their AI agent projects.

---

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Quick Start](#quick-start)
4. [LLM Integration](#llm-integration)
5. [Advanced Usage](#advanced-usage)
6. [Error Handling](#error-handling)
7. [Logging](#logging)
8. [API Reference](#api-reference)

---

## Installation

```bash
pip install ai-tools-executor
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add ai-tools-executor
```

**Requirements:** Python ≥ 3.10 — zero external dependencies.

---

## Core Concepts

### The Problem

Traditional agent frameworks dump **all tool schemas** into LLM context every turn. With 50 tools at ~500 tokens each, that's ~25,000 tokens wasted on tools that may never be used.

### The Solution: 3 Meta-Tools

Instead of exposing every tool, the agent sees only **3 meta-tools**:

| Meta-Tool             | What It Does                                     |
| --------------------- | ------------------------------------------------ |
| `search_tools(query)` | Discover tools by describing a capability        |
| `execute(calls)`      | Run tool calls using Python function-call syntax |
| `describe_tool(name)` | Get full documentation for a specific tool       |

### Why Function-Call Syntax?

Tools are invoked using **Python syntax**, not JSON:

```python
# ❌ Traditional JSON (verbose, error-prone)
{"tool": "get_stock_price", "params": {"symbol": "GOOG"}}

# ✅ This package (native, token-efficient)
get_stock_price(symbol="GOOG")
```

The parser uses Python's `ast` module — **no code is ever executed**. It's the same technique IDEs and linters use.

---

## Quick Start

**Note: check [full_example.py](/playground/full_example.py) for a complete example.**

### Step 1: Register Tools

Use the `@tool` decorator to register functions:

```python
from ai_tools_executor import tool

@tool(
    description="Fetch real-time stock price. symbol: ticker like 'GOOG', 'AAPL'.",
    category="finance",
    tags=["stock", "price", "market"],
)
def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price for a ticker symbol.

    Args:
        symbol: Stock ticker (e.g. 'GOOG', 'AAPL').

    Examples:
        get_stock_price(symbol="GOOG")
    """
    # Your real API call here
    return {"symbol": symbol, "price": 182.63, "currency": "USD"}


@tool(
    description="Search the web for information. query: natural language search string.",
    category="search",
    tags=["web", "google"],
)
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using a text query.

    Args:
        query: Natural language search query.
        max_results: Number of results to return (1-20).
    """
    return [{"title": "...", "url": "...", "snippet": "..."}]
```

**Key points about `@tool`:**

- `description` — short, one-line summary shown in search results
- `category` — logical grouping for search scoring (e.g. `"finance"`, `"search"`)
- `tags` — extra keywords to improve search matching
- The function's actual **docstring** is used by `describe_tool()` for detailed docs
- The function is returned **unmodified** — the decorator is transparent

> **💡 Recommended:** Include brief argument descriptions in `description` when the parameter name alone isn't self-explanatory. The `description` is what `search_tools` shows to the LLM — if it includes arg hints, the LLM can often call `execute` directly without needing `describe_tool`, saving a round-trip.
>
> ```python
> # ❌ Vague — LLM may not know what "symbol" means
> @tool(description="Fetch real-time stock price.")
>
> # ✅ Clear — LLM can execute immediately after search
> @tool(description="Fetch real-time stock price. symbol: ticker like 'GOOG', 'AAPL'.")
> ```

### Step 2: Create an Executor

```python
from ai_tools_executor import ToolExecutor

executor = ToolExecutor()
```

The executor picks up all `@tool`-decorated functions from the default registry automatically.

### Step 3: Search for Tools

```python
result = executor.search_tools("stock price lookup")
print(result)
```

Output:

```
# Tools matching: "stock price lookup"
def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price."""
```

### Step 4: Execute Tool Calls

```python
# Single call
results = executor.execute("get_stock_price(symbol='GOOG')")
r = results[0]
print(r.ok)      # True
print(r.tool)    # "get_stock_price"
print(r.result)  # {"symbol": "GOOG", "price": 182.63, "currency": "USD"}

# Multiple calls at once
results = executor.execute(
    "[get_stock_price(symbol='GOOG'), search_web(query='market trends')]"
)
for r in results:
    print(f"{r.tool}: ok={r.ok}, result={r.result}")
```

### Step 5: Handle Results

Each result is a `ToolCallResult` dataclass:

```python
r = results[0]

# Check success
if r.ok:
    print(r.result)     # The tool function's return value
else:
    print(r.error)      # Structured error string

# Serialise for transport
r.to_dict()   # Plain dict (status enum → string)
r.to_json()   # JSON string
```

### Step 6: Get Detailed Docs (Optional)

```python
print(executor.describe_tool("search_web"))
```

Output:

```
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using a text query.

    Args:
        query: Natural language search query.
        max_results: Number of results to return (1-20).
    """
```

---

## LLM Integration

The package provides two helpers to plug directly into any OpenAI-compatible API (OpenAI, LiteLLM, Groq, etc.):

### `get_meta_tools_schema()`

Returns the 3 meta-tool definitions in OpenAI function-calling format:

```python
from ai_tools_executor import get_meta_tools_schema

tools = get_meta_tools_schema()
# Pass `tools` to your LLM's completion call
```

### `handle_tool_call(executor, tool_name, arguments)`

Routes a meta-tool call to the executor — no manual `if/elif` needed:

```python
from ai_tools_executor import handle_tool_call

result_str = handle_tool_call(executor, "search_tools", {"query": "stock"})
# Returns a string ready to send back to the LLM
```

### Full Agent Loop Example

```python
import json
import litellm
from ai_tools_executor import (
    ToolExecutor,
    get_meta_tools_schema,
    handle_tool_call,
    tool,
)

# 1. Register your tools (import them or define them here)
#    ... @tool-decorated functions ...

# 2. Set up
executor = ToolExecutor()
meta_tools = get_meta_tools_schema()

SYSTEM_PROMPT = """You have 3 meta-tools:
1. search_tools(query) — discover tools by capability
2. execute(calls) — run tools using Python syntax
3. describe_tool(name) — get detailed docs for a tool

Workflow: search → execute → summarise for user.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's Google's stock price?"},
]

# 3. Agent loop
for turn in range(10):
    response = litellm.completion(
        model="your-model",
        messages=messages,
        tools=meta_tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message.model_dump())

        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            # Single dispatch — handles all 3 meta-tools
            result = handle_tool_call(executor, fn_name, fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        # Final text answer
        print(message.content)
        break
```

See [`playground/full_example.py`](../playground/full_example.py) for a complete runnable version.

---

## Advanced Usage

### Custom Search Strategy

Swap the default keyword search for something more sophisticated:

```python
from ai_tools_executor import ToolExecutor, SearchStrategy

class SemanticSearch(SearchStrategy):
    """Embedding-based search over registered tools."""

    def search(self, query, tools, *, max_results=5):
        # Your vector similarity logic here
        ...

executor = ToolExecutor(search_strategy=SemanticSearch())
```

### Custom Registry

Use a separate registry (useful for multi-tenant setups or testing):

```python
from ai_tools_executor import ToolRegistry, ToolExecutor

my_registry = ToolRegistry()

# Manually register tools into this registry
# (The @tool decorator uses the default registry)
executor = ToolExecutor(registry=my_registry)
```

### Async Execution

Run independent tool calls concurrently:

```python
results = await executor.execute_async(
    "[get_stock_price(symbol='GOOG'), get_weather(city='London')]"
)
```

Uses `asyncio.to_thread` under the hood — sync tool functions are automatically wrapped.

### Hot-Reload

Register and unregister tools at runtime without restarting:

```python
from ai_tools_executor import get_default_registry

registry = get_default_registry()
registry.unregister("deprecated_tool")
# New @tool decorations are picked up immediately
```

---

## Error Handling

### Exception Hierarchy

```
ToolExecutorError (base)
├── ToolNotFoundError      — unknown tool name
├── ToolAlreadyRegisteredError — duplicate registration
├── ParseError             — ast.parse() failure (syntax)
├── ValidationError        — bad params, missing args, etc.
└── ExecutionError         — tool function raised at runtime
```

### Structured Error Format

All errors follow a consistent format so LLMs can self-correct:

```
ValidationError: Missing required parameter(s) for 'get_stock_price': symbol
  Input:    get_stock_price()
  Error:    Missing required parameter(s) for 'get_stock_price': symbol
  Expected: def get_stock_price(symbol: str) -> dict:
            """Fetch real-time stock price."""
  Hint:     Required: symbol
```

Fields:

- **Input** — what the agent sent
- **Error** — what went wrong
- **Expected** — the correct function signature (so the agent can retry)
- **Hint** — an actionable suggestion (optional)

### Catching Errors Programmatically

```python
from ai_tools_executor import (
    ToolExecutorError,
    ParseError,
    ValidationError,
    ExecutionError,
    ToolNotFoundError,
)

try:
    results = executor.execute("bad_syntax((")
except ToolExecutorError as e:
    print(e.format())  # Structured error string
```

> **Note:** `executor.execute()` catches all errors internally and returns them as `ToolCallResult` objects with `status="error"`. You only need `try/except` if calling lower-level APIs directly.

### Partial Failure

When executing multiple calls, each call is independent:

```python
results = executor.execute(
    "[get_stock_price(symbol='GOOG'), nonexistent_tool()]"
)
results[0].ok     # True  — succeeded
results[1].ok     # False — ToolNotFoundError
results[1].error  # Structured error string
```

---

## Logging

The package uses Python's standard `logging` module. By default, **no output is produced** (NullHandler pattern).

### Quick Setup

```python
from ai_tools_executor import setup_logging

setup_logging()          # INFO level → stderr
setup_logging("DEBUG")   # Verbose — shows parse attempts, registry ops, etc.
```

### Standard Library Configuration

```python
import logging

logging.basicConfig(level=logging.DEBUG)
# Or target just this package:
logging.getLogger("ai_tools_executor").setLevel(logging.DEBUG)
```

### What Gets Logged

| Level     | Events                                                                          |
| --------- | ------------------------------------------------------------------------------- |
| `DEBUG`   | Parse attempts, search queries, registry register/unregister/clear, tool lookup |
| `INFO`    | Search match count, successful tool execution                                   |
| `WARNING` | Parse/validation failures, tool-not-found in describe                           |
| `ERROR`   | Tool function runtime exceptions                                                |

---

## API Reference

### Decorator

| Symbol                                  | Description                                            |
| --------------------------------------- | ------------------------------------------------------ |
| `@tool(description, *, category, tags)` | Register a function as a tool                          |
| `ToolInfo`                              | Frozen dataclass — full metadata for a registered tool |
| `ParameterInfo`                         | Frozen dataclass — metadata for a single parameter     |

### Executor

| Method                                       | Description                             |
| -------------------------------------------- | --------------------------------------- |
| `ToolExecutor(*, registry, search_strategy)` | Create an executor                      |
| `.search_tools(query, *, max_results=5)`     | Discover tools matching a query         |
| `.execute(calls)`                            | Parse + validate + run tool call(s)     |
| `.execute_async(calls)`                      | Async version with concurrent execution |
| `.describe_tool(name)`                       | Get full signature + docstring          |

### Result Models

| Symbol           | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `ToolCallResult` | Frozen dataclass with `.tool`, `.status`, `.result`, `.error` |
| `CallStatus`     | Enum: `OK`, `ERROR`                                           |
| `.ok`            | Property — `True` when status is OK                           |
| `.to_dict()`     | Serialise to plain dict                                       |
| `.to_json()`     | Serialise to JSON string                                      |

### Meta-Tool Helpers

| Function                                           | Description                                 |
| -------------------------------------------------- | ------------------------------------------- |
| `get_meta_tools_schema()`                          | Returns 3 tool definitions in OpenAI format |
| `handle_tool_call(executor, tool_name, arguments)` | Dispatch + return string result             |

### Registry

| Symbol                     | Description                                  |
| -------------------------- | -------------------------------------------- |
| `ToolRegistry`             | Thread-safe store for tool metadata          |
| `get_default_registry()`   | Get/create the process-wide default registry |
| `reset_default_registry()` | Replace with a fresh instance (testing)      |

### Search

| Symbol                  | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `SearchStrategy`        | ABC — implement `.search()` for custom strategies |
| `KeywordSearchStrategy` | Default keyword-overlap scorer                    |

### Logging

| Function                        | Description                 |
| ------------------------------- | --------------------------- |
| `setup_logging(level, handler)` | Quick logging configuration |

### Exceptions

| Exception                    | When                              |
| ---------------------------- | --------------------------------- |
| `ToolExecutorError`          | Base class for all package errors |
| `ToolNotFoundError`          | Unknown tool name                 |
| `ToolAlreadyRegisteredError` | Duplicate `@tool` registration    |
| `ParseError`                 | `ast.parse()` failure             |
| `ValidationError`            | Bad/missing parameters            |
| `ExecutionError`             | Tool function raised at runtime   |
