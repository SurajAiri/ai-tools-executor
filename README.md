# AI Tools Executor

> An executor layer between AI agents and tools. The agent gets only **3 meta-tools** — tools are discovered on-demand and invoked via **Python function call syntax**, not JSON.

## The Problem

Every agent framework dumps **all tool schemas** into the LLM context on every turn. With 50 tools at ~500 tokens each, that's **25,000 tokens wasted** — on tools that will never be used.

## The Solution

The agent sees exactly **3 meta-tools**:

| Meta-Tool             | Purpose                               |
| --------------------- | ------------------------------------- |
| `search_tools(query)` | Discover available tools              |
| `execute(calls)`      | Run one or more tool calls            |
| `describe_tool(name)` | Get detailed docs for a specific tool |

Everything else — validation, execution, result formatting — happens behind the scenes.

## Installation

```bash
pip install ai-tools-executor
```

**Or with [uv](https://github.com/astral-sh/uv):**

```bash
uv add ai-tools-executor
```

## Quick Start

### 1. Register Tools

```python
from ai_tools_executor import tool

@tool(description="Fetch real-time stock price.", category="finance", tags=["stock", "price"])
def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price for a ticker symbol.

    Args:
        symbol: Stock ticker (e.g. 'GOOG', 'AAPL').

    Examples:
        get_stock_price(symbol="GOOG")
    """
    return {"symbol": symbol, "price": 182.63, "currency": "USD"}


@tool(description="Search the web for information.", category="search", tags=["web", "google"])
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using a text query.

    Args:
        query: Natural language search query.
        max_results: Number of results to return (1-20).
    """
    return [{"title": "...", "url": "...", "snippet": "..."}]
```

### 2. Use the Executor

```python
from ai_tools_executor import ToolExecutor

executor = ToolExecutor()

# Meta-tool 1: Search for tools by intent
print(executor.search_tools("stock price lookup"))
# def get_stock_price(symbol: str) -> dict:
#     """Fetch real-time stock price."""

# Meta-tool 2: Execute tool calls using Python syntax
results = executor.execute("get_stock_price(symbol='GOOG')")
r = results[0]
r.ok        # True
r.tool      # "get_stock_price"
r.result    # {"symbol": "GOOG", "price": 182.63, "currency": "USD"}

# Execute multiple calls at once
results = executor.execute(
    "[get_stock_price(symbol='GOOG'), search_web(query='market trends')]"
)

# Serialise for transport when needed
r.to_dict()  # plain dict
r.to_json()  # JSON string

# Meta-tool 3: Get detailed docs when needed
print(executor.describe_tool("search_web"))
```

## Key Features

### Function Call Syntax (not JSON)

```python
# Traditional agent tool calling (JSON — verbose, error-prone)
{"tool": "get_stock_price", "params": {"symbol": "GOOG"}}

# This package (Python syntax — native, token-efficient)
get_stock_price(symbol="GOOG")
```

|                 | JSON Tool Calling            | Function Call Syntax                           |
| --------------- | ---------------------------- | ---------------------------------------------- |
| **Tokens**      | ~20 per call                 | **~7 per call** (65% less)                     |
| **LLM fluency** | Synthetic format             | Native — trained on billions of function calls |
| **Validation**  | Custom JSON schema validator | **`ast.parse()`** — same as IDEs/linters       |
| **Error rate**  | Higher (JSON syntax errors)  | Lower (function calls are natural to LLMs)     |

### Safe AST Parsing

No code is ever executed. The parser uses Python's `ast` module to extract function names and arguments — exactly like an IDE or linter:

```python
import ast
tree = ast.parse('get_stock_price(symbol="GOOG")', mode="eval")
# Extracts: function_name="get_stock_price", kwargs={"symbol": "GOOG"}
# Validates against registry, then calls the real function
```

### Partial Failure on Multi-Call

When some calls succeed and others fail, you get **both**:

```python
results = executor.execute(
    "[get_stock_price(symbol='GOOG'), bad_tool(x=1)]"
)
results[0].ok      # True  — stock price succeeded
results[0].result  # {"symbol": "GOOG", "price": 182.63, ...}
results[1].ok      # False — bad_tool failed
results[1].error   # "ToolNotFoundError: Tool 'bad_tool' not found ..."
```

### Structured Error Messages

Errors follow a consistent format so the agent can self-correct:

```
ValidationError: Missing required parameter(s) for 'get_stock_price': symbol
  Input:    get_stock_price()
  Error:    Missing required parameter(s) for 'get_stock_price': symbol
  Expected: def get_stock_price(symbol: str) -> dict:
            """Fetch real-time stock price."""
  Hint:     Required: symbol
```

### Pluggable Search

Swap search strategies via the `SearchStrategy` ABC:

```python
from ai_tools_executor import ToolExecutor, SearchStrategy

class SemanticSearch(SearchStrategy):
    def search(self, query, tools, *, max_results=5):
        # Your embedding-based search here
        ...

executor = ToolExecutor(search_strategy=SemanticSearch())
```

### Async Execution

Run independent tool calls concurrently:

```python
results = await executor.execute_async(
    "[get_stock_price(symbol='GOOG'), get_weather(city='London')]"
)
```

### Hot-Reload

Register and unregister tools at runtime:

```python
from ai_tools_executor import get_default_registry

registry = get_default_registry()
registry.unregister("old_tool")
# Register new tools with @tool — no restart needed
```

## Architecture

```
AI Agent (LLM)
  │  Only sees: search_tools + execute + describe_tool
  │
  ├── search_tools(query) ──► Tool Search ──► Registry ──► Ranked results
  │
  ├── execute(calls) ──► AST Parser ──► Validator ──► Tool Function ──► Results
  │
  └── describe_tool(name) ──► Registry ──► Full docstring
```

For the complete architecture document, see [docs/architecture.md](docs/architecture.md).

## Project Structure

```
src/ai_tools_executor/
├── __init__.py      # Public API
├── decorator.py     # @tool decorator, ToolInfo, ParameterInfo
├── exceptions.py    # Structured error hierarchy
├── executor.py      # ToolExecutor (3 meta-tools)
├── models.py        # ToolCallResult, CallStatus (frozen dataclasses)
├── parser.py        # AST call parser + Layer 1 validation
├── registry.py      # Thread-safe ToolRegistry
├── meta_tools.py    # Meta-tool schema + handle_tool_call handler
└── search.py        # Pluggable search strategies
```

## Development

```bash
# Clone and install
git clone https://github.com/surajairi/ai-tools-executor.git
cd ai-tools-executor
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

## Requirements

- Python ≥ 3.10
- No external dependencies (stdlib only)

## License

[MIT](LICENSE)
