# AI Tools Executor — Architecture

> **One-liner**: An executor layer that sits between AI agents and tools. The agent gets only **3 meta-tools** (`search_tools` + `execute` + `describe_tool`). Tools are discovered on-demand and invoked via **Python function call syntax** — not JSON.

---

## The Problem

Every AI agent framework today dumps **all tool schemas** into the LLM's context on every request.

```
Agent Context = System Prompt + ALL Tool Schemas + Conversation History
```

With 50 tools at ~500 tokens each, that's **25,000 tokens wasted every turn** — on tools that will never be used. This pollutes the context, wastes money, and degrades tool selection accuracy.

**Analogy**: It's like carrying an entire library on your back to read one chapter. You should carry the **index** and fetch chapters on demand.

---

## The Solution

The agent sees **exactly 3 tools** in its context:

| Meta-Tool | Purpose |
|---|---|
| `search_tools(query)` | Discover what tools are available |
| `execute(calls)` | Run one or more tool calls |
| `describe_tool(name)` | Get detailed docs + examples for a specific tool |

`search_tools` returns concise signatures. `describe_tool` is the deep-dive — only called when the agent is confused about how a tool works. Everything else — validation, execution, result formatting — happens **behind the scenes** inside the executor.

> **Note**: `describe_tool` is not compulsory. The system works with just `search_tools` + `execute`. It's a good-to-have that reduces retry loops when the agent needs more context about a tool's usage.

---

## How It Works

### Step 1 — Agent Searches

The agent describes **what capability it needs** — not a bare keyword, but the intent:

```python
# ✅ Good — expresses intent clearly
search_tools(query="web search for stock price information")
search_tools(query="send an email notification")
search_tools(query="get current weather for a city")

# ❌ Bad — too vague, no intent
search_tools(query="google")
search_tools(query="price")
```

The executor matches against tool names, descriptions, and tags, then returns **function signatures with concise docstrings**:

```python
# Results for: "web search for stock price information"

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information using a text query."""

def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price. symbol: ticker like 'GOOG', 'AAPL'."""
```

This is what the agent sees — **exact signatures it can call** plus a one-line docstring explaining what each tool does and returns. No verbose JSON schemas, no ambiguity.

**Cost**: ~100 tokens returned. Compare with loading 50 full schemas = 25,000 tokens.

### Step 2 — Agent Executes

The agent now knows the function signatures. It returns **Python function call syntax**:

```python
execute("get_stock_price(symbol='GOOG')")
```

Or **multiple calls at once**:

```python
execute("[get_stock_price(symbol='GOOG'), get_stock_price(symbol='AAPL'), search_web(query='tech market trends')]")
```

### Step 3 — Executor Parses, Validates, Runs

```
Agent returns:  get_stock_price(symbol='GOOG')
                        ↓
               ast.parse() — extract function name + args
                        ↓
               Validate against tool registry schema
                        ↓
               Call the actual Python function
                        ↓
               Return structured result to agent
```

No JSON parsing. No sandbox. No code execution risk. The executor **parses** the syntax tree (like an IDE does), extracts the function name and arguments, validates them, and calls the real function.

### Partial Failure on Multi-Call

When `execute` receives multiple calls and some succeed while others fail, it returns **partial results** — successes inline with failures:

```python
# Agent sends:
execute("[get_stock_price(symbol='GOOG'), get_weather(city='')]")

# Executor returns:
[
  {tool: "get_stock_price", status: "ok", result: {price: 182.63}},
  {tool: "get_weather", status: "error", error: "ExecutionError: ..."}
]
```

We don't throw away a valid stock price just because the weather call had a bad parameter. The agent gets everything that worked and can retry only what failed.

---

## Why Function Calls Instead of JSON?

This is the key design decision. Traditional tool calling uses JSON:

```json
{"tool": "get_stock_price", "params": {"symbol": "GOOG"}}
```

We use Python function call syntax instead:

```python
get_stock_price(symbol="GOOG")
```

### Comparison

| | JSON Tool Calling | Function Call Syntax |
|---|---|---|
| **Tokens** | ~20 per call | **~7 per call** (65% less) |
| **LLM fluency** | Synthetic format, less training data | Native — LLMs trained on billions of function calls |
| **Validation** | Custom JSON schema validator | **`ast.parse()`** — same as IDEs/linters |
| **Multi-call** | Verbose array of objects | `[fn1(), fn2(), fn3()]` — natural Python |
| **Error rate** | Higher (JSON syntax errors, missing braces) | Lower (LLMs write function calls all day) |
| **Composability** (future) | Can't express data flow | Can pass results: `fn2(data=fn1())` |

### Validation via AST

Python's `ast` module parses function call syntax safely without executing anything:

```python
import ast

raw = 'get_stock_price(symbol="GOOG")'
tree = ast.parse(raw, mode="eval")
# → Call(func=Name(id='get_stock_price'), keywords=[keyword(arg='symbol', value=Constant(value='GOOG'))])

# Extract: function_name = "get_stock_price", kwargs = {"symbol": "GOOG"}
# Validate: check function exists in registry, check arg types match schema
# Execute: call the actual registered function
```

This is exactly what IDEs, linters, and code highlighters do — parse the syntax tree, validate it, flag errors. Zero execution risk.

### Error Recovery

When the agent produces invalid syntax or bad parameters, the executor returns a **structured error** with the **exact exception message** — no translation, no context loss:

```
ExecutionError: Failed to parse tool call
  Input:    get_stock_price(symbol="GOOG'
  Error:    SyntaxError: EOL while scanning string literal (line 1, col 34)
  Expected: get_stock_price(symbol: str)
            """Fetch real-time stock price. symbol: ticker like 'GOOG', 'AAPL'."""
  Hint:     Use matching quotes — either "GOOG" or 'GOOG'
```

| Field | Required | What it contains |
|---|---|---|
| **Input** | ✅ | Exactly what the agent sent |
| **Error** | ✅ | Raw exception message — no translation, keep full context |
| **Expected** | Optional | The function signature + `description` (same as `search_tools` returns) — helps the agent self-correct |
| **Hint** | Optional | Actionable fix if an intelligent layer can generate one — skip if not available |

The `Expected` field returns the same `description` that `search_tools` would return — the agent already knows how to read that format.

### Parameter Validation

Two layers of validation happen before a tool runs:

**Layer 1 — AST-level** (handled by the executor):
- Syntax valid? (`ast.parse`)
- Function exists in registry?
- Required parameters provided?
- Parameter names match the signature?

On error → return the function signature + `description` so the agent can self-correct.

**Layer 2 — Value-level** (handled by the tool developer via Pydantic):
- Is `symbol` a valid ticker format?
- Is `max_results` within an acceptable range?

This lives in the tool's own function definition, not in the executor. The executor catches Pydantic `ValidationError` and passes it through in the same structured format with the **raw error message preserved**:

```
ExecutionError: Parameter validation failed
  Input:    get_stock_price(symbol="123INVALID")
  Error:    ValidationError: symbol — string does not match regex '^[A-Z]{1,5}$'
  Expected: get_stock_price(symbol: str)
            """Fetch real-time stock price. symbol: ticker like 'GOOG', 'AAPL'."""
```

Developer Pydantic errors are kept as-is. The executor's job is structuring them into the `Input` / `Error` / `Expected` format, not rewriting the messages.

### `describe_tool` — Deep-Dive When Confused

If `search_tools` returns concise signatures but the agent still isn't sure how to use a tool, it can call:

```python
describe_tool(name="search_web")
```

This returns the **full docstring with examples**:

```python
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information using a text query.

    Args:
        query: Natural language search query. Be specific for better results.
        max_results: Number of results to return (1-20).

    Examples:
        search_web(query="Python asyncio tutorial")
        search_web(query="GOOG stock price today", max_results=3)
    """
```

This is **only called when needed** — most of the time, the concise signature from `search_tools` is enough. Think of it as the agent flipping from the index to reading the full chapter.

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│                   AI Agent (LLM)                  │
│   Only sees: search_tools + execute + describe    │
└──────────┬──────────────────┬─────────────────────┘
           │                  │
     search_tools(q)     execute(calls)
           │                  │
┌──────────▼──────────────────▼─────────────────────┐
│               AI Tools Executor                    │
│                                                    │
│  ┌──────────────┐      ┌───────────────────────┐  │
│  │ Tool Search   │      │ Call Parser (AST)      │  │
│  │ • keyword     │      │ • ast.parse()          │  │
│  │ • fuzzy match │      │ • extract fn + args    │  │
│  │ • categories  │      │ • validate types       │  │
│  └──────┬───────┘      └───────────┬───────────┘  │
│         │                          │               │
│  ┌──────▼──────────────────────────▼───────────┐  │
│  │              Tool Registry                    │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │
│  │  │ Tool A  │ │ Tool B  │ │ Tool N  │ ...    │  │
│  │  └─────────┘ └─────────┘ └─────────┘       │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

### Components

**Tool Registry** — Stores all registered tools with their metadata (name, description, parameter schemas, categories, tags). Tools are registered via a `@tool` decorator. The registry is searchable but its contents are **never loaded into the agent's context**.

**Tool Search** — Receives natural language queries, matches against tool names/descriptions/tags, and returns compact summaries with function signatures. Strategies are pluggable (keyword, fuzzy, semantic embeddings).

**Call Parser** — Uses `ast.parse()` to safely extract function names and keyword arguments from the agent's response. Validates against the registry schema (type checking, required params, unknown params). No code is ever executed by the parser.

**Tool Executor** — After validation passes, calls the actual Python function registered for that tool name and returns the result.

---

## Tool Registration

Tools are Python functions with a `@tool` decorator:

```python
from ai_tools_executor import tool

@tool(
    description="Fetch real-time stock price. symbol: ticker like 'GOOG', 'AAPL'.",
    category="finance",
    tags=["stock", "price", "market"]
)
def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price for a given ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g. 'GOOG', 'AAPL', 'MSFT').
                Must be 1-5 uppercase letters.

    Examples:
        get_stock_price(symbol="GOOG")
        get_stock_price(symbol="AAPL")
    """
    # ... implementation
    return {"symbol": symbol, "price": 182.63, "currency": "USD"}


@tool(
    description="Search the web for information using a text query.",
    category="search",
    tags=["web", "search", "google"]
)
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information.

    Args:
        query: Natural language search query. Be specific for better results.
        max_results: Number of results to return (1-20).

    Examples:
        search_web(query="Python asyncio tutorial")
        search_web(query="GOOG stock price today", max_results=3)
    """
    # ... implementation
    return [{"title": "...", "url": "...", "snippet": "..."}]
```

The decorator registers two separate description fields in the registry:

| Field | Used by | Content | Source |
|---|---|---|---|
| `description` | `search_tools`, error `Expected` field | Short — what the tool does + param hints if names aren't clear | Decorator `description` param (required) |
| `doc_str` | `describe_tool` | Full — args, constraints, examples | **Always auto-processed from the function's docstring** |

`doc_str` is never set manually in the decorator — it's always extracted from the function's own docstring. This keeps the source of truth in one place and follows standard Python conventions.

The decorator also:
1. Extracts parameter names, types, and defaults from the function signature automatically
2. Registers the tool in the global registry
3. Does **not** inject anything into the agent's context

---

## Complete Flow Example

**User asks**: _"What's Google's stock price and what's the weather in London?"_

```
┌─ Turn 1 ──────────────────────────────────────────────────────┐
│ Agent context: system prompt + 3 meta-tools (~500 tokens)     │
│                                                                │
│ Agent thinks: "I need to look up a stock price and check       │
│                weather for a city"                             │
│                                                                │
│ Agent calls:                                                   │
│   search_tools(query="get stock price and weather for a city") │
│                                                                │
│ Executor returns:                                              │
│   def get_stock_price(symbol: str) -> dict:                    │
│       """Fetch real-time stock price.                          │
│       symbol: ticker like 'GOOG', 'AAPL'."""                  │
│                                                                │
│   def get_weather(city: str, units: str = "celsius") -> dict:  │
│       """Get current weather for a city."""                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌─ Turn 2 ──────────────────────────────────────────────────┐
│ Agent now knows the signatures                             │
│ Agent calls:                                               │
│   execute("[                                               │
│     get_stock_price(symbol='GOOG'),                        │
│     get_weather(city='London')                             │
│   ]")                                                      │
│                                                            │
│ Executor:                                                  │
│   1. ast.parse() → extracts 2 calls                        │
│   2. Validates both against registry ✓                     │
│   3. Runs both (can parallelize — they're independent)     │
│   4. Returns combined results                              │
│                                                            │
│ Results:                                                   │
│   [                                                        │
│     {tool: "get_stock_price", result: {price: 182.63}},    │
│     {tool: "get_weather", result: {temp: 12, sky: "cloudy"}} │
│   ]                                                        │
└────────────────────────────────────────────────────────────┘

┌─ Turn 3 ──────────────────────────────────────────────────┐
│ Agent responds to user with both results                   │
│ Total tool tokens used: ~500 (meta-tools) + ~100 (search)  │
│ Traditional approach would have used: ~25,000+             │
└────────────────────────────────────────────────────────────┘
```

---

## Token Savings

| Approach | Tokens for tools per turn | With 100 tools |
|---|---|---|
| Traditional (all schemas in context) | ~500 × N tools | **50,000** |
| JSON tool calling (lazy search) | ~500 + lazy load | ~1,000 |
| **Function call syntax (this project)** | **~500 + ~100** | **~600** |

Additional savings from function call syntax vs JSON per individual tool call: **~65% fewer tokens**.

---

## Prior Art & Inspiration

| Project | What they do | Our difference |
|---|---|---|
| **Cloudflare Code Mode** | 2 tools (search + execute), TypeScript code in V8 sandbox | 3 meta-tools, Python function syntax + AST parsing, no sandbox needed |
| **Anthropic Tool Search** | Lazy search, but still JSON tool calling after discovery | We skip JSON entirely, use function call syntax |
| **Spring AI Dynamic Discovery** | Semantic search for tools, portable across LLM providers | Similar search approach, but we add function-call execution |

---

## Design Principles

1. **Minimal context** — Agent only ever sees 3 meta-tools, never the full registry
2. **Native syntax** — Function calls, not JSON. LLMs are trained on code.
3. **Safe parsing** — `ast.parse()` validates syntax without execution. Like an IDE, not a runtime.
4. **Fail informatively** — Errors always include what was sent, what was expected, and why it failed. The agent should never need to guess.
5. **Registry never touches context** — No auto-injection, no framework magic. The only way tools reach the agent is through `search_tools`.
6. **Parallel execution** — Multiple independent calls in one `execute()` run concurrently, with partial results on partial failure.
7. **Provider agnostic** — Works with any LLM that can generate text (OpenAI, Anthropic, Gemini, local models)
8. **Pluggable search** — Swap keyword/fuzzy/semantic strategies via config
9. **Hot-reload** — Register and unregister tools at runtime without restarting
