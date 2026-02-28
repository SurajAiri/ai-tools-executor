# AI Tools Executor — Architecture

> **One-liner**: An executor layer that sits between AI agents and tools. The agent gets only **2 meta-tools** (`search_tools` + `execute`). Tools are discovered on-demand and invoked via **Python function call syntax** — not JSON.

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

The agent sees **exactly 2 tools** in its context:

| Meta-Tool | Purpose |
|---|---|
| `search_tools(query)` | Discover what tools are available |
| `execute(calls)` | Run one or more tool calls |

Everything else — discovery, validation, execution, result formatting — happens **behind the scenes** inside the executor.

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

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│                   AI Agent (LLM)                  │
│          Only sees: search_tools + execute        │
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
    description="Get real-time stock price",
    category="finance",
    tags=["stock", "price", "market"]
)
def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price. symbol: ticker like 'GOOG', 'AAPL'."""
    # ... implementation
    return {"symbol": symbol, "price": 182.63, "currency": "USD"}


@tool(
    description="Search the web for information",
    category="search",
    tags=["web", "search", "google"]
)
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information using a text query."""
    # ... implementation
    return [{"title": "...", "url": "...", "snippet": "..."}]
```

The decorator:
1. Extracts parameter names, types, and defaults from the function signature
2. Registers the tool in the global registry
3. Does **not** inject anything into the agent's context

---

## Complete Flow Example

**User asks**: _"What's Google's stock price and what's the weather in London?"_

```
┌─ Turn 1 ──────────────────────────────────────────────────────┐
│ Agent context: system prompt + 2 meta-tools (~500 tokens)     │
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
| **Cloudflare Code Mode** | 2 tools (search + execute), TypeScript code in V8 sandbox | We use Python function syntax + AST parsing, no sandbox needed |
| **Anthropic Tool Search** | Lazy search, but still JSON tool calling after discovery | We skip JSON entirely, use function call syntax |
| **Spring AI Dynamic Discovery** | Semantic search for tools, portable across LLM providers | Similar search approach, but we add function-call execution |

---

## Design Principles

1. **Minimal context** — Agent only ever sees 2 tools, never the full registry
2. **Native syntax** — Function calls, not JSON. LLMs are trained on code.
3. **Safe parsing** — `ast.parse()` validates syntax without execution. Like an IDE, not a runtime.
4. **Parallel execution** — Multiple independent calls in one `execute()` run concurrently
5. **Provider agnostic** — Works with any LLM that can generate text (OpenAI, Anthropic, Gemini, local models)
6. **Pluggable search** — Swap keyword/fuzzy/semantic strategies via config
7. **Hot-reload** — Register and unregister tools at runtime without restarting
