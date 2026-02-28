"""Full end-to-end example: AI Tools Executor + LiteLLM.

Demonstrates a real agent loop where the LLM only sees the 3 meta-tools
(search_tools, execute, describe_tool) and uses them to discover and
invoke tools dynamically.

Requirements:
    pip install litellm

Usage:
    # Set your API key
    export GEMINI_API_KEY="your-key-here"

    # Run
    python playground/full_example.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import litellm

from ai_tools_executor import ToolExecutor, get_meta_tools_schema, handle_tool_call, tool

# ─── Register some sample tools ──────────────────────────────────────


@tool(
    description="Fetch real-time stock price. symbol: ticker like 'GOOG', 'AAPL'.",
    category="finance",
    tags=["stock", "price", "market"],
)
def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price for a ticker symbol.

    Args:
        symbol: Stock ticker (e.g. 'GOOG', 'AAPL', 'MSFT').

    Examples:
        get_stock_price(symbol="GOOG")
        get_stock_price(symbol="AAPL")
    """
    # Simulated data — replace with a real API call
    prices = {
        "GOOG": 182.63,
        "AAPL": 227.45,
        "MSFT": 415.20,
        "AMZN": 198.30,
    }
    price = prices.get(symbol.upper(), round(100 + hash(symbol) % 200, 2))
    return {"symbol": symbol.upper(), "price": price, "currency": "USD"}


@tool(
    description="Search the web for information using a text query.",
    category="search",
    tags=["web", "search", "google"],
)
def search_web(query: str, max_results: int = 3) -> list[dict]:
    """Search the web for information.

    Args:
        query: Natural language search query.
        max_results: Number of results to return (1-10).

    Examples:
        search_web(query="Python asyncio tutorial")
        search_web(query="GOOG stock price today", max_results=3)
    """
    # Simulated — replace with a real search API
    return [
        {
            "title": f"Result {i + 1} for '{query}'",
            "url": f"https://example.com/{i + 1}",
            "snippet": f"Relevant information about {query}...",
        }
        for i in range(max_results)
    ]


@tool(
    description="Get current weather for a city.",
    category="weather",
    tags=["weather", "temperature", "forecast"],
)
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: City name (e.g. 'London', 'New York', 'Tokyo').
        units: Temperature units — 'celsius' or 'fahrenheit'.

    Examples:
        get_weather(city="London")
        get_weather(city="New York", units="fahrenheit")
    """
    # Simulated
    temps = {"London": 12, "New York": 8, "Tokyo": 15, "Mumbai": 32}
    temp = temps.get(city, 20)
    if units == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
    return {"city": city, "temperature": temp, "units": units, "sky": "cloudy"}


@tool(
    description="Calculate a mathematical expression.",
    category="math",
    tags=["calculate", "math", "arithmetic"],
)
def calculate(expression: str) -> dict:
    """Safely evaluate a mathematical expression.

    Args:
        expression: A math expression like '2 + 3 * 4' or '100 / 7'.

    Examples:
        calculate(expression="2 + 3 * 4")
        calculate(expression="(100 - 20) / 4")
    """
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        raise ValueError(f"Invalid characters in expression: {expression}")
    result = eval(expression)  # noqa: S307 — input is sanitised
    return {"expression": expression, "result": round(result, 6)}


# ─── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant with access to tools.

You have exactly 3 meta-tools to interact with external capabilities:

1. **search_tools(query)** — Discover tools by describing what you need.
   Returns function signatures you can call.

2. **execute(calls)** — Run tool calls using Python function-call syntax.
   Single: execute("[tool_name(arg='value')]")
   Multiple: execute("[tool1(a=1), tool2(b=2)]")

3. **describe_tool(name)** — Get detailed documentation for a specific tool.
   Use this only when you need more detail about how a tool works.

**Workflow:**
- First, search for relevant tools
- Then, execute the tools you found
- Finally, summarize the results for the user

Always search before executing — never guess tool names.
When calling execute, use the exact function signatures returned by search_tools.
"""

# ─── Build the 3 meta-tool schemas for LiteLLM ──────────────────────

META_TOOLS = get_meta_tools_schema()


# ─── Agent loop ──────────────────────────────────────────────────────


def run_agent(
    user_message: str,
    model: str = "groq/openai/gpt-oss-20b",
    # model: str = "groq/qwen/qwen3-32b",
    max_turns: int = 10,
) -> str:
    """Run a full agent loop with tool calling.

    The LLM only sees the 3 meta-tools. It searches for tools,
    executes them, and formulates a final answer.
    """
    executor = ToolExecutor()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        response = litellm.completion(
            model=model,
            messages=messages,
            tools=META_TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        # If the model wants to call tools
        if message.tool_calls:
            # Add assistant message with tool calls
            messages.append(message.model_dump())

            for tc in message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                print(f"  🔧 {fn_name}({fn_args})")
                result = handle_tool_call(executor, fn_name, fn_args)

                # Print errors fully in red, truncate successful results
                RED = "\033[91m"
                RESET = "\033[0m"
                try:
                    result_json = json.loads(result)
                    if isinstance(result_json, list):
                        for r in result_json:
                            if r.get("status") == "error":
                                print(f"{RED}  ❌ {json.dumps(r, indent=2)}{RESET}")
                            else:
                                r_str = json.dumps(r)
                                print(
                                    f"  📎 {r_str[:200]}{'...' if len(r_str) > 200 else ''}"
                                )
                    else:
                        if result_json.get("status") == "error":
                            print(f"{RED}  ❌ {result}{RESET}")
                        else:
                            print(
                                f"  📎 {result[:200]}{'...' if len(result) > 200 else ''}"
                            )
                except json.JSONDecodeError:
                    print(f"  📎 {result[:200]}{'...' if len(result) > 200 else ''}")

                # Add tool result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        else:
            # Model gave a final text response
            final = message.content
            print(f"\n✅ Final answer:\n{final}")
            return final

    return "Max turns reached without a final answer."


# ─── Main ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Default question, or pass your own as a CLI argument
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What's Google's stock price and what's the weather in London? Also calculate 42 * 17."
    )

    print(f"❓ User: {question}")
    run_agent(question)
