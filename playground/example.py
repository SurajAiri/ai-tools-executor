from ai_tools_executor import ToolExecutor, tool


@tool(
    description="Fetch real-time stock price.",
    category="finance",
    tags=["stock", "price"],
)
def get_stock_price(symbol: str) -> dict:
    """Fetch real-time stock price for a ticker symbol.

    Args:
        symbol: Stock ticker (e.g. 'GOOG', 'AAPL').

    Examples:
        get_stock_price(symbol="GOOG")
    """
    return {"symbol": symbol, "price": 182.63, "currency": "USD"}


@tool(
    description="Search the web for information.",
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


@tool(
    description="Get current weather for a city.",
    category="weather",
    tags=["weather", "temperature", "forecast"],
)
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: City name.
        units: 'celsius' or 'fahrenheit'.
    """
    return {"city": city, "temperature": 25, "units": units}


# math
@tool(description="Add two numbers.", category="math", tags=["add"])
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool(description="Subtract two numbers.", category="math", tags=["subtract"])
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


@tool(description="Multiply two numbers.", category="math", tags=["multiply"])
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool(description="Divide two numbers.", category="math", tags=["divide"])
def divide(a: int, b: int) -> int:
    """Divide two numbers."""
    return a / b


def main():
    executor = ToolExecutor()

    # Meta-tool 1: Search for tools by intent
    print("=== Search: 'stock price lookup' ===")
    print(executor.search_tools("stock price lookup"))
    print()

    # Meta-tool 2: Execute a single call
    print("=== Execute: get_stock_price ===")
    results = executor.execute("get_stock_price(symbol='GOOG')")
    r = results[0]
    print(f"  ok:     {r.ok}")
    print(f"  tool:   {r.tool}")
    print(f"  result: {r.result}")
    print()

    # Execute multiple calls at once
    print("=== Execute: multi-call ===")
    results = executor.execute(
        "[get_stock_price(symbol='GOOG'), search_web(query='market trends')]"
    )
    for r in results:
        print(f"  {r.tool}: ok={r.ok}, result={r.result}")
    print()

    # Serialise for transport
    print("=== Serialisation ===")
    print(f"  to_dict: {results[0].to_dict()}")
    print(f"  to_json: {results[0].to_json()}")
    print()

    # Meta-tool 3: Get detailed docs when needed
    print("=== Describe: search_web ===")
    print(executor.describe_tool("search_web"))


if __name__ == "__main__":
    main()
