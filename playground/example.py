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

@tool(description="Get current weather for a city.", category="weather", tags=["weather", "temperature", "forecast"])
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

from ai_tools_executor import ToolExecutor
def main():
    executor = ToolExecutor()

    # Meta-tool 1: Search for tools by intent
    print(executor.search_tools("stock price lookup"))
    # def get_stock_price(symbol: str) -> dict:
    #     """Fetch real-time stock price."""

    # Meta-tool 2: Execute tool calls using Python syntax
    results = executor.execute("get_stock_price(symbol='GOOG')")
    print(f"Results: {results}")
    # [{'tool': 'get_stock_price', 'status': 'ok', 'result': {'symbol': 'GOOG', 'price': 182.63, 'currency': 'USD'}}]

    # Execute multiple calls at once
    results = executor.execute("[get_stock_price(symbol='GOOG'), search_web(query='market trends')]")
    print(f"Results: {results}")

    # Meta-tool 3: Get detailed docs when needed
    # print(executor.describe_tool("search_web"))

if __name__ == "__main__":
    main()
