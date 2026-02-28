"""Shared test fixtures."""

from __future__ import annotations

import pytest

from ai_tools_executor.decorator import ParameterInfo, ToolInfo
from ai_tools_executor.registry import ToolRegistry, reset_default_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the global registry before every test."""
    reset_default_registry()
    yield
    reset_default_registry()


@pytest.fixture()
def registry() -> ToolRegistry:
    """Return a fresh, isolated registry."""
    return ToolRegistry()


# ── Sample tools (plain ToolInfo, not using @tool) ────────────────────


def _stock_price(symbol: str) -> dict:
    return {"symbol": symbol, "price": 182.63, "currency": "USD"}


def _search_web(query: str, max_results: int = 5) -> list[dict]:
    return [{"title": "Result", "url": "https://example.com", "snippet": query}]


def _get_weather(city: str, units: str = "celsius") -> dict:
    return {"city": city, "temp": 12, "units": units}


def _failing_tool(x: int) -> int:
    raise ValueError(f"Bad value: {x}")


STOCK_PRICE_INFO = ToolInfo(
    name="get_stock_price",
    description=(
        "Fetch real-time stock price."
        " symbol: ticker like 'GOOG', 'AAPL'."
    ),
    doc_str=(
        "Fetch real-time stock price for a given"
        " ticker symbol.\n\nArgs:\n"
        "    symbol: Stock ticker symbol."
    ),
    category="finance",
    tags=["stock", "price", "market"],
    parameters=[
        ParameterInfo(name="symbol", annotation=str, required=True),
    ],
    return_type="dict",
    func=_stock_price,
)

SEARCH_WEB_INFO = ToolInfo(
    name="search_web",
    description=(
        "Search the web for information"
        " using a text query."
    ),
    doc_str=(
        "Search the web for information.\n\nArgs:\n"
        "    query: Natural language search query.\n"
        "    max_results: Number of results"
        " to return (1-20)."
    ),
    category="search",
    tags=["web", "search", "google"],
    parameters=[
        ParameterInfo(
            name="query", annotation=str, required=True,
        ),
        ParameterInfo(
            name="max_results", annotation=int,
            default=5, required=False,
        ),
    ],
    return_type="list[dict]",
    func=_search_web,
)

WEATHER_INFO = ToolInfo(
    name="get_weather",
    description="Get current weather for a city.",
    doc_str=(
        "Get current weather.\n\nArgs:\n"
        "    city: City name.\n"
        "    units: 'celsius' or 'fahrenheit'."
    ),
    category="weather",
    tags=["weather", "temperature", "forecast"],
    parameters=[
        ParameterInfo(
            name="city", annotation=str, required=True,
        ),
        ParameterInfo(
            name="units", annotation=str,
            default="celsius", required=False,
        ),
    ],
    return_type="dict",
    func=_get_weather,
)

FAILING_TOOL_INFO = ToolInfo(
    name="failing_tool",
    description="A tool that always fails.",
    doc_str="A tool that always raises ValueError.",
    category="test",
    tags=["fail"],
    parameters=[ParameterInfo(name="x", annotation=int, required=True)],
    return_type="int",
    func=_failing_tool,
)


@pytest.fixture()
def populated_registry(registry: ToolRegistry) -> ToolRegistry:
    """Registry pre-loaded with sample tools."""
    for info in (STOCK_PRICE_INFO, SEARCH_WEB_INFO, WEATHER_INFO, FAILING_TOOL_INFO):
        registry.register(info)
    return registry
