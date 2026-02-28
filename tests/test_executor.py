"""Tests for the ``ToolExecutor`` — the 3 meta-tool interface."""

from __future__ import annotations

import pytest

from ai_tools_executor.executor import ToolExecutor
from ai_tools_executor.registry import ToolRegistry

from .conftest import (
    FAILING_TOOL_INFO,
    SEARCH_WEB_INFO,
    STOCK_PRICE_INFO,
    WEATHER_INFO,
)


@pytest.fixture()
def executor(populated_registry: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(registry=populated_registry)


class TestSearchTools:
    def test_returns_matching_signatures(self, executor: ToolExecutor):
        result = executor.search_tools("stock price")
        assert "get_stock_price" in result
        assert "def get_stock_price" in result

    def test_no_match(self, executor: ToolExecutor):
        result = executor.search_tools("quantum computing")
        assert "No tools found" in result


class TestExecute:
    def test_single_call_success(self, executor: ToolExecutor):
        results = executor.execute("get_stock_price(symbol='GOOG')")
        assert len(results) == 1
        assert results[0]["status"] == "ok"
        assert results[0]["tool"] == "get_stock_price"
        assert results[0]["result"]["price"] == 182.63

    def test_multi_call_success(self, executor: ToolExecutor):
        raw = "[get_stock_price(symbol='GOOG'), get_weather(city='London')]"
        results = executor.execute(raw)
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)
        assert results[0]["result"]["symbol"] == "GOOG"
        assert results[1]["result"]["city"] == "London"

    def test_partial_failure(self, executor: ToolExecutor):
        raw = "[get_stock_price(symbol='GOOG'), failing_tool(x=42)]"
        results = executor.execute(raw)
        assert len(results) == 2
        # First call succeeds
        assert results[0]["status"] == "ok"
        assert results[0]["result"]["price"] == 182.63
        # Second call fails
        assert results[1]["status"] == "error"
        assert "Bad value: 42" in results[1]["error"]

    def test_syntax_error_returns_error_dict(self, executor: ToolExecutor):
        results = executor.execute("broken(")
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "tool" in results[0]

    def test_unknown_tool_returns_error_dict(self, executor: ToolExecutor):
        results = executor.execute("nonexistent(x=1)")
        assert len(results) == 1
        assert results[0]["status"] == "error"

    def test_missing_param_returns_error_dict(self, executor: ToolExecutor):
        results = executor.execute("get_stock_price()")
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "Missing required" in results[0]["error"]

    def test_optional_param_uses_default(self, executor: ToolExecutor):
        results = executor.execute("search_web(query='test')")
        assert results[0]["status"] == "ok"

    def test_optional_param_override(self, executor: ToolExecutor):
        results = executor.execute("search_web(query='test', max_results=3)")
        assert results[0]["status"] == "ok"


class TestDescribeTool:
    def test_returns_full_docstring(self, executor: ToolExecutor):
        result = executor.describe_tool("get_stock_price")
        assert "def get_stock_price" in result
        assert "ticker symbol" in result

    def test_unknown_tool(self, executor: ToolExecutor):
        result = executor.describe_tool("nonexistent")
        assert "not found" in result


class TestErrorFormat:
    def test_error_includes_input(self, executor: ToolExecutor):
        results = executor.execute("get_stock_price()")
        error_text = results[0]["error"]
        assert "Input:" in error_text

    def test_error_includes_expected(self, executor: ToolExecutor):
        results = executor.execute("get_stock_price()")
        error_text = results[0]["error"]
        assert "Expected:" in error_text

    def test_execution_error_includes_tool_name(self, executor: ToolExecutor):
        results = executor.execute("failing_tool(x=99)")
        assert results[0]["tool"] == "failing_tool"
        assert results[0]["status"] == "error"
        assert "Bad value: 99" in results[0]["error"]


class TestExecuteAsync:
    @pytest.mark.asyncio
    async def test_async_single_call(self, executor: ToolExecutor):
        results = await executor.execute_async("get_stock_price(symbol='GOOG')")
        assert len(results) == 1
        assert results[0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_async_multi_call(self, executor: ToolExecutor):
        raw = "[get_stock_price(symbol='GOOG'), get_weather(city='NYC')]"
        results = await executor.execute_async(raw)
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)

    @pytest.mark.asyncio
    async def test_async_partial_failure(self, executor: ToolExecutor):
        raw = "[get_stock_price(symbol='GOOG'), failing_tool(x=0)]"
        results = await executor.execute_async(raw)
        assert results[0]["status"] == "ok"
        assert results[1]["status"] == "error"
