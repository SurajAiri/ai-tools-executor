"""Tests for ``ai_tools_executor.meta_tools``."""

from __future__ import annotations

import json

from ai_tools_executor.executor import ToolExecutor
from ai_tools_executor.meta_tools import get_meta_tools_schema, handle_tool_call
from ai_tools_executor.registry import ToolRegistry

from .conftest import STOCK_PRICE_INFO, WEATHER_INFO


class TestGetMetaToolsSchema:
    """Tests for :func:`get_meta_tools_schema`."""

    def test_returns_list_of_three(self):
        schema = get_meta_tools_schema()
        assert isinstance(schema, list)
        assert len(schema) == 3

    def test_each_entry_has_type_function(self):
        for entry in get_meta_tools_schema():
            assert entry["type"] == "function"
            assert "function" in entry

    def test_function_keys_present(self):
        for entry in get_meta_tools_schema():
            fn = entry["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_tool_names(self):
        names = [e["function"]["name"] for e in get_meta_tools_schema()]
        assert names == ["search_tools", "execute", "describe_tool"]

    def test_search_tools_required_params(self):
        fn = get_meta_tools_schema()[0]["function"]
        assert fn["parameters"]["required"] == ["query"]
        assert "query" in fn["parameters"]["properties"]

    def test_execute_required_params(self):
        fn = get_meta_tools_schema()[1]["function"]
        assert fn["parameters"]["required"] == ["calls"]
        assert "calls" in fn["parameters"]["properties"]

    def test_describe_tool_required_params(self):
        fn = get_meta_tools_schema()[2]["function"]
        assert fn["parameters"]["required"] == ["name"]
        assert "name" in fn["parameters"]["properties"]

    def test_all_property_types_are_string(self):
        """Every parameter should be typed as 'string'."""
        for entry in get_meta_tools_schema():
            props = entry["function"]["parameters"]["properties"]
            for prop in props.values():
                assert prop["type"] == "string"

    def test_returns_fresh_copy(self):
        """Each call should return a new list (no shared mutable state)."""
        a = get_meta_tools_schema()
        b = get_meta_tools_schema()
        assert a is not b
        assert a == b


class TestHandleToolCall:
    """Tests for :func:`handle_tool_call`."""

    @staticmethod
    def _make_executor() -> ToolExecutor:
        reg = ToolRegistry()
        reg.register(STOCK_PRICE_INFO)
        reg.register(WEATHER_INFO)
        return ToolExecutor(registry=reg)

    def test_search_tools(self):
        executor = self._make_executor()
        result = handle_tool_call(
            executor, "search_tools", {"query": "stock"},
        )
        assert isinstance(result, str)
        assert "get_stock_price" in result

    def test_execute_returns_json(self):
        executor = self._make_executor()
        result = handle_tool_call(
            executor,
            "execute",
            {"calls": "get_stock_price(symbol='GOOG')"},
        )
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["status"] == "ok"
        assert parsed[0]["result"]["price"] == 182.63

    def test_describe_tool(self):
        executor = self._make_executor()
        result = handle_tool_call(
            executor, "describe_tool", {"name": "get_stock_price"},
        )
        assert isinstance(result, str)
        assert "get_stock_price" in result

    def test_unknown_meta_tool(self):
        executor = self._make_executor()
        result = handle_tool_call(
            executor, "nonexistent", {},
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Unknown meta-tool" in parsed["error"]
