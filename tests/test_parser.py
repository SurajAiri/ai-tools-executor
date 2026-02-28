"""Tests for the AST call parser."""

from __future__ import annotations

import pytest

from ai_tools_executor.exceptions import ParseError, ValidationError
from ai_tools_executor.parser import parse_calls
from ai_tools_executor.registry import ToolRegistry

from .conftest import SEARCH_WEB_INFO, STOCK_PRICE_INFO, WEATHER_INFO


class TestSingleCall:
    def test_simple_keyword(self, populated_registry: ToolRegistry):
        results = parse_calls("get_stock_price(symbol='GOOG')", populated_registry)
        assert len(results) == 1
        assert results[0].name == "get_stock_price"
        assert results[0].kwargs == {"symbol": "GOOG"}

    def test_double_quoted_string(self, populated_registry: ToolRegistry):
        results = parse_calls('get_stock_price(symbol="AAPL")', populated_registry)
        assert results[0].kwargs["symbol"] == "AAPL"

    def test_optional_param_omitted(self, populated_registry: ToolRegistry):
        results = parse_calls("search_web(query='hello')", populated_registry)
        assert results[0].kwargs == {"query": "hello"}

    def test_optional_param_provided(self, populated_registry: ToolRegistry):
        results = parse_calls(
            "search_web(query='hello', max_results=3)", populated_registry
        )
        assert results[0].kwargs == {"query": "hello", "max_results": 3}

    def test_positional_arg(self, populated_registry: ToolRegistry):
        results = parse_calls("get_stock_price('GOOG')", populated_registry)
        assert results[0].kwargs == {"symbol": "GOOG"}


class TestMultiCall:
    def test_list_of_calls(self, populated_registry: ToolRegistry):
        raw = "[get_stock_price(symbol='GOOG'), search_web(query='test')]"
        results = parse_calls(raw, populated_registry)
        assert len(results) == 2
        assert results[0].name == "get_stock_price"
        assert results[1].name == "search_web"

    def test_three_calls(self, populated_registry: ToolRegistry):
        raw = (
            "[get_stock_price(symbol='A'),"
            " get_stock_price(symbol='B'),"
            " get_weather(city='NYC')]"
        )
        results = parse_calls(raw, populated_registry)
        assert len(results) == 3


class TestParseErrors:
    def test_syntax_error(self, populated_registry: ToolRegistry):
        with pytest.raises(ParseError, match="Failed to parse"):
            parse_calls("get_stock_price(symbol='GOOG'", populated_registry)

    def test_empty_list(self, populated_registry: ToolRegistry):
        with pytest.raises(ParseError, match="Empty call list"):
            parse_calls("[]", populated_registry)

    def test_non_call_in_list(self, populated_registry: ToolRegistry):
        with pytest.raises(ParseError, match="only function calls"):
            parse_calls("[42]", populated_registry)

    def test_not_a_call(self, populated_registry: ToolRegistry):
        with pytest.raises(ParseError, match="Expected a function call"):
            parse_calls("42", populated_registry)

    def test_attribute_call_rejected(self, populated_registry: ToolRegistry):
        with pytest.raises(ParseError, match="simple function names"):
            parse_calls("obj.method()", populated_registry)


class TestValidationErrors:
    def test_unknown_tool(self, populated_registry: ToolRegistry):
        with pytest.raises(Exception, match="not found"):
            parse_calls("nonexistent(x=1)", populated_registry)

    def test_unknown_param(self, populated_registry: ToolRegistry):
        with pytest.raises(ValidationError, match="Unknown parameter"):
            parse_calls("get_stock_price(symbol='A', bad=1)", populated_registry)

    def test_missing_required(self, populated_registry: ToolRegistry):
        with pytest.raises(ValidationError, match="Missing required"):
            parse_calls("get_stock_price()", populated_registry)

    def test_too_many_positional(self, populated_registry: ToolRegistry):
        with pytest.raises(ValidationError, match="Too many positional"):
            parse_calls("get_stock_price('A', 'B')", populated_registry)

    def test_duplicate_arg(self, populated_registry: ToolRegistry):
        with pytest.raises(ValidationError, match="Duplicate"):
            parse_calls("get_stock_price('GOOG', symbol='GOOG')", populated_registry)


class TestLiteralResolution:
    def test_int(self, populated_registry: ToolRegistry):
        raw = "search_web(query='q', max_results=10)"
        results = parse_calls(raw, populated_registry)
        assert results[0].kwargs["max_results"] == 10

    def test_negative_int(self, populated_registry: ToolRegistry):
        raw = "search_web(query='q', max_results=-1)"
        results = parse_calls(raw, populated_registry)
        assert results[0].kwargs["max_results"] == -1

    def test_bool(self, populated_registry: ToolRegistry):
        # Booleans parse as constants in AST
        raw = "search_web(query='q', max_results=True)"
        results = parse_calls(raw, populated_registry)
        assert results[0].kwargs["max_results"] is True

    def test_list_value(self, populated_registry: ToolRegistry):
        # Even if search_web expects int, the parser resolves
        # the value — type checking is Layer 2.
        raw = "search_web(query='q', max_results=[1,2])"
        results = parse_calls(raw, populated_registry)
        assert results[0].kwargs["max_results"] == [1, 2]
