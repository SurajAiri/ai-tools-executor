"""Tests for the tool search module."""

from __future__ import annotations

import pytest

from ai_tools_executor.registry import ToolRegistry
from ai_tools_executor.search import KeywordSearchStrategy, format_search_results

from .conftest import SEARCH_WEB_INFO, STOCK_PRICE_INFO, WEATHER_INFO


@pytest.fixture()
def strategy():
    return KeywordSearchStrategy()


@pytest.fixture()
def all_tools():
    return [STOCK_PRICE_INFO, SEARCH_WEB_INFO, WEATHER_INFO]


class TestKeywordSearch:
    def test_matches_by_name(self, strategy, all_tools):
        results = strategy.search("stock price", all_tools)
        names = [t.name for t in results]
        assert "get_stock_price" in names

    def test_matches_by_tag(self, strategy, all_tools):
        results = strategy.search("google", all_tools)
        names = [t.name for t in results]
        assert "search_web" in names

    def test_matches_by_category(self, strategy, all_tools):
        results = strategy.search("finance", all_tools)
        names = [t.name for t in results]
        assert "get_stock_price" in names

    def test_matches_by_description(self, strategy, all_tools):
        results = strategy.search("weather city", all_tools)
        names = [t.name for t in results]
        assert "get_weather" in names

    def test_no_match_returns_empty(self, strategy, all_tools):
        results = strategy.search("quantum entanglement", all_tools)
        assert results == []

    def test_max_results_honoured(self, strategy, all_tools):
        results = strategy.search(
            "search web stock price weather",
            all_tools,
            max_results=2,
        )
        assert len(results) <= 2

    def test_ranking_prefers_name(self, strategy, all_tools):
        results = strategy.search("stock", all_tools)
        # "stock" appears in the name of get_stock_price, should rank first
        assert results[0].name == "get_stock_price"

    def test_empty_query(self, strategy, all_tools):
        results = strategy.search("", all_tools)
        assert results == []

    def test_single_char_tokens_ignored(self, strategy, all_tools):
        # Tokens shorter than 2 chars are filtered out
        results = strategy.search("a b c", all_tools)
        assert results == []


class TestFormatResults:
    def test_formats_with_header(self):
        output = format_search_results([STOCK_PRICE_INFO], "stock")
        assert '# Results for: "stock"' in output
        assert "def get_stock_price" in output
        assert '"""Fetch real-time stock price' in output

    def test_no_results_message(self):
        output = format_search_results([], "quantum")
        assert "No tools found" in output
        assert "quantum" in output

    def test_multiple_tools(self):
        output = format_search_results([STOCK_PRICE_INFO, SEARCH_WEB_INFO], "search")
        assert "get_stock_price" in output
        assert "search_web" in output
