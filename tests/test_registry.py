"""Tests for the ``ToolRegistry``."""

from __future__ import annotations

import pytest

from ai_tools_executor.exceptions import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
)
from ai_tools_executor.registry import ToolRegistry

from .conftest import SEARCH_WEB_INFO, STOCK_PRICE_INFO


class TestRegistration:
    def test_register_and_get(self, registry: ToolRegistry):
        registry.register(STOCK_PRICE_INFO)
        info = registry.get("get_stock_price")
        assert info.name == "get_stock_price"

    def test_duplicate_raises(self, registry: ToolRegistry):
        registry.register(STOCK_PRICE_INFO)
        with pytest.raises(ToolAlreadyRegisteredError, match="already registered"):
            registry.register(STOCK_PRICE_INFO)

    def test_unregister(self, registry: ToolRegistry):
        registry.register(STOCK_PRICE_INFO)
        registry.unregister("get_stock_price")
        assert "get_stock_price" not in registry

    def test_unregister_missing_raises(self, registry: ToolRegistry):
        with pytest.raises(ToolNotFoundError):
            registry.unregister("nonexistent")

    def test_clear(self, registry: ToolRegistry):
        registry.register(STOCK_PRICE_INFO)
        registry.register(SEARCH_WEB_INFO)
        assert len(registry) == 2
        registry.clear()
        assert len(registry) == 0


class TestLookup:
    def test_get_missing_raises(self, registry: ToolRegistry):
        with pytest.raises(ToolNotFoundError, match="not found"):
            registry.get("nope")

    def test_has(self, registry: ToolRegistry):
        assert not registry.has("get_stock_price")
        registry.register(STOCK_PRICE_INFO)
        assert registry.has("get_stock_price")

    def test_contains(self, registry: ToolRegistry):
        registry.register(STOCK_PRICE_INFO)
        assert "get_stock_price" in registry
        assert "missing" not in registry

    def test_list_all(self, registry: ToolRegistry):
        registry.register(STOCK_PRICE_INFO)
        registry.register(SEARCH_WEB_INFO)
        all_tools = registry.list_all()
        names = {t.name for t in all_tools}
        assert names == {"get_stock_price", "search_web"}

    def test_len(self, registry: ToolRegistry):
        assert len(registry) == 0
        registry.register(STOCK_PRICE_INFO)
        assert len(registry) == 1

    def test_hint_shows_available_tools(self, registry: ToolRegistry):
        registry.register(STOCK_PRICE_INFO)
        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.get("nope")
        assert "get_stock_price" in exc_info.value.hint
