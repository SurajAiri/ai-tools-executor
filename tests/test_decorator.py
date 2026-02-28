"""Tests for the ``@tool`` decorator and ``ToolInfo`` / ``ParameterInfo``."""

from __future__ import annotations

import pytest

from ai_tools_executor import get_default_registry, tool
from ai_tools_executor.decorator import ParameterInfo, ToolInfo, _extract_parameters


class TestParameterInfo:
    def test_required_repr(self):
        p = ParameterInfo(name="x", annotation=int, required=True)
        assert repr(p) == "x: int"

    def test_optional_repr(self):
        p = ParameterInfo(name="n", annotation=int, default=5, required=False)
        assert repr(p) == "n: int = 5"

    def test_type_str_none(self):
        p = ParameterInfo(name="x", annotation=None, required=True)
        assert p.type_str == "Any"


class TestToolDecorator:
    def test_registers_tool(self):
        @tool(description="Add two numbers.", category="math", tags=["add"])
        def add(a: int, b: int) -> int:
            """Add *a* and *b*."""
            return a + b

        reg = get_default_registry()
        info = reg.get("add")
        assert info.name == "add"
        assert info.description == "Add two numbers."
        assert info.category == "math"
        assert info.tags == ["add"]
        assert info.return_type == "int"
        assert len(info.parameters) == 2

    def test_extracts_docstring(self):
        @tool(description="Short desc.")
        def greet(name: str) -> str:
            """Say hello to someone.

            Args:
                name: The person's name.
            """
            return f"Hello, {name}!"

        info = get_default_registry().get("greet")
        assert "Say hello to someone." in info.doc_str
        assert "Args:" in info.doc_str

    def test_preserves_original_function(self):
        @tool(description="Double a number.")
        def double(n: int) -> int:
            """Double *n*."""
            return n * 2

        assert double(5) == 10  # decorator is transparent

    def test_defaults_extracted(self):
        @tool(description="Search.")
        def search(query: str, limit: int = 10) -> list:
            """Search with limit."""
            return []

        info = get_default_registry().get("search")
        params = {p.name: p for p in info.parameters}
        assert params["query"].required is True
        assert params["limit"].required is False
        assert params["limit"].default == 10

    def test_no_tags_defaults_to_empty(self):
        @tool(description="No tags.")
        def notags() -> None:
            pass

        info = get_default_registry().get("notags")
        assert info.tags == []

    def test_default_category(self):
        @tool(description="General.")
        def general_tool() -> None:
            pass

        info = get_default_registry().get("general_tool")
        assert info.category == "general"


class TestToolInfoFormatting:
    def test_signature_line(self):
        info = ToolInfo(
            name="add",
            description="Add numbers.",
            doc_str="Add numbers.",
            category="math",
            tags=[],
            parameters=[
                ParameterInfo(name="a", annotation=int, required=True),
                ParameterInfo(name="b", annotation=int, required=True),
            ],
            return_type="int",
            func=lambda a, b: a + b,
        )
        sig = info.signature_line()
        assert sig == "def add(a: int, b: int) -> int:"

    def test_short_summary(self):
        info = ToolInfo(
            name="echo",
            description="Echo input.",
            doc_str="Echo back the input string.",
            category="util",
            tags=[],
            parameters=[ParameterInfo(name="msg", annotation=str, required=True)],
            return_type="str",
            func=lambda msg: msg,
        )
        summary = info.short_summary()
        assert "def echo(msg: str) -> str:" in summary
        assert '"""Echo input."""' in summary

    def test_full_description_uses_doc_str(self):
        info = ToolInfo(
            name="echo",
            description="Short.",
            doc_str="Full docstring with details.",
            category="util",
            tags=[],
            parameters=[],
            return_type="str",
            func=lambda: "",
        )
        full = info.full_description()
        assert "Full docstring with details." in full
