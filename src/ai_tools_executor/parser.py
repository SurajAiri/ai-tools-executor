"""AST-based call parser — Layer 1 validation.

Safely parses Python function-call syntax produced by the LLM,
extracts function names and keyword arguments, and validates them
against the :class:`ToolRegistry`.

No code is ever *executed* — only the syntax tree is inspected, exactly
like an IDE or linter would do.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_tools_executor.exceptions import ParseError, ValidationError

if TYPE_CHECKING:
    from ai_tools_executor.decorator import ToolInfo
    from ai_tools_executor.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ParsedCall:
    """Result of successfully parsing and validating a single call."""

    name: str
    kwargs: dict[str, Any]
    tool_info: ToolInfo


# ── Public API ────────────────────────────────────────────────────────


def parse_calls(raw: str, registry: ToolRegistry) -> list[ParsedCall]:
    """Parse *raw* into one or more validated :class:`ParsedCall` objects.

    Parameters
    ----------
    raw:
        A string containing either a single function call
        (``'fn(a=1)'``) or a list of calls
        (``'[fn1(a=1), fn2(b=2)]'``).
    registry:
        The :class:`ToolRegistry` used for name / parameter lookups.

    Raises
    ------
    ParseError
        If ``ast.parse`` fails (syntax error).
    ValidationError
        If the call references an unknown tool or has parameter
        issues.
    """
    raw = raw.strip()
    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError as exc:
        raise ParseError(
            "Failed to parse tool call",
            input_text=raw,
            hint=f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})",
        ) from exc

    body = tree.body

    # Single call: ``fn(a=1)``
    if isinstance(body, ast.Call):
        return [_validate_call(body, raw, registry)]

    # List of calls: ``[fn1(), fn2()]``
    if isinstance(body, ast.List):
        if not body.elts:
            raise ParseError(
                "Empty call list",
                input_text=raw,
                hint="Provide at least one function call inside the list.",
            )
        results: list[ParsedCall] = []
        for node in body.elts:
            if not isinstance(node, ast.Call):
                raise ParseError(
                    "List must contain only function calls",
                    input_text=raw,
                    hint=f"Found {type(node).__name__} instead of a Call.",
                )
            results.append(_validate_call(node, raw, registry))
        return results

    raise ParseError(
        "Expected a function call or a list of function calls",
        input_text=raw,
        hint=f"Got {type(body).__name__} instead.",
    )


# ── Internal helpers ──────────────────────────────────────────────────


def _extract_func_name(node: ast.Call) -> str:
    """Return the plain function name from an ``ast.Call`` node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    raise ParseError(
        "Only simple function names are supported",
        hint="Attribute calls like obj.method() are not allowed.",
    )


def _resolve_value(node: ast.expr) -> Any:
    """Safely convert an AST literal node to a Python value.

    Only constant literals, lists, dicts, tuples, sets, and
    unary operators (e.g. ``-1``) are allowed.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_resolve_value(el) for el in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_resolve_value(el) for el in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _resolve_value(k): _resolve_value(v)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    if isinstance(node, ast.Set):
        return {_resolve_value(el) for el in node.elts}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _resolve_value(node.operand)
        return +operand if isinstance(node.op, ast.UAdd) else -operand

    raise ParseError(
        "Unsupported expression in arguments",
        hint=(
            f"Only literals (str, int, float, bool, None, list, dict) are "
            f"allowed.  Got {type(node).__name__}."
        ),
    )


def _validate_call(
    node: ast.Call,
    raw: str,
    registry: ToolRegistry,
) -> ParsedCall:
    """Validate a single ``ast.Call`` against the registry."""
    func_name = _extract_func_name(node)

    # Resolve tool info (raises ToolNotFoundError if missing)
    tool_info = registry.get(func_name)

    # Collect keyword arguments
    kwargs: dict[str, Any] = {}

    # Handle positional arguments — map them to parameter names in order
    param_names = [p.name for p in tool_info.parameters]
    if node.args:
        if len(node.args) > len(param_names):
            raise ValidationError(
                f"Too many positional arguments for '{func_name}'",
                input_text=raw,
                expected=tool_info.short_summary(),
                hint=(
                    f"Expected at most {len(param_names)} arguments, "
                    f"got {len(node.args)}."
                ),
            )
        for i, arg_node in enumerate(node.args):
            kwargs[param_names[i]] = _resolve_value(arg_node)

    # Handle keyword arguments
    for kw in node.keywords:
        if kw.arg is None:
            raise ValidationError(
                "**kwargs unpacking is not supported in tool calls",
                input_text=raw,
                expected=tool_info.short_summary(),
            )
        if kw.arg in kwargs:
            raise ValidationError(
                f"Duplicate argument '{kw.arg}' for '{func_name}'",
                input_text=raw,
                expected=tool_info.short_summary(),
            )
        kwargs[kw.arg] = _resolve_value(kw.value)

    # Validate parameter names
    valid_names = {p.name for p in tool_info.parameters}
    unknown = set(kwargs) - valid_names
    if unknown:
        raise ValidationError(
            f"Unknown parameter(s) for '{func_name}': {', '.join(sorted(unknown))}",
            input_text=raw,
            expected=tool_info.short_summary(),
            hint=f"Valid parameters: {', '.join(sorted(valid_names))}",
        )

    # Validate required parameters
    required = {p.name for p in tool_info.parameters if p.required}
    missing = required - set(kwargs)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValidationError(
            f"Missing required parameter(s) for "
            f"'{func_name}': {missing_str}",
            input_text=raw,
            expected=tool_info.short_summary(),
            hint=f"Required: {', '.join(sorted(required))}",
        )

    return ParsedCall(name=func_name, kwargs=kwargs, tool_info=tool_info)
