"""``@tool`` decorator and ``ToolInfo`` / ``ParameterInfo`` data models.

The decorator inspects the wrapped function's signature and docstring,
builds a :class:`ToolInfo`, and registers it in the global registry.
The original function is returned **unmodified** — the decorator is
transparent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

# Lazy import to avoid circular dependency — registry imports nothing
# from this module; this module imports the registry at decoration time.
_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class ParameterInfo:
    """Metadata for a single function parameter."""

    name: str
    annotation: type | str | None = None
    default: Any = _SENTINEL  # _SENTINEL means "no default → required"
    required: bool = True

    @property
    def type_str(self) -> str:
        """Human-readable type string."""
        if self.annotation is None:
            return "Any"
        if isinstance(self.annotation, type):
            return self.annotation.__name__
        return str(self.annotation)

    def __repr__(self) -> str:
        default_part = "" if self.required else f" = {self.default!r}"
        return f"{self.name}: {self.type_str}{default_part}"


@dataclass(slots=True)
class ToolInfo:
    """Complete metadata for a registered tool."""

    name: str
    description: str
    doc_str: str
    category: str
    tags: list[str]
    parameters: list[ParameterInfo]
    return_type: str
    func: Callable[..., Any]

    # ── Formatting helpers used by search_tools / describe_tool ───────

    def signature_line(self) -> str:
        """Compact ``def name(params) -> ret_type:`` string."""
        params = ", ".join(repr(p) for p in self.parameters)
        return f"def {self.name}({params}) -> {self.return_type}:"

    def short_summary(self) -> str:
        """Signature + one-line description (what ``search_tools`` returns)."""
        return f'{self.signature_line()}\n    """{self.description}"""'

    def full_description(self) -> str:
        """Signature + full docstring (what ``describe_tool`` returns)."""
        doc = self.doc_str or self.description
        return f'{self.signature_line()}\n    """{doc}"""'


def _extract_parameters(func: Callable[..., Any]) -> list[ParameterInfo]:
    """Inspect *func* and return a list of :class:`ParameterInfo`."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 — runtime hints can fail
        hints = {}

    params: list[ParameterInfo] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = None

        has_default = param.default is not inspect.Parameter.empty
        params.append(
            ParameterInfo(
                name=name,
                annotation=annotation,
                default=param.default if has_default else _SENTINEL,
                required=not has_default,
            )
        )
    return params


def _extract_return_type(func: Callable[..., Any]) -> str:
    """Return the stringified return-type annotation, or ``'Any'``."""
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001
        hints = {}
    ret = hints.get("return", inspect.signature(func).return_annotation)
    if ret is inspect.Parameter.empty:
        return "Any"
    if isinstance(ret, type):
        return ret.__name__
    return str(ret)


# ── The public decorator ──────────────────────────────────────────────


def tool(
    description: str,
    *,
    category: str = "general",
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as an executable tool.

    Parameters
    ----------
    description:
        Short, one-line description used by ``search_tools`` results and
        error ``Expected`` fields.
    category:
        Logical grouping (``"finance"``, ``"search"``, …).
    tags:
        Extra keywords used for search matching.
    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        # Import here to avoid circular imports at module load time.
        from ai_tools_executor.registry import get_default_registry

        info = ToolInfo(
            name=func.__name__,
            description=description,
            doc_str=inspect.getdoc(func) or "",
            category=category,
            tags=list(tags) if tags else [],
            parameters=_extract_parameters(func),
            return_type=_extract_return_type(func),
            func=func,
        )
        get_default_registry().register(info)
        return func

    return wrapper
