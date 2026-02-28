"""Pluggable tool search strategies.

The default :class:`KeywordSearchStrategy` tokenises the query and
scores each tool by how many tokens match its name, description,
category, and tags.  Higher-quality strategies (fuzzy, semantic
embeddings) can be swapped in by implementing the
:class:`SearchStrategy` protocol.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_tools_executor.decorator import ToolInfo


# ── Abstract interface ────────────────────────────────────────────────


class SearchStrategy(ABC):
    """Protocol for pluggable search backends."""

    @abstractmethod
    def search(
        self,
        query: str,
        tools: list[ToolInfo],
        *,
        max_results: int = 5,
    ) -> list[ToolInfo]:
        """Return the best-matching tools for *query*, ranked by relevance."""


# ── Default keyword strategy ──────────────────────────────────────────

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenise(text: str) -> set[str]:
    """Lower-case and split on non-alphanumeric boundaries."""
    return {t for t in _SPLIT_RE.split(text.lower()) if len(t) >= 2}


def _name_tokens(name: str) -> set[str]:
    """Split a ``snake_case`` or ``camelCase`` name into tokens."""
    # Convert camelCase to snake_case first
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return _tokenise(snake)


@dataclass
class _ScoredTool:
    tool: ToolInfo
    score: float


class KeywordSearchStrategy(SearchStrategy):
    """Score tools by keyword overlap with the query.

    Scoring weights:
    - Name token match: 3 points
    - Tag match: 2 points
    - Category match: 2 points
    - Description token match: 1 point
    """

    def search(
        self,
        query: str,
        tools: list[ToolInfo],
        *,
        max_results: int = 5,
    ) -> list[ToolInfo]:
        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        scored: list[_ScoredTool] = []
        for t in tools:
            score = 0.0

            # Name matching (highest weight)
            name_tokens = _name_tokens(t.name)
            score += len(query_tokens & name_tokens) * 3

            # Tag matching
            tag_tokens: set[str] = set()
            for tag in t.tags:
                tag_tokens |= _tokenise(tag)
            score += len(query_tokens & tag_tokens) * 2

            # Category matching
            cat_tokens = _tokenise(t.category)
            score += len(query_tokens & cat_tokens) * 2

            # Description matching (lowest weight)
            desc_tokens = _tokenise(t.description)
            score += len(query_tokens & desc_tokens) * 1

            if score > 0:
                scored.append(_ScoredTool(tool=t, score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return [s.tool for s in scored[:max_results]]


# ── Result formatting (what the agent sees) ───────────────────────────


def format_search_results(tools: list[ToolInfo], query: str) -> str:
    """Format matched tools as concise function signatures.

    This is exactly the format shown in the architecture doc under
    "Step 1 — Agent Searches".
    """
    if not tools:
        return f'No tools found for: "{query}"'

    header = f'# Tools matching: "{query}"\n'
    blocks = [header]
    for t in tools:
        blocks.append(t.short_summary())
    return "\n".join(blocks)
