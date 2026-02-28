"""Package-level logging configuration.

Follows the Python library logging best practice: attach a
:class:`~logging.NullHandler` so that no output is produced unless the
*application* developer explicitly configures logging.

Quick opt-in::

    from ai_tools_executor import setup_logging
    setup_logging()          # INFO to stderr
    setup_logging("DEBUG")   # DEBUG to stderr

Or use the standard ``logging`` module directly::

    import logging
    logging.getLogger("ai_tools_executor").setLevel(logging.DEBUG)
"""

from __future__ import annotations

import logging
import sys

# ── Package logger ────────────────────────────────────────────────────

logger = logging.getLogger("ai_tools_executor")
logger.addHandler(logging.NullHandler())


# ── Convenience helper ────────────────────────────────────────────────


def setup_logging(
    level: str | int = logging.INFO,
    handler: logging.Handler | None = None,
) -> None:
    """Configure logging for the ``ai_tools_executor`` package.

    Parameters
    ----------
    level:
        Logging level — a string like ``"DEBUG"`` or an ``int`` constant
        such as :data:`logging.DEBUG`.
    handler:
        A custom :class:`~logging.Handler`.  When *None* a
        :class:`~logging.StreamHandler` writing to *stderr* is created
        with a sensible format.

    Examples
    --------
    >>> from ai_tools_executor import setup_logging
    >>> setup_logging()           # INFO to stderr
    >>> setup_logging("DEBUG")    # verbose
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    # Remove the NullHandler before adding a real one
    for h in logger.handlers[:]:
        if isinstance(h, logging.NullHandler):
            logger.removeHandler(h)

    logger.addHandler(handler)
    logger.setLevel(level)
