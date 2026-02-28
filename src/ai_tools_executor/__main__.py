"""Allow ``python -m ai_tools_executor``."""

from ai_tools_executor import __version__


def main() -> None:
    print(f"ai-tools-executor v{__version__}")
    print("Import and use: from ai_tools_executor import tool, ToolExecutor")


if __name__ == "__main__":
    main()