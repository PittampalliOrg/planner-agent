#!/usr/bin/env python3
"""
Planner Agent - Main Entry Point

A Claude Agent SDK application that replicates Claude Code's plan mode.
Run this script to start the planner agent.

Usage:
    python main.py                          # Interactive mode
    python main.py "Add user authentication"  # Direct prompt mode
    python main.py --cwd /path/to/repo "Add feature"  # Specify working directory
"""

import asyncio
import sys

from claude_agent_sdk import CLINotFoundError, ProcessError, CLIJSONDecodeError

from planner_agent import main


async def run_with_error_handling() -> None:
    """Run the main function with SDK-specific error handling."""
    try:
        await main()
    except CLINotFoundError:
        print("\nError: Claude Code CLI not found.")
        print("Install it with:")
        print("  curl -fsSL https://claude.ai/install.sh | bash")
        print("\nOr see: https://code.claude.com/docs/en/setup")
        sys.exit(1)
    except ProcessError as e:
        print(f"\nProcess error: {e}")
        if e.exit_code:
            print(f"Exit code: {e.exit_code}")
        if e.stderr:
            print(f"Details: {e.stderr}")
        sys.exit(1)
    except CLIJSONDecodeError as e:
        print(f"\nFailed to parse CLI response: {e}")
        print("This may indicate a CLI version mismatch.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        sys.exit(0)


def run() -> None:
    """Synchronous entry point for the planner-agent console script."""
    asyncio.run(run_with_error_handling())


if __name__ == "__main__":
    asyncio.run(run_with_error_handling())
