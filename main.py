#!/usr/bin/env python3
"""
Native Planner Agent - Main Entry Point

A Claude CLI-based application using native Claude Code tools for planning.
Uses TodoWrite for task creation instead of custom plan managers.

Usage:
    python main.py                          # Interactive mode
    python main.py "Add user authentication"  # Direct prompt mode
    python main.py --cwd /path/to/repo "Add feature"  # Specify working directory
"""

import asyncio
import shutil
import sys

from planner_agent import main


async def run_with_error_handling() -> None:
    """Run the main function with error handling."""
    try:
        # Check if claude CLI is available
        if not shutil.which("claude"):
            print("\nError: Claude Code CLI not found.")
            print("Install it with:")
            print("  curl -fsSL https://claude.ai/install.sh | bash")
            print("\nOr see: https://code.claude.com/docs/en/setup")
            sys.exit(1)

        await main()
    except FileNotFoundError as e:
        print(f"\nFile not found error: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"\nPermission error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_with_error_handling())
