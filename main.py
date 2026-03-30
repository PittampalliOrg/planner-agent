#!/usr/bin/env python3
"""
Planner Agent - Main Entry Point

A Claude Agent SDK application that replicates Claude Code's plan mode.
Run this script to start the planner agent.

Usage:
    python main.py                                    # Interactive mode
    python main.py "Add user authentication"          # Direct prompt mode
    python main.py --cwd /path/to/repo "Add feature" # Specify working directory
    python main.py --version                          # Print version and exit
    python main.py --verbose "Add feature X"          # Run with debug logging
    python main.py -v --cwd /path/to/repo "Add feature X"  # Verbose + custom cwd
"""

import argparse
import asyncio
import logging
import sys

from claude_agent_sdk import CLINotFoundError, ProcessError, CLIJSONDecodeError

from planner_agent import PlannerAgent
from version import __version__

VERBOSE = False


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Planner Agent - Claude Code Plan Mode Replica"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"planner-agent {__version__}",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory (git repository) to work in",
    )
    parser.add_argument(
        "--plans-dir",
        type=str,
        default=None,
        help="Directory to store plans and tasks",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        default=None,
        help="Feature request to plan (if not provided, runs interactively)",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    """Configure root logging based on verbosity flag."""
    global VERBOSE
    VERBOSE = verbose
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)


async def run_with_error_handling(args: argparse.Namespace) -> None:
    """Run the planner agent with SDK-specific error handling."""
    try:
        agent = PlannerAgent(cwd=args.cwd, plans_dir=args.plans_dir)
        if args.prompt:
            await agent.run_planning_session(args.prompt)
        else:
            await agent.run_interactive()
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


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    asyncio.run(run_with_error_handling(args))
