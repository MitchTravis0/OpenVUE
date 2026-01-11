"""
Action Executor - Claude Computer Use Integration

This module provides the interface for executing user commands
using Claude Computer Use for visual understanding and control.
"""

import os
from dotenv import load_dotenv
from claude_computer_use import ClaudeComputerUse, ClaudeConfig

load_dotenv()

# Initialize Claude Computer Use agent
_claude_agent = None


def get_agent() -> ClaudeComputerUse:
    """Get or create the Claude Computer Use agent singleton."""
    global _claude_agent
    if _claude_agent is None:
        _claude_agent = ClaudeComputerUse()
    return _claude_agent


def digest_prompts(prompt: str, on_status=None) -> dict:
    """
    Process a user command using Claude Computer Use.

    Args:
        prompt: Natural language command from the user
        on_status: Optional callback for status updates (status_text, action_name)

    Returns:
        Dict with 'success', 'message', 'iterations', 'actions_taken'
    """
    print(f"(!) Processing with Claude CUA: {prompt}")

    agent = get_agent()

    # Set up status callback if provided
    if on_status:
        agent.on_action_start = lambda action_type, action: on_status(
            f"Executing: {action_type}", action_type
        )
        agent.on_task_complete = lambda success, msg: on_status(
            "Complete" if success else "Failed", msg
        )

    # Run the task
    result = agent.run_task(prompt)

    print(f"(!) Task complete. Success: {result['success']}")
    print(f"(!) Message: {result['message']}")
    print(f"(!) Actions taken: {len(result['actions_taken'])}")

    return result


# For backward compatibility - alias
def get_llm_interpretation(user_command: str) -> dict:
    """
    Legacy function - now uses Claude Computer Use.
    Returns result dict instead of action list.
    """
    return digest_prompts(user_command)


if __name__ == "__main__":
    print("Action Executor - Claude Computer Use")
    print("=" * 40)

    command = input("Enter command: ")
    if command.strip():
        result = digest_prompts(command)
        print(f"\nResult: {result['success']}")
        print(f"Message: {result['message']}")
