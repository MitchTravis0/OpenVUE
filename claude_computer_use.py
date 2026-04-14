"""
Claude Computer Use Integration for AIVue

Provides visual understanding and self-correcting computer control
using Claude's Computer Use API. This module implements the agent loop
pattern where Claude requests actions, we execute them, capture screenshots,
and return results until the task is complete.

Usage:
    from claude_computer_use import ClaudeComputerUse

    agent = ClaudeComputerUse()
    result = agent.run_task("Open Chrome and search for Python tutorials")
"""

import anthropic
import base64
import io
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple
from PIL import Image
import pyautogui
from dotenv import load_dotenv

load_dotenv()


class ActionType(Enum):
    """Supported Claude Computer Use actions."""
    SCREENSHOT = "screenshot"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    TRIPLE_CLICK = "triple_click"
    MIDDLE_CLICK = "middle_click"
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK_DRAG = "left_click_drag"
    LEFT_MOUSE_DOWN = "left_mouse_down"
    LEFT_MOUSE_UP = "left_mouse_up"
    TYPE = "type"
    KEY = "key"
    HOLD_KEY = "hold_key"
    SCROLL = "scroll"
    WAIT = "wait"


@dataclass
class ClaudeConfig:
    """Configuration for Claude Computer Use."""
    api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    max_iterations: int = 20
    display_width: int = 1024
    display_height: int = 768
    tool_version: str = "computer_20250124"
    beta_flag: str = "computer-use-2025-01-24"

    # Screenshot settings
    screenshot_quality: int = 85  # JPEG quality (1-100)
    use_jpeg: bool = True  # Use JPEG for smaller size

    # Timing settings
    action_delay: float = 0.3  # Delay after each action
    typing_interval: float = 0.02  # Delay between keystrokes

    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
            max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "4096")),
            max_iterations=int(os.getenv("CLAUDE_MAX_ITERATIONS", "20")),
            display_width=int(os.getenv("CLAUDE_DISPLAY_WIDTH", "1024")),
            display_height=int(os.getenv("CLAUDE_DISPLAY_HEIGHT", "768")),
        )


@dataclass
class SafetyConfig:
    """Safety configuration for computer use."""

    # Regions to avoid clicking (x, y, width, height)
    avoid_regions: List[Tuple[int, int, int, int]] = field(
        default_factory=lambda: [(0, 0, 5, 5)]  # Top-left corner
    )

    # Regex patterns that trigger confirmation for destructive commands
    sensitive_patterns: List[str] = field(
        default_factory=lambda: [
            r"\brm\s+-rf\b",
            r"\bdel\s+/[sfq]",
            r"\bformat\s+[a-z]:",
            r"\bshutdown\b",
            r"\brestart\b",
            r"\brmdir\s+/s",
            r"\bdiskpart\b",
            r"\breg\s+delete\b",
            r"\bsudo\b",
            r"\buninstall\b",
        ]
    )

    # Maximum actions per task
    max_actions_per_task: int = 50

    # Action timeout in seconds
    action_timeout: float = 30.0


class SafetyChecker:
    """Validates actions before execution for safety."""

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.action_count = 0

    def validate_action(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate an action before execution.

        Returns:
            Tuple of (is_safe, reason)
        """
        self.action_count += 1

        # Circuit breaker - too many actions
        if self.action_count > self.config.max_actions_per_task:
            return False, f"Maximum actions ({self.config.max_actions_per_task}) exceeded"

        action_type = action.get("action", "")

        # Check coordinate safety
        if "coordinate" in action:
            x, y = action["coordinate"]
            for rx, ry, rw, rh in self.config.avoid_regions:
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    return False, f"Coordinate ({x}, {y}) in restricted region"

        # Check text for sensitive command patterns
        if action_type == "type":
            import re
            text = action.get("text", "").lower()
            for pattern in self.config.sensitive_patterns:
                if re.search(pattern, text):
                    return False, f"Sensitive command pattern matched: {pattern}"

        return True, "OK"

    def reset(self):
        """Reset action counter for new task."""
        self.action_count = 0


class ClaudeComputerUse:
    """
    Main class for Claude Computer Use integration.

    Implements the agent loop pattern where Claude requests actions,
    we execute them, capture screenshots, and return results until
    the task is complete.

    Attributes:
        config: ClaudeConfig instance with API and display settings
        safety: SafetyChecker for action validation

    Callbacks:
        on_action_start: Called before each action (action_type, action_dict)
        on_action_complete: Called after each action (action_type, success, error)
        on_screenshot: Called when screenshot is taken (PIL.Image)
        on_iteration: Called each agent loop iteration (iteration_num, total)
        on_task_complete: Called when task finishes (success, message)
    """

    def __init__(self, config: Optional[ClaudeConfig] = None):
        """
        Initialize Claude Computer Use agent.

        Args:
            config: Optional ClaudeConfig. If None, loads from environment.

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set.
        """
        self.config = config or ClaudeConfig.from_env()

        if not self.config.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Please set it in your .env file."
            )

        self.client = anthropic.Anthropic(api_key=self.config.api_key)
        self.safety = SafetyChecker()

        # Get actual screen dimensions
        self.actual_width, self.actual_height = pyautogui.size()
        self._calculate_scale_factors()

        # Callbacks for UI integration
        self.on_action_start: Optional[Callable] = None
        self.on_action_complete: Optional[Callable] = None
        self.on_screenshot: Optional[Callable] = None
        self.on_iteration: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None

        # Disable pyautogui failsafe for accessibility use
        pyautogui.FAILSAFE = False

    def _calculate_scale_factors(self):
        """Calculate scale factors for coordinate transformation."""
        self.scale_x = self.config.display_width / self.actual_width
        self.scale_y = self.config.display_height / self.actual_height

        print(f"Screen: {self.actual_width}x{self.actual_height}")
        print(f"Claude display: {self.config.display_width}x{self.config.display_height}")
        print(f"Scale factors: {self.scale_x:.3f}x, {self.scale_y:.3f}y")

    # =========================================================================
    # SCREENSHOT CAPTURE
    # =========================================================================

    def take_screenshot(self) -> Tuple[str, str]:
        """
        Capture the screen and return as base64 encoded image.

        Resizes to configured display dimensions for Claude and
        optionally compresses as JPEG for smaller payload.

        Returns:
            Tuple of (base64_data, media_type)
        """
        try:
            # Capture full screen
            screenshot = pyautogui.screenshot()

            # Resize to Claude's expected resolution
            screenshot = screenshot.resize(
                (self.config.display_width, self.config.display_height),
                Image.Resampling.LANCZOS
            )

            # Convert to base64
            buffer = io.BytesIO()

            if self.config.use_jpeg:
                # Use JPEG for smaller file size
                screenshot = screenshot.convert('RGB')
                screenshot.save(
                    buffer,
                    format="JPEG",
                    quality=self.config.screenshot_quality,
                    optimize=True
                )
                media_type = "image/jpeg"
            else:
                screenshot.save(buffer, format="PNG", optimize=True)
                media_type = "image/png"

            base64_image = base64.standard_b64encode(buffer.getvalue()).decode()

            # Callback for UI
            if self.on_screenshot:
                self.on_screenshot(screenshot)

            return base64_image, media_type

        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            raise

    # =========================================================================
    # COORDINATE SCALING
    # =========================================================================

    def _scale_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """
        Scale Claude's coordinates to actual screen resolution.

        Claude works with configured display dimensions (e.g., 1024x768),
        but the actual screen may be different (e.g., 1920x1080).
        This transforms coordinates appropriately.

        Args:
            x: X coordinate in Claude's space
            y: Y coordinate in Claude's space

        Returns:
            Tuple (screen_x, screen_y) in actual screen coordinates
        """
        screen_x = int(x / self.scale_x)
        screen_y = int(y / self.scale_y)

        # Clamp to screen bounds
        screen_x = max(0, min(screen_x, self.actual_width - 1))
        screen_y = max(0, min(screen_y, self.actual_height - 1))

        return screen_x, screen_y

    def _scale_from_screen(self, x: int, y: int) -> Tuple[int, int]:
        """
        Scale screen coordinates to Claude's coordinate space.

        Args:
            x: X coordinate in screen space
            y: Y coordinate in screen space

        Returns:
            Tuple (claude_x, claude_y) in Claude's coordinate space
        """
        claude_x = int(x * self.scale_x)
        claude_y = int(y * self.scale_y)
        return claude_x, claude_y

    # =========================================================================
    # ACTION EXECUTION
    # =========================================================================

    def execute_action(self, action: Dict[str, Any]) -> Tuple[str, str]:
        """
        Execute a Claude tool action and return screenshot result.

        Args:
            action: Dict containing 'action' type and parameters

        Returns:
            Tuple of (base64_screenshot, media_type)
        """
        action_type = action.get("action", "unknown")

        # Safety check
        is_safe, reason = self.safety.validate_action(action)
        if not is_safe:
            print(f"Safety check failed: {reason}")
            # Still return screenshot so Claude can see current state
            return self.take_screenshot()

        # Callback before action
        if self.on_action_start:
            self.on_action_start(action_type, action)

        success = True
        error_msg = None

        # Debug: log full action
        print(f"Executing: {action}")

        try:
            if action_type == ActionType.SCREENSHOT.value:
                pass  # Just return screenshot below

            elif action_type == ActionType.LEFT_CLICK.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                self._safe_click(screen_x, screen_y)

            elif action_type == ActionType.RIGHT_CLICK.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                self._safe_click(screen_x, screen_y, button='right')

            elif action_type == ActionType.DOUBLE_CLICK.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                self._safe_click(screen_x, screen_y, clicks=2)

            elif action_type == ActionType.TRIPLE_CLICK.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                self._safe_click(screen_x, screen_y, clicks=3)

            elif action_type == ActionType.MIDDLE_CLICK.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                self._safe_click(screen_x, screen_y, button='middle')

            elif action_type == ActionType.MOUSE_MOVE.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                pyautogui.moveTo(screen_x, screen_y, duration=0.2)

            elif action_type == ActionType.LEFT_CLICK_DRAG.value:
                start_x, start_y = action["start_coordinate"]
                end_x, end_y = action["end_coordinate"]
                start_screen = self._scale_to_screen(start_x, start_y)
                end_screen = self._scale_to_screen(end_x, end_y)

                pyautogui.moveTo(*start_screen)
                time.sleep(0.1)
                pyautogui.mouseDown()
                pyautogui.moveTo(*end_screen, duration=0.3)
                pyautogui.mouseUp()

            elif action_type == ActionType.LEFT_MOUSE_DOWN.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                pyautogui.moveTo(screen_x, screen_y)
                pyautogui.mouseDown()

            elif action_type == ActionType.LEFT_MOUSE_UP.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                pyautogui.moveTo(screen_x, screen_y)
                pyautogui.mouseUp()

            elif action_type == ActionType.TYPE.value:
                text = action["text"]
                # pyautogui.write doesn't handle all unicode well
                # Use typewrite for ASCII, write for unicode
                try:
                    pyautogui.write(text, interval=self.config.typing_interval)
                except Exception:
                    # Fallback: type character by character
                    for char in text:
                        pyautogui.press(char) if len(char) == 1 else None
                        time.sleep(self.config.typing_interval)

            elif action_type == ActionType.KEY.value:
                # Claude may send key in different formats
                key_combo = action.get("key") or action.get("text") or ""
                if not key_combo:
                    print(f"Warning: No key found in action. Full action: {action}")
                else:
                    print(f"Key combo: {key_combo}")
                    self._press_key_combo(key_combo)

            elif action_type == ActionType.HOLD_KEY.value:
                key = action["key"]
                self._press_key_combo(key, hold=True)

            elif action_type == ActionType.SCROLL.value:
                x, y = action["coordinate"]
                screen_x, screen_y = self._scale_to_screen(x, y)
                direction = action.get("scroll_direction", "down")
                amount = action.get("scroll_amount", 3)

                pyautogui.moveTo(screen_x, screen_y)
                scroll_value = amount if direction == "up" else -amount
                pyautogui.scroll(scroll_value)

            elif action_type == ActionType.WAIT.value:
                duration = action.get("duration", 1)
                time.sleep(duration)

            else:
                print(f"Unknown action type: {action_type}")

            # Delay after action for UI to update
            time.sleep(self.config.action_delay)

        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"Error executing action {action_type}: {e}")

        # Callback after action
        if self.on_action_complete:
            self.on_action_complete(action_type, success, error_msg)

        # Always return screenshot after action
        return self.take_screenshot()

    def _safe_click(
        self,
        x: int,
        y: int,
        button: str = 'left',
        clicks: int = 1
    ):
        """
        Perform a click with safety checks.

        Args:
            x: Screen X coordinate
            y: Screen Y coordinate
            button: 'left', 'right', or 'middle'
            clicks: Number of clicks (1, 2, or 3)
        """
        # Safety check for corner clicks (failsafe trigger area)
        if x < 5 and y < 5:
            print("Warning: Adjusting click away from failsafe corner (0,0)")
            x, y = max(10, x), max(10, y)

        print(f"Click: ({x}, {y}) button={button} clicks={clicks}")
        pyautogui.click(x, y, clicks=clicks, button=button)

    def _press_key_combo(self, key_combo: str, hold: bool = False):
        """
        Press a key or key combination.

        Supports formats like: 'enter', 'ctrl+c', 'cmd+shift+s'

        Args:
            key_combo: Key or combination string
            hold: If True, hold the key (for hold_key action)
        """
        # Normalize and split
        keys = key_combo.lower().replace(' ', '').split('+')

        # Map common key names to pyautogui names
        key_map = {
            'cmd': 'command',
            'ctrl': 'ctrl',
            'control': 'ctrl',
            'alt': 'alt',
            'option': 'alt',
            'shift': 'shift',
            'enter': 'enter',
            'return': 'enter',
            'esc': 'escape',
            'escape': 'escape',
            'tab': 'tab',
            'space': 'space',
            'backspace': 'backspace',
            'delete': 'delete',
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'home': 'home',
            'end': 'end',
            'pageup': 'pageup',
            'pagedown': 'pagedown',
            'capslock': 'capslock',
            'win': 'win',
            'windows': 'win',
            'super': 'win',
            'meta': 'win',
        }

        mapped_keys = [key_map.get(k, k) for k in keys]
        print(f"Key combo: {mapped_keys}")

        if len(mapped_keys) == 1:
            if hold:
                pyautogui.keyDown(mapped_keys[0])
            else:
                pyautogui.press(mapped_keys[0])
        else:
            pyautogui.hotkey(*mapped_keys)

    # =========================================================================
    # AGENT LOOP
    # =========================================================================

    def run_task(
        self,
        user_command: str,
        initial_screenshot: bool = True,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a task using Claude Computer Use agent loop.

        This implements the core agent pattern:
        1. Send user command (optionally with screenshot)
        2. Claude requests tool actions
        3. Execute actions and capture screenshots
        4. Return results to Claude
        5. Repeat until task complete or max iterations

        Args:
            user_command: Natural language description of the task
            initial_screenshot: Whether to include initial screenshot
            system_prompt: Optional custom system prompt

        Returns:
            Dict with keys:
                - success: bool
                - message: str (final response or error)
                - iterations: int
                - actions_taken: List[Dict]
        """
        print(f"\n{'='*60}")
        print(f"Starting Claude Computer Use task:")
        print(f"  Command: {user_command}")
        print(f"  Max iterations: {self.config.max_iterations}")
        print(f"{'='*60}\n")

        # Reset safety checker for new task
        self.safety.reset()

        # Default system prompt for Windows
        if system_prompt is None:
            system_prompt = """You are a computer use assistant helping a user on Windows 11.

IMPORTANT GUIDELINES:
1. To open applications on Windows, press the Windows key (use key "win") to open Start, then type the app name and press Enter.
2. Common Windows apps: "notepad" (not "notes"), "chrome", "firefox", "edge", "explorer", "cmd", "powershell"
3. Be efficient - don't take unnecessary screenshots or wait actions unless needed.
4. After clicking or typing, wait briefly for the UI to respond before the next action.
5. Use keyboard shortcuts when possible (Ctrl+C, Ctrl+V, Alt+F4, etc.)

For the current task, work step by step and complete it efficiently."""

        # Build initial message
        messages = []

        if initial_screenshot:
            # Include current screen state with command
            screenshot_data, media_type = self.take_screenshot()
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": screenshot_data
                        }
                    },
                    {
                        "type": "text",
                        "text": user_command
                    }
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": user_command
            })

        # Define tools
        tools = [
            {
                "type": self.config.tool_version,
                "name": "computer",
                "display_width_px": self.config.display_width,
                "display_height_px": self.config.display_height,
            }
        ]

        actions_taken = []
        iterations = 0

        while iterations < self.config.max_iterations:
            iterations += 1
            print(f"\n--- Iteration {iterations}/{self.config.max_iterations} ---")

            if self.on_iteration:
                self.on_iteration(iterations, self.config.max_iterations)

            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.client.beta.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        system=system_prompt,
                        messages=messages,
                        tools=tools,
                        betas=[self.config.beta_flag]
                    )
                    break
                except anthropic.RateLimitError:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        print(f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        error_msg = "Rate limit exceeded after retries"
                        print(error_msg)
                        if self.on_task_complete:
                            self.on_task_complete(False, error_msg)
                        return {
                            "success": False, "message": error_msg,
                            "iterations": iterations, "actions_taken": actions_taken
                        }
                except anthropic.APIConnectionError:
                    if attempt < max_retries - 1:
                        print(f"Connection error, retrying in 1s...")
                        time.sleep(1)
                    else:
                        error_msg = "API connection failed after retries"
                        print(error_msg)
                        if self.on_task_complete:
                            self.on_task_complete(False, error_msg)
                        return {
                            "success": False, "message": error_msg,
                            "iterations": iterations, "actions_taken": actions_taken
                        }
                except anthropic.APIError as e:
                    error_msg = f"API error: {e}"
                    print(error_msg)
                    if self.on_task_complete:
                        self.on_task_complete(False, error_msg)
                    return {
                        "success": False, "message": error_msg,
                        "iterations": iterations, "actions_taken": actions_taken
                    }

            # Add assistant response to conversation
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Check stop reason
            print(f"Stop reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
                # Claude finished the task
                final_message = self._extract_text_response(response.content)
                print(f"\nTask complete: {final_message}")

                if self.on_task_complete:
                    self.on_task_complete(True, final_message)

                return {
                    "success": True,
                    "message": final_message,
                    "iterations": iterations,
                    "actions_taken": actions_taken
                }

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    action = block.input
                    action_type = action.get("action", "unknown")

                    print(f"Action requested: {action_type}")
                    actions_taken.append(action)

                    screenshot_data, media_type = self.execute_action(action)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": screenshot_data
                                }
                            }
                        ]
                    })

            if not tool_results:
                # No tool calls - task might be complete
                final_message = self._extract_text_response(response.content)
                print(f"\nNo more actions. Final message: {final_message}")

                if self.on_task_complete:
                    self.on_task_complete(True, final_message)

                return {
                    "success": True,
                    "message": final_message,
                    "iterations": iterations,
                    "actions_taken": actions_taken
                }

            # Add tool results for next iteration
            messages.append({
                "role": "user",
                "content": tool_results
            })

        # Max iterations reached
        warning_msg = f"Max iterations ({self.config.max_iterations}) reached without completing task"
        print(f"\n{warning_msg}")

        if self.on_task_complete:
            self.on_task_complete(False, warning_msg)

        return {
            "success": False,
            "message": warning_msg,
            "iterations": iterations,
            "actions_taken": actions_taken
        }

    def _extract_text_response(self, content: List) -> str:
        """Extract text content from Claude's response blocks."""
        text_parts = []
        for block in content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return " ".join(text_parts) if text_parts else "Task completed"


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_task(command: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run a task with default configuration.

    Args:
        command: Natural language task description
        **kwargs: Additional arguments passed to run_task()

    Returns:
        Task result dictionary
    """
    agent = ClaudeComputerUse()
    return agent.run_task(command, **kwargs)


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("Claude Computer Use - Test Mode")
    print("=" * 40)

    # Check configuration
    config = ClaudeConfig.from_env()

    if config.api_key == "your-api-key-here" or not config.api_key:
        print("\nERROR: Please set ANTHROPIC_API_KEY in .env file")
        print("Current value:", config.api_key[:20] + "..." if config.api_key else "Not set")
        exit(1)

    print(f"API Key: {config.api_key[:8]}...")
    print(f"Model: {config.model}")
    print(f"Display: {config.display_width}x{config.display_height}")

    # Test screenshot
    print("\nTesting screenshot capture...")
    agent = ClaudeComputerUse(config)
    screenshot_data, media_type = agent.take_screenshot()
    print(f"Screenshot captured: {len(screenshot_data)} bytes ({media_type})")

    # Interactive test
    print("\n" + "=" * 40)
    test_command = input("Enter a test command (or 'skip' to exit): ")

    if test_command.lower() != 'skip':
        result = agent.run_task(test_command)
        print("\n" + "=" * 40)
        print("Result:")
        print(f"  Success: {result['success']}")
        print(f"  Message: {result['message']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Actions: {len(result['actions_taken'])}")
