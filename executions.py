from platform import system

from pynput.keyboard import Key, Controller
from pynput.mouse import Button, Controller as MouseController
import pyautogui
import uuid
import threading
import time
import os
from PIL import ImageGrab
from datetime import datetime
import requests
import base64
import json
import re
import pyautogui
from PIL import Image, ImageDraw, ImageEnhance
from typing import Tuple


keyboard = Controller()
mouse = MouseController()
held_keys = {}

DEFAULT_SCREEN_WIDTH = 3024  # Default target resolution width
DEFAULT_SCREEN_HEIGHT = 1964  # Default target resolution height

KEY_MAP = {
    'enter': Key.enter,
    'return': Key.enter,
    'tab': Key.tab,
    'esc': Key.esc,
    'escape': Key.esc,
    'space': Key.space,
    'up': Key.up,
    'down': Key.down,
    'left': Key.left,
    'right': Key.right,
    'cmd': Key.cmd,           # macOS Command key
    'command': Key.cmd,
    'option': Key.alt,        # macOS Option key maps to alt
    'alt': Key.alt,
    'shift': Key.shift,
    'delete': Key.delete,
    'backspace': Key.backspace,
    'home': Key.home,
    'end': Key.end,
    'pageup': Key.page_up,
    'pagedown': Key.page_down,
    'capslock': Key.caps_lock,
    # Function keys
    **{f'f{i}': getattr(Key, f'f{i}') for i in range(1, 13)}
}

# misc functions

def hold_key(key):
    if key in held_keys:
        print(f"Key '{key}' already held.")
        return

    def hold():
        keyboard.press(key)
        while key in held_keys:
            time.sleep(0.1)

    t = threading.Thread(target=hold)
    t.start()
    held_keys[key] = t
    print(f"Holding key '{key}'...")

def release_key(key):
    if key in held_keys:
        del held_keys[key]
        keyboard.release(key)
        print(f"Released key '{key}'.")
    else:
        print(f"Key '{key}' is not currently held.")

def open_application_via_search(app_name):
    """
    A reliable way to open applications by using the Spotlight Search

    Args:
        app_name: Name of the application to open (e.g., 'discord', 'chrome')

    Returns:
        True if the action sequence completed, False on error
    """
    try:
        print(f"ACTION: Opening {app_name} via Windows search")

        # Open Spotlight search
        hold_key(Key.cmd)  # NOT 'key.cmd' as a string
        time.sleep(1)
        hold_key(' ')  # spacebar
        time.sleep(2)
        release_key(Key.cmd)
        time.sleep(1)
        release_key(' ')

        # Type the application name
        keyboard.type(app_name)
        time.sleep(1)

        # Press Enter to launch
        keyboard.press(Key.enter)
        time.sleep(3)  # Wait for app to launch

        print(f"Successfully initiated launch sequence for {app_name}")
        return True
    except Exception as e:
        print(f"Error opening {app_name}: {e}")
        return False

def type_text(text_to_type):
    """Types the given text using the keyboard."""
    print(f"ACTION: Typing text: '{text_to_type[:50]}...'") # Log snippet
    try:
        # Add a small delay before typing starts, helps ensure focus
        time.sleep(0.3)
        keyboard.type(text_to_type)
        return True
    except Exception as e:
        print(f"Error typing text: {e}")
        return False

def press_key(key_name):
    """Presses a special key using pynput (macOS compatible)."""
    key_lower = key_name.lower()
    key = KEY_MAP.get(key_lower)

    if key:
        print(f"ACTION: Pressing key: '{key_lower}'")
        try:
            keyboard.press(key)
            keyboard.release(key)
            time.sleep(0.3)  # Small pause after key press
            return True
        except Exception as e:
            print(f"Error pressing key '{key_lower}': {e}")
            return False
    else:
        print(f"Warning: Invalid or unsupported key requested: '{key_name}'")
        return False

def wait(seconds):
    """Pauses execution for a number of seconds."""
    print(f"ACTION: Waiting for {seconds} seconds...")
    try:
        time.sleep(float(seconds))
        return True
    except ValueError:
        print(f"Error: Invalid wait time '{seconds}'")
        return False

def normalize_coordinates(x, y):
    """
    Normalizes coordinates to match the actual screen resolution.
    Takes coordinates that might be based on default 1920x1080 resolution
    and adjusts them to the actual screen resolution.

    Args:
        x: X coordinate (potentially based on 1920x1080)
        y: Y coordinate (potentially based on 1920x1080)

    Returns:
        Tuple (x, y) with normalized coordinates for actual screen
    """
    try:
        # Convert to integers if they are not already
        x, y = int(float(x)), int(float(y))

        # Get actual screen size
        actual_width, actual_height = pyautogui.size()

        # If we're already using the target resolution, no need to adjust
        if actual_width == DEFAULT_SCREEN_WIDTH and actual_height == DEFAULT_SCREEN_HEIGHT:
            return x, y

        # Calculate scale factors for width and height
        width_scale = actual_width / DEFAULT_SCREEN_WIDTH
        height_scale = actual_height / DEFAULT_SCREEN_HEIGHT

        # Scale coordinates
        scaled_x = int(x * width_scale)
        scaled_y = int(y * height_scale)

        # Ensure coordinates are within screen bounds
        scaled_x = max(0, min(scaled_x, actual_width - 1))
        scaled_y = max(0, min(scaled_y, actual_height - 1))

        # Debug information
        if scaled_x != x or scaled_y != y:
            print(f"Normalized coordinates: ({x}, {y}) -> ({scaled_x}, {scaled_y})")
            print(
                f"Screen resolution: {actual_width}x{actual_height}, Scale factors: {width_scale:.2f}x{height_scale:.2f}")

        return scaled_x, scaled_y
    except Exception as e:
        print(f"Error normalizing coordinates: {e}")
        # Return original coordinates if something went wrong
        return x, y


def click_location(x, y):
    """Moves the mouse to (x, y) and clicks using pynput. Uses pyautogui for screen size only."""
    try:
        # Convert to integers if they are not already
        x, y = int(float(x)), int(float(y))

        # Normalize coordinates
        x_norm, y_norm = normalize_coordinates(x, y)

        print(f"ACTION: Clicking at ({x_norm}, {y_norm})")

        # Get screen size using pyautogui
        screen_width, screen_height = pyautogui.size()

        # Safety check to avoid (0,0) unless it's inside Discord taskbar area
        if (x == 0 and y == 0) and not (192 <= x <= 384 and y >= 1000):
            print("Warning: Avoiding click at (0,0) which can trigger failsafe-like behavior")
            x, y = screen_width // 2, screen_height - 40
            print(f"Using safer default position instead: ({x}, {y})")

        # Bounds check
        if not (0 <= x < screen_width and 0 <= y < screen_height):
            print(f"Warning: Click coordinates ({x},{y}) may be out of screen bounds ({screen_width}x{screen_height}).")

        # Move and click
        mouse.position = (x, y)
        time.sleep(0.05)  # Let the mouse settle
        mouse.click(Button.left, 1)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"Error clicking at ({x}, {y}): {e}")
        return False

def execute_comment(comment_text):
    """Prints a comment from the plan (useful for non-executable steps)."""
    print(f"COMMENT: {comment_text}")
    return True

def right_click(x, y):
    """
    Right-clicks at the specified coordinates.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        True if successful, False otherwise
    """
    try:
        x, y = int(float(x)), int(float(y))

        # Normalize coordinates for actual screen resolution
        x, y = normalize_coordinates(x, y)

        # Safety check to avoid right-clicking at (0,0) which triggers PyAutoGUI failsafe
        if (x == 0 and y == 0) and not (192 <= x <= 384 and y >= 1000):  # Allow Discord taskbar area
            print("Warning: Avoiding right-click at (0,0) which can trigger failsafe")
            screen_width, screen_height = pyautogui.size()
            x, y = screen_width // 2, screen_height - 40
            print(f"Using safer default position instead: ({x}, {y})")

        print(f"ACTION: Right-clicking at ({x}, {y})")

        # Move the mouse to the coordinates and perform a right-click
        mouse.position = (x, y)
        time.sleep(0.5)
        mouse.click(Button.right, 1)  # Right-click
        return True

    except Exception as e:
        print(f"Error right-clicking at ({x}, {y}): {e}")
        return False

def double_click(x, y):
    """
    Double-clicks at the specified coordinates.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        True if successful, False otherwise
    """
    try:
        x, y = int(float(x)), int(float(y))

        # Normalize coordinates for actual screen resolution
        x, y = normalize_coordinates(x, y)

        # Safety check to avoid double-clicking at (0,0) which triggers PyAutoGUI failsafe
        if (x == 0 and y == 0) and not (192 <= x <= 384 and y >= 1000):  # Allow Discord taskbar area
            print("Warning: Avoiding double-click at (0,0) which can trigger failsafe")
            screen_width, screen_height = pyautogui.size()
            x, y = screen_width // 2, screen_height - 40
            print(f"Using safer default position instead: ({x}, {y})")

        print(f"ACTION: Double-clicking at ({x}, {y})")

        # Move the mouse to the coordinates and perform a double-click
        mouse.position = (x, y)
        mouse.click(Button.left, 2)  # Double-click (left button)
        time.sleep(0.3)
        return True

    except Exception as e:
        print(f"Error double-clicking at ({x}, {y}): {e}")
        return False

def search_browser(term):
    try:
        print(f"ACTION: Searching {term} via shortcuts")

        # Open Spotlight search
        hold_key(Key.cmd)  # NOT 'key.cmd' as a string
        time.sleep(1)
        hold_key('l')  # spacebar
        time.sleep(2)
        release_key(Key.cmd)
        time.sleep(1)
        release_key('l')

        # Type the application name
        keyboard.type(term)
        time.sleep(1)

        # Press Enter to launch
        keyboard.press(Key.enter)
        time.sleep(3)  # Wait for app to launch

        print(f"Successfully searched term {term}")
        return True
    except Exception as e:
        print(f"Error searching {term}: {e}")
        return False


# --- Plan Parsers ---

def execute_plan(action_obj, use_json_format=True):
    """Execute a single action object or command and return success status."""

    if use_json_format:
        # JSON format
        if not isinstance(action_obj, dict) or "action" not in action_obj:
            print(f"Skipping invalid action object: {action_obj}")
            return False

        command_name = action_obj["action"]
        try:
            # Original actions
            if command_name == "run_app":
                # This is now deprecated - we should warn the user
                app = action_obj.get("app_name")
                open_application_via_search(app)
                time.sleep(1)
                return True
            elif command_name == "search":
                search_term = action_obj.get("term")
                time.sleep(1)
                search_browser(search_term)
            elif command_name == "type_text":
                text = action_obj.get("text")
                if text is not None:
                    return type_text(text)
                else:
                    print("Error: 'text' missing for type_text")
            elif command_name == "press_key":
                key = action_obj.get("key")
                if key:
                    return press_key(key)
                else:
                    print("Error: 'key' missing for press_key")
            elif command_name == "go_to":
                n = action_obj.get("nth_term")
                press_key("tab")
                wait(1)
                press_key("return")
                for i in range(0, n):
                    press_key("tab")
            elif command_name == "wait":
                sec = action_obj.get("seconds")
                if sec is not None:
                    return wait(sec)
                else:
                    print("Error: 'seconds' missing for wait")
            # elif command_name == "click":
            #     x = action_obj.get("x")
            #     y = action_obj.get("y")
            #     if x is not None and y is not None:
            #         return click_location(x, y)
            #     else:
            #         print("Error: 'x' or 'y' missing for click")
            elif command_name == "comment":
                comment = action_obj.get("comment_text")
                if comment:
                    return execute_comment(comment)
                else:
                    print("Error: 'comment_text' missing for comment")

            elif command_name == "right_click":
                x = action_obj.get("x")
                y = action_obj.get("y")

                if x is not None and y is not None:
                    return right_click(x, y)
                else:
                    print("Error: 'x' or 'y' missing for right_click")

            elif command_name == "double_click":
                x = action_obj.get("x")
                y = action_obj.get("y")

                if x is not None and y is not None:
                    return double_click(x, y)
                else:
                    print("Error: 'x' or 'y' missing for double_click")

        except Exception as e:
            print(f"Critical error executing {action_obj}: {e}")
            return False

    return False  # Default return if no action was executed
