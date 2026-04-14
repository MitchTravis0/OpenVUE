"""
OpenVUE Configuration System

Provides persistent settings via settings.json with sensible defaults.
"""

import json
import os
from dataclasses import dataclass, field, asdict

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")


@dataclass
class TrackingConfig:
    camera_index: int = 1
    dead_zone_pixels: int = 8
    click_mode: str = "wink"  # "wink" or "dwell"
    dwell_time: float = 1.0
    dwell_tolerance: int = 15
    acceleration_exponent: float = 0.7
    mediapipe_detection_confidence: float = 0.5
    mediapipe_tracking_confidence: float = 0.55
    mediapipe_reset_minutes: int = 5
    kalman_base_r: float = 400.0
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007


@dataclass
class STTConfig:
    engine: str = "whisper-cpp"
    model: str = "base"
    device: str = "cpu"


@dataclass
class AppConfig:
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stt: STTConfig = field(default_factory=STTConfig)


def load_config() -> AppConfig:
    """Load configuration from settings.json, falling back to defaults."""
    config = AppConfig()
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
            # Merge tracking settings
            if "tracking" in data:
                for key, value in data["tracking"].items():
                    if hasattr(config.tracking, key):
                        setattr(config.tracking, key, value)
            # Merge STT settings
            if "stt" in data:
                for key, value in data["stt"].items():
                    if hasattr(config.stt, key):
                        setattr(config.stt, key, value)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings.json: {e}. Using defaults.")
    return config


def save_config(config: AppConfig):
    """Save configuration to settings.json."""
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(asdict(config), f, indent=2)
