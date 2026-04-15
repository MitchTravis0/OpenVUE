# OpenVUE - Free Hands-Free Computer Control for Accessibility

**OpenVUE** is a free, open-source accessibility solution designed for individuals with physical disabilities or motor function limitations. It enables complete hands-free computer control through head movements, eye gestures, voice commands, and AI-powered natural language task execution.

## Why OpenVUE?

Many people with conditions like ALS, paralysis, limited motor control, or other physical disabilities cannot use traditional input devices. Commercial accessibility solutions are often expensive and proprietary. OpenVUE provides a **completely free alternative** that combines:

- **Head Gaze Tracking** - Control your cursor by moving your head
- **Eye Gesture Detection** - Click using winks and blinks
- **Dwell Clicking** - Hold the cursor still to click (for users who find winking difficult)
- **Hybrid Gaze Tracking** - Iris-based coarse positioning combined with head-based fine control
- **Head Gesture Scrolling** - Tilt your head to scroll
- **Speech-to-Text** - Type and dictate using your voice
- **AI Computer Use Agent** - Execute complex tasks through natural language using Claude AI

## Features

### Head-Controlled Mouse
- Real-time cursor control through head movements
- 6-DOF head pose estimation via OpenCV solvePnP
- 5-point calibration system for personalized accuracy
- One Euro Filter smoothing for minimal jitter and lag
- Velocity-adaptive Kalman filter for smooth, stable cursor movement
- Non-linear acceleration curve (precise near center, fast at edges)
- Configurable dead zone to eliminate micro-jitter
- Quick re-center hotkey (no full recalibration needed)
- Adaptive calibration that improves over time with outlier rejection
- Periodic MediaPipe reset to prevent long-session drift
- Automatic camera disconnect recovery
- Saved calibration profiles

### Click Modes
- **Wink mode** (default):
  - Left wink/blink for left mouse click
  - Right wink/blink for right mouse click
  - Double wink for double-click
  - Eye Aspect Ratio (EAR) detection for reliability
- **Dwell mode**: Hold the cursor still for a configurable duration to click
  - Visual progress ring indicator
  - Configurable dwell time and tolerance zone
  - Toggle between modes with the `D` key

### Hybrid Gaze + Head Tracking
- Uses MediaPipe iris landmarks for coarse gaze-based cursor jumps
- Head tracking handles fine positioning within the gazed region
- Dramatically improves cursor travel speed
- Toggle with the `G` key

### Head Gesture Scrolling
- Tilt head up/down beyond a threshold to scroll
- Scroll speed proportional to tilt magnitude
- Toggle with the `S` key

### Voice Input
- Multiple speech-to-text engines with automatic fallback:
  - **whisper-cpp** (recommended) - Fast, lightweight
  - **faster-whisper** - Optimized Whisper variant
  - **OpenAI Whisper** - Full implementation
  - **Vosk** - Lightweight offline recognition
- Voice Activity Detection for automatic speech segmentation
- Noise filtering for common spurious outputs
- Bounded result queue with overflow protection
- "Processing speech..." status indicator during transcription

### AI-Powered Task Execution
- Natural language commands like "Open Chrome and search for Python tutorials"
- Uses Claude's Computer Use API for visual understanding
- Self-correcting agent loop: screenshot → analyze → act → verify
- 18+ action types including click, drag, type, scroll, keyboard shortcuts
- Built-in safety limits, regex-based destructive command detection, and restricted region checks
- Automatic retry with exponential backoff on rate limits and connection errors

### Settings & Configuration
- Persistent settings stored in `settings.json`
- PyQt6 settings dialog accessible from launcher (⚙️ button):
  - Camera index
  - Dead zone size
  - Click mode (wink/dwell) and dwell parameters
  - Acceleration curve
  - MediaPipe reset interval
  - STT engine, model, and device

### Session Statistics
- Live display on the tracking window:
  - Session duration
  - Total clicks
  - Adaptive refinement count
  - FPS

### Modern Control Panel
- Auto-hiding panel that appears when cursor approaches screen edge
- Quick access to all features:
  - **Talk** - Toggle speech-to-text
  - **Assist** - Launch AI task execution with voice input
  - **Pause/Resume** - Control head tracking
  - **Recall** - Restart calibration
- Keyboard shortcut (Ctrl+Shift+H) for quick access

## System Requirements

- Python 3.10 or higher (3.12 recommended; Python 3.14 may require MSVC build tools for some dependencies)
- Webcam (for head/eye tracking)
- Microphone (for voice input)
- Windows 10/11 (primary support) / macOS / Linux
- Anthropic API key (for Claude Computer Use features)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MitchTravis0/OpenVUE.git
   cd OpenVUE
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

4. **Optional STT configuration** (in `.env` or via the settings dialog):
   ```
   STT_ENGINE=whisper-cpp    # Options: whisper-cpp, faster-whisper, vosk, whisper
   WHISPER_MODEL=base        # Options: tiny, base, small, medium, large
   WHISPER_DEVICE=cpu        # Options: cpu, cuda
   ```

## Usage

### Starting the Application

**Option 1 - Full System with Control Panel:**
```bash
python pyhandler.py
```
This launches the complete system with the auto-hiding control panel.

**Option 2 - GUI Launcher (recommended):**
```bash
python pygui.py
```
Opens a launcher interface with Start, Settings (⚙️), and Quit buttons.

**Option 3 - Head Tracking Only:**
```bash
python headtracker.py
```
Runs standalone head gaze tracking.

### Calibration

1. Launch the application
2. Follow the on-screen instructions for 5-point calibration
3. Look at each calibration point (center + 4 corners) as directed
4. The system adapts to your specific head movements
5. Press `P` at any time during tracking to save the current calibration as a profile

### Tracking Window Hotkeys

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `C` | Recalibrate (full 5-point) |
| `R` | Quick re-center (no full recalibration) |
| `D` | Toggle click mode (wink/dwell) |
| `S` | Toggle head gesture scrolling |
| `G` | Toggle hybrid gaze jumping |
| `P` | Save current calibration as profile |

### Using Head Tracking

- Move your head to control cursor position
- The system uses your calibrated range of motion
- If the cursor drifts, press `R` for an instant re-center (faster than full recalibration)

### Using Eye Gestures (Wink mode)

- **Left wink** = Left mouse click
- **Right wink** = Right mouse click
- **Double wink** = Double-click

### Using Dwell Mode

Press `D` in the tracking window to switch to dwell mode:
- Hold the cursor still over the target
- A progress ring shows the dwell countdown
- Click triggers automatically after the configured duration

### Using Head Gesture Scrolling

With scroll enabled (`S` key):
- Tilt your head **down** to scroll down
- Tilt your head **up** to scroll up
- Hold the tilt for ~0.5s to engage scrolling
- Scroll speed increases with tilt magnitude

### Using Voice Input

1. Click **Talk** on the control panel (or use keyboard shortcut)
2. Speak clearly toward your microphone
3. Text is automatically typed at cursor position
4. Click **Talk** again to stop listening

### Using AI Assistant

1. Click **Assist** on the control panel
2. Speak or type your command (e.g., "Open Chrome and search for weather")
3. Claude analyzes your screen and executes the task
4. The AI takes screenshots, plans actions, and self-corrects

## Project Structure

| File | Description |
|------|-------------|
| `pyhandler.py` | Main control panel and system manager |
| `headtracker.py` | Head tracking, eye gestures, dwell click, gaze, scrolling |
| `STT.py` | Speech-to-text engine with multiple backends |
| `claude_computer_use.py` | Claude AI Computer Use integration |
| `action_executor.py` | Action executor wrapper for Claude |
| `executions.py` | Low-level keyboard/mouse control |
| `pygui.py` | GUI launcher |
| `settings_dialog.py` | PyQt6 settings dialog |
| `config.py` | Persistent configuration system (`settings.json`) |
| `logger.py` | Logging utilities |
| `camera_diagnostic.py` | Camera troubleshooting tool |
| `camera_test.py` | Quick camera format testing |
| `profiles/` | Saved calibration profiles (JSON) |

## Troubleshooting

### Camera Issues
Run the diagnostic tool:
```bash
python camera_diagnostic.py
```
This tests multiple camera indices, backends, and formats. You can also change the camera index in the settings dialog.

If the camera disconnects during tracking, OpenVUE will automatically attempt reconnection.

### Head Tracking Drift
- Press `R` for instant re-center
- If drift persists, press `C` for full recalibration
- Ensure good lighting conditions
- Make sure your face is clearly visible to the camera

### Dwell or Scroll Too Sensitive
Open the settings dialog and adjust:
- **Dead Zone** for cursor stability
- **Dwell Time** and **Dwell Tolerance** for dwell clicking
- **Acceleration** for cursor speed

### Speech Recognition Issues
- Check microphone permissions
- Verify audio input device in system settings
- Try a different STT engine in the settings dialog

### Claude Computer Use Not Working
- Verify your `ANTHROPIC_API_KEY` is set correctly
- Check your API key has access to Claude Computer Use beta

### Dependency Build Errors on Python 3.14
Some packages may not have pre-built wheels for Python 3.14 yet. Either:
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Or use Python 3.12/3.13

## Technology Stack

- **Computer Vision**: MediaPipe (FaceMesh with iris landmarks), OpenCV (solvePnP)
- **Signal Processing**: FilterPy (Kalman), OneEuroFilter
- **Speech Recognition**: whisper-cpp, faster-whisper, OpenAI Whisper, Vosk
- **AI**: Anthropic Claude (Computer Use API)
- **UI Framework**: PyQt6
- **Input Control**: PyAutoGUI, pynput

## Contributing

Contributions are welcome! OpenVUE is a community project aimed at making computer accessibility free and available to everyone.

## License

This project is licensed under the GNU General Public License v3 (see `GNUGPL-LICENSE`).

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face mesh and iris tracking
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Anthropic Claude](https://www.anthropic.com/) for AI computer use
- [FilterPy](https://github.com/rlabbe/filterpy) for Kalman filtering
- [OneEuroFilter](https://cristal.univ-lille.fr/~casiez/1euro/) for adaptive pose smoothing
- [Vosk](https://alphacephei.com/vosk/) for offline speech recognition

---

**OpenVUE** - Making computers accessible to everyone, for free.
