# OpenVUE - Free Hands-Free Computer Control for Accessibility

**OpenVUE** is a free, open-source accessibility solution designed for individuals with physical disabilities or motor function limitations. It enables complete hands-free computer control through head movements, eye gestures, voice commands, and AI-powered natural language task execution.

## Why OpenVUE?

Many people with conditions like ALS, paralysis, limited motor control, or other physical disabilities cannot use traditional input devices. Commercial accessibility solutions are often expensive and proprietary. OpenVUE provides a **completely free alternative** that combines:

- **Head Gaze Tracking** - Control your cursor by moving your head
- **Eye Gesture Detection** - Click using winks and blinks
- **Speech-to-Text** - Type and dictate using your voice
- **AI Computer Use Agent** - Execute complex tasks through natural language using Claude AI

## Features

### Head-Controlled Mouse
- Real-time cursor control through head movements
- 5-point calibration system for personalized accuracy
- Kalman filtering for smooth, stable cursor movement
- Adaptive calibration that improves over time

### Eye Gesture Detection
- Left wink/blink for left mouse click
- Right wink/blink for right mouse click
- Double wink for double-click
- Eye Aspect Ratio (EAR) detection for reliability

### Voice Input
- Multiple speech-to-text engines with automatic fallback:
  - **whisper-cpp** (recommended) - Fast, lightweight
  - **faster-whisper** - Optimized Whisper variant
  - **OpenAI Whisper** - Full implementation
  - **Vosk** - Lightweight offline recognition
- Voice Activity Detection for automatic speech segmentation
- Noise filtering for common spurious outputs

### AI-Powered Task Execution
- Natural language commands like "Open Chrome and search for Python tutorials"
- Uses Claude's Computer Use API for visual understanding
- Self-correcting agent loop: screenshot → analyze → act → verify
- 18+ action types including click, drag, type, scroll, keyboard shortcuts
- Built-in safety limits and restricted region detection

### Modern Control Panel
- Auto-hiding panel that appears when cursor approaches screen edge
- Quick access to all features:
  - **Talk** - Toggle speech-to-text
  - **Assist** - Launch AI task execution with voice input
  - **Pause/Resume** - Control head tracking
  - **Recall** - Restart calibration
- Keyboard shortcut (Ctrl+Shift+H) for quick access

## System Requirements

- Python 3.8 or higher
- Webcam (for head/eye tracking)
- Microphone (for voice input)
- macOS (primary support) / Windows / Linux
- Anthropic API key (for Claude Computer Use features)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/OpenVUE.git
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

4. **Optional STT configuration** (in `.env`):
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

**Option 2 - GUI Launcher:**
```bash
python pygui.py
```
Opens a launcher interface to start calibration.

**Option 3 - Head Tracking Only:**
```bash
python pipefacemac.py
```
Runs standalone head gaze tracking.

### Calibration

1. Launch the application
2. Follow the on-screen instructions for 5-point calibration
3. Look at each calibration point (center + 4 corners) as directed
4. The system adapts to your specific head movements

### Using Head Tracking

- Move your head to control cursor position
- The system uses your calibrated range of motion
- Kalman filtering ensures smooth cursor movement

### Using Eye Gestures

- **Left wink** = Left mouse click
- **Right wink** = Right mouse click
- **Double wink** = Double-click

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
| `pipefacemac.py` | Head tracking and eye gesture detection |
| `STT.py` | Speech-to-text engine with multiple backends |
| `claude_computer_use.py` | Claude AI Computer Use integration |
| `action_exectuor_mac.py` | Action executor wrapper for Claude |
| `executions.py` | Low-level keyboard/mouse control |
| `pygui.py` | Simple GUI launcher |
| `logger.py` | Logging utilities |
| `camera_diagnostic.py` | Camera troubleshooting tool |
| `camera_test.py` | Quick camera format testing |

## Troubleshooting

### Camera Issues
Run the diagnostic tool:
```bash
python camera_diagnostic.py
```
This tests multiple camera indices, backends, and formats.

### Head Tracking Not Working
- Ensure good lighting conditions
- Run calibration again
- Make sure your face is clearly visible to the camera

### Speech Recognition Issues
- Check microphone permissions
- Verify audio input device in system settings
- Try a different STT engine via environment variable

### Claude Computer Use Not Working
- Verify your `ANTHROPIC_API_KEY` is set correctly
- Check your API key has access to Claude Computer Use beta

## Technology Stack

- **Computer Vision**: MediaPipe, OpenCV
- **Speech Recognition**: Whisper, Vosk
- **AI**: Anthropic Claude (Computer Use API)
- **UI Framework**: PyQt6
- **Input Control**: PyAutoGUI, pynput
- **Signal Processing**: FilterPy (Kalman filtering)

## Contributing

Contributions are welcome! OpenVUE is a community project aimed at making computer accessibility free and available to everyone.

## License

This project is open source and free to use for accessibility purposes.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face mesh tracking
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Anthropic Claude](https://www.anthropic.com/) for AI computer use
- [FilterPy](https://github.com/rlabbe/filterpy) for Kalman filtering
- [Vosk](https://alphacephei.com/vosk/) for offline speech recognition

---

**OpenVUE** - Making computers accessible to everyone, for free.
