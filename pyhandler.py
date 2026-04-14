import sys
import os
import subprocess
import psutil
import time
import cv2
import action_executor

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QMessageBox,
    QSizePolicy,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QDialog,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    QProcess, Qt, QPropertyAnimation, QRect, QEasingCurve,
    QTimer, pyqtProperty, QPoint
)
from PyQt6.QtGui import QFont, QShortcut, QKeySequence, QCursor

from headtracker import HeadGazeTracker
from STT import (toggle_recording, initialize_model, is_recording, is_processing, result_queue)

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPEFACE_SCRIPT = os.path.join(SCRIPT_DIR, "headtracker.py")
PYGUI_SCRIPT = os.path.join(SCRIPT_DIR, "pygui.py")  # CHANGE PATH
PYTHON_EXECUTABLE = sys.executable

# --- UI Configuration ---
PANEL_WIDTH = 80  # Width of the control panel
ANIMATION_DURATION = 200  # Animation duration in milliseconds
RIGHT_SIDE_THRESHOLD = 150  # Cursor within this many pixels of right edge triggers show
CURSOR_POLL_INTERVAL = 50  # How often to check cursor position (ms)
AUTO_HIDE_DELAY = 300  # Delay before auto-hiding after cursor leaves (ms)

# Color scheme - Modern dark theme
COLORS = {
    'bg_primary': '#1a1a2e',
    'bg_secondary': '#16213e',
    'accent': '#e94560',
    'accent_hover': '#ff6b6b',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0a0',
    'border': '#0f3460',
    'success': '#4ecca3',
    'warning': '#ffc107',
}
# --------------------


class AssistWidget(QDialog):
    """Widget that shows a text input and send button for Claude CUA with voice input."""

    def __init__(self, pause_callback=None, resume_callback=None):
        super().__init__()
        self.setWindowTitle("Claude Assistant")
        self.setModal(False)  # Non-modal so user can see screen
        self.setFixedSize(420, 260)
        self.is_running = False
        self.is_listening = False
        self.stt_thread = None
        self.pause_callback = pause_callback
        self.resume_callback = resume_callback

        # Modern styling for the dialog
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 14px;
                font-weight: 500;
            }}
            QLineEdit {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 1px solid {COLORS['accent']};
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_secondary']};
            }}
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.title_label = QLabel("Claude Computer Use")
        self.input = QLineEdit()
        self.input.setPlaceholderText("Listening... speak your command")
        self.input.returnPressed.connect(self.stop_listening_and_send)

        self.status_label = QLabel("Listening...")
        self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")

        # Button row
        button_layout = QHBoxLayout()

        self.send_button = QPushButton("Stop & Send")
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.clicked.connect(self.stop_listening_and_send)
        self.send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #5fd9b3;
            }}
        """)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.clicked.connect(self.cancel_and_close)
        self.cancel_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
                color: {COLORS['text_primary']};
            }}
        """)

        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.cancel_button)

        layout.addWidget(self.title_label)
        layout.addWidget(self.input)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.prompt_result = None

        # Auto-start listening when dialog opens
        QTimer.singleShot(100, self.start_listening)

    def start_listening(self):
        """Start speech-to-text recording."""
        if self.is_listening or self.is_running:
            return

        self.is_listening = True
        self.status_label.setText("Listening... speak your command")
        self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        self.input.setPlaceholderText("Listening... speak your command")
        self.input.clear()

        # Start STT recording
        if not is_recording:
            toggle_recording()

        # Start thread to capture STT results
        import threading
        self.stt_thread = threading.Thread(target=self._capture_stt_results, daemon=True)
        self.stt_thread.start()

        print("[AssistWidget] Started listening for voice input")

    def _capture_stt_results(self):
        """Background thread to capture STT results and update input field."""
        import queue as q
        from STT import is_processing as stt_is_processing

        while self.is_listening and not self.is_running:
            try:
                # Show processing indicator while STT is transcribing
                import STT
                if STT.is_processing:
                    self.status_label.setText("Processing speech...")

                item = result_queue.get(timeout=0.1)
                if item is None:
                    continue

                # Double-check we're still listening and not running CUA
                if not self.is_listening or self.is_running:
                    break

                type_flag, text = item
                if text and type_flag == "final":
                    # Append transcribed text to input field
                    current = self.input.text()
                    new_text = f"{current} {text}".strip() if current else text
                    self.input.setText(new_text)
                    self.status_label.setText("Listening...")
                    print(f"[AssistWidget] Captured: {text}")

            except q.Empty:
                continue
            except Exception as e:
                print(f"[AssistWidget] Error capturing STT: {e}")
                break

        print("[AssistWidget] STT capture thread stopped")

    def stop_listening(self):
        """Stop speech-to-text recording."""
        if not self.is_listening:
            return

        self.is_listening = False

        # Stop STT recording
        if is_recording:
            toggle_recording()

        # Clear any pending transcriptions from the queue
        import queue as q
        while True:
            try:
                result_queue.get_nowait()
            except q.Empty:
                break

        print("[AssistWidget] Stopped listening and cleared queue")

    def stop_listening_and_send(self):
        """Stop listening and send the command to Claude."""
        self.stop_listening()

        message = self.input.text().strip()
        if not message or self.is_running:
            if not message:
                self.status_label.setText("No command detected. Try again or type.")
                self.status_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 12px;")
                # Restart listening
                QTimer.singleShot(500, self.start_listening)
            return

        self.prompt_result = message
        self.is_running = True
        print(f"[AssistWidget] Sending to Claude CUA: {message}")

        # Pause head tracking so it doesn't interfere with Claude's mouse control
        if self.pause_callback:
            print("[AssistWidget] Pausing head tracking...")
            self.pause_callback()

        # Update UI
        self.send_button.setEnabled(False)
        self.send_button.setText("Processing...")
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Starting Claude CUA...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.input.setEnabled(False)

        # Run in background thread
        import threading

        def run_task():
            try:
                def status_callback(status, detail):
                    # Update status label
                    try:
                        self.status_label.setText(f"{status}")
                    except:
                        pass

                result = action_executor.digest_prompts(message, on_status=status_callback)

                # Update UI on completion
                success = result.get('success', False)
                final_msg = result.get('message', 'Done')[:50]
                self.status_label.setText(f"{'Done' if success else 'Failed'}: {final_msg}")

            except Exception as e:
                print(f"[AssistWidget] Error in Claude CUA: {e}")
                self.status_label.setText(f"Error: {str(e)[:30]}")

            finally:
                # Resume head tracking
                if self.resume_callback:
                    print("[AssistWidget] Resuming head tracking...")
                    self.resume_callback()

                self.is_running = False
                self.send_button.setEnabled(True)
                self.send_button.setText("Stop & Send")
                self.cancel_button.setEnabled(True)
                self.input.setEnabled(True)
                self.input.clear()
                # Restart listening for next command
                QTimer.singleShot(1000, self.start_listening)

        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()

    def cancel_and_close(self):
        """Cancel and close the dialog."""
        self.stop_listening()
        self.close()

    def closeEvent(self, event):
        """Ensure we stop listening when dialog closes."""
        self.stop_listening()
        event.accept()


class ProcessControlApp(QWidget):
    """Main application window for controlling external Python scripts.

    Features:
    - Slide animation to hide/show panel
    - Hover trigger zone at screen edge
    - Pull tab visible when hidden
    - Keyboard shortcut (Ctrl+Shift+H) to toggle
    - Modern dark theme styling
    """

    def __init__(self):
        super().__init__()
        # psutil handle for headtracker.py
        self.pipeface_process: psutil.Process | None = None
        self.is_paused = False
        self.is_hidden = True  # Start hidden
        self.is_animating = False

        # Store screen geometry
        self.screen_geo = QApplication.primaryScreen().availableGeometry()

        # Cursor position polling timer
        self.cursor_poll_timer = QTimer()
        self.cursor_poll_timer.timeout.connect(self._check_cursor_position)

        # Auto-hide delay timer
        self.auto_hide_timer = QTimer()
        self.auto_hide_timer.setSingleShot(True)
        self.auto_hide_timer.timeout.connect(self._on_auto_hide_timeout)

        self.init_ui()
        self._setup_keyboard_shortcuts()

        # Start cursor polling
        self.cursor_poll_timer.start(CURSOR_POLL_INTERVAL)
        initialize_model()
        self._start_pipeface()

    def init_ui(self):
        # Remove window frame and keep always on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)

        # Main container with styling
        self.container = QWidget(self)
        self.container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_primary']};
                border-left: 2px solid {COLORS['border']};
            }}
        """)

        # Button styling
        button_style = f"""
            QPushButton {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
                padding: 8px;
                margin: 4px 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent']};
            }}
        """

        # Main action buttons
        self.btn_talk = QPushButton("Talk")
        self.btn_talk.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btn_talk.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_talk.setStyleSheet(button_style)

        self.btn_assist = QPushButton("Assist")
        self.btn_assist.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btn_assist.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_assist.setStyleSheet(button_style)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btn_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pause.setStyleSheet(button_style)

        self.btn_recall = QPushButton("Recall")
        self.btn_recall.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btn_recall.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_recall.setStyleSheet(button_style)

        self.btn_quit = QPushButton("Quit")
        self.btn_quit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btn_quit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_quit.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['accent']};
                border: 1px solid {COLORS['accent']};
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
                padding: 8px;
                margin: 4px 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
                color: {COLORS['text_primary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_hover']};
            }}
        """)

        # Connect signals
        self.btn_talk.clicked.connect(self._handle_talk_click)
        self.btn_assist.clicked.connect(self.show_assist_ui)
        self.btn_pause.clicked.connect(self._toggle_pause_resume)
        self.btn_recall.clicked.connect(self._recall_pipeface)
        self.btn_quit.clicked.connect(self._quit_all)

        # Layout for container
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(4, 12, 4, 8)
        container_layout.setSpacing(4)

        # Action buttons
        container_layout.addWidget(self.btn_talk)
        container_layout.addWidget(self.btn_assist)
        container_layout.addWidget(self.btn_pause)
        container_layout.addWidget(self.btn_recall)
        container_layout.addStretch()
        container_layout.addWidget(self.btn_quit)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)

        self._position_on_right()
        self._setup_animation()
        self.show()

    def _setup_animation(self):
        """Setup animation for slide effect."""
        self.slide_animation = QPropertyAnimation(self, b"geometry")
        self.slide_animation.setDuration(ANIMATION_DURATION)
        self.slide_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.slide_animation.finished.connect(self._on_animation_finished)

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for panel control."""
        # Ctrl+Shift+H to toggle panel visibility
        self.shortcut_toggle = QShortcut(QKeySequence("Ctrl+Shift+H"), self)
        self.shortcut_toggle.activated.connect(self.toggle_panel)

        # Make shortcut work globally (when panel is hidden)
        self.shortcut_toggle.setContext(Qt.ShortcutContext.ApplicationShortcut)

    def _position_on_right(self):
        """Position the panel on the right side of the screen (starts hidden)."""
        window_height = self.screen_geo.height()

        # Start off-screen (hidden)
        self.setGeometry(
            self.screen_geo.width(),  # Off-screen to the right
            0,
            PANEL_WIDTH,
            window_height
        )
        self.container.setGeometry(0, 0, PANEL_WIDTH, window_height)

    def toggle_panel(self):
        """Toggle between hidden and visible states."""
        if self.is_animating:
            return

        if self.is_hidden:
            self.show_panel()
        else:
            self.hide_panel()

    def hide_panel(self):
        """Animate panel sliding off screen to the right."""
        if self.is_hidden or self.is_animating:
            return

        self.is_animating = True

        # Calculate positions
        start_rect = self.geometry()
        end_rect = QRect(
            self.screen_geo.width(),  # Off screen to the right
            0,
            PANEL_WIDTH,
            self.screen_geo.height()
        )

        # Setup and start animation
        self.slide_animation.setStartValue(start_rect)
        self.slide_animation.setEndValue(end_rect)
        self.slide_animation.start()

        self.is_hidden = True

    def show_panel(self):
        """Animate panel sliding back onto screen."""
        if not self.is_hidden or self.is_animating:
            return

        self.is_animating = True

        # Make sure we're visible before animating
        self.show()

        # Calculate positions
        start_rect = QRect(
            self.screen_geo.width(),  # Off screen
            0,
            PANEL_WIDTH,
            self.screen_geo.height()
        )
        end_rect = QRect(
            self.screen_geo.width() - PANEL_WIDTH,  # Visible position
            0,
            PANEL_WIDTH,
            self.screen_geo.height()
        )

        # Setup and start animation
        self.slide_animation.setStartValue(start_rect)
        self.slide_animation.setEndValue(end_rect)
        self.slide_animation.start()

        self.is_hidden = False

    def _on_animation_finished(self):
        """Called when slide animation completes."""
        self.is_animating = False

    def _check_cursor_position(self):
        """Check cursor position and show/hide panel accordingly."""
        if self.is_animating:
            return

        cursor_pos = QCursor.pos()
        right_edge = self.screen_geo.width()
        threshold = right_edge - RIGHT_SIDE_THRESHOLD

        # Cursor is on the right side of the screen
        if cursor_pos.x() >= threshold:
            self.auto_hide_timer.stop()
            if self.is_hidden:
                self.show_panel()
        else:
            # Cursor moved away from right side - start hide timer if not already started
            if not self.is_hidden and not self.auto_hide_timer.isActive():
                self.auto_hide_timer.start(AUTO_HIDE_DELAY)

    def _on_auto_hide_timeout(self):
        """Auto-hide the panel when timer expires."""
        if self.is_animating:
            return

        # Double-check cursor is still away from right side
        cursor_pos = QCursor.pos()
        right_edge = self.screen_geo.width()
        threshold = right_edge - RIGHT_SIDE_THRESHOLD

        if cursor_pos.x() < threshold and not self.is_hidden:
            self.hide_panel()

    def _handle_talk_click(self):
        toggle_recording()
        if is_recording:
            self.btn_talk.setText("Stop Listening")
        else:
            self.btn_talk.setText("Talk")

    def show_assist_ui(self):
        """Show the Claude Assistant dialog with voice input."""
        def pause_tracking():
            """Pause head tracking."""
            if self._is_pipeface_running() and not self.is_paused:
                try:
                    self.pipeface_process.suspend()
                    self.is_paused = True
                    self.btn_pause.setText("Resume")
                    print("[ProcessControlApp] Head tracking paused for CUA")
                except Exception as e:
                    print(f"[ProcessControlApp] Error pausing: {e}")

        def resume_tracking():
            """Resume head tracking."""
            if self._is_pipeface_running() and self.is_paused:
                try:
                    self.pipeface_process.resume()
                    self.is_paused = False
                    self.btn_pause.setText("Pause")
                    print("[ProcessControlApp] Head tracking resumed after CUA")
                except Exception as e:
                    print(f"[ProcessControlApp] Error resuming: {e}")

        assist_popup = AssistWidget(
            pause_callback=pause_tracking,
            resume_callback=resume_tracking
        )
        assist_popup.exec()

        if assist_popup.prompt_result:
            print(f"Prompt was: {assist_popup.prompt_result}")

    def on_assist_send(self):
        print("Assist widget closed")

    def _is_pipeface_running(self) -> bool:
        """Checks if the managed headtracker.py process is running."""
        if self.pipeface_process:
            try:
                return self.pipeface_process.is_running()
            except psutil.NoSuchProcess:
                self.pipeface_process = None
                return False
            except Exception as e:
                print(f"Error checking process status: {e}")
                return False
        return False

    def _start_pipeface(self) -> bool:
        """Starts or restarts the headtracker.py script using psutil."""
        if not os.path.exists(PIPEFACE_SCRIPT):
            print(f"Error: Script not found at {PIPEFACE_SCRIPT}")
            QMessageBox.warning(
                self, "Error", f"Script not found:\n{PIPEFACE_SCRIPT}")
            return False

        if self._is_pipeface_running():
            print("Terminating existing headtracker.py before restart.")
            self._terminate_process(self.pipeface_process)

        try:
            print(f"Starting script: {PIPEFACE_SCRIPT}")
            # Use creationflags on Windows if console window needs hiding (adjust if needed)
            kwargs = {}
            # if sys.platform == "win32": kwargs['creationflags'] = 0x08000000 # CREATE_NO_WINDOW

            process_popen = subprocess.Popen(
                [PYTHON_EXECUTABLE, PIPEFACE_SCRIPT], **kwargs)
            self.pipeface_process = psutil.Process(process_popen.pid)
            self.is_paused = False
            self.btn_pause.setText("Pause")
            print(
                f"Started headtracker.py with PID: {self.pipeface_process.pid}")
            return True
        except Exception as e:
            print(f"Error starting {PIPEFACE_SCRIPT}: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to start script:\n{PIPEFACE_SCRIPT}\n\n{e}")
            self.pipeface_process = None
            return False

    def _toggle_pause_resume(self):
        """Pauses or resumes the headtracker.py script via psutil."""
        if not self._is_pipeface_running():
            print("Cannot pause/resume: headtracker.py is not running.")
            return

        try:
            if self.is_paused:
                print(f"Resuming PID {self.pipeface_process.pid}...")
                self.pipeface_process.resume()
                self.btn_pause.setText("Pause")
                self.is_paused = False
            else:
                print(f"Pausing PID {self.pipeface_process.pid}...")
                self.pipeface_process.suspend()
                self.btn_pause.setText("Resume")
                self.is_paused = True
        except psutil.Error as e:
            print(
                f"Error pausing/resuming process {self.pipeface_process.pid}: {e}")
            QMessageBox.warning(
                self, "Error", f"Failed to pause/resume process:\n{e}")
            # Attempt to revert state on error
            self.is_paused = not self.is_paused
            self.btn_pause.setText("Pause" if not self.is_paused else "Resume")

    def _recall_pipeface(self):
        """Terminates and restarts the headtracker.py script."""
        print("Recall requested...")
        if self._is_pipeface_running():
            print("Terminating existing headtracker.py process...")
            if self._terminate_process(self.pipeface_process):
                time.sleep(0.5)  # Brief pause before restart
            else:
                print(
                    "Failed to terminate existing process cleanly, attempting restart anyway.")
        else:
            print("headtracker.py was not running. Starting it now.")

        self._start_pipeface()

    def _terminate_process(self, process: psutil.Process | None) -> bool:
        """Attempts to terminate a process gracefully (terminate -> wait -> kill)."""
        if not process:
            return False
        try:
            if process.is_running():
                print(f"Terminating process PID {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=2)
                    print(f"Process {process.pid} terminated.")
                    return True
                except psutil.TimeoutExpired:
                    print(f"Process {process.pid} kill required...")
                    process.kill()
                    process.wait(timeout=1)
                    print(f"Process {process.pid} killed.")
                    return True
        except psutil.NoSuchProcess:
            print(f"Process {process.pid} already terminated.")
            return True
        except psutil.Error as e:
            print(f"Error terminating process {process.pid}: {e}")
            return False
        finally:
            if process == self.pipeface_process:
                self.pipeface_process = None  # Clear handle if it was the managed one

    def _find_and_kill_processes(self, script_full_paths: list[str]):
        """Finds running python processes matching script paths and kills them."""
        killed_count = 0
        target_scripts = [os.path.normpath(p) for p in script_full_paths]
        print(f"Attempting to find and kill processes for: {target_scripts}")

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                # Check if it's a python process running one of the target scripts
                if cmdline and len(cmdline) > 1 and sys.executable in cmdline[0]:
                    script_path = os.path.normpath(cmdline[1])
                    if script_path in target_scripts:
                        print(
                            f"Found target process: PID={proc.pid}, Script={script_path}")
                        if self._terminate_process(psutil.Process(proc.pid)):
                            killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue  # Ignore processes that vanished or are inaccessible
            except Exception as e:
                # Log other unexpected errors
                print(f"Error iterating processes: {e}")
        print(f"Killed {killed_count} target process(es).")

    def _cleanup_ui_components(self):
        """Clean up timers."""
        if hasattr(self, 'cursor_poll_timer') and self.cursor_poll_timer:
            self.cursor_poll_timer.stop()
        if hasattr(self, 'auto_hide_timer') and self.auto_hide_timer:
            self.auto_hide_timer.stop()

    def _quit_all(self):
        """Terminates target scripts and closes the application."""
        print("Quit requested. Terminating target processes...")

        # Clean up UI components first
        self._cleanup_ui_components()

        # Terminate the specific instance started by this app
        self._terminate_process(self.pipeface_process)

        # Find and kill any other instances of target scripts
        # Tracker cleanup handled by _terminate_process above
        scripts_to_kill = []
        if os.path.exists(PIPEFACE_SCRIPT):
            scripts_to_kill.append(PIPEFACE_SCRIPT)
        if os.path.exists(PYGUI_SCRIPT):
            scripts_to_kill.append(PYGUI_SCRIPT)
        cv2.destroyAllWindows()

        if scripts_to_kill:
            self._find_and_kill_processes(scripts_to_kill)
        else:
            print("No target script paths found to search for.")

        print("Closing controller application.")
        self.close()  # Close this PyQt application

    def closeEvent(self, event):
        """Ensures cleanup when the window is closed via 'X'."""
        print("Close event triggered. Cleaning up...")

        # Clean up UI components
        self._cleanup_ui_components()

        # Run termination logic on window close as well
        self._terminate_process(self.pipeface_process)
        scripts_to_kill = []
        if os.path.exists(PIPEFACE_SCRIPT):
            scripts_to_kill.append(PIPEFACE_SCRIPT)
        if os.path.exists(PYGUI_SCRIPT):
            scripts_to_kill.append(PYGUI_SCRIPT)
        if scripts_to_kill:
            self._find_and_kill_processes(scripts_to_kill)
        # cv2.destroyAllWindows()

        event.accept()


if __name__ == '__main__':
    # Optional: Basic check if the primary controlled script exists
    if not os.path.exists(PIPEFACE_SCRIPT):
        print(
            f"WARNING: Script 'headtracker.py' not found at {PIPEFACE_SCRIPT}. Recall/Pause will fail.")
        # Consider exiting if critical:
        # QMessageBox.critical(None, "Startup Error", f"Required script 'headtracker.py' not found.\n{PIPEFACE_SCRIPT}")
        # sys.exit(1)

    app = QApplication(sys.argv)
    controller = ProcessControlApp()
    sys.exit(app.exec())
