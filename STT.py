"""
Speech-to-Text Module for AIVue
Supports multiple engines: faster-whisper (recommended), Vosk (fallback)
"""

import sounddevice as sd
import queue
import threading
import json
import pyautogui
import sys
import os
import time
import numpy as np
from pathlib import Path

# --- Configuration ---
# Engine options: "whisper-cpp", "whisper", "faster-whisper", "vosk"
# whisper-cpp is recommended - fast local inference without PyTorch dependency
STT_ENGINE = os.environ.get("STT_ENGINE", "whisper-cpp")

# Whisper settings (all whisper variants)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")  # cpu or cuda
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float32")  # int8, float16, float32 (faster-whisper only)

# Vosk settings (fallback)
BASE_DIR = Path(__file__).resolve().parent
VOSK_MODEL_PATH = BASE_DIR / "vosk-model-en-us-0.42-gigaspeech"

# Audio settings
SAMPLE_RATE = 16000
DEVICE_ID = 0
BLOCK_SIZE = 8000

# --- Global Variables ---
audio_queue = queue.Queue()
result_queue = queue.Queue(maxsize=100)
is_recording = False
is_processing = False
stream = None
model = None
recognizer = None  # For Vosk
processing_thread = None
typing_thread = None
current_engine = None


def _safe_queue_put(item):
    """Put item into result_queue with overflow protection."""
    try:
        _safe_queue_put(item, block=False)
    except queue.Full:
        try:
            result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _safe_queue_put(item, block=False)
        except queue.Full:
            pass


# --- Engine Loading ---

def load_whisper():
    """Load OpenAI whisper model"""
    global model, current_engine
    try:
        print("[STT] Attempting to import openai-whisper...")
        import whisper
        print(f"[STT] Loading whisper model '{WHISPER_MODEL}'...")
        model = whisper.load_model(WHISPER_MODEL)
        current_engine = "whisper"
        print(f"[STT] OpenAI whisper model loaded successfully!")
        return True
    except ImportError as e:
        print(f"[STT] openai-whisper not installed: {e}")
        return False
    except Exception as e:
        import traceback
        print(f"[STT] Error loading whisper model: {e}")
        traceback.print_exc()
        return False


def load_faster_whisper():
    """Load faster-whisper model"""
    global model, current_engine
    try:
        print("[STT] Attempting to import faster-whisper...")
        from faster_whisper import WhisperModel
        print(f"[STT] Loading faster-whisper model '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
        print(f"[STT] Compute type: {WHISPER_COMPUTE_TYPE}")
        model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        current_engine = "faster-whisper"
        print(f"[STT] faster-whisper model loaded successfully!")
        return True
    except ImportError as e:
        print(f"[STT] faster-whisper not installed: {e}")
        return False
    except Exception as e:
        import traceback
        print(f"[STT] Error loading faster-whisper model: {e}")
        traceback.print_exc()
        return False


def load_vosk_model():
    """Load Vosk model (fallback)"""
    global model, current_engine
    try:
        import vosk
        model_path_str = str(VOSK_MODEL_PATH)

        if not os.path.exists(model_path_str):
            print(f"ERROR: Vosk model folder not found at '{model_path_str}'")
            print("Please download a model from https://alphacephei.com/vosk/models")
            return False

        print(f"Loading Vosk model from: {model_path_str}")
        model = vosk.Model(model_path_str)
        current_engine = "vosk"
        print("Vosk model loaded successfully.")
        return True
    except ImportError:
        print("Vosk not installed. Install with: pip install vosk")
        return False
    except Exception as e:
        print(f"Error loading Vosk model: {e}")
        return False


def load_whisper_cpp():
    """Load whisper.cpp model (no PyTorch dependency)"""
    global model, current_engine
    try:
        print("[STT] Attempting to import pywhispercpp...")
        from pywhispercpp.model import Model as WhisperCppModel

        # Map model names to whisper.cpp model names
        model_map = {
            "tiny": "tiny",
            "base": "base",
            "small": "small",
            "medium": "medium",
            "large": "large-v3"
        }
        model_name = model_map.get(WHISPER_MODEL, "base")

        print(f"[STT] Loading whisper.cpp model '{model_name}'...")
        print("[STT] Note: First run will download the model (~150MB for base)")

        model = WhisperCppModel(model_name)
        current_engine = "whisper-cpp"
        print(f"[STT] whisper.cpp model loaded successfully!")
        return True
    except ImportError as e:
        print(f"[STT] pywhispercpp not installed: {e}")
        print("[STT] Install with: pip install pywhispercpp")
        return False
    except Exception as e:
        import traceback
        print(f"[STT] Error loading whisper.cpp model: {e}")
        traceback.print_exc()
        return False


def initialize_model():
    """Initialize the STT model based on configuration"""
    global model, current_engine

    print(f"[STT] Initializing STT with engine: {STT_ENGINE}")

    if STT_ENGINE == "whisper-cpp":
        if load_whisper_cpp():
            return True
        print("[STT] Falling back to faster-whisper...")
        if load_faster_whisper():
            return True
        print("[STT] Falling back to Vosk...")
        return load_vosk_model()
    elif STT_ENGINE == "whisper":
        if load_whisper():
            return True
        print("[STT] Falling back to whisper-cpp...")
        if load_whisper_cpp():
            return True
        print("[STT] Falling back to faster-whisper...")
        if load_faster_whisper():
            return True
        print("[STT] Falling back to Vosk...")
        return load_vosk_model()
    elif STT_ENGINE == "faster-whisper":
        if load_faster_whisper():
            return True
        print("[STT] Falling back to whisper-cpp...")
        if load_whisper_cpp():
            return True
        print("[STT] Falling back to Vosk...")
        return load_vosk_model()
    elif STT_ENGINE == "vosk":
        return load_vosk_model()
    else:
        print(f"[STT] Unknown STT engine: {STT_ENGINE}. Trying whisper-cpp...")
        if load_whisper_cpp():
            return True
        return load_vosk_model()


# --- Audio Callback ---

def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream - puts audio data in queue"""
    if status:
        print(status, file=sys.stderr)
    if is_recording:
        try:
            # Store as numpy array for faster-whisper, bytes for vosk
            audio_queue.put(indata.copy())
        except Exception as e:
            print(f"Error putting data into queue: {e}")


# --- OpenAI Whisper Processing ---

class OpenAIWhisperProcessor:
    """Processes audio using OpenAI whisper with chunked approach"""

    def __init__(self, model):
        self.model = model
        self.audio_buffer = []
        self.min_chunk_duration = 1.5  # Minimum seconds before processing
        self.max_chunk_duration = 15.0  # Maximum seconds before forced processing
        self.silence_threshold = 500  # Amplitude threshold for silence detection
        self.silence_duration = 0.8  # Seconds of silence to trigger processing

    def process_audio(self):
        """Main processing loop for OpenAI whisper"""
        global is_recording

        print("[STT] OpenAI whisper processing thread started.")
        self.audio_buffer = []
        last_speech_time = time.time()

        while is_recording:
            try:
                # Get audio data from queue
                data = audio_queue.get(timeout=0.1)
                if not is_recording:
                    break

                # Add to buffer
                self.audio_buffer.append(data)

                # Calculate buffer duration
                buffer_samples = sum(len(chunk) for chunk in self.audio_buffer)
                buffer_duration = buffer_samples / SAMPLE_RATE

                # Check for silence (simple VAD)
                amplitude = np.abs(data).mean()
                if amplitude > self.silence_threshold:
                    last_speech_time = time.time()

                silence_time = time.time() - last_speech_time

                # Process if we have enough audio and silence, or max duration reached
                should_process = (
                    (buffer_duration >= self.min_chunk_duration and silence_time >= self.silence_duration) or
                    buffer_duration >= self.max_chunk_duration
                )

                if should_process and self.audio_buffer:
                    self._transcribe_buffer()
                    self.audio_buffer = []
                    last_speech_time = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in whisper processing loop: {e}")
                break

        # Process any remaining audio
        if self.audio_buffer:
            self._transcribe_buffer()

        print("[STT] OpenAI whisper processing thread finished.")
        _safe_queue_put(None)

    def _transcribe_buffer(self):
        """Transcribe the accumulated audio buffer"""
        global is_processing
        if not self.audio_buffer:
            return

        try:
            is_processing = True
            # Concatenate audio chunks
            audio_data = np.concatenate(self.audio_buffer, axis=0)

            # Convert to float32 and normalize for whisper
            audio_float = audio_data.astype(np.float32).flatten() / 32768.0

            # OpenAI whisper expects audio at 16kHz
            # Transcribe using OpenAI whisper API
            result = self.model.transcribe(
                audio_float,
                language="en",
                fp16=False  # Use fp32 for CPU
            )

            full_text = result.get("text", "").strip()

            if full_text:
                # Filter common spurious outputs
                if full_text.lower() not in ["the", "a", "um", "uh", ".", "you"]:
                    _safe_queue_put(("final", full_text))
                    print(f"[STT] Transcribed: {full_text}")

        except Exception as e:
            import traceback
            print(f"[STT] Error transcribing audio: {e}")
            traceback.print_exc()
        finally:
            is_processing = False


# --- whisper.cpp Processing ---

class WhisperCppProcessor:
    """Processes audio using whisper.cpp (no PyTorch dependency)"""

    def __init__(self, model):
        self.model = model
        self.audio_buffer = []
        self.min_chunk_duration = 1.0  # Minimum seconds before processing
        self.max_chunk_duration = 10.0  # Maximum seconds before forced processing
        self.silence_threshold = 500  # Amplitude threshold for silence detection
        self.silence_duration = 0.5  # Seconds of silence to trigger processing

    def process_audio(self):
        """Main processing loop for whisper.cpp"""
        global is_recording

        print("[STT] whisper.cpp processing thread started.")
        self.audio_buffer = []
        last_speech_time = time.time()

        while is_recording:
            try:
                # Get audio data from queue
                data = audio_queue.get(timeout=0.1)
                if not is_recording:
                    break

                # Add to buffer
                self.audio_buffer.append(data)

                # Calculate buffer duration
                buffer_samples = sum(len(chunk) for chunk in self.audio_buffer)
                buffer_duration = buffer_samples / SAMPLE_RATE

                # Check for silence (simple VAD)
                amplitude = np.abs(data).mean()
                if amplitude > self.silence_threshold:
                    last_speech_time = time.time()

                silence_time = time.time() - last_speech_time

                # Process if we have enough audio and silence, or max duration reached
                should_process = (
                    (buffer_duration >= self.min_chunk_duration and silence_time >= self.silence_duration) or
                    buffer_duration >= self.max_chunk_duration
                )

                if should_process and self.audio_buffer:
                    self._transcribe_buffer()
                    self.audio_buffer = []
                    last_speech_time = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[STT] Error in whisper.cpp processing loop: {e}")
                break

        # Process any remaining audio
        if self.audio_buffer:
            self._transcribe_buffer()

        print("[STT] whisper.cpp processing thread finished.")
        _safe_queue_put(None)

    def _transcribe_buffer(self):
        """Transcribe the accumulated audio buffer"""
        global is_processing
        if not self.audio_buffer:
            return

        try:
            is_processing = True
            # Concatenate audio chunks
            audio_data = np.concatenate(self.audio_buffer, axis=0)

            # Convert to float32 and normalize for whisper
            audio_float = audio_data.astype(np.float32).flatten() / 32768.0

            # whisper.cpp transcribe expects float32 audio at 16kHz
            segments = self.model.transcribe(audio_float)

            # Collect results from segments
            full_text = ""
            for segment in segments:
                full_text += segment.text

            full_text = full_text.strip()

            if full_text:
                # Filter common spurious outputs
                spurious = ["the", "a", "um", "uh", ".", "you", "[music]", "(music)",
                           "[blank_audio]", "(blank_audio)", "blank_audio", "[silence]"]
                if full_text.lower() not in spurious and not full_text.startswith("[") :
                    _safe_queue_put(("final", full_text))
                    print(f"[STT] Transcribed: {full_text}")
                else:
                    print(f"[STT] (filtered: {full_text})")

        except Exception as e:
            import traceback
            print(f"[STT] Error transcribing audio: {e}")
            traceback.print_exc()
        finally:
            is_processing = False


# --- faster-whisper Processing ---

class FasterWhisperProcessor:
    """Processes audio using faster-whisper with chunked approach"""

    def __init__(self, model):
        self.model = model
        self.audio_buffer = []
        self.min_chunk_duration = 1.0  # Minimum seconds before processing
        self.max_chunk_duration = 10.0  # Maximum seconds before forced processing
        self.silence_threshold = 500  # Amplitude threshold for silence detection
        self.silence_duration = 0.5  # Seconds of silence to trigger processing

    def process_audio(self):
        """Main processing loop for faster-whisper"""
        global is_recording

        print("faster-whisper processing thread started.")
        self.audio_buffer = []
        last_speech_time = time.time()

        while is_recording:
            try:
                # Get audio data from queue
                data = audio_queue.get(timeout=0.1)
                if not is_recording:
                    break

                # Add to buffer
                self.audio_buffer.append(data)

                # Calculate buffer duration
                buffer_samples = sum(len(chunk) for chunk in self.audio_buffer)
                buffer_duration = buffer_samples / SAMPLE_RATE

                # Check for silence (simple VAD)
                amplitude = np.abs(data).mean()
                if amplitude > self.silence_threshold:
                    last_speech_time = time.time()

                silence_time = time.time() - last_speech_time

                # Process if we have enough audio and silence, or max duration reached
                should_process = (
                    (buffer_duration >= self.min_chunk_duration and silence_time >= self.silence_duration) or
                    buffer_duration >= self.max_chunk_duration
                )

                if should_process and self.audio_buffer:
                    self._transcribe_buffer()
                    self.audio_buffer = []
                    last_speech_time = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in faster-whisper processing loop: {e}")
                break

        # Process any remaining audio
        if self.audio_buffer:
            self._transcribe_buffer()

        print("faster-whisper processing thread finished.")
        _safe_queue_put(None)

    def _transcribe_buffer(self):
        """Transcribe the accumulated audio buffer"""
        global is_processing
        if not self.audio_buffer:
            return

        try:
            is_processing = True
            # Concatenate audio chunks
            audio_data = np.concatenate(self.audio_buffer, axis=0)

            # Convert to float32 and normalize for whisper
            audio_float = audio_data.astype(np.float32).flatten() / 32768.0

            # Transcribe
            segments, info = self.model.transcribe(
                audio_float,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )

            # Collect results
            full_text = ""
            for segment in segments:
                full_text += segment.text

            full_text = full_text.strip()

            if full_text:
                # Filter common spurious outputs
                if full_text.lower() not in ["the", "a", "um", "uh", "."]:
                    _safe_queue_put(("final", full_text))
                    print(f"Transcribed: {full_text}")

        except Exception as e:
            print(f"Error transcribing audio: {e}")
        finally:
            is_processing = False


# --- Vosk Processing ---

class VoskProcessor:
    """Processes audio using Vosk"""

    def __init__(self, model):
        self.model = model
        self.recognizer = None

    def process_audio(self):
        """Main processing loop for Vosk"""
        global is_recording
        import vosk

        self.recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(False)

        print("Vosk processing thread started.")

        while is_recording:
            try:
                data = audio_queue.get(timeout=0.1)
                if not is_recording or self.recognizer is None:
                    break

                # Convert numpy array to bytes for Vosk
                audio_bytes = data.tobytes()

                if self.recognizer.AcceptWaveform(audio_bytes):
                    result_json = self.recognizer.Result()
                    result_dict = json.loads(result_json)
                    final_text = result_dict.get('text', '').strip()
                    if final_text.lower() == "the":
                        print("Filtered spurious 'the'")
                    elif final_text:
                        _safe_queue_put(("final", final_text))
                        print(f"Final: {final_text}")
                else:
                    partial_json = self.recognizer.PartialResult()
                    partial_dict = json.loads(partial_json)
                    partial_text = partial_dict.get('partial', '')
                    if partial_text:
                        _safe_queue_put(("partial", partial_text))

            except queue.Empty:
                continue
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            except Exception as e:
                print(f"Error in Vosk processing loop: {e}")
                break

        # Process final chunk
        print("Processing final audio chunk...")
        try:
            if self.recognizer:
                final_result_json = self.recognizer.FinalResult()
                final_result_dict = json.loads(final_result_json)
                final_text = final_result_dict.get('text', '').strip()
                if final_text.lower() == "the":
                    print("Filtered final 'the'")
                elif final_text:
                    _safe_queue_put(("final", final_text))
                    print(f"Final (at end): {final_text}")
        except Exception as e:
            print(f"Error getting final result: {e}")

        print("Vosk processing thread finished.")
        _safe_queue_put(None)


# --- Typing Thread ---

def type_results():
    """Thread that types out transcribed text"""
    global is_recording
    print("Typing thread started.")
    current_segment_text = ""

    while True:
        try:
            item = result_queue.get()
            if item is None:
                if current_segment_text:
                    pyautogui.write(' ')
                    current_segment_text = ""
                break

            type_flag, text = item
            if not text and type_flag != "final":
                continue

            if type_flag == "partial":
                # Handle partial results (mainly for Vosk)
                if text.startswith(current_segment_text):
                    new_chars = text[len(current_segment_text):]
                    if new_chars:
                        pyautogui.write(new_chars, interval=0.01)
                        current_segment_text = text
                else:
                    if len(text) > len(current_segment_text):
                        new_chars = text[len(current_segment_text):]
                        pyautogui.write(new_chars, interval=0.01)
                        current_segment_text = text

            elif type_flag == "final":
                if text.startswith(current_segment_text):
                    new_chars = text[len(current_segment_text):]
                    if new_chars:
                        pyautogui.write(new_chars, interval=0.01)
                elif text:
                    pyautogui.write(text, interval=0.01)
                pyautogui.write(' ', interval=0.01)
                current_segment_text = ""

        except Exception as e:
            print(f"Error in typing thread: {e}")
            break

    print("Typing thread finished.")


# --- Toggle Recording ---

def toggle_recording():
    """Toggle recording on/off"""
    global is_recording, stream, processing_thread, typing_thread, model, current_engine

    if not is_recording:
        # Start recording
        if not model:
            print("ERROR: STT model not loaded. Cannot start recording.")
            return

        try:
            # Clear queues
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break

            is_recording = True

            # Check audio device
            try:
                sd.check_input_settings(
                    device=DEVICE_ID, samplerate=SAMPLE_RATE, channels=1)
            except Exception as e:
                print(f"Error checking input device settings: {e}")
                is_recording = False
                return

            # Start audio stream
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=DEVICE_ID,
                channels=1,
                dtype='int16',
                callback=audio_callback,
            )
            stream.start()
            print(f"Recording started with {current_engine}...")

            # Start processing thread based on engine
            if current_engine == "whisper":
                processor = OpenAIWhisperProcessor(model)
                processing_thread = threading.Thread(
                    target=processor.process_audio, daemon=True)
            elif current_engine == "whisper-cpp":
                processor = WhisperCppProcessor(model)
                processing_thread = threading.Thread(
                    target=processor.process_audio, daemon=True)
            elif current_engine == "faster-whisper":
                processor = FasterWhisperProcessor(model)
                processing_thread = threading.Thread(
                    target=processor.process_audio, daemon=True)
            else:  # vosk
                processor = VoskProcessor(model)
                processing_thread = threading.Thread(
                    target=processor.process_audio, daemon=True)

            typing_thread = threading.Thread(target=type_results, daemon=True)
            processing_thread.start()
            typing_thread.start()

        except Exception as e:
            print(f"Error starting recording: {e}")
            is_recording = False
            if stream and stream.active:
                try:
                    stream.stop()
                    stream.close()
                except Exception as e_close:
                    print(f"Error closing stream during start error: {e_close}")
            stream = None
    else:
        # Stop recording
        print("Stopping recording...")
        is_recording = False

        if stream:
            try:
                if stream.active:
                    stream.stop()
                stream.close()
            except Exception as e_close:
                print(f"Error stopping/closing stream: {e_close}")
            finally:
                stream = None

        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=2.0)
        if typing_thread and typing_thread.is_alive():
            typing_thread.join(timeout=1.5)

        print("Recording stopped.")


# --- Utility Functions ---

def get_current_engine():
    """Return the current STT engine name"""
    return current_engine


def set_engine(engine_name):
    """Switch STT engine (requires restart)"""
    global STT_ENGINE
    valid_engines = ["whisper-cpp", "whisper", "faster-whisper", "vosk"]
    if engine_name in valid_engines:
        os.environ["STT_ENGINE"] = engine_name
        STT_ENGINE = engine_name
        print(f"STT engine set to {engine_name}. Restart required to apply.")
    else:
        print(f"Unknown engine: {engine_name}. Options: {', '.join(valid_engines)}")


# --- Main Execution ---

def list_audio_devices():
    """List available audio input devices"""
    print("\n--- Available Audio Input Devices ---")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            marker = " <-- CURRENT" if i == DEVICE_ID else ""
            print(f"  [{i}] {dev['name']} (inputs: {dev['max_input_channels']}){marker}")
    print(f"\nCurrent DEVICE_ID: {DEVICE_ID}")
    print("To change, set DEVICE_ID in STT.py or use 'd <number>' command\n")


def main():
    """Main function for standalone testing"""
    global DEVICE_ID

    # Show audio devices first
    list_audio_devices()

    if initialize_model():
        print(f"STT is ready with {current_engine} engine.")
    else:
        print("Failed to load any STT model.")
        return

    print("Enter 'r' to toggle recording, 'e' to show engine, 'd' to list devices, or 'q' to quit.")

    while True:
        cmd = input("Command (r/e/d/q): ").strip().lower()
        if cmd == 'r':
            toggle_recording()
        elif cmd == 'e':
            print(f"Current engine: {current_engine}")
            print(f"Configured engine: {STT_ENGINE}")
        elif cmd == 'd':
            list_audio_devices()
        elif cmd.startswith('d '):
            try:
                new_id = int(cmd.split()[1])
                DEVICE_ID = new_id
                print(f"Device ID changed to {new_id}. Will use on next recording.")
            except (ValueError, IndexError):
                print("Usage: d <device_number>")
        elif cmd == 'q':
            if is_recording:
                toggle_recording()
            break
        else:
            print("Commands: r=record, e=engine, d=devices, d <num>=set device, q=quit")


if __name__ == "__main__":
    main()
