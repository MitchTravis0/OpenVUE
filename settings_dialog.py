"""
OpenVUE Settings Dialog

PyQt6 dialog for configuring tracking, STT, and application settings.
Reads/writes to settings.json via the config module.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from config import load_config, save_config, AppConfig


DARK_BG = "#16213e"
ACCENT = "#f8a5c2"
TEXT = "#ffffff"
TEXT_DIM = "#a0a0b0"


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenVUE Settings")
        self.setFixedSize(480, 620)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {DARK_BG}; }}
            QLabel {{ color: {TEXT}; font-size: 12px; }}
            QGroupBox {{
                color: {TEXT}; font-size: 13px; font-weight: bold;
                border: 1px solid #2a3a5e; border-radius: 6px;
                margin-top: 10px; padding-top: 14px;
            }}
            QGroupBox::title {{ subcontrol-position: top left; padding: 2px 8px; }}
            QSlider::groove:horizontal {{
                height: 6px; background: #2a3a5e; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 14px; height: 14px;
                margin: -4px 0; border-radius: 7px;
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: #1a2744; color: {TEXT}; border: 1px solid #2a3a5e;
                border-radius: 4px; padding: 4px 8px; font-size: 12px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QRadioButton {{ color: {TEXT}; font-size: 12px; }}
            QRadioButton::indicator {{ width: 14px; height: 14px; }}
            QPushButton {{
                background-color: {ACCENT}; color: white; border: none;
                border-radius: 8px; font-size: 14px; font-weight: 600;
                padding: 8px 24px;
            }}
            QPushButton:hover {{ background-color: #ff758f; }}
            QPushButton:pressed {{ background-color: #f55674; }}
        """)

        self.config = load_config()
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("Settings")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # --- Tracking Group ---
        tracking_group = QGroupBox("Tracking")
        tracking_form = QFormLayout()
        tracking_form.setSpacing(8)

        # Camera index
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 5)
        tracking_form.addRow("Camera Index:", self.camera_spin)

        # Dead zone
        self.deadzone_slider = QSlider(Qt.Orientation.Horizontal)
        self.deadzone_slider.setRange(0, 30)
        self.deadzone_label = QLabel()
        dz_row = QHBoxLayout()
        dz_row.addWidget(self.deadzone_slider)
        dz_row.addWidget(self.deadzone_label)
        self.deadzone_slider.valueChanged.connect(
            lambda v: self.deadzone_label.setText(f"{v} px"))
        tracking_form.addRow("Dead Zone:", dz_row)

        # Acceleration
        self.accel_slider = QSlider(Qt.Orientation.Horizontal)
        self.accel_slider.setRange(50, 150)  # 0.5 to 1.5, stored as x100
        self.accel_label = QLabel()
        accel_row = QHBoxLayout()
        accel_row.addWidget(self.accel_slider)
        accel_row.addWidget(self.accel_label)
        self.accel_slider.valueChanged.connect(
            lambda v: self.accel_label.setText(f"{v/100:.2f}"))
        tracking_form.addRow("Acceleration:", accel_row)

        # MediaPipe reset interval
        self.mp_reset_spin = QSpinBox()
        self.mp_reset_spin.setRange(1, 30)
        self.mp_reset_spin.setSuffix(" min")
        tracking_form.addRow("MediaPipe Reset:", self.mp_reset_spin)

        tracking_group.setLayout(tracking_form)
        layout.addWidget(tracking_group)

        # --- Click Mode Group ---
        click_group = QGroupBox("Click Mode")
        click_layout = QVBoxLayout()

        self.click_mode_group = QButtonGroup(self)
        self.wink_radio = QRadioButton("Wink (eye blink to click)")
        self.dwell_radio = QRadioButton("Dwell (hold cursor still to click)")
        self.click_mode_group.addButton(self.wink_radio, 0)
        self.click_mode_group.addButton(self.dwell_radio, 1)
        click_layout.addWidget(self.wink_radio)
        click_layout.addWidget(self.dwell_radio)

        # Dwell time
        self.dwell_slider = QSlider(Qt.Orientation.Horizontal)
        self.dwell_slider.setRange(3, 30)  # 0.3 to 3.0, stored as x10
        self.dwell_label = QLabel()
        dwell_row = QHBoxLayout()
        dwell_lbl = QLabel("Dwell Time:")
        dwell_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        dwell_row.addWidget(dwell_lbl)
        dwell_row.addWidget(self.dwell_slider)
        dwell_row.addWidget(self.dwell_label)
        self.dwell_slider.valueChanged.connect(
            lambda v: self.dwell_label.setText(f"{v/10:.1f}s"))
        click_layout.addLayout(dwell_row)

        # Dwell tolerance
        self.dwell_tol_slider = QSlider(Qt.Orientation.Horizontal)
        self.dwell_tol_slider.setRange(5, 40)
        self.dwell_tol_label = QLabel()
        tol_row = QHBoxLayout()
        tol_lbl = QLabel("Dwell Tolerance:")
        tol_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        tol_row.addWidget(tol_lbl)
        tol_row.addWidget(self.dwell_tol_slider)
        tol_row.addWidget(self.dwell_tol_label)
        self.dwell_tol_slider.valueChanged.connect(
            lambda v: self.dwell_tol_label.setText(f"{v} px"))
        click_layout.addLayout(tol_row)

        click_group.setLayout(click_layout)
        layout.addWidget(click_group)

        # --- STT Group ---
        stt_group = QGroupBox("Speech-to-Text")
        stt_form = QFormLayout()

        self.stt_engine_combo = QComboBox()
        self.stt_engine_combo.addItems(["whisper-cpp", "faster-whisper", "whisper", "vosk"])
        stt_form.addRow("Engine:", self.stt_engine_combo)

        self.stt_model_combo = QComboBox()
        self.stt_model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        stt_form.addRow("Model:", self.stt_model_combo)

        self.stt_device_combo = QComboBox()
        self.stt_device_combo.addItems(["cpu", "cuda"])
        stt_form.addRow("Device:", self.stt_device_combo)

        stt_group.setLayout(stt_form)
        layout.addWidget(stt_group)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: {TEXT};
                border: 1px solid #2a3a5e; border-radius: 8px;
                font-size: 14px; padding: 8px 24px;
            }}
            QPushButton:hover {{ background-color: #2a3a5e; }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _load_values(self):
        tc = self.config.tracking
        sc = self.config.stt

        self.camera_spin.setValue(tc.camera_index)
        self.deadzone_slider.setValue(tc.dead_zone_pixels)
        self.accel_slider.setValue(int(tc.acceleration_exponent * 100))
        self.mp_reset_spin.setValue(tc.mediapipe_reset_minutes)

        if tc.click_mode == "dwell":
            self.dwell_radio.setChecked(True)
        else:
            self.wink_radio.setChecked(True)
        self.dwell_slider.setValue(int(tc.dwell_time * 10))
        self.dwell_tol_slider.setValue(tc.dwell_tolerance)

        idx = self.stt_engine_combo.findText(sc.engine)
        if idx >= 0:
            self.stt_engine_combo.setCurrentIndex(idx)
        idx = self.stt_model_combo.findText(sc.model)
        if idx >= 0:
            self.stt_model_combo.setCurrentIndex(idx)
        idx = self.stt_device_combo.findText(sc.device)
        if idx >= 0:
            self.stt_device_combo.setCurrentIndex(idx)

    def _save(self):
        tc = self.config.tracking
        sc = self.config.stt

        tc.camera_index = self.camera_spin.value()
        tc.dead_zone_pixels = self.deadzone_slider.value()
        tc.acceleration_exponent = self.accel_slider.value() / 100.0
        tc.mediapipe_reset_minutes = self.mp_reset_spin.value()
        tc.click_mode = "dwell" if self.dwell_radio.isChecked() else "wink"
        tc.dwell_time = self.dwell_slider.value() / 10.0
        tc.dwell_tolerance = self.dwell_tol_slider.value()

        sc.engine = self.stt_engine_combo.currentText()
        sc.model = self.stt_model_combo.currentText()
        sc.device = self.stt_device_combo.currentText()

        save_config(self.config)
        print("Settings saved to settings.json")
        self.accept()
