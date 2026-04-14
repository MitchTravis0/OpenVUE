# -*- coding: utf-8 -*-
import subprocess
from tarfile import TruncatedHeaderError

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import os
import sys
import platform
import traceback
from logger import log_event

try:
    from OneEuroFilter import OneEuroFilter
except ImportError:
    print("Warning: OneEuroFilter not found. Install with: pip install OneEuroFilter")
    OneEuroFilter = None

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except ImportError:
    print("---------------------------------------------------------")
    print("Error: filterpy library not found or Q_discrete_white_noise not available.")
    print("Please install or update filterpy:")
    print("  pip install filterpy")
    print("---------------------------------------------------------")
    sys.exit(1)


class HeadGazeTracker:
    def __init__(self, config=None):
        print("Initializing HeadGazeTracker...")
        from config import TrackingConfig
        self.config = config or TrackingConfig()

        if platform.system() == "Darwin":
            print(
                "macOS detected. Ensure necessary permissions (Accessibility, Camera) are granted.")
        elif platform.system() == "Windows":
            print("Windows detected.")
        else:
            print(f"{platform.system()} detected.")

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.mediapipe_detection_confidence,
            min_tracking_confidence=self.config.mediapipe_tracking_confidence
        )

        # Eye landmarks for EAR
        self.LEFT_EYE_V = [159, 145]
        self.RIGHT_EYE_V = [386, 374]
        self.LEFT_EYE_H = [33, 133]
        self.RIGHT_EYE_H = [362, 263]

        # Iris landmarks for gaze estimation (MediaPipe refine_landmarks=True provides these)
        self.LEFT_IRIS = [468, 469, 470, 471, 472]   # center + 4 cardinal
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        self.LEFT_EYE_CORNERS = [33, 133]   # outer, inner
        self.RIGHT_EYE_CORNERS = [362, 263]
        self.gaze_enabled = True
        self.gaze_jump_threshold = 0.25     # normalized gaze shift to trigger jump
        self.gaze_smoothing = 0.7           # EMA factor for gaze position
        self._smooth_gaze_x = 0.5
        self._smooth_gaze_y = 0.5
        self._prev_gaze_region = None
        self._gaze_jump_cooldown = 0.5      # seconds between gaze jumps
        self._last_gaze_jump_time = 0

        # Get screen dimensions automatically
        try:
            self.screen_width, self.screen_height = pyautogui.size()
            print(
                f"Screen resolution: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            print(
                f"Warning: Failed to get screen size: {e}. Defaulting to 3024x1964.")
            self.screen_width, self.screen_height = 3024, 1964

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        # Calibration settings - 5-point calibration (corners + center)
        # This provides sufficient coverage for polynomial fitting while being faster
        self.calibration_points = [
            (0.5, 0.5),  # Center first (establishes neutral reference)
            (0.1, 0.1),  # Top-left
            (0.9, 0.1),  # Top-right
            (0.1, 0.9),  # Bottom-left
            (0.9, 0.9),  # Bottom-right
        ]
        self.calibration_data = []
        self.calibration_complete = False
        self.calibration_matrix = None
        self.reference_head_pose = None

        # Adaptive calibration settings
        self.adaptive_calibration_enabled = True
        self.adaptive_samples = []  # Stores (head_pose, screen_target) pairs for refinement
        self.adaptive_sample_count = 0
        self.max_adaptive_samples = 50  # Rolling buffer size
        self.adaptive_refinement_threshold = 10  # Refine after this many samples
        self.last_refinement_time = 0
        self.refinement_cooldown = 15.0  # Seconds between refinements
        self._use_simple_model = True  # Track which calibration model is in use

        # --- Kalman Filter Setup ---
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([[self.screen_width / 2],
                              [self.screen_height / 2],
                              [0.],
                              [0.]])
        self.kf.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])

        self.kf.R = np.diag([self.config.kalman_base_r, self.config.kalman_base_r])  # Adaptive R overridden each frame
        self.process_noise_std = 3.0

        self.kf.Q = np.eye(4)
        self.kf.P = np.eye(4) * 500.
        self.last_time = time.time()

        # --- Dead Zone ---
        self.dead_zone_pixels = self.config.dead_zone_pixels

        # --- Periodic MediaPipe Reset ---
        self.mediapipe_reset_interval = self.config.mediapipe_reset_minutes * 60 * 30  # at ~30fps

        # --- Face Lost Recovery ---
        self.face_was_lost = False

        # --- Camera Recovery ---
        self.consecutive_frame_failures = 0
        self.max_frame_failures = 30  # ~1 second at 30fps
        self.camera_index = None

        # --- Wink Detection ---
        self.EAR_THRESHOLD = 0.20
        self.WINK_CONSEC_FRAMES = 2
        self.CLICK_COOLDOWN = 0.5
        self.DOUBLE_CLICK_WINDOW = 0.4
        self.left_wink_counter = 0
        self.right_wink_counter = 0
        self.left_eye_closed = False
        self.right_eye_closed = False
        self.last_left_click_time = 0
        self.last_right_click_time = 0
        self.last_left_wink_detect_time = 0

        # --- solvePnP Head Pose Model ---
        self.face_model_3d = np.array([
            (0.0, 0.0, 0.0),           # Nose tip (landmark 1)
            (0.0, -330.0, -65.0),       # Chin (landmark 152)
            (-225.0, 170.0, -135.0),    # Left eye left corner (landmark 33)
            (225.0, 170.0, -135.0),     # Right eye right corner (landmark 263)
            (-150.0, -150.0, -125.0),   # Left mouth corner (landmark 61)
            (150.0, -150.0, -125.0),    # Right mouth corner (landmark 291)
        ], dtype=np.float64)
        self.face_landmark_indices = [1, 152, 33, 263, 61, 291]
        self._prev_rvec = None
        self._prev_tvec = None
        self._pnp_max_angle = 45.0  # Degrees for normalization to [-1, 1]

        # --- Dwell Clicking ---
        self.dwell_enabled = (self.config.click_mode == "dwell")
        self.dwell_time = self.config.dwell_time
        self.dwell_tolerance = self.config.dwell_tolerance
        self.dwell_start_time = None
        self.dwell_anchor = None

        # --- Session Statistics ---
        self.session_start_time = time.time()
        self.total_clicks = 0
        self.adaptive_refinement_count = 0
        self.face_detection_count = 0
        self.face_lost_count = 0

        # --- Head Gesture Scrolling ---
        self.scroll_enabled = True
        self.scroll_pitch_threshold = 0.6   # normalized pitch to trigger scroll
        self.scroll_sustain_time = 0.5      # seconds before scroll starts
        self.scroll_start_time = None
        self.scroll_direction = 0           # -1=up, 1=down, 0=none
        self.scroll_base_speed = 3          # scroll clicks per frame

        # --- Raw Pose Smoothing (One Euro Filter) ---
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        if OneEuroFilter is not None:
            self.yaw_filter = OneEuroFilter(freq=30.0, min_cutoff=self.config.one_euro_min_cutoff, beta=self.config.one_euro_beta, d_cutoff=1.0)
            self.pitch_filter = OneEuroFilter(freq=30.0, min_cutoff=self.config.one_euro_min_cutoff, beta=self.config.one_euro_beta, d_cutoff=1.0)
        else:
            self.yaw_filter = None
            self.pitch_filter = None
        self.first_pose = True

        print("Initialization complete.")

    def _calculate_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return 0.0
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _calculate_ear(self, landmarks, eye_v_indices, eye_h_indices):
        try:
            if not landmarks or max(eye_v_indices + eye_h_indices) >= len(landmarks):
                return 0.3
            p_top = landmarks[eye_v_indices[0]]
            p_bottom = landmarks[eye_v_indices[1]]
            p_left = landmarks[eye_h_indices[0]]
            p_right = landmarks[eye_h_indices[1]]
            v_dist = self._calculate_distance(p_top, p_bottom)
            h_dist = self._calculate_distance(p_left, p_right)
            if h_dist < 1e-6:
                return 0.3
            ear = v_dist / h_dist
            return ear
        except Exception:
            return 0.3

    def _detect_winks(self, landmarks, head_pose=None, screen_pos=None):
        """
        Detect winks for click actions and optionally collect adaptive calibration samples.

        Args:
            landmarks: Face landmarks from MediaPipe
            head_pose: Optional tuple (yaw, pitch) for adaptive calibration
            screen_pos: Optional tuple (x, y) screen position for adaptive calibration
        """
        current_time = time.time()
        left_ear = self._calculate_ear(
            landmarks, self.LEFT_EYE_V, self.LEFT_EYE_H)
        right_ear = self._calculate_ear(
            landmarks, self.RIGHT_EYE_V, self.RIGHT_EYE_H)
        ear_thresh_low = self.EAR_THRESHOLD - 0.02
        ear_thresh_high = self.EAR_THRESHOLD + 0.02
        self.left_eye_closed = left_ear < ear_thresh_low
        self.right_eye_closed = right_ear < ear_thresh_low

        if self.left_eye_closed and right_ear > ear_thresh_high:
            self.left_wink_counter += 1
        else:
            self.left_wink_counter = 0
        if (self.left_wink_counter == self.WINK_CONSEC_FRAMES and
                current_time - self.last_left_click_time > self.CLICK_COOLDOWN):
            time_since_last_detect = current_time - self.last_left_wink_detect_time
            if time_since_last_detect < self.DOUBLE_CLICK_WINDOW:
                print(
                    f"Double Left Click Triggered! (dT={time_since_last_detect:.2f}s)")
                try:
                    pyautogui.doubleClick(button='left')
                    self.total_clicks += 2
                except Exception as e:
                    print(f"PyAutoGUI Error: {e}")
                self.last_left_click_time = current_time
                self.last_left_wink_detect_time = 0
            else:
                print(
                    f"Left Click Triggered! (dT={time_since_last_detect:.2f}s)")
                try:
                    pyautogui.click(button='left')
                    self.total_clicks += 1
                    # Collect adaptive calibration sample on left click
                    if head_pose is not None and screen_pos is not None:
                        self.add_adaptive_sample(head_pose[0], head_pose[1],
                                                screen_pos[0], screen_pos[1])
                except Exception as e:
                    print(f"PyAutoGUI Error: {e}")
                self.last_left_click_time = current_time
                self.last_left_wink_detect_time = current_time
            self.left_wink_counter = 0
        elif self.left_wink_counter > self.WINK_CONSEC_FRAMES * 2:
            self.left_wink_counter = 0

        if self.right_eye_closed and left_ear > ear_thresh_high:
            self.right_wink_counter += 1
        else:
            self.right_wink_counter = 0
        if (self.right_wink_counter == self.WINK_CONSEC_FRAMES and
                current_time - self.last_right_click_time > self.CLICK_COOLDOWN):
            print("Right Click Triggered!")
            try:
                pyautogui.click(button='right')
                self.total_clicks += 1
            except Exception as e:
                print(f"PyAutoGUI Error: {e}")
            self.last_right_click_time = current_time
            self.right_wink_counter = 0
        elif self.right_wink_counter > self.WINK_CONSEC_FRAMES * 2:
            self.right_wink_counter = 0

        return left_ear, right_ear

    def _get_head_pose(self, landmarks, image_shape):
        """Main head pose method - delegates to solvePnP with geometric fallback."""
        return self._get_head_pose_pnp(landmarks, image_shape)

    def _get_head_pose_pnp(self, landmarks, image_shape):
        """Head pose estimation using cv2.solvePnP for robust 3D pose recovery."""
        h, w = image_shape[:2]
        if not landmarks or not all(idx < len(landmarks) for idx in self.face_landmark_indices):
            return self._get_head_pose_geometric(landmarks, image_shape)
        try:
            # Extract 2D image points for the 6 landmarks
            image_points = np.array([
                (landmarks[idx].x * w, landmarks[idx].y * h)
                for idx in self.face_landmark_indices
            ], dtype=np.float64)

            # Camera matrix approximation using frame width as focal length
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            # Use previous vectors as initial guess for temporal stability
            if self._prev_rvec is not None and self._prev_tvec is not None:
                success, rvec, tvec = cv2.solvePnP(
                    self.face_model_3d, image_points, camera_matrix, dist_coeffs,
                    rvec=self._prev_rvec.copy(), tvec=self._prev_tvec.copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                success, rvec, tvec = cv2.solvePnP(
                    self.face_model_3d, image_points, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            if not success:
                return self._get_head_pose_geometric(landmarks, image_shape)

            self._prev_rvec = rvec
            self._prev_tvec = tvec

            # Convert rotation vector to rotation matrix and extract Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                np.hstack((rotation_matrix, tvec))
            )

            # euler_angles: [pitch, yaw, roll] in degrees
            pitch_deg = euler_angles[0, 0]
            yaw_deg = euler_angles[1, 0]

            # Normalize to approximately [-1, 1] using max angle constant
            head_yaw = np.clip(yaw_deg / self._pnp_max_angle, -1.0, 1.0)
            head_pitch = np.clip(pitch_deg / self._pnp_max_angle, -1.0, 1.0)

            return self._apply_pose_smoothing(head_yaw, head_pitch)
        except Exception as e:
            return self._get_head_pose_geometric(landmarks, image_shape)

    def _get_head_pose_geometric(self, landmarks, image_shape):
        """Fallback geometric head pose estimation using nose-to-eye ratios."""
        h, w = image_shape[:2]
        required_indices = [1, 152, 133, 362, 10, 33, 263]
        if not landmarks or not all(idx < len(landmarks) for idx in required_indices):
            ref_pose = self.reference_head_pose if self.reference_head_pose else (
                0.0, 0.0)
            return ref_pose[0], ref_pose[1]
        try:
            nose_tip = landmarks[1]
            chin = landmarks[152]
            left_eye_inner = landmarks[133]
            right_eye_inner = landmarks[362]
            forehead = landmarks[10]

            nose_x = nose_tip.x * w
            left_eye_x = left_eye_inner.x * w
            right_eye_x = right_eye_inner.x * w
            eye_center_x = (left_eye_x + right_eye_x) / 2.0
            interocular_dist = abs(right_eye_x - left_eye_x)
            if interocular_dist < 10:
                interocular_dist = 10
            nose_offset_x = nose_x - eye_center_x
            head_yaw = (nose_offset_x / interocular_dist) * 1.3
            head_yaw = np.clip(head_yaw, -1.0, 1.0)

            nose_y = nose_tip.y * h
            chin_y = chin.y * h
            forehead_y = forehead.y * h
            eye_center_y = ((left_eye_inner.y + right_eye_inner.y) / 2.0) * h
            face_height = abs(chin_y - forehead_y)
            if face_height < 20:
                face_height = 20
            nose_offset_y = (nose_y - eye_center_y) / face_height
            neutral_pitch_offset = 0.1
            head_pitch = (nose_offset_y - neutral_pitch_offset) * 2.2
            head_pitch = np.clip(head_pitch, -0.8, 0.8)

            return self._apply_pose_smoothing(head_yaw, head_pitch)
        except Exception as e:
            ref_pose = self.reference_head_pose if self.reference_head_pose else (
                0.0, 0.0)
            return (self.smooth_yaw if not self.first_pose else ref_pose[0],
                    self.smooth_pitch if not self.first_pose else ref_pose[1])

    def _apply_pose_smoothing(self, head_yaw, head_pitch):
        """Apply One Euro Filter (or EMA fallback) smoothing to raw pose values."""
        if self.first_pose:
            if OneEuroFilter is not None:
                self.yaw_filter = OneEuroFilter(freq=30.0, min_cutoff=self.config.one_euro_min_cutoff, beta=self.config.one_euro_beta, d_cutoff=1.0)
                self.pitch_filter = OneEuroFilter(freq=30.0, min_cutoff=self.config.one_euro_min_cutoff, beta=self.config.one_euro_beta, d_cutoff=1.0)
            self._prev_rvec = None
            self._prev_tvec = None
            self.first_pose = False

        if self.yaw_filter is not None and self.pitch_filter is not None:
            self.smooth_yaw = self.yaw_filter(head_yaw)
            self.smooth_pitch = self.pitch_filter(head_pitch)
        else:
            alpha = 0.4
            self.smooth_yaw = alpha * self.smooth_yaw + (1 - alpha) * head_yaw
            self.smooth_pitch = alpha * self.smooth_pitch + (1 - alpha) * head_pitch

        return self.smooth_yaw, self.smooth_pitch

    def _estimate_gaze(self, landmarks, image_shape):
        """
        Estimate gaze direction using iris position relative to eye corners.
        Returns (gaze_x, gaze_y) normalized to [0, 1] where 0.5 is center.
        """
        h, w = image_shape[:2]
        try:
            max_idx = max(max(self.LEFT_IRIS), max(self.RIGHT_IRIS))
            if len(landmarks) <= max_idx:
                return None

            # Left eye: iris center relative to eye corners
            l_iris = landmarks[self.LEFT_IRIS[0]]  # center
            l_outer = landmarks[self.LEFT_EYE_CORNERS[0]]
            l_inner = landmarks[self.LEFT_EYE_CORNERS[1]]
            l_eye_width = abs(l_inner.x - l_outer.x)
            if l_eye_width < 1e-6:
                return None
            l_gaze_x = (l_iris.x - l_outer.x) / l_eye_width

            # Right eye: iris center relative to eye corners
            r_iris = landmarks[self.RIGHT_IRIS[0]]
            r_outer = landmarks[self.RIGHT_EYE_CORNERS[0]]
            r_inner = landmarks[self.RIGHT_EYE_CORNERS[1]]
            r_eye_width = abs(r_inner.x - r_outer.x)
            if r_eye_width < 1e-6:
                return None
            r_gaze_x = (r_iris.x - r_outer.x) / r_eye_width

            # Average both eyes for horizontal gaze
            gaze_x = (l_gaze_x + r_gaze_x) / 2.0

            # Vertical: iris center relative to eye top/bottom
            l_top = landmarks[self.LEFT_EYE_V[0]]
            l_bottom = landmarks[self.LEFT_EYE_V[1]]
            l_eye_height = abs(l_bottom.y - l_top.y)
            r_top = landmarks[self.RIGHT_EYE_V[0]]
            r_bottom = landmarks[self.RIGHT_EYE_V[1]]
            r_eye_height = abs(r_bottom.y - r_top.y)
            if l_eye_height < 1e-6 or r_eye_height < 1e-6:
                gaze_y = 0.5
            else:
                l_gaze_y = (l_iris.y - l_top.y) / l_eye_height
                r_gaze_y = (r_iris.y - r_top.y) / r_eye_height
                gaze_y = (l_gaze_y + r_gaze_y) / 2.0

            # Smooth the gaze
            self._smooth_gaze_x = self.gaze_smoothing * self._smooth_gaze_x + (1 - self.gaze_smoothing) * gaze_x
            self._smooth_gaze_y = self.gaze_smoothing * self._smooth_gaze_y + (1 - self.gaze_smoothing) * gaze_y

            return (self._smooth_gaze_x, self._smooth_gaze_y)
        except Exception:
            return None

    def _gaze_to_screen_region(self, gaze_x, gaze_y):
        """Map gaze position to a screen region (3x3 grid)."""
        col = 0 if gaze_x < 0.35 else (2 if gaze_x > 0.65 else 1)
        row = 0 if gaze_y < 0.35 else (2 if gaze_y > 0.65 else 1)
        return (row, col)

    def _region_to_screen_center(self, region):
        """Convert a (row, col) region to screen center coordinates."""
        row, col = region
        x = (col + 0.5) / 3.0 * self.screen_width
        y = (row + 0.5) / 3.0 * self.screen_height
        return x, y

    def custom_calibration(self):
        """
        This is your custom calibration function.
        Replace this with your own calibration work.
        For demonstration, it simply prints a message and sets a dummy reference pose.
        """
        print("Running custom calibration...")
        # Simulate calibration work here (e.g., additional processing, logging, etc.)
        # For example, set a dummy reference head pose.
        self.reference_head_pose = (0.0, 0.0)
        # You might update self.calibration_data here as needed.
        print("Custom calibration completed. Reference head pose set to:",
              self.reference_head_pose)
        return True

    def _show_transition_animation(self, calib_win_name, actual_w, actual_h,
                                      from_pos, to_pos, duration=0.3):
        """
        Show smooth transition animation between calibration points.

        Args:
            calib_win_name: Window name for display
            actual_w, actual_h: Window dimensions
            from_pos: (x, y) starting position tuple
            to_pos: (x, y) ending position tuple
            duration: Animation duration in seconds
        """
        steps = int(duration * 30)  # 30 fps animation
        if steps < 1:
            steps = 1

        for step in range(steps + 1):
            t = step / steps
            # Ease-in-out interpolation
            t = t * t * (3 - 2 * t)

            current_x = int(from_pos[0] + (to_pos[0] - from_pos[0]) * t)
            current_y = int(from_pos[1] + (to_pos[1] - from_pos[1]) * t)

            # Draw transition frame
            trans_img = np.full((actual_h, actual_w, 3), (235, 235, 235), dtype=np.uint8)
            # Outer ring (shrinks as we approach target)
            ring_size = int(30 + 20 * (1 - t))
            cv2.circle(trans_img, (current_x, current_y), ring_size, (200, 200, 200), 3)
            cv2.circle(trans_img, (current_x, current_y), 15, (180, 180, 180), -1)
            cv2.imshow(calib_win_name, trans_img)
            cv2.waitKey(int(1000 / 30))

    def _draw_calibration_point(self, img, target_x, target_y, point_num, total_points,
                                 progress_fraction, remaining_time, is_neutral=False):
        """
        Draw a calibration point with consistent styling.

        Args:
            img: Image to draw on
            target_x, target_y: Point coordinates
            point_num: Current point number (1-indexed)
            total_points: Total number of points
            progress_fraction: Progress (0-1) for the progress bar
            remaining_time: Time remaining in seconds
            is_neutral: Whether this is the neutral pose calibration
        """
        actual_h, actual_w = img.shape[:2]

        # Draw main target circle
        cv2.circle(img, (target_x, target_y), 30, (238, 0, 255), 6)
        cv2.circle(img, (target_x, target_y), 15, (255, 0, 0), -1)
        cv2.circle(img, (target_x, target_y), 5, (255, 255, 255), -1)  # Center dot

        # Label
        if is_neutral:
            label = "Look at center - establishing neutral position"
        else:
            label = f"Point {point_num} of {total_points}"

        # Position label above the point
        label_x = max(10, min(target_x - 100, actual_w - 250))
        label_y = target_y - 50 if target_y > 80 else target_y + 100
        cv2.putText(img, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Progress bar below target
        bar_width = 120
        bar_height = 12
        bar_x = max(10, min(target_x - bar_width // 2, actual_w - bar_width - 10))
        bar_y = target_y + 60 if target_y < actual_h - 100 else target_y - 80
        filled_width = int(bar_width * progress_fraction)

        # Progress bar background
        cv2.rectangle(img, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
        # Progress bar fill
        cv2.rectangle(img, (bar_x, bar_y),
                     (bar_x + filled_width, bar_y + bar_height), (0, 200, 0), -1)
        # Progress bar border
        cv2.rectangle(img, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 1)

        # Timer text
        timer_text = f"{remaining_time:.1f}s"
        cv2.putText(img, timer_text, (bar_x + bar_width + 10, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Instructions at bottom of screen
        instruction = "Keep your head still and look at the target"
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = (actual_w - text_size[0]) // 2
        cv2.putText(img, instruction, (text_x, actual_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

        # Overall progress indicator at top
        if not is_neutral:
            overall_progress = f"Calibration: {point_num}/{total_points}"
            cv2.putText(img, overall_progress, (actual_w - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    def run_calibration(self, cap):
        """
        Run the 5-point calibration process.

        Improvements over 9-point calibration:
        - Faster: 5 points instead of 9 (~40% time reduction)
        - Smoother: Animated transitions between points
        - Clearer: Better visual feedback and instructions
        - Adaptive: Calibration improves over time with use
        """
        baseline_pose = None
        move_threshold = 0.15  # TUNABLE
        print("\n" + "="*50)
        print("  QUICK CALIBRATION (5-point)")
        print("="*50)
        print("This will take about 15 seconds.")
        print("Move your head to follow the target dot.")
        print("Press 'Q' at any time to cancel.")
        print("="*50)

        self.calibration_data = []
        self.reference_head_pose = None
        self.calibration_complete = False
        self.reset_adaptive_calibration()  # Clear adaptive data on recalibration

        # Call your custom calibration work first
        if not self.custom_calibration():
            print("Custom calibration failed. Exiting calibration process.")
            return False
        else:
            print("Custom calibration succeeded, proceeding with full calibration.")

        # Create calibration window and force it to cover the entire screen
        calib_win_name = "Calibration Target"
        feed_win_name = "Camera Feed (Calibration)"
        cv2.namedWindow(calib_win_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(calib_win_name, 0, 0)
        cv2.resizeWindow(calib_win_name, self.screen_width, self.screen_height)
        cv2.setWindowProperty(
            calib_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(50)
        x, y, actual_w, actual_h = cv2.getWindowImageRect(calib_win_name)
        if actual_w <= 0 or actual_h <= 0:
            actual_w, actual_h = self.screen_width, self.screen_height
        print(f"Calibration window: {actual_w}x{actual_h}")

        # Create feed window (smaller preview)
        feed_w = self.screen_width // 8
        feed_h = self.screen_height // 8
        cv2.namedWindow(feed_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(feed_win_name, feed_w, feed_h)
        cv2.moveWindow(feed_win_name, 30, 30)

        # --- Welcome screen ---
        welcome_img = np.full((actual_h, actual_w, 3), (235, 235, 235), dtype=np.uint8)
        welcome_texts = [
            "Quick 5-Point Calibration",
            "",
            "1. Look at the center target first",
            "2. Then follow the dot to each corner",
            "3. Keep your head still at each point",
            "",
            "Starting in 2 seconds..."
        ]
        y_offset = actual_h // 3
        for text in welcome_texts:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (actual_w - text_size[0]) // 2
            cv2.putText(welcome_img, text, (text_x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
            y_offset += 45
        cv2.imshow(calib_win_name, welcome_img)
        cv2.waitKey(2000)

        # --- Neutral Pose Calibration (center point) ---
        print("\nStep 1/2: Neutral Pose (center)")
        neutral_data_raw = []
        center_x, center_y = actual_w // 2, actual_h // 2
        collection_duration = 2.5  # Reduced from 3.0 seconds
        min_neutral_samples = 12  # Reduced from 15
        start_time = time.time()
        self.first_pose = True
        baseline_pose = None

        while time.time() - start_time < collection_duration:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame.flags.writeable = True

            progress_fraction = min(1.0, (time.time() - start_time) / collection_duration)
            remaining_time = collection_duration - (time.time() - start_time)

            calib_img = np.full((actual_h, actual_w, 3), (235, 235, 235), dtype=np.uint8)
            self._draw_calibration_point(calib_img, center_x, center_y, 0,
                                        len(self.calibration_points),
                                        progress_fraction, remaining_time, is_neutral=True)
            cv2.imshow(calib_win_name, calib_img)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                try:
                    head_yaw, head_pitch = self._get_head_pose(landmarks, frame.shape)
                    if baseline_pose is None:
                        baseline_pose = (head_yaw, head_pitch)
                    else:
                        if (abs(head_yaw - baseline_pose[0]) > move_threshold or
                            abs(head_pitch - baseline_pose[1]) > move_threshold):
                            start_time = time.time()
                            neutral_data_raw = []
                            baseline_pose = (head_yaw, head_pitch)
                            # Show "moved too much" feedback
                            cv2.putText(calib_img, "Hold still - resetting...",
                                       (actual_w // 2 - 100, actual_h - 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
                            cv2.imshow(calib_win_name, calib_img)
                    if not self.first_pose:
                        neutral_data_raw.append((head_yaw, head_pitch))
                    self._draw_landmarks(frame, landmarks)
                except Exception:
                    pass
            cv2.imshow(feed_win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Calibration cancelled.")
                cv2.destroyAllWindows()
                return False

        if len(neutral_data_raw) >= min_neutral_samples:
            avg_yaw = np.mean([d[0] for d in neutral_data_raw])
            avg_pitch = np.mean([d[1] for d in neutral_data_raw])
            self.reference_head_pose = (avg_yaw, avg_pitch)
            print(f"  Neutral pose captured: yaw={avg_yaw:.3f}, pitch={avg_pitch:.3f}")
        else:
            print(f"  Failed: Only {len(neutral_data_raw)}/{min_neutral_samples} samples. Try again.")
            cv2.destroyAllWindows()
            return False

        # --- Calibration Points (corners) ---
        print("\nStep 2/2: Corner Points")
        collection_duration_point = 1.2  # Reduced from 1.5 seconds
        min_point_samples = 4  # Reduced from 5
        last_pos = (center_x, center_y)

        for i, (x_ratio, y_ratio) in enumerate(self.calibration_points):
            target_x = int(x_ratio * actual_w)
            target_y = int(y_ratio * actual_h)

            # Show smooth transition to next point
            self._show_transition_animation(calib_win_name, actual_w, actual_h,
                                           last_pos, (target_x, target_y), duration=0.25)
            last_pos = (target_x, target_y)

            print(f"  Point {i+1}/{len(self.calibration_points)} at ({x_ratio:.1f}, {y_ratio:.1f})")
            point_data_raw = []
            baseline_pose = None
            start_time = time.time()
            self.first_pose = True

            while time.time() - start_time < collection_duration_point:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.face_mesh.process(frame_rgb)
                frame.flags.writeable = True

                progress_fraction = min(1.0, (time.time() - start_time) / collection_duration_point)
                remaining_time = collection_duration_point - (time.time() - start_time)

                point_img = np.full((actual_h, actual_w, 3), (235, 235, 235), dtype=np.uint8)
                self._draw_calibration_point(point_img, target_x, target_y, i + 1,
                                            len(self.calibration_points),
                                            progress_fraction, remaining_time)
                cv2.imshow(calib_win_name, point_img)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    try:
                        head_yaw, head_pitch = self._get_head_pose(landmarks, frame.shape)
                        if baseline_pose is None:
                            baseline_pose = (head_yaw, head_pitch)
                        else:
                            if (abs(head_yaw - baseline_pose[0]) > move_threshold or
                                abs(head_pitch - baseline_pose[1]) > move_threshold):
                                start_time = time.time()
                                point_data_raw = []
                                baseline_pose = (head_yaw, head_pitch)
                        if not self.first_pose:
                            point_data_raw.append((head_yaw, head_pitch))
                        self._draw_landmarks(frame, landmarks)
                    except Exception:
                        pass
                cv2.putText(frame, f"Point {i+1}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(feed_win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Calibration cancelled.")
                    cv2.destroyAllWindows()
                    return False

            if len(point_data_raw) >= min_point_samples:
                avg_yaw = np.mean([p[0] for p in point_data_raw])
                avg_pitch = np.mean([p[1] for p in point_data_raw])
                self.calibration_data.append((avg_yaw, avg_pitch, x_ratio, y_ratio))
                print(f"    Recorded: yaw={avg_yaw:.3f}, pitch={avg_pitch:.3f}")
            else:
                print(f"    Skipped: insufficient samples ({len(point_data_raw)}/{min_point_samples})")

        cv2.destroyWindow(calib_win_name)
        cv2.destroyWindow(feed_win_name)

        # --- Compute calibration matrix ---
        print("\n--- Computing Calibration ---")
        # For 5-point calibration, we need at least 4 valid points
        min_required_points = 4
        if len(self.calibration_data) >= min_required_points:
            if self._compute_calibration_matrix():
                # Show success message
                success_img = np.full((self.screen_height, self.screen_width, 3),
                                     (235, 235, 235), dtype=np.uint8)
                success_texts = [
                    "Calibration Complete!",
                    "",
                    f"Captured {len(self.calibration_data)} points",
                    "Calibration will improve as you use the system",
                    "",
                    "Starting tracking..."
                ]
                y_offset = self.screen_height // 3
                for text in success_texts:
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = (self.screen_width - text_size[0]) // 2
                    color = (0, 150, 0) if "Complete" in text else (50, 50, 50)
                    cv2.putText(success_img, text, (text_x, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 45
                cv2.namedWindow(calib_win_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(calib_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(calib_win_name, success_img)
                cv2.waitKey(1500)
                cv2.destroyWindow(calib_win_name)

                print("Calibration successful!")
                self.calibration_complete = True
                self.kf.x = np.array([[self.screen_width / 2],
                                      [self.screen_height / 2],
                                      [0.],
                                      [0.]])
                self.kf.P = np.eye(4) * 500.
                self.last_time = time.time()
                self.left_wink_counter, self.right_wink_counter = 0, 0
                self.last_left_click_time, self.last_right_click_time = 0, 0
                self.last_left_wink_detect_time = 0
                self.first_pose = True
                return True
            else:
                print("Calibration matrix computation failed.")
                self.calibration_complete = False
                return False
        else:
            print(f"Calibration failed: Only {len(self.calibration_data)} points captured (need {min_required_points}+).")
            self.calibration_complete = False
            return False

    def _compute_calibration_matrix(self, calibration_data=None, sample_weights=None):
        """
        Compute calibration matrix from calibration data.
        Uses polynomial fitting: screen_pos = f(yaw, pitch, yaw^2, pitch^2, yaw*pitch, 1)

        With 5 points, we use a simplified linear model (fewer polynomial terms)
        to avoid overfitting while maintaining accuracy.

        Args:
            calibration_data: Optional override data (default: self.calibration_data)
            sample_weights: Optional per-sample weights for weighted least squares
        """
        data = calibration_data if calibration_data is not None else self.calibration_data
        if not data or self.reference_head_pose is None:
            return False

        ref_yaw, ref_pitch = self.reference_head_pose
        rel_yaw = np.array([p[0] - ref_yaw for p in data])
        rel_pitch = np.array([p[1] - ref_pitch for p in data])
        target_x_ratio = np.array([p[2] for p in data])
        target_y_ratio = np.array([p[3] for p in data])

        # For 5-point calibration, use a simpler model to avoid overfitting
        # Linear model with cross term: ax + by + cxy + d
        if len(data) <= 5:
            feature_matrix = np.column_stack(
                [rel_yaw, rel_pitch, rel_yaw * rel_pitch, np.ones(len(rel_yaw))])
        else:
            # Full polynomial model for more points (including adaptive refinement)
            feature_matrix = np.column_stack(
                [rel_yaw, rel_pitch, rel_yaw**2, rel_pitch**2, rel_yaw * rel_pitch, np.ones(len(rel_yaw))])

        try:
            if sample_weights is not None:
                # Weighted least squares: W^(1/2) * A * x = W^(1/2) * b
                W = np.diag(np.sqrt(sample_weights))
                Aw = W @ feature_matrix
                bx_w = W @ target_x_ratio
                by_w = W @ target_y_ratio
                x_coeffs, _, _, _ = np.linalg.lstsq(Aw, bx_w, rcond=None)
                y_coeffs, _, _, _ = np.linalg.lstsq(Aw, by_w, rcond=None)
            else:
                x_coeffs, _, _, _ = np.linalg.lstsq(
                    feature_matrix, target_x_ratio, rcond=None)
                y_coeffs, _, _, _ = np.linalg.lstsq(
                    feature_matrix, target_y_ratio, rcond=None)
            self.calibration_matrix = (x_coeffs, y_coeffs)
            self._use_simple_model = len(data) <= 5
            print(f"Calibration matrix computed ({'simple' if self._use_simple_model else 'full'} model, {len(data)} points).")
            return True
        except Exception as e:
            print(f"Error computing calibration matrix: {e}")
            self.calibration_matrix = None
            return False

    def add_adaptive_sample(self, head_yaw, head_pitch, click_x, click_y):
        """
        Add a sample for adaptive calibration refinement.
        Called when user clicks, assuming the gaze was targeting the click location.

        Args:
            head_yaw, head_pitch: Current head pose
            click_x, click_y: Screen coordinates where click occurred
        """
        if not self.adaptive_calibration_enabled or not self.calibration_complete:
            return

        # Convert screen coords to ratios
        x_ratio = click_x / self.screen_width
        y_ratio = click_y / self.screen_height

        # Store sample
        sample = (head_yaw, head_pitch, x_ratio, y_ratio)
        self.adaptive_samples.append(sample)
        self.adaptive_sample_count += 1

        # Maintain rolling buffer
        if len(self.adaptive_samples) > self.max_adaptive_samples:
            self.adaptive_samples.pop(0)

        # Check if we should refine the calibration
        current_time = time.time()
        if (self.adaptive_sample_count >= self.adaptive_refinement_threshold and
            current_time - self.last_refinement_time > self.refinement_cooldown):
            self._refine_calibration()
            self.last_refinement_time = current_time
            self.adaptive_sample_count = 0

    def _refine_calibration(self):
        """
        Refine calibration using adaptive samples combined with original calibration data.
        Applies outlier rejection (z-score > 2.0) and exponential decay weighting.
        """
        if len(self.adaptive_samples) < 5:
            return

        # Outlier rejection: compute residual errors against current calibration
        errors = []
        for sample in self.adaptive_samples:
            predicted_x, predicted_y = self.map_head_to_screen(sample[0], sample[1])
            actual_x = sample[2] * self.screen_width
            actual_y = sample[3] * self.screen_height
            error = math.sqrt((predicted_x - actual_x)**2 + (predicted_y - actual_y)**2)
            errors.append(error)

        mean_err = np.mean(errors)
        std_err = np.std(errors)

        if std_err > 1e-6:
            valid_samples = [s for s, e in zip(self.adaptive_samples, errors)
                             if abs(e - mean_err) / std_err < 2.0]
        else:
            valid_samples = list(self.adaptive_samples)

        if len(valid_samples) < 3:
            print(f"Adaptive refinement: too few valid samples ({len(valid_samples)}/{len(self.adaptive_samples)})")
            return

        rejected = len(self.adaptive_samples) - len(valid_samples)
        if rejected > 0:
            print(f"Adaptive refinement: rejected {rejected} outlier(s)")

        # Build combined data with original calibration points duplicated for higher weight
        combined_data = []
        for point in self.calibration_data:
            combined_data.append(point)
            combined_data.append(point)
        for sample in valid_samples:
            combined_data.append(sample)

        # Compute exponential decay weights: newer adaptive samples weighted higher
        weights = []
        for _ in range(len(self.calibration_data) * 2):
            weights.append(2.0)  # Original calibration points: fixed weight
        for i in range(len(valid_samples)):
            age = len(valid_samples) - i  # oldest=len, newest=1
            weights.append(max(0.1, math.exp(-0.05 * age)))

        # Recompute calibration matrix with weights
        old_matrix = self.calibration_matrix
        if self._compute_calibration_matrix(combined_data, sample_weights=np.array(weights)):
            self.adaptive_refinement_count += 1
            print(f"Adaptive calibration refinement applied ({len(valid_samples)} valid samples).")
        else:
            self.calibration_matrix = old_matrix
            print("Adaptive refinement failed, keeping previous calibration.")

    def reset_adaptive_calibration(self):
        """Reset adaptive calibration data (useful when recalibrating)."""
        self.adaptive_samples = []
        self.adaptive_sample_count = 0
        self.last_refinement_time = 0

    def save_calibration_profile(self, name="default"):
        """Save current calibration to a named profile JSON file."""
        import json
        if not self.calibration_complete or self.calibration_matrix is None:
            print("No calibration to save.")
            return False
        profile = {
            "name": name,
            "reference_head_pose": list(self.reference_head_pose),
            "calibration_data": [list(p) for p in self.calibration_data],
            "calibration_matrix": {
                "x_coeffs": self.calibration_matrix[0].tolist(),
                "y_coeffs": self.calibration_matrix[1].tolist(),
            },
            "use_simple_model": self._use_simple_model,
            "camera_index": self.camera_index,
        }
        profile_dir = os.path.join(os.path.dirname(__file__), "profiles")
        os.makedirs(profile_dir, exist_ok=True)
        path = os.path.join(profile_dir, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f"Calibration profile saved: {path}")
        return True

    def load_calibration_profile(self, name="default"):
        """Load a named calibration profile, skipping the calibration sequence."""
        import json
        profile_dir = os.path.join(os.path.dirname(__file__), "profiles")
        path = os.path.join(profile_dir, f"{name}.json")
        if not os.path.exists(path):
            print(f"Profile not found: {path}")
            return False
        try:
            with open(path) as f:
                profile = json.load(f)
            self.reference_head_pose = tuple(profile["reference_head_pose"])
            self.calibration_data = [tuple(p) for p in profile["calibration_data"]]
            x_coeffs = np.array(profile["calibration_matrix"]["x_coeffs"])
            y_coeffs = np.array(profile["calibration_matrix"]["y_coeffs"])
            self.calibration_matrix = (x_coeffs, y_coeffs)
            self._use_simple_model = profile.get("use_simple_model", True)
            self.calibration_complete = True
            print(f"Calibration profile loaded: {name}")
            return True
        except Exception as e:
            print(f"Error loading profile: {e}")
            return False

    def list_calibration_profiles(self):
        """List available calibration profiles."""
        profile_dir = os.path.join(os.path.dirname(__file__), "profiles")
        if not os.path.exists(profile_dir):
            return []
        return [f[:-5] for f in os.listdir(profile_dir) if f.endswith(".json")]

    def map_head_to_screen(self, head_yaw, head_pitch):
        if not self.calibration_complete or self.calibration_matrix is None or self.reference_head_pose is None:
            return self.kf.x[0, 0], self.kf.x[1, 0]

        ref_yaw, ref_pitch = self.reference_head_pose
        x_coeffs, y_coeffs = self.calibration_matrix
        rel_yaw = head_yaw - ref_yaw
        rel_pitch = head_pitch - ref_pitch

        # Use appropriate feature set based on model type
        if getattr(self, '_use_simple_model', True):
            # Simple linear model with cross term: ax + by + cxy + d
            features = np.array([rel_yaw, rel_pitch, rel_yaw * rel_pitch, 1.0])
        else:
            # Full polynomial model
            features = np.array([rel_yaw, rel_pitch, rel_yaw**2,
                                rel_pitch**2, rel_yaw * rel_pitch, 1.0])

        screen_x_ratio = np.dot(features, x_coeffs)
        screen_y_ratio = np.dot(features, y_coeffs)
        screen_x_ratio = np.clip(screen_x_ratio, -0.1, 1.1)
        screen_y_ratio = np.clip(screen_y_ratio, -0.1, 1.1)
        screen_x = screen_x_ratio * self.screen_width
        screen_y = screen_y_ratio * self.screen_height

        # Apply acceleration curve: small movements stay precise, large movements amplified
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        dx = screen_x - center_x
        dy = screen_y - center_y
        distance = math.sqrt(dx*dx + dy*dy)
        max_distance = math.sqrt(center_x**2 + center_y**2)
        if distance > 0 and max_distance > 0:
            normalized = distance / max_distance
            accelerated = normalized ** self.config.acceleration_exponent
            scale = accelerated / normalized
            screen_x = center_x + dx * scale
            screen_y = center_y + dy * scale

        margin = 1
        screen_x = max(margin, min(self.screen_width - 1 - margin, screen_x))
        screen_y = max(margin, min(self.screen_height - 1 - margin, screen_y))
        return screen_x, screen_y

    def _draw_landmarks(self, frame, landmarks):
        if not landmarks:
            return
        h, w = frame.shape[:2]
        left_color = (0, 255, 255) if self.left_eye_closed else (255, 255, 0)
        right_color = (0, 255, 255) if self.right_eye_closed else (255, 255, 0)
        for idx in self.LEFT_EYE_V + self.LEFT_EYE_H:
            if idx < len(landmarks):
                cv2.circle(
                    frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 2, left_color, -1)
        for idx in self.RIGHT_EYE_V + self.RIGHT_EYE_H:
            if idx < len(landmarks):
                cv2.circle(
                    frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 2, right_color, -1)
        if 1 < len(landmarks):
            cv2.circle(
                frame, (int(landmarks[1].x * w), int(landmarks[1].y * h)), 3, (0, 0, 255), -1)
        # Draw iris landmarks when gaze is enabled
        if self.gaze_enabled:
            for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                if idx < len(landmarks):
                    cv2.circle(
                        frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 2, (0, 255, 0), -1)

    def start_tracking(self):
        cap = None
        camera_found = False
        preferred_indices = [self.config.camera_index]
        other_indices = [0] + [i for i in range(2, 6)]
        for i in preferred_indices + other_indices:
            print(f"Attempting camera index: {i}")
            # Try DirectShow backend on Windows
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                print(f"  Camera {i} opened.")
                # Let camera use its native resolution
                time.sleep(2.0)
                # Discard first 30 frames (camera warmup)
                print("  Warming up camera...")
                for _ in range(30):
                    cap.grab()
                ret, _ = cap.read()
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if ret and actual_width > 0 and actual_height > 0:
                    print(
                        f"  Camera {i} ready. Res: {int(actual_width)}x{int(actual_height)}")
                    # Quick camera test - show 30 frames
                    print("Testing camera feed for 2 seconds...")
                    for _ in range(30):
                        ret_test, frame_test = cap.read()
                        if ret_test:
                            cv2.imshow("Camera Test", frame_test)
                            cv2.waitKey(66)
                    cv2.destroyWindow("Camera Test")
                    print("Camera test complete.")
                    self.camera_index = i
                    camera_found = True
                    break
                else:
                    print(f"  Camera {i} failed read/res check. Closing.")
                    cap.release()
                    cap = None
            else:
                print(f"  Failed to open camera {i}.")
            if cap:
                cap.release()
                cap = None

        if not camera_found or cap is None:
            print("Error: Could not open webcam.")
            return

        if not self.run_calibration(cap):
            print("Calibration failed/aborted. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return

        print("\n--- Starting Tracking ---")
        main_win_name = 'Head Tracking Feed'
        cv2.namedWindow(main_win_name, cv2.WINDOW_NORMAL)
        feed_w = self.screen_width // 2
        feed_h = self.screen_height // 2
        cv2.resizeWindow(main_win_name, feed_w, feed_h)
        try:
            cv2.moveWindow(main_win_name, 30, self.screen_height - feed_h - 30)  # Positioning it at the bottom left
        except:
            cv2.moveWindow(main_win_name, 30, 30)

        frame_count = 0
        total_frame_count = 0
        fps = 0
        fps_start_time = time.time()
        self.last_time = time.time()
        self.first_pose = True

        while True:
            current_time = time.time()
            dt = current_time - self.last_time
            if dt <= 1e-6:
                dt = 1/30.0
            elif dt > 0.5:
                dt = 1/30.0
            self.last_time = current_time

            ret, frame = cap.read()
            if not ret:
                self.consecutive_frame_failures += 1
                if self.consecutive_frame_failures > self.max_frame_failures:
                    print("Camera lost. Attempting reconnection...")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        time.sleep(1.0)
                        for _ in range(10):
                            cap.grab()
                        self.consecutive_frame_failures = 0
                        print("Camera reconnected.")
                    else:
                        print("Reconnection failed. Retrying in 3 seconds...")
                        time.sleep(3.0)
                continue
            self.consecutive_frame_failures = 0
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            try:
                results = self.face_mesh.process(frame_rgb)
            except Exception as e:
                print(f"MediaPipe Error: {e}")
                frame.flags.writeable = True
                continue
            frame.flags.writeable = True

            left_ear, right_ear = 0.3, 0.3

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                try:
                    head_yaw, head_pitch = self._get_head_pose(
                        landmarks, frame.shape)
                    measured_x, measured_y = self.map_head_to_screen(
                        head_yaw, head_pitch)
                    measurement = np.array([[measured_x], [measured_y]])
                    # Snap Kalman filter on face re-detection to prevent cursor jump
                    self.face_detection_count += 1
                    if self.face_was_lost:
                        self.kf.x[0, 0] = measured_x
                        self.kf.x[1, 0] = measured_y
                        self.kf.x[2, 0] = 0.0
                        self.kf.x[3, 0] = 0.0
                        self.kf.P = np.eye(4) * 500.
                        self.face_was_lost = False
                        print("Face re-detected, Kalman filter reset.")
                    # Hybrid gaze: detect large gaze shifts and jump cursor to target region
                    if self.gaze_enabled and self.calibration_complete:
                        gaze = self._estimate_gaze(landmarks, frame.shape)
                        if gaze is not None:
                            region = self._gaze_to_screen_region(gaze[0], gaze[1])
                            current_time_gaze = time.time()
                            if (self._prev_gaze_region is not None and
                                    region != self._prev_gaze_region and
                                    current_time_gaze - self._last_gaze_jump_time > self._gaze_jump_cooldown):
                                # Gaze shifted to a new region - jump Kalman to region center
                                jump_x, jump_y = self._region_to_screen_center(region)
                                self.kf.x[0, 0] = jump_x
                                self.kf.x[1, 0] = jump_y
                                self.kf.x[2, 0] = 0.0
                                self.kf.x[3, 0] = 0.0
                                self.kf.P = np.eye(4) * 500.
                                self._last_gaze_jump_time = current_time_gaze
                            self._prev_gaze_region = region
                    self.kf.F[0, 2] = dt
                    self.kf.F[1, 3] = dt
                    q_block_half = Q_discrete_white_noise(
                        dim=2, dt=dt, var=self.process_noise_std**2)
                    self.kf.Q = np.zeros((4, 4))
                    self.kf.Q[0, 0] = self.kf.Q[1, 1] = q_block_half[0, 0]
                    self.kf.Q[2, 2] = self.kf.Q[3, 3] = q_block_half[1, 1]
                    self.kf.Q[0, 2] = self.kf.Q[2, 0] = q_block_half[0, 1]
                    self.kf.Q[1, 3] = self.kf.Q[3, 1] = q_block_half[0, 1]
                    # Adaptive measurement noise: trust measurements more during fast movement
                    velocity = math.sqrt(self.kf.x[2, 0]**2 + self.kf.x[3, 0]**2)
                    R_base = self.config.kalman_base_r
                    R_adaptive = max(100.0, R_base / (1.0 + velocity * 0.015))
                    self.kf.R = np.diag([R_adaptive, R_adaptive])
                    self.kf.predict()
                    self.kf.update(measurement)
                    filtered_x = self.kf.x[0, 0]
                    filtered_y = self.kf.x[1, 0]
                    margin = 0
                    smooth_x = max(margin, min(
                        self.screen_width - 1 - margin, filtered_x))
                    smooth_y = max(margin, min(
                        self.screen_height - 1 - margin, filtered_y))
                    # Dead zone: skip move if delta is below threshold to prevent micro-jitter
                    current_x, current_y = pyautogui.position()
                    dx = smooth_x - current_x
                    dy = smooth_y - current_y
                    if math.sqrt(dx*dx + dy*dy) < self.dead_zone_pixels:
                        smooth_x, smooth_y = current_x, current_y
                    try:
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0.0)
                    except Exception as e:
                        print(f"PyAutoGUI Error: {e}")
                    # Dwell click detection
                    if self.dwell_enabled:
                        current_pos = (smooth_x, smooth_y)
                        if self.dwell_anchor is None:
                            self.dwell_anchor = current_pos
                            self.dwell_start_time = time.time()
                        else:
                            ddx = current_pos[0] - self.dwell_anchor[0]
                            ddy = current_pos[1] - self.dwell_anchor[1]
                            dwell_dist = math.sqrt(ddx*ddx + ddy*ddy)
                            if dwell_dist > self.dwell_tolerance:
                                self.dwell_anchor = current_pos
                                self.dwell_start_time = time.time()
                            elif time.time() - self.dwell_start_time >= self.dwell_time:
                                pyautogui.click(button='left')
                                self.total_clicks += 1
                                print(f"Dwell Click at ({int(smooth_x)}, {int(smooth_y)})")
                                self.dwell_anchor = None
                                self.dwell_start_time = None
                    # Pass head pose and screen position for adaptive calibration
                    left_ear, right_ear = self._detect_winks(
                        landmarks,
                        head_pose=(head_yaw, head_pitch),
                        screen_pos=(smooth_x, smooth_y)
                    )
                    # Head gesture scrolling
                    if self.scroll_enabled and self.reference_head_pose is not None:
                        ref_yaw, ref_pitch = self.reference_head_pose
                        rel_pitch = head_pitch - ref_pitch
                        if abs(rel_pitch) > self.scroll_pitch_threshold:
                            direction = 1 if rel_pitch > 0 else -1
                            if self.scroll_direction == direction and self.scroll_start_time is not None:
                                if time.time() - self.scroll_start_time >= self.scroll_sustain_time:
                                    # Scroll speed proportional to pitch magnitude beyond threshold
                                    magnitude = abs(rel_pitch) - self.scroll_pitch_threshold
                                    clicks = int(self.scroll_base_speed * (1.0 + magnitude * 5.0))
                                    pyautogui.scroll(-direction * clicks)
                            else:
                                self.scroll_direction = direction
                                self.scroll_start_time = time.time()
                        else:
                            self.scroll_direction = 0
                            self.scroll_start_time = None
                    # Draw dwell progress indicator on camera feed
                    if self.dwell_enabled and self.dwell_start_time is not None:
                        progress = min(1.0, (time.time() - self.dwell_start_time) / self.dwell_time)
                        angle = int(360 * progress)
                        cv2.ellipse(frame, (frame_width//2, frame_height//2), (20, 20),
                                    0, 0, angle, (0, 255, 0), 2)
                    self._draw_landmarks(frame, landmarks)
                    cv2.putText(frame, f"Pos: ({int(smooth_x)}, {int(smooth_y)})", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"Error in main loop processing: {e}")
                    traceback.print_exc()
                    cv2.putText(frame, "Processing Error", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                cv2.putText(frame, "Face not detected", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                self.kf.x[2, 0] = 0.0
                self.kf.x[3, 0] = 0.0
                self.first_pose = True
                self.face_was_lost = True
                self.face_lost_count += 1
                self.left_wink_counter, self.right_wink_counter = 0, 0
                self.left_eye_closed, self.right_eye_closed = False, False
                self.last_left_wink_detect_time = 0

            frame_count += 1
            total_frame_count += 1
            # Periodic MediaPipe reset to prevent landmark drift during long sessions
            if total_frame_count % self.mediapipe_reset_interval == 0:
                print("Periodic MediaPipe reset...")
                self.face_mesh.close()
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.config.mediapipe_detection_confidence,
                    min_tracking_confidence=self.config.mediapipe_tracking_confidence
                )
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()
            # Session stats display
            session_mins = int((time.time() - self.session_start_time) / 60)
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"{session_mins}m | Clk:{self.total_clicks} Ref:{self.adaptive_refinement_count}",
                        (frame_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(frame, f"L EAR: {left_ear:.2f}", (10, frame_height - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"R EAR: {right_ear:.2f}", (10, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            click_mode = "Dwell" if self.dwell_enabled else "Wink"
            scroll_str = "S:On" if self.scroll_enabled else "S:Off"
            cv2.putText(frame, f"Q C R D:[{click_mode}] {scroll_str}", (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow(main_win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                print("Recalibrating...")
                cv2.destroyWindow(main_win_name)
                if not self.run_calibration(cap):
                    print("Recalibration failed/aborted. Exiting.")
                    break
                else:
                    print("\n--- Resuming Tracking ---")
                    cv2.namedWindow(main_win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(main_win_name, feed_w, feed_h)
                    try:
                        cv2.moveWindow(main_win_name, 30,
                                       self.screen_height - feed_h - 50)
                    except:
                        cv2.moveWindow(main_win_name, 30, 30)
                    frame_count = 0
                    fps_start_time = time.time()
                    self.last_time = time.time()
                    self.first_pose = True
            elif key == ord('r'):
                # Quick re-center: snap reference to current pose without full recalibration
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    current_yaw, current_pitch = self._get_head_pose(landmarks, frame.shape)
                    self.reference_head_pose = (current_yaw, current_pitch)
                    self.first_pose = True
                    # Reset Kalman filter to screen center with zero velocity
                    self.kf.x = np.array([[self.screen_width / 2],
                                          [self.screen_height / 2],
                                          [0.],
                                          [0.]])
                    self.kf.P = np.eye(4) * 500.
                    print("Re-centered to current head position.")
            elif key == ord('d'):
                self.dwell_enabled = not self.dwell_enabled
                self.dwell_anchor = None
                self.dwell_start_time = None
                mode = "Dwell" if self.dwell_enabled else "Wink"
                print(f"Click mode: {mode}")
            elif key == ord('s'):
                self.scroll_enabled = not self.scroll_enabled
                self.scroll_direction = 0
                self.scroll_start_time = None
                print(f"Scroll: {'enabled' if self.scroll_enabled else 'disabled'}")
            elif key == ord('p'):
                self.save_calibration_profile("default")
            elif key == ord('g'):
                self.gaze_enabled = not self.gaze_enabled
                self._prev_gaze_region = None
                print(f"Gaze jump: {'enabled' if self.gaze_enabled else 'disabled'}")

        print("Releasing camera and closing windows.")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == "__main__":

    print("-------------------------------------")
    print(" Head Gaze Tracker Initializing... ")
    print("-------------------------------------")
    try:
        import cv2
        import mediapipe
        import numpy
        import pyautogui
        import filterpy
        print("Core dependencies found.")


    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}")
        sys.exit(1)


    from config import load_config
    app_config = load_config()
    tracker = HeadGazeTracker(config=app_config.tracking)
    tracker.start_tracking()
