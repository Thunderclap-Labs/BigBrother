"""
phone_detector.py — YOLOv8-based phone detection module for BigBrother.

Uses Ultralytics YOLOv8 to detect cell phones (COCO class 67) in webcam
frames. Tracks detection persistence so we only trigger scare events when
a phone has been visible for a configurable duration.
"""

import time
import logging
import collections
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID for cell phone
CELL_PHONE_CLASS_ID = 67

# COCO class IDs that represent screens / flat devices (suppress phone if overlapping)
SCREEN_CLASS_IDS = [62, 63]  # 62=tv/monitor, 63=laptop (covers iPads sometimes)

# Maximum camera indices to probe when listing available cameras
_CAMERA_SCAN_LIMIT = 4  # Most setups have 0-3 cameras; probing further is slow


@dataclass
class PhoneDetectionResult:
    """Result from a single frame of phone detection."""
    detected: bool = False
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    annotated_frame: Optional[np.ndarray] = None
    detection_duration: float = 0.0
    # Active-use fields: phone detected + user NOT looking at camera/screen
    face_visible: bool = True   # True = user is facing the camera
    actively_on_phone: bool = False  # True = phone visible AND face NOT visible


class PhoneDetector:
    """
    Real-time phone detector using YOLOv8.

    Captures frames from the webcam, runs YOLOv8 inference filtered to
    COCO class 67 (cell phone), and tracks how long a phone has been
    continuously visible.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        camera_index: int = 0,
        roi: Optional[Tuple[int, int, int, int]] = None,
        calibration: Optional[Dict] = None,
        filter_screens: bool = True,
        screen_overlap_threshold: float = 0.4,
        max_phone_area_fraction: float = 0.20,
    ):
        """
        filter_screens: suppress phone detections that overlap with
                        a TV/monitor/laptop bounding box or are too large
                        to plausibly be a hand-held phone (likely an iPad).
        screen_overlap_threshold: IoU fraction at which we decide the
                                  'phone' is really a monitor/tablet.
        max_phone_area_fraction: if the phone bbox covers this fraction
                                 of the total frame, ignore it (iPad/laptop).
        """
        # Lazy-import ultralytics so we fail fast with a clear message
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for phone detection. "
                "Install it with: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.camera_index = camera_index
        self.roi = roi  # (x1, y1, x2, y2) or None
        # calibration keys: width, height, fps, brightness, contrast, exposure
        self.calibration: Dict = calibration or {}

        self.filter_screens = filter_screens
        self.screen_overlap_threshold = screen_overlap_threshold
        self.max_phone_area_fraction = max_phone_area_fraction

        # ── Motion-based active-use detection ────────────────────────────
        # Strategy: track the phone's centroid (cx, cy) over a rolling time
        # window. A phone lying still on the desk barely moves; a phone being
        # picked up and held moves noticeably.
        #
        # Works for ANY camera angle — front, side, overhead — because it
        # relies purely on bbox movement, not spatial overlap or face direction.
        #
        # _motion_window   : seconds of history to look back (default 2.5 s)
        # _motion_threshold: pixels of total displacement that constitutes
        #                    "movement" (default 35 px; tune up/down)
        # _motion_grace    : seconds after first detection treated as active
        #                    so we catch the pick-up moment before the phone
        #                    has had time to become "stationary"
        self._motion_window: float = 2.5
        self._motion_threshold: float = 35.0
        self._motion_grace: float = 2.0
        # Rolling buffer of (timestamp, cx, cy)
        self._centroid_history: Deque[Tuple[float, float, float]] = collections.deque()
        self._phone_first_seen: Optional[float] = None  # time phone first appeared

        self._detection_start: Optional[float] = None
        self._phone_detected: bool = False
        self._active_start: Optional[float] = None
        self._actively_on_phone: bool = False
        self._last_person_boxes: List[Tuple[int, int, int, int]] = []  # kept for compat
        self._latest_result: Optional[PhoneDetectionResult] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._switch_lock = __import__('threading').Lock()
        self._pending_camera_index: Optional[int] = None

        logger.info(
            "PhoneDetector initialized — model=%s  conf=%.2f  cam=%d  filter_screens=%s",
            model_path, confidence, camera_index, filter_screens,
        )

    # ------------------------------------------------------------------
    # Camera enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def list_cameras(max_index: int = _CAMERA_SCAN_LIMIT) -> List[Dict]:
        """
        Probe camera indices 0..max_index-1 and return a list of dicts
        describing each available camera::

            [{"index": 0, "width": 1280, "height": 720, "fps": 30.0}, ...]
        """
        found: List[Dict] = []
        import sys
        use_dshow = sys.platform == "win32"  # DirectShow skips MSMF EOS spam
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if use_dshow else cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    info = {
                        "index": idx,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                    }
                    found.append(info)
                    logger.info(
                        "Camera %d found: %dx%d @ %.0f fps",
                        idx, info["width"], info["height"], info["fps"],
                    )
                cap.release()
        if not found:
            logger.warning("No cameras detected during scan.")
        return found

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def open_camera(self) -> bool:
        """Open the webcam and apply calibration settings. Returns True on success."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("Could not open camera index %d", self.camera_index)
            return False

        # Apply calibration / capture properties
        cal = self.calibration
        if cal.get("width") and cal.get("height"):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal["width"])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal["height"])
        if cal.get("fps"):
            self._cap.set(cv2.CAP_PROP_FPS, cal["fps"])
        if cal.get("brightness") is not None:
            self._cap.set(cv2.CAP_PROP_BRIGHTNESS, cal["brightness"])
        if cal.get("contrast") is not None:
            self._cap.set(cv2.CAP_PROP_CONTRAST, cal["contrast"])
        if cal.get("exposure") is not None:
            # Disable auto-exposure first, then set manual value
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual
            self._cap.set(cv2.CAP_PROP_EXPOSURE, cal["exposure"])

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera %d opened — %dx%d @ %.0f fps",
            self.camera_index, actual_w, actual_h, actual_fps,
        )
        return True

    def release_camera(self):
        """Release the webcam resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def switch_camera(self, new_index: int) -> bool:
        """
        Hot-swap to a different camera while the detection loop keeps running.
        Releases the current capture and reopens on new_index.
        Returns True on success.
        """
        with self._switch_lock:
            logger.info("Switching camera from %d to %d", self.camera_index, new_index)
            self.release_camera()
            self.camera_index = new_index
            success = self.open_camera()
            if success:
                logger.info("Camera switched to index %d", new_index)
            else:
                logger.error("Failed to switch to camera %d", new_index)
            return success


    def grab_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the webcam. Returns None on failure."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> PhoneDetectionResult:
        """
        Run YOLOv8 inference on *frame*.

        Filters detections to COCO class 67 (cell phone).  If an ROI is
        configured, only detections whose centre falls inside the ROI are
        kept.

        Returns a PhoneDetectionResult with confidence, bounding box, and
        annotated frame.
        """
        result = PhoneDetectionResult()

        # Detect phones and screen classes in one pass
        detect_classes = [CELL_PHONE_CLASS_ID] + (SCREEN_CLASS_IDS if self.filter_screens else [])
        preds = self.model.predict(
            frame,
            classes=detect_classes,
            conf=self.confidence,
            verbose=False,
        )

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        annotated = frame.copy()
        best_conf = 0.0
        best_box = None

        screen_boxes: List[Tuple[int, int, int, int]] = []
        phone_candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []

        for det in preds:
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if cls_id in SCREEN_CLASS_IDS:
                    screen_boxes.append((x1, y1, x2, y2))
                    color = (80, 80, 200)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(annotated, f"Screen {conf:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                elif cls_id == CELL_PHONE_CLASS_ID:
                    phone_candidates.append((conf, (x1, y1, x2, y2)))

        for conf, (x1, y1, x2, y2) in phone_candidates:
            # ROI filter
            if self.roi is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                rx1, ry1, rx2, ry2 = self.roi
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue

            if self.filter_screens:
                box_area = (x2 - x1) * (y2 - y1)

                # Size filter: if box is too large relative to the frame it is
                # likely an iPad, a tablet or a monitor — not a hand-held phone.
                if box_area > frame_area * self.max_phone_area_fraction:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (100, 100, 100), 1)
                    cv2.putText(annotated, "iPad/Screen (ignored)", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                    continue

                # Overlap filter: suppress if the phone bbox overlaps heavily
                # with any detected tv/laptop/monitor box.
                suppressed = False
                for sx1, sy1, sx2, sy2 in screen_boxes:
                    inter_x1 = max(x1, sx1)
                    inter_y1 = max(y1, sy1)
                    inter_x2 = min(x2, sx2)
                    inter_y2 = min(y2, sy2)
                    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                        continue
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    overlap_frac = inter_area / max(box_area, 1)
                    if overlap_frac >= self.screen_overlap_threshold:
                        suppressed = True
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (100, 100, 100), 1)
                        cv2.putText(annotated, "Suppressed (screen overlap)", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
                        break
                if suppressed:
                    continue

            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Phone {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw ROI if set
        if self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        # Update detection state
        now = time.time()
        detected = best_conf > 0

        # ── Motion-based active-use detection (ON DESK — DISABLED) ──────
        # Commented out: stationary phones were treated as passive ("on desk")
        # and would not trigger the scare system. With this disabled every
        # detected phone is treated as active use regardless of movement.
        #
        # if detected and best_box is not None:
        #     cx = (best_box[0] + best_box[2]) / 2.0
        #     cy = (best_box[1] + best_box[3]) / 2.0
        #     if self._phone_first_seen is None:
        #         self._phone_first_seen = now
        #     self._centroid_history.append((now, cx, cy))
        #     cutoff = now - self._motion_window
        #     while self._centroid_history and self._centroid_history[0][0] < cutoff:
        #         self._centroid_history.popleft()
        #     in_grace = (now - self._phone_first_seen) < self._motion_grace
        #     if in_grace:
        #         active = True
        #     elif len(self._centroid_history) >= 2:
        #         xs = [e[1] for e in self._centroid_history]
        #         ys = [e[2] for e in self._centroid_history]
        #         displacement = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        #         active = displacement >= self._motion_threshold
        #     face_visible = not active
        #     status_color = (0, 80, 255) if active else (0, 180, 0)
        #     status_text  = "HELD" if active else "on desk"
        #     cv2.putText(annotated, status_text, (best_box[0], best_box[3] + 18),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        # else:
        #     self._centroid_history.clear()
        #     self._phone_first_seen = None

        face_visible = True  # kept for API compatibility
        active = detected    # any detected phone is treated as active use

        # --- Raw phone state ---
        if detected:
            if not self._phone_detected:
                self._detection_start = now
            self._phone_detected = True
        else:
            self._phone_detected = False
            self._detection_start = None

        # --- Active-use state (what triggers the scare) ---
        if active:
            if not self._actively_on_phone:
                self._active_start = now
            self._actively_on_phone = True
        else:
            self._actively_on_phone = False
            self._active_start = None

        duration = 0.0
        if self._actively_on_phone and self._active_start is not None:
            duration = now - self._active_start
        elif self._phone_detected and self._detection_start is not None:
            # Expose raw duration even when not "active" (for UI info)
            pass

        result.detected = detected
        result.confidence = best_conf
        result.bbox = best_box
        result.annotated_frame = annotated
        result.detection_duration = duration
        result.face_visible = face_visible
        result.actively_on_phone = active

        self._latest_result = result
        self._latest_frame = annotated

        return result

    def is_phone_persistent(self, min_duration: float = 2.0) -> bool:
        """
        Returns True if the phone has been actively held/used for at least
        *min_duration* seconds.

        Active = the phone's centroid has moved more than _motion_threshold
        pixels within the last _motion_window seconds, OR the phone just
        appeared (grace period — catches the pick-up moment).

        Works for any camera angle — front, side, or overhead — because it
        relies purely on bbox movement, not face direction or spatial overlap.
        """
        if self._latest_result is None:
            return False
        return (
            self._latest_result.actively_on_phone
            and self._latest_result.detection_duration >= min_duration
        )

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame with YOLO bounding boxes drawn."""
        return self._latest_frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def update_settings(
        self,
        confidence: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        filter_screens: Optional[bool] = None,
        screen_overlap_threshold: Optional[float] = None,
        max_phone_area_fraction: Optional[float] = None,
        motion_threshold: Optional[float] = None,
        motion_window: Optional[float] = None,
    ):
        """Hot-update detection settings without recreating the detector."""
        if confidence is not None:
            self.confidence = confidence
            logger.info("Confidence threshold updated to %.2f", confidence)
        if roi is not None:
            self.roi = roi
            logger.info("ROI updated to %s", roi)
        if filter_screens is not None:
            self.filter_screens = filter_screens
            logger.info("filter_screens updated to %s", filter_screens)
        if screen_overlap_threshold is not None:
            self.screen_overlap_threshold = screen_overlap_threshold
        if max_phone_area_fraction is not None:
            self.max_phone_area_fraction = max_phone_area_fraction
        if motion_threshold is not None:
            self._motion_threshold = motion_threshold
            logger.info("motion_threshold updated to %.1f px", motion_threshold)
        if motion_window is not None:
            self._motion_window = motion_window
            logger.info("motion_window updated to %.1f s", motion_window)
