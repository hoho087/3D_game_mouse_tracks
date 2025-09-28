from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("OpenCV (cv2) is required: pip install opencv-python") from exc

try:
    from mss import mss
except ModuleNotFoundError as exc:
    raise SystemExit("mss is required for screen capture: pip install mss") from exc

from overlay import OverlayDrawer

TARGET_FPS = 60
FRAME_INTERVAL = 1.0 / TARGET_FPS
RESET_INTERVAL_SECONDS = 5.0
JSON_FLUSH_INTERVAL = 1.0
JSON_OUTPUT_PATH = Path("flow_trace.json")

MONITOR_INDEX = 1  # primary monitor
OVERLAY_ORIGIN = (40, 40)
OVERLAY_SIZE = (320, 180)
MOTION_SCALE = 0.3
TRAJECTORY_COLOR = (0, 255, 0)
TRAJECTORY_WIDTH = 2
BOUNDARY_COLOR = (255, 255, 255)
BOUNDARY_WIDTH = 1

FRAME_DOWNSCALE = 1
GAUSSIAN_KERNEL = (5, 5)
FLOW_MAG_THRESHOLD = 0.08  
MIN_ACTIVE_RATIO = 0.003  
SMOOTHING_ALPHA = 0.15  
PHASE_SMOOTHING_ALPHA = 0.4 
STATIC_SUPPRESS_THRESHOLD = 0.04 
STATIC_DECAY = 0.55 
STOP_EPS = 0.002
DRAW_MIN_STEP = 0.8
DRAW_MAX_SEGMENT = 6.0

LARGE_MOTION_MAG = 3.5
LARGE_ACTIVE_RATIO = 0.40
PHASECORR_DOWNSCALE = 0.25
PHASECORR_MIN_RESPONSE = 0.05


def _ensure_output_path(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _capture_gray_frame(sct: mss, monitor: dict) -> np.ndarray:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    height, width = gray.shape[:2]
    crop_size = min(640, height, width)
    if crop_size < height or crop_size < width:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        gray = gray[top : top + crop_size, left : left + crop_size]

    if FRAME_DOWNSCALE != 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=FRAME_DOWNSCALE,
            fy=FRAME_DOWNSCALE,
            interpolation=cv2.INTER_AREA,
        )
    return cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)


def _phase_correlate_shift(prev_gray: np.ndarray, gray_frame: np.ndarray) -> tuple[np.ndarray, float]:
    small_prev = prev_gray
    small_curr = gray_frame
    if PHASECORR_DOWNSCALE != 1.0:
        small_prev = cv2.resize(
            prev_gray,
            None,
            fx=PHASECORR_DOWNSCALE,
            fy=PHASECORR_DOWNSCALE,
            interpolation=cv2.INTER_AREA,
        )
        small_curr = cv2.resize(
            gray_frame,
            None,
            fx=PHASECORR_DOWNSCALE,
            fy=PHASECORR_DOWNSCALE,
            interpolation=cv2.INTER_AREA,
        )

    hann = cv2.createHanningWindow((small_prev.shape[1], small_prev.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(
        (small_prev.astype(np.float32) * hann),
        (small_curr.astype(np.float32) * hann),
    )
    shift_vec = np.array([shift[0], shift[1]], dtype=np.float32)
    if PHASECORR_DOWNSCALE != 1.0:
        shift_vec /= PHASECORR_DOWNSCALE
    return shift_vec, float(response)


def _write_records(path: Path, records: list[dict[str, float]]) -> None:
    path.write_text(json.dumps(records, indent=2))


def _clip_point(point: np.ndarray) -> np.ndarray:
    x_min, y_min = OVERLAY_ORIGIN
    width, height = OVERLAY_SIZE
    x_max = x_min + width
    y_max = y_min + height
    point[0] = float(np.clip(point[0], x_min, x_max))
    point[1] = float(np.clip(point[1], y_min, y_max))
    return point


def _reset_overlay(drawer: OverlayDrawer, trajectory: deque[np.ndarray], origin_point: np.ndarray) -> None:
    drawer.clear()
    drawer.add_rect(
        OVERLAY_ORIGIN[0],
        OVERLAY_ORIGIN[1],
        OVERLAY_SIZE[0],
        OVERLAY_SIZE[1],
        BOUNDARY_COLOR,
        BOUNDARY_WIDTH,
    )
    trajectory.clear()
    trajectory.append(origin_point.copy())


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main() -> None:
    drawer = OverlayDrawer(max_shapes=2000, refresh_rate=TARGET_FPS)
    drawer.start()

    _ensure_output_path(JSON_OUTPUT_PATH)
    flow_records: list[dict[str, float]] = []

    trajectory: deque[np.ndarray] = deque(maxlen=2000)
    origin_point = np.array(
        [
            OVERLAY_ORIGIN[0] + OVERLAY_SIZE[0] / 2,
            OVERLAY_ORIGIN[1] + OVERLAY_SIZE[1] / 2,
        ],
        dtype=np.float32,
    )

    _reset_overlay(drawer, trajectory, origin_point)

    smoothed_flow = np.zeros(2, dtype=np.float32)
    current_point = origin_point.copy()
    last_drawn_point = origin_point.copy()

    try:
        with mss() as sct:
            monitor = sct.monitors[MONITOR_INDEX]
            prev_gray = _capture_gray_frame(sct, monitor)

            start_time = time.time()
            last_flush_time = start_time
            last_reset_time = start_time

            while True:
                frame_start = time.perf_counter()

                gray_frame = _capture_gray_frame(sct, monitor)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray_frame,
                    None,
                    0.5,
                    3,
                    31,
                    3,
                    5,
                    1.2,
                    0,
                )

                magnitude = np.linalg.norm(flow, axis=2)
                motion_mask = magnitude > FLOW_MAG_THRESHOLD
                active_ratio = float(motion_mask.mean())

                if motion_mask.any() and active_ratio >= MIN_ACTIVE_RATIO:
                    mean_flow = flow[motion_mask].mean(axis=0)
                    mean_mag = float(magnitude[motion_mask].mean())
                else:
                    mean_flow = np.zeros(2, dtype=np.float32)
                    mean_mag = 0.0

                estimator = "farneback"
                phase_response = 0.0

                if (
                    mean_mag > LARGE_MOTION_MAG
                    or active_ratio > LARGE_ACTIVE_RATIO
                ):
                    phase_flow, phase_response = _phase_correlate_shift(prev_gray, gray_frame)
                    if phase_response >= PHASECORR_MIN_RESPONSE:
                        mean_flow = phase_flow
                        mean_mag = float(np.linalg.norm(mean_flow))
                        estimator = "phase_correlate"

                if mean_mag <= STATIC_SUPPRESS_THRESHOLD:
                    smoothed_flow *= (1.0 - STATIC_DECAY)
                else:
                    alpha = PHASE_SMOOTHING_ALPHA if estimator == "phase_correlate" else SMOOTHING_ALPHA
                    smoothed_flow = (
                        (1.0 - alpha) * smoothed_flow
                        + alpha * mean_flow.astype(np.float32)
                    )

                smoothed_mag = float(np.linalg.norm(smoothed_flow))
                if smoothed_mag < STOP_EPS:
                    smoothed_flow[:] = 0.0
                    smoothed_mag = 0.0

                dx = float(smoothed_flow[0])
                dy = float(smoothed_flow[1])
                raw_dx = float(mean_flow[0])
                raw_dy = float(mean_flow[1])
                timestamp = time.time()

                flow_records.append(
                    {
                        "timestamp": timestamp,
                        "raw_dx": raw_dx,
                        "raw_dy": raw_dy,
                        "dx": dx,
                        "dy": dy,
                        "active_ratio": active_ratio,
                        "mean_magnitude": mean_mag,
                        "smoothed_magnitude": smoothed_mag,
                        "estimator": estimator,
                        "phase_response": phase_response,
                    }
                )
                if timestamp - last_flush_time >= JSON_FLUSH_INTERVAL:
                    _write_records(JSON_OUTPUT_PATH, flow_records)
                    last_flush_time = timestamp

                delta = smoothed_flow * MOTION_SCALE
                current_point = _clip_point(current_point + delta)

                segment_len = _distance(last_drawn_point, current_point)
                if segment_len >= DRAW_MIN_STEP:
                    steps = max(1, int(np.ceil(segment_len / DRAW_MAX_SEGMENT)))
                    for idx in range(1, steps + 1):
                        intermediate = last_drawn_point + (current_point - last_drawn_point) * (idx / steps)
                        intermediate = _clip_point(intermediate)
                        drawer.add_line(
                            int(last_drawn_point[0]),
                            int(last_drawn_point[1]),
                            int(intermediate[0]),
                            int(intermediate[1]),
                            TRAJECTORY_COLOR,
                            TRAJECTORY_WIDTH,
                        )
                        last_drawn_point = intermediate
                        trajectory.append(intermediate.copy())
                    current_point = last_drawn_point.copy()

                if timestamp - last_reset_time >= RESET_INTERVAL_SECONDS:
                    _reset_overlay(drawer, trajectory, origin_point)
                    current_point = origin_point.copy()
                    last_drawn_point = origin_point.copy()
                    smoothed_flow[:] = 0.0
                    last_reset_time = timestamp

                prev_gray = gray_frame

                elapsed = time.perf_counter() - frame_start
                remaining = FRAME_INTERVAL - elapsed
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        _write_records(JSON_OUTPUT_PATH, flow_records)
        drawer.stop()


if __name__ == "__main__":
    main()
