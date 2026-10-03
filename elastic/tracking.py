"""Extract a beam-tip time series from a vibration video.

The two detector profiles consolidate the alternatives in the supplied notebook.
Coordinates assume a vertical beam with its clamp at the bottom of the image.
No material property is inferred by the detector itself.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd


@dataclass
class TrackingConfig:
    # 'legacy' reproduces the last active detector in the supplied notebook.
    # 'component' selects its morphology/component and median-tip alternative.
    profile: str = "legacy"
    hsv_lower: tuple | None = None
    hsv_upper: tuple | None = None
    gap_threshold_px: int = 3
    lateral_tolerance_px: float = 100.0
    max_jump_px: float | None = None
    roi: tuple | None = None  # (left, top, width, height), in original image pixels


def beam_pixels(image, config):
    """Return (image x, image y) pixels after configurable HSV segmentation."""
    if config.profile not in {"legacy", "component"}:
        raise ValueError("profile must be 'legacy' or 'component'")
    low = config.hsv_lower or ((40, 40, 40) if config.profile == "legacy" else (45, 60, 25))
    high = config.hsv_upper or ((80, 255, 255) if config.profile == "legacy" else (85, 255, 160))
    limits = np.array([179, 255, 255])
    if len(low) != 3 or len(high) != 3 or np.any(np.array(low) < 0) or np.any(np.array(high) > limits) or np.any(np.array(low) > high):
        raise ValueError("HSV bounds must satisfy 0 <= lower <= upper <= (179,255,255)")
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), np.array(low, np.uint8), np.array(high, np.uint8))
    if config.roi is not None:
        left, top, width, height = config.roi
        if min(left, top) < 0 or min(width, height) <= 0 or left + width > image.shape[1] or top + height > image.shape[0]:
            raise ValueError("ROI must lie inside the image")
        roi_mask = np.zeros_like(mask)
        roi_mask[top:top + height, left:left + width] = 255
        mask = cv2.bitwise_and(mask, roi_mask)
    if config.profile == "component":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best_label, best_score = None, -1
        for label in range(1, count):
            _, _, width, height, area = stats[label]
            aspect = max(width, height) / max(min(width, height), 1)
            if area >= 40 and aspect > 3 and area * aspect > best_score:
                best_label, best_score = label, area * aspect
        if best_label is not None:
            mask = np.uint8(labels == best_label) * 255
    rows, columns = np.where(mask > 0)
    return list(zip(columns.tolist(), rows.tolist()))


def root_coordinates(pixels):
    """Map (image x, image y) to (-y, -x) and remove the near-root offset."""
    if not pixels:
        return []
    points = [(-y, -x) for x, y in pixels]
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    if xmax <= xmin:
        return []
    edges = []
    for divisor in (15, 20):
        column = round(xmin + (xmax - xmin) / divisor)
        values = [y for x, y in points if x == column]
        if not values:
            values = [y for x, y in points if abs(x - column) <= 2]
        if not values:
            return []
        edges.extend((min(values), max(values)))
    root_y = np.mean(edges)
    return [(x - xmin, y - root_y) for x, y in points]


def continuous_pixels(points, gap_threshold=3, lateral_tolerance=100):
    """Retain the source notebook's root-to-tip continuity filter."""
    if not points:
        return []
    points = sorted(points, key=lambda p: p[0])
    kept, previous_y = [points[0]], [points[0][1]]
    for i in range(1, len(points)):
        x, y = points[i]
        previous_x = points[i - 1][0]
        if x - previous_x > gap_threshold:
            break
        if abs(y - np.mean(previous_y)) <= lateral_tolerance:
            kept.append((x, y))
            previous_y = previous_y + [y] if x == previous_x else [y]
    return kept


def tip_and_thickness(points, profile):
    """Return tip axial position, transverse position, and a thickness diagnostic."""
    if not points:
        return np.nan, np.nan, np.nan
    xmax = max(x for x, _ in points)
    if profile == "legacy":
        values = [y for x, y in points if x == xmax]
        tip_y = max(values, key=abs)  # Original far-edge definition, not tip center.
    else:
        tip_y = float(np.median([y for x, y in points if x >= xmax - 5]))
    column = round(xmax / 3)
    thickness_values = [y for x, y in points if x == column]
    if not thickness_values:
        thickness_values = [y for x, y in points if abs(x - column) <= 2]
    thickness = np.ptp(thickness_values) if thickness_values else np.nan
    return xmax, tip_y, thickness


def track_video(video_path, output_csv, *, acquisition_fps, config=None, preview_every=0):
    """Track every decoded frame; retain failed detections as NaN rows.

    acquisition_fps is the physical camera rate, NOT slow-motion playback FPS.
    max_jump_px flags suspect frames as NaN; it never silently holds the last tip.
    Previews and metadata are written beside output_csv. Existing outputs require
    a new filename, so a rerun cannot silently replace a previous experiment.
    """
    from dataclasses import asdict

    config = config or TrackingConfig()
    if not np.isfinite(acquisition_fps) or acquisition_fps <= 0:
        raise ValueError("Set the measured physical acquisition_fps to a positive value")
    if config.max_jump_px is not None and (not np.isfinite(config.max_jump_px) or config.max_jump_px <= 0):
        raise ValueError("max_jump_px must be positive or None")
    if preview_every < 0 or int(preview_every) != preview_every:
        raise ValueError("preview_every must be a nonnegative integer")
    video_path, output_csv = Path(video_path), Path(output_csv)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    metadata_path = output_csv.with_suffix(".metadata.json")
    preview_dir = output_csv.parent / (output_csv.stem + "_previews")
    if output_csv.exists() or metadata_path.exists() or (preview_every and preview_dir.exists()):
        raise FileExistsError("Choose a new output name for this run")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot decode video: {video_path}")
    playback_fps = cap.get(cv2.CAP_PROP_FPS)
    expected_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, previous_raw = [], None
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = len(rows)
            detected = beam_pixels(frame, config)
            points = continuous_pixels(root_coordinates(detected), config.gap_threshold_px, config.lateral_tolerance_px)
            tip_x, raw_y, thickness = tip_and_thickness(points, config.profile)
            status, saved_y = "accepted", raw_y
            if not np.isfinite(raw_y):
                status = "no_detection"
            elif config.max_jump_px is not None and previous_raw is not None and abs(raw_y - previous_raw) > config.max_jump_px:
                status, saved_y = "jump_flagged", np.nan
            previous_raw = raw_y if np.isfinite(raw_y) else None
            rows.append(dict(frame_index=frame_index, time_s=frame_index / acquisition_fps,
                             tip_displacement_px=saved_y, raw_tip_displacement_px=raw_y,
                             beam_thickness_px=thickness, tip_axial_px=tip_x, tracking_status=status))
            if preview_every and frame_index % preview_every == 0:
                preview_dir.mkdir(exist_ok=True)
                overlay = frame.copy()
                if detected:
                    xy = np.asarray(detected)
                    overlay[xy[:, 1], xy[:, 0]] = (0, 0, 255)
                preview = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                if not cv2.imwrite(str(preview_dir / f"frame_{frame_index:06d}.png"), preview):
                    raise OSError("Could not write diagnostic preview")
    finally:
        cap.release()
    if not rows:
        raise ValueError("Video contains no decodable frames")
    if expected_count > 0 and len(rows) != expected_count:
        raise ValueError(f"Decoded {len(rows)} of {expected_count} reported frames; check the video before fitting")
    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)
    metadata_path.write_text(json.dumps(dict(input_video=video_path.name, acquisition_fps=acquisition_fps,
        playback_fps=playback_fps, decoded_frames=len(rows), config=asdict(config)), indent=2), encoding="utf-8")
    return result
