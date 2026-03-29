"""S5 — Geometric skew correction via Hough Line Transform.

Detects dominant line angles in the plate image using probabilistic
Hough lines, computes the median skew angle, and applies an affine
rotation to straighten the plate.  Skew angles beyond ±max_degrees
are clamped; angles within ±skip_threshold are treated as already
straight and the image is returned unchanged.

All operations use OpenCV's optimized C++ backend for minimal latency.

Spec reference: §4 stage S5.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from uni_vision.common.config import DeskewConfig


class HoughStraightener:
    """Preprocessor that corrects rotational skew on plate crops.

    Implements the ``Preprocessor`` protocol (``contracts/preprocessor.py``).
    """

    def __init__(self, config: DeskewConfig) -> None:
        self._config = config

    # ── Preprocessor protocol ─────────────────────────────────────

    @property
    def name(self) -> str:
        return "hough_straightener"

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Estimate skew and apply affine correction if needed."""
        if not self.enabled:
            return image

        angle = self._estimate_skew(image)

        if abs(angle) <= self._config.skip_threshold_degrees:
            return image

        angle = float(np.clip(
            angle,
            -self._config.max_skew_degrees,
            self._config.max_skew_degrees,
        ))

        return self._rotate(image, angle)

    # ── Internal helpers ──────────────────────────────────────────

    def _estimate_skew(self, image: NDArray[np.uint8]) -> float:
        """Return the median skew angle (degrees) from Hough lines."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(
            gray,
            self._config.canny_low,
            self._config.canny_high,
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._config.hough_threshold,
            minLineLength=self._config.hough_min_line_length,
            maxLineGap=self._config.hough_max_line_gap,
        )

        if lines is None or len(lines) == 0:
            return 0.0

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 1e-6:
                continue
            angle_deg = float(np.degrees(np.arctan2(dy, dx)))
            # Keep only near-horizontal lines (±45° from horizontal)
            if abs(angle_deg) <= 45.0:
                angles.append(angle_deg)

        if not angles:
            return 0.0

        return float(np.median(angles))

    @staticmethod
    def _rotate(image: NDArray[np.uint8], angle: float) -> NDArray[np.uint8]:
        """Apply affine rotation around the image centre."""
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        cos_a = abs(rotation_matrix[0, 0])
        sin_a = abs(rotation_matrix[0, 1])
        new_w = int(np.ceil(h * sin_a + w * cos_a))
        new_h = int(np.ceil(h * cos_a + w * sin_a))

        rotation_matrix[0, 2] += (new_w - w) / 2.0
        rotation_matrix[1, 2] += (new_h - h) / 2.0

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return np.ascontiguousarray(rotated)
