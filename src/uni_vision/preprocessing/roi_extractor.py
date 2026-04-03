"""S4 — Plate ROI extraction with configurable padding margin.

Crops the detected plate region from the full frame, applying a
symmetric pixel-padding margin clamped to frame boundaries.  This is
the GPU→CPU transfer point in the architecture; however, the actual
tensor transfer is handled upstream by the detection layer.  This
module operates entirely on CPU-resident numpy arrays.

Spec reference: §4 stage S4 — <1 ms latency target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from uni_vision.contracts.dtos import BoundingBox


def extract_plate_roi(
    frame: NDArray[np.uint8],
    bbox: BoundingBox,
    padding_px: int = 5,
) -> NDArray[np.uint8]:
    """Crop the plate region from *frame* with symmetric padding.

    Parameters
    ----------
    frame:
        Full-resolution BGR frame, shape ``(H, W, 3)``, dtype ``uint8``.
    bbox:
        Plate bounding box with pixel coordinates.
    padding_px:
        Symmetric padding added to each side of the bbox, clamped to
        frame boundaries.  Must be non-negative.

    Returns
    -------
    A contiguous numpy array of the cropped plate region.
    """
    h, w = frame.shape[:2]

    x1 = max(bbox.x1 - padding_px, 0)
    y1 = max(bbox.y1 - padding_px, 0)
    x2 = min(bbox.x2 + padding_px, w)
    y2 = min(bbox.y2 + padding_px, h)

    roi = frame[y1:y2, x1:x2]
    return np.ascontiguousarray(roi)
