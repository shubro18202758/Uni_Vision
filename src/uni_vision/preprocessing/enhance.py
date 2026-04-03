"""S6 — Photometric enhancement for plate images.

Applies a configurable sequence of image-processing sub-stages to
maximise OCR readability:

  1. **Resize** — upscale small plates to a minimum height (INTER_CUBIC).
  2. **CLAHE** — localised contrast equalisation on the L-channel of
     LAB colour space, avoiding global histogram distortion.
  3. **Gaussian blur** — mild smoothing to suppress sensor noise.
  4. **Bilateral filter** — edge-preserving smoothing that retains
     character boundaries while reducing background texture.

Each sub-stage is independently toggleable via ``EnhanceConfig``.
All operations use OpenCV's optimised C++ backend.

Spec reference: §4 stage S6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from uni_vision.common.config import EnhanceConfig


class PhotometricEnhancer:
    """Preprocessor that applies photometric normalisation to plate crops.

    Implements the ``Preprocessor`` protocol (``contracts/preprocessor.py``).
    """

    def __init__(self, config: EnhanceConfig) -> None:
        self._config = config

        # Pre-create the CLAHE object (stateless, reusable)
        if config.clahe_enabled:
            tile = tuple(config.clahe_tile_grid_size)
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,
                tileGridSize=tile,
            )

    # ── Preprocessor protocol ─────────────────────────────────────

    @property
    def name(self) -> str:
        return "photometric_enhancer"

    @property
    def enabled(self) -> bool:
        return (
            self._config.resize_enabled
            or self._config.clahe_enabled
            or self._config.gaussian_blur_enabled
            or self._config.bilateral_enabled
        )

    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply the enabled enhancement sub-stages sequentially."""
        result = image

        if self._config.resize_enabled:
            result = self._resize(result)

        if self._config.clahe_enabled:
            result = self._apply_clahe(result)

        if self._config.gaussian_blur_enabled:
            result = self._gaussian_blur(result)

        if self._config.bilateral_enabled:
            result = self._bilateral_filter(result)

        return result

    # ── Sub-stage implementations ─────────────────────────────────

    def _resize(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Upscale to minimum height while preserving aspect ratio."""
        h, w = image.shape[:2]
        min_h = self._config.resize_min_height

        if h >= min_h:
            return image

        scale = min_h / h
        new_w = round(w * scale)
        resized = cv2.resize(
            image,
            (new_w, min_h),
            interpolation=cv2.INTER_CUBIC,
        )
        return np.ascontiguousarray(resized)

    def _apply_clahe(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """CLAHE on the L-channel of LAB colour space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        l_ch = self._clahe.apply(l_ch)

        merged = cv2.merge((l_ch, a_ch, b_ch))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return np.ascontiguousarray(enhanced)

    def _gaussian_blur(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Mild Gaussian smoothing to suppress sensor noise."""
        ksize = tuple(self._config.gaussian_kernel_size)
        blurred = cv2.GaussianBlur(image, ksize, 0)
        return np.ascontiguousarray(blurred)

    def _bilateral_filter(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Edge-preserving bilateral smoothing."""
        filtered = cv2.bilateralFilter(
            image,
            d=self._config.bilateral_d,
            sigmaColor=self._config.bilateral_sigma_color,
            sigmaSpace=self._config.bilateral_sigma_space,
        )
        return np.ascontiguousarray(filtered)
