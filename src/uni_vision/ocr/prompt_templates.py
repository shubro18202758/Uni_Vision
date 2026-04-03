"""OCR utility constants — no LLM image processing.

Gemma 4 E2B is the Manager Agent brain only — it does NOT process
images or perform OCR.  All image processing is handled by dedicated
pre-trained models and libraries (EasyOCR, PaddleOCR, TrOCR, etc.)
that the Manager Agent provisions at runtime.

This module retains minimal constants for backward compatibility
with modules that may reference it.
"""

from __future__ import annotations

# Retained for backward compatibility with response_parser imports.
# These are NOT used for LLM image processing.
PLATE_ALLOWLIST: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
