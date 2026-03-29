"""Deterministic validation layer — S8 first pass.

Applies fast, code-only corrections to the raw OCR text before any
LLM adjudication is considered:

1. **Character confusion correction** — positionally aware substitution
   of visually similar alphanumeric pairs (``O↔0``, ``I↔1``, ``S↔5``,
   ``B↔8``, ``D↔0``, ``Z↔2``, ``G↔6``).  The correction direction is
   determined by the *expected character category* at each position
   according to the locale regex (alpha-slot vs. digit-slot).

2. **Locale-aware regex validation** — the corrected text is tested
   against one or more regex patterns for the configured locale.

Returns a ``ValidationVerdict`` indicating whether the deterministic
layer accepted the text or whether agentic adjudication is required.

Spec reference: §4 S8 — character correction + regex validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from uni_vision.common.config import ValidationConfig


class Verdict(str, Enum):
    """Outcome of the deterministic validation pass."""

    ACCEPTED = "accepted"
    CORRECTED = "corrected"  # accepted after char substitution
    REGEX_FAIL = "regex_fail"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True)
class ValidationVerdict:
    """Result of the deterministic validator."""

    verdict: Verdict
    corrected_text: str
    original_text: str
    corrections_applied: Dict[str, str]
    matched_locale: Optional[str]


# ── Position-category masks for common Indian plate format ────────
# Indian plates: two alpha, two digit, one/two alpha, four digit
# e.g. MH12AB1234  →  AADDAADDDD  or  AADDADDDD
# We encode expected category per position to guide correction
# direction.  'A' = must be alpha, 'D' = must be digit.


def _infer_position_mask(pattern: str) -> Optional[str]:
    """Derive an A/D position mask from a simple regex pattern.

    Only supports patterns built from ``[A-Z]``, ``[0-9]``, and the
    quantifiers ``{n}`` and ``{n,m}``.  Returns ``None`` for patterns
    too complex to decompose.
    """
    mask: List[str] = []
    i = 0
    while i < len(pattern):
        # Match [A-Z], [A-Z0-9], [0-9]
        m = re.match(r"\[([A-Z0-9\-]+)\]", pattern[i:])
        if m:
            token = m.group(1)
            i += m.end()
            # Determine category
            has_alpha = "A-Z" in token or "A" in token
            has_digit = "0-9" in token or "0" in token
            if has_alpha and not has_digit:
                cat = "A"
            elif has_digit and not has_alpha:
                cat = "D"
            else:
                cat = "X"  # either

            # Check for quantifier
            qm = re.match(r"\{(\d+)(?:,(\d+))?\}", pattern[i:])
            if qm:
                low = int(qm.group(1))
                high = int(qm.group(2)) if qm.group(2) else low
                mask.extend([cat] * high)
                i += qm.end()
            else:
                mask.append(cat)
        elif pattern[i] == "^" or pattern[i] == "$":
            i += 1
        else:
            return None  # unsupported token
    return "".join(mask) if mask else None


# ── Alpha↔Digit confusion maps ───────────────────────────────────

# When a position expects a DIGIT but we have an ALPHA
_ALPHA_TO_DIGIT: Dict[str, str] = {
    "O": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "D": "0",
    "Z": "2",
    "G": "6",
    "T": "7",
    "A": "4",
}

# When a position expects an ALPHA but we have a DIGIT
_DIGIT_TO_ALPHA: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z",
    "6": "G",
    "7": "T",
    "4": "A",
}


class DeterministicValidator:
    """Fast, code-only plate-text validator.

    Parameters
    ----------
    config : ValidationConfig
        Locale patterns, char corrections, confidence threshold.
    """

    def __init__(self, config: ValidationConfig) -> None:
        self._config = config

        # Compile locale regex patterns
        self._patterns: Dict[str, re.Pattern[str]] = {
            locale: re.compile(pattern)
            for locale, pattern in config.locale_patterns.items()
        }

        # Pre-compute position masks for each locale pattern
        self._masks: Dict[str, Optional[str]] = {
            locale: _infer_position_mask(pattern)
            for locale, pattern in config.locale_patterns.items()
        }

    def validate(
        self,
        plate_text: str,
        confidence: float,
    ) -> ValidationVerdict:
        """Run deterministic validation on the raw plate text.

        Steps:
          1. Normalise text (strip whitespace, uppercase).
          2. Try regex match on the raw text — if it passes and
             confidence is above threshold, accept immediately.
          3. If regex fails, apply positional char corrections and
             re-test.
          4. If corrected text passes regex, accept as ``CORRECTED``.
          5. Otherwise return ``REGEX_FAIL``.
          6. If regex passes but confidence is low, mark as
             ``LOW_CONFIDENCE`` (triggers adjudication upstream).
        """
        raw = plate_text.strip().upper()
        threshold = self._config.adjudication_confidence_threshold

        # ── Step 1: Try raw text against all locale patterns ──────
        matched_locale = self._match_any_locale(raw)
        if matched_locale is not None:
            if confidence >= threshold:
                return ValidationVerdict(
                    verdict=Verdict.ACCEPTED,
                    corrected_text=raw,
                    original_text=raw,
                    corrections_applied={},
                    matched_locale=matched_locale,
                )
            return ValidationVerdict(
                verdict=Verdict.LOW_CONFIDENCE,
                corrected_text=raw,
                original_text=raw,
                corrections_applied={},
                matched_locale=matched_locale,
            )

        # ── Step 2: Apply positional corrections ──────────────────
        corrected, corrections = self._apply_corrections(raw)

        matched_locale = self._match_any_locale(corrected)
        if matched_locale is not None:
            v = Verdict.CORRECTED if confidence >= threshold else Verdict.LOW_CONFIDENCE
            return ValidationVerdict(
                verdict=v,
                corrected_text=corrected,
                original_text=raw,
                corrections_applied=corrections,
                matched_locale=matched_locale,
            )

        # ── Step 3: Regex fail ────────────────────────────────────
        return ValidationVerdict(
            verdict=Verdict.REGEX_FAIL,
            corrected_text=raw,
            original_text=raw,
            corrections_applied={},
            matched_locale=None,
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _match_any_locale(self, text: str) -> Optional[str]:
        """Return the first locale whose pattern matches ``text``."""
        # Try default locale first for efficiency
        default = self._config.default_locale
        if default in self._patterns and self._patterns[default].fullmatch(text):
            return default

        for locale, pattern in self._patterns.items():
            if locale == default:
                continue
            if pattern.fullmatch(text):
                return locale
        return None

    def _apply_corrections(
        self,
        text: str,
    ) -> Tuple[str, Dict[str, str]]:
        """Apply positional character confusion corrections.

        Uses the position mask from the default locale pattern to
        determine whether each position expects an alpha or a digit,
        then substitutes accordingly.
        """
        default = self._config.default_locale
        mask = self._masks.get(default)

        if mask is None:
            # No position mask available — try simple global correction
            return self._global_correction(text)

        corrected = list(text)
        corrections: Dict[str, str] = {}

        for i, ch in enumerate(corrected):
            if i >= len(mask):
                break

            expected = mask[i]
            if expected == "A" and ch.isdigit():
                replacement = _DIGIT_TO_ALPHA.get(ch)
                if replacement:
                    corrections[f"pos{i}:{ch}"] = replacement
                    corrected[i] = replacement
            elif expected == "D" and ch.isalpha():
                replacement = _ALPHA_TO_DIGIT.get(ch)
                if replacement:
                    corrections[f"pos{i}:{ch}"] = replacement
                    corrected[i] = replacement

        return "".join(corrected), corrections

    def _global_correction(
        self,
        text: str,
    ) -> Tuple[str, Dict[str, str]]:
        """Fallback: apply generic confusion map from config."""
        corrected = list(text)
        corrections: Dict[str, str] = {}

        for i, ch in enumerate(corrected):
            replacement = self._config.char_corrections.get(ch)
            if replacement and replacement != ch:
                corrections[f"pos{i}:{ch}"] = replacement
                corrected[i] = replacement

        return "".join(corrected), corrections
