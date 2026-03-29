"""Tests for the deterministic plate-text validator."""

from __future__ import annotations

import pytest


class TestPositionMaskInference:
    """Test the internal _infer_position_mask helper."""

    def test_indian_pattern(self):
        from uni_vision.postprocessing.validator import _infer_position_mask

        # Indian format: AA00AA0000 or AA00A0000
        mask = _infer_position_mask(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")
        assert mask is not None
        # Should start with AA, then DD, then 2 A's (max quantifier), then DDDD
        assert mask.startswith("AA")
        assert mask[2:4] == "DD"

    def test_generic_pattern(self):
        from uni_vision.postprocessing.validator import _infer_position_mask

        mask = _infer_position_mask(r"^[A-Z0-9]{4,10}$")
        assert mask is not None
        # All positions are 'X' (either alpha or digit)
        assert all(c == "X" for c in mask)

    def test_unsupported_pattern_returns_none(self):
        from uni_vision.postprocessing.validator import _infer_position_mask

        assert _infer_position_mask(r"foo.*bar") is None


class TestDeterministicValidator:
    """Test char correction and regex validation."""

    def test_valid_plate_accepted(self, validation_config):
        from uni_vision.postprocessing.validator import DeterministicValidator, Verdict

        v = DeterministicValidator(validation_config)
        result = v.validate("MH12AB1234", confidence=0.9)
        assert result.verdict == Verdict.ACCEPTED
        assert result.corrected_text == "MH12AB1234"
        assert result.corrections_applied == {}
        assert result.matched_locale == "IN"

    def test_lowercase_normalised(self, validation_config):
        from uni_vision.postprocessing.validator import DeterministicValidator, Verdict

        v = DeterministicValidator(validation_config)
        result = v.validate("  mh12ab1234  ", confidence=0.9)
        assert result.verdict == Verdict.ACCEPTED
        assert result.corrected_text == "MH12AB1234"

    def test_low_confidence_flagged(self, validation_config):
        from uni_vision.postprocessing.validator import DeterministicValidator, Verdict

        v = DeterministicValidator(validation_config)
        # Valid regex but below confidence threshold (0.75)
        result = v.validate("MH12AB1234", confidence=0.5)
        assert result.verdict == Verdict.LOW_CONFIDENCE

    def test_char_correction_o_to_zero(self):
        from uni_vision.common.config import ValidationConfig
        from uni_vision.postprocessing.validator import DeterministicValidator, Verdict

        # Use IN-only locale so GENERIC doesn't match the raw text first
        cfg = ValidationConfig(
            locale_patterns={"IN": "^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"},
        )
        v = DeterministicValidator(cfg)
        # "MH12AB12O4" — O in digit position → should be corrected to 0
        result = v.validate("MH12AB12O4", confidence=0.9)
        assert result.verdict == Verdict.CORRECTED
        assert result.corrected_text == "MH12AB1204"

    def test_regex_fail_garbage(self, validation_config):
        from uni_vision.postprocessing.validator import DeterministicValidator, Verdict

        v = DeterministicValidator(validation_config)
        result = v.validate("GARBAGE!!!", confidence=0.9)
        assert result.verdict == Verdict.REGEX_FAIL

    def test_generic_locale_fallback(self, validation_config):
        from uni_vision.postprocessing.validator import DeterministicValidator, Verdict

        v = DeterministicValidator(validation_config)
        # Doesn't match IN pattern but matches GENERIC
        result = v.validate("ABC123", confidence=0.9)
        assert result.verdict == Verdict.ACCEPTED
        assert result.matched_locale == "GENERIC"


class TestAlphaDigitMaps:
    """Verify the confusion maps are symmetric and complete."""

    def test_alpha_to_digit_map(self):
        from uni_vision.postprocessing.validator import _ALPHA_TO_DIGIT

        assert _ALPHA_TO_DIGIT["O"] == "0"
        assert _ALPHA_TO_DIGIT["I"] == "1"
        assert _ALPHA_TO_DIGIT["S"] == "5"
        assert _ALPHA_TO_DIGIT["B"] == "8"
        assert _ALPHA_TO_DIGIT["Z"] == "2"
        assert _ALPHA_TO_DIGIT["G"] == "6"
        assert _ALPHA_TO_DIGIT["T"] == "7"

    def test_digit_to_alpha_map(self):
        from uni_vision.postprocessing.validator import _DIGIT_TO_ALPHA

        assert _DIGIT_TO_ALPHA["0"] == "O"
        assert _DIGIT_TO_ALPHA["1"] == "I"
        assert _DIGIT_TO_ALPHA["5"] == "S"
        assert _DIGIT_TO_ALPHA["8"] == "B"
        assert _DIGIT_TO_ALPHA["2"] == "Z"
        assert _DIGIT_TO_ALPHA["6"] == "G"
