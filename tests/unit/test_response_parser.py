"""Tests for the XML OCR response parser."""

from __future__ import annotations

import pytest


class TestParseValidResponse:
    """Test successful parsing of well-formed XML responses."""

    def test_full_response(self):
        from uni_vision.ocr.response_parser import parse_llm_response

        xml = """
        <result>
            <plate_text>MH12AB1234</plate_text>
            <confidence>0.92</confidence>
            <char_bboxes>10,20,30,40;50,60,70,80</char_bboxes>
            <reasoning>Clear plate with high contrast</reasoning>
        </result>
        """
        parsed = parse_llm_response(xml)
        assert parsed.plate_text == "MH12AB1234"
        assert parsed.confidence == 0.92
        assert parsed.char_bboxes is not None
        assert len(parsed.char_bboxes) == 2
        assert parsed.char_bboxes[0] == (10, 20, 30, 40)
        assert parsed.reasoning == "Clear plate with high contrast"

    def test_without_char_bboxes(self):
        from uni_vision.ocr.response_parser import parse_llm_response

        xml = """
        <result>
            <plate_text>KA01HG5678</plate_text>
            <confidence>0.85</confidence>
            <reasoning>Blurry but readable</reasoning>
        </result>
        """
        parsed = parse_llm_response(xml)
        assert parsed.plate_text == "KA01HG5678"
        assert parsed.char_bboxes is None

    def test_char_bboxes_none_string(self):
        from uni_vision.ocr.response_parser import parse_llm_response

        xml = """
        <result>
            <plate_text>DL3CAF1234</plate_text>
            <confidence>0.78</confidence>
            <char_bboxes>NONE</char_bboxes>
            <reasoning>Night time capture</reasoning>
        </result>
        """
        parsed = parse_llm_response(xml)
        assert parsed.char_bboxes is None

    def test_confidence_clamped_above_one(self):
        from uni_vision.ocr.response_parser import parse_llm_response

        xml = """
        <result>
            <plate_text>XX99ZZ0000</plate_text>
            <confidence>1.5</confidence>
            <reasoning>test</reasoning>
        </result>
        """
        parsed = parse_llm_response(xml)
        assert parsed.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        from uni_vision.ocr.response_parser import parse_llm_response

        xml = """
        <result>
            <plate_text>XX99ZZ0000</plate_text>
            <confidence>-0.3</confidence>
            <reasoning>test</reasoning>
        </result>
        """
        parsed = parse_llm_response(xml)
        assert parsed.confidence == 0.0

    def test_surrounding_text_ignored(self):
        from uni_vision.ocr.response_parser import parse_llm_response

        xml = """Some preamble text here.
        <result>
            <plate_text>AP09CD5678</plate_text>
            <confidence>0.88</confidence>
            <reasoning>OK</reasoning>
        </result>
        More trailing text."""
        parsed = parse_llm_response(xml)
        assert parsed.plate_text == "AP09CD5678"


class TestParseErrors:
    """Test error handling for malformed responses."""

    def test_no_result_block(self):
        from uni_vision.common.exceptions import LLMParseError
        from uni_vision.ocr.response_parser import parse_llm_response

        with pytest.raises(LLMParseError, match="No <result> block"):
            parse_llm_response("Just some text without XML")

    def test_empty_plate_text(self):
        from uni_vision.common.exceptions import LLMParseError
        from uni_vision.ocr.response_parser import parse_llm_response

        xml = """
        <result>
            <plate_text>  </plate_text>
            <confidence>0.5</confidence>
            <reasoning>empty</reasoning>
        </result>
        """
        with pytest.raises(LLMParseError, match="Empty plate_text"):
            parse_llm_response(xml)

    def test_invalid_confidence(self):
        from uni_vision.common.exceptions import LLMParseError
        from uni_vision.ocr.response_parser import parse_llm_response

        xml = """
        <result>
            <plate_text>AB12CD3456</plate_text>
            <confidence>not_a_number</confidence>
            <reasoning>test</reasoning>
        </result>
        """
        with pytest.raises(LLMParseError, match="Invalid confidence"):
            parse_llm_response(xml)


class TestCharBboxParsing:
    """Test the internal _parse_char_bboxes function."""

    def test_none_input(self):
        from uni_vision.ocr.response_parser import _parse_char_bboxes

        assert _parse_char_bboxes(None) is None

    def test_empty_string(self):
        from uni_vision.ocr.response_parser import _parse_char_bboxes

        assert _parse_char_bboxes("") is None

    def test_none_string(self):
        from uni_vision.ocr.response_parser import _parse_char_bboxes

        assert _parse_char_bboxes("NONE") is None
        assert _parse_char_bboxes("none") is None

    def test_valid_bboxes(self):
        from uni_vision.ocr.response_parser import _parse_char_bboxes

        result = _parse_char_bboxes("1,2,3,4;5,6,7,8")
        assert result is not None
        assert len(result) == 2
        assert result[0] == (1, 2, 3, 4)
        assert result[1] == (5, 6, 7, 8)

    def test_malformed_segment_skipped(self):
        from uni_vision.ocr.response_parser import _parse_char_bboxes

        # Second segment has only 3 values — should be skipped
        result = _parse_char_bboxes("1,2,3,4;5,6,7")
        assert result is not None
        assert len(result) == 1

    def test_all_malformed_returns_none(self):
        from uni_vision.ocr.response_parser import _parse_char_bboxes

        assert _parse_char_bboxes("a,b,c;x,y") is None
