"""XML response parser for the Ollama LLM OCR engine.

Parses the ``<result>`` XML block returned by the model into typed
fields.  On parse failure, raises ``LLMParseError`` so the retry
loop in ``llm_ocr.py`` can append the error to context and re-prompt.

Spec reference: §8.3 error recovery loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from uni_vision.common.exceptions import LLMParseError


@dataclass(frozen=True)
class ParsedOCRResponse:
    """Typed representation of a successfully parsed LLM response."""

    plate_text: str
    confidence: float
    char_bboxes: Optional[List[Tuple[int, int, int, int]]]
    reasoning: str


_RESULT_RE = re.compile(
    r"<result>\s*"
    r"<plate_text>\s*(?P<plate_text>.*?)\s*</plate_text>\s*"
    r"<confidence>\s*(?P<confidence>.*?)\s*</confidence>\s*"
    r"(?:<char_bboxes>\s*(?P<char_bboxes>.*?)\s*</char_bboxes>\s*)?"
    r"<reasoning>\s*(?P<reasoning>.*?)\s*</reasoning>\s*"
    r"</result>",
    re.DOTALL,
)


def parse_llm_response(raw: str) -> ParsedOCRResponse:
    """Extract structured fields from the LLM's ``<result>`` XML.

    Raises
    ------
    LLMParseError
        If the response does not match the expected schema.
    """
    match = _RESULT_RE.search(raw)
    if match is None:
        raise LLMParseError(
            f"No <result> block found in LLM response: {raw[:200]}"
        )

    plate_text = match.group("plate_text").strip()
    if not plate_text:
        raise LLMParseError("Empty plate_text in LLM response")

    try:
        confidence = float(match.group("confidence").strip())
    except (ValueError, TypeError) as exc:
        raise LLMParseError(
            f"Invalid confidence value: {match.group('confidence')}"
        ) from exc

    confidence = max(0.0, min(1.0, confidence))

    char_bboxes = _parse_char_bboxes(match.group("char_bboxes"))
    reasoning = match.group("reasoning").strip()

    return ParsedOCRResponse(
        plate_text=plate_text,
        confidence=confidence,
        char_bboxes=char_bboxes,
        reasoning=reasoning,
    )


def _parse_char_bboxes(
    raw: Optional[str],
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Parse semicolon-separated ``x1,y1,x2,y2`` bounding boxes.

    Returns ``None`` if the raw string is empty, ``NONE``, or absent.
    """
    if raw is None:
        return None

    raw = raw.strip()
    if not raw or raw.upper() == "NONE":
        return None

    bboxes: List[Tuple[int, int, int, int]] = []
    for segment in raw.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split(",")
        if len(parts) != 4:
            continue
        try:
            bboxes.append(tuple(int(p.strip()) for p in parts))  # type: ignore[arg-type]
        except ValueError:
            continue

    return bboxes if bboxes else None
