"""Optical Character Extraction module — S7.

Re-exports the public API for the OCR subsystem:
  * ``ComponentOCREngine`` — adapter for Manager-provisioned OCR components
  * ``EasyOCRFallback``    — lightweight CPU-only default engine
  * ``OCRStrategy``        — multi-engine strategy (Manager adds engines at runtime)
  * ``parse_llm_response`` — XML response parser
"""

from uni_vision.ocr.fallback_ocr import EasyOCRFallback
from uni_vision.ocr.llm_ocr import ComponentOCREngine
from uni_vision.ocr.response_parser import ParsedOCRResponse, parse_llm_response
from uni_vision.ocr.strategy import OCRStrategy

__all__ = [
    "ComponentOCREngine",
    "EasyOCRFallback",
    "OCRStrategy",
    "ParsedOCRResponse",
    "parse_llm_response",
]
