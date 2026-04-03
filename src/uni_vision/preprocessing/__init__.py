"""Plate preprocessing — stages S4 (crop), S5 (deskew), S6 (enhance)."""

from uni_vision.preprocessing.chain import PreprocessingChain
from uni_vision.preprocessing.deskew import HoughStraightener
from uni_vision.preprocessing.enhance import PhotometricEnhancer
from uni_vision.preprocessing.roi_extractor import extract_plate_roi

__all__ = [
    "HoughStraightener",
    "PhotometricEnhancer",
    "PreprocessingChain",
    "extract_plate_roi",
]
