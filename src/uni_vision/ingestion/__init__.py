"""Stream ingestion and temporal sampling — pipeline stages S0 & S1."""

from uni_vision.ingestion.phash import compute_phash, hamming_distance
from uni_vision.ingestion.rtsp_source import RTSPFrameSource
from uni_vision.ingestion.sampler import TemporalSampler

__all__ = [
    "compute_phash",
    "hamming_distance",
    "RTSPFrameSource",
    "TemporalSampler",
]
