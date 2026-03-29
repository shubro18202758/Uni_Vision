"""Post-processing package — S8 cognitive orchestrator + event dispatch.

Public API:
  CognitiveOrchestrator      — the PostProcessor implementation (validator + adjudicator)
  DeterministicValidator      — fast char-correction + locale regex (Layer 1)
  ConsensusAdjudicator        — multi-engine consensus voting (Layer 2)
  SlidingWindowDeduplicator   — temporal dedup across consecutive frames
  MultiDispatcher             — async persistence dispatcher (DB + object store)
"""

from uni_vision.postprocessing.adjudicator import ConsensusAdjudicator
from uni_vision.postprocessing.deduplicator import SlidingWindowDeduplicator
from uni_vision.postprocessing.dispatcher import MultiDispatcher
from uni_vision.postprocessing.orchestrator import CognitiveOrchestrator
from uni_vision.postprocessing.validator import (
    DeterministicValidator,
    ValidationVerdict,
    Verdict,
)

__all__ = [
    "CognitiveOrchestrator",
    "DeterministicValidator",
    "ConsensusAdjudicator",
    "SlidingWindowDeduplicator",
    "MultiDispatcher",
    "ValidationVerdict",
    "Verdict",
]