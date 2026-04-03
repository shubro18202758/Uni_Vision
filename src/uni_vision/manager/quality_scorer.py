"""Component Quality Scorer — multi-dimensional Bayesian scoring.

Assigns a composite quality score to each component based on:

  * **Latency efficiency** — normalised against the budget for that stage.
  * **Accuracy / confidence** — output confidence relative to peers.
  * **Reliability** — success rate with recent-bias weighting.
  * **VRAM efficiency** — quality per MB of VRAM consumed.

Scores are updated online with Bayesian smoothing so that new or
rarely-used components start with a prior and converge to their
true quality as data accumulates.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uni_vision.components.base import ComponentCapability

log = logging.getLogger(__name__)

# Default prior (Beta distribution parameters for reliability)
_ALPHA_PRIOR = 2.0
_BETA_PRIOR = 1.0

# Weight vector for composite score (must sum to 1.0)
_W_LATENCY = 0.25
_W_CONFIDENCE = 0.30
_W_RELIABILITY = 0.30
_W_VRAM = 0.15


@dataclass
class _BetaTracker:
    """Bayesian Beta-Binomial tracker for reliability."""

    alpha: float = _ALPHA_PRIOR
    beta: float = _BETA_PRIOR

    def update_success(self) -> None:
        self.alpha += 1.0

    def update_failure(self) -> None:
        self.beta += 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def confidence(self) -> float:
        """How confident we are in the estimate (0–1)."""
        n = self.alpha + self.beta - _ALPHA_PRIOR - _BETA_PRIOR
        return min(1.0, n / 50.0)  # saturates after 50 observations


@dataclass
class ComponentScore:
    """Full quality score for a component."""

    component_id: str
    capability: ComponentCapability

    # Sub-scores (0.0 – 1.0)
    latency_score: float = 0.5
    confidence_score: float = 0.5
    reliability_tracker: _BetaTracker = field(default_factory=_BetaTracker)
    vram_score: float = 0.5

    # Metadata
    observations: int = 0
    last_updated: float = field(default_factory=time.monotonic)

    # Raw accumulators for normalisation
    _latency_sum: float = 0.0
    _confidence_sum: float = 0.0

    @property
    def reliability_score(self) -> float:
        return self.reliability_tracker.mean

    @property
    def composite(self) -> float:
        """Weighted composite quality score."""
        return (
            _W_LATENCY * self.latency_score
            + _W_CONFIDENCE * self.confidence_score
            + _W_RELIABILITY * self.reliability_score
            + _W_VRAM * self.vram_score
        )

    @property
    def confidence_in_score(self) -> float:
        """How confident we are in this composite score."""
        return self.reliability_tracker.confidence

    def summary(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "capability": self.capability.value,
            "composite_score": round(self.composite, 4),
            "latency_score": round(self.latency_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "reliability_score": round(self.reliability_score, 4),
            "vram_score": round(self.vram_score, 4),
            "observations": self.observations,
            "score_confidence": round(self.confidence_in_score, 3),
        }


class QualityScorer:
    """Multi-dimensional component quality scorer with Bayesian updating.

    Parameters
    ----------
    latency_budget_ms:
        Expected per-stage latency budget for normalisation.
    vram_budget_mb:
        Total VRAM budget for normalisation.
    """

    def __init__(
        self,
        *,
        latency_budget_ms: float = 100.0,
        vram_budget_mb: int = 2400,
    ) -> None:
        self._latency_budget = latency_budget_ms
        self._vram_budget = vram_budget_mb
        self._scores: dict[str, ComponentScore] = {}

    def ensure_component(
        self,
        component_id: str,
        capability: ComponentCapability,
    ) -> ComponentScore:
        """Ensure a score entry exists for the component."""
        if component_id not in self._scores:
            self._scores[component_id] = ComponentScore(
                component_id=component_id,
                capability=capability,
            )
        return self._scores[component_id]

    def record_execution(
        self,
        component_id: str,
        capability: ComponentCapability,
        *,
        latency_ms: float,
        success: bool,
        confidence: float | None = None,
        vram_mb: int | None = None,
    ) -> float:
        """Record an execution and return the updated composite score.

        Returns
        -------
        float
            The updated composite quality score.
        """
        cs = self.ensure_component(component_id, capability)
        cs.observations += 1
        cs.last_updated = time.monotonic()

        # Reliability (Bayesian update)
        if success:
            cs.reliability_tracker.update_success()
        else:
            cs.reliability_tracker.update_failure()

        # Latency score: 1.0 = at or under budget, decays exponentially
        if latency_ms <= 0:
            latency_norm = 1.0
        else:
            ratio = latency_ms / self._latency_budget
            latency_norm = math.exp(-max(0, ratio - 1))
        # EWMA update
        alpha = 0.15
        cs.latency_score = alpha * latency_norm + (1 - alpha) * cs.latency_score

        # Confidence score
        if confidence is not None:
            cs.confidence_score = alpha * confidence + (1 - alpha) * cs.confidence_score

        # VRAM efficiency score
        if vram_mb is not None and vram_mb > 0:
            # Higher quality per MB is better; compute relative to budget
            efficiency = 1.0 - (vram_mb / self._vram_budget)
            cs.vram_score = alpha * max(0.0, efficiency) + (1 - alpha) * cs.vram_score

        return cs.composite

    def get_score(self, component_id: str) -> float | None:
        """Get composite score for a component (or None if unknown)."""
        cs = self._scores.get(component_id)
        return cs.composite if cs else None

    def get_full_score(self, component_id: str) -> dict[str, Any] | None:
        cs = self._scores.get(component_id)
        return cs.summary() if cs else None

    def rank_by_capability(
        self,
        capability: ComponentCapability,
    ) -> list[tuple[str, float]]:
        """Return components for a capability ranked by composite score."""
        items = [(cid, cs.composite) for cid, cs in self._scores.items() if cs.capability == capability]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def get_best_for_capability(
        self,
        capability: ComponentCapability,
        *,
        min_observations: int = 3,
    ) -> str | None:
        """Return the component ID with the highest score for a capability
        that has enough observations."""
        ranked = [
            (cid, cs.composite, cs.observations)
            for cid, cs in self._scores.items()
            if cs.capability == capability and cs.observations >= min_observations
        ]
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[0][0]

    def prune_stale(self, max_age_s: float = 600.0) -> list[str]:
        """Remove scores for components not seen in a while."""
        now = time.monotonic()
        stale = [cid for cid, cs in self._scores.items() if (now - cs.last_updated) > max_age_s]
        for cid in stale:
            del self._scores[cid]
        return stale

    def status(self) -> dict[str, Any]:
        """Full scorer status for monitoring."""
        return {
            "tracked_components": len(self._scores),
            "latency_budget_ms": self._latency_budget,
            "vram_budget_mb": self._vram_budget,
            "scores": {cid: cs.summary() for cid, cs in self._scores.items()},
        }
