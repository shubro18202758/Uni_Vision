"""Fallback Chain — ordered component alternatives per capability.

When a primary component fails or degrades, the fallback chain
provides a pre-ranked list of alternatives to try.  Chains are
built from the component registry + HuggingFace Hub search results
and continuously re-ranked by the quality scorer.

Design:
  * Each capability maps to an ordered list of component candidates.
  * Primary → Secondary → Tertiary chains with automatic promotion/demotion.
  * Graceful degradation: when all GPU options exhaust, fall back to CPU.
  * Chain state is persisted in-memory (no DB); rebuilt on startup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from uni_vision.components.base import ComponentCapability
    from uni_vision.manager.schemas import ComponentCandidate

log = structlog.get_logger(__name__)


class FallbackTier(str, Enum):
    """Tier in the fallback chain."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    EMERGENCY = "emergency"  # CPU-only last resort


@dataclass
class FallbackEntry:
    """A single entry in a fallback chain."""

    candidate: ComponentCandidate
    tier: FallbackTier
    score: float = 0.5  # quality score (0.0 – 1.0)
    consecutive_failures: int = 0
    total_uses: int = 0
    total_failures: int = 0
    last_used: float = 0.0
    disabled: bool = False
    disable_reason: str = ""

    @property
    def reliability(self) -> float:
        if self.total_uses == 0:
            return 0.5  # prior
        return 1.0 - (self.total_failures / self.total_uses)

    @property
    def effective_score(self) -> float:
        """Score adjusted by reliability and failure streaks."""
        penalty = min(self.consecutive_failures * 0.15, 0.6)
        return max(0.0, self.score * self.reliability - penalty)


@dataclass
class FallbackChain:
    """Ordered fallback chain for a single capability."""

    capability: ComponentCapability
    entries: list[FallbackEntry] = field(default_factory=list)

    def active_entries(self) -> list[FallbackEntry]:
        """Return enabled entries sorted by effective score descending."""
        active = [e for e in self.entries if not e.disabled]
        active.sort(key=lambda e: e.effective_score, reverse=True)
        return active

    @property
    def primary(self) -> FallbackEntry | None:
        """Current best active entry."""
        active = self.active_entries()
        return active[0] if active else None

    @property
    def depth(self) -> int:
        return len([e for e in self.entries if not e.disabled])


class FallbackChainManager:
    """Manages fallback chains for all capabilities.

    Parameters
    ----------
    max_consecutive_failures:
        Number of consecutive failures before disabling a component.
    auto_recover_after_s:
        Seconds after which a disabled component is re-enabled for probing.
    """

    def __init__(
        self,
        *,
        max_consecutive_failures: int = 5,
        auto_recover_after_s: float = 120.0,
    ) -> None:
        self._chains: dict[ComponentCapability, FallbackChain] = {}
        self._max_failures = max_consecutive_failures
        self._auto_recover_s = auto_recover_after_s

    def register_candidate(
        self,
        capability: ComponentCapability,
        candidate: ComponentCandidate,
        *,
        tier: FallbackTier = FallbackTier.SECONDARY,
        initial_score: float = 0.5,
    ) -> None:
        """Add a component candidate to the fallback chain."""
        if capability not in self._chains:
            self._chains[capability] = FallbackChain(capability=capability)

        chain = self._chains[capability]

        # Avoid duplicates
        for entry in chain.entries:
            if entry.candidate.component_id == candidate.component_id:
                return

        chain.entries.append(
            FallbackEntry(
                candidate=candidate,
                tier=tier,
                score=initial_score,
            )
        )
        log.debug(
            "fallback_registered",
            capability=capability.value,
            component=candidate.component_id,
            tier=tier.value,
        )

    def get_next_fallback(
        self,
        capability: ComponentCapability,
        *,
        exclude: set[str] | None = None,
    ) -> ComponentCandidate | None:
        """Get the next best fallback candidate for a capability.

        Parameters
        ----------
        exclude:
            Component IDs to skip (e.g. the one that just failed).
        """
        self._try_auto_recover(capability)
        chain = self._chains.get(capability)
        if chain is None:
            return None

        exclude = exclude or set()
        for entry in chain.active_entries():
            if entry.candidate.component_id not in exclude:
                return entry.candidate
        return None

    def record_success(self, capability: ComponentCapability, component_id: str) -> None:
        """Record a successful use of a component."""
        entry = self._find_entry(capability, component_id)
        if entry is None:
            return
        entry.total_uses += 1
        entry.consecutive_failures = 0
        entry.last_used = time.monotonic()

    def record_failure(self, capability: ComponentCapability, component_id: str) -> None:
        """Record a failure and potentially disable the component."""
        entry = self._find_entry(capability, component_id)
        if entry is None:
            return
        entry.total_uses += 1
        entry.total_failures += 1
        entry.consecutive_failures += 1
        entry.last_used = time.monotonic()

        if entry.consecutive_failures >= self._max_failures:
            entry.disabled = True
            entry.disable_reason = f"Disabled after {entry.consecutive_failures} consecutive failures"
            log.warning(
                "fallback_disabled",
                capability=capability.value,
                component=component_id,
                reason=entry.disable_reason,
            )

    def update_score(
        self,
        capability: ComponentCapability,
        component_id: str,
        new_score: float,
    ) -> None:
        """Update the quality score for a component (from quality scorer)."""
        entry = self._find_entry(capability, component_id)
        if entry is not None:
            entry.score = max(0.0, min(1.0, new_score))

    def get_chain_status(self, capability: ComponentCapability) -> dict[str, Any]:
        """Return the status of a fallback chain."""
        chain = self._chains.get(capability)
        if chain is None:
            return {"status": "no_chain"}
        return {
            "capability": capability.value,
            "total_entries": len(chain.entries),
            "active_entries": chain.depth,
            "primary": chain.primary.candidate.component_id if chain.primary else None,
            "entries": [
                {
                    "component_id": e.candidate.component_id,
                    "tier": e.tier.value,
                    "score": round(e.score, 3),
                    "effective_score": round(e.effective_score, 3),
                    "reliability": round(e.reliability, 3),
                    "consecutive_failures": e.consecutive_failures,
                    "disabled": e.disabled,
                }
                for e in chain.entries
            ],
        }

    def promote_demote(self, capability: ComponentCapability) -> None:
        """Re-assign tiers based on current effective scores."""
        chain = self._chains.get(capability)
        if chain is None:
            return

        active = chain.active_entries()
        tiers = [FallbackTier.PRIMARY, FallbackTier.SECONDARY, FallbackTier.TERTIARY]
        for i, entry in enumerate(active):
            if i < len(tiers):
                entry.tier = tiers[i]
            else:
                entry.tier = FallbackTier.EMERGENCY

    def status(self) -> dict[str, Any]:
        return {
            "chains": len(self._chains),
            "total_candidates": sum(len(c.entries) for c in self._chains.values()),
            "capabilities": {
                cap.value: {
                    "depth": chain.depth,
                    "primary": chain.primary.candidate.component_id if chain.primary else None,
                }
                for cap, chain in self._chains.items()
            },
        }

    # ── Internal ──────────────────────────────────────────────

    def _find_entry(self, capability: ComponentCapability, component_id: str) -> FallbackEntry | None:
        chain = self._chains.get(capability)
        if chain is None:
            return None
        for entry in chain.entries:
            if entry.candidate.component_id == component_id:
                return entry
        return None

    def _try_auto_recover(self, capability: ComponentCapability) -> None:
        """Re-enable components that have been disabled long enough."""
        chain = self._chains.get(capability)
        if chain is None:
            return
        now = time.monotonic()
        for entry in chain.entries:
            if entry.disabled and (now - entry.last_used) > self._auto_recover_s:
                entry.disabled = False
                entry.consecutive_failures = 0
                entry.disable_reason = ""
                log.info(
                    "fallback_auto_recovered",
                    capability=capability.value,
                    component=entry.candidate.component_id,
                )
