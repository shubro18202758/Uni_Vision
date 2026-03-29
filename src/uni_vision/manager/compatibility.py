"""Component Compatibility Matrix — tracks inter-component compatibility.

Maintains an empirical record of which components work well together
and which cause conflicts (Python package clashes, CUDA version mismatches,
tensor format incompatibilities, etc.).

The matrix is built from:
  1. **Static rules** — known incompatibilities declared upfront.
  2. **Runtime observations** — recorded when components actually co-load or fail.

This data is consumed by:
  * **Pipeline Composer** — avoids composing incompatible stages.
  * **Conflict Resolver** — provides evidence for conflict decisions.
  * **Component Resolver** — filters candidates by compatibility.
"""

from __future__ import annotations

import structlog
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

log = structlog.get_logger(__name__)


class CompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"
    CONDITIONAL = "conditional"  # works under certain conditions


@dataclass
class CompatibilityRecord:
    """A record of compatibility between two components."""

    component_a: str
    component_b: str
    status: CompatibilityStatus
    reason: str = ""
    confidence: float = 0.5  # 0.0–1.0, increases with observations
    observations: int = 0
    last_updated: float = field(default_factory=time.monotonic)

    @property
    def pair_key(self) -> FrozenSet[str]:
        return frozenset([self.component_a, self.component_b])


@dataclass
class PackageConflict:
    """A known Python package version conflict."""

    package_a: str
    version_a: str
    package_b: str
    version_b: str
    conflict_type: str  # "version_clash" | "abi_mismatch" | "cuda_version"
    description: str = ""


class CompatibilityMatrix:
    """Component compatibility tracker.

    Parameters
    ----------
    conflict_threshold:
        Number of co-load failures before marking as incompatible.
    """

    def __init__(self, *, conflict_threshold: int = 3) -> None:
        self._records: Dict[FrozenSet[str], CompatibilityRecord] = {}
        self._package_conflicts: List[PackageConflict] = []
        self._conflict_threshold = conflict_threshold

        # Static incompatibilities (known from the ecosystem)
        self._init_static_rules()

    def check(self, component_a: str, component_b: str) -> CompatibilityStatus:
        """Check compatibility between two components."""
        key = frozenset([component_a, component_b])
        record = self._records.get(key)
        if record is None:
            return CompatibilityStatus.UNKNOWN
        return record.status

    def is_compatible(self, component_a: str, component_b: str) -> bool:
        """Return True if the pair is compatible or unknown."""
        status = self.check(component_a, component_b)
        return status in (CompatibilityStatus.COMPATIBLE, CompatibilityStatus.UNKNOWN)

    def check_set(self, component_ids: Set[str]) -> List[Tuple[str, str, CompatibilityStatus]]:
        """Check all pairwise compatibilities in a set of components.

        Returns a list of incompatible or conditional pairs.
        """
        issues = []
        comp_list = sorted(component_ids)
        for i, a in enumerate(comp_list):
            for b in comp_list[i + 1:]:
                status = self.check(a, b)
                if status in (CompatibilityStatus.INCOMPATIBLE, CompatibilityStatus.CONDITIONAL):
                    issues.append((a, b, status))
        return issues

    def record_success(self, component_a: str, component_b: str) -> None:
        """Record that two components successfully co-loaded."""
        key = frozenset([component_a, component_b])
        rec = self._records.get(key)
        if rec is None:
            rec = CompatibilityRecord(
                component_a=component_a,
                component_b=component_b,
                status=CompatibilityStatus.COMPATIBLE,
            )
            self._records[key] = rec
        rec.observations += 1
        rec.confidence = min(1.0, rec.confidence + 0.05)
        rec.last_updated = time.monotonic()
        # If was unknown or conditional, upgrade to compatible with enough data
        if rec.status == CompatibilityStatus.UNKNOWN and rec.observations >= 3:
            rec.status = CompatibilityStatus.COMPATIBLE

    def record_failure(self, component_a: str, component_b: str, reason: str = "") -> None:
        """Record that two components failed to co-load."""
        key = frozenset([component_a, component_b])
        rec = self._records.get(key)
        if rec is None:
            rec = CompatibilityRecord(
                component_a=component_a,
                component_b=component_b,
                status=CompatibilityStatus.UNKNOWN,
            )
            self._records[key] = rec
        rec.observations += 1
        rec.reason = reason or rec.reason
        rec.last_updated = time.monotonic()

        # Count failures  
        # Use a simple heuristic: if confidence was going up, failures bring it down
        rec.confidence = max(0.0, rec.confidence - 0.15)

        # Mark incompatible after enough failures
        if rec.confidence < 0.2 and rec.observations >= self._conflict_threshold:
            rec.status = CompatibilityStatus.INCOMPATIBLE
            log.warning(
                "components_marked_incompatible",
                a=component_a,
                b=component_b,
                reason=reason,
            )

    def declare_incompatible(
        self,
        component_a: str,
        component_b: str,
        reason: str,
    ) -> None:
        """Explicitly declare two components as incompatible."""
        key = frozenset([component_a, component_b])
        self._records[key] = CompatibilityRecord(
            component_a=component_a,
            component_b=component_b,
            status=CompatibilityStatus.INCOMPATIBLE,
            reason=reason,
            confidence=1.0,
        )

    def declare_compatible(
        self,
        component_a: str,
        component_b: str,
    ) -> None:
        """Explicitly declare two components as compatible."""
        key = frozenset([component_a, component_b])
        self._records[key] = CompatibilityRecord(
            component_a=component_a,
            component_b=component_b,
            status=CompatibilityStatus.COMPATIBLE,
            confidence=1.0,
        )

    def add_package_conflict(self, conflict: PackageConflict) -> None:
        """Register a known package-level conflict."""
        self._package_conflicts.append(conflict)

    def get_incompatible_with(self, component_id: str) -> List[str]:
        """Return all components known to be incompatible with the given one."""
        result = []
        for key, rec in self._records.items():
            if component_id in key and rec.status == CompatibilityStatus.INCOMPATIBLE:
                for other in key:
                    if other != component_id:
                        result.append(other)
        return result

    def status(self) -> Dict[str, Any]:
        compatible = sum(
            1 for r in self._records.values()
            if r.status == CompatibilityStatus.COMPATIBLE
        )
        incompatible = sum(
            1 for r in self._records.values()
            if r.status == CompatibilityStatus.INCOMPATIBLE
        )
        return {
            "total_pairs": len(self._records),
            "compatible": compatible,
            "incompatible": incompatible,
            "package_conflicts": len(self._package_conflicts),
            "incompatible_pairs": [
                {
                    "a": rec.component_a,
                    "b": rec.component_b,
                    "reason": rec.reason,
                }
                for rec in self._records.values()
                if rec.status == CompatibilityStatus.INCOMPATIBLE
            ],
        }

    # ── Static rules ──────────────────────────────────────────

    def _init_static_rules(self) -> None:
        """Declare known static incompatibilities."""
        # Example: different ONNX runtime builds can conflict
        self.add_package_conflict(PackageConflict(
            package_a="onnxruntime-gpu",
            version_a="*",
            package_b="onnxruntime",
            version_b="*",
            conflict_type="version_clash",
            description="Cannot install both onnxruntime and onnxruntime-gpu",
        ))
        # TensorRT and standard ONNX often clash
        self.add_package_conflict(PackageConflict(
            package_a="tensorrt",
            version_a=">=8",
            package_b="onnxruntime-gpu",
            version_b="<1.16",
            conflict_type="abi_mismatch",
            description="TensorRT 8+ requires onnxruntime-gpu >= 1.16",
        ))
