"""Component Registry — tracks all known and loaded components.

The registry is the single source of truth for:
  * What components exist (registered — available but not loaded)
  * What components are currently loaded and ready
  * What capabilities are satisfied by loaded components
  * VRAM accounting for loaded GPU components

The Manager Agent queries the registry to decide whether to reuse
existing components or download new ones.
"""

from __future__ import annotations

import logging
import threading

from uni_vision.components.base import (
    ComponentCapability,
    ComponentState,
    CVComponent,
)

log = logging.getLogger(__name__)


class ComponentRegistry:
    """Thread-safe registry of all pipeline components.

    Components progress through states:
      REGISTERED → LOADING → READY → (SUSPENDED | UNLOADING → REGISTERED)

    The registry does NOT manage lifecycle transitions — that's the
    LifecycleManager's job.  The registry is a passive data store
    that the Manager Agent and other subsystems query.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._components: dict[str, CVComponent] = {}

        # Reverse index: capability → set of component_ids that provide it
        self._capability_index: dict[ComponentCapability, set[str]] = {}

    # ── Registration ──────────────────────────────────────────────

    def register(self, component: CVComponent) -> None:
        """Add a component to the registry."""
        cid = component.metadata.component_id
        with self._lock:
            if cid in self._components:
                log.warning("component_already_registered component_id=%s", cid)
                return

            self._components[cid] = component

            # Update capability index
            for cap in component.metadata.capabilities:
                if cap not in self._capability_index:
                    self._capability_index[cap] = set()
                self._capability_index[cap].add(cid)

        log.info(
            "component_registered component_id=%s type=%s capabilities=%s",
            cid,
            component.metadata.component_type.value,
            [c.value for c in component.metadata.capabilities],
        )

    def unregister(self, component_id: str) -> CVComponent | None:
        """Remove a component from the registry entirely."""
        with self._lock:
            component = self._components.pop(component_id, None)
            if component is None:
                return None

            # Clean capability index
            for cap in component.metadata.capabilities:
                cap_set = self._capability_index.get(cap)
                if cap_set:
                    cap_set.discard(component_id)
                    if not cap_set:
                        del self._capability_index[cap]

        log.info("component_unregistered component_id=%s", component_id)
        return component

    # ── Queries ───────────────────────────────────────────────────

    def get(self, component_id: str) -> CVComponent | None:
        """Look up a component by ID."""
        with self._lock:
            return self._components.get(component_id)

    def get_all(self) -> list[CVComponent]:
        """Return all registered components."""
        with self._lock:
            return list(self._components.values())

    def get_loaded(self) -> list[CVComponent]:
        """Return all components in READY state."""
        with self._lock:
            return [c for c in self._components.values() if c.state == ComponentState.READY]

    def get_by_capability(
        self,
        capability: ComponentCapability,
        *,
        only_ready: bool = False,
    ) -> list[CVComponent]:
        """Find components that provide a given capability."""
        with self._lock:
            cids = self._capability_index.get(capability, set())
            components = [self._components[cid] for cid in cids if cid in self._components]

        if only_ready:
            components = [c for c in components if c.state == ComponentState.READY]

        return components

    def has_capability_loaded(self, capability: ComponentCapability) -> bool:
        """Check if any READY component provides this capability."""
        return len(self.get_by_capability(capability, only_ready=True)) > 0

    def get_missing_capabilities(
        self,
        required: set[ComponentCapability],
    ) -> set[ComponentCapability]:
        """Return capabilities not satisfied by any READY component."""
        return {cap for cap in required if not self.has_capability_loaded(cap)}

    # ── VRAM accounting ───────────────────────────────────────────

    def get_loaded_vram_mb(self) -> int:
        """Total VRAM used by all READY components (estimated)."""
        with self._lock:
            return sum(
                c.metadata.resource_estimate.vram_mb
                for c in self._components.values()
                if c.state == ComponentState.READY
            )

    def get_component_vram_mb(self, component_id: str) -> int:
        """VRAM estimate for a specific component."""
        comp = self.get(component_id)
        if comp is None:
            return 0
        return comp.metadata.resource_estimate.vram_mb

    # ── Summary / serialisation ───────────────────────────────────

    def summary(self) -> list[dict]:
        """Return a JSON-serialisable summary for LLM prompts."""
        with self._lock:
            return [
                {
                    "component_id": c.metadata.component_id,
                    "name": c.metadata.name,
                    "type": c.metadata.component_type.value,
                    "source": c.metadata.source,
                    "capabilities": [cap.value for cap in c.metadata.capabilities],
                    "vram_mb": c.metadata.resource_estimate.vram_mb,
                    "state": c.state.value,
                    "trusted": c.metadata.trusted,
                }
                for c in self._components.values()
            ]

    def loaded_summary(self) -> list[dict]:
        """Summary of currently loaded (READY) components only."""
        return [entry for entry in self.summary() if entry["state"] == "ready"]

    def __len__(self) -> int:
        with self._lock:
            return len(self._components)

    def __contains__(self, component_id: str) -> bool:
        with self._lock:
            return component_id in self._components
