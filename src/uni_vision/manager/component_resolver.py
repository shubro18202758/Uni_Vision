"""Component Resolver — discovers, downloads, and provisions components.

Given a set of required capabilities, the resolver:
  1. Checks the local ComponentRegistry for existing components.
  2. For missing capabilities, uses LLM-guided open internet search OR
     falls back to curated seed knowledge.
  3. Has the LLM evaluate and select the best candidate.
  4. Downloads and installs the best candidate.
  5. Wraps it in the appropriate CVComponent wrapper and registers it.
  6. Calls lifecycle.load_component() to bring the component to READY state.

The hardcoded package lists are SEED KNOWLEDGE (performance shortcuts) —
the LLM can discover and onboard ANY component from the open internet.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from uni_vision.components.base import ComponentCapability, ComponentState
from uni_vision.components.wrappers import (
    HuggingFaceModelComponent,
    PipPackageComponent,
    TorchHubComponent,
)
from uni_vision.manager.component_registry import ComponentRegistry
from uni_vision.manager.hub_client import HubClient
from uni_vision.manager.schemas import (
    ComponentCandidate,
    DiscoveryQuery,
    ResolutionResult,
    ResolutionStatus,
)

if TYPE_CHECKING:
    from uni_vision.manager.lifecycle import ComponentLifecycle
    from uni_vision.manager.dependency_resolver import DependencyConflictResolver

log = logging.getLogger(__name__)


# Well-known pip packages for common CV capabilities (SEED KNOWLEDGE —
# these are performance shortcuts, NOT the upper bound.  The LLM can
# discover any package from the open internet via open_search.)
_KNOWN_PIP_PACKAGES: Dict[ComponentCapability, Dict] = {
    ComponentCapability.PLATE_OCR: {
        "package": "paddleocr",
        "name": "PaddleOCR",
        "source_id": "paddleocr",
        "vram_mb": 400,
        "model_class": "paddleocr.PaddleOCR",
        "requirements": ["paddleocr", "paddlepaddle"],
    },
    ComponentCapability.TRACKING: {
        "package": "deep_sort_realtime",
        "name": "DeepSort",
        "source_id": "deep_sort_realtime",
        "vram_mb": 250,
        "model_class": "deep_sort_realtime.deepsort_tracker.DeepSort",
        "requirements": ["deep-sort-realtime"],
    },
    ComponentCapability.SCENE_CLASSIFICATION: {
        "package": "timm",
        "name": "timm-classifier",
        "source_id": "timm",
        "vram_mb": 300,
        "model_class": "timm.create_model",
        "requirements": ["timm"],
        "load_pattern": "create_model",
    },
    ComponentCapability.IMAGE_ENHANCE: {
        "package": "kornia",
        "name": "Kornia Enhancement",
        "source_id": "kornia",
        "vram_mb": 100,
        "model_class": "kornia.enhance.AdjustBrightness",
        "requirements": ["kornia"],
    },
    ComponentCapability.IMAGE_DENOISE: {
        "package": "albumentations",
        "name": "Albumentations",
        "source_id": "albumentations",
        "vram_mb": 0,
        "model_class": "albumentations.Compose",
        "requirements": ["albumentations"],
    },
    ComponentCapability.ZERO_SHOT_DETECTION: {
        "package": "open-clip-torch",
        "name": "OpenCLIP",
        "source_id": "open-clip-torch",
        "vram_mb": 600,
        "model_class": "open_clip.create_model_and_transforms",
        "requirements": ["open-clip-torch"],
    },
}


class ComponentResolver:
    """Resolve capability requirements to concrete, installable components.

    Supports BOTH standard enum-based capabilities and free-form dynamic
    capabilities discovered by the LLM.  For standard capabilities, seed
    knowledge provides fast initial lookups; for dynamic capabilities (or
    when the LLM is available), open internet search discovers components
    that go far beyond any curated list.

    Parameters
    ----------
    registry:
        The component registry to check for existing components.
    hub_client:
        Client for searching HuggingFace Hub, PyPI, TorchHub, and GitHub.
    llm_client:
        Optional async LLM client for generating search queries and
        evaluating candidates.  When provided, enables true LLM-driven
        discovery beyond curated lists.
    lifecycle:
        Optional lifecycle manager. When provided, ``provision_candidate``
        will also *load* the component after registering it.
    vram_limit_mb:
        Hard VRAM ceiling.  Candidates exceeding this are deprioritised.
    prefer_trusted:
        If True, trusted sources score higher.
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        hub_client: HubClient,
        *,
        llm_client: Optional[Any] = None,
        lifecycle: Optional["ComponentLifecycle"] = None,
        dependency_resolver: Optional["DependencyConflictResolver"] = None,
        vram_limit_mb: int = 8192,
        prefer_trusted: bool = True,
    ) -> None:
        self._registry = registry
        self._hub = hub_client
        self._llm = llm_client
        self._lifecycle = lifecycle
        self._dep_resolver = dependency_resolver
        self._vram_limit = vram_limit_mb
        self._prefer_trusted = prefer_trusted
        # Cache failed resolutions to avoid repeated network calls
        self._failed_caps: dict[ComponentCapability, "ResolutionResult"] = {}

    # ── Public API ────────────────────────────────────────────────

    async def resolve_capabilities(
        self,
        required: Set[ComponentCapability],
        *,
        available_vram_mb: Optional[int] = None,
    ) -> List[ResolutionResult]:
        """Resolve each required capability to a component.

        Returns a ResolutionResult for every capability in *required*.
        """
        vram_budget = available_vram_mb or self._vram_limit
        results: List[ResolutionResult] = []

        for cap in required:
            # Return cached result immediately (avoids repeated network calls
            # and repeated hub searches for capabilities already resolved)
            if cap in self._failed_caps:
                results.append(self._failed_caps[cap])
                continue

            result = await self._resolve_single(cap, vram_budget)
            results.append(result)

            # Cache ALL resolution results (including FOUND) — provisioning
            # failures are cached separately in ManagerAgent
            self._failed_caps[cap] = result

            # Account for selected component VRAM
            if result.selected_candidate and result.status in (
                ResolutionStatus.FOUND,
                ResolutionStatus.INSTALLED,
            ):
                vram_budget -= result.selected_candidate.vram_mb

        return results

    async def provision_candidate(
        self,
        candidate: ComponentCandidate,
    ) -> Optional[object]:
        """Download, install, wrap, register, and *load* a candidate.

        Returns the CVComponent instance in READY state, or None on failure.
        """
        log.info(
            "provisioning_component component_id=%s source=%s",
            candidate.component_id,
            candidate.source,
        )

        # Install Python requirements
        installed_reqs: list[str] = []
        for req in candidate.python_requirements:
            ok = await self._hub.install_package(req, verify_import=True)
            if not ok:
                log.error("provision_dep_failed package=%s", req)
                return None
            installed_reqs.append(req)

        # Run dependency conflict check & auto-resolve after all installs
        if installed_reqs and self._dep_resolver is not None:
            dep_report = await self._dep_resolver.check_and_resolve(installed_reqs)
            if not dep_report.resolved and dep_report.final_conflicts:
                log.error(
                    "provision_dependency_conflicts_unresolved",
                    component_id=candidate.component_id,
                    conflicts=dep_report.final_conflicts,
                )
                # Do NOT abort — the component may still load fine.
                # The conflicts are logged for observability.

        # Download model weights if HuggingFace source
        if candidate.source == "huggingface" and candidate.source_id:
            dl_ok = await self._hub.download_model(candidate.source_id)
            if not dl_ok:
                log.warning("model_download_failed source_id=%s (will try loading anyway)",
                            candidate.source_id)

        # Create appropriate wrapper
        component = self._create_wrapper(candidate)
        if component is None:
            return None

        # Register it
        self._registry.register(component)
        log.info("component_provisioned component_id=%s", candidate.component_id)

        # Load it (bring to READY state) if lifecycle is available
        if self._lifecycle is not None:
            try:
                await self._lifecycle.load_component(candidate.component_id)
                log.info("component_loaded component_id=%s", candidate.component_id)
            except Exception as exc:
                log.error("component_load_failed component_id=%s error=%s",
                          candidate.component_id, exc)
                return None

        return component

    def _create_wrapper(self, candidate: ComponentCandidate) -> Optional[object]:
        """Instantiate the correct CVComponent wrapper for a candidate."""
        metadata = candidate.metadata or {}
        load_pattern = metadata.get("load_pattern", "from_pretrained")
        hub_repo = metadata.get("hub_repo", "")

        if candidate.source == "torchhub":
            # TorchHub models: hub_repo is e.g. "ultralytics/yolov5",
            # model name is extracted from model_class or source_id
            parts = candidate.source_id.split(":", 1)
            repo = parts[0] if len(parts) > 1 else hub_repo or candidate.source_id
            model_name = parts[1] if len(parts) > 1 else candidate.model_class.rsplit(".", 1)[-1]
            return TorchHubComponent(
                component_id=candidate.component_id,
                name=candidate.name,
                hub_repo=repo,
                model_name=model_name,
                capabilities=candidate.capabilities,
                vram_mb=candidate.vram_mb,
                python_requirements=candidate.python_requirements,
            )

        if candidate.source == "huggingface":
            return HuggingFaceModelComponent(
                component_id=candidate.component_id,
                name=candidate.name,
                repo_id=candidate.source_id,
                capabilities=candidate.capabilities,
                model_class=candidate.model_class,
                vram_mb=candidate.vram_mb,
                python_requirements=candidate.python_requirements,
                load_pattern=load_pattern,
                hub_repo=hub_repo or None,
            )

        # PyPI / other — use PipPackageComponent
        return PipPackageComponent(
            component_id=candidate.component_id,
            name=candidate.name,
            package_name=candidate.source_id,
            capabilities=candidate.capabilities,
            entry_class=candidate.model_class,  # auto-splits dotted path
            vram_mb=candidate.vram_mb,
        )

    async def resolve_dynamic_capabilities(
        self,
        dynamic_caps: Set[str],
        discovery_queries: Optional[List[DiscoveryQuery]] = None,
        *,
        available_vram_mb: Optional[int] = None,
    ) -> List[ResolutionResult]:
        """Resolve free-form capability strings discovered by the LLM.

        Unlike ``resolve_capabilities`` (which works with the fixed
        ``ComponentCapability`` enum), this method handles arbitrary string
        capability labels that the LLM invented during open-ended analysis.

        Parameters
        ----------
        dynamic_caps:
            Set of free-form capability labels (e.g. ``"shadow_removal"``,
            ``"infrared_enhancement"``).
        discovery_queries:
            Optional pre-generated search queries from the LLM.  If provided,
            these are executed directly via ``hub_client.open_search()``.
            If not provided and an LLM client is available, the resolver
            will ask the LLM to generate queries on the fly.
        available_vram_mb:
            Current VRAM budget.
        """
        vram_budget = available_vram_mb or self._vram_limit
        results: List[ResolutionResult] = []

        # Build a lookup from capability label → its discovery queries
        query_map: Dict[str, List[DiscoveryQuery]] = {}
        for dq in (discovery_queries or []):
            hint = dq.capability_hint or ""
            query_map.setdefault(hint, []).append(dq)

        for cap_label in dynamic_caps:
            result = await self._resolve_dynamic_single(
                cap_label,
                query_map.get(cap_label, []),
                vram_budget,
            )
            results.append(result)

            if result.selected_candidate and result.status in (
                ResolutionStatus.FOUND,
                ResolutionStatus.INSTALLED,
            ):
                vram_budget -= result.selected_candidate.vram_mb

        return results

    # ── Internals ─────────────────────────────────────────────────

    async def _resolve_single(
        self,
        capability: ComponentCapability,
        vram_budget: int,
    ) -> ResolutionResult:
        """Resolve one standard enum capability.

        Resolution order:
          1. Local registry (already loaded)
          2. Seed knowledge (curated pip packages — fast path)
          3. LLM-guided open internet search (when LLM client available)
          4. Broad HuggingFace / PyPI / TorchHub search (fallback)
          5. LLM evaluates and picks the best candidate
        """

        # 1. Check registry for existing ready or registered component
        existing = self._registry.get_by_capability(capability)
        if existing:
            best = sorted(existing, key=lambda c: c.state == ComponentState.READY, reverse=True)[0]
            return ResolutionResult(
                capability=capability,
                status=ResolutionStatus.FOUND,
                selected_candidate=self._component_to_candidate(best),
                all_candidates=[self._component_to_candidate(c) for c in existing],
            )

        # 2. Check well-known pip packages (SEED KNOWLEDGE — fast path)
        candidates: List[ComponentCandidate] = []
        known = _KNOWN_PIP_PACKAGES.get(capability)
        if known:
            candidates.append(ComponentCandidate(
                component_id=f"pip.{known['source_id']}",
                name=known["name"],
                source="pypi",
                source_id=known["source_id"],
                capabilities={capability},
                vram_mb=known["vram_mb"],
                score=0.8,
                python_requirements=known["requirements"],
                model_class=known["model_class"],
                trusted=True,
                metadata={"load_pattern": known.get("load_pattern", "from_pretrained")},
            ))

        # 3. LLM-guided open internet search (goes beyond seed knowledge)
        if self._llm is not None:
            llm_candidates = await self._llm_open_search(
                capability.value, vram_budget,
            )
            for c in llm_candidates:
                if not any(x.source_id == c.source_id for x in candidates):
                    candidates.append(c)

        # 4. Standard broad search (HuggingFace, PyPI curated, TorchHub)
        query = capability.value.replace("_", " ")
        hf_results = await self._hub.search_models(
            query,
            capability=capability,
            max_results=5,
        )
        for info in hf_results:
            cand = self._hub.hf_model_to_candidate(
                info,
                capabilities={capability},
                vram_mb=self._estimate_vram(info.model_id),
            )
            if not any(x.source_id == cand.source_id for x in candidates):
                candidates.append(cand)

        pypi_results = await self._hub.search_pypi(query, capability=capability)
        for pypi_cand in pypi_results:
            if not any(c.source_id == pypi_cand.source_id for c in candidates):
                candidates.append(pypi_cand)

        torch_results = await self._hub.search_torchhub(query, capability=capability)
        for th_cand in torch_results:
            if not any(c.source_id == th_cand.source_id for c in candidates):
                candidates.append(th_cand)

        if not candidates:
            return ResolutionResult(
                capability=capability,
                status=ResolutionStatus.FAILED,
                error=f"No candidates found for {capability.value}",
            )

        # 5. LLM evaluates and picks the best (or simple ranking fallback)
        if self._llm is not None and len(candidates) > 1:
            best = await self._llm_evaluate_candidates(
                candidates, capability.value, vram_budget,
            )
            if best is not None:
                ranked = [best] + [c for c in candidates if c.source_id != best.source_id]
                return ResolutionResult(
                    capability=capability,
                    status=ResolutionStatus.FOUND,
                    selected_candidate=ranked[0],
                    all_candidates=ranked,
                )

        # Fallback: simple scoring
        ranked = self._rank_candidates(candidates, vram_budget)

        return ResolutionResult(
            capability=capability,
            status=ResolutionStatus.FOUND,
            selected_candidate=ranked[0],
            all_candidates=ranked,
        )

    async def _resolve_dynamic_single(
        self,
        cap_label: str,
        queries: List[DiscoveryQuery],
        vram_budget: int,
    ) -> ResolutionResult:
        """Resolve a single free-form dynamic capability label.

        Uses LLM-generated DiscoveryQuery objects to drive open internet
        search, then has the LLM evaluate candidates.
        """
        candidates: List[ComponentCandidate] = []

        # Execute pre-generated discovery queries from context analysis
        for dq in queries:
            try:
                found = await self._hub.open_search(
                    query=dq.query,
                    source=dq.source,
                    vram_budget_mb=dq.vram_budget_mb or vram_budget,
                    max_results=5,
                )
                candidates.extend(found)
            except Exception as exc:
                log.warning(
                    "discovery_query_failed query=%r source=%s error=%s",
                    dq.query, dq.source, exc,
                )

        # If no pre-generated queries, use the LLM to generate them
        if not queries and self._llm is not None:
            llm_candidates = await self._llm_open_search(
                cap_label, vram_budget,
            )
            candidates.extend(llm_candidates)

        # Also try a direct broad search using the label as query text
        if not candidates:
            broad_query = cap_label.replace("_", " ")
            for source in ("huggingface", "pypi", "github"):
                try:
                    found = await self._hub.open_search(
                        query=broad_query,
                        source=source,
                        vram_budget_mb=vram_budget,
                        max_results=3,
                    )
                    candidates.extend(found)
                except Exception:
                    pass

        # Deduplicate by source_id
        seen: Set[str] = set()
        unique: List[ComponentCandidate] = []
        for c in candidates:
            if c.source_id not in seen:
                seen.add(c.source_id)
                unique.append(c)
        candidates = unique

        if not candidates:
            return ResolutionResult(
                capability=None,
                status=ResolutionStatus.FAILED,
                error=f"No candidates found for dynamic capability '{cap_label}'",
            )

        # LLM evaluates and picks the best
        if self._llm is not None and len(candidates) > 1:
            best = await self._llm_evaluate_candidates(
                candidates, cap_label, vram_budget,
            )
            if best is not None:
                ranked = [best] + [c for c in candidates if c.source_id != best.source_id]
                return ResolutionResult(
                    capability=None,
                    status=ResolutionStatus.FOUND,
                    selected_candidate=ranked[0],
                    all_candidates=ranked,
                )

        ranked = self._rank_candidates(candidates, vram_budget)
        return ResolutionResult(
            capability=None,
            status=ResolutionStatus.FOUND,
            selected_candidate=ranked[0],
            all_candidates=ranked,
        )

    # ── LLM-driven helpers ────────────────────────────────────────

    async def _llm_open_search(
        self,
        capability_label: str,
        vram_budget: int,
    ) -> List[ComponentCandidate]:
        """Ask the LLM what to search for, then execute the searches.

        The LLM generates structured search queries based on the capability
        description, and we fan out across HuggingFace, PyPI, and GitHub.
        """
        if self._llm is None:
            return []

        prompt = (
            f"I need to find a Python package or ML model for this "
            f"computer-vision capability: '{capability_label}'.\n"
            f"Available VRAM: {vram_budget} MB.\n"
            f"Generate 2-3 search queries I should use on HuggingFace Hub, "
            f"PyPI, and GitHub to find the best component.  Return JSON:\n"
            f'{{"queries": [{{"query": "...", "source": "huggingface|pypi|github"}}]}}'
        )

        try:
            raw = await self._llm.generate(prompt)
            parsed = json.loads(raw)
            search_queries = parsed.get("queries", [])
        except Exception as exc:
            log.warning("llm_search_query_gen_failed error=%s", exc)
            # Fallback: use the label directly
            search_queries = [
                {"query": capability_label.replace("_", " "), "source": "huggingface"},
                {"query": capability_label.replace("_", " "), "source": "pypi"},
            ]

        candidates: List[ComponentCandidate] = []
        seen_ids: Set[str] = set()

        for sq in search_queries:
            try:
                results = await self._hub.open_search(
                    query=sq.get("query", capability_label),
                    source=sq.get("source", "huggingface"),
                    vram_budget_mb=vram_budget,
                    max_results=5,
                )
                for c in results:
                    if c.source_id not in seen_ids:
                        seen_ids.add(c.source_id)
                        candidates.append(c)
            except Exception as exc:
                log.debug("llm_search_execution_failed query=%r error=%s",
                          sq, exc)

        return candidates

    async def _llm_evaluate_candidates(
        self,
        candidates: List[ComponentCandidate],
        capability_label: str,
        vram_budget: int,
    ) -> Optional[ComponentCandidate]:
        """Ask the LLM to evaluate candidates and pick the best one.

        Uses the CANDIDATE_EVALUATION_PROMPT to have Gemma 4 E2B reason about
        relevance, VRAM fit, trust, quality, efficiency, and compatibility.
        """
        if self._llm is None:
            return None

        from uni_vision.manager.prompts import CANDIDATE_EVALUATION_PROMPT

        candidates_json = json.dumps([
            {
                "id": c.component_id,
                "name": c.name,
                "source": c.source,
                "source_id": c.source_id,
                "vram_mb": c.vram_mb,
                "score": c.score,
                "trusted": c.trusted,
                "model_class": c.model_class,
            }
            for c in candidates
        ], indent=2)

        prompt = CANDIDATE_EVALUATION_PROMPT.format(
            capability_description=capability_label,
            frame_context="current pipeline frame",
            vram_available_mb=vram_budget,
            candidates_json=candidates_json,
        )

        try:
            raw = await self._llm.generate(prompt)
            parsed = json.loads(raw)
            chosen_id = parsed.get("selected_component_id", "")

            for c in candidates:
                if c.component_id == chosen_id:
                    log.info(
                        "llm_selected_candidate capability=%s chosen=%s reason=%s",
                        capability_label, chosen_id,
                        parsed.get("reasoning", ""),
                    )
                    return c

            log.debug("llm_selected_unknown_id id=%s", chosen_id)
        except Exception as exc:
            log.warning("llm_candidate_eval_failed error=%s", exc)

        return None

    def _rank_candidates(
        self,
        candidates: List[ComponentCandidate],
        vram_budget: int,
    ) -> List[ComponentCandidate]:
        """Score and rank candidates.  Higher = better."""

        def score(c: ComponentCandidate) -> float:
            s = c.score
            # Boost trusted
            if self._prefer_trusted and c.trusted:
                s += 0.3
            # Boost local (already installed)
            if c.is_local:
                s += 0.5
            # Penalise over-budget
            if c.vram_mb > vram_budget:
                s -= 0.5
            # Prefer smaller models (faster)
            s -= (c.vram_mb / 10_000)
            return s

        return sorted(candidates, key=score, reverse=True)

    def _estimate_vram(self, model_id: str) -> int:
        """Rough VRAM estimate from model ID heuristics."""
        lower = model_id.lower()
        if "large" in lower:
            return 2000
        if "base" in lower:
            return 800
        if "small" in lower or "tiny" in lower or "nano" in lower:
            return 300
        return 500  # default

    @staticmethod
    def _component_to_candidate(component: object) -> ComponentCandidate:
        """Convert a CVComponent to a ComponentCandidate."""
        meta = component.metadata  # type: ignore[attr-defined]
        return ComponentCandidate(
            component_id=meta.component_id,
            name=meta.name,
            source=meta.source,
            source_id=meta.source_id,
            capabilities=meta.capabilities,
            vram_mb=meta.resource_estimate.vram_mb,
            score=1.0,
            python_requirements=meta.python_requirements,
            model_class="",
            trusted=meta.trusted,
        )
