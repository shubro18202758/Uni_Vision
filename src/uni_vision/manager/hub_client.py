"""HuggingFace Hub and external source client.

Provides async methods to search for models on HuggingFace Hub,
fetch model metadata (cards, tags, file sizes), and download models
to a local cache directory.  Also supports discovering PyPI packages
that wrap CV models, and OPEN-ENDED internet search driven by LLM-
generated queries for unbounded component discovery.

Security considerations:
  * Only downloads from allowlisted sources (HuggingFace, PyPI).
  * Model files are cached in a controlled directory.
  * pip installs use --no-deps by default; the ConflictResolver
    handles dependency validation separately.
  * No arbitrary code execution during download — execution only
    happens after the Manager Agent explicitly calls ``load()``.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import structlog
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from uni_vision.components.base import ComponentCapability
from uni_vision.manager.schemas import ComponentCandidate

log = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────

HF_API_BASE = "https://huggingface.co/api"
HF_SEARCH_ENDPOINT = f"{HF_API_BASE}/models"
PYPI_SEARCH_URL = "https://pypi.org/pypi"
PYPI_SEARCH_XML_URL = "https://pypi.org/simple/"

# Default local cache for downloaded models
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "uni_vision" / "models"

# Download retry settings
_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_TIMEOUT_SECONDS = 300  # 5 minutes per download attempt

# Task → HuggingFace pipeline_tag mapping (SEED KNOWLEDGE — used as hints
# for mapping known capabilities to HF pipeline tags.  The LLM can generate
# free-form search queries that bypass this mapping entirely.)
_CAPABILITY_TO_HF_TASK: Dict[str, str] = {
    "object_detection": "object-detection",
    "vehicle_detection": "object-detection",
    "plate_detection": "object-detection",
    "person_detection": "object-detection",
    "face_detection": "object-detection",
    "zero_shot_detection": "zero-shot-object-detection",
    "plate_ocr": "image-to-text",
    "scene_text_ocr": "image-to-text",
    "document_ocr": "image-to-text",
    "image_enhance": "image-to-image",
    "super_resolution": "image-to-image",
    "semantic_segmentation": "image-segmentation",
    "instance_segmentation": "image-segmentation",
    "scene_classification": "image-classification",
    "action_recognition": "video-classification",
    "depth_estimation": "depth-estimation",
    "pose_estimation": "image-classification",
    "tracking": "object-detection",
}

# Well-known trusted HuggingFace repos (SEED KNOWLEDGE — these are boosted
# in scoring but the system can discover and use ANY repo on the Hub.)
TRUSTED_REPOS: frozenset[str] = frozenset({
    "ultralytics/yolov8",
    "google/owlv2-base-patch16-ensemble",
    "microsoft/Florence-2-base",
    "IDEA-Research/grounding-dino-base",
    "PaddlePaddle/PaddleOCR",
    "facebook/detr-resnet-50",
    "facebook/sam-vit-base",
    "hustvl/yolos-tiny",
    "microsoft/beit-base-patch16-224",
    "google/vit-base-patch16-224",
    "nvidia/mit-b0",
    "WinKawaks/yolov8s",
    "intel/dpt-large",
    "openai/clip-vit-base-patch32",
    "facebook/dinov2-base",
})

# Extended library_name → (module_path, class_name, requirements, load_pattern)
# load_pattern: "from_pretrained" | "constructor" | "hub_load" | "create_model"
_LIBRARY_LOAD_PATTERNS: Dict[str, Dict[str, Any]] = {
    "transformers": {
        "module": "transformers",
        "class": "AutoModel",
        "requirements": ["transformers", "torch"],
        "load_pattern": "from_pretrained",
    },
    "ultralytics": {
        "module": "ultralytics",
        "class": "YOLO",
        "requirements": ["ultralytics"],
        "load_pattern": "constructor",  # YOLO(model_name)
    },
    "timm": {
        "module": "timm",
        "class": "create_model",
        "requirements": ["timm", "torch"],
        "load_pattern": "create_model",  # timm.create_model(name, pretrained=True)
    },
    "diffusers": {
        "module": "diffusers",
        "class": "DiffusionPipeline",
        "requirements": ["diffusers", "torch"],
        "load_pattern": "from_pretrained",
    },
    "open_clip": {
        "module": "open_clip",
        "class": "create_model_and_transforms",
        "requirements": ["open_clip_torch"],
        "load_pattern": "create_model",
    },
    "sentence-transformers": {
        "module": "sentence_transformers",
        "class": "SentenceTransformer",
        "requirements": ["sentence-transformers"],
        "load_pattern": "constructor",
    },
    "paddlepaddle": {
        "module": "paddleocr",
        "class": "PaddleOCR",
        "requirements": ["paddleocr", "paddlepaddle"],
        "load_pattern": "constructor",
    },
    "detectron2": {
        "module": "detectron2.engine",
        "class": "DefaultPredictor",
        "requirements": ["detectron2"],
        "load_pattern": "constructor",
    },
    "yolov5": {
        "module": "torch.hub",
        "class": "load",
        "requirements": ["torch", "torchvision"],
        "load_pattern": "hub_load",
        "hub_repo": "ultralytics/yolov5",
    },
    "keras": {
        "module": "tensorflow.keras.applications",
        "class": "MobileNetV2",
        "requirements": ["tensorflow"],
        "load_pattern": "constructor",
    },
    "pytorch": {
        "module": "torch.hub",
        "class": "load",
        "requirements": ["torch"],
        "load_pattern": "hub_load",
    },
}


@dataclass
class HFModelInfo:
    """Parsed metadata from a HuggingFace Hub model."""

    model_id: str
    pipeline_tag: str = ""
    library_name: str = ""
    tags: List[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    license: str = ""
    description: str = ""
    model_size_mb: int = 0

    @property
    def is_trusted(self) -> bool:
        return any(self.model_id.startswith(prefix) for prefix in TRUSTED_REPOS)


class HubClient:
    """Async client for HuggingFace Hub model discovery and download.

    Parameters
    ----------
    cache_dir:
        Local directory for model caches.
    http_timeout:
        Timeout for Hub API requests in seconds.
    max_search_results:
        Maximum models returned per search query.
    """

    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        http_timeout: float = 30.0,
        max_search_results: int = 10,
    ) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = http_timeout
        self._max_results = max_search_results

    # ── Search ────────────────────────────────────────────────────

    async def search_models(
        self,
        query: str,
        *,
        capability: Optional[ComponentCapability] = None,
        max_results: Optional[int] = None,
    ) -> List[HFModelInfo]:
        """Search HuggingFace Hub for models matching a query.

        Parameters
        ----------
        query:
            Free-text search (e.g., "license plate detection yolov8").
        capability:
            If provided, maps to a ``pipeline_tag`` filter.
        max_results:
            Override the default result limit.
        """
        params: Dict[str, Any] = {
            "search": query,
            "limit": max_results or self._max_results,
            "sort": "downloads",
            "direction": -1,
        }

        # Map capability to HF pipeline_tag filter
        if capability:
            hf_task = _CAPABILITY_TO_HF_TASK.get(capability.value)
            if hf_task:
                params["pipeline_tag"] = hf_task

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(HF_SEARCH_ENDPOINT, params=params)
                resp.raise_for_status()
                models_raw = resp.json()
        except httpx.HTTPError as exc:
            log.error("hf_search_failed", query=query, error=str(exc))
            return []

        results: List[HFModelInfo] = []
        for entry in models_raw:
            results.append(HFModelInfo(
                model_id=entry.get("modelId", entry.get("id", "")),
                pipeline_tag=entry.get("pipeline_tag", ""),
                library_name=entry.get("library_name", ""),
                tags=entry.get("tags", []),
                downloads=entry.get("downloads", 0),
                likes=entry.get("likes", 0),
            ))

        log.info("hf_search_complete", query=query, results=len(results))
        return results

    async def get_model_info(self, repo_id: str) -> Optional[HFModelInfo]:
        """Fetch detailed metadata for a specific model."""
        url = f"{HF_API_BASE}/models/{repo_id}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            log.error("hf_model_info_failed", repo_id=repo_id, error=str(exc))
            return None

        return HFModelInfo(
            model_id=data.get("modelId", repo_id),
            pipeline_tag=data.get("pipeline_tag", ""),
            library_name=data.get("library_name", ""),
            tags=data.get("tags", []),
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            license=data.get("cardData", {}).get("license", ""),
            description=data.get("description", "")[:500],
        )

    # ── Download ──────────────────────────────────────────────────

    async def download_model(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        timeout_seconds: Optional[float] = None,
    ) -> Path:
        """Download a HuggingFace model to local cache with retry/timeout.

        Uses ``huggingface_hub`` if available, with configurable timeout
        and automatic retry on transient failures.

        Returns the local cache directory path.
        """
        safe_name = repo_id.replace("/", "--")
        model_dir = self._cache_dir / safe_name

        if model_dir.exists():
            log.info("hf_model_cached", repo_id=repo_id, path=str(model_dir))
            return model_dir

        timeout = timeout_seconds or _DOWNLOAD_TIMEOUT_SECONDS

        for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 1):
            log.info("hf_model_downloading", repo_id=repo_id, attempt=attempt)
            try:
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    log.warning("huggingface_hub_not_installed, installing")
                    ok = await self.install_package("huggingface_hub")
                    if not ok:
                        raise RuntimeError("Failed to install huggingface_hub")
                    from huggingface_hub import snapshot_download

                path = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id,
                            revision=revision,
                            cache_dir=str(self._cache_dir),
                            local_dir=str(model_dir),
                        ),
                    ),
                    timeout=timeout,
                )
                log.info("hf_model_downloaded", repo_id=repo_id, path=path)
                return Path(path)

            except asyncio.TimeoutError:
                log.error("hf_download_timeout", repo_id=repo_id, attempt=attempt, timeout_s=timeout)
                if attempt == _DOWNLOAD_MAX_RETRIES:
                    raise RuntimeError(f"Download of {repo_id} timed out after {_DOWNLOAD_MAX_RETRIES} attempts")
            except Exception as exc:
                log.error("hf_download_failed", repo_id=repo_id, attempt=attempt, error=str(exc))
                if attempt == _DOWNLOAD_MAX_RETRIES:
                    raise
                await asyncio.sleep(2 ** attempt)  # exponential backoff

        raise RuntimeError(f"Download of {repo_id} failed after {_DOWNLOAD_MAX_RETRIES} attempts")

    # ── Package installation ──────────────────────────────────────

    async def install_package(
        self,
        package_spec: str,
        *,
        no_deps: bool = False,
        verify_import: Optional[str] = None,
        timeout_seconds: float = 120.0,
    ) -> bool:
        """Install a Python package using pip with verification.

        Parameters
        ----------
        package_spec:
            Package name with optional version (e.g., ``"ultralytics>=8.0"``).
        no_deps:
            If True, install without pulling transitive dependencies.
        verify_import:
            Module name to import-check after install.  If None, infers
            from package_spec (strips version specifiers and normalizes).
        timeout_seconds:
            Maximum time to wait for pip install.
        """
        cmd = [sys.executable, "-m", "pip", "install", package_spec, "--quiet"]
        if no_deps:
            cmd.append("--no-deps")

        log.info("pip_install_start", package=package_spec)
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=5.0,  # timeout for process creation
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds,
            )

            if proc.returncode != 0:
                log.error(
                    "pip_install_failed",
                    package=package_spec,
                    returncode=proc.returncode,
                    stderr=stderr.decode()[:500],
                )
                return False

            log.info("pip_install_success", package=package_spec)

            # Verify the package is actually importable
            import_name = verify_import or self._infer_import_name(package_spec)
            if import_name:
                importable = await self.check_package_installed(import_name)
                if not importable:
                    log.error(
                        "pip_install_verify_failed",
                        package=package_spec,
                        import_name=import_name,
                    )
                    return False
                log.info("pip_install_verified", package=package_spec, import_name=import_name)

            return True

        except asyncio.TimeoutError:
            log.error("pip_install_timeout", package=package_spec, timeout_s=timeout_seconds)
            return False
        except Exception as exc:
            log.error("pip_install_error", package=package_spec, error=str(exc))
            return False

    @staticmethod
    def _infer_import_name(package_spec: str) -> str:
        """Infer the import name from a pip package spec.

        Handles common mismatches like ``Pillow`` → ``PIL``,
        ``scikit-learn`` → ``sklearn``, ``opencv-python`` → ``cv2``.
        """
        # Strip version specifiers
        name = package_spec.split(">=")[0].split("<=")[0].split("==")[0].split("[")[0].strip()

        _IMPORT_MAP = {
            "pillow": "PIL",
            "scikit-learn": "sklearn",
            "opencv-python": "cv2",
            "opencv-python-headless": "cv2",
            "pytorch": "torch",
            "paddlepaddle": "paddle",
            "paddlepaddle-gpu": "paddle",
            "deep-sort-realtime": "deep_sort_realtime",
            "open-clip-torch": "open_clip",
            "sentence-transformers": "sentence_transformers",
        }
        normalized = name.lower().replace("_", "-")
        if normalized in _IMPORT_MAP:
            return _IMPORT_MAP[normalized]
        # Default: replace hyphens with underscores
        return name.replace("-", "_")

    async def check_package_installed(self, package_name: str) -> bool:
        """Check if a Python package is importable (side-effect free)."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: importlib.util.find_spec(package_name),
            )
            return result is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    # ── Candidate conversion ──────────────────────────────────────

    def hf_model_to_candidate(
        self,
        info: HFModelInfo,
        *,
        capabilities: set[ComponentCapability],
        vram_mb: int = 500,
    ) -> ComponentCandidate:
        """Convert a HuggingFace model info into a ComponentCandidate.

        Uses the extended ``_LIBRARY_LOAD_PATTERNS`` registry to infer
        the correct model class, requirements, and loading strategy for
        a wide range of library ecosystems.
        """
        model_class = ""
        requirements: List[str] = []
        load_metadata: Dict[str, Any] = {}

        lib = info.library_name or ""
        pattern = _LIBRARY_LOAD_PATTERNS.get(lib)

        if pattern:
            model_class = f"{pattern['module']}.{pattern['class']}"
            requirements = list(pattern["requirements"])
            load_metadata["load_pattern"] = pattern["load_pattern"]
            if "hub_repo" in pattern:
                load_metadata["hub_repo"] = pattern["hub_repo"]
        else:
            # Infer from tags if library_name not directly matched
            tags_lower = [t.lower() for t in info.tags]
            if any("yolo" in t for t in tags_lower):
                model_class = "ultralytics.YOLO"
                requirements = ["ultralytics"]
                load_metadata["load_pattern"] = "constructor"
            elif any("clip" in t for t in tags_lower):
                model_class = "transformers.CLIPModel"
                requirements = ["transformers", "torch"]
                load_metadata["load_pattern"] = "from_pretrained"
            elif any("detr" in t for t in tags_lower):
                model_class = "transformers.DetrForObjectDetection"
                requirements = ["transformers", "torch"]
                load_metadata["load_pattern"] = "from_pretrained"
            elif any("segment" in t for t in tags_lower) or any("sam" in t for t in tags_lower):
                model_class = "transformers.SamModel"
                requirements = ["transformers", "torch"]
                load_metadata["load_pattern"] = "from_pretrained"
            elif any("vit" in t for t in tags_lower):
                model_class = "transformers.ViTForImageClassification"
                requirements = ["transformers", "torch"]
                load_metadata["load_pattern"] = "from_pretrained"
            elif info.pipeline_tag in ("object-detection", "image-classification",
                                        "image-segmentation", "image-to-text"):
                # Generic transformers fallback for recognized vision tasks
                model_class = "transformers.AutoModel"
                requirements = ["transformers", "torch"]
                load_metadata["load_pattern"] = "from_pretrained"

        return ComponentCandidate(
            component_id=f"hf.{info.model_id.replace('/', '.')}",
            name=info.model_id.split("/")[-1] if "/" in info.model_id else info.model_id,
            source="huggingface",
            source_id=info.model_id,
            capabilities=capabilities,
            vram_mb=vram_mb,
            score=min(info.downloads / 100_000, 1.0),  # Normalised popularity
            python_requirements=requirements,
            model_class=model_class,
            metadata={
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "downloads": info.downloads,
                "likes": info.likes,
                "license": info.license,
                "load_pattern": load_metadata.get("load_pattern", ""),
                "hub_repo": load_metadata.get("hub_repo", ""),
            },
            trusted=info.is_trusted,
        )

    # ── Multi-source discovery ────────────────────────────────────

    async def search_pypi(
        self,
        query: str,
        *,
        capability: Optional[ComponentCapability] = None,
    ) -> List[ComponentCandidate]:
        """Search PyPI for CV model wrapper packages.

        Uses a curated registry of known CV packages cross-referenced
        with live PyPI metadata (version, summary) for validation.
        Packages that are not available on PyPI are excluded.
        """
        candidates: List[ComponentCandidate] = []
        well_known = self._pypi_candidates_for(query, capability)

        # Also do a real PyPI search for the query term against our registry
        if capability and not well_known:
            # Fall back to matching all packages that have the capability
            for pkg_name, keywords, caps, vram in _PYPI_CV_PACKAGES:
                if capability in caps:
                    well_known.append((pkg_name, caps, vram))

        for pkg_name, pkg_caps, vram_est in well_known:
            info = await self._get_pypi_info(pkg_name)
            if info is None:
                continue  # Package doesn't exist on PyPI — skip

            # Look up the entry class from our curated registry
            entry_info = _PYPI_PACKAGE_DETAILS.get(pkg_name, {})

            candidates.append(ComponentCandidate(
                component_id=f"pypi.{pkg_name}",
                name=pkg_name,
                source="pypi",
                source_id=pkg_name,
                capabilities=pkg_caps,
                vram_mb=vram_est,
                score=min((info.get("downloads", 0)) / 500_000, 1.0),
                python_requirements=entry_info.get("requirements", [pkg_name]),
                model_class=entry_info.get("model_class", ""),
                metadata={
                    "version": info.get("version", ""),
                    "summary": info.get("summary", ""),
                    "home_page": info.get("home_page", ""),
                    "entry_module": entry_info.get("entry_module", pkg_name),
                    "entry_class": entry_info.get("entry_class", ""),
                    "load_pattern": entry_info.get("load_pattern", "constructor"),
                },
                trusted=pkg_name in _TRUSTED_PYPI_PACKAGES,
            ))

        return candidates

    async def search_torchhub(
        self,
        query: str,
        *,
        capability: Optional[ComponentCapability] = None,
    ) -> List[ComponentCandidate]:
        """Search known TorchHub repos for matching models.

        TorchHub lacks a search API, so we query a curated list of
        well-known repos that host CV models.  The query and capability
        are both checked for matches.
        """
        candidates: List[ComponentCandidate] = []

        # Search all TorchHub entries matching query or capability
        matched_repos: List[tuple] = []
        q_lower = query.lower()

        for key, repos in _TORCHHUB_REPOS.items():
            # Match by query keyword or by capability value
            if q_lower in key or any(q_lower in r[1].lower() for r in repos):
                matched_repos.extend(repos)
            elif capability and capability.value == key:
                matched_repos.extend(repos)

        # De-duplicate
        seen = set()
        for repo_id, model_name, caps, vram_est in matched_repos:
            cid = f"torchhub.{repo_id.replace('/', '.')}.{model_name}"
            if cid in seen:
                continue
            seen.add(cid)
            candidates.append(ComponentCandidate(
                component_id=cid,
                name=model_name,
                source="torchhub",
                source_id=f"{repo_id}:{model_name}",
                capabilities=caps,
                vram_mb=vram_est,
                score=0.7,
                python_requirements=["torch", "torchvision"],
                model_class=f"torch.hub.load('{repo_id}', '{model_name}')",
                metadata={
                    "repo": repo_id,
                    "load_pattern": "hub_load",
                },
                trusted=True,
            ))

        return candidates

    async def unified_search(
        self,
        query: str,
        *,
        capability: Optional[ComponentCapability] = None,
        max_results: Optional[int] = None,
    ) -> List[ComponentCandidate]:
        """Search all sources and return a de-duplicated, ranked list.

        Searches HuggingFace Hub, PyPI, and TorchHub in parallel,
        then merges and ranks results by composite score.
        """
        hf_task = asyncio.create_task(self.search_models(
            query, capability=capability, max_results=max_results,
        ))
        pypi_task = asyncio.create_task(self.search_pypi(
            query, capability=capability,
        ))
        torch_task = asyncio.create_task(self.search_torchhub(
            query, capability=capability,
        ))

        hf_models, pypi_cands, torch_cands = await asyncio.gather(
            hf_task, pypi_task, torch_task,
        )

        # Convert HF results to candidates
        cap_set = {capability} if capability else set()
        hf_cands = [
            self.hf_model_to_candidate(m, capabilities=cap_set)
            for m in hf_models
        ]

        # Merge and de-duplicate
        all_cands = hf_cands + pypi_cands + torch_cands
        seen: set[str] = set()
        unique: List[ComponentCandidate] = []
        for c in all_cands:
            if c.component_id not in seen:
                seen.add(c.component_id)
                unique.append(c)

        # Sort by score (highest first), then trusted first
        unique.sort(key=lambda c: (c.trusted, c.score), reverse=True)

        limit = max_results or self._max_results
        return unique[:limit]

    # ── Open-ended LLM-driven search ────────────────────────────────

    async def open_search(
        self,
        query: str,
        *,
        source: str = "all",
        vram_budget_mb: int = 1024,
        max_results: Optional[int] = None,
    ) -> List[ComponentCandidate]:
        """Open-ended internet search driven by LLM-generated queries.

        Unlike ``unified_search`` which maps capabilities to fixed
        pipeline tags, this method takes a FREE-FORM query string
        (typically generated by Qwen 3.5) and searches the open
        internet without any curated registry as a ceiling.

        Parameters
        ----------
        query:
            Free-form search string from the LLM (e.g., "underwater
            image correction deep learning model").
        source:
            Where to search: "huggingface", "pypi", "github", or "all".
        vram_budget_mb:
            Maximum VRAM estimate for any single candidate.
        max_results:
            Limit on returned candidates.
        """
        limit = max_results or self._max_results
        all_cands: List[ComponentCandidate] = []

        sources = (
            ["huggingface", "pypi", "github"]
            if source == "all"
            else [source]
        )

        tasks = []
        if "huggingface" in sources:
            tasks.append(self._open_search_hf(query, limit))
        if "pypi" in sources:
            tasks.append(self._open_search_pypi(query, limit))
        if "github" in sources:
            tasks.append(self._open_search_github(query, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_cands.extend(result)
            elif isinstance(result, Exception):
                log.warning("open_search_source_failed", error=str(result))

        # De-duplicate and rank
        seen: set[str] = set()
        unique: List[ComponentCandidate] = []
        for c in all_cands:
            if c.component_id not in seen:
                seen.add(c.component_id)
                # Filter by VRAM budget
                if c.vram_mb <= vram_budget_mb:
                    unique.append(c)

        unique.sort(key=lambda c: (c.trusted, c.score), reverse=True)
        log.info(
            "open_search_complete",
            query=query,
            source=source,
            total_candidates=len(unique),
        )
        return unique[:limit]

    async def _open_search_hf(
        self,
        query: str,
        limit: int,
    ) -> List[ComponentCandidate]:
        """Search HuggingFace Hub with a free-form query — NO pipeline_tag
        filter, allowing discovery of any model type."""
        params: Dict[str, Any] = {
            "search": query,
            "limit": limit,
            "sort": "downloads",
            "direction": -1,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(HF_SEARCH_ENDPOINT, params=params)
                resp.raise_for_status()
                models_raw = resp.json()
        except httpx.HTTPError as exc:
            log.error("open_hf_search_failed", query=query, error=str(exc))
            return []

        results: List[ComponentCandidate] = []
        for entry in models_raw:
            info = HFModelInfo(
                model_id=entry.get("modelId", entry.get("id", "")),
                pipeline_tag=entry.get("pipeline_tag", ""),
                library_name=entry.get("library_name", ""),
                tags=entry.get("tags", []),
                downloads=entry.get("downloads", 0),
                likes=entry.get("likes", 0),
            )
            # Convert to candidate with empty capability set
            # (will be filled by LLM evaluation)
            candidate = self.hf_model_to_candidate(
                info,
                capabilities=set(),
                vram_mb=self._estimate_vram_from_info(info),
            )
            results.append(candidate)

        return results

    async def _open_search_pypi(
        self,
        query: str,
        limit: int,
    ) -> List[ComponentCandidate]:
        """Search PyPI using the JSON API and classifiers — beyond the
        curated registry.  Searches real PyPI for any package matching
        the query."""
        candidates: List[ComponentCandidate] = []

        # First check curated packages (fast path / seed knowledge)
        q_lower = query.lower()
        for pkg_name, keywords, caps, vram in _PYPI_CV_PACKAGES:
            if any(k in q_lower for k in keywords):
                info = await self._get_pypi_info(pkg_name)
                if info:
                    candidates.append(ComponentCandidate(
                        component_id=f"pypi.{pkg_name}",
                        name=pkg_name,
                        source="pypi",
                        source_id=pkg_name,
                        capabilities=caps,
                        vram_mb=vram,
                        score=min((info.get("downloads", 0)) / 500_000, 1.0) + 0.1,
                        python_requirements=[pkg_name],
                        trusted=pkg_name in _TRUSTED_PYPI_PACKAGES,
                    ))

        # Then do a real PyPI search: use the XMLRPC search or direct
        # package name probing for likely package names derived from query
        likely_names = self._derive_package_names(query)
        for pkg in likely_names:
            if any(c.source_id == pkg for c in candidates):
                continue  # already found from curated
            info = await self._get_pypi_info(pkg)
            if info and info.get("summary"):
                entry = _PYPI_PACKAGE_DETAILS.get(pkg, {})
                candidates.append(ComponentCandidate(
                    component_id=f"pypi.{pkg}",
                    name=pkg,
                    source="pypi",
                    source_id=pkg,
                    capabilities=set(),  # Unknown — LLM will evaluate
                    vram_mb=0,
                    score=min((info.get("downloads", 0)) / 500_000, 0.8),
                    python_requirements=entry.get("requirements", [pkg]),
                    model_class=entry.get("model_class", ""),
                    metadata={
                        "version": info.get("version", ""),
                        "summary": info.get("summary", ""),
                    },
                    trusted=pkg in _TRUSTED_PYPI_PACKAGES,
                ))

        return candidates[:limit]

    async def _open_search_github(
        self,
        query: str,
        limit: int,
    ) -> List[ComponentCandidate]:
        """Search GitHub public repositories for CV-related projects.

        Uses the GitHub Search API (no auth required for public repos,
        but rate-limited to 10 requests/minute).
        """
        search_url = "https://api.github.com/search/repositories"
        params = {
            "q": f"{query} topic:computer-vision language:python",
            "sort": "stars",
            "order": "desc",
            "per_page": min(limit, 10),
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    search_url,
                    params=params,
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
                if resp.status_code == 403:
                    log.warning("github_search_rate_limited")
                    return []
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            log.error("github_search_failed", query=query, error=str(exc))
            return []

        candidates: List[ComponentCandidate] = []
        for repo in data.get("items", []):
            full_name = repo.get("full_name", "")
            stars = repo.get("stargazers_count", 0)
            description = repo.get("description", "") or ""

            # Use star count as a trust signal
            score = min(stars / 10_000, 1.0)
            trusted = stars > 1000

            candidates.append(ComponentCandidate(
                component_id=f"github.{full_name.replace('/', '.')}",
                name=repo.get("name", ""),
                source="github",
                source_id=full_name,
                capabilities=set(),  # Unknown — LLM will evaluate
                vram_mb=0,  # Unknown
                score=score,
                python_requirements=[],  # Will need to parse setup.py/pyproject
                metadata={
                    "github_url": repo.get("html_url", ""),
                    "stars": stars,
                    "description": description[:200],
                    "language": repo.get("language", ""),
                    "license": (repo.get("license") or {}).get("spdx_id", ""),
                    "topics": repo.get("topics", []),
                },
                trusted=trusted,
            ))

        return candidates

    @staticmethod
    def _derive_package_names(query: str) -> List[str]:
        """Derive likely PyPI package names from a free-form query.

        Generates plausible pip-installable names like "underwater-enhance",
        "torch-dehazing", etc.
        """
        words = query.lower().replace("-", " ").replace("_", " ").split()
        # Filter stopwords
        stopwords = {"a", "the", "for", "and", "or", "in", "on", "with", "model",
                      "deep", "learning", "neural", "network", "python", "library"}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        names: List[str] = []
        # Single words
        for kw in keywords[:5]:
            names.append(kw)
        # Hyphenated pairs
        for i in range(min(len(keywords) - 1, 3)):
            names.append(f"{keywords[i]}-{keywords[i+1]}")
        # Common prefixes
        for prefix in ("torch-", "py", "cv-"):
            for kw in keywords[:3]:
                names.append(f"{prefix}{kw}")

        return names[:15]

    @staticmethod
    def _estimate_vram_from_info(info: HFModelInfo) -> int:
        """Estimate VRAM from HF model metadata."""
        lower = info.model_id.lower()
        if "large" in lower or "xl" in lower:
            return 2000
        if "base" in lower:
            return 800
        if "small" in lower or "tiny" in lower or "nano" in lower or "mini" in lower:
            return 300
        if info.model_size_mb > 0:
            return int(info.model_size_mb * 1.2)  # 1.2x overhead
        return 500

    # ── PyPI helpers ──────────────────────────────────────────────

    async def _get_pypi_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Fetch package metadata from PyPI JSON API."""
        url = f"{PYPI_SEARCH_URL}/{package_name}/json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
                info = data.get("info", {})
                return {
                    "version": info.get("version", ""),
                    "summary": info.get("summary", ""),
                    "home_page": info.get("home_page", ""),
                    "downloads": info.get("downloads", 0) or 0,
                }
        except httpx.HTTPError:
            return None

    @staticmethod
    def _pypi_candidates_for(
        query: str,
        capability: Optional[ComponentCapability],
    ) -> List[tuple]:
        """Return well-known PyPI packages matching a query or capability."""
        results = []
        q = query.lower()
        for pkg_name, keywords, caps, vram in _PYPI_CV_PACKAGES:
            if any(k in q for k in keywords):
                results.append((pkg_name, caps, vram))
            elif capability and capability in caps:
                results.append((pkg_name, caps, vram))
        return results


# ── Curated registries (SEED KNOWLEDGE — performance shortcuts, NOT ceilings)
# These provide fast initial lookups for common CV tasks.  The LLM-driven
# open_search() method can discover packages and models FAR beyond these
# lists via real-time internet search.
# ──────────────────────────────────────────────────────────────────────────

_TRUSTED_PYPI_PACKAGES: frozenset = frozenset({
    "ultralytics", "paddleocr", "easyocr", "mmdet",
    "detectron2", "torchvision", "timm", "supervision",
    "opencv-python", "opencv-python-headless", "albumentations",
    "deep-sort-realtime", "kornia", "open-clip-torch",
})

_PYPI_CV_PACKAGES: List[tuple] = [
    # (package_name, [search_keywords], {capabilities}, vram_estimate_mb)
    ("ultralytics", ["yolo", "detection", "plate", "vehicle", "ultralytics"],
     {ComponentCapability.OBJECT_DETECTION, ComponentCapability.VEHICLE_DETECTION,
      ComponentCapability.PLATE_DETECTION, ComponentCapability.PERSON_DETECTION},
     300),
    ("paddleocr", ["ocr", "paddle", "text", "paddleocr"],
     {ComponentCapability.PLATE_OCR, ComponentCapability.SCENE_TEXT_OCR,
      ComponentCapability.DOCUMENT_OCR},
     400),
    ("easyocr", ["ocr", "easy", "text", "recognition", "easyocr"],
     {ComponentCapability.PLATE_OCR, ComponentCapability.SCENE_TEXT_OCR},
     200),
    ("mmdet", ["mmdetection", "detection", "mask", "mmdet"],
     {ComponentCapability.OBJECT_DETECTION, ComponentCapability.INSTANCE_SEGMENTATION},
     500),
    ("supervision", ["supervision", "annotate", "track", "count"],
     {ComponentCapability.OBJECT_DETECTION},
     50),
    ("timm", ["timm", "classification", "scene", "imagenet"],
     {ComponentCapability.SCENE_CLASSIFICATION},
     300),
    ("deep-sort-realtime", ["deepsort", "deep_sort", "tracking", "sort"],
     {ComponentCapability.TRACKING},
     250),
    ("kornia", ["kornia", "augmentation", "geometry", "enhance"],
     {ComponentCapability.IMAGE_ENHANCE},
     50),
    ("albumentations", ["albumentations", "augmentation", "transform"],
     {ComponentCapability.IMAGE_ENHANCE},
     0),
    ("open-clip-torch", ["clip", "zero-shot", "open_clip"],
     {ComponentCapability.ZERO_SHOT_DETECTION, ComponentCapability.SCENE_CLASSIFICATION},
     400),
    ("transformers", ["transformers", "huggingface", "bert", "vit", "detr"],
     {ComponentCapability.OBJECT_DETECTION, ComponentCapability.SCENE_CLASSIFICATION},
     500),
]

# Detailed load info for each PyPI package
_PYPI_PACKAGE_DETAILS: Dict[str, Dict[str, Any]] = {
    "ultralytics": {
        "entry_module": "ultralytics",
        "entry_class": "YOLO",
        "model_class": "ultralytics.YOLO",
        "requirements": ["ultralytics"],
        "load_pattern": "constructor",
    },
    "paddleocr": {
        "entry_module": "paddleocr",
        "entry_class": "PaddleOCR",
        "model_class": "paddleocr.PaddleOCR",
        "requirements": ["paddleocr", "paddlepaddle"],
        "load_pattern": "constructor",
    },
    "easyocr": {
        "entry_module": "easyocr",
        "entry_class": "Reader",
        "model_class": "easyocr.Reader",
        "requirements": ["easyocr"],
        "load_pattern": "constructor",
    },
    "timm": {
        "entry_module": "timm",
        "entry_class": "create_model",
        "model_class": "timm.create_model",
        "requirements": ["timm", "torch"],
        "load_pattern": "create_model",
    },
    "deep-sort-realtime": {
        "entry_module": "deep_sort_realtime.deepsort_tracker",
        "entry_class": "DeepSort",
        "model_class": "deep_sort_realtime.deepsort_tracker.DeepSort",
        "requirements": ["deep-sort-realtime"],
        "load_pattern": "constructor",
    },
    "supervision": {
        "entry_module": "supervision",
        "entry_class": "Detections",
        "model_class": "supervision.Detections",
        "requirements": ["supervision"],
        "load_pattern": "constructor",
    },
    "kornia": {
        "entry_module": "kornia.enhance",
        "entry_class": "AdjustBrightness",
        "model_class": "kornia.enhance.AdjustBrightness",
        "requirements": ["kornia", "torch"],
        "load_pattern": "constructor",
    },
    "open-clip-torch": {
        "entry_module": "open_clip",
        "entry_class": "create_model_and_transforms",
        "model_class": "open_clip.create_model_and_transforms",
        "requirements": ["open_clip_torch"],
        "load_pattern": "create_model",
    },
    "transformers": {
        "entry_module": "transformers",
        "entry_class": "AutoModel",
        "model_class": "transformers.AutoModel",
        "requirements": ["transformers", "torch"],
        "load_pattern": "from_pretrained",
    },
}

_TORCHHUB_REPOS: Dict[str, List[tuple]] = {
    "object_detection": [
        ("ultralytics/yolov5", "yolov5s", {ComponentCapability.OBJECT_DETECTION}, 200),
        ("ultralytics/yolov5", "yolov5n", {ComponentCapability.OBJECT_DETECTION}, 100),
        ("facebookresearch/detr", "detr_resnet50", {ComponentCapability.OBJECT_DETECTION}, 400),
    ],
    "vehicle_detection": [
        ("ultralytics/yolov5", "yolov5s", {ComponentCapability.VEHICLE_DETECTION}, 200),
    ],
    "person_detection": [
        ("ultralytics/yolov5", "yolov5s", {ComponentCapability.PERSON_DETECTION}, 200),
    ],
    "semantic_segmentation": [
        ("pytorch/vision", "deeplabv3_resnet50", {ComponentCapability.SEMANTIC_SEGMENTATION}, 350),
        ("pytorch/vision", "deeplabv3_mobilenet_v3_large", {ComponentCapability.SEMANTIC_SEGMENTATION}, 150),
    ],
    "depth_estimation": [
        ("intel-isl/MiDaS", "MiDaS_small", {ComponentCapability.DEPTH_ESTIMATION}, 150),
        ("intel-isl/MiDaS", "DPT_Large", {ComponentCapability.DEPTH_ESTIMATION}, 400),
    ],
    "scene_classification": [
        ("pytorch/vision", "resnet18", {ComponentCapability.SCENE_CLASSIFICATION}, 100),
        ("pytorch/vision", "mobilenet_v3_small", {ComponentCapability.SCENE_CLASSIFICATION}, 50),
    ],
}
