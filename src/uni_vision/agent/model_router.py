"""Ollama model router — manages VRAM-exclusive model activation.

On the 8 GB RTX 4070 only ONE model can occupy VRAM at a time.
This router provides clean activation/deactivation of models
through the Ollama HTTP API:

- **Pre-Launch** (design phase): Navarasa 2.0 7B is active for chat,
  translation, and agentic workflow design.
- **Post-Launch** (pipeline phase): Qwen 3.5 9B is active for
  pipeline reasoning, tool execution, and CV orchestration.
  Navarasa goes silent.

Ollama model lifecycle:
  - Load model:   POST /api/generate  {"model": "...", "keep_alive": "10m"}
  - Unload model:  POST /api/generate  {"model": "...", "keep_alive": "0"}
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ModelPhase(str, Enum):
    """Current VRAM occupant phase."""

    PRE_LAUNCH = "pre_launch"    # Navarasa active, Qwen dormant
    POST_LAUNCH = "post_launch"  # Qwen active, Navarasa dormant
    TRANSITIONING = "transitioning"
    IDLE = "idle"                # Neither loaded


@dataclass
class ModelState:
    """Snapshot of the current model routing state."""

    phase: ModelPhase
    active_model: str
    navarasa_loaded: bool
    qwen_loaded: bool


class OllamaModelRouter:
    """Manages exclusive VRAM model activation via the Ollama API.

    Parameters
    ----------
    ollama_base_url : str
        Ollama server URL (e.g. ``http://localhost:11434``).
    navarasa_model : str
        Ollama model tag for Navarasa 2.0 7B.
    qwen_model : str
        Ollama model tag for Qwen 3.5 9B.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        navarasa_model: str = "uni-vision-navarasa",
        qwen_model: str = "qwen3.5:9b-q4_K_M",
    ) -> None:
        self._base_url = ollama_base_url
        self._navarasa_model = navarasa_model
        self._qwen_model = qwen_model
        self._phase = ModelPhase.IDLE
        self._lock = asyncio.Lock()

        self._client = httpx.AsyncClient(
            base_url=ollama_base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    # ── Public API ────────────────────────────────────────────────

    async def activate_navarasa(self) -> ModelState:
        """Switch to pre-launch phase: load Navarasa, unload Qwen.

        Called at startup and when the pipeline is stopped.
        """
        async with self._lock:
            self._phase = ModelPhase.TRANSITIONING
            logger.info("model_router_activating_navarasa")

            # Unload Qwen first to free VRAM
            await self._unload_model(self._qwen_model)
            # Load Navarasa
            await self._load_model(self._navarasa_model)

            self._phase = ModelPhase.PRE_LAUNCH
            state = self.get_state()
            logger.info(
                "model_router_navarasa_active phase=%s model=%s",
                state.phase.value,
                state.active_model,
            )
            return state

    async def activate_qwen(self) -> ModelState:
        """Switch to post-launch phase: load Qwen, unload Navarasa.

        Called when the pipeline Launch button is pressed.
        """
        async with self._lock:
            self._phase = ModelPhase.TRANSITIONING
            logger.info("model_router_activating_qwen")

            # Unload Navarasa first to free VRAM
            await self._unload_model(self._navarasa_model)
            # Load Qwen
            await self._load_model(self._qwen_model)

            self._phase = ModelPhase.POST_LAUNCH
            state = self.get_state()
            logger.info(
                "model_router_qwen_active phase=%s model=%s",
                state.phase.value,
                state.active_model,
            )
            return state

    def get_state(self) -> ModelState:
        """Return current model routing state."""
        if self._phase == ModelPhase.PRE_LAUNCH:
            return ModelState(
                phase=ModelPhase.PRE_LAUNCH,
                active_model=self._navarasa_model,
                navarasa_loaded=True,
                qwen_loaded=False,
            )
        elif self._phase == ModelPhase.POST_LAUNCH:
            return ModelState(
                phase=ModelPhase.POST_LAUNCH,
                active_model=self._qwen_model,
                navarasa_loaded=False,
                qwen_loaded=True,
            )
        elif self._phase == ModelPhase.TRANSITIONING:
            return ModelState(
                phase=ModelPhase.TRANSITIONING,
                active_model="",
                navarasa_loaded=False,
                qwen_loaded=False,
            )
        else:
            return ModelState(
                phase=ModelPhase.IDLE,
                active_model="",
                navarasa_loaded=False,
                qwen_loaded=False,
            )

    @property
    def phase(self) -> ModelPhase:
        return self._phase

    @property
    def is_navarasa_active(self) -> bool:
        return self._phase == ModelPhase.PRE_LAUNCH

    @property
    def is_qwen_active(self) -> bool:
        return self._phase == ModelPhase.POST_LAUNCH

    # ── Internal Ollama API calls ─────────────────────────────────

    async def _load_model(self, model: str) -> None:
        """Warm-load a model into VRAM via Ollama generate with keep_alive."""
        try:
            resp = await self._client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "keep_alive": "10m",
                    "stream": False,
                },
            )
            if resp.status_code == 200:
                logger.info("model_loaded model=%s", model)
            else:
                logger.warning(
                    "model_load_unexpected_status model=%s status=%d",
                    model,
                    resp.status_code,
                )
        except httpx.TimeoutException:
            logger.warning("model_load_timeout model=%s", model)
        except httpx.HTTPError as exc:
            logger.warning("model_load_error model=%s error=%s", model, exc)

    async def _unload_model(self, model: str) -> None:
        """Instruct Ollama to immediately release a model from VRAM."""
        try:
            resp = await self._client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "keep_alive": "0",
                    "stream": False,
                },
            )
            if resp.status_code == 200:
                logger.info("model_unloaded model=%s", model)
            else:
                logger.warning(
                    "model_unload_unexpected_status model=%s status=%d",
                    model,
                    resp.status_code,
                )
        except httpx.TimeoutException:
            # Model might already be unloaded / not exist
            logger.debug("model_unload_timeout model=%s (likely already unloaded)", model)
        except httpx.HTTPError as exc:
            logger.debug("model_unload_error model=%s error=%s (non-critical)", model, exc)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
