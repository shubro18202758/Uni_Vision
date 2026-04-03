"""Shared async Ollama client for the agentic reasoning engine.

Provides a thin wrapper over the Ollama ``/api/chat`` endpoint,
reusing the same httpx connection-pooling pattern as
``ocr/llm_ocr.py`` but configured for the agent's longer context
window and structured JSON tool-call output.

Unlike the OCR client which expects XML, this client works with
JSON-structured reasoning and tool-call output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from uni_vision.common.config import OllamaConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Parsed response from the agent LLM."""

    content: str
    role: str = "assistant"
    raw_body: dict[str, Any] | None = None
    total_duration_ns: int = 0
    eval_count: int = 0


class AgentLLMClient:
    """Async Ollama client for the agentic reasoning loop.

    Parameters
    ----------
    config : OllamaConfig
        Ollama connection and model parameters.
    timeout_s : float
        Request timeout (agent requests may take longer than OCR).
    """

    def __init__(
        self,
        config: OllamaConfig,
        *,
        timeout_s: float = 30.0,
        max_tokens: int = 1024,
    ) -> None:
        self._cfg = config
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens

        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama.

        Parameters
        ----------
        messages : list[dict]
            The full conversation history [{role, content}, ...].
        temperature : float
            Sampling temperature (lower = more deterministic).
        max_tokens : int, optional
            Override the default max generation tokens.

        Returns
        -------
        LLMResponse
            The parsed assistant response.
        """
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": self._cfg.num_ctx,
                "temperature": temperature,
                "top_p": self._cfg.top_p,
                "top_k": self._cfg.top_k,
                "num_predict": max_tokens or self._max_tokens,
                "num_gpu": self._cfg.num_gpu,
                "seed": self._cfg.seed,
            },
        }

        try:
            resp = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException:
            logger.error("agent_llm_timeout after=%ds", self._timeout_s)
            raise
        except httpx.HTTPError as exc:
            logger.error("agent_llm_http_error error=%s", exc)
            raise

        if resp.status_code != 200:
            logger.error(
                "agent_llm_bad_status status=%d body=%s",
                resp.status_code,
                resp.text[:200],
            )
            raise httpx.HTTPStatusError(
                f"Ollama returned {resp.status_code}",
                request=resp.request,
                response=resp,
            )

        body = resp.json()
        content = body.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            role=body.get("message", {}).get("role", "assistant"),
            raw_body=body,
            total_duration_ns=body.get("total_duration", 0),
            eval_count=body.get("eval_count", 0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
