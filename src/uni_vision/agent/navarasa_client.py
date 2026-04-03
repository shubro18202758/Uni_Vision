"""Async Ollama client for the Navarasa 2.0 7B conversational UI LLM.

Provides an Alpaca-format chat interface to the Navarasa model served
via Ollama.  Navarasa is a Gemma 7B fine-tuned on 15 Indian languages
and uses the prompt pattern:

    ### Instruction: {instruction}
    ### Input: {input}
    ### Response:

The client shares the Ollama endpoint with the Gemma 4 E2B Manager
Agent but targets the ``uni-vision-navarasa`` model tag.  Ollama
handles sequential model swapping — only one model occupies VRAM at
a time on the 8 GB RTX 4070.

Responsibility:
  - Conversational and interactive generative LLM for the frontend UI.
    Converses naturally with users in any of 15 Indian languages,
    translates between Indian languages and English for Gemma 4
    processing, explains system status and guides users through the
    interface, and translates alerts/notifications.
  - All pipeline intelligence (plate interpretation, OCR, state
    codes, detection enrichment, CV component management) is
    handled exclusively by the Gemma 4 E2B Manager Agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from uni_vision.common.config import NavarasaConfig

logger = logging.getLogger(__name__)


# ── Language code → display name ──────────────────────────────────

LANGUAGE_NAMES: dict[str, str] = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
    "as": "Assamese",
    "kok": "Konkani",
    "ne": "Nepali",
    "sd": "Sindhi",
    "en": "English",
}


@dataclass
class NavarasaResponse:
    """Parsed response from the Navarasa model."""

    content: str
    role: str = "assistant"
    raw_body: dict[str, Any] | None = None
    total_duration_ns: int = 0
    eval_count: int = 0
    language: str = "en"


class NavarasaClient:
    """Async Ollama client for the Navarasa 2.0 7B conversational UI LLM.

    Handles natural conversation in Indian languages, multilingual
    translation, and interactive guidance for the frontend interface.
    All pipeline/CV intelligence is handled by the Gemma 4 E2B
    Manager Agent — Navarasa focuses on user-facing interaction.

    Parameters
    ----------
    config : NavarasaConfig
        Navarasa connection and model parameters.
    """

    def __init__(self, config: NavarasaConfig) -> None:
        self._cfg = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        self._supported_langs = set(config.supported_languages.split(","))

    # ── Core chat interface ───────────────────────────────────────

    async def chat(
        self,
        instruction: str,
        input_text: str = "",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> NavarasaResponse:
        """Send an Alpaca-format request to Navarasa via Ollama /api/chat.

        Parameters
        ----------
        instruction : str
            The task instruction (maps to ``### Instruction:``).
        input_text : str
            The input data (maps to ``### Input:``).
        temperature : float, optional
            Override default temperature.
        max_tokens : int, optional
            Override default num_predict.

        Returns
        -------
        NavarasaResponse
            The parsed Navarasa response.
        """
        user_content = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "num_ctx": self._cfg.num_ctx,
                "temperature": temperature or self._cfg.temperature,
                "top_p": self._cfg.top_p,
                "top_k": self._cfg.top_k,
                "num_predict": max_tokens or self._cfg.num_predict,
                "num_gpu": self._cfg.num_gpu,
                "seed": self._cfg.seed,
            },
        }

        try:
            resp = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException:
            logger.error("navarasa_timeout after=%ds", self._cfg.timeout_s)
            raise
        except httpx.HTTPError as exc:
            logger.error("navarasa_http_error error=%s", exc)
            raise

        if resp.status_code != 200:
            logger.error(
                "navarasa_bad_status status=%d body=%s",
                resp.status_code,
                resp.text[:200],
            )
            raise httpx.HTTPStatusError(
                f"Ollama (Navarasa) returned {resp.status_code}",
                request=resp.request,
                response=resp,
            )

        body = resp.json()
        content = body.get("message", {}).get("content", "")

        return NavarasaResponse(
            content=content,
            role=body.get("message", {}).get("role", "assistant"),
            raw_body=body,
            total_duration_ns=body.get("total_duration", 0),
            eval_count=body.get("eval_count", 0),
        )

    # ── Conversational interface ────────────────────────────────────

    async def converse(
        self,
        user_message: str,
        language: str = "",
        *,
        system_context: str = "",
        max_tokens: int | None = None,
    ) -> NavarasaResponse:
        """Have a free-form conversation with the user in their language.

        Unlike :meth:`translate`, this is open-ended dialogue — Navarasa
        can answer questions, explain system status, give guidance, or
        simply chat in any of the 15 supported Indian languages.

        Parameters
        ----------
        user_message : str
            The user's message (in any supported language).
        language : str
            ISO 639-1 code for the response language.  Defaults to
            ``default_language`` from config.
        system_context : str, optional
            Extra context about current system state (e.g. active
            cameras, recent detections) to ground the conversation.
        max_tokens : int, optional
            Override default num_predict.

        Returns
        -------
        NavarasaResponse
            Navarasa's conversational response.
        """
        lang = language or self._cfg.default_language
        lang_name = LANGUAGE_NAMES.get(lang, lang)

        instruction_parts = [
            f"Respond conversationally in {lang_name}.",
            "Be helpful, warm, and concise.",
        ]
        if system_context:
            instruction_parts.append(f"Current system context: {system_context}")

        resp = await self.chat(
            instruction=" ".join(instruction_parts),
            input_text=user_message,
            max_tokens=max_tokens or 512,
        )
        resp.language = lang
        return resp

    # ── Translation methods ───────────────────────────────────────

    async def translate(
        self,
        text: str,
        target_language: str = "",
    ) -> str:
        """Translate text into the specified Indian language.

        Parameters
        ----------
        text : str
            Text to translate (typically English → Indian language
            or Indian language → English).
        target_language : str
            ISO 639-1 code (e.g. ``hi``, ``ta``, ``te``).
            Defaults to ``default_language`` from config.

        Returns
        -------
        str
            Translated text, or original if translation fails.
        """
        lang = target_language or self._cfg.default_language
        if lang not in self._supported_langs:
            logger.warning("navarasa_unsupported_language lang=%s", lang)
            return text

        if lang == "en":
            return text

        lang_name = LANGUAGE_NAMES.get(lang, lang)

        try:
            resp = await self.chat(
                instruction=f"Translate the following text to {lang_name}. "
                f"Provide ONLY the translation, no explanation.",
                input_text=text,
                max_tokens=256,
            )
            translated = resp.content.strip()
            return translated if translated else text
        except Exception as exc:
            logger.warning("navarasa_translate_error lang=%s error=%s", lang, exc)
            return text

    async def translate_to_english(self, text: str, source_language: str = "") -> str:
        """Translate text from an Indian language into English.

        Parameters
        ----------
        text : str
            Text in an Indian language to translate to English.
        source_language : str
            ISO 639-1 code of the source language (e.g. ``hi``).

        Returns
        -------
        str
            English translation, or original text if translation fails.
        """
        lang = source_language or self._cfg.default_language
        lang_name = LANGUAGE_NAMES.get(lang, lang)

        try:
            resp = await self.chat(
                instruction=f"Translate the following {lang_name} text to English. "
                f"Provide ONLY the English translation, no explanation.",
                input_text=text,
                max_tokens=256,
            )
            translated = resp.content.strip()
            return translated if translated else text
        except Exception as exc:
            logger.warning("navarasa_translate_to_en_error lang=%s error=%s", lang, exc)
            return text

    async def translate_alert(
        self,
        alert_message: str,
        target_language: str = "",
    ) -> dict[str, str]:
        """Translate a WebSocket alert into the target Indian language.

        Returns both the original and translated message.
        """
        lang = target_language or self._cfg.default_language
        translated = await self.translate(alert_message, lang)

        return {
            "original": alert_message,
            "translated": translated,
            "language": lang,
            "language_name": LANGUAGE_NAMES.get(lang, lang),
        }

    # ── Utility ───────────────────────────────────────────────────

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if a language code is supported by Navarasa."""
        return lang_code in self._supported_langs

    @property
    def supported_languages(self) -> dict[str, str]:
        """Return dict of supported language codes → names."""
        return {code: LANGUAGE_NAMES.get(code, code) for code in sorted(self._supported_langs)}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
