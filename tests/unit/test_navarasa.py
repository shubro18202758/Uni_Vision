"""Unit tests for Navarasa 2.0 7B — conversational & interactive UI LLM.

Covers:
  - NavarasaConfig defaults and env_prefix behaviour
  - NavarasaClient language support, chat formatting, translation helpers
  - Conversational interface (converse method)
  - LANGUAGE_NAMES registry completeness
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uni_vision.common.config import NavarasaConfig


# ═══════════════════════════════════════════════════════════════════
# ──  NavarasaConfig tests  ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestNavarasaConfig:
    """Verify default values and supported_languages parsing."""

    def test_defaults(self):
        cfg = NavarasaConfig()
        assert cfg.enabled is True
        assert cfg.model == "uni-vision-navarasa"
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.timeout_s == 15.0
        assert cfg.num_ctx == 4096
        assert cfg.temperature == 0.15
        assert cfg.num_predict == 512
        assert cfg.default_language == "hi"
        assert cfg.translate_alerts is True

    def test_supported_languages_list(self):
        cfg = NavarasaConfig()
        langs = cfg.supported_languages.split(",")
        assert len(langs) == 16
        assert "hi" in langs
        assert "en" in langs
        assert "ta" in langs
        assert "te" in langs

    def test_env_prefix(self):
        """Env vars with UV_NAVARASA_ prefix should override defaults."""
        cfg = NavarasaConfig(
            _env_file=None,
            enabled=False,
            model="custom-model",
            default_language="ta",
        )
        assert cfg.enabled is False
        assert cfg.model == "custom-model"
        assert cfg.default_language == "ta"


# ═══════════════════════════════════════════════════════════════════
# ──  NavarasaClient tests  ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestNavarasaClient:
    """Test NavarasaClient language support and translation methods."""

    def _make_client(self, **overrides):
        from uni_vision.agent.navarasa_client import NavarasaClient
        cfg = NavarasaConfig(**overrides)
        with patch("uni_vision.agent.navarasa_client.httpx.AsyncClient"):
            client = NavarasaClient(cfg)
        return client

    def test_is_language_supported(self):
        client = self._make_client()
        assert client.is_language_supported("hi") is True
        assert client.is_language_supported("ta") is True
        assert client.is_language_supported("en") is True
        assert client.is_language_supported("fr") is False
        assert client.is_language_supported("zh") is False

    def test_supported_languages_property(self):
        client = self._make_client()
        langs = client.supported_languages
        assert isinstance(langs, dict)
        assert langs["hi"] == "Hindi"
        assert langs["te"] == "Telugu"
        assert langs["en"] == "English"
        assert len(langs) == 16

    @pytest.mark.asyncio
    async def test_chat_builds_alpaca_format(self):
        """Verify the chat method builds Alpaca-format payload."""
        from uni_vision.agent.navarasa_client import NavarasaClient

        cfg = NavarasaConfig()
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "test response"},
            "total_duration": 100000,
            "eval_count": 10,
        }
        mock_http.post = AsyncMock(return_value=mock_resp)

        with patch("uni_vision.agent.navarasa_client.httpx.AsyncClient", return_value=mock_http):
            client = NavarasaClient(cfg)

        resp = await client.chat("Do something", "Some input")
        assert resp.content == "test response"
        assert resp.eval_count == 10

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        user_msg = payload["messages"][0]["content"]
        assert "### Instruction:" in user_msg
        assert "### Input:" in user_msg
        assert "### Response:" in user_msg
        assert payload["model"] == "uni-vision-navarasa"
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_translate_calls_chat(self):
        """translate() should delegate to chat() with correct instruction."""
        client = self._make_client()
        client.chat = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "अनुवादित पाठ"
        client.chat.return_value = mock_resp

        result = await client.translate("Hello world", "hi")
        assert result == "अनुवादित पाठ"
        client.chat.assert_awaited_once()
        assert "Hindi" in str(client.chat.call_args)

    @pytest.mark.asyncio
    async def test_translate_to_english_calls_chat(self):
        """translate_to_english() should translate to English."""
        client = self._make_client()
        client.chat = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "Translated text"
        client.chat.return_value = mock_resp

        result = await client.translate_to_english("नमस्ते दुनिया", "hi")
        assert result == "Translated text"
        client.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_translate_alert_returns_dict(self):
        """translate_alert() should return structured alert translation."""
        client = self._make_client()
        client.chat = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "प्लेट पता चली"
        client.chat.return_value = mock_resp

        result = await client.translate_alert("Plate detected: MH12AB1234", "hi")
        assert isinstance(result, dict)
        assert result["original"] == "Plate detected: MH12AB1234"
        assert result["translated"] == "प्लेट पता चली"
        assert result["language"] == "hi"
        assert result["language_name"] == "Hindi"

    @pytest.mark.asyncio
    async def test_translate_alert_unsupported_language_passthrough(self):
        """Unsupported language should return original text."""
        client = self._make_client()
        result = await client.translate_alert("Plate detected", "fr")
        assert result["translated"] == "Plate detected"
        assert result["language"] == "fr"

    @pytest.mark.asyncio
    async def test_translate_alert_english_passthrough(self):
        """English target should skip translation."""
        client = self._make_client()
        result = await client.translate_alert("Plate detected", "en")
        assert result["translated"] == "Plate detected"
        assert result["language"] == "en"

    # ── Conversational interface tests ─────────────────────────────

    @pytest.mark.asyncio
    async def test_converse_delegates_to_chat(self):
        """converse() should call chat() with conversational instruction."""
        client = self._make_client()
        client.chat = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "नमस्ते! मैं आपकी मदद कर सकती हूँ।"
        mock_resp.language = "en"  # will be overwritten
        client.chat.return_value = mock_resp

        resp = await client.converse("कैसे हो", "hi")
        assert resp.content == "नमस्ते! मैं आपकी मदद कर सकती हूँ।"
        assert resp.language == "hi"
        client.chat.assert_awaited_once()
        call_args = client.chat.call_args
        assert "Hindi" in str(call_args)

    @pytest.mark.asyncio
    async def test_converse_uses_default_language(self):
        """converse() with no language should use config default."""
        client = self._make_client(default_language="ta")
        client.chat = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "வணக்கம்!"
        mock_resp.language = "en"
        client.chat.return_value = mock_resp

        resp = await client.converse("வணக்கம்")
        assert resp.language == "ta"
        client.chat.assert_awaited_once()
        assert "Tamil" in str(client.chat.call_args)

    @pytest.mark.asyncio
    async def test_converse_with_system_context(self):
        """converse() should include system_context in the instruction."""
        client = self._make_client()
        client.chat = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "2 cameras active"
        mock_resp.language = "en"
        client.chat.return_value = mock_resp

        await client.converse(
            "What's happening?",
            "hi",
            system_context="2 cameras active, 5 plates detected",
        )
        instruction_arg = client.chat.call_args.kwargs.get(
            "instruction"
        ) or client.chat.call_args[0][0]
        assert "2 cameras active" in instruction_arg

    @pytest.mark.asyncio
    async def test_converse_custom_max_tokens(self):
        """converse() should forward max_tokens override."""
        client = self._make_client()
        client.chat = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "ok"
        mock_resp.language = "en"
        client.chat.return_value = mock_resp

        await client.converse("Hi", "hi", max_tokens=1024)
        call_kwargs = client.chat.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 1024

    @pytest.mark.asyncio
    async def test_close_closes_http_client(self):
        """close() should close the underlying httpx client."""
        from uni_vision.agent.navarasa_client import NavarasaClient

        cfg = NavarasaConfig()
        mock_http = AsyncMock()
        with patch("uni_vision.agent.navarasa_client.httpx.AsyncClient", return_value=mock_http):
            client = NavarasaClient(cfg)

        await client.close()
        mock_http.aclose.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════
# ──  Language registry tests  ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════


class TestLanguageRegistry:
    """Verify the LANGUAGE_NAMES registry is comprehensive."""

    def test_language_names_count(self):
        from uni_vision.agent.navarasa_client import LANGUAGE_NAMES
        assert len(LANGUAGE_NAMES) == 16
        assert "hi" in LANGUAGE_NAMES
        assert "en" in LANGUAGE_NAMES

    def test_all_scheduled_languages_present(self):
        from uni_vision.agent.navarasa_client import LANGUAGE_NAMES
        expected = {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur", "ne", "sd", "kok", "en"}
        assert set(LANGUAGE_NAMES.keys()) == expected
