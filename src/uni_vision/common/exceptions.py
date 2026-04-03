"""Custom exception hierarchy — spec §13 failure taxonomy (F01–F13).

Each exception class maps to a specific failure class from the
architecture specification.  Recovery strategies are documented on
each class as a guideline for the orchestrator and circuit breakers.
"""

from __future__ import annotations


class UniVisionError(Exception):
    """Root exception for every recoverable error in the pipeline."""


# ── F01  Camera stream disconnection ──────────────────────────────


class StreamError(UniVisionError):
    """F01 — Camera stream unreachable or connection lost.

    Recovery: exponential backoff reconnect (1 s → 16 s).
    Alert after 3 consecutive failures.
    """


class StreamReconnectExhausted(StreamError):
    """All reconnection attempts for a camera source have been exhausted."""


# ── F02  Frame queue overflow ─────────────────────────────────────


class QueueOverflowError(UniVisionError):
    """F02 — Frame evicted from a full ring-buffer queue.

    Recovery: oldest frame evicted; ``frames_dropped`` counter incremented;
    adaptive FPS throttle engaged.
    """


# ── F03 / F04  Detection misses ──────────────────────────────────


class DetectionError(UniVisionError):
    """Base for detection-stage failures (F03, F04)."""


class NoVehicleDetected(DetectionError):
    """F03 — No vehicle found in frame.  Severity: INFO.

    Recovery: frame discarded, logged to audit trail.
    """


class NoPlateDetected(DetectionError):
    """F04 — No plate found within the vehicle ROI.  Severity: INFO.

    Recovery: event discarded, logged.
    """


# ── F05  VRAM overflow ────────────────────────────────────────────


class VRAMError(UniVisionError):
    """F05 — GPU out-of-memory or VRAM budget exceeded.

    Recovery: immediate CPU fallback for the current event,
    ``torch.cuda.empty_cache()``, alert raised.
    Persistent → degrade to Mode 3 (full CPU).
    """


class VRAMBudgetExceeded(VRAMError):
    """A VRAM region has exceeded its allocated budget."""

    def __init__(self, region: str, budget_mb: float, used_mb: float) -> None:
        self.region = region
        self.budget_mb = budget_mb
        self.used_mb = used_mb
        super().__init__(f"VRAM region '{region}' exceeded budget: {used_mb:.1f} MB used / {budget_mb:.1f} MB budget")


# ── F06  Ollama timeout ──────────────────────────────────────────


class OllamaError(UniVisionError):
    """F06 — Ollama service unresponsive or returned an unexpected status.

    Recovery: 5 s timeout → skip LLM OCR, route to fallback engine.
    Circuit breaker opens after 3 consecutive timeouts.
    """


class OllamaTimeoutError(OllamaError):
    """Ollama HTTP request exceeded the configured timeout."""


class OllamaCircuitOpen(OllamaError):
    """Circuit breaker is OPEN — all OCR requests bypass Ollama."""


# ── F07  LLM parse failure ────────────────────────────────────────


class LLMParseError(UniVisionError):
    """F07 — LLM output could not be parsed into the expected schema.

    Recovery: append error to context, re-prompt (max 2 retries).
    All retries exhausted → ``PARSE_FAIL`` status.
    """


# ── F08  LLM repetition loop ─────────────────────────────────────


class LLMRepetitionError(UniVisionError):
    """F08 — LLM entered a repetition loop (output > 2× expected tokens).

    Recovery: abort request, route to fallback OCR engine.
    """


# ── F09  Database write failure ───────────────────────────────────


class StorageError(UniVisionError):
    """Base for persistence-layer failures (F09, F10)."""


class DatabaseWriteError(StorageError):
    """F09 — PostgreSQL write failed after retries.

    Recovery: retry with exponential backoff (3 attempts).
    Buffer in-memory (max 100 records).
    Alert on persistent failure.
    """


class DatabaseConnectionError(StorageError):
    """Unable to establish a database connection."""


# ── F10  Object store upload failure ──────────────────────────────


class ObjectStoreError(StorageError):
    """F10 — S3 / MinIO image upload failed.

    Recovery: retry 2×.  On failure → store path as ``upload_pending``,
    background job retries later.
    """


# ── Configuration errors ──────────────────────────────────────────


class ConfigurationError(UniVisionError):
    """YAML or environment configuration is invalid or missing."""


# ── Pipeline lifecycle ────────────────────────────────────────────


class PipelineShutdownError(UniVisionError):
    """Error during graceful pipeline shutdown (resource release)."""


# ── F11  Agent errors ─────────────────────────────────────────────


class AgentError(UniVisionError):
    """F11 — Base exception for the agentic reasoning sub-system.

    Recovery: return partial result or fallback to simpler heuristic.
    """


class ToolExecutionError(AgentError):
    """F12 — A tool invoked by the agent failed during execution.

    Recovery: the agent receives the error in its observation and
    can retry with different parameters or choose another tool.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' failed: {reason}")


class AgentTimeoutError(AgentError):
    """F13 — Agent reasoning loop exceeded the iteration limit.

    Recovery: return the best partial answer collected so far.
    """
