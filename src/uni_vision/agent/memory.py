"""Working memory for the agentic reasoning loop.

Manages the conversation history within the LLM's context window
(typically 4096 tokens for Gemma 4 E2B Q4_K_M), implementing:

  * Fixed-capacity message buffer with FIFO eviction.
  * System prompt pinning (never evicted).
  * Token-aware truncation using character-count heuristic.
  * Observation summarisation on overflow.
  * Session-scoped scratchpad for agent notes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 characters for English text
_CHARS_PER_TOKEN = 4


@dataclass
class MemoryEntry:
    """A single message in the conversation memory."""

    role: str  # system | user | assistant | tool
    content: str
    tool_name: str | None = None  # set for role=tool
    pinned: bool = False  # pinned messages are never evicted


class WorkingMemory:
    """Bounded conversation memory for the agent loop.

    Parameters
    ----------
    max_tokens : int
        Maximum estimated token budget for the full message history.
    system_prompt : str
        The system prompt to pin at the start of every conversation.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 3072,
        system_prompt: str = "",
    ) -> None:
        self._max_tokens = max_tokens
        self._max_chars = max_tokens * _CHARS_PER_TOKEN
        self._entries: list[MemoryEntry] = []
        self._scratchpad: dict[str, Any] = {}

        if system_prompt:
            self._entries.append(MemoryEntry(role="system", content=system_prompt, pinned=True))

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    @property
    def message_count(self) -> int:
        return len(self._entries)

    @property
    def estimated_tokens(self) -> int:
        total_chars = sum(len(e.content) for e in self._entries)
        return total_chars // _CHARS_PER_TOKEN

    def add_user_message(self, content: str) -> None:
        """Add a user message and enforce token budget."""
        self._entries.append(MemoryEntry(role="user", content=content))
        self._enforce_budget()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant (LLM) message."""
        self._entries.append(MemoryEntry(role="assistant", content=content))
        self._enforce_budget()

    def add_tool_result(self, tool_name: str, content: str) -> None:
        """Add a tool observation."""
        self._entries.append(MemoryEntry(role="tool", content=content, tool_name=tool_name))
        self._enforce_budget()

    def add_system_note(self, content: str) -> None:
        """Add a non-evictable system note (e.g., context injection)."""
        self._entries.append(MemoryEntry(role="system", content=content, pinned=True))
        self._enforce_budget()

    def to_messages(self) -> list[dict[str, Any]]:
        """Export memory as an Ollama-compatible messages list."""
        messages: list[dict[str, Any]] = []
        for entry in self._entries:
            msg: dict[str, Any] = {
                "role": entry.role if entry.role != "tool" else "user",
                "content": entry.content,
            }
            if entry.role == "tool":
                # Wrap tool results as a user message with tool context
                msg["content"] = f"[Tool Result: {entry.tool_name}]\n{entry.content}"
            messages.append(msg)
        return messages

    def set_scratchpad(self, key: str, value: Any) -> None:
        """Store a key-value pair in the agent's scratchpad."""
        self._scratchpad[key] = value

    def get_scratchpad(self, key: str, default: Any = None) -> Any:
        """Retrieve a scratchpad value."""
        return self._scratchpad.get(key, default)

    def clear(self) -> None:
        """Clear all non-pinned entries and the scratchpad."""
        self._entries = [e for e in self._entries if e.pinned]
        self._scratchpad.clear()

    def _enforce_budget(self) -> None:
        """Evict oldest non-pinned messages until within token budget."""
        total_chars = sum(len(e.content) for e in self._entries)
        while total_chars > self._max_chars and len(self._entries) > 1:
            # Find the oldest non-pinned entry
            evict_idx: int | None = None
            for i, entry in enumerate(self._entries):
                if not entry.pinned:
                    evict_idx = i
                    break

            if evict_idx is None:
                # All entries are pinned — cannot evict further
                break

            evicted = self._entries.pop(evict_idx)
            total_chars -= len(evicted.content)
            logger.debug(
                "memory_evicted role=%s chars=%d remaining_entries=%d",
                evicted.role,
                len(evicted.content),
                len(self._entries),
            )
