"""Session manager for persistent agent conversations.

Provides multi-turn conversation support by associating a session ID
with a running conversation context.  Each session keeps its own
WorkingMemory so that follow-up queries share conversational state
(e.g., "now filter that by camera-1").

Sessions are automatically purged after ``ttl_seconds`` of inactivity.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from uni_vision.agent.memory import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single exchange within a session."""

    role: str  # user | assistant
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: int = 0  # number of tool calls in this turn
    elapsed_ms: float = 0.0


@dataclass
class Session:
    """A persistent conversation session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_active

    def touch(self) -> None:
        """Update last-active timestamp."""
        self.last_active = time.time()

    def add_turn(
        self,
        role: str,
        content: str,
        tool_calls: int = 0,
        elapsed_ms: float = 0.0,
    ) -> ConversationTurn:
        turn = ConversationTurn(
            role=role,
            content=content,
            tool_calls=tool_calls,
            elapsed_ms=elapsed_ms,
        )
        self.turns.append(turn)
        self.touch()
        return turn

    def get_context_summary(self, max_turns: int = 6) -> str:
        """Build a condensed summary of recent conversation for context injection."""
        if not self.turns:
            return ""

        recent = self.turns[-max_turns:]
        lines: List[str] = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Agent"
            # Truncate long turns to keep context small
            text = t.content[:300] + "..." if len(t.content) > 300 else t.content
            lines.append(f"{prefix}: {text}")

        return "\n".join(lines)


class SessionManager:
    """Manages named conversation sessions with auto-expiry.

    Parameters
    ----------
    ttl_seconds:
        Maximum idle time before a session is purged (default 30 min).
    max_sessions:
        Hard limit on concurrent sessions (LRU eviction).
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 1800.0,
        max_sessions: int = 100,
    ) -> None:
        self._sessions: Dict[str, Session] = {}
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions

    def create_session(self, session_id: str | None = None) -> Session:
        """Create or return an existing session."""
        self._purge_expired()

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.touch()
            return session

        sid = session_id or uuid.uuid4().hex[:16]
        session = Session(session_id=sid)
        self._sessions[sid] = session

        # Evict oldest if at capacity
        if len(self._sessions) > self._max_sessions:
            self._evict_oldest()

        logger.debug("session_created id=%s", sid)
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve an existing session by ID, or None."""
        self._purge_expired()
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if found and deleted."""
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions (summary only)."""
        self._purge_expired()
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "last_active": s.last_active,
                "turn_count": s.turn_count,
                "idle_seconds": round(s.idle_seconds, 1),
            }
            for s in self._sessions.values()
        ]

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def _purge_expired(self) -> None:
        """Remove sessions that have exceeded the TTL."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.last_active) > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug("session_expired id=%s", sid)

    def _evict_oldest(self) -> None:
        """Evict the least-recently-used session."""
        if not self._sessions:
            return
        oldest_sid = min(self._sessions, key=lambda k: self._sessions[k].last_active)
        del self._sessions[oldest_sid]
        logger.debug("session_evicted id=%s", oldest_sid)
