"""System-wide health check logic.

Aggregates GPU telemetry, Ollama reachability, database connectivity,
and camera stream status into a single ``HealthStatus`` DTO.  Used by
the ``GET /health`` API endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import httpx
import structlog

from uni_vision.contracts.dtos import HealthStatus, OffloadMode
from uni_vision.monitoring.vram_monitor import VRAMMonitor

log = structlog.get_logger()


class HealthService:
    """Evaluates liveness and readiness of the Uni_Vision pipeline.

    Parameters
    ----------
    vram_monitor:
        Reference to the running VRAM monitor.
    ollama_url:
        Ollama base URL for the ``/api/tags`` liveness probe.
    db_dsn:
        PostgreSQL DSN for a lightweight connectivity check.
    """

    def __init__(
        self,
        *,
        vram_monitor: VRAMMonitor,
        ollama_url: str = "http://localhost:11434",
        db_dsn: str = "",
        stream_count: int = 0,
    ) -> None:
        self._vram_monitor = vram_monitor
        self._ollama_url = ollama_url.rstrip("/")
        self._db_dsn = db_dsn
        self._stream_count = stream_count
        self._connected_streams: int = 0

    def update_stream_status(self, connected: int, total: int) -> None:
        self._connected_streams = connected
        self._stream_count = total

    async def check(self) -> HealthStatus:
        """Run all sub-checks concurrently and return aggregated status."""
        gpu_ok, ollama_ok, db_ok = await asyncio.gather(
            self._check_gpu(),
            self._check_ollama(),
            self._check_database(),
        )

        details: Dict[str, str] = {}
        if not gpu_ok:
            details["gpu"] = "unavailable or pynvml not installed"
        if not ollama_ok:
            details["ollama"] = f"unreachable at {self._ollama_url}"
        if not db_ok:
            details["database"] = "connection failed"

        healthy = gpu_ok and ollama_ok and db_ok

        return HealthStatus(
            healthy=healthy,
            gpu_available=gpu_ok,
            ollama_reachable=ollama_ok,
            database_connected=db_ok,
            streams_connected=self._connected_streams,
            streams_total=self._stream_count,
            offload_mode=self._vram_monitor.offload_mode,
            details=details,
        )

    # ── Sub-checks ────────────────────────────────────────────────

    async def _check_gpu(self) -> bool:
        telemetry = self._vram_monitor.latest_telemetry()
        return telemetry is not None

    async def _check_ollama(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def _check_database(self) -> bool:
        if not self._db_dsn:
            return False
        try:
            import asyncpg

            conn = await asyncio.wait_for(
                asyncpg.connect(self._db_dsn),
                timeout=3.0,
            )
            await conn.execute("SELECT 1")
            await conn.close()
            return True
        except Exception:
            return False
