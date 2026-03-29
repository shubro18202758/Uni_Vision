"""Temporal frame sampler with pHash deduplication — spec §S1.

``TemporalSampler`` sits between the per-camera stream queues
(Layer 1, §6.2) and the central inference queue (Layer 2, §6.3).
It enforces two distinct filters on the raw frame stream:

1. **FPS gating** — only one frame per ``1 / fps_target`` second
   interval is forwarded.  Intermediate frames are silently skipped.
2. **Perceptual deduplication** — the pHash of every FPS-accepted
   frame is compared to the last *forwarded* hash for that camera.
   If the Hamming distance is ≤ ``hamming_threshold``, the frame is
   discarded as a near-duplicate (stationary scene, parked vehicle,
   etc.).

Both operations are pure CPU / system-RAM — zero GPU VRAM impact.

Dynamic configuration
---------------------
* ``hamming_threshold`` is exposed as a property and can be adjusted
  at runtime (e.g., through the REST API or a hot-reload hook).
* ``throttle_factor`` (0.1–1.0) is propagated from the pipeline's
  adaptive throttle mechanism (spec §6.4) and widens the FPS gate.

The sampler operates in its own long-lived thread, polling all
registered ``RTSPFrameSource`` instances in a weighted round-robin
and dispatching accepted frames to a callback (typically
``Pipeline.enqueue_frame``).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import structlog

from uni_vision.contracts.dtos import FramePacket
from uni_vision.ingestion.phash import compute_phash, hamming_distance
from uni_vision.ingestion.rtsp_source import RTSPFrameSource
from uni_vision.monitoring.metrics import FRAMES_DEDUPLICATED

log = structlog.get_logger()


class TemporalSampler:
    """FPS-gated, pHash-deduplicated frame dispatcher.

    Parameters
    ----------
    sources:
        Live ``RTSPFrameSource`` instances (one per enabled camera).
    enqueue_callback:
        Async callable that accepts a ``FramePacket`` and submits it
        to the inference queue.  Expected signature matches
        ``Pipeline.enqueue_frame``.
    event_loop:
        The running ``asyncio`` event loop — required so the sampler
        thread can schedule the async callback via
        ``loop.call_soon_threadsafe``.
    hamming_threshold:
        Maximum Hamming distance at which two pHashes are considered
        duplicates.  Default **5** (spec ``frame_sampling.hamming_distance_threshold``).
    phash_size:
        Side length of the DCT tile used by the hasher (default 32).
    poll_interval_s:
        Minimum sleep between round-robin polling ticks when no
        frames are available (prevents busy-spin).
    """

    def __init__(
        self,
        sources: List[RTSPFrameSource],
        *,
        enqueue_callback: Callable[[FramePacket], object],
        event_loop: asyncio.AbstractEventLoop,
        hamming_threshold: int = 5,
        phash_size: int = 32,
        poll_interval_s: float = 0.002,
    ) -> None:
        self._sources = sources
        self._enqueue = enqueue_callback
        self._loop = event_loop
        self._phash_size = phash_size
        self._poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Configurable at runtime
        self._hamming_threshold = hamming_threshold

        # Per-camera state: last accepted pHash & timestamp
        self._last_hash: Dict[str, np.uint64] = {}
        self._last_accept_ts: Dict[str, float] = {}

    # ── Public configuration knobs ────────────────────────────────

    @property
    def hamming_threshold(self) -> int:
        return self._hamming_threshold

    @hamming_threshold.setter
    def hamming_threshold(self, value: int) -> None:
        self._hamming_threshold = max(0, value)
        log.info("hamming_threshold_updated", value=self._hamming_threshold)

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the dispatcher thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="temporal-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the dispatcher thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ── Internal: dispatch loop ───────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Round-robin poll all sources; gate by FPS and deduplicate."""
        while not self._stop_event.is_set():
            any_frame = False

            for source in self._sources:
                if self._stop_event.is_set():
                    return

                frame = source.read_frame()
                if frame is None:
                    continue

                any_frame = True
                cam = frame.camera_id

                # ── FPS gating ────────────────────────────────────
                # Compute the effective interval from the source's
                # *current* throttle-adjusted FPS.
                fps = max(1.0, source._camera.fps_target * source._throttle_factor)
                min_interval = 1.0 / fps
                now = time.monotonic()
                last_ts = self._last_accept_ts.get(cam, 0.0)
                if (now - last_ts) < min_interval:
                    continue  # too soon — skip

                # ── Perceptual deduplication ──────────────────────
                current_hash = compute_phash(
                    frame.image,
                    hash_size=self._phash_size,
                )

                prev_hash = self._last_hash.get(cam)
                if prev_hash is not None:
                    dist = hamming_distance(current_hash, prev_hash)
                    if dist <= self._hamming_threshold:
                        FRAMES_DEDUPLICATED.labels(camera_id=cam).inc()
                        log.debug(
                            "frame_deduplicated",
                            camera_id=cam,
                            hamming=dist,
                            threshold=self._hamming_threshold,
                        )
                        continue  # near-duplicate — discard

                # ── Accept frame ──────────────────────────────────
                self._last_hash[cam] = current_hash
                self._last_accept_ts[cam] = now

                # Schedule the async enqueue on the event loop from
                # this worker thread.
                asyncio.run_coroutine_threadsafe(
                    self._enqueue(frame),
                    self._loop,
                )

            # Avoid busy-spin when all queues are empty.
            if not any_frame:
                self._stop_event.wait(timeout=self._poll_interval_s)
