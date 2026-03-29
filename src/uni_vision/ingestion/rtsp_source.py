"""RTSP / file-based frame source with automatic reconnection — spec §S0.

Each ``RTSPFrameSource`` owns a dedicated background I/O thread that
reads from the underlying ``cv2.VideoCapture`` and pushes decoded
frames into a per-camera ``queue.Queue`` (Layer-1 stream queue, §6.2).

Key behaviours:
  * Exponential backoff reconnection: 1 s → 2 s → 4 s → 8 s → 16 s (max).
  * After ``max_reconnect_attempts`` consecutive failures the source
    enters a permanent *disconnected* state and raises
    ``StreamReconnectExhausted``.
  * The read thread is a daemon — it will not block interpreter exit.
  * When the internal ring-buffer queue is full, the **oldest** frame
    is evicted so the pipeline always processes the most recent data.
  * ``release()`` tears down the capture and joins the reader thread.

OpenCV transport preferences
----------------------------
``cv2.VideoCapture`` is opened with the FFMPEG backend and the
``CAP_PROP_BUFFERSIZE=1`` hint so that the driver does not buffer
stale I-frames.  For RTSP, the transport is forced to TCP via the
``OPENCV_FFMPEG_CAPTURE_OPTIONS`` environment prelude (set once at
module load).

All frame data lives in system RAM (numpy ``uint8`` arrays).
No GPU VRAM is touched.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

from uni_vision.common.exceptions import StreamError, StreamReconnectExhausted
from uni_vision.contracts.dtos import CameraSource, FramePacket
from uni_vision.monitoring.metrics import FRAMES_INGESTED, STREAM_STATUS

log = structlog.get_logger()

# Force RTSP-over-TCP for reliability (prevents UDP packet loss).
# This env var is read by the FFmpeg backend inside OpenCV.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

_DEFAULT_MAX_RECONNECT = 10
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 16.0


class RTSPFrameSource:
    """Threaded RTSP / file video source implementing the ``FrameSource`` protocol.

    Parameters
    ----------
    camera:
        Parsed ``CameraSource`` DTO from ``cameras.yaml``.
    stream_queue_maxsize:
        Maximum depth of the per-camera ring-buffer queue (spec default 50).
    max_reconnect_attempts:
        Hard ceiling on consecutive reconnections before giving up.
    """

    def __init__(
        self,
        camera: CameraSource,
        *,
        stream_queue_maxsize: int = 50,
        max_reconnect_attempts: int = _DEFAULT_MAX_RECONNECT,
    ) -> None:
        self._camera = camera
        self._max_reconnect = max_reconnect_attempts

        # Per-camera ring-buffer queue (Layer-1, spec §6.2).
        self._queue: queue.Queue[FramePacket] = queue.Queue(
            maxsize=stream_queue_maxsize,
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index: int = 0
        self._connected = False
        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        # Dynamic throttle factor — the pipeline sets this to < 1.0
        # when the inference queue is under pressure (spec §6.4).
        self._throttle_factor: float = 1.0

    # ── FrameSource protocol properties ───────────────────────────

    @property
    def camera_id(self) -> str:
        return self._camera.camera_id

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Open the capture and spawn the background reader thread."""
        self._stop_event.clear()
        self._open_capture()
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name=f"stream-{self._camera.camera_id}",
            daemon=True,
        )
        self._reader_thread.start()

    def release(self) -> None:
        """Signal the reader thread to exit and release the capture."""
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
            self._reader_thread = None
        self._close_capture()

    def read_frame(self) -> Optional[FramePacket]:
        """Non-blocking read from the internal queue.

        Returns the next available ``FramePacket`` or ``None`` if the
        queue is empty (caller should retry on the next tick).
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    # ── Throttle interface (called by pipeline backpressure) ──────

    def throttle(self, factor: float) -> None:
        """Reduce effective FPS by *factor* (0.0–1.0)."""
        self._throttle_factor = max(0.1, min(factor, 1.0))
        log.debug(
            "stream_throttled",
            camera_id=self._camera.camera_id,
            factor=self._throttle_factor,
        )

    def unthrottle(self) -> None:
        """Restore full capture rate."""
        self._throttle_factor = 1.0

    # ── Internal: capture open / close ────────────────────────────

    def _open_capture(self) -> None:
        """Attempt to open ``cv2.VideoCapture`` with FFMPEG backend."""
        self._close_capture()
        cap = cv2.VideoCapture(self._camera.source_url, cv2.CAP_FFMPEG)
        # Minimise internal buffer to avoid stale frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            self._cap = cap
            self._connected = True
            STREAM_STATUS.labels(camera_id=self._camera.camera_id).set(1)
            log.info(
                "stream_connected",
                camera_id=self._camera.camera_id,
                source=self._camera.source_url,
            )
        else:
            cap.release()
            self._connected = False
            STREAM_STATUS.labels(camera_id=self._camera.camera_id).set(0)
            raise StreamError(
                f"Failed to open stream for {self._camera.camera_id}: "
                f"{self._camera.source_url}"
            )

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False
        STREAM_STATUS.labels(camera_id=self._camera.camera_id).set(0)

    # ── Internal: reconnection with exponential backoff ───────────

    def _reconnect(self) -> bool:
        """Try to restore the capture with exponential backoff.

        Returns ``True`` if reconnection succeeded, ``False`` if all
        attempts are exhausted.
        """
        for attempt in range(1, self._max_reconnect + 1):
            if self._stop_event.is_set():
                return False
            delay = min(_BACKOFF_BASE_S * (2 ** (attempt - 1)), _BACKOFF_MAX_S)
            log.warning(
                "stream_reconnecting",
                camera_id=self._camera.camera_id,
                attempt=attempt,
                delay_s=delay,
            )
            self._stop_event.wait(timeout=delay)
            if self._stop_event.is_set():
                return False
            try:
                self._open_capture()
                return True
            except StreamError:
                continue

        log.error(
            "stream_reconnect_exhausted",
            camera_id=self._camera.camera_id,
            attempts=self._max_reconnect,
        )
        return False

    # ── Internal: main read loop (runs in dedicated thread) ───────

    def _read_loop(self) -> None:
        """Continuous frame read with FPS gating and reconnection.

        This method never touches GPU VRAM.  Decoded frames are plain
        NumPy arrays (``uint8``, BGR) in system RAM.
        """
        fps_target = self._camera.fps_target
        if fps_target <= 0:
            fps_target = 15  # sane fallback

        while not self._stop_event.is_set():
            # Effective inter-frame interval, respecting throttle.
            effective_fps = max(1.0, fps_target * self._throttle_factor)
            interval = 1.0 / effective_fps

            # --- Grab frame ---
            if self._cap is None or not self._cap.isOpened():
                if not self._reconnect():
                    return  # permanent disconnect
                continue

            ok, raw_frame = self._cap.read()
            if not ok or raw_frame is None:
                self._connected = False
                STREAM_STATUS.labels(camera_id=self._camera.camera_id).set(0)
                log.warning("stream_read_failed", camera_id=self._camera.camera_id)
                if not self._reconnect():
                    return
                continue

            # Build immutable FramePacket
            packet = FramePacket(
                camera_id=self._camera.camera_id,
                timestamp_utc=time.time(),
                frame_index=self._frame_index,
                image=raw_frame,
            )
            self._frame_index += 1
            FRAMES_INGESTED.labels(camera_id=self._camera.camera_id).inc()

            # Ring-buffer eviction: drop oldest when full (§6.2)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            self._queue.put_nowait(packet)

            # FPS gate — sleep for the remaining interval period
            self._stop_event.wait(timeout=interval)
