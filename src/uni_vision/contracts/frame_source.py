"""Protocol contract for stream frame sources (S0)."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from uni_vision.contracts.dtos import FramePacket


@runtime_checkable
class FrameSource(Protocol):
    """Reads frames from a video stream source.

    Implementations: OpenCV VideoCapture, FFmpeg subprocess, test stub.
    Each FrameSource instance is bound to exactly one camera source.
    """

    @property
    def camera_id(self) -> str:
        """Unique identifier of the bound camera."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the underlying stream is currently connected."""
        ...

    def read_frame(self) -> Optional[FramePacket]:
        """Read the next frame from the source.

        Returns ``None`` when the stream is exhausted or temporarily
        unavailable (reconnection is handled internally).
        """
        ...

    def release(self) -> None:
        """Release all resources held by this source (file handles, sockets)."""
        ...
