"""Protocol contract for result dispatchers (S8 — dispatch phase)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uni_vision.contracts.dtos import DetectionRecord


@runtime_checkable
class Dispatcher(Protocol):
    """Persists and distributes a finalised detection record.

    Implementations: MultiDispatcher (PostgreSQL + S3 + Redis Pub/Sub),
    LogOnlyDispatcher (dev/test — writes to structlog only).

    Dispatch is async because it involves I/O to the database,
    object store, and message broker.
    """

    async def dispatch(self, record: DetectionRecord) -> None:
        """Persist the record and notify downstream consumers.

        Args:
            record: Fully validated detection record ready for storage.

        Raises:
            DispatchError: If all persistence targets fail after retries.
        """
        ...
