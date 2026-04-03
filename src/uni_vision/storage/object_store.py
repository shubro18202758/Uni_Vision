"""S3-compatible object-store archiver — spec §4 S8 storage layer.

Archives plate crop images (visual evidence) to an S3-compatible
endpoint (MinIO / AWS S3 / local FS fallback).  Images are stored
with a deterministic key derived from the detection record ID.

Uses ``aioboto3`` for fully async uploads so the dispatch task never
blocks the event loop.

Spec reference: §4 S8 — Images: S3-compatible store.
Failure taxonomy: F10 — ObjectStoreError.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

from uni_vision.common.exceptions import ObjectStoreError
from uni_vision.monitoring.metrics import DISPATCH_ERRORS

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from uni_vision.common.config import DispatchConfig, StorageConfig

logger = logging.getLogger(__name__)

# Lazy-load cv2 once at module level on first call, not per-image.
_cv2_module = None


def _get_cv2():
    global _cv2_module
    if _cv2_module is None:
        import cv2

        _cv2_module = cv2
    return _cv2_module


def _encode_image(image: NDArray[np.uint8], fmt: str) -> bytes:
    """Encode a BGR numpy array to PNG/JPEG bytes."""
    cv2 = _get_cv2()
    if fmt.lower() in ("jpg", "jpeg"):
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


class ObjectStoreArchiver:
    """Async S3-compatible image archiver.

    Parameters
    ----------
    storage_config : StorageConfig
        Endpoint, bucket, credentials, region.
    dispatch_config : DispatchConfig
        Upload timeout and retry parameters.
    """

    def __init__(
        self,
        storage_config: StorageConfig,
        dispatch_config: DispatchConfig,
    ) -> None:
        self._endpoint = storage_config.endpoint
        self._bucket = storage_config.bucket
        self._access_key = storage_config.access_key
        self._secret_key = storage_config.secret_key
        self._region = storage_config.region
        self._timeout = dispatch_config.image_upload_timeout_s
        self._max_retries = dispatch_config.max_retries
        self._retry_delay = dispatch_config.retry_base_delay_s
        self._image_fmt = dispatch_config.image_format
        self._session: object = None  # lazily created aioboto3.Session

    async def _get_client(self):
        """Lazily create an ``aioboto3`` S3 client."""
        import aioboto3  # type: ignore[import-untyped]

        if self._session is None:
            self._session = aioboto3.Session()
        return self._session.client(
            "s3",
            endpoint_url=self._endpoint,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            region_name=self._region,
        )

    async def ensure_bucket(self) -> None:
        """Create the target bucket if it does not already exist."""
        async with await self._get_client() as s3:
            try:
                await s3.head_bucket(Bucket=self._bucket)
            except Exception:
                await s3.create_bucket(Bucket=self._bucket)
                logger.info("object_store_bucket_created bucket=%s", self._bucket)

    async def upload_plate_image(
        self,
        record_id: str,
        camera_id: str,
        plate_image: NDArray[np.uint8],
    ) -> str:
        """Encode and upload a plate crop, returning the object key.

        Retries with exponential backoff on transient failures.

        Returns
        -------
        str
            The S3 object key (path within the bucket).
        """
        ext = self._image_fmt.lower()
        content_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        key = f"plates/{camera_id}/{record_id}.{ext}"
        body = _encode_image(plate_image, ext)

        delay = self._retry_delay
        last_exc: Exception | None = None

        for attempt in range(1 + self._max_retries):
            try:
                async with await self._get_client() as s3:
                    await asyncio.wait_for(
                        s3.put_object(
                            Bucket=self._bucket,
                            Key=key,
                            Body=io.BytesIO(body),
                            ContentType=content_type,
                        ),
                        timeout=self._timeout,
                    )
                logger.debug("image_uploaded key=%s size=%d", key, len(body))
                return key

            except (asyncio.TimeoutError, OSError, Exception) as exc:
                last_exc = exc
                logger.warning(
                    "image_upload_retry attempt=%d/%d key=%s err=%s",
                    attempt + 1,
                    1 + self._max_retries,
                    key,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2

        DISPATCH_ERRORS.labels(target="object_store").inc()
        raise ObjectStoreError(f"Image upload failed after {1 + self._max_retries} attempts: {last_exc}") from last_exc

    async def close(self) -> None:
        """Release resources (session is lightweight — no-op)."""
        self._session = None
