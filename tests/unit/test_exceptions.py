"""Tests for the exception hierarchy — spec §13 failure taxonomy F01–F10."""

from __future__ import annotations

import pytest


class TestExceptionHierarchy:
    """Verify the exception class tree."""

    def test_root_exception(self):
        from uni_vision.common.exceptions import UniVisionError

        assert issubclass(UniVisionError, Exception)

    def test_stream_errors(self):
        from uni_vision.common.exceptions import (
            StreamError,
            StreamReconnectExhausted,
            UniVisionError,
        )

        assert issubclass(StreamError, UniVisionError)
        assert issubclass(StreamReconnectExhausted, StreamError)

    def test_queue_overflow(self):
        from uni_vision.common.exceptions import QueueOverflowError, UniVisionError

        assert issubclass(QueueOverflowError, UniVisionError)

    def test_detection_errors(self):
        from uni_vision.common.exceptions import (
            DetectionError,
            NoPlateDetected,
            NoVehicleDetected,
            UniVisionError,
        )

        assert issubclass(DetectionError, UniVisionError)
        assert issubclass(NoVehicleDetected, DetectionError)
        assert issubclass(NoPlateDetected, DetectionError)

    def test_vram_errors(self):
        from uni_vision.common.exceptions import (
            UniVisionError,
            VRAMBudgetExceeded,
            VRAMError,
        )

        assert issubclass(VRAMError, UniVisionError)
        assert issubclass(VRAMBudgetExceeded, VRAMError)

    def test_vram_budget_exceeded_message(self):
        from uni_vision.common.exceptions import VRAMBudgetExceeded

        exc = VRAMBudgetExceeded("region_A", 5120.0, 5500.0)
        assert exc.region == "region_A"
        assert exc.budget_mb == 5120.0
        assert exc.used_mb == 5500.0
        assert "region_A" in str(exc)
        assert "5500.0" in str(exc)

    def test_ollama_errors(self):
        from uni_vision.common.exceptions import (
            OllamaCircuitOpen,
            OllamaError,
            OllamaTimeoutError,
            UniVisionError,
        )

        assert issubclass(OllamaError, UniVisionError)
        assert issubclass(OllamaTimeoutError, OllamaError)
        assert issubclass(OllamaCircuitOpen, OllamaError)

    def test_llm_errors(self):
        from uni_vision.common.exceptions import (
            LLMParseError,
            LLMRepetitionError,
            UniVisionError,
        )

        assert issubclass(LLMParseError, UniVisionError)
        assert issubclass(LLMRepetitionError, UniVisionError)

    def test_storage_errors(self):
        from uni_vision.common.exceptions import (
            DatabaseConnectionError,
            DatabaseWriteError,
            ObjectStoreError,
            StorageError,
            UniVisionError,
        )

        assert issubclass(StorageError, UniVisionError)
        assert issubclass(DatabaseWriteError, StorageError)
        assert issubclass(DatabaseConnectionError, StorageError)
        assert issubclass(ObjectStoreError, StorageError)

    def test_all_are_catchable(self):
        """Every pipeline exception should be catchable as UniVisionError."""
        from uni_vision.common.exceptions import (
            DatabaseWriteError,
            LLMParseError,
            NoPlateDetected,
            OllamaTimeoutError,
            StreamReconnectExhausted,
            UniVisionError,
            VRAMBudgetExceeded,
        )

        simple_exceptions = [
            StreamReconnectExhausted,
            NoPlateDetected,
            OllamaTimeoutError,
            LLMParseError,
            DatabaseWriteError,
        ]
        for exc_cls in simple_exceptions:
            try:
                raise exc_cls("test")
            except UniVisionError:
                pass  # expected
            except Exception:
                pytest.fail(f"{exc_cls.__name__} not caught by UniVisionError")

        # VRAMBudgetExceeded requires (region, budget_mb, used_mb)
        try:
            raise VRAMBudgetExceeded("A", 5120.0, 6000.0)
        except UniVisionError:
            pass
        except Exception:
            pytest.fail("VRAMBudgetExceeded not caught by UniVisionError")
