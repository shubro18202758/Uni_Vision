"""Tests for the profiling decorator and ring buffer."""

from __future__ import annotations

import asyncio

import pytest


class TestProfileStage:
    """Test the @profile_stage decorator."""

    def test_sync_function_profiled(self):
        from uni_vision.monitoring.profiler import (
            clear_profile_history,
            get_profile_history,
            profile_stage,
            set_profiling_enabled,
        )

        set_profiling_enabled(True)
        clear_profile_history()

        @profile_stage("test_sync", track_vram=False)
        def dummy_fn(x: int) -> int:
            return x * 2

        result = dummy_fn(21)
        assert result == 42

        history = get_profile_history()
        assert len(history) == 1
        assert history[0].stage == "test_sync"
        assert history[0].wall_time_ms >= 0

    def test_async_function_profiled(self):
        from uni_vision.monitoring.profiler import (
            clear_profile_history,
            get_profile_history,
            profile_stage,
            set_profiling_enabled,
        )

        set_profiling_enabled(True)
        clear_profile_history()

        @profile_stage("test_async", track_vram=False)
        async def dummy_async(x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1

        result = asyncio.get_event_loop().run_until_complete(dummy_async(10))
        assert result == 11

        history = get_profile_history()
        assert len(history) == 1
        assert history[0].stage == "test_async"
        assert history[0].wall_time_ms >= 10.0  # at least 10ms sleep

    def test_profiling_disabled_bypasses(self):
        from uni_vision.monitoring.profiler import (
            clear_profile_history,
            get_profile_history,
            profile_stage,
            set_profiling_enabled,
        )

        set_profiling_enabled(False)
        clear_profile_history()

        @profile_stage("test_disabled", track_vram=False)
        def dummy_fn() -> str:
            return "hello"

        result = dummy_fn()
        assert result == "hello"

        # No profiling record should be created
        assert len(get_profile_history()) == 0

        # Restore
        set_profiling_enabled(True)

    def test_function_preserves_return_value(self):
        from uni_vision.monitoring.profiler import profile_stage

        @profile_stage("test_return", track_vram=False)
        def returns_dict() -> dict:
            return {"key": "value"}

        result = returns_dict()
        assert result == {"key": "value"}


class TestProfileHistory:
    """Test the ring buffer behaviour."""

    def test_history_is_copy(self):
        from uni_vision.monitoring.profiler import (
            clear_profile_history,
            get_profile_history,
        )

        clear_profile_history()
        h1 = get_profile_history()
        h1.append(None)  # type: ignore[arg-type]
        h2 = get_profile_history()
        assert len(h2) == 0  # original unmodified

    def test_ring_buffer_eviction(self):
        from uni_vision.monitoring.profiler import (
            StageProfile,
            _record_profile,
            clear_profile_history,
            get_profile_history,
        )

        clear_profile_history()

        # Fill beyond the 512 max
        for i in range(520):
            _record_profile(StageProfile(
                stage=f"s_{i}",
                wall_time_ms=1.0,
                vram_before_mb=0.0,
                vram_after_mb=0.0,
                vram_delta_mb=0.0,
            ))

        history = get_profile_history()
        assert len(history) == 512
        # Oldest entries were evicted — first entry should be s_8
        assert history[0].stage == "s_8"
        assert history[-1].stage == "s_519"

    def test_clear_history(self):
        from uni_vision.monitoring.profiler import (
            StageProfile,
            _record_profile,
            clear_profile_history,
            get_profile_history,
        )

        _record_profile(StageProfile(
            stage="x", wall_time_ms=1.0,
            vram_before_mb=0.0, vram_after_mb=0.0, vram_delta_mb=0.0,
        ))
        clear_profile_history()
        assert len(get_profile_history()) == 0


class TestVRAMQueryHelpers:
    """Test VRAM query graceful fallbacks (no GPU in CI)."""

    def test_query_vram_used_returns_float(self):
        from uni_vision.monitoring.profiler import _query_vram_used_mb

        # Without a real GPU, should return -1.0 gracefully
        result = _query_vram_used_mb()
        assert isinstance(result, float)

    def test_query_torch_allocated_returns_float(self):
        from uni_vision.monitoring.profiler import _query_torch_allocated_mb

        result = _query_torch_allocated_mb()
        assert isinstance(result, float)
