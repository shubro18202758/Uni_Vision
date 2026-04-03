"""Unit tests for JobLifecycleManager.

Covers:
  - Job creation and lookup
  - Dynamic component registration
  - Anomaly-driven completion logic
  - Job flushing (unload + pip uninstall)
  - Phase transitions and broadcaster events
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uni_vision.manager.job_lifecycle import (
    JobLifecycleConfig,
    JobLifecycleManager,
    JobPhase,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def mock_registry():
    reg = MagicMock()
    reg.unregister = MagicMock()
    return reg


@pytest.fixture()
def mock_lifecycle():
    lm = MagicMock()
    lm.unload_component = AsyncMock(return_value=True)
    return lm


@pytest.fixture()
def mock_broadcaster():
    bc = MagicMock()
    bc.emit_custom = AsyncMock()
    return bc


@pytest.fixture()
def config():
    return JobLifecycleConfig(
        post_anomaly_stable_threshold=3,
        max_frames_per_job=20,
        uninstall_pip_packages=True,
        flush_timeout_s=10.0,
    )


@pytest.fixture()
def manager(mock_registry, mock_lifecycle, mock_broadcaster, config):
    return JobLifecycleManager(
        registry=mock_registry,
        lifecycle=mock_lifecycle,
        config=config,
        broadcaster=mock_broadcaster,
    )


# ── Job creation ──────────────────────────────────────────────────


class TestJobCreation:
    @pytest.mark.asyncio
    async def test_create_job(self, manager):
        job = await manager.create_job("j1", "cam-01")
        assert job.job_id == "j1"
        assert job.camera_id == "cam-01"
        assert job.phase == JobPhase.INITIALIZING
        assert job.flushed is False

    @pytest.mark.asyncio
    async def test_create_job_broadcasts(self, manager, mock_broadcaster):
        await manager.create_job("j1", "cam-01")
        mock_broadcaster.emit_custom.assert_called_once()
        call_args = mock_broadcaster.emit_custom.call_args
        assert call_args.kwargs["event_type"] == "job_created"
        assert call_args.kwargs["data"]["job_id"] == "j1"

    @pytest.mark.asyncio
    async def test_get_job(self, manager):
        await manager.create_job("j1", "cam-01")
        assert manager.get_job("j1") is not None
        assert manager.get_job("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_job_for_camera(self, manager):
        await manager.create_job("j1", "cam-01")
        result = manager.get_job_for_camera("cam-01")
        assert result is not None
        assert result.job_id == "j1"

    @pytest.mark.asyncio
    async def test_get_job_for_camera_ignores_completed(self, manager):
        job = await manager.create_job("j1", "cam-01")
        job.phase = JobPhase.COMPLETED
        result = manager.get_job_for_camera("cam-01")
        assert result is None


# ── Dynamic component registration ───────────────────────────────


class TestComponentRegistration:
    @pytest.mark.asyncio
    async def test_register_component(self, manager):
        await manager.create_job("j1", "cam-01")
        await manager.register_dynamic_component("j1", "yolov8_detect")
        job = manager.get_job("j1")
        assert "yolov8_detect" in job.dynamic_components

    @pytest.mark.asyncio
    async def test_register_component_with_pip(self, manager):
        await manager.create_job("j1", "cam-01")
        await manager.register_dynamic_component("j1", "yolov8_detect", pip_package="ultralytics")
        job = manager.get_job("j1")
        assert "ultralytics" in job.dynamic_pip_packages

    @pytest.mark.asyncio
    async def test_register_component_broadcasts(self, manager, mock_broadcaster):
        await manager.create_job("j1", "cam-01")
        mock_broadcaster.emit_custom.reset_mock()
        await manager.register_dynamic_component("j1", "comp-a", pip_package="pkg-a")
        mock_broadcaster.emit_custom.assert_called_once()
        data = mock_broadcaster.emit_custom.call_args.kwargs["data"]
        assert data["component_id"] == "comp-a"
        assert data["pip_package"] == "pkg-a"

    @pytest.mark.asyncio
    async def test_register_unknown_job_is_noop(self, manager):
        # Should not raise
        await manager.register_dynamic_component("no-such-job", "comp")


# ── Phase transitions ─────────────────────────────────────────────


class TestPhaseTransitions:
    @pytest.mark.asyncio
    async def test_update_phase(self, manager):
        await manager.create_job("j1", "cam-01")
        await manager.update_phase("j1", JobPhase.DISCOVERING)
        assert manager.get_job("j1").phase == JobPhase.DISCOVERING

    @pytest.mark.asyncio
    async def test_update_phase_broadcasts(self, manager, mock_broadcaster):
        await manager.create_job("j1", "cam-01")
        mock_broadcaster.emit_custom.reset_mock()
        await manager.update_phase("j1", JobPhase.PROVISIONING)
        mock_broadcaster.emit_custom.assert_called_once()
        data = mock_broadcaster.emit_custom.call_args.kwargs["data"]
        assert data["old_phase"] == "initializing"
        assert data["new_phase"] == "provisioning"


# ── Anomaly-driven completion ─────────────────────────────────────


class TestAnomalyCompletion:
    @pytest.mark.asyncio
    async def test_no_anomaly_no_complete(self, manager):
        """Normal frames don't trigger completion."""
        await manager.create_job("j1", "cam-01")
        for _ in range(5):
            done = await manager.record_frame_result("j1", anomaly_detected=False)
            assert done is False

    @pytest.mark.asyncio
    async def test_anomaly_then_stable_triggers_complete(self, manager, config):
        """After anomaly + N stable frames, job should complete."""
        await manager.create_job("j1", "cam-01")

        # Detect anomaly
        await manager.record_frame_result("j1", anomaly_detected=True)
        job = manager.get_job("j1")
        assert job.phase == JobPhase.ANOMALY_DETECTED

        # Send stable frames (threshold is 3)
        for _i in range(config.post_anomaly_stable_threshold - 1):
            done = await manager.record_frame_result("j1", anomaly_detected=False)
            assert done is False

        # The Nth stable frame triggers completion
        done = await manager.record_frame_result("j1", anomaly_detected=False)
        assert done is True
        assert job.phase == JobPhase.COMPLETING
        assert job.anomaly.anomaly_fully_analysed is True

    @pytest.mark.asyncio
    async def test_anomaly_reset_stable_counter(self, manager, config):
        """New anomaly resets the stable counter."""
        await manager.create_job("j1", "cam-01")

        await manager.record_frame_result("j1", anomaly_detected=True)
        # 2 stable frames (threshold=3)
        await manager.record_frame_result("j1", anomaly_detected=False)
        await manager.record_frame_result("j1", anomaly_detected=False)
        # Another anomaly resets counter
        await manager.record_frame_result("j1", anomaly_detected=True)

        job = manager.get_job("j1")
        assert job.anomaly.post_anomaly_stable_frames == 0
        assert job.phase != JobPhase.COMPLETING

    @pytest.mark.asyncio
    async def test_max_frames_forces_completion(self, manager, config):
        """Max frames limit triggers completion even without anomalies."""
        await manager.create_job("j1", "cam-01")
        for _i in range(config.max_frames_per_job - 1):
            done = await manager.record_frame_result("j1", anomaly_detected=False)
            assert done is False
        done = await manager.record_frame_result("j1", anomaly_detected=False)
        assert done is True

    @pytest.mark.asyncio
    async def test_anomaly_data_stored(self, manager):
        """Anomaly data dicts are accumulated."""
        await manager.create_job("j1", "cam-01")
        await manager.record_frame_result("j1", anomaly_detected=True, anomaly_data={"type": "fire"})
        await manager.record_frame_result("j1", anomaly_detected=True, anomaly_data={"type": "smoke"})
        job = manager.get_job("j1")
        assert len(job.anomaly.anomaly_results) == 2
        assert job.anomaly.anomaly_results[0]["type"] == "fire"

    @pytest.mark.asyncio
    async def test_record_unknown_job(self, manager):
        done = await manager.record_frame_result("nope", anomaly_detected=False)
        assert done is False


# ── Flushing ──────────────────────────────────────────────────────


class TestFlushJob:
    @pytest.mark.asyncio
    async def test_flush_unloads_components(self, manager, mock_lifecycle, mock_registry):
        await manager.create_job("j1", "cam-01")
        await manager.register_dynamic_component("j1", "comp-a")
        await manager.register_dynamic_component("j1", "comp-b")

        summary = await manager.flush_job("j1")

        assert set(summary["unloaded"]) == {"comp-a", "comp-b"}
        assert mock_lifecycle.unload_component.call_count == 2
        assert mock_registry.unregister.call_count == 2

    @pytest.mark.asyncio
    async def test_flush_uninstalls_pip(self, manager):
        await manager.create_job("j1", "cam-01")
        await manager.register_dynamic_component("j1", "comp", pip_package="ultralytics")

        with patch("uni_vision.manager.job_lifecycle.subprocess") as mock_sub:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_sub.run.return_value = mock_result
            summary = await manager.flush_job("j1")

        assert "ultralytics" in summary["uninstalled"]

    @pytest.mark.asyncio
    async def test_flush_marks_completed(self, manager):
        await manager.create_job("j1", "cam-01")
        await manager.flush_job("j1")
        job = manager.get_job("j1")
        assert job.flushed is True
        assert job.phase == JobPhase.COMPLETED
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_flush_already_flushed(self, manager):
        await manager.create_job("j1", "cam-01")
        await manager.flush_job("j1")
        summary = await manager.flush_job("j1")
        assert summary.get("already_flushed") is True

    @pytest.mark.asyncio
    async def test_flush_unknown_job(self, manager):
        summary = await manager.flush_job("no-job")
        assert summary.get("error") == "job_not_found"

    @pytest.mark.asyncio
    async def test_flush_handles_unload_errors(self, manager, mock_lifecycle):
        mock_lifecycle.unload_component = AsyncMock(side_effect=RuntimeError("boom"))
        await manager.create_job("j1", "cam-01")
        await manager.register_dynamic_component("j1", "comp-a")
        summary = await manager.flush_job("j1")
        assert "comp-a" in summary["unload_errors"]


# ── Status ────────────────────────────────────────────────────────


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_snapshot(self, manager):
        await manager.create_job("j1", "cam-01")
        await manager.register_dynamic_component("j1", "yolo")
        status = manager.status()
        assert "j1" in status
        assert status["j1"]["phase"] == "initializing"
        assert status["j1"]["dynamic_components"] == 1
        assert status["j1"]["flushed"] is False
