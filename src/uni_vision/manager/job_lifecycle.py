"""Job Lifecycle Manager — per-video-job state tracking & component flushing.

Tracks which CV components were dynamically pulled in for each video
processing job, monitors anomaly detection progress, and orchestrates
full cleanup (unregister + pip-uninstall) when the job concludes.

The Manager Agent (Gemma 4 E2B) acts as a *manager*, not a direct vision
processor.  When it dynamically discovers and installs packages from
the open internet for a specific video, those packages are scoped to
the job's lifecycle.  Once anomaly detection completes, this module
flushes every dynamically-provisioned component — leaving the system
clean for the next upload.
"""

from __future__ import annotations

import asyncio
import enum
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog

from uni_vision.components.base import ComponentState
from uni_vision.manager.component_registry import ComponentRegistry
from uni_vision.manager.lifecycle import LifecycleManager

log = structlog.get_logger(__name__)


# ── Job state machine ─────────────────────────────────────────────


class JobPhase(str, enum.Enum):
    """Pipeline job processing phases."""

    INITIALIZING = "initializing"
    DISCOVERING = "discovering"      # Manager Agent analysing scene
    PROVISIONING = "provisioning"    # Downloading / installing packages
    PROCESSING = "processing"        # CV pipeline running
    ANOMALY_DETECTED = "anomaly_detected"
    COMPLETING = "completing"        # Anomaly analysis done, wrapping up
    FLUSHING = "flushing"            # Unloading + uninstalling packages
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AnomalyState:
    """Tracks anomaly detection progress for a job."""

    total_frames_processed: int = 0
    anomaly_frames: int = 0
    consecutive_anomaly_frames: int = 0
    anomaly_first_detected_at: Optional[float] = None
    anomaly_fully_analysed: bool = False
    # Number of consecutive non-anomaly frames after detection
    # to confirm the anomaly series is complete
    post_anomaly_stable_frames: int = 0
    # Aggregated anomaly results
    anomaly_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class JobRecord:
    """Full state for a single video processing job."""

    job_id: str
    camera_id: str
    phase: JobPhase = JobPhase.INITIALIZING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Components dynamically provisioned for THIS job (component IDs)
    dynamic_components: Set[str] = field(default_factory=set)
    # Pip packages installed for THIS job (package names)
    dynamic_pip_packages: Set[str] = field(default_factory=set)

    # Anomaly tracking
    anomaly: AnomalyState = field(default_factory=AnomalyState)

    # Whether this job's resources have been flushed
    flushed: bool = False


# ── Configuration ─────────────────────────────────────────────────


@dataclass
class JobLifecycleConfig:
    """Tunables for anomaly-driven completion."""

    # After first anomaly: how many stable (non-anomaly) frames
    # confirm the anomaly series is fully analysed
    post_anomaly_stable_threshold: int = 10

    # Maximum frames to process before forcing completion
    # (safety net for long/infinite streams)
    max_frames_per_job: int = 5000

    # If True, pip-uninstall packages on flush; otherwise just unload
    uninstall_pip_packages: bool = True

    # Timeout for the entire flush operation (seconds)
    flush_timeout_s: float = 60.0


# ── Job Lifecycle Manager ─────────────────────────────────────────


class JobLifecycleManager:
    """Manages per-video-job component lifecycle and anomaly-driven completion.

    Responsibilities:
      1. Track which components are dynamically provisioned per job
      2. Monitor anomaly detection state per job
      3. Determine when a job's anomaly analysis is complete
      4. Flush (unload + uninstall) all dynamic packages on completion
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        lifecycle: LifecycleManager,
        *,
        config: Optional[JobLifecycleConfig] = None,
        broadcaster: Optional[Any] = None,
    ) -> None:
        self._registry = registry
        self._lifecycle = lifecycle
        self._config = config or JobLifecycleConfig()
        self._broadcaster = broadcaster
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    # ── Job creation / lookup ─────────────────────────────────────

    async def create_job(self, job_id: str, camera_id: str) -> JobRecord:
        """Register a new video processing job."""
        async with self._lock:
            job = JobRecord(job_id=job_id, camera_id=camera_id)
            self._jobs[job_id] = job
            log.info("job_created", job_id=job_id, camera_id=camera_id)
            if self._broadcaster is not None:
                try:
                    await self._broadcaster.emit_custom(
                        event_type="job_created",
                        data={"job_id": job_id, "camera_id": camera_id},
                    )
                except Exception:
                    pass
            return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def get_job_for_camera(self, camera_id: str) -> Optional[JobRecord]:
        """Find the active job for a camera (most recent non-completed)."""
        for job in reversed(list(self._jobs.values())):
            if job.camera_id == camera_id and job.phase not in (
                JobPhase.COMPLETED, JobPhase.ERROR,
            ):
                return job
        return None

    # ── Dynamic component tracking ────────────────────────────────

    async def register_dynamic_component(
        self,
        job_id: str,
        component_id: str,
        pip_package: Optional[str] = None,
    ) -> None:
        """Record that a component was dynamically provisioned for a job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                log.warning("register_component_no_job", job_id=job_id,
                            component_id=component_id)
                return
            job.dynamic_components.add(component_id)
            if pip_package:
                job.dynamic_pip_packages.add(pip_package)
            log.info(
                "dynamic_component_registered",
                job_id=job_id,
                component_id=component_id,
                pip_package=pip_package,
                total_dynamic=len(job.dynamic_components),
            )
            if self._broadcaster is not None:
                try:
                    await self._broadcaster.emit_custom(
                        event_type="component_provisioned",
                        data={
                            "job_id": job_id,
                            "component_id": component_id,
                            "pip_package": pip_package,
                            "total_dynamic": len(job.dynamic_components),
                        },
                    )
                except Exception:
                    pass

    async def update_phase(self, job_id: str, phase: JobPhase) -> None:
        """Transition a job to a new phase."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            old = job.phase
            job.phase = phase
            log.info("job_phase_changed", job_id=job_id,
                      old_phase=old, new_phase=phase)
            if self._broadcaster is not None:
                try:
                    await self._broadcaster.emit_custom(
                        event_type="job_phase_changed",
                        data={
                            "job_id": job_id,
                            "old_phase": old.value,
                            "new_phase": phase.value,
                        },
                    )
                except Exception:
                    pass

    # ── Anomaly tracking ──────────────────────────────────────────

    async def record_frame_result(
        self,
        job_id: str,
        *,
        anomaly_detected: bool,
        anomaly_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record a frame's analysis result.  Returns True if the job
        should be considered complete (anomaly series fully analysed
        OR max frames reached).
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False

            state = job.anomaly
            state.total_frames_processed += 1

            if anomaly_detected:
                state.anomaly_frames += 1
                state.consecutive_anomaly_frames += 1
                state.post_anomaly_stable_frames = 0

                if state.anomaly_first_detected_at is None:
                    state.anomaly_first_detected_at = time.time()
                    job.phase = JobPhase.ANOMALY_DETECTED
                    log.info("anomaly_first_detected", job_id=job_id,
                              frame=state.total_frames_processed)

                if anomaly_data:
                    state.anomaly_results.append(anomaly_data)
            else:
                state.consecutive_anomaly_frames = 0
                if state.anomaly_first_detected_at is not None:
                    state.post_anomaly_stable_frames += 1

            # Completion criteria:
            # 1. Anomaly detected + enough stable post-anomaly frames
            # 2. Max frames reached
            if (
                state.anomaly_first_detected_at is not None
                and state.post_anomaly_stable_frames
                    >= self._config.post_anomaly_stable_threshold
            ):
                state.anomaly_fully_analysed = True
                job.phase = JobPhase.COMPLETING
                log.info(
                    "anomaly_series_complete",
                    job_id=job_id,
                    anomaly_frames=state.anomaly_frames,
                    total_frames=state.total_frames_processed,
                    stable_after=state.post_anomaly_stable_frames,
                )
                return True

            if state.total_frames_processed >= self._config.max_frames_per_job:
                job.phase = JobPhase.COMPLETING
                log.warning(
                    "job_max_frames_reached",
                    job_id=job_id,
                    max_frames=self._config.max_frames_per_job,
                )
                return True

            return False

    # ── Flushing ──────────────────────────────────────────────────

    async def flush_job(self, job_id: str) -> Dict[str, Any]:
        """Unload and optionally uninstall all dynamic components for a job.

        Returns a summary dict of what was flushed.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return {"error": "job_not_found"}

        if job.flushed:
            return {"already_flushed": True}

        job.phase = JobPhase.FLUSHING
        log.info(
            "job_flush_start",
            job_id=job_id,
            dynamic_components=len(job.dynamic_components),
            pip_packages=len(job.dynamic_pip_packages),
        )

        unloaded: List[str] = []
        unload_errors: List[str] = []
        uninstalled: List[str] = []
        uninstall_errors: List[str] = []

        # 1. Unload all dynamic components from GPU/CPU
        for cid in list(job.dynamic_components):
            try:
                ok = await self._lifecycle.unload_component(cid)
                if ok:
                    # Also unregister so the registry is clean
                    self._registry.unregister(cid)
                    unloaded.append(cid)
                else:
                    unload_errors.append(cid)
            except Exception as exc:
                log.error("flush_unload_error", component_id=cid, error=str(exc))
                unload_errors.append(cid)

        # 2. Pip-uninstall packages that were dynamically installed
        if self._config.uninstall_pip_packages:
            for pkg in list(job.dynamic_pip_packages):
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda p=pkg: subprocess.run(
                            [sys.executable, "-m", "pip", "uninstall", "-y", p],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        ),
                    )
                    if result.returncode == 0:
                        uninstalled.append(pkg)
                        log.info("pip_uninstalled", package=pkg, job_id=job_id)
                    else:
                        uninstall_errors.append(pkg)
                        log.warning(
                            "pip_uninstall_failed",
                            package=pkg,
                            stderr=result.stderr[:200],
                        )
                except Exception as exc:
                    log.error("pip_uninstall_error", package=pkg, error=str(exc))
                    uninstall_errors.append(pkg)

        job.flushed = True
        job.phase = JobPhase.COMPLETED
        job.completed_at = time.time()

        summary = {
            "job_id": job_id,
            "unloaded": unloaded,
            "unload_errors": unload_errors,
            "uninstalled": uninstalled,
            "uninstall_errors": uninstall_errors,
            "total_frames": job.anomaly.total_frames_processed,
            "anomaly_frames": job.anomaly.anomaly_frames,
        }

        log.info("job_flush_complete", **summary)
        return summary

    # ── Status ────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of all tracked jobs."""
        return {
            jid: {
                "phase": j.phase.value,
                "camera_id": j.camera_id,
                "dynamic_components": len(j.dynamic_components),
                "dynamic_pip_packages": len(j.dynamic_pip_packages),
                "frames_processed": j.anomaly.total_frames_processed,
                "anomaly_frames": j.anomaly.anomaly_frames,
                "flushed": j.flushed,
            }
            for jid, j in self._jobs.items()
        }
