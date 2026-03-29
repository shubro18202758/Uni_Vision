"""MLflow experiment tracking for pipeline inference stages.

Logs per-stage metrics (latency, confidence, VRAM usage) to an
MLflow experiment so that model performance can be tracked, compared,
and analysed over time.  Supports:

  * **Run-level metrics** — latency_ms, ocr_confidence, vram_delta_mb
    for each pipeline stage (S2 vehicle-detect, S3 plate-detect,
    S7 OCR, S8 validation).
  * **Model registration** — registers the active model configuration
    (engine tag, quantisation format) as an MLflow model artefact.
  * **System metrics** — GPU utilisation, total VRAM, throughput.
  * **Batch logging** — accumulates N frames before flushing to avoid
    per-frame tracking overhead on the hot inference path.

All tracking is local-first (file-backed) but compatible with
Databricks-hosted MLflow Tracking Server.

Requires: ``pip install 'uni-vision[databricks]'``
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports
_mlflow = None


def _ensure_imports() -> None:
    global _mlflow
    if _mlflow is not None:
        return
    try:
        import mlflow  # type: ignore[import-untyped]
        _mlflow = mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow not installed. Run: pip install 'uni-vision[databricks]'"
        ) from exc


class InferenceTracker:
    """MLflow-backed experiment tracker for Uni_Vision inference.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI. Local path or remote server URL.
    experiment_name : str
        MLflow experiment name.
    log_every_n_frames : int
        Flush accumulated metrics every N frames.
    log_system_metrics : bool
        Log GPU / system metrics alongside inference metrics.
    run_name_prefix : str
        Prefix for auto-generated run names.
    """

    def __init__(
        self,
        tracking_uri: str = "./data/mlflow",
        experiment_name: str = "uni-vision-inference",
        log_every_n_frames: int = 50,
        log_system_metrics: bool = True,
        run_name_prefix: str = "uv-pipeline",
    ) -> None:
        _ensure_imports()
        self._tracking_uri = str(Path(tracking_uri).resolve())
        self._experiment_name = experiment_name
        self._log_every_n = log_every_n_frames
        self._log_system = log_system_metrics
        self._run_name_prefix = run_name_prefix

        # Metric accumulation buffer
        self._buffer: List[Dict[str, Any]] = []
        self._frame_count = 0
        self._run_id: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._start_time: Optional[float] = None

    # ── Lifecycle ─────────────────────────────────────────────────

    def initialise(self) -> None:
        """Set up MLflow tracking and start the first run."""
        os.makedirs(self._tracking_uri, exist_ok=True)
        _mlflow.set_tracking_uri(self._tracking_uri)

        # Create or get experiment
        experiment = _mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            self._experiment_id = _mlflow.create_experiment(
                self._experiment_name,
                artifact_location=os.path.join(self._tracking_uri, "artifacts"),
            )
        else:
            self._experiment_id = experiment.experiment_id

        self._start_run()
        logger.info(
            "mlflow_tracker_initialised experiment=%s run=%s",
            self._experiment_name,
            self._run_id,
        )

    def _start_run(self) -> None:
        """Begin a new MLflow run."""
        run_name = f"{self._run_name_prefix}-{int(time.time())}"
        run = _mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
        )
        self._run_id = run.info.run_id
        self._start_time = time.time()

    def shutdown(self) -> None:
        """Flush remaining metrics and close the active run."""
        self._flush_buffer()
        if self._run_id is not None:
            try:
                _mlflow.end_run()
            except Exception:
                pass
            self._run_id = None
        logger.info("mlflow_tracker_shutdown frames_logged=%d", self._frame_count)

    # ── Stage metric recording ────────────────────────────────────

    def log_stage_metrics(
        self,
        stage: str,
        latency_ms: float,
        confidence: Optional[float] = None,
        vram_before_mb: Optional[float] = None,
        vram_after_mb: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record metrics for a single pipeline stage execution.

        Parameters
        ----------
        stage : str
            Stage identifier (e.g. "S2_vehicle_detect", "S7_ocr").
        latency_ms : float
            Wall-clock time for the stage in milliseconds.
        confidence : float, optional
            Detection / OCR confidence score.
        vram_before_mb, vram_after_mb : float, optional
            VRAM usage before and after the stage.
        extra : dict, optional
            Additional key-value metrics to log.
        """
        metrics: Dict[str, float] = {
            f"{stage}_latency_ms": latency_ms,
        }
        if confidence is not None:
            metrics[f"{stage}_confidence"] = confidence
        if vram_before_mb is not None and vram_after_mb is not None:
            metrics[f"{stage}_vram_delta_mb"] = vram_after_mb - vram_before_mb
            metrics[f"{stage}_vram_after_mb"] = vram_after_mb

        if extra:
            for k, v in extra.items():
                if isinstance(v, (int, float)):
                    metrics[f"{stage}_{k}"] = float(v)

        self._buffer.append(metrics)
        self._frame_count += 1

        if len(self._buffer) >= self._log_every_n:
            self._flush_buffer()

    def log_detection_result(
        self,
        plate_number: str,
        camera_id: str,
        ocr_engine: str,
        ocr_confidence: float,
        validation_status: str,
        total_pipeline_ms: float,
    ) -> None:
        """Log a complete detection result as an MLflow metric set."""
        self._buffer.append({
            "detection_confidence": ocr_confidence,
            "detection_pipeline_ms": total_pipeline_ms,
            "detection_valid": 1.0 if validation_status == "valid" else 0.0,
        })
        self._frame_count += 1

        if len(self._buffer) >= self._log_every_n:
            self._flush_buffer()

    # ── Model parameter logging ───────────────────────────────────

    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log model configuration as MLflow parameters.

        Call once at pipeline startup to record the active model
        configuration (engine tag, quantisation, input size, etc.).
        """
        if self._run_id is None:
            return
        safe_params = {k: str(v) for k, v in params.items()}
        _mlflow.log_params(safe_params)
        logger.debug("mlflow_params_logged keys=%s", list(safe_params.keys()))

    def log_model_artifact(
        self,
        model_name: str,
        model_info: Dict[str, Any],
    ) -> None:
        """Register model metadata as an MLflow artifact."""
        if self._run_id is None:
            return
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=f"model_{model_name}_"
        ) as f:
            json.dump(model_info, f, indent=2, default=str)
            tmp_path = f.name

        _mlflow.log_artifact(tmp_path, artifact_path="models")
        os.unlink(tmp_path)
        logger.debug("mlflow_model_artifact_logged name=%s", model_name)

    def tag_run(self, tags: Dict[str, str]) -> None:
        """Apply tags to the active run for categorisation and filtering."""
        if self._run_id is None:
            return
        try:
            _mlflow.set_tags(tags)
        except Exception as exc:
            logger.warning("mlflow_tag_failed err=%s", exc)

    def get_health(self) -> Dict[str, Any]:
        """Return a health summary for monitoring dashboards."""
        return {
            "active": self._run_id is not None,
            "experiment": self._experiment_name,
            "run_id": self._run_id,
            "frames_logged": self._frame_count,
            "buffer_pending": len(self._buffer),
            "uptime_s": round(time.time() - (self._start_time or time.time()), 1),
        }

    # ── Buffer management ─────────────────────────────────────────

    def _flush_buffer(self) -> None:
        """Aggregate buffered metrics and log to MLflow in a single batch.

        Computes mean, min, max, and p95 for each metric key across
        the accumulated buffer, giving richer insight than averages alone.
        """
        if not self._buffer or self._run_id is None:
            return

        # Collect all values per metric key
        collected: Dict[str, List[float]] = {}
        for entry in self._buffer:
            for k, v in entry.items():
                collected.setdefault(k, []).append(v)

        metrics_out: Dict[str, float] = {}
        for k, values in collected.items():
            n = len(values)
            avg = sum(values) / n
            metrics_out[k] = avg
            # Only emit extended stats for keys with enough data points
            if n >= 3:
                sorted_vals = sorted(values)
                metrics_out[f"{k}_min"] = sorted_vals[0]
                metrics_out[f"{k}_max"] = sorted_vals[-1]
                p95_idx = int(n * 0.95)
                metrics_out[f"{k}_p95"] = sorted_vals[min(p95_idx, n - 1)]

        # Add throughput metric (frames per second since last flush)
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                metrics_out["throughput_fps"] = round(self._frame_count / elapsed, 2)

        try:
            _mlflow.log_metrics(metrics_out, step=self._frame_count)
        except Exception as exc:
            logger.warning("mlflow_flush_failed err=%s", exc)

        self._buffer.clear()

    # ── Query interface ───────────────────────────────────────────

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Return a summary of the current experiment."""
        if self._experiment_id is None:
            return {"active": False}

        experiment = _mlflow.get_experiment(self._experiment_id)
        runs = _mlflow.search_runs(
            experiment_ids=[self._experiment_id],
            max_results=100,
        )

        return {
            "active": True,
            "experiment_name": self._experiment_name,
            "experiment_id": self._experiment_id,
            "current_run_id": self._run_id,
            "total_runs": len(runs) if not runs.empty else 0,
            "total_frames_logged": self._frame_count,
            "buffer_size": len(self._buffer),
            "tracking_uri": self._tracking_uri,
            "uptime_s": round(time.time() - (self._start_time or time.time()), 1),
        }

    def get_run_metrics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve metrics for a specific run (defaults to active run)."""
        target_run = run_id or self._run_id
        if target_run is None:
            return {"error": "No active run"}

        run = _mlflow.get_run(target_run)
        return {
            "run_id": target_run,
            "status": run.info.status,
            "start_time": str(run.info.start_time),
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        }

    def get_metric_history(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve the logged history for a specific metric."""
        if self._run_id is None:
            return []

        client = _mlflow.tracking.MlflowClient(self._tracking_uri)
        history = client.get_metric_history(self._run_id, metric_name)

        return [
            {
                "step": h.step,
                "value": h.value,
                "timestamp": h.timestamp,
            }
            for h in history[-limit:]
        ]
