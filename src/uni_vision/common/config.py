"""YAML configuration loader with environment variable resolution.

Loads config/default.yaml, merges with config/cameras.yaml and
config/models.yaml, then resolves UV_-prefixed environment overrides
via pydantic-settings.  Provides a single validated AppConfig instance
used throughout the application.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Locate config directory relative to project root ──────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/uni_vision/common → project root
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a single YAML file and return its content as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


# ── Pydantic sub-models ──────────────────────────────────────────


class HardwareConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UV_")

    device: str = "cuda"
    cuda_device_index: int = 0
    vram_ceiling_mb: int = 8192
    vram_safety_margin_mb: int = 256
    vram_poll_interval_ms: int = 500


class VRAMBudgets(BaseSettings):
    """Static VRAM region budgets — spec §2.1, revised §7.3.

    Region A: LLM weights (Gemma 4 E2B Q4_K_M GGUF, 7.2 GB disk).
    Region B: KV-cache (4096 tokens, Ollama-managed scratch).
    Region C: Vision workspace (2× YOLOv8n INT8 TensorRT engines).
    Region D: OS + driver + CUDA context overhead.

    Gemma 4 E2B (5.1B total / 2.3B effective MoE) fits entirely
    on an 8 GB GPU with ~2168 MB headroom — no CPU offload needed.
    """

    llm_weights_mb: int = 5000
    kv_cache_mb: int = 256
    vision_workspace_mb: int = 256
    system_overhead_mb: int = 512


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UV_")

    stream_queue_maxsize: int = 50
    inference_queue_maxsize: int = 10
    inference_queue_high_water: int = 8
    inference_queue_low_water: int = 3
    adaptive_throttle_factor: float = 0.5


class OllamaConfig(BaseSettings):
    """Ollama inference parameters — aligned with Modelfile defaults.

    KV-cache budget: 512 MB for 4096 tokens (Region B).
    Generation is hard-capped at ``num_predict`` tokens; stop sequences
    (``</result>``, ``</adjudication>``) terminate output earlier.
    """

    model_config = SettingsConfigDict(env_prefix="UV_OLLAMA_")

    base_url: str = "http://localhost:11434"
    model: str = "gemma4:e2b"
    timeout_s: int = 5
    num_ctx: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    repeat_penalty: float = 1.15
    repeat_last_n: int = 128
    num_predict: int = 256
    num_batch: int = 256
    num_gpu: int = -1  # -1 = offload all layers to GPU
    seed: int = 42
    stream: bool = False
    max_retries: int = 2


class ProfilingConfig(BaseSettings):
    """Runtime profiling toggle and history parameters — spec §12."""

    model_config = SettingsConfigDict(env_prefix="UV_PROFILING_")

    enabled: bool = True
    device_index: int = 0
    history_size: int = 512
    validate_vram_budget_on_start: bool = True


class QuantizationConfig(BaseSettings):
    """Model quantization descriptors — documentation only, not runtime.

    These mirror the Modelfile ``FROM`` tags and TensorRT engine formats
    so operators can verify the expected model configuration at a glance.
    """

    model_config = SettingsConfigDict(env_prefix="UV_QUANT_")

    llm_format: str = "Q4_K_M"  # GGUF mixed 4-bit K-quants
    llm_model_tag: str = "gemma4:e2b"
    vision_format: str = "INT8"  # TensorRT INT8 calibrated
    vision_engine_ext: str = ".engine"


class CircuitBreakerConfig(BaseSettings):
    failure_threshold: int = 3
    failure_window_s: int = 60
    recovery_timeout_s: int = 30


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UV_POSTGRES_")

    dsn: str = "postgresql://uni_vision:changeme@localhost:5432/uni_vision"
    pool_min: int = 2
    pool_max: int = 8


class StorageConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UV_S3_")

    endpoint: str = "http://localhost:9000"
    bucket: str = "uni-vision-images"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    region: str = "us-east-1"


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UV_REDIS_")

    url: str = "redis://localhost:6379/0"


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UV_API_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    api_keys: str = ""  # comma-separated API keys; empty = auth disabled
    rate_limit_rpm: int = 120  # requests per minute per IP; 0 = disabled
    cors_origins: str = ""  # comma-separated origins; empty = allow all (dev only)


class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UV_")

    log_level: str = "INFO"
    log_format: str = "json"


class CameraSourceConfig(BaseSettings):
    camera_id: str = ""
    source_url: str = ""
    location_tag: str = ""
    fps_target: int = 15
    enabled: bool = True


class FrameSamplingConfig(BaseSettings):
    """pHash-based temporal dedup parameters — spec §S1."""

    model_config = SettingsConfigDict(env_prefix="UV_FRAME_SAMPLING_")

    phash_size: int = 32
    hamming_distance_threshold: int = 5


class ModelEntry(BaseSettings):
    model_path: str = ""
    model_format: str = "onnx"
    input_size: List[int] = Field(default_factory=lambda: [640, 640])
    confidence_threshold: float = 0.60
    nms_iou_threshold: float = 0.45
    classes: Dict[int, str] = Field(default_factory=dict)
    multi_plate_policy: str = "highest_confidence"


class DeskewConfig(BaseSettings):
    """Geometric correction parameters — spec §4 S5."""

    enabled: bool = True
    max_skew_degrees: float = 30.0
    skip_threshold_degrees: float = 3.0
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 80
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10


class EnhanceConfig(BaseSettings):
    """Photometric enhancement parameters — spec §4 S6."""

    resize_enabled: bool = True
    resize_min_height: int = 200

    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: List[int] = Field(default_factory=lambda: [8, 8])

    gaussian_blur_enabled: bool = True
    gaussian_kernel_size: List[int] = Field(default_factory=lambda: [3, 3])

    bilateral_enabled: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0


class PreprocessingConfig(BaseSettings):
    """Detection preprocessing chain — spec §4 S4-S6."""

    padding_px: int = 5
    deskew: DeskewConfig = Field(default_factory=DeskewConfig)
    enhance: EnhanceConfig = Field(default_factory=EnhanceConfig)


class FallbackOCRConfig(BaseSettings):
    """Lightweight fallback OCR engine parameters — spec §9.1 Strategy."""

    model_config = SettingsConfigDict(env_prefix="UV_FALLBACK_OCR_")

    engine: str = "easyocr"  # easyocr | paddleocr
    languages: List[str] = Field(default_factory=lambda: ["en"])
    gpu: bool = False  # fallback runs on CPU to respect VRAM budget
    confidence_threshold: float = 0.4
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class ValidationConfig(BaseSettings):
    """Deterministic validation parameters — spec §4 S8.

    Character correction map: keys are OCR-confused characters,
    values are their correct counterparts at that position category
    (alpha or digit).  Applied positionally based on locale regex.
    """

    model_config = SettingsConfigDict(env_prefix="UV_VALIDATION_")

    # Locale-aware regex patterns (evaluated top-to-bottom, first match wins)
    locale_patterns: Dict[str, str] = Field(default_factory=lambda: {
        "IN": r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$",
        "GENERIC": r"^[A-Z0-9]{4,10}$",
    })
    default_locale: str = "IN"

    # Character confusion substitution pairs (bidirectional)
    char_corrections: Dict[str, str] = Field(default_factory=lambda: {
        "O": "0", "0": "O",
        "I": "1", "1": "I",
        "S": "5", "5": "S",
        "B": "8", "8": "B",
        "D": "0",
        "Z": "2", "2": "Z",
        "G": "6", "6": "G",
    })

    # Confidence below this triggers LLM adjudication even if regex passes
    adjudication_confidence_threshold: float = 0.75


class AdjudicationConfig(BaseSettings):
    """LLM adjudicator parameters for S8 agentic re-evaluation."""

    model_config = SettingsConfigDict(env_prefix="UV_ADJUDICATION_")

    enabled: bool = True
    timeout_s: float = 5.0
    max_retries: int = 1


class DeduplicationConfig(BaseSettings):
    """Sliding-window temporal deduplication — spec §4 S8.

    Suppresses redundant detections of the same object across
    consecutive frames within the configured time window.
    """

    model_config = SettingsConfigDict(env_prefix="UV_DEDUP_")

    window_seconds: float = 10.0
    purge_interval_seconds: float = 5.0


class DispatchConfig(BaseSettings):
    """Async dispatch parameters — spec §4 S8.

    Controls the decoupled persistence task that commits metadata to
    the database and archives images to object storage.
    """

    model_config = SettingsConfigDict(env_prefix="UV_DISPATCH_")

    queue_maxsize: int = 256
    db_write_timeout_s: float = 2.0
    image_upload_timeout_s: float = 3.0
    max_retries: int = 2
    retry_base_delay_s: float = 0.25
    image_format: str = "png"


class RetentionConfig(BaseSettings):
    """Automated data retention policy for production deployments.

    Controls periodic cleanup of old detection records and audit logs
    to keep the database size manageable.
    """

    model_config = SettingsConfigDict(env_prefix="UV_RETENTION_")

    enabled: bool = False
    max_age_days: int = 90
    audit_max_age_days: int = 180
    check_interval_hours: float = 24.0
    batch_size: int = 1000


class ManagerConfig(BaseSettings):
    """Manager Agent configuration — dynamic CV pipeline orchestrator.

    Controls the Gemma 4 E2B Manager Agent that discovers, downloads,
    loads, and composes specialised CV components into context-adaptive
    pipelines.  Gemma 4 acts as a meta-orchestrator, not a direct OCR or
    reasoning model — it decides *which* open-source models and
    libraries to load for each frame context.
    """

    model_config = SettingsConfigDict(env_prefix="UV_MANAGER_")

    enabled: bool = True
    hub_cache_dir: str = "~/.cache/uni_vision/hub"
    vram_total_mb: int = 8192
    vram_reserved_for_llm_mb: int = 5500
    max_search_results: int = 10
    http_timeout_s: float = 30.0
    default_scene: str = "unknown"
    prefer_trusted: bool = True
    auto_download: bool = True
    max_concurrent_downloads: int = 2
    component_warmup: bool = True
    trusted_sources: str = ""  # comma-separated; empty = built-in list only


class NavarasaConfig(BaseSettings):
    """Navarasa 2.0 7B conversational & interactive UI LLM configuration.

    Navarasa (Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0)
    is a Gemma 7B model fine-tuned on 15 Indian languages + English.
    It runs alongside Gemma 4 E2B via Ollama (sequential model swapping).

    Role:
      - Conversational and interactive generative LLM for the frontend.
        Converses naturally with users in any of the 15 supported
        Indian languages, translates between Indian languages and
        English for Gemma 4 processing, explains system status, and
        guides users through the interface.
      - All pipeline / CV intelligence (plate interpretation, OCR,
        state codes, detection enrichment, component management)
        remains with the Gemma 4 E2B Manager Agent.
    """

    model_config = SettingsConfigDict(env_prefix="UV_NAVARASA_")

    enabled: bool = True
    model: str = "uni-vision-navarasa"
    base_url: str = "http://localhost:11434"
    timeout_s: float = 15.0
    num_ctx: int = 4096
    temperature: float = 0.15
    top_p: float = 0.9
    top_k: int = 30
    num_predict: int = 512
    num_gpu: int = -1
    seed: int = 42
    max_retries: int = 2
    default_language: str = "hi"  # ISO 639-1 code for Hindi
    supported_languages: str = (
        "hi,te,ta,kn,ml,mr,bn,gu,pa,or,ur,as,kok,ne,sd,en"
    )
    translate_alerts: bool = True  # translate WebSocket alerts


class AgentConfig(BaseSettings):
    """Agentic sub-system parameters — Phase 21+.

    Controls the ReAct reasoning loop that lets Gemma 4 E2B
    autonomously manage pipelines and answer natural-language queries.
    """

    model_config = SettingsConfigDict(env_prefix="UV_AGENT_")

    enabled: bool = True
    max_iterations: int = 10
    llm_timeout_s: float = 30.0
    max_tokens: int = 1024
    temperature: float = 0.3
    memory_budget_tokens: int = 3072
    tool_timeout_s: float = 10.0


# ── Databricks add-on configuration ──────────────────────────────


class DeltaLakeConfig(BaseSettings):
    """Delta Lake detection sink — ACID versioned storage."""

    model_config = SettingsConfigDict(env_prefix="UV_DELTA_")

    table_path: str = "./data/delta/detections"
    audit_table_path: str = "./data/delta/audit_log"
    partition_columns: List[str] = Field(default_factory=lambda: ["camera_id"])
    checkpoint_interval: int = 50
    vacuum_retain_hours: int = 168  # 7 days
    enable_time_travel: bool = True
    max_versions: int = 500


class MLflowConfig(BaseSettings):
    """MLflow inference experiment tracking."""

    model_config = SettingsConfigDict(env_prefix="UV_MLFLOW_")

    tracking_uri: str = "./data/mlflow"
    experiment_name: str = "uni-vision-inference"
    log_every_n_frames: int = 50
    log_system_metrics: bool = True
    register_models: bool = True
    run_name_prefix: str = "uv-pipeline"


class SparkAnalyticsConfig(BaseSettings):
    """PySpark batch analytics engine."""

    model_config = SettingsConfigDict(env_prefix="UV_SPARK_")

    master: str = "local[*]"
    app_name: str = "UniVisionAnalytics"
    driver_memory: str = "2g"
    executor_memory: str = "1g"
    shuffle_partitions: int = 4
    delta_table_path: str = "./data/delta/detections"


class FAISSConfig(BaseSettings):
    """FAISS vector search for plate similarity and agent RAG."""

    model_config = SettingsConfigDict(env_prefix="UV_FAISS_")

    index_path: str = "./data/faiss/detection_index.bin"
    metadata_path: str = "./data/faiss/detection_metadata.json"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    nprobe: int = 10
    top_k: int = 20
    similarity_threshold: float = 0.65
    rebuild_interval_s: float = 3600.0


class DatabricksConfig(BaseSettings):
    """Top-level Databricks integration toggle and sub-configs.

    When ``enabled`` is *False*, all Databricks add-ons are skipped
    and the system behaves identically to the base architecture.
    """

    model_config = SettingsConfigDict(env_prefix="UV_DATABRICKS_")

    enabled: bool = False
    delta: DeltaLakeConfig = Field(default_factory=DeltaLakeConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    spark: SparkAnalyticsConfig = Field(default_factory=SparkAnalyticsConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)


# ── Top-level application config ─────────────────────────────────


class AppConfig(BaseSettings):
    """Aggregated, validated configuration for the entire pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="UV_",
        env_nested_delimiter="__",
    )

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    vram_budgets: VRAMBudgets = Field(default_factory=VRAMBudgets)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cameras: List[CameraSourceConfig] = Field(default_factory=list)
    models: Dict[str, ModelEntry] = Field(default_factory=dict)
    frame_sampling: FrameSamplingConfig = Field(default_factory=FrameSamplingConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    fallback_ocr: FallbackOCRConfig = Field(default_factory=FallbackOCRConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    adjudication: AdjudicationConfig = Field(default_factory=AdjudicationConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    dispatch: DispatchConfig = Field(default_factory=DispatchConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    manager: ManagerConfig = Field(default_factory=ManagerConfig)
    navarasa: NavarasaConfig = Field(default_factory=NavarasaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    databricks: DatabricksConfig = Field(default_factory=DatabricksConfig)
    visualizer_enabled: bool = False


def load_config(
    config_dir: Optional[Path] = None,
) -> AppConfig:
    """Build the validated ``AppConfig`` from YAML files + env vars.

    Resolution order (later overrides earlier):
      1. config/default.yaml
      2. config/cameras.yaml  → ``cameras`` list
      3. config/models.yaml   → ``models`` dict
      4. UV_-prefixed environment variables
    """
    config_dir = config_dir or _CONFIG_DIR

    # 1. Base configuration from default.yaml
    base: Dict[str, Any] = _load_yaml(config_dir / "default.yaml")

    # 2. Merge camera sources
    cameras_raw = _load_yaml(config_dir / "cameras.yaml")
    if "cameras" in cameras_raw:
        base["cameras"] = cameras_raw["cameras"]

    # 3. Merge model definitions + frame_sampling + preprocessing config
    models_raw = _load_yaml(config_dir / "models.yaml")
    if "frame_sampling" in models_raw:
        base["frame_sampling"] = models_raw["frame_sampling"]
    if "preprocessing" in models_raw:
        base["preprocessing"] = models_raw["preprocessing"]

    # Extract detector entries — top-level keys that are not frame_sampling/preprocessing
    detector_entries: Dict[str, Any] = {}
    for key, val in models_raw.items():
        if key in ("frame_sampling", "preprocessing"):
            continue
        if isinstance(val, dict):
            detector_entries[key] = val
    if detector_entries:
        base["models"] = detector_entries

    # 4. Construct config — pydantic-settings auto-reads env vars
    return AppConfig(**base)
