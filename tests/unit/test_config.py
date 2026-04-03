"""Tests for config loading, defaults, and sub-model construction."""

from __future__ import annotations


class TestConfigDefaults:
    """Verify that AppConfig and sub-models have correct defaults."""

    def test_hardware_defaults(self):
        from uni_vision.common.config import HardwareConfig

        hw = HardwareConfig()
        assert hw.device == "cuda"
        assert hw.cuda_device_index == 0
        assert hw.vram_ceiling_mb == 8192
        assert hw.vram_poll_interval_ms == 500

    def test_vram_budgets_defaults(self):
        from uni_vision.common.config import VRAMBudgets

        vb = VRAMBudgets()
        assert vb.llm_weights_mb == 5000
        assert vb.kv_cache_mb == 256
        assert vb.vision_workspace_mb == 256
        assert vb.system_overhead_mb == 512
        # Total should fit within ~8 GB (Ollama auto-offloads overflow to CPU)
        total = vb.llm_weights_mb + vb.kv_cache_mb + vb.vision_workspace_mb + vb.system_overhead_mb
        assert total == 6024
        assert total <= 10240  # generous ceiling; fits entirely on 8 GB GPU

    def test_ollama_defaults(self):
        from uni_vision.common.config import OllamaConfig

        oc = OllamaConfig()
        assert oc.base_url == "http://localhost:11434"
        assert oc.model == "gemma4:e2b"
        assert oc.timeout_s == 5
        assert oc.num_ctx == 4096
        assert oc.num_predict == 256
        assert oc.seed == 42

    def test_pipeline_defaults(self):
        from uni_vision.common.config import PipelineConfig

        pc = PipelineConfig()
        assert pc.stream_queue_maxsize == 50
        assert pc.inference_queue_maxsize == 10

    def test_database_defaults(self):
        from uni_vision.common.config import DatabaseConfig

        db = DatabaseConfig()
        assert db.pool_min == 2
        assert db.pool_max == 8
        assert "postgresql://" in db.dsn

    def test_frame_sampling_defaults(self):
        from uni_vision.common.config import FrameSamplingConfig

        fs = FrameSamplingConfig()
        assert fs.phash_size == 32
        assert fs.hamming_distance_threshold == 5

    def test_validation_defaults(self):
        from uni_vision.common.config import ValidationConfig

        vc = ValidationConfig()
        assert "IN" in vc.locale_patterns
        assert "GENERIC" in vc.locale_patterns
        assert vc.adjudication_confidence_threshold == 0.75

    def test_dedup_defaults(self):
        from uni_vision.common.config import DeduplicationConfig

        dc = DeduplicationConfig()
        assert dc.window_seconds == 10.0
        assert dc.purge_interval_seconds == 5.0

    def test_dispatch_defaults(self):
        from uni_vision.common.config import DispatchConfig

        dc = DispatchConfig()
        assert dc.queue_maxsize == 256
        assert dc.db_write_timeout_s == 2.0
        assert dc.max_retries == 2

    def test_circuit_breaker_defaults(self):
        from uni_vision.common.config import CircuitBreakerConfig

        cb = CircuitBreakerConfig()
        assert cb.failure_threshold == 3
        assert cb.failure_window_s == 60
        assert cb.recovery_timeout_s == 30

    def test_profiling_defaults(self):
        from uni_vision.common.config import ProfilingConfig

        pc = ProfilingConfig()
        assert pc.enabled is True
        assert pc.history_size == 512

    def test_quantization_defaults(self):
        from uni_vision.common.config import QuantizationConfig

        qc = QuantizationConfig()
        assert qc.llm_format == "Q4_K_M"
        assert qc.vision_format == "INT8"

    def test_deskew_defaults(self):
        from uni_vision.common.config import DeskewConfig

        dc = DeskewConfig()
        assert dc.enabled is True
        assert dc.max_skew_degrees == 30.0

    def test_enhance_defaults(self):
        from uni_vision.common.config import EnhanceConfig

        ec = EnhanceConfig()
        assert ec.clahe_enabled is True
        assert ec.clahe_clip_limit == 2.0
        assert ec.bilateral_d == 9

    def test_fallback_ocr_defaults(self):
        from uni_vision.common.config import FallbackOCRConfig

        fc = FallbackOCRConfig()
        assert fc.engine == "easyocr"
        assert fc.gpu is False  # CPU-only to respect VRAM budget
        assert fc.allowlist == "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class TestAppConfig:
    """Test the top-level AppConfig aggregation."""

    def test_app_config_construction(self):
        from uni_vision.common.config import AppConfig

        cfg = AppConfig()
        # Verify nested sub-models are present
        assert cfg.hardware.device == "cuda"
        assert cfg.ollama.model == "gemma4:e2b"
        assert cfg.validation.adjudication_confidence_threshold == 0.75
        assert cfg.deduplication.window_seconds == 10.0
        assert cfg.cameras == []
        assert cfg.models == {}
        assert cfg.visualizer_enabled is False
