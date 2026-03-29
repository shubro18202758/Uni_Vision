"""Integration tests: Config loading & overlay logic.

Verifies AppConfig construction, YAML merging, env var overrides,
and sub-config wiring without external dependencies.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from uni_vision.common.config import (
    APIConfig,
    AppConfig,
    DatabaseConfig,
    DispatchConfig,
    HardwareConfig,
    PipelineConfig,
    RedisConfig,
    StorageConfig,
    load_config,
)


# ── Tests ─────────────────────────────────────────────────────────


class TestAppConfigDefaults:
    """Verify default config builds without any YAML or env vars."""

    def test_default_construction(self):
        config = AppConfig()
        assert config.hardware is not None
        assert config.pipeline is not None
        assert config.database is not None
        assert config.api is not None

    def test_pipeline_defaults(self):
        config = AppConfig()
        assert config.pipeline.inference_queue_maxsize == 10
        assert config.pipeline.inference_queue_high_water == 8
        assert config.pipeline.inference_queue_low_water == 3

    def test_hardware_defaults(self):
        config = AppConfig()
        assert config.hardware.cuda_device_index == 0
        assert config.hardware.vram_ceiling_mb == 8192

    def test_database_defaults(self):
        config = AppConfig()
        assert "postgresql" in config.database.dsn
        assert config.database.pool_min == 2
        assert config.database.pool_max == 8

    def test_api_defaults(self):
        config = AppConfig()
        assert config.api.port == 8000
        assert config.api.rate_limit_rpm == 120

    def test_dispatch_defaults(self):
        config = AppConfig()
        assert config.dispatch.queue_maxsize == 256
        assert config.dispatch.max_retries == 2


class TestSubConfigNesting:
    """Verify that nested sub-configs wire correctly from dict input."""

    def test_override_pipeline_via_dict(self):
        config = AppConfig(
            pipeline=PipelineConfig(inference_queue_maxsize=20)
        )
        assert config.pipeline.inference_queue_maxsize == 20
        # Other defaults untouched
        assert config.pipeline.inference_queue_high_water == 8

    def test_override_api_via_dict(self):
        config = AppConfig(
            api=APIConfig(port=9000, rate_limit_rpm=60)
        )
        assert config.api.port == 9000
        assert config.api.rate_limit_rpm == 60

    def test_override_database(self):
        config = AppConfig(
            database=DatabaseConfig(pool_max=16)
        )
        assert config.database.pool_max == 16
        assert config.database.pool_min == 2

    def test_cameras_list(self):
        """Cameras list defaults to empty."""
        config = AppConfig()
        assert config.cameras == []

    def test_models_dict(self):
        """Models dict defaults to empty."""
        config = AppConfig()
        assert config.models == {}


class TestEnvVarOverrides:
    """Verify environment variable overrides for critical settings."""

    def test_api_port_override(self, monkeypatch):
        monkeypatch.setenv("UV_API_PORT", "9999")
        config = APIConfig()
        assert config.port == 9999

    def test_database_pool_override(self, monkeypatch):
        monkeypatch.setenv("UV_POSTGRES_POOL_MAX", "32")
        config = DatabaseConfig()
        assert config.pool_max == 32

    def test_pipeline_queue_size_override(self, monkeypatch):
        monkeypatch.setenv("UV_INFERENCE_QUEUE_MAXSIZE", "50")
        config = PipelineConfig()
        assert config.inference_queue_maxsize == 50


class TestLoadConfigFromYAML:
    """Verify that load_config reads YAML files properly."""

    def test_load_from_temp_dir(self, tmp_path):
        """Create minimal YAML files and verify they are loaded."""
        # default.yaml
        (tmp_path / "default.yaml").write_text(
            "hardware:\n  vram_ceiling_mb: 4096\n"
            "api:\n  port: 7777\n"
        )
        # cameras.yaml
        (tmp_path / "cameras.yaml").write_text(
            "cameras:\n"
            "  - camera_id: cam-test\n"
            "    source_url: rtsp://test\n"
            "    location_tag: lab\n"
            "    fps_target: 15\n"
            "    enabled: true\n"
        )
        # models.yaml (empty is fine)
        (tmp_path / "models.yaml").write_text("{}\n")

        config = load_config(config_dir=tmp_path)
        assert config.hardware.vram_ceiling_mb == 4096
        assert config.api.port == 7777
        assert len(config.cameras) == 1
        assert config.cameras[0].camera_id == "cam-test"

    def test_load_config_missing_cameras_yaml(self, tmp_path):
        """load_config should handle missing cameras.yaml gracefully."""
        (tmp_path / "default.yaml").write_text("{}\n")
        (tmp_path / "cameras.yaml").write_text("{}\n")
        (tmp_path / "models.yaml").write_text("{}\n")

        config = load_config(config_dir=tmp_path)
        assert config.cameras == []


class TestConfigConsistency:
    """Cross-field and cross-config coherence checks."""

    def test_high_water_less_than_max(self):
        config = AppConfig()
        assert config.pipeline.inference_queue_high_water < config.pipeline.inference_queue_maxsize

    def test_low_water_less_than_high(self):
        config = AppConfig()
        assert config.pipeline.inference_queue_low_water < config.pipeline.inference_queue_high_water

    def test_pool_min_lt_pool_max(self):
        config = AppConfig()
        assert config.database.pool_min <= config.database.pool_max

    def test_dispatch_queue_positive(self):
        config = AppConfig()
        assert config.dispatch.queue_maxsize > 0

    def test_visualizer_off_by_default(self):
        config = AppConfig()
        assert config.visualizer_enabled is False
