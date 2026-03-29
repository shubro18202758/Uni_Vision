"""Tests for the VRAM budget enforcer."""

from __future__ import annotations

import pytest


class TestComputeBudget:
    """Test the budget computation logic."""

    def test_default_budget_fits(self):
        from uni_vision.monitoring.vram_budget import compute_budget

        report = compute_budget()
        assert report.fits is True
        assert report.ceiling_mb == 8192
        assert report.total_allocated_mb == 7168
        assert report.headroom_mb == 1024
        assert report.quantization == "Q4_K_M"
        assert report.context_window_tokens == 4096

    def test_budget_regions_add_up(self):
        from uni_vision.monitoring.vram_budget import compute_budget

        report = compute_budget()
        total = (
            report.region_a_llm_weights_mb
            + report.region_b_kv_cache_mb
            + report.region_c_vision_workspace_mb
            + report.region_d_system_overhead_mb
        )
        assert total == report.total_allocated_mb

    def test_budget_exceeds_ceiling(self):
        from uni_vision.monitoring.vram_budget import compute_budget

        report = compute_budget(ceiling_mb=4096)
        assert report.fits is False
        assert report.headroom_mb < 0

    def test_low_headroom_warning(self):
        from uni_vision.monitoring.vram_budget import compute_budget

        # Set ceiling just barely above total to trigger warning
        report = compute_budget(ceiling_mb=7300)
        assert report.fits is True
        assert any("Headroom critically low" in w for w in report.warnings)

    def test_high_context_warning(self):
        from uni_vision.monitoring.vram_budget import compute_budget

        report = compute_budget(context_tokens=8192)
        assert any("Context window" in w for w in report.warnings)

    def test_custom_regions(self):
        from uni_vision.monitoring.vram_budget import compute_budget

        report = compute_budget(
            ceiling_mb=16384,
            llm_weights_mb=8000,
            kv_cache_mb=2000,
            vision_workspace_mb=2000,
            system_overhead_mb=1000,
        )
        assert report.total_allocated_mb == 13000
        assert report.headroom_mb == 3384
        assert report.fits is True


class TestValidateBudget:
    """Test validate_budget which raises on overflow."""

    def test_validate_passes(self):
        from uni_vision.monitoring.vram_budget import validate_budget

        report = validate_budget()
        assert report.fits is True

    def test_validate_raises_on_overflow(self):
        from uni_vision.monitoring.vram_budget import validate_budget

        with pytest.raises(MemoryError, match="VRAM budget exceeded"):
            validate_budget(ceiling_mb=2048)


class TestMaxContextForBudget:
    """Test the adaptive context window calculator."""

    def test_default_returns_positive(self):
        from uni_vision.monitoring.vram_budget import max_context_for_budget

        tokens = max_context_for_budget()
        assert tokens > 0
        assert tokens <= 8192  # clamped

    def test_no_room_returns_zero(self):
        from uni_vision.monitoring.vram_budget import max_context_for_budget

        tokens = max_context_for_budget(ceiling_mb=5120)
        # With 5120 LLM + 1024 vision + 512 system + 256 safety = 6912 > 5120
        assert tokens == 0

    def test_larger_ceiling_more_tokens(self):
        from uni_vision.monitoring.vram_budget import max_context_for_budget

        tokens_8g = max_context_for_budget(ceiling_mb=8192)
        tokens_16g = max_context_for_budget(ceiling_mb=16384)
        assert tokens_16g >= tokens_8g


class TestConstants:
    """Verify VRAM constants match spec §2.1."""

    def test_ceiling(self):
        from uni_vision.monitoring.vram_budget import VRAM_CEILING_MB

        assert VRAM_CEILING_MB == 8192

    def test_qwen_weight_size(self):
        from uni_vision.monitoring.vram_budget import QWEN_9B_Q4_KM_WEIGHTS_MB

        assert QWEN_9B_Q4_KM_WEIGHTS_MB == 5120

    def test_yolov8n_engine_size(self):
        from uni_vision.monitoring.vram_budget import YOLOV8N_INT8_SINGLE_MB

        assert YOLOV8N_INT8_SINGLE_MB == 45

    def test_kv_base_tokens(self):
        from uni_vision.monitoring.vram_budget import KV_CACHE_BASE_TOKENS

        assert KV_CACHE_BASE_TOKENS == 4096
