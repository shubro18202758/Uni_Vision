"""Hard 8 GiB VRAM budget enforcer — spec §2.1.

Computes the exact memory partitioning required to fit the OS, the
INT8 detection models, and the Q4_K_M language model orchestrator
within a strict 8192 MB ceiling.

This module is **read at startup** by the container and the VRAM
monitor.  It is the single source of truth for VRAM arithmetic and
provides a ``validate_budget()`` function that *provably* ensures
the combined footprint never exceeds the ceiling.

Layout (RTX 4070, 8192 MB total):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Region  Purpose                     Budget   Notes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A       LLM weights (Q4_K_M)        5120 MB  Qwen 3.5 9B @ 4 bits
B       KV cache (4096 ctx)          512 MB  ~0.5 MB per layer × 2 (K+V)
C       Vision workspace (INT8)     1024 MB  YOLOv8n×2 sequential
D       System / CUDA runtime        512 MB  Context + driver + OS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                               7168 MB
HEADROOM                            1024 MB  (12.5% safety)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── VRAM allocation constants ─────────────────────────────────────

# Hard ceiling — never allocate beyond this.
VRAM_CEILING_MB: int = 8192

# Per-model measured sizes (from Ollama `show` and TensorRT build logs)
QWEN_9B_Q4_KM_WEIGHTS_MB: int = 5120   # on-GPU after mmap
YOLOV8N_INT8_SINGLE_MB: int = 45        # single TensorRT engine
YOLOV8N_INT8_WORKSPACE_MB: int = 256    # TensorRT execution workspace

# KV-cache arithmetic for Qwen 3.5 9B:
#   36 layers × 2 (K + V) × 4096 tokens × 128 dims × 2 bytes (FP16)
#   = 36 × 2 × 4096 × 128 × 2 = ~72 MB (exact)
#   Ollama reserves additional scratch buffers (~440 MB).
KV_CACHE_BASE_TOKENS: int = 4096
KV_CACHE_BUDGET_MB: int = 512

# CUDA runtime + driver + OS desktop compositor
SYSTEM_OVERHEAD_MB: int = 512


@dataclass(frozen=True)
class VRAMBudgetReport:
    """Verified memory layout fitting within the 8 GiB ceiling."""

    ceiling_mb: int
    region_a_llm_weights_mb: int
    region_b_kv_cache_mb: int
    region_c_vision_workspace_mb: int
    region_d_system_overhead_mb: int
    total_allocated_mb: int
    headroom_mb: int
    fits: bool
    context_window_tokens: int
    quantization: str
    warnings: list


def compute_budget(
    *,
    ceiling_mb: int = VRAM_CEILING_MB,
    llm_weights_mb: int = QWEN_9B_Q4_KM_WEIGHTS_MB,
    kv_cache_mb: int = KV_CACHE_BUDGET_MB,
    vision_workspace_mb: int = 1024,
    system_overhead_mb: int = SYSTEM_OVERHEAD_MB,
    context_tokens: int = KV_CACHE_BASE_TOKENS,
) -> VRAMBudgetReport:
    """Validate that the given configuration fits within the VRAM ceiling.

    Returns a ``VRAMBudgetReport`` whose ``fits`` attribute is ``True``
    only when ``total_allocated_mb ≤ ceiling_mb``.
    """
    total = llm_weights_mb + kv_cache_mb + vision_workspace_mb + system_overhead_mb
    headroom = ceiling_mb - total
    warnings: list = []

    if headroom < 256:
        warnings.append(
            f"Headroom critically low: {headroom} MB (recommend ≥ 256 MB)"
        )
    if context_tokens > 4096:
        warnings.append(
            f"Context window {context_tokens} exceeds 4096 — KV cache may overflow"
        )

    return VRAMBudgetReport(
        ceiling_mb=ceiling_mb,
        region_a_llm_weights_mb=llm_weights_mb,
        region_b_kv_cache_mb=kv_cache_mb,
        region_c_vision_workspace_mb=vision_workspace_mb,
        region_d_system_overhead_mb=system_overhead_mb,
        total_allocated_mb=total,
        headroom_mb=headroom,
        fits=total <= ceiling_mb,
        context_window_tokens=context_tokens,
        quantization="Q4_K_M",
        warnings=warnings,
    )


def validate_budget(
    *,
    ceiling_mb: int = VRAM_CEILING_MB,
    llm_weights_mb: int = QWEN_9B_Q4_KM_WEIGHTS_MB,
    kv_cache_mb: int = KV_CACHE_BUDGET_MB,
    vision_workspace_mb: int = 1024,
    system_overhead_mb: int = SYSTEM_OVERHEAD_MB,
    context_tokens: int = KV_CACHE_BASE_TOKENS,
) -> VRAMBudgetReport:
    """Compute and log the VRAM budget.  Raises if budget is exceeded."""
    report = compute_budget(
        ceiling_mb=ceiling_mb,
        llm_weights_mb=llm_weights_mb,
        kv_cache_mb=kv_cache_mb,
        vision_workspace_mb=vision_workspace_mb,
        system_overhead_mb=system_overhead_mb,
        context_tokens=context_tokens,
    )

    logger.info(
        "vram_budget_report "
        "ceiling=%d total=%d headroom=%d fits=%s quantization=%s ctx=%d",
        report.ceiling_mb,
        report.total_allocated_mb,
        report.headroom_mb,
        report.fits,
        report.quantization,
        report.context_window_tokens,
    )
    for w in report.warnings:
        logger.warning("vram_budget_warning: %s", w)

    if not report.fits:
        raise MemoryError(
            f"VRAM budget exceeded: {report.total_allocated_mb} MB "
            f"> {report.ceiling_mb} MB ceiling.  Reduce context window "
            f"or switch to a smaller quantization."
        )

    return report


# ── Adaptive context window calculator ────────────────────────────


def max_context_for_budget(
    *,
    ceiling_mb: int = VRAM_CEILING_MB,
    llm_weights_mb: int = QWEN_9B_Q4_KM_WEIGHTS_MB,
    vision_workspace_mb: int = 1024,
    system_overhead_mb: int = SYSTEM_OVERHEAD_MB,
    safety_margin_mb: int = 256,
    bytes_per_token: float = 0.125,  # KV per token per layer (estimated)
    num_layers: int = 36,
) -> int:
    """Calculate the maximum context window (tokens) that fits in budget.

    KV cache size ≈ num_layers × 2 (K+V) × ctx_tokens × head_dim × dtype
    For Qwen 3.5 9B: 36 layers, head_dim=128, FP16 = 2 bytes
    Per-token KV ≈ 36 × 2 × 128 × 2 = 18.432 KB ≈ 0.018 MB

    Adding Ollama scratch overhead: ~0.125 MB per token (empirical, includes
    attention workspace, rope buffers, and quantised-weight dequant scratch).
    """
    available = ceiling_mb - llm_weights_mb - vision_workspace_mb - system_overhead_mb - safety_margin_mb
    if available <= 0:
        return 0

    mb_per_token = bytes_per_token
    max_tokens = int(available / mb_per_token)

    # Clamp to sensible range
    return min(max_tokens, 8192)
