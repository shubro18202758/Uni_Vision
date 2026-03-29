"""Monitoring sub-package — profiling, VRAM telemetry, and budget enforcement.

Public API::

    from uni_vision.monitoring import (
        # Profiler
        profile_stage,
        set_profiling_enabled,
        vram_sampler,
        PipelineTelemetryHook,
        get_profile_history,
        clear_profile_history,
        # VRAM budget
        compute_budget,
        validate_budget,
        max_context_for_budget,
        # Live VRAM monitor
        VRAMMonitor,
    )
"""

from uni_vision.monitoring.profiler import (
    EventTelemetry,
    PipelineTelemetryHook,
    StageProfile,
    VRAMSnapshot,
    clear_profile_history,
    get_profile_history,
    profile_stage,
    set_profiling_enabled,
    vram_sampler,
)
from uni_vision.monitoring.vram_budget import (
    VRAMBudgetReport,
    compute_budget,
    max_context_for_budget,
    validate_budget,
)
from uni_vision.monitoring.vram_monitor import VRAMMonitor

__all__ = [
    # profiler
    "profile_stage",
    "set_profiling_enabled",
    "vram_sampler",
    "PipelineTelemetryHook",
    "EventTelemetry",
    "StageProfile",
    "VRAMSnapshot",
    "get_profile_history",
    "clear_profile_history",
    # budget
    "compute_budget",
    "validate_budget",
    "max_context_for_budget",
    "VRAMBudgetReport",
    # live monitor
    "VRAMMonitor",
]
