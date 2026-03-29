"""Knowledge-base tools for the agentic subsystem.

Provides tools that let the agent query and interact with the
KnowledgeBase — plate patterns, error profiles, feedback, and
anomaly detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from uni_vision.agent.tools import tool


def _get_kb(context: Any) -> Any:
    """Extract the KnowledgeBase from the execution context."""
    kb = getattr(context, "knowledge_base", None)
    if kb is None:
        raise RuntimeError("KnowledgeBase not available in execution context")
    return kb


def _get_pg_client(context: Any) -> Any:
    return getattr(context, "pg_client", None)


# ── Tools ─────────────────────────────────────────────────────────


@tool(
    name="get_knowledge_stats",
    description=(
        "Get summary statistics of the knowledge base: total observations, "
        "feedback entries, unique plates, and cameras profiled."
    ),
)
async def get_knowledge_stats(context: Any = None) -> Dict[str, Any]:
    kb = _get_kb(context)
    return kb.get_stats()


@tool(
    name="get_frequent_plates",
    description=(
        "Get the most frequently detected plates across all cameras. "
        "Useful for identifying permanent/regular vehicles."
    ),
    param_descriptions={
        "top_n": "Number of top plates to return (default: 20)",
    },
)
async def get_frequent_plates(
    top_n: int = 20, context: Any = None
) -> Dict[str, Any]:
    kb = _get_kb(context)
    plates = kb.get_plate_frequency(top_n=top_n)
    return {
        "top_plates": [{"plate": p, "count": c} for p, c in plates],
        "total_unique": len(kb._plate_frequency),
    }


@tool(
    name="get_camera_error_profile",
    description=(
        "Get the OCR error profile for a specific camera, including "
        "error rate, common character confusions, and average confidence. "
        "Use to diagnose persistent OCR issues for a camera."
    ),
    param_descriptions={
        "camera_id": "Camera identifier to profile",
    },
)
async def get_camera_error_profile(
    camera_id: str, context: Any = None
) -> Dict[str, Any]:
    kb = _get_kb(context)
    profile = kb.get_camera_profile(camera_id)
    if profile is None:
        return {"error": f"No profile for camera '{camera_id}'", "camera_id": camera_id}

    return {
        "camera_id": profile.camera_id,
        "total_detections": profile.total_detections,
        "error_count": profile.error_count,
        "error_rate": round(profile.error_rate, 4),
        "avg_confidence": round(profile.avg_confidence, 4),
        "top_confusions": [
            {"pattern": k, "count": v} for k, v in profile.top_confusions(5)
        ],
        "last_updated": profile.last_updated,
    }


@tool(
    name="get_all_camera_profiles",
    description=(
        "Get error profiles for all cameras. Provides a quick overview "
        "of which cameras have the most OCR issues."
    ),
)
async def get_all_camera_profiles(context: Any = None) -> Dict[str, Any]:
    kb = _get_kb(context)
    profiles = kb.get_all_camera_profiles()

    if not profiles:
        return {"cameras": [], "message": "No camera profiles recorded yet"}

    cameras = []
    for cam_id, profile in profiles.items():
        cameras.append({
            "camera_id": cam_id,
            "total_detections": profile.total_detections,
            "error_rate": round(profile.error_rate, 4),
            "avg_confidence": round(profile.avg_confidence, 4),
        })

    # Sort by error rate descending
    cameras.sort(key=lambda x: -x["error_rate"])
    return {"cameras": cameras, "total_cameras": len(cameras)}


@tool(
    name="get_cross_camera_plates",
    description=(
        "Find plates that have been detected across multiple cameras. "
        "Useful for tracking vehicle movement through the coverage area."
    ),
    param_descriptions={
        "min_cameras": "Minimum number of cameras a plate must appear on (default: 2)",
    },
)
async def get_cross_camera_plates(
    min_cameras: int = 2, context: Any = None
) -> Dict[str, Any]:
    kb = _get_kb(context)
    cross = kb.get_cross_camera_plates(min_cameras=min_cameras)

    results = []
    for plate, cameras in sorted(cross.items(), key=lambda x: -len(x[1])):
        results.append({"plate": plate, "cameras": cameras, "camera_count": len(cameras)})

    return {
        "cross_camera_plates": results[:50],
        "total_found": len(cross),
        "min_cameras_filter": min_cameras,
    }


@tool(
    name="get_ocr_error_patterns",
    description=(
        "Get common OCR character confusion patterns, optionally filtered "
        "by camera. Shows which characters are most frequently misread."
    ),
    param_descriptions={
        "camera_id": "Optional camera ID to filter patterns (empty for all cameras)",
    },
)
async def get_ocr_error_patterns(
    camera_id: str = "", context: Any = None
) -> Dict[str, Any]:
    kb = _get_kb(context)
    patterns = kb.get_error_patterns(
        camera_id=camera_id if camera_id else None
    )

    result: Dict[str, Any] = {}
    for cam_id, pattern_list in patterns.items():
        result[cam_id] = [
            {"confusion": k, "count": v} for k, v in pattern_list
        ]

    return {"patterns": result, "cameras_analyzed": len(result)}


@tool(
    name="detect_plate_anomalies",
    description=(
        "Detect anomalous plate activity in the recent time window. "
        "Identifies frequency spikes, unusual patterns, and new plates."
    ),
    param_descriptions={
        "hours_back": "Hours to look back (default: 1)",
        "spike_threshold": "Factor above average to flag as spike (default: 3.0)",
    },
)
async def detect_plate_anomalies(
    hours_back: float = 1.0,
    spike_threshold: float = 3.0,
    context: Any = None,
) -> Dict[str, Any]:
    kb = _get_kb(context)
    anomalies = kb.get_anomalies(
        hours_back=hours_back, spike_threshold=spike_threshold
    )

    return {
        "anomalies": anomalies,
        "count": len(anomalies),
        "window_hours": hours_back,
        "spike_threshold": spike_threshold,
    }


@tool(
    name="record_plate_feedback",
    description=(
        "Record operator feedback on a detection result. Use 'confirm' to "
        "mark a reading as correct, 'correct' to provide the right plate, "
        "or 'reject' to mark a reading as invalid."
    ),
    param_descriptions={
        "detection_id": "ID of the detection to give feedback on",
        "feedback_type": "One of: confirm, correct, reject",
        "corrected_plate": "The correct plate text (required for 'correct' type)",
        "camera_id": "Camera that captured the detection",
        "original_plate": "The plate text that was detected",
        "notes": "Optional operator notes",
    },
)
async def record_plate_feedback(
    detection_id: str,
    feedback_type: str,
    original_plate: str,
    camera_id: str = "unknown",
    corrected_plate: str = "",
    notes: str = "",
    context: Any = None,
) -> Dict[str, Any]:
    from uni_vision.agent.knowledge import FeedbackEntry

    kb = _get_kb(context)

    if feedback_type not in ("confirm", "correct", "reject"):
        return {"error": f"Invalid feedback_type: {feedback_type}. Must be confirm/correct/reject"}

    if feedback_type == "correct" and not corrected_plate:
        return {"error": "corrected_plate is required when feedback_type is 'correct'"}

    entry = FeedbackEntry(
        detection_id=detection_id,
        original_plate=original_plate,
        corrected_plate=corrected_plate or original_plate,
        feedback_type=feedback_type,
        camera_id=camera_id,
        notes=notes,
    )
    kb.record_feedback(entry)

    return {
        "status": "recorded",
        "feedback_type": feedback_type,
        "detection_id": detection_id,
        "plate": corrected_plate or original_plate,
    }


@tool(
    name="get_recent_feedback",
    description=(
        "Retrieve recent operator feedback entries. Use to review "
        "corrections and confirmations."
    ),
    param_descriptions={
        "hours_back": "Hours to look back (default: 24)",
        "limit": "Maximum entries to return (default: 20)",
    },
)
async def get_recent_feedback(
    hours_back: float = 24,
    limit: int = 20,
    context: Any = None,
) -> Dict[str, Any]:
    kb = _get_kb(context)
    entries = kb.get_recent_feedback(hours_back=hours_back, limit=limit)

    return {
        "feedback": [
            {
                "detection_id": e.detection_id,
                "type": e.feedback_type,
                "original": e.original_plate,
                "corrected": e.corrected_plate,
                "camera": e.camera_id,
                "timestamp": e.timestamp,
                "notes": e.notes,
            }
            for e in entries
        ],
        "total": len(entries),
        "window_hours": hours_back,
    }


@tool(
    name="get_camera_hints",
    description=(
        "Get camera-specific OCR hints based on historical error patterns. "
        "These hints can be injected into OCR prompts to improve accuracy."
    ),
    param_descriptions={
        "camera_id": "Camera identifier",
    },
)
async def get_camera_hints(
    camera_id: str, context: Any = None
) -> Dict[str, Any]:
    kb = _get_kb(context)
    hints = kb.get_camera_hints(camera_id)

    return {
        "camera_id": camera_id,
        "hints": hints,
        "has_hints": bool(hints),
    }


@tool(
    name="save_knowledge",
    description=(
        "Persist the current knowledge base to PostgreSQL. "
        "Call periodically to ensure learning is not lost on restart."
    ),
)
async def save_knowledge(context: Any = None) -> Dict[str, Any]:
    kb = _get_kb(context)
    pg = _get_pg_client(context)

    if pg is None:
        return {"status": "skipped", "reason": "No database connection"}

    await kb.save_to_db(pg)
    stats = kb.get_stats()
    return {"status": "saved", **stats}
