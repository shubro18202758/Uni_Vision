"""LLM prompt templates for the Manager Agent (Gemma 4 E2B).

The Manager Agent uses Gemma 4 not for direct OCR but as a reasoning
engine to decide WHICH components to load, HOW to compose them into
a pipeline, and HOW to resolve conflicts.  All interactions use
structured JSON output for reliable parsing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MANAGER_SYSTEM_PROMPT = """\
You are the Manager Agent for Uni_Vision — an adaptive real-time computer
vision pipeline running on a single NVIDIA RTX 4070 (8 GB VRAM).

Your role is NOT to perform inference directly.  Instead, you orchestrate
the pipeline by:

1. ANALYSING frame context to determine what CV tasks are needed.
2. SELECTING the best available components (models, libraries, algorithms)
   for each task from the registry or external sources (HuggingFace, PyPI,
   TorchHub).
3. COMPOSING a pipeline of stages that processes frames end-to-end.
4. RESOLVING conflicts between components (VRAM budget, dependency clashes,
   device mismatches, inter-component compatibility).
5. VALIDATING that the assembled pipeline is functional.
6. ADAPTING in real-time based on performance feedback, scene transitions,
   and environmental changes.

You have access to tools.  When you need to take an action, respond with
a JSON object:

{{
  "thought": "your step-by-step reasoning",
  "action": "tool_name",
  "arguments": {{ ... }}
}}

When you have a final answer, respond with:

{{
  "thought": "your reasoning",
  "answer": "your final response or decision summary",
  "pipeline_actions": {{
    "decision": "use_existing | swap_components | build_new_pipeline | download_and_integrate | offload_for_vram",
    "components_to_load": ["component_id_1"],
    "components_to_unload": ["component_id_2"],
    "components_to_download": [{{"source": "huggingface | pypi | torchhub", "source_id": "repo/model", "reason": "..."}}],
    "pipeline_stages": ["stage_1_capability", "stage_2_capability"],
    "reasoning": "why this configuration"
  }}
}}

CONSTRAINTS:
- Total VRAM budget: {vram_ceiling_mb} MB.  Currently used: {vram_used_mb} MB.
- Available VRAM: {vram_available_mb} MB.
- Prefer TRUSTED/BUILT-IN components when they satisfy requirements.
- Only download external components when no local component can do the job.
- Always check compatibility before loading new components.
- Minimise unnecessary component swaps — they cost latency.
- Keep total loaded model VRAM under budget at all times.
- Ensure the pipeline stages chain correctly (output of stage N = input of stage N+1).

ADAPTIVE PIPELINE CONTEXT:
{adaptive_context}

AVAILABLE TOOLS:
{tool_descriptions}

Currently loaded components:
{loaded_components}

Component registry (available locally):
{registry_summary}
"""


CONTEXT_ANALYSIS_PROMPT = """\
Analyse the following frame information and determine what computer vision
tasks and capabilities are required.

Frame metadata:
- Camera ID: {camera_id}
- Resolution: {width}x{height}
- Timestamp: {timestamp}
- Previous context: {previous_context}
- Scene hints: {scene_hints}

Based on the YOLO-E / zero-shot learning paradigm, determine:

1. Scene type (traffic / parking / surveillance / industrial / indoor / unknown)
2. Required capabilities — what MUST be detected/processed
3. Optional capabilities — nice-to-have if VRAM allows
4. Processing priority (critical / high / normal / low)

Respond in JSON:
{{
  "scene_type": "...",
  "required_capabilities": ["capability_1", "capability_2"],
  "optional_capabilities": ["capability_3"],
  "priority": "normal",
  "reasoning": "why these capabilities are needed"
}}
"""


CONFLICT_RESOLUTION_PROMPT = """\
The following conflicts were detected when trying to compose a pipeline:

{conflicts_description}

Current VRAM status:
- Total: {vram_total_mb} MB
- Used: {vram_used_mb} MB
- Available: {vram_available_mb} MB

Loaded components:
{loaded_components}

Proposed new components:
{proposed_components}

Resolve these conflicts.  For each conflict, provide:
1. Which component(s) to keep vs. remove
2. Whether any components can be offloaded to CPU
3. Whether alternative lighter-weight components exist

Respond in JSON:
{{
  "resolutions": [
    {{
      "conflict_index": 0,
      "action": "unload_component | offload_to_cpu | swap_for_lighter | skip_optional",
      "target_component_id": "...",
      "replacement_component_id": "..." or null,
      "reasoning": "..."
    }}
  ],
  "final_feasible": true
}}
"""


COMPONENT_SEARCH_PROMPT = """\
I need a component with the following capability: {capability}

Context: {context_description}

Requirements:
- Must support: {device} device
- VRAM budget remaining: {vram_available_mb} MB
- Preferred source: {preferred_source}

Currently available in registry:
{registry_candidates}

If no suitable local component exists, suggest a HuggingFace model or
PyPI package to download.  Prefer well-known, well-maintained models
with permissive licenses.

Respond in JSON:
{{
  "use_local": true/false,
  "selected_component_id": "..." or null,
  "download_suggestion": {{
    "source": "huggingface | pypi",
    "source_id": "repo/model or package_name",
    "model_class": "module.ClassName",
    "estimated_vram_mb": 500,
    "python_requirements": ["package1", "package2"],
    "reasoning": "why this model/package"
  }} or null
}}
"""


PIPELINE_VALIDATION_PROMPT = """\
Validate the following pipeline blueprint:

Blueprint: {blueprint_name}
Stages:
{stages_description}

For each stage, verify:
1. The assigned component has the required capability
2. The output type of stage N matches the input type of stage N+1
3. The total VRAM fits within {vram_available_mb} MB

Respond in JSON:
{{
  "valid": true/false,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1"]
}}
"""


ADAPTATION_DECISION_PROMPT = """\
The adaptive feedback subsystem has detected performance issues:

{adaptation_signals}

Current pipeline performance:
{performance_summary}

Temporal context (multi-frame awareness):
{temporal_context}

Quality scores (Bayesian-ranked components):
{quality_rankings}

Degraded components:
{degraded_components}

Available fallback alternatives:
{fallback_alternatives}

Decide the best adaptation strategy.  Options:
1. "swap" — Replace a specific stage's component with a higher-ranked fallback
2. "recompose" — Rebuild the entire pipeline from scratch for current context
3. "downgrade" — Switch to lighter components to relieve VRAM pressure
4. "tolerate" — Current performance is acceptable given constraints

Respond in JSON:
{{
  "strategy": "swap | recompose | downgrade | tolerate",
  "target_stages": ["stage_name"],
  "replacement_components": {{"stage_name": "new_component_id"}},
  "reasoning": "why this adaptation"
}}
"""


SCENE_TRANSITION_PROMPT = """\
A scene transition has been detected on camera {camera_id}:

Previous scene: {old_scene}
New scene: {new_scene}
Transition confidence: {confidence}

Temporal context:
{temporal_summary}

The current pipeline was optimised for the previous scene type.
Determine what capability changes are needed for the new scene.

Respond in JSON:
{{
  "capabilities_to_add": ["capability_1"],
  "capabilities_to_remove": ["capability_2"],
  "urgency": "immediate | next_frame | gradual",
  "reasoning": "why these changes"
}}
"""


OPEN_DISCOVERY_PROMPT = """\
You are the adaptive intelligence engine for Uni_Vision — a real-time CV
pipeline on an NVIDIA RTX 4070 (8 GB VRAM).  Your task is to DEEPLY REASON
about what computer vision processing this frame requires, going BEYOND any
fixed list of known capabilities.

Frame metadata:
- Camera ID: {camera_id}
- Resolution: {width}x{height}
- Timestamp: {timestamp}
- Average brightness: {brightness}/255
- Previous scene: {previous_context}
- Scene hints: {scene_hints}
- VRAM available: {vram_available_mb} MB

Currently loaded components:
{loaded_components}

Well-known capabilities (these are HINTS, not an exhaustive list):
{known_capabilities}

INSTRUCTIONS:
1. Determine what the frame needs based on camera context, scene hints, and
   any patterns you can infer.
2. Go BEYOND the well-known list — if the situation warrants a specialised
   capability (e.g., weather detection, wildlife recognition, medical imaging,
   underwater correction, infrared processing), suggest it.
3. For each capability you identify, generate one or more internet search
   queries that would find the best open-source model or library to fulfil it.
4. Think about the FULL processing pipeline: preprocessing needs, core
   detection/recognition, postprocessing, and any domain-specific steps.

Respond in JSON:
{{
  "scene_type": "traffic | parking | surveillance | industrial | indoor | unknown | <custom>",
  "reasoning": "your detailed analysis of what the frame needs and why",
  "required_capabilities": [
    {{
      "name": "capability_name",
      "is_standard": true,
      "rationale": "why this is needed"
    }}
  ],
  "optional_capabilities": [
    {{
      "name": "capability_name",
      "is_standard": false,
      "rationale": "why this could help"
    }}
  ],
  "discovery_queries": [
    {{
      "query": "search string for HuggingFace/PyPI/GitHub",
      "source": "huggingface | pypi | github | all",
      "capability_hint": "what capability this addresses",
      "context_rationale": "why this search is needed",
      "priority": 1
    }}
  ],
  "priority": "critical | high | normal | low"
}}
"""


CANDIDATE_EVALUATION_PROMPT = """\
You are the intelligent component selection engine for Uni_Vision.
Evaluate the following candidate components discovered from the open internet
and decide which ones to install for the current pipeline needs.

Required capability: {capability_description}
Frame context: {frame_context}
VRAM available: {vram_available_mb} MB

Discovered candidates:
{candidates_json}

EVALUATION CRITERIA (reason carefully about each):
1. **Relevance**: Does the candidate actually solve the required capability?
2. **VRAM fit**: Will it fit within the available VRAM budget?
3. **Trust**: Is it from a well-known source? High downloads? Permissive license?
4. **Quality**: Is it actively maintained? Compatible with PyTorch/CUDA?
5. **Efficiency**: Smaller models that perform well are preferred for real-time use.
6. **Compatibility**: Will it work with Python 3.10+ and CUDA on RTX 4070?

Respond in JSON:
{{
  "selected_candidate_index": 0,
  "reasoning": "why this candidate is the best choice",
  "rejected_reasons": {{
    "1": "why candidate at index 1 was not selected",
    "2": "why candidate at index 2 was not selected"
  }},
  "install_confidence": 0.85,
  "vram_estimate_mb": 300,
  "alternative_search_queries": ["query if none of the candidates are good enough"]
}}
"""


DYNAMIC_PIPELINE_ORDERING_PROMPT = """\
You are composing a processing pipeline for Uni_Vision. Some capabilities
in this pipeline are non-standard (discovered dynamically by the LLM).
Determine the correct execution order and data flow for all stages.

Standard stage ordering (for reference — lower number = earlier):
{standard_ordering}

Pipeline capabilities to order:
{capabilities_list}

For each capability, determine:
1. Execution order (10-99, lower = earlier in pipeline)
2. Input key name (what data it reads from the pipeline state)
3. Output key name (what data it produces)

Standard categories:
- Preprocessing (10-29): Denoising, enhancement, super-resolution, correction
- Classification (30-39): Scene classification, weather detection
- Detection (40-49): Object/vehicle/person/plate detection
- Segmentation (50-59): Instance/semantic segmentation
- Tracking (60-79): Object tracking, plate detection
- Recognition/OCR (80-89): Text recognition, OCR
- Analysis (90-99): Anomaly detection, action recognition, depth estimation

Respond in JSON:
{{
  "stage_ordering": [
    {{
      "capability": "capability_name",
      "order": 40,
      "input_key": "frame",
      "output_key": "detections",
      "reasoning": "why this position and these IO keys"
    }}
  ]
}}
"""


def build_manager_system_prompt(
    *,
    vram_ceiling_mb: int,
    vram_used_mb: int,
    loaded_components: List[Dict[str, Any]],
    registry_summary: List[Dict[str, Any]],
    tool_descriptions: str,
    adaptive_context: Optional[str] = None,
) -> str:
    """Build the complete system prompt for the Manager Agent.

    Parameters
    ----------
    adaptive_context:
        Optional string summarising current adaptive state (degraded
        components, scene status, temporal context, quality scores).
    """
    loaded_str = "\n".join(
        f"  - {c['component_id']} ({c['type']}) — {', '.join(c.get('capabilities', []))}"
        f"  [{c.get('vram_mb', 0)} MB VRAM, state={c.get('state', '?')}]"
        for c in loaded_components
    ) or "  (none loaded)"

    registry_str = "\n".join(
        f"  - {c['component_id']} ({c['source']}) — {', '.join(c.get('capabilities', []))}"
        for c in registry_summary
    ) or "  (registry empty)"

    adaptive_str = adaptive_context or "  No adaptive data yet — first frame."

    return MANAGER_SYSTEM_PROMPT.format(
        vram_ceiling_mb=vram_ceiling_mb,
        vram_used_mb=vram_used_mb,
        vram_available_mb=vram_ceiling_mb - vram_used_mb,
        tool_descriptions=tool_descriptions,
        loaded_components=loaded_str,
        registry_summary=registry_str,
        adaptive_context=adaptive_str,
    )
