"""Agentic subsystem — Qwen 3.5 9B as an autonomous reasoning engine.

This package transforms the Uni_Vision pipeline from a passive
LLM-assisted OCR system into a truly agentic architecture where the
LLM actively manages pipelines, performs autonomous diagnostics,
answers natural-language queries, and optimises its own behaviour.

Architecture overview:

  ┌─────────────────────────────────────────────┐
  │              AgentCoordinator                │
  │  (multi-step ReAct loop + tool dispatch)     │
  ├─────────────────────────────────────────────┤
  │  ToolRegistry  │  WorkingMemory  │ LLMClient │
  ├────────────────┴─────────────────┴──────────┤
  │              Tool implementations            │
  │  (query_detections, pipeline_stats, etc.)    │
  └─────────────────────────────────────────────┘

Modules:
  tools.py     — Tool base class, registry, decorator
  llm_client.py — Shared async Ollama client for agent reasoning
  prompts.py   — ReAct system prompt + tool schema generation
  memory.py    — Working memory (conversation + context window)
  loop.py      — Multi-step ReAct agent loop
  coordinator.py — Top-level agent coordinator
"""
