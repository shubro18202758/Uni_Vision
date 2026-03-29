"""Manager Agent subsystem — Qwen 3.5 9B as meta-orchestrator.

Instead of acting as a direct OCR / reasoning LLM, Qwen operates as
a Manager Agent that:

  1. Analyses incoming frame context to decide what CV tasks are required.
  2. Discovers and downloads specialised open-source models/libraries
     from HuggingFace Hub, PyPI, or GitHub.
  3. Resolves dependency conflicts between components.
  4. Composes context-adaptive pipelines dynamically.
  5. Validates assembled pipelines before activating them.
  6. Manages component lifecycle (load/unload/swap) under VRAM budget.
"""

from uni_vision.manager.agent import ManagerAgent

__all__ = ["ManagerAgent"]
