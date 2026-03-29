"""Component abstraction layer for dynamic pipeline composition.

Every CV model, library, or algorithmic package — whether built-in or
downloaded from HuggingFace/PyPI — is wrapped as a ``CVComponent`` that
the Manager Agent can discover, load, compose into pipelines, and swap
at runtime without touching the core orchestrator.
"""

from uni_vision.components.base import (
    CVComponent,
    ComponentCapability,
    ComponentType,
    ComponentState,
)

__all__ = [
    "CVComponent",
    "ComponentCapability",
    "ComponentType",
    "ComponentState",
]
