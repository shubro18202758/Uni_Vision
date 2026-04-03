"""Composable preprocessing chain — sequences multiple Preprocessors.

Applies an ordered list of ``Preprocessor``-protocol-compatible stages,
skipping any whose ``enabled`` property returns ``False``.  Each stage
receives the output of the previous stage as input.

Used by the pipeline to run S5 → S6 (and any future stages) in a
single composable call.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from uni_vision.monitoring.metrics import STAGE_LATENCY

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from uni_vision.contracts.preprocessor import Preprocessor

log = structlog.get_logger()


class PreprocessingChain:
    """Ordered pipeline of ``Preprocessor`` stages.

    Parameters
    ----------
    stages:
        Preprocessor instances in execution order.  Disabled stages
        are automatically skipped at runtime.
    """

    def __init__(self, stages: list[Preprocessor]) -> None:
        self._stages = stages

    def run(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Execute every enabled stage sequentially, returning the
        final processed image."""
        result = image

        for stage in self._stages:
            if not stage.enabled:
                continue

            t0 = time.perf_counter()
            result = stage.process(result)
            elapsed = time.perf_counter() - t0

            STAGE_LATENCY.labels(stage=stage.name).observe(elapsed)
            log.debug("preprocess_stage_done", stage=stage.name, elapsed_ms=elapsed * 1000)

        return result
