"""Protocol contract for image preprocessors (S5, S6)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class Preprocessor(Protocol):
    """A single, composable image transformation step.

    Implementations: HoughStraightener (S5), CLAHEBilateralEnhancer (S6),
    identity pass-through.

    Preprocessors are pure CPU operations.  They receive a numpy array
    and return a new numpy array — never mutating the input.
    """

    @property
    def name(self) -> str:
        """Identifier for metrics labels and structured logs."""
        ...

    @property
    def enabled(self) -> bool:
        """Whether this step is active (read from config)."""
        ...

    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply the transformation.

        Args:
            image: ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Transformed image as a **new** array (input is not mutated).
        """
        ...
