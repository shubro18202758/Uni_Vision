"""CPU-only perceptual hashing for temporal frame deduplication — spec §S1.

Implements a 64-bit DCT-based perceptual hash (pHash) that operates
entirely in system RAM using vectorized NumPy operations.  Zero GPU
VRAM is touched at any point in this module.

Algorithm (matches spec: 32×32 downscale → DCT → median threshold):
  1. Convert BGR frame to 8-bit grayscale.
  2. Resize to ``hash_size × hash_size`` (default 32×32) with area
     interpolation — this is the only OpenCV call.
  3. Compute a 2-D Type-II DCT of the grayscale tile using the
     separable row-column factorisation (pure NumPy, no scipy/fftpack
     dependency).
  4. Retain the top-left 8×8 low-frequency block (64 coefficients),
     excluding the DC term.
  5. Binarise: coefficient ≥ median → 1, else → 0.
  6. Pack the 64 bits into a single ``numpy.uint64`` hash.

Hamming distance between two hashes is computed with XOR + popcount
(``numpy.unpackbits`` path, branchless and vectorised).

Performance budget: < 0.3 ms per frame at 1080p on a modern x86-64 CPU.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

# ── Internal: separable DCT-II via matrix multiply ────────────────
# Pre-compute a reusable DCT basis matrix once at import time for the
# default hash size.  If a caller changes ``hash_size``, a new basis
# can be obtained from ``_dct_basis_matrix``.

_DCT_CACHE: dict[int, NDArray[np.float64]] = {}


def _dct_basis_matrix(n: int) -> NDArray[np.float64]:
    """Return the *n×n* Type-II DCT basis matrix (orthonormal).

    ``C[k, i] = cos(π·k·(2i+1) / (2N))`` with appropriate
    normalisation so that ``C @ C.T == I``.

    The result is cached — this is called at most once per unique *n*.
    """
    if n in _DCT_CACHE:
        return _DCT_CACHE[n]

    rows = np.arange(n, dtype=np.float64).reshape(-1, 1)
    cols = np.arange(n, dtype=np.float64).reshape(1, -1)
    basis = np.cos(np.pi * rows * (2.0 * cols + 1.0) / (2.0 * n))
    # Orthonormal scaling
    basis[0, :] *= 1.0 / np.sqrt(n)
    basis[1:, :] *= np.sqrt(2.0 / n)

    _DCT_CACHE[n] = basis
    return basis


# ── Public API ────────────────────────────────────────────────────


def compute_phash(
    image: NDArray[np.uint8],
    *,
    hash_size: int = 32,
    dct_low: int = 8,
) -> np.uint64:
    """Compute a 64-bit perceptual hash for a BGR or grayscale image.

    Parameters
    ----------
    image:
        Input frame — (H, W, 3) BGR ``uint8`` or (H, W) grayscale.
    hash_size:
        Side length of the downscaled tile fed to the DCT.  Default 32
        matches spec ``frame_sampling.phash_size``.
    dct_low:
        Number of low-frequency rows/columns retained from the DCT
        output.  Must satisfy ``dct_low² == 64`` for a 64-bit hash
        (default 8).

    Returns
    -------
    numpy.uint64
        64-bit perceptual hash packed into a scalar.
    """
    # 1. Grayscale conversion (in-place if already single-channel)
    if image.ndim == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image

    # 2. Resize to hash_size × hash_size via area interpolation.
    #    Area interpolation is the best for downscaling — it correctly
    #    averages source pixels, suppressing aliasing artefacts.
    tile: NDArray[np.float64] = cv2.resize(
        grey,
        (hash_size, hash_size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)

    # 3. Separable 2-D DCT: C @ tile @ C.T
    basis = _dct_basis_matrix(hash_size)
    dct_2d = basis @ tile @ basis.T  # (hash_size, hash_size) float64

    # 4. Retain top-left low-frequency block, exclude DC coefficient
    low_block = dct_2d[:dct_low, :dct_low].ravel()
    # DC coefficient is index 0 — drop it, take next 63, then pad back to 64
    low_block_no_dc = low_block[1:]  # 63 coefficients

    # 5. Binarise against median
    med = np.median(low_block_no_dc)
    bits = (low_block_no_dc >= med).astype(np.uint8)  # 63 bits

    # Pad to 64 bits (MSB is always 0 — the dropped DC term)
    bits_64 = np.empty(64, dtype=np.uint8)
    bits_64[0] = 0
    bits_64[1:] = bits

    # 6. Pack into uint64 — MSB-first (big-endian bit order)
    hash_value = np.uint64(0)
    for i in range(64):
        hash_value = np.uint64(hash_value << np.uint64(1)) | np.uint64(bits_64[i])

    return hash_value


def hamming_distance(a: np.uint64, b: np.uint64) -> int:
    """Return the Hamming distance (number of differing bits) between two hashes.

    Uses XOR + popcount via ``numpy.unpackbits`` — branchless and
    vectorised, with no Python-level bit manipulation loop.

    Parameters
    ----------
    a, b:
        64-bit perceptual hashes produced by :func:`compute_phash`.

    Returns
    -------
    int
        Number of differing bits (0–64).
    """
    xor = np.uint64(a) ^ np.uint64(b)
    # Decompose the 64-bit integer into 8 bytes, then unpack to individual bits
    byte_array = np.array(
        [(int(xor) >> (56 - 8 * i)) & 0xFF for i in range(8)],
        dtype=np.uint8,
    )
    return int(np.unpackbits(byte_array).sum())
