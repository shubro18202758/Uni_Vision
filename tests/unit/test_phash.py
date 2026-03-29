"""Tests for the pHash computation — spec §S1."""

from __future__ import annotations

import numpy as np
import pytest


class TestPHash:
    """Test the DCT-based perceptual hash implementation."""

    def test_deterministic_output(self, sample_bgr_image):
        from uni_vision.ingestion.phash import compute_phash

        h1 = compute_phash(sample_bgr_image)
        h2 = compute_phash(sample_bgr_image)
        assert h1 == h2

    def test_returns_uint64(self, sample_bgr_image):
        from uni_vision.ingestion.phash import compute_phash

        h = compute_phash(sample_bgr_image)
        assert h.dtype == np.uint64

    def test_grayscale_input(self, sample_grayscale_image):
        from uni_vision.ingestion.phash import compute_phash

        h = compute_phash(sample_grayscale_image)
        assert h.dtype == np.uint64

    def test_different_images_different_hashes(self):
        from uni_vision.ingestion.phash import compute_phash

        rng = np.random.default_rng(1)
        img_a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        rng2 = np.random.default_rng(999)
        img_b = rng2.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        ha = compute_phash(img_a)
        hb = compute_phash(img_b)
        # Very different random images should have different hashes
        assert ha != hb

    def test_similar_images_close_hashes(self):
        from uni_vision.ingestion.phash import compute_phash, hamming_distance

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
        # Add mild noise
        noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        h1 = compute_phash(img)
        h2 = compute_phash(noisy)
        dist = hamming_distance(h1, h2)
        # Similar images should have ≤ 5 Hamming distance (threshold)
        assert dist <= 10  # generous bound for noisy test

    def test_custom_hash_size(self, sample_bgr_image):
        from uni_vision.ingestion.phash import compute_phash

        h = compute_phash(sample_bgr_image, hash_size=16, dct_low=8)
        assert h.dtype == np.uint64


class TestHammingDistance:
    """Test the Hamming distance utility."""

    def test_identical_hashes(self):
        from uni_vision.ingestion.phash import hamming_distance

        h = np.uint64(0xABCDEF0123456789)
        assert hamming_distance(h, h) == 0

    def test_single_bit_difference(self):
        from uni_vision.ingestion.phash import hamming_distance

        a = np.uint64(0)
        b = np.uint64(1)
        assert hamming_distance(a, b) == 1

    def test_all_bits_different(self):
        from uni_vision.ingestion.phash import hamming_distance

        a = np.uint64(0)
        b = np.uint64(0xFFFFFFFFFFFFFFFF)
        assert hamming_distance(a, b) == 64

    def test_known_pair(self):
        from uni_vision.ingestion.phash import hamming_distance

        a = np.uint64(0b1010)
        b = np.uint64(0b0101)
        assert hamming_distance(a, b) == 4


class TestDCTBasisMatrix:
    """Test the cached DCT basis matrix computation."""

    def test_orthonormality(self):
        from uni_vision.ingestion.phash import _dct_basis_matrix

        C = _dct_basis_matrix(8)
        identity = C @ C.T
        np.testing.assert_allclose(identity, np.eye(8), atol=1e-10)

    def test_cache_reuse(self):
        from uni_vision.ingestion.phash import _dct_basis_matrix

        c1 = _dct_basis_matrix(32)
        c2 = _dct_basis_matrix(32)
        assert c1 is c2  # same object from cache
