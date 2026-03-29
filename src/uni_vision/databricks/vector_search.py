"""FAISS vector search for plate similarity and agent RAG.

Provides a vector index that embeds plate observation text using
``sentence-transformers`` and indexes them with Facebook AI
Similarity Search (FAISS) for:

  * **Plate similarity search** — find plates that look similar to
    a query (catches OCR variations, partial reads, typos).
  * **Agent knowledge RAG** — retrieve relevant historical context
    for the agentic sub-system's reasoning loop.
  * **Anomaly detection** — flag plates whose embeddings are distant
    from any cluster (novel / unexpected formats).
  * **Fuzzy deduplication** — identify plates that are likely the
    same vehicle despite OCR character errors.

The index is persisted to disk and rebuilt periodically or on demand.

Requires: ``pip install 'uni-vision[databricks]'``
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports
_faiss = None
_SentenceTransformer = None
_np = None


def _ensure_imports() -> None:
    global _faiss, _SentenceTransformer, _np
    if _faiss is not None:
        return
    try:
        import faiss  # type: ignore[import-untyped]
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        _faiss = faiss
        _SentenceTransformer = SentenceTransformer
        _np = np
    except ImportError as exc:
        raise ImportError(
            "FAISS / sentence-transformers not installed. "
            "Run: pip install 'uni-vision[databricks]'"
        ) from exc


class VectorSearchEngine:
    """FAISS-backed vector search for plate text similarity.

    Parameters
    ----------
    index_path : str
        Path to persist the FAISS index binary.
    metadata_path : str
        Path to persist plate metadata JSON.
    embedding_model : str
        Sentence-transformers model for text encoding.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    nprobe : int
        FAISS search probe count (higher = better recall, slower).
    top_k : int
        Default number of results to return.
    similarity_threshold : float
        Minimum cosine similarity for a result to be included.
    """

    def __init__(
        self,
        index_path: str = "./data/faiss/plate_index.bin",
        metadata_path: str = "./data/faiss/plate_metadata.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        nprobe: int = 10,
        top_k: int = 20,
        similarity_threshold: float = 0.65,
    ) -> None:
        _ensure_imports()
        self._index_path = str(Path(index_path).resolve())
        self._metadata_path = str(Path(metadata_path).resolve())
        self._model_name = embedding_model
        self._dim = embedding_dim
        self._nprobe = nprobe
        self._top_k = top_k
        self._threshold = similarity_threshold

        self._model = None
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._id_counter = 0

    # ── Lifecycle ─────────────────────────────────────────────────

    def initialise(self) -> None:
        """Load or create the FAISS index and embedding model."""
        # Load embedding model
        self._model = _SentenceTransformer(self._model_name)

        # Load or create FAISS index
        if os.path.exists(self._index_path):
            self._index = _faiss.read_index(self._index_path)
            logger.info(
                "faiss_index_loaded path=%s vectors=%d",
                self._index_path,
                self._index.ntotal,
            )
        else:
            # Use IndexFlatIP (inner product) for cosine similarity
            # after L2-normalising all vectors
            self._index = _faiss.IndexFlatIP(self._dim)
            logger.info("faiss_index_created dim=%d", self._dim)

        # Load metadata
        if os.path.exists(self._metadata_path):
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            self._id_counter = len(self._metadata)
        else:
            self._metadata = []

    def shutdown(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        self._save()
        logger.info(
            "faiss_index_saved vectors=%d",
            self._index.ntotal if self._index else 0,
        )

    def _save(self) -> None:
        """Write index and metadata to disk."""
        if self._index is None:
            return

        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        _faiss.write_index(self._index, self._index_path)

        os.makedirs(os.path.dirname(self._metadata_path), exist_ok=True)
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    # ── Embedding ─────────────────────────────────────────────────

    def _encode(self, texts: List[str]):
        """Encode text(s) into L2-normalised vectors."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return _np.array(embeddings, dtype=_np.float32)

    # ── Indexing ──────────────────────────────────────────────────

    def add_plate_observation(
        self,
        plate_text: str,
        camera_id: str,
        confidence: float,
        engine: str,
        validation_status: str,
        timestamp: Optional[float] = None,
    ) -> int:
        """Add a plate observation to the vector index.

        The plate text is embedded and added to the FAISS index.
        Returns the internal vector ID.
        """
        if self._index is None or self._model is None:
            raise RuntimeError("VectorSearchEngine not initialised")

        # Create a rich text representation for embedding
        text_repr = f"plate:{plate_text} camera:{camera_id} engine:{engine} status:{validation_status}"
        vec = self._encode([text_repr])

        vec_id = self._id_counter
        self._index.add(vec)
        self._metadata.append({
            "id": vec_id,
            "plate_text": plate_text,
            "camera_id": camera_id,
            "confidence": confidence,
            "engine": engine,
            "validation_status": validation_status,
            "timestamp": timestamp or time.time(),
        })
        self._id_counter += 1

        # Periodic auto-save every 500 additions
        if self._id_counter % 500 == 0:
            self._save()

        return vec_id

    def add_batch(self, observations: List[Dict[str, Any]]) -> int:
        """Batch-add multiple observations for efficiency."""
        if not observations:
            return 0

        texts = []
        for obs in observations:
            text = (
                f"plate:{obs['plate_text']} "
                f"camera:{obs.get('camera_id', '')} "
                f"engine:{obs.get('engine', '')} "
                f"status:{obs.get('validation_status', '')}"
            )
            texts.append(text)

        vecs = self._encode(texts)
        self._index.add(vecs)

        for obs in observations:
            self._metadata.append({
                "id": self._id_counter,
                "plate_text": obs["plate_text"],
                "camera_id": obs.get("camera_id", ""),
                "confidence": obs.get("confidence", 0.0),
                "engine": obs.get("engine", ""),
                "validation_status": obs.get("validation_status", ""),
                "timestamp": obs.get("timestamp", time.time()),
            })
            self._id_counter += 1

        self._save()
        return len(observations)

    # ── Search ────────────────────────────────────────────────────

    def search_similar_plates(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Find plates similar to the query text.

        Parameters
        ----------
        query : str
            Plate text or partial plate to search for.
        top_k : int, optional
            Number of results (default: configured top_k).
        threshold : float, optional
            Minimum similarity score (default: configured threshold).

        Returns
        -------
        list[dict]
            Matching plates with similarity scores and metadata.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k or self._top_k, self._index.ntotal)
        thresh = threshold or self._threshold

        # Embed the query with context
        query_repr = f"plate:{query}"
        query_vec = self._encode([query_repr])

        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            sim = float(score)
            if sim < thresh:
                continue

            meta = self._metadata[idx]
            results.append({
                "plate_text": meta["plate_text"],
                "similarity": round(sim, 4),
                "camera_id": meta["camera_id"],
                "confidence": meta["confidence"],
                "engine": meta["engine"],
                "validation_status": meta["validation_status"],
                "timestamp": meta["timestamp"],
            })

        return results

    def search_by_camera(
        self, query: str, camera_id: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search similar plates filtered to a specific camera."""
        # Get more results than needed, then filter
        all_results = self.search_similar_plates(
            query, top_k=top_k * 3, threshold=0.3
        )
        filtered = [r for r in all_results if r["camera_id"] == camera_id]
        return filtered[:top_k]

    def find_potential_duplicates(
        self, similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Find plate pairs that may be OCR variations of each other.

        Uses an optimised approach: encode unique plates in batch,
        then perform FAISS nearest-neighbour search on the unique set
        to avoid O(n²) individual queries.
        """
        if self._index is None or self._index.ntotal < 2:
            return []

        # Collect unique plate texts
        unique_plates: Dict[str, List[Dict[str, Any]]] = {}
        for meta in self._metadata:
            pt = meta["plate_text"]
            if pt not in unique_plates:
                unique_plates[pt] = []
            unique_plates[pt].append(meta)

        plate_list = list(unique_plates.keys())
        if len(plate_list) < 2:
            return []

        # Batch-encode all unique plates at once for efficiency
        texts = [f"plate:{p}" for p in plate_list]
        vecs = self._encode(texts)

        # Search each unique plate against the full index
        k = min(6, self._index.ntotal)
        scores, indices = self._index.search(vecs, k)

        duplicates = []
        seen_pairs: set = set()

        for i, plate_text in enumerate(plate_list):
            for j in range(k):
                idx = int(indices[i][j])
                sim = float(scores[i][j])
                if idx < 0 or idx >= len(self._metadata):
                    continue
                if sim < similarity_threshold:
                    continue
                match_text = self._metadata[idx]["plate_text"]
                if match_text == plate_text:
                    continue
                pair_key = tuple(sorted([plate_text, match_text]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                duplicates.append({
                    "plate_a": plate_text,
                    "plate_b": match_text,
                    "similarity": round(sim, 4),
                    "observations_a": len(unique_plates.get(plate_text, [])),
                    "observations_b": len(unique_plates.get(match_text, [])),
                })

        return sorted(duplicates, key=lambda x: -x["similarity"])

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return vector index statistics."""
        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "embedding_dim": self._dim,
            "embedding_model": self._model_name,
            "index_path": self._index_path,
            "metadata_entries": len(self._metadata),
            "nprobe": self._nprobe,
            "similarity_threshold": self._threshold,
        }

    def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the FAISS index from stored metadata.

        Useful after parameter changes or to compact the index.
        """
        if not self._metadata:
            return {"status": "empty", "vectors": 0}

        # Re-encode all stored observations
        texts = []
        for meta in self._metadata:
            text = (
                f"plate:{meta['plate_text']} "
                f"camera:{meta.get('camera_id', '')} "
                f"engine:{meta.get('engine', '')} "
                f"status:{meta.get('validation_status', '')}"
            )
            texts.append(text)

        vecs = self._encode(texts)

        # Rebuild index
        self._index = _faiss.IndexFlatIP(self._dim)
        self._index.add(vecs)
        self._save()

        logger.info("faiss_index_rebuilt vectors=%d", self._index.ntotal)
        return {"status": "rebuilt", "vectors": self._index.ntotal}

    # ── Time-Range Search ─────────────────────────────────────────

    def search_by_time_range(
        self,
        query: str,
        start_ts: float,
        end_ts: float,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search similar plates within a specific time window.

        Parameters
        ----------
        query : str
            Plate text or partial plate to search for.
        start_ts : float
            Start of time window (UNIX timestamp).
        end_ts : float
            End of time window (UNIX timestamp).
        """
        # Get more candidates so we have enough after filtering
        results = self.search_similar_plates(
            query,
            top_k=(top_k or self._top_k) * 5,
            threshold=threshold,
        )
        filtered = [
            r for r in results
            if start_ts <= r.get("timestamp", 0) <= end_ts
        ]
        return filtered[: (top_k or self._top_k)]

    # ── Cluster Analysis ──────────────────────────────────────────

    def get_cluster_analysis(
        self, n_clusters: int = 8, max_samples: int = 2000
    ) -> Dict[str, Any]:
        """Group indexed plates into embedding clusters for pattern recognition.

        Uses K-Means on the FAISS vectors to identify clusters of
        similar plates — useful for spotting OCR error families,
        frequent visitors, or geographic plate patterns.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to fit (capped at unique-plate count).
        max_samples : int
            Maximum vectors to cluster (for performance).

        Returns
        -------
        dict
            Cluster sizes, representative plates, and coherence scores.
        """
        if self._index is None or self._index.ntotal < n_clusters:
            return {"error": "not_enough_data", "total_vectors": self._index.ntotal if self._index else 0}

        total = min(self._index.ntotal, max_samples)
        n_clusters = min(n_clusters, total)

        # Reconstruct vectors from the flat index
        vecs = _np.zeros((total, self._dim), dtype=_np.float32)
        for i in range(total):
            vecs[i] = self._index.reconstruct(i)

        # Use FAISS K-Means (faster than sklearn for float32 vectors)
        kmeans = _faiss.Kmeans(self._dim, n_clusters, niter=20, verbose=False)
        kmeans.train(vecs)
        _, assignments = kmeans.index.search(vecs, 1)
        assignments = assignments.flatten()

        clusters = []
        for cid in range(n_clusters):
            member_ids = _np.where(assignments == cid)[0]
            if len(member_ids) == 0:
                continue

            # Representative plates for this cluster
            plates_in_cluster: Dict[str, int] = {}
            for mid in member_ids:
                if mid < len(self._metadata):
                    pt = self._metadata[int(mid)]["plate_text"]
                    plates_in_cluster[pt] = plates_in_cluster.get(pt, 0) + 1

            # Coherence = how tight the cluster is (avg intra-cluster similarity)
            centroid = kmeans.centroids[cid: cid + 1]
            sims = _np.dot(vecs[member_ids], centroid.T).flatten()
            coherence = float(_np.mean(sims)) if len(sims) > 0 else 0.0

            top_plates = sorted(plates_in_cluster.items(), key=lambda x: -x[1])[:5]
            clusters.append({
                "cluster_id": cid,
                "size": int(len(member_ids)),
                "coherence": round(coherence, 4),
                "top_plates": [{"plate": p, "count": c} for p, c in top_plates],
            })

        clusters.sort(key=lambda c: -c["size"])
        return {
            "n_clusters": n_clusters,
            "total_vectors_analysed": total,
            "clusters": clusters,
        }

    # ── Health ────────────────────────────────────────────────────

    def get_health(self) -> Dict[str, Any]:
        """Return health summary for monitoring dashboards."""
        unique_plates = len({m["plate_text"] for m in self._metadata})
        unique_cameras = len({m["camera_id"] for m in self._metadata})
        return {
            "status": "active" if self._index and self._index.ntotal > 0 else "empty",
            "total_vectors": self._index.ntotal if self._index else 0,
            "unique_plates": unique_plates,
            "unique_cameras": unique_cameras,
            "embedding_model": self._model_name,
            "embedding_dim": self._dim,
            "index_type": "IndexFlatIP",
            "index_size_bytes": (
                os.path.getsize(self._index_path) if os.path.exists(self._index_path) else 0
            ),
        }
