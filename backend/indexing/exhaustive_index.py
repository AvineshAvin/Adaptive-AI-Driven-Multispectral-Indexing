import time
import numpy as np
from typing import List, Dict, Any, Tuple

class ExhaustiveIndex:
    """
    Baseline 1: Static Flat / Brute-Force Index.
    Scans the entire database linearly to compute cosine similarities.
    Candidate search space is 100% of the repository.
    """
    def __init__(self, ids: List[str], features: np.ndarray, metadata: List[Dict[str, Any]]):
        self.ids = ids
        self.features = features  # Shape: (N, D), assumed L2-normalized
        self.metadata = metadata
        self.total_count = len(ids)

    def search(self, query_feat: np.ndarray, top_k: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.perf_counter()
        
        # Ensure query is 1D normalized
        q = query_feat.flatten()
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-7:
            q = q / q_norm

        # Vectorized dot product (cosine similarity)
        sims = np.dot(self.features, q)

        # Get top-k indices
        if top_k >= self.total_count:
            top_indices = np.argsort(-sims)
        else:
            top_indices = np.argpartition(-sims, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "score": float(sims[idx]),
                "class_name": self.metadata[idx]["class_name"],
                "path": self.metadata[idx]["path"],
                "tier": "COLD_FLAT"
            })

        stats = {
            "search_time_ms": elapsed_ms,
            "candidates_evaluated": self.total_count,
            "search_space_ratio": 1.0,
            "mode": "Exhaustive Flat Baseline"
        }
        return results, stats
