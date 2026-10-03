import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ..config import (
    HOT_TIER_THRESHOLD,
    WARM_TIER_THRESHOLD,
    HOT_CONFIDENCE_THRESHOLD,
    WARM_CONFIDENCE_THRESHOLD,
)

class AdaptiveTieredIndex:
    """
    Proposed Adaptive Multi-Tier Indexing Architecture.
    Organizes repository into:
      - HOT Tier: High-activity, in-memory, immediate search.
      - WARM Tier: Moderate-activity, inverted cluster lists.
      - COLD Tier: Low-activity archive, scanned on fallback.
    """
    def __init__(self,
                 ids: List[str],
                 features: np.ndarray,
                 spectral_features: np.ndarray,
                 spatial_features: np.ndarray,
                 metadata: List[Dict[str, Any]]):
        self.ids = ids
        self.features = features                      # Fused features (N, D)
        self.spectral_features = spectral_features    # Spectral branch (N, D1)
        self.spatial_features = spatial_features      # Spatial branch (N, D2)
        self.metadata = metadata
        self.total_count = len(ids)
        self.id_to_idx = {img_id: i for i, img_id in enumerate(ids)}

        # Tracking statistics for each item
        # item_stats[i] = {"access_count": 0, "query_count": 0, "last_step": 0, "feedback_score": 0.0, "activity_score": 0.0, "tier": "WARM"}
        self.item_stats = [
            {
                "access_count": 0,
                "query_count": 0,
                "last_step": 0,
                "feedback_score": 0.0,
                "activity_score": 0.3,
                "tier": "COLD"
            }
            for _ in range(self.total_count)
        ]

        # Tier membership sets (indices)
        self.hot_indices: set = set()
        self.warm_indices: set = set()
        self.cold_indices: set = set(range(self.total_count))

        self.current_step = 0
        self.migration_history: List[Dict[str, Any]] = []
        self._update_tier_arrays()

    def _update_tier_arrays(self):
        self.hot_arr = np.array(sorted(self.hot_indices), dtype=np.int32) if self.hot_indices else np.array([], dtype=np.int32)
        self.warm_arr = np.array(sorted(self.warm_indices), dtype=np.int32) if self.warm_indices else np.array([], dtype=np.int32)
        self.cold_arr = np.array(sorted(self.cold_indices), dtype=np.int32) if self.cold_indices else np.array([], dtype=np.int32)

    def set_tiers(self, hot_set: set, warm_set: set, cold_set: set):
        self.hot_indices = hot_set
        self.warm_indices = warm_set
        self.cold_indices = cold_set
        self._update_tier_arrays()
        for idx in self.hot_indices:
            self.item_stats[idx]["tier"] = "HOT"
        for idx in self.warm_indices:
            self.item_stats[idx]["tier"] = "WARM"
        for idx in self.cold_indices:
            self.item_stats[idx]["tier"] = "COLD"

    def get_tier_counts(self) -> Dict[str, int]:
        return {
            "HOT": len(self.hot_indices),
            "WARM": len(self.warm_indices),
            "COLD": len(self.cold_indices),
            "TOTAL": self.total_count
        }

    def record_access(self, doc_idx: int, relevance: float = 0.0):
        self.current_step += 1
        st = self.item_stats[doc_idx]
        st["access_count"] += 1
        st["last_step"] = self.current_step
        if relevance != 0.0:
            st["feedback_score"] += relevance

    def record_query(self, doc_idx: int):
        self.current_step += 1
        st = self.item_stats[doc_idx]
        st["query_count"] += 1
        st["last_step"] = self.current_step

    def search_candidates(self,
                          cand_indices: Any,
                          query_spec: np.ndarray,
                          query_spat: np.ndarray,
                          alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes high-speed dynamic dual-branch similarity over candidate subset:
        S_combined = alpha * S_spectral + (1 - alpha) * S_spatial
        """
        if isinstance(cand_indices, np.ndarray):
            arr_idx = cand_indices
        elif isinstance(cand_indices, (list, tuple, set)):
            if not cand_indices:
                return np.array([]), np.array([]), np.array([])
            arr_idx = np.array(list(cand_indices), dtype=np.int32)
        else:
            return np.array([]), np.array([]), np.array([])

        if len(arr_idx) == 0:
            return np.array([]), np.array([]), np.array([])

        # Spectral branch dot product (BLAS vectorized)
        spec_cands = self.spectral_features[arr_idx]
        sim_spec = np.dot(spec_cands, query_spec)

        # Spatial branch dot product (BLAS vectorized)
        spat_cands = self.spatial_features[arr_idx]
        sim_spat = np.dot(spat_cands, query_spat)

        # Combined adaptive similarity
        sim_combined = alpha * sim_spec + (1.0 - alpha) * sim_spat

        return sim_combined, sim_spec, sim_spat
