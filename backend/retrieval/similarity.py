import numpy as np
from typing import Tuple

def compute_adaptive_similarity(query_spec: np.ndarray,
                                query_spat: np.ndarray,
                                doc_spec: np.ndarray,
                                doc_spat: np.ndarray,
                                alpha: float = 0.55) -> Tuple[float, float, float]:
    """
    Computes adaptive dual-branch similarity:
      S_spectral = dot(query_spec, doc_spec)
      S_spatial  = dot(query_spat, doc_spat)
      S_combined = alpha * S_spectral + (1 - alpha) * S_spatial
    Returns: (combined_score, spectral_score, spatial_score)
    """
    s_spec = float(np.dot(query_spec, doc_spec))
    s_spat = float(np.dot(query_spat, doc_spat))
    s_comb = alpha * s_spec + (1.0 - alpha) * s_spat
    return s_comb, s_spec, s_spat
