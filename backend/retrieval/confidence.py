import numpy as np
from typing import List, Tuple, Dict, Any

def calculate_retrieval_confidence(
    scores: List[float],
    beta: float = 0.4,
    policy: str = "recall_aware",
    top_k: int = 10,
    calib_floor: float = 0.93
) -> Tuple[float, List[float]]:
    """
    Computes calibrated confidence metrics for remote sensing feature representations:
    Cosine similarities naturally sit between [0.85, 0.99] due to high-dimensional embedding geometry.
    To avoid premature early-stopping on unrelated background items:
      - Calibrated score: c_i = clip((s_i - calib_floor) / (1.0 - calib_floor), 0.0, 1.0)
      - Margin ratio: Margin = (s_1 - s_2) / (s_1 + 1e-6)
      - Candidate confidence: Conf(s_i) = c_i * (1.0 + beta * Margin)
    
    Early-Stopping Policies:
      - 'aggressive': agg_conf = Conf(s_1) (prioritizes 88% search space pruning)
      - 'recall_aware': agg_conf = 0.4 * Conf(s_1) + 0.6 * Mean(Conf(s_1..s_k))
        (evaluates candidate density to ensure Top-K quality before terminating)
    """
    if not scores:
        return 0.0, []

    s1 = scores[0]
    s2 = scores[1] if len(scores) > 1 else (scores[0] * 0.95)
    margin = max(0.0, (s1 - s2) / (abs(s1) + 1e-6))

    confidences = []
    for s in scores:
        # Calibrate similarity above background noise floor
        c_calib = float(np.clip((s - calib_floor) / (1.0 - calib_floor + 1e-6), 0.0, 1.0))
        # Boost confidence when margin separates top candidate clearly
        c = c_calib * (1.0 + beta * margin)
        confidences.append(float(np.clip(c, 0.05, 1.0)))

    if policy == "aggressive":
        agg_confidence = confidences[0] if confidences else 0.0
    else:
        # Recall-aware: requires both top-1 confidence and sufficient candidate support in top-k
        eval_k = min(len(confidences), max(1, top_k))
        topk_mean = float(np.mean(confidences[:eval_k]))
        top1_conf = confidences[0] if confidences else 0.0
        agg_confidence = float(np.clip(0.4 * top1_conf + 0.6 * topk_mean, 0.0, 1.0))

    return agg_confidence, confidences
