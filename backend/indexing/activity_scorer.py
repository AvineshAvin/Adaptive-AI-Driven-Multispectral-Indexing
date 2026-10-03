import math
import time
from typing import Dict, Any
from ..config import (
    WEIGHT_ACCESS_FREQ,
    WEIGHT_QUERY_FREQ,
    WEIGHT_RECENCY,
    WEIGHT_FEEDBACK,
    RECENCY_DECAY_LAMBDA,
)

class ActivityScorer:
    """
    Computes adaptive dynamic activity scores for each image item.
    Factors:
      1. Access frequency (times retrieved in top-K)
      2. Query frequency (times used as query image)
      3. Recency decay: exp(-lambda * delta_time)
      4. Explicit relevance feedback (+1 for relevant, -0.5 for irrelevant)
    """
    def __init__(self,
                 w_access: float = WEIGHT_ACCESS_FREQ,
                 w_query: float = WEIGHT_QUERY_FREQ,
                 w_recency: float = WEIGHT_RECENCY,
                 w_feedback: float = WEIGHT_FEEDBACK,
                 decay_lambda: float = RECENCY_DECAY_LAMBDA):
        self.w_access = w_access
        self.w_query = w_query
        self.w_recency = w_recency
        self.w_feedback = w_feedback
        self.decay_lambda = decay_lambda

    def compute_score(self, item_stats: Dict[str, Any], current_step: int) -> float:
        """
        Calculates normalized activity score A_i in [0.0, 1.0].
        """
        access_count = item_stats.get("access_count", 0)
        query_count = item_stats.get("query_count", 0)
        last_step = item_stats.get("last_step", 0)
        feedback_val = item_stats.get("feedback_score", 0.0)

        # Non-linear saturating frequency terms: 1 - exp(-0.2 * count)
        norm_access = 1.0 - math.exp(-0.25 * access_count)
        norm_query = 1.0 - math.exp(-0.4 * query_count)

        # Recency term: exponential decay since last interaction
        dt = max(0, current_step - last_step)
        recency = math.exp(-self.decay_lambda * dt)

        # Feedback term: sigmoid-scaled feedback score
        # e.g. feedback_score ranges from -5 to +10
        norm_feedback = 1.0 / (1.0 + math.exp(-0.8 * feedback_val))

        raw_score = (
            self.w_access * norm_access +
            self.w_query * norm_query +
            self.w_recency * recency +
            self.w_feedback * norm_feedback
        )

        return float(min(1.0, max(0.0, raw_score)))
