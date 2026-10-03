import time
from typing import Dict, Any, List, Tuple
from .adaptive_tiers import AdaptiveTieredIndex
from .activity_scorer import ActivityScorer
from ..config import HOT_TIER_THRESHOLD, WARM_TIER_THRESHOLD

class DynamicTierManager:
    """
    Self-learning Tier Migration Engine.
    Periodically or reactively computes activity scores across the repository
    and dynamically promotes or demotes images between HOT, WARM, and COLD tiers.
    """
    def __init__(self,
                 index: AdaptiveTieredIndex,
                 scorer: ActivityScorer = None,
                 hot_threshold: float = HOT_TIER_THRESHOLD,
                 warm_threshold: float = WARM_TIER_THRESHOLD):
        self.index = index
        self.scorer = scorer if scorer is not None else ActivityScorer()
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
        self.migration_counter = 0

    def rebalance_tiers(self) -> Dict[str, Any]:
        """
        Executes tier reorganization based on current activity scores.
        Returns a summary report of migrations (promotions and demotions).
        """
        start_time = time.perf_counter()
        self.migration_counter += 1

        promotions = 0
        demotions = 0
        stayed = 0

        new_hot = set()
        new_warm = set()
        new_cold = set()

        migration_events = []

        for idx, stats in enumerate(self.index.item_stats):
            old_tier = stats["tier"]
            score = self.scorer.compute_score(stats, self.index.current_step)
            stats["activity_score"] = score

            if score >= self.hot_threshold:
                new_tier = "HOT"
                new_hot.add(idx)
            elif score >= self.warm_threshold:
                new_tier = "WARM"
                new_warm.add(idx)
            else:
                new_tier = "COLD"
                new_cold.add(idx)

            if new_tier != old_tier:
                # Migration occurred
                if (old_tier == "COLD" and new_tier in ["WARM", "HOT"]) or (old_tier == "WARM" and new_tier == "HOT"):
                    promotions += 1
                    event_type = "PROMOTION"
                else:
                    demotions += 1
                    event_type = "DEMOTION"
                
                if len(migration_events) < 50:
                    migration_events.append({
                        "id": self.index.ids[idx],
                        "class": self.index.metadata[idx]["class_name"],
                        "from_tier": old_tier,
                        "to_tier": new_tier,
                        "score": round(score, 4),
                        "type": event_type
                    })
            else:
                stayed += 1

        # Apply new tiers
        self.index.set_tiers(new_hot, new_warm, new_cold)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        summary = {
            "migration_id": self.migration_counter,
            "step": self.index.current_step,
            "duration_ms": elapsed_ms,
            "promotions": promotions,
            "demotions": demotions,
            "stayed": stayed,
            "tier_counts": self.index.get_tier_counts(),
            "sample_events": migration_events[:15]
        }

        self.index.migration_history.append(summary)
        return summary
