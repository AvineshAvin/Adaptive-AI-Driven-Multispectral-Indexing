import random
import numpy as np
from typing import List, Dict, Any
from ..retrieval.engine import RetrievalEngine

class QueryActivitySimulator:
    """
    Simulates remote sensing query workloads to analyze how the adaptive index
    reorganizes its HOT-WARM-COLD tiers according to access behavior.
    """
    def __init__(self, engine: RetrievalEngine):
        self.engine = engine

    def run_simulation(self,
                       scenario: str = "zipfian",
                       num_queries: int = 50,
                       rebalance_interval: int = 10) -> Dict[str, Any]:
        """
        Executes query simulation under different access profiles.
        Scenarios:
          - 'disaster_monitoring': River, SeaLake, Forest focus (flood / wildfire)
          - 'urban_expansion': Highway, Industrial, Residential focus
          - 'agriculture_survey': AnnualCrop, PermanentCrop, Pasture focus
          - 'zipfian': Power-law access skew (Pareto 80/20)
        """
        all_metadata = self.engine.tiered_index.metadata
        total_items = len(all_metadata)

        # Categorize indices by class
        class_to_indices = {}
        for idx, meta in enumerate(all_metadata):
            c = meta["class_name"]
            class_to_indices.setdefault(c, []).append(idx)

        # Build sampling probability distribution
        weights = np.ones(total_items, dtype=np.float32)

        if scenario == "disaster_monitoring":
            target_classes = ["River", "SeaLake", "Forest"]
            for c in target_classes:
                for idx in class_to_indices.get(c, []):
                    weights[idx] = 12.0
        elif scenario == "urban_expansion":
            target_classes = ["Highway", "Industrial", "Residential"]
            for c in target_classes:
                for idx in class_to_indices.get(c, []):
                    weights[idx] = 12.0
        elif scenario == "agriculture_survey":
            target_classes = ["AnnualCrop", "PermanentCrop", "Pasture"]
            for c in target_classes:
                for idx in class_to_indices.get(c, []):
                    weights[idx] = 12.0
        else:
            # Zipfian power-law distribution
            ranks = np.arange(1, total_items + 1)
            weights = 1.0 / (ranks ** 0.8)
            np.random.shuffle(weights)

        probs = weights / np.sum(weights)

        # Track history
        step_logs = []
        tier_snapshots = [self.engine.tiered_index.get_tier_counts()]

        for step in range(1, num_queries + 1):
            # Sample query image
            q_idx = int(np.random.choice(total_items, p=probs))
            q_meta = all_metadata[q_idx]

            # Mark query interaction
            self.engine.tiered_index.record_query(q_idx)

            # Perform adaptive retrieval
            q_feat = {
                "spectral_features": self.engine.tiered_index.spectral_features[q_idx],
                "spatial_features": self.engine.tiered_index.spatial_features[q_idx],
                "fused_features": self.engine.tiered_index.features[q_idx],
                "alpha": 0.55
            }
            results, stats = self.engine.retrieve(q_feat, mode="adaptive", top_k=5)

            # Periodic self-learning tier reorganization
            if step % rebalance_interval == 0:
                rebal_info = self.engine.tier_manager.rebalance_tiers()
                tier_snapshots.append(rebal_info["tier_counts"])
                step_logs.append({
                    "step": step,
                    "query_id": q_meta["id"],
                    "query_class": q_meta["class_name"],
                    "promotions": rebal_info["promotions"],
                    "demotions": rebal_info["demotions"],
                    "tier_counts": rebal_info["tier_counts"],
                    "sample_promoted": [e["id"] for e in rebal_info.get("sample_events", []) if e["type"] == "PROMOTION"][:3]
                })

        final_tier_distribution = self.engine.tiered_index.get_tier_counts()

        # Class representation in HOT tier
        hot_class_distribution = {}
        for h_idx in self.engine.tiered_index.hot_indices:
            cls = all_metadata[h_idx]["class_name"]
            hot_class_distribution[cls] = hot_class_distribution.get(cls, 0) + 1

        return {
            "scenario": scenario,
            "queries_executed": num_queries,
            "rebalance_events": len(step_logs),
            "initial_tiers": tier_snapshots[0],
            "final_tiers": final_tier_distribution,
            "tier_progression": tier_snapshots,
            "hot_class_distribution": hot_class_distribution,
            "reorganization_steps": step_logs
        }
