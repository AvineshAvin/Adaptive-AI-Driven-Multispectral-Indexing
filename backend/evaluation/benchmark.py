import time
import numpy as np
from typing import List, Dict, Any
from .metrics import compute_precision_at_k, compute_average_precision
from ..retrieval.engine import RetrievalEngine

class BenchmarkRunner:
    """
    Evaluates and compares the three indexing & retrieval frameworks:
      1. Baseline Exhaustive Flat Search
      2. Baseline Static Inverted Index
      3. Proposed Adaptive AI Multi-Tier Retrieval
    """
    def __init__(self, engine: RetrievalEngine):
        self.engine = engine

    def run_benchmark(self, query_indices: List[int], top_k: int = 10) -> Dict[str, Any]:
        """
        Executes benchmark over specified query sample indices.
        Returns aggregated metrics for each system mode.
        """
        modes = ["flat", "inverted", "adaptive"]
        results = {m: {
            "p1": [], "p5": [], "p10": [], "ap": [],
            "latencies_ms": [], "candidates": [], "reductions": []
        } for m in modes}

        total_queries = len(query_indices)
        
        for q_idx in query_indices:
            q_meta = self.engine.tiered_index.metadata[q_idx]
            q_class = q_meta["class_name"]
            q_feat_dict = {
                "spectral_features": self.engine.tiered_index.spectral_features[q_idx],
                "spatial_features": self.engine.tiered_index.spatial_features[q_idx],
                "fused_features": self.engine.tiered_index.features[q_idx],
                "alpha": 0.55
            }

            for mode in modes:
                # Do not count self-match if querying within repository
                res_items, stats = self.engine.retrieve(q_feat_dict, mode=mode, top_k=top_k + 1)
                filtered = [it for it in res_items if it.get("id") != q_meta["id"]][:top_k]

                p1 = compute_precision_at_k(filtered, q_class, 1)
                p5 = compute_precision_at_k(filtered, q_class, 5)
                p10 = compute_precision_at_k(filtered, q_class, 10)
                ap = compute_average_precision(filtered, q_class, 10)

                results[mode]["p1"].append(p1)
                results[mode]["p5"].append(p5)
                results[mode]["p10"].append(p10)
                results[mode]["ap"].append(ap)
                results[mode]["latencies_ms"].append(stats["search_time_ms"])
                results[mode]["candidates"].append(stats["candidates_evaluated"])
                results[mode]["reductions"].append(stats["search_space_reduction_pct"])

        # Compute summary averages
        summary = {}
        flat_avg_lat = np.mean(results["flat"]["latencies_ms"]) if results["flat"]["latencies_ms"] else 1.0

        for m in modes:
            avg_lat = float(np.mean(results[m]["latencies_ms"]))
            speedup = flat_avg_lat / max(0.001, avg_lat)
            summary[m] = {
                "precision_at_1": round(float(np.mean(results[m]["p1"])), 4),
                "precision_at_5": round(float(np.mean(results[m]["p5"])), 4),
                "precision_at_10": round(float(np.mean(results[m]["p10"])), 4),
                "mAP": round(float(np.mean(results[m]["ap"])), 4),
                "avg_search_time_ms": round(avg_lat, 3),
                "avg_candidates_evaluated": round(float(np.mean(results[m]["candidates"])), 1),
                "search_space_reduction_pct": round(float(np.mean(results[m]["reductions"])), 2),
                "speedup_factor": round(float(speedup), 2)
            }

        return {
            "num_queries_evaluated": total_queries,
            "metrics": summary,
            "tier_distribution": self.engine.tiered_index.get_tier_counts()
        }
