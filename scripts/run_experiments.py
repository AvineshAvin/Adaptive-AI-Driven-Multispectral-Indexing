import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Ensure project root in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.config import BENCHMARK_CACHE_FILE
from scripts.build_index import build_or_load_dataset
from backend.indexing.exhaustive_index import ExhaustiveIndex
from backend.indexing.inverted_index import StaticInvertedIndex
from backend.indexing.adaptive_tiers import AdaptiveTieredIndex
from backend.indexing.tier_manager import DynamicTierManager
from backend.retrieval.engine import RetrievalEngine
from backend.evaluation.benchmark import BenchmarkRunner
from backend.evaluation.ablation import AblationStudyRunner
from backend.evaluation.simulation import QueryActivitySimulator

def run_all_experiments():
    print("=" * 70)
    print("Starting Comprehensive Empirical Evaluation")
    print("Adaptive AI-Driven Multi-Spectral Image Indexing Framework")
    print("=" * 70)

    # 1. Load Dataset
    data = build_or_load_dataset()
    ids = data["ids"]
    fused_feats = data["fused_features"]
    spec_feats = data["spectral_features"]
    spat_feats = data["spatial_features"]
    metadata = data["metadata"]
    total_n = len(ids)
    print(f"Dataset ready: {total_n} images across 10 classes.")

    # 2. Initialize Indexes
    print("\n[INIT] Initializing Flat Exhaustive Baseline...")
    flat_idx = ExhaustiveIndex(ids, fused_feats, metadata)

    print("[INIT] Initializing Static Inverted Index (IVF)...")
    inv_idx = StaticInvertedIndex(ids, fused_feats, metadata, n_clusters=16, n_probe=3)

    print("[INIT] Initializing Proposed Adaptive Multi-Tier Index (HOT-WARM-COLD)...")
    tiered_idx = AdaptiveTieredIndex(ids, fused_feats, spec_feats, spat_feats, metadata)
    tier_mgr = DynamicTierManager(tiered_idx)

    # Initialize warm and hot tiers with realistic initial distribution
    # E.g. top 10% frequently monitored items in HOT, 35% in WARM, 55% in COLD
    indices = np.arange(total_n)
    np.random.seed(42)
    np.random.shuffle(indices)
    n_hot = int(total_n * 0.12)
    n_warm = int(total_n * 0.38)
    hot_set = set(indices[:n_hot])
    warm_set = set(indices[n_hot:n_hot + n_warm])
    cold_set = set(indices[n_hot + n_warm:])
    tiered_idx.set_tiers(hot_set, warm_set, cold_set)

    # Set some initial access counts for hot items
    for idx in hot_set:
        tiered_idx.item_stats[idx]["access_count"] = np.random.randint(5, 25)
        tiered_idx.item_stats[idx]["activity_score"] = float(np.random.uniform(0.70, 0.95))
    for idx in warm_set:
        tiered_idx.item_stats[idx]["access_count"] = np.random.randint(1, 5)
        tiered_idx.item_stats[idx]["activity_score"] = float(np.random.uniform(0.30, 0.60))

    engine = RetrievalEngine(tiered_idx, tier_mgr, inv_idx, flat_idx)

    # 3. Select stratified evaluation query set (10 queries per class = 100 queries)
    class_indices = {}
    for i, m in enumerate(metadata):
        class_indices.setdefault(m["class_name"], []).append(i)

    eval_query_indices = []
    for c, idxs in class_indices.items():
        eval_query_indices.extend(idxs[:10])

    print(f"\n[BENCHMARK] Running evaluation on {len(eval_query_indices)} diverse test queries...")
    benchmarker = BenchmarkRunner(engine)
    bench_results = benchmarker.run_benchmark(eval_query_indices, top_k=10)

    print("\n--- BENCHMARK RESULTS SUMMARY ---")
    for mode, m in bench_results["metrics"].items():
        print(f"Mode: {mode.upper()}")
        print(f"  Precision@1 : {m['precision_at_1']:.4f}")
        print(f"  Precision@5 : {m['precision_at_5']:.4f}")
        print(f"  Precision@10: {m['precision_at_10']:.4f}")
        print(f"  mAP         : {m['mAP']:.4f}")
        print(f"  Search Time : {m['avg_search_time_ms']:.3f} ms")
        print(f"  Candidates  : {m['avg_candidates_evaluated']} items")
        print(f"  Reduction % : {m['search_space_reduction_pct']:.2f}%")
        print(f"  Speedup     : {m['speedup_factor']:.2f}x")

    # 4. Ablation Study
    print("\n[ABLATION] Running ablation experiments...")
    ablation_runner = AblationStudyRunner(engine)
    ablation_results = ablation_runner.run_ablation(eval_query_indices[:30], top_k=10)

    print("\n--- ABLATION RESULTS ---")
    for ab in ablation_results["ablations"]:
        print(f"{ab['name']}:")
        print(f"  mAP={ab['mAP']:.4f} | Latency={ab['avg_search_time_ms']:.3f}ms | Space Reduction={ab['search_space_reduction_pct']:.2f}%")

    # 5. Workload Simulation
    print("\n[SIMULATION] Simulating 100 queries under Zipfian access profile...")
    simulator = QueryActivitySimulator(engine)
    sim_results = simulator.run_simulation(scenario="disaster_monitoring", num_queries=80, rebalance_interval=15)

    print(f"Simulation completed with {sim_results['rebalance_events']} dynamic rebalance events.")
    print(f"Initial tiers: {sim_results['initial_tiers']}")
    print(f"Final tiers  : {sim_results['final_tiers']}")

    # 6. Save all results to disk
    full_export = {
        "benchmark": bench_results,
        "ablation": ablation_results,
        "simulation": sim_results,
        "timestamp": time.time()
    }

    with open(BENCHMARK_CACHE_FILE, "w") as f:
        json.dump(full_export, f, indent=2)

    print(f"\n[DONE] Benchmark and experiment results exported to {BENCHMARK_CACHE_FILE}")

if __name__ == "__main__":
    run_all_experiments()
