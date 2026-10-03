import os
import sys
import webbrowser
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import uvicorn
from backend.config import FEATURES_CACHE_FILE, BENCHMARK_CACHE_FILE
from scripts.build_index import build_or_load_dataset
from scripts.run_experiments import run_all_experiments

def main():
    print("=" * 75)
    print("  ADAPTIVE AI-DRIVEN MULTI-SPECTRAL IMAGE INDEXING FRAMEWORK")
    print("  FOR HIGH-SPEED REMOTE SENSING RETRIEVAL (EuroSAT Sentinel-2)")
    print("=" * 75)

    # 1. Check or build dataset features
    if not FEATURES_CACHE_FILE.exists():
        print("\n[*] Initializing dataset features (first-time run)...")
        build_or_load_dataset(samples_per_class=150)
    else:
        print("\n[+] Verified feature cache: data_cache/eurosat_features.pkl")

    # 2. Check or run benchmark evaluation
    if not BENCHMARK_CACHE_FILE.exists():
        print("[*] Generating empirical benchmark evaluation...")
        run_all_experiments()
    else:
        print("[+] Verified empirical benchmark results: data_cache/benchmark_results.json")

    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print(f"\n[+] Starting Web Dashboard at {url}")
    print("[+] Press Ctrl+C to stop server.\n")

    try:
        # Launch browser tab automatically
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("backend.api.app:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
