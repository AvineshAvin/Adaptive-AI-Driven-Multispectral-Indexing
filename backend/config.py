import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_MSI = PROJECT_ROOT / "archive" / "EuroSATallBands"
DATA_DIR_RGB = PROJECT_ROOT / "archive" / "EuroSAT"
CACHE_DIR = PROJECT_ROOT / "data_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_CACHE_FILE = CACHE_DIR / "eurosat_features.pkl"
BENCHMARK_CACHE_FILE = CACHE_DIR / "benchmark_results.json"

# EuroSAT Sentinel-2 13 Spectral Bands Information
# Band index -> (Band Name, Central Wavelength, Resolution, Description)
BAND_INFO = {
    0: ("B01", "443 nm", "60m", "Coastal Aerosol"),
    1: ("B02", "490 nm", "10m", "Blue"),
    2: ("B03", "560 nm", "10m", "Green"),
    3: ("B04", "665 nm", "10m", "Red"),
    4: ("B05", "705 nm", "20m", "Vegetation Red Edge 1"),
    5: ("B06", "740 nm", "20m", "Vegetation Red Edge 2"),
    6: ("B07", "783 nm", "20m", "Vegetation Red Edge 3"),
    7: ("B08", "842 nm", "10m", "NIR (Near Infrared)"),
    8: ("B8A", "865 nm", "20m", "Narrow NIR"),
    9: ("B09", "940 nm", "60m", "Water Vapour"),
    10: ("B10", "1375 nm", "60m", "SWIR - Cirrus"),
    11: ("B11", "1610 nm", "20m", "SWIR 1"),
    12: ("B12", "2190 nm", "20m", "SWIR 2"),
}

# EuroSAT 10 Land Cover Classes
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Activity Scoring Weights
# A_i(t) = w_f * F_acc + w_q * F_query + w_r * exp(-lambda * dt) + w_fb * R_feedback
WEIGHT_ACCESS_FREQ = 0.35
WEIGHT_QUERY_FREQ = 0.25
WEIGHT_RECENCY = 0.20
WEIGHT_FEEDBACK = 0.20
RECENCY_DECAY_LAMBDA = 0.05

# Tier Thresholds (Activity Score ranges in [0.0, 1.0])
HOT_TIER_THRESHOLD = 0.65     # >= 0.65 -> HOT
WARM_TIER_THRESHOLD = 0.25    # 0.25 <= score < 0.65 -> WARM; < 0.25 -> COLD

# Dynamic Search Parameters
HOT_CONFIDENCE_THRESHOLD = 0.82   # If top candidate confidence >= 0.82 in HOT, stop search early!
WARM_CONFIDENCE_THRESHOLD = 0.70  # Threshold for warm tier retrieval stopping
DEFAULT_TOP_K = 10
INV_CLUSTERS_K = 16               # Number of Voronoi partitions for Inverted Index
INV_PROBE_CLUSTERS = 3            # Number of clusters probed during inverted retrieval

# Default alpha (Spectral vs Spatial weight)
DEFAULT_ALPHA = 0.55
