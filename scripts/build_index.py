import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import glob
import pickle
import time
import argparse
import numpy as np
import tifffile
from PIL import Image

from backend.config import DATA_DIR_MSI, DATA_DIR_RGB, CACHE_DIR, FEATURES_CACHE_FILE, CLASSES
from backend.multispectral.feature_extractor import DualBranchFeatureExtractor
from backend.multispectral.loader import generate_rgb_composite, percentile_stretch

def build_or_load_dataset(samples_per_class: int = 200, force_rebuild: bool = False):
    """
    Builds or loads cached dataset features.
    If cache exists and force_rebuild is False, loads instantly.
    Otherwise, extracts dual-branch features from EuroSAT GeoTIFFs.
    """
    if FEATURES_CACHE_FILE.exists() and not force_rebuild:
        print(f"[CACHE] Loading cached features from {FEATURES_CACHE_FILE}...")
        start_t = time.perf_counter()
        with open(FEATURES_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"[CACHE] Loaded {len(data['ids'])} images in {(time.perf_counter() - start_t):.3f}s")
        return data

    print(f"[BUILD] Extracting features from {DATA_DIR_MSI} ({samples_per_class} per class)...")
    extractor = DualBranchFeatureExtractor()
    preview_dir = CACHE_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    ids = []
    spectral_feats = []
    spatial_feats = []
    fused_feats = []
    alphas = []
    band_signatures = []
    metadata = []

    total_start = time.perf_counter()

    for class_name in CLASSES:
        class_msi_dir = DATA_DIR_MSI / class_name
        class_rgb_dir = DATA_DIR_RGB / class_name
        
        tif_files = sorted(glob.glob(str(class_msi_dir / "*.tif")))
        if samples_per_class > 0:
            tif_files = tif_files[:samples_per_class]

        print(f"  -> Processing class '{class_name}': {len(tif_files)} images...")

        for fpath_str in tif_files:
            fpath = Path(fpath_str)
            img_id = fpath.stem

            try:
                raw_tif = tifffile.imread(str(fpath))
                if raw_tif.ndim != 3 or raw_tif.shape[2] != 13:
                    continue

                # Generate and save fast JPEG preview for web UI
                preview_path = preview_dir / f"{img_id}.jpg"
                if not preview_path.exists():
                    rgb_comp = generate_rgb_composite(raw_tif)
                    pil_img = Image.fromarray(rgb_comp)
                    pil_img = pil_img.resize((96, 96), Image.Resampling.NEAREST)
                    pil_img.save(preview_path, format="JPEG", quality=85)

                # Feature extraction
                extracted = extractor.extract_all(raw_tif)

                ids.append(img_id)
                spectral_feats.append(extracted["spectral_features"])
                spatial_feats.append(extracted["spatial_features"])
                fused_feats.append(extracted["fused_features"])
                alphas.append(extracted["alpha"])
                band_signatures.append(extracted["band_signature"])

                metadata.append({
                    "id": img_id,
                    "class_name": class_name,
                    "path": str(fpath),
                    "preview_path": str(preview_path),
                    "alpha": extracted["alpha"]
                })
            except Exception as e:
                print(f"Warning: Failed to process {fpath}: {e}")
                continue

    dataset_dict = {
        "ids": ids,
        "spectral_features": np.array(spectral_feats, dtype=np.float32),
        "spatial_features": np.array(spatial_feats, dtype=np.float32),
        "fused_features": np.array(fused_feats, dtype=np.float32),
        "alphas": np.array(alphas, dtype=np.float32),
        "band_signatures": np.array(band_signatures, dtype=np.float32),
        "metadata": metadata
    }

    print(f"[BUILD] Feature extraction completed in {(time.perf_counter() - total_start):.2f}s")
    print(f"[BUILD] Caching features to {FEATURES_CACHE_FILE}...")
    with open(FEATURES_CACHE_FILE, "wb") as f:
        pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200, help="Samples per class (default 200 = 2000 total)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild cache")
    args = parser.parse_args()

    build_or_load_dataset(samples_per_class=args.samples, force_rebuild=args.rebuild)
