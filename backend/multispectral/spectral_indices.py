import numpy as np
from typing import Dict

def compute_spectral_indices(img: np.ndarray, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    """
    Computes key remote sensing spectral indices from Sentinel-2 13-band image (H, W, 13).
    Bands:
      B02: Blue  (index 1)
      B03: Green (index 2)
      B04: Red   (index 3)
      B05: RE1   (index 4)
      B08: NIR   (index 7)
      B11: SWIR1 (index 11)
    """
    fimg = img.astype(np.float32)
    b_blue = fimg[:, :, 1]
    b_green = fimg[:, :, 2]
    b_red = fimg[:, :, 3]
    b_re1 = fimg[:, :, 4]
    b_nir = fimg[:, :, 7]
    b_swir1 = fimg[:, :, 11]

    # NDVI: Normalized Difference Vegetation Index
    ndvi = (b_nir - b_red) / (b_nir + b_red + eps)

    # NDWI: Normalized Difference Water Index (McFeeters)
    ndwi = (b_green - b_nir) / (b_green + b_nir + eps)

    # NDBI: Normalized Difference Built-up Index
    ndbi = (b_swir1 - b_nir) / (b_swir1 + b_nir + eps)

    # SAVI: Soil Adjusted Vegetation Index (L=0.5)
    savi = ((b_nir - b_red) / (b_nir + b_red + 0.5 * 10000.0)) * 1.5

    # NDRE: Normalized Difference Red Edge
    ndre = (b_nir - b_re1) / (b_nir + b_re1 + eps)

    # BSI: Bare Soil Index
    bsi_num = (b_swir1 + b_red) - (b_nir + b_blue)
    bsi_den = (b_swir1 + b_red) + (b_nir + b_blue) + eps
    bsi = bsi_num / bsi_den

    return {
        "NDVI": ndvi,
        "NDWI": ndwi,
        "NDBI": ndbi,
        "SAVI": savi,
        "NDRE": ndre,
        "BSI": bsi
    }

def summarize_index(index_map: np.ndarray) -> np.ndarray:
    """
    Summarize a 2D spatial index map into compact statistical moments:
    [mean, std, p10, p50, p90]
    """
    flat = index_map.ravel()
    p10, p50, p90 = np.percentile(flat, [10, 50, 90])
    return np.array([float(np.mean(flat)), float(np.std(flat)), float(p10), float(p50), float(p90)], dtype=np.float32)
