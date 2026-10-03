import numpy as np
from scipy import ndimage
from typing import Tuple, Dict, Any
from .preprocessor import preprocess_multispectral
from .spectral_indices import compute_spectral_indices, summarize_index

class DualBranchFeatureExtractor:
    """
    Extracts dual-branch Spectral and Spatial feature representations
    from 13-band Sentinel-2 multispectral images.
    """
    def __init__(self):
        pass

    def extract_spectral_features(self, img_norm: np.ndarray) -> np.ndarray:
        """
        Branch 1: Spectral Feature Extraction (13 bands + 6 spectral indices + band ratios).
        """
        features = []
        H, W, num_bands = img_norm.shape

        # 1. Band-wise statistical moments (mean, std, p10, p50, p90)
        for b in range(num_bands):
            band = img_norm[:, :, b].ravel()
            p10, p50, p90 = np.percentile(band, [10, 50, 90])
            mean_val = np.mean(band)
            std_val = np.std(band)
            features.extend([mean_val, std_val, p10, p50, p90])

        # 2. Spectral Indices statistics
        indices = compute_spectral_indices(img_norm * 10000.0)
        for key in ["NDVI", "NDWI", "NDBI", "SAVI", "NDRE", "BSI"]:
            summary = summarize_index(indices[key])
            features.extend(summary.tolist())

        # 3. Inter-band ratios (diagnostic remote sensing ratios)
        eps = 1e-6
        b_blue = img_norm[:, :, 1].mean()
        b_green = img_norm[:, :, 2].mean()
        b_red = img_norm[:, :, 3].mean()
        b_nir = img_norm[:, :, 7].mean()
        b_swir1 = img_norm[:, :, 11].mean()

        features.append(b_nir / (b_red + eps))
        features.append(b_green / (b_red + eps))
        features.append(b_swir1 / (b_nir + eps))
        features.append(b_nir / (b_blue + eps))

        spec_vec = np.array(features, dtype=np.float32)
        # L2-normalize
        norm = np.linalg.norm(spec_vec)
        if norm > 1e-7:
            spec_vec /= norm
        return spec_vec

    def extract_spatial_features(self, img_norm: np.ndarray) -> np.ndarray:
        """
        Branch 2: Spatial Feature Extraction.
        Captures texture, spatial gradients, and quadrant geometric layouts
        across structural bands (Red B04, NIR B08, Green B03).
        """
        features = []
        key_bands = [2, 3, 7]  # Green, Red, NIR

        for b_idx in key_bands:
            band = img_norm[:, :, b_idx]

            # 1. Spatial Gradients (Sobel filters)
            sobel_x = ndimage.sobel(band, axis=0)
            sobel_y = ndimage.sobel(band, axis=1)
            grad_mag = np.hypot(sobel_x, sobel_y)

            features.append(float(np.mean(grad_mag)))
            features.append(float(np.std(grad_mag)))
            features.append(float(np.max(grad_mag)))

            # 2. Spatial Laplacian (Edge sharpness / texture variance)
            laplacian = ndimage.laplace(band)
            features.append(float(np.var(laplacian)))

            # 3. Quadrant Spatial Layout (2x2 grid pooling)
            h_mid, w_mid = band.shape[0] // 2, band.shape[1] // 2
            q1 = np.mean(band[:h_mid, :w_mid])
            q2 = np.mean(band[:h_mid, w_mid:])
            q3 = np.mean(band[h_mid:, :w_mid])
            q4 = np.mean(band[h_mid:, w_mid:])
            features.extend([float(q1), float(q2), float(q3), float(q4)])

        spat_vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(spat_vec)
        if norm > 1e-7:
            spat_vec /= norm
        return spat_vec

    def estimate_adaptive_alpha(self, img_norm: np.ndarray) -> float:
        """
        Estimates adaptive alpha (spectral vs spatial weight) based on image characteristics.
        Natural land cover (forest, water, crop) has high spectral distinction -> alpha ~ 0.65 - 0.80
        Man-made structures (roads, buildings) has sharp edge gradients -> alpha ~ 0.35 - 0.45
        """
        # Calculate gradient magnitude on NIR band
        nir = img_norm[:, :, 7]
        grad_mag = np.hypot(ndimage.sobel(nir, axis=0), ndimage.sobel(nir, axis=1))
        edge_energy = float(np.mean(grad_mag))

        # Calculate NDVI variance
        indices = compute_spectral_indices(img_norm * 10000.0)
        ndvi_std = float(np.std(indices["NDVI"]))

        # Higher edge energy -> higher spatial emphasis (lower alpha)
        # Higher spectral variation -> higher spectral emphasis (higher alpha)
        balance = ndvi_std * 2.5 - edge_energy * 3.0
        alpha = 0.55 + 0.25 * float(np.tanh(balance))
        return float(np.clip(alpha, 0.25, 0.85))

    def extract_all(self, raw_geotiff: np.ndarray) -> Dict[str, Any]:
        """
        Full feature extraction pipeline:
        Returns spectral_vector, spatial_vector, fused_vector, and adaptive_alpha.
        """
        norm_img = preprocess_multispectral(raw_geotiff)
        spec_vec = self.extract_spectral_features(norm_img)
        spat_vec = self.extract_spatial_features(norm_img)
        alpha = self.estimate_adaptive_alpha(norm_img)

        # Fused vector combining spectral and spatial representation
        fused = np.concatenate([spec_vec, spat_vec])
        norm = np.linalg.norm(fused)
        if norm > 1e-7:
            fused /= norm

        # Compute average 13-band signature for visualization
        band_signature = [float(np.mean(norm_img[:, :, b])) for b in range(13)]

        return {
            "spectral_features": spec_vec,
            "spatial_features": spat_vec,
            "fused_features": fused,
            "alpha": alpha,
            "band_signature": band_signature
        }
