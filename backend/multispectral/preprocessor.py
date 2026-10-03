import numpy as np

def preprocess_multispectral(img: np.ndarray) -> np.ndarray:
    """
    Validates and preprocesses 13-band Sentinel-2 image.
    - Handles NaN/inf values
    - Clips extreme values
    - Normalizes reflectance to [0.0, 1.0] range (Sentinel-2 scale factor 10,000)
    """
    fimg = img.astype(np.float32)
    
    # Handle NaN or Inf if any
    fimg = np.nan_to_num(fimg, nan=0.0, posinf=10000.0, neginf=0.0)
    
    # Sentinel-2 physical reflectance normalization
    # Normal surface reflectance rarely exceeds 1.0 (10000)
    fimg = np.clip(fimg, 0.0, 10000.0) / 10000.0
    
    return fimg
