import io
import base64
import numpy as np
from PIL import Image
import tifffile
from pathlib import Path
from typing import Tuple, Optional, Union

def load_geotiff(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load a multispectral GeoTIFF image.
    EuroSAT GeoTIFFs have shape (64, 64, 13) and dtype uint16.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Multispectral file not found: {file_path}")
    
    img = tifffile.imread(str(path_obj))
    if img.ndim != 3 or img.shape[2] != 13:
        raise ValueError(f"Expected image with 13 bands (H, W, 13), got shape {img.shape}")
    
    return img

def percentile_stretch(channel: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Robust 2% - 98% percentile linear contrast stretch for satellite bands.
    """
    low, high = np.percentile(channel, (p_low, p_high))
    if high <= low:
        high = low + 1e-5
    stretched = (channel - low) / (high - low)
    stretched = np.clip(stretched, 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)

def generate_rgb_composite(img: np.ndarray) -> np.ndarray:
    """
    True Color RGB: Band 3 (Red, B04), Band 2 (Green, B03), Band 1 (Blue, B02).
    """
    r = percentile_stretch(img[:, :, 3].astype(np.float32))
    g = percentile_stretch(img[:, :, 2].astype(np.float32))
    b = percentile_stretch(img[:, :, 1].astype(np.float32))
    return np.stack([r, g, b], axis=2)

def generate_false_color_cir(img: np.ndarray) -> np.ndarray:
    """
    Color Infrared (CIR): NIR (B08, index 7), Red (B04, index 3), Green (B03, index 2).
    Vegetation reflects intensely in NIR and appears bright red.
    """
    r = percentile_stretch(img[:, :, 7].astype(np.float32))
    g = percentile_stretch(img[:, :, 3].astype(np.float32))
    b = percentile_stretch(img[:, :, 2].astype(np.float32))
    return np.stack([r, g, b], axis=2)

def generate_swir_composite(img: np.ndarray) -> np.ndarray:
    """
    SWIR composite: SWIR1 (B11, index 11), NIR (B08, index 7), Red (B04, index 3).
    """
    r = percentile_stretch(img[:, :, 11].astype(np.float32))
    g = percentile_stretch(img[:, :, 7].astype(np.float32))
    b = percentile_stretch(img[:, :, 3].astype(np.float32))
    return np.stack([r, g, b], axis=2)

def image_to_base64(img_array: np.ndarray) -> str:
    """
    Convert a uint8 RGB numpy array (H, W, 3) to base64 JPEG data URL.
    """
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    # Resize to 128x128 for crisp display in web dashboard
    pil_img = pil_img.resize((128, 128), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=92)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"
