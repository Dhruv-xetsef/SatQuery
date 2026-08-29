import os
import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def load_and_inspect_image(file_path: str):
    """
    Loads an optical or SAR remote sensing image (GeoTIFF/TIFF, PNG, JPEG),
    extracts rich geospatial & sensor metadata (format, dimensions, CRS, GSD, bands, modality),
    and returns a normalized 3-channel RGB numpy array [H, W, 3] + metadata dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    file_size_kb = round(os.path.getsize(file_path) / 1024, 2)

    metadata = {
        "filename": filename,
        "filepath": file_path,
        "file_size_kb": file_size_kb,
        "format": ext,
        "is_geotiff": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "crs": "N/A",
        "gsd_m": "N/A",
        "bounds": None,
        "modality_guess": "Unknown",
        "sensor_type": "Optical"
    }

    if HAS_RASTERIO and ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(file_path) as src:
                metadata["is_geotiff"] = True
                metadata["width"] = src.width
                metadata["height"] = src.height
                metadata["channels"] = src.count
                metadata["crs"] = str(src.crs) if src.crs else "EPSG:4326 (Local Pixel Grid)"
                
                # Estimate Ground Sample Distance (GSD in meters)
                if src.res:
                    res_val = round(abs(src.res[0]), 3)
                    metadata["gsd_m"] = f"{res_val}m" if res_val < 100 else "10.0m (Sentinel-2 Nominal)"
                else:
                    metadata["gsd_m"] = "10.0m (Sentinel-2 Nominal)"

                if src.bounds:
                    metadata["bounds"] = {
                        "left": round(src.bounds.left, 5),
                        "bottom": round(src.bounds.bottom, 5),
                        "right": round(src.bounds.right, 5),
                        "top": round(src.bounds.top, 5)
                    }

                data = src.read()  # Shape: (count, H, W)
                
                # Sensor & Modality classification
                is_sar_filename = any(k in filename.lower() for k in ["sar", "radar", "risat", "sentinel1", "s1"])
                if src.count in [1, 2] or is_sar_filename:
                    metadata["sensor_type"] = "SAR"
                    metadata["modality_guess"] = "SAR Microwave Radar (C-band / Double-bounce)"
                    # Render SAR grayscale / 3-channel
                    vv = data[0].astype(np.float32)
                    p2, p98 = np.percentile(vv, 2), np.percentile(vv, 98)
                    vv_norm = np.clip((vv - p2) / (p98 - p2 + 1e-6), 0, 1)
                    rgb_img = (np.stack([vv_norm] * 3, axis=-1) * 255).astype(np.uint8)
                else:
                    metadata["sensor_type"] = "Optical"
                    metadata["modality_guess"] = f"Multispectral Optical ({src.count} bands)"
                    if src.count >= 3:
                        r, g, b = data[0], data[1], data[2]
                    else:
                        r = g = b = data[0]
                    
                    r_norm = np.clip((r - np.percentile(r, 2)) / (np.percentile(r, 98) - np.percentile(r, 2) + 1e-6), 0, 1)
                    g_norm = np.clip((g - np.percentile(g, 2)) / (np.percentile(g, 98) - np.percentile(g, 2) + 1e-6), 0, 1)
                    b_norm = np.clip((b - np.percentile(b, 2)) / (np.percentile(b, 98) - np.percentile(b, 2) + 1e-6), 0, 1)
                    rgb_img = (np.stack([r_norm, g_norm, b_norm], axis=-1) * 255).astype(np.uint8)

                return rgb_img, metadata
        except Exception as e:
            print(f"Rasterio warning loading {file_path}: {e}")

    # Fallback with PIL for standard PNG/JPEG
    pil_img = Image.open(file_path).convert("RGB")
    rgb_img = np.array(pil_img)
    metadata["width"] = pil_img.width
    metadata["height"] = pil_img.height
    metadata["channels"] = 3
    metadata["gsd_m"] = "10.0m (Standard Benchmark Resolution)"
    
    if "sar" in filename.lower() or "radar" in filename.lower():
        metadata["sensor_type"] = "SAR"
        metadata["modality_guess"] = "SAR Microwave Radar"
    else:
        metadata["sensor_type"] = "Optical"
        metadata["modality_guess"] = "Optical RGB Image"

    return rgb_img, metadata
