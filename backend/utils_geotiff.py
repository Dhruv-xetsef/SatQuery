import os
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def load_and_inspect_image(file_path: str, max_dim: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads an optical or SAR remote sensing image (GeoTIFF/TIFF, PNG, JPEG),
    extracts geospatial & sensor metadata (format, dimensions, CRS, affine transform, bounds, bands, modality),
    and returns a normalized 3-channel RGB numpy array [H, W, 3] + metadata dictionary.
    
    Supports dynamic range normalization (percentile clipping, NaN/nodata handling) and tiling/windowing for large imagery.
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
        "dtype": "uint8",
        "crs": "N/A",
        "transform": None,
        "gsd_m": "N/A",
        "bounds": None,
        "modality": "Unknown",
        "sensor_type": "Optical",
        "acquisition_date": "N/A",
        "nodata_val": None
    }

    if HAS_RASTERIO and ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(file_path) as src:
                metadata["is_geotiff"] = True
                metadata["width"] = src.width
                metadata["height"] = src.height
                metadata["channels"] = src.count
                metadata["dtype"] = str(src.dtypes[0])
                metadata["crs"] = str(src.crs) if src.crs else "EPSG:4326 (Unspecified)"
                metadata["transform"] = list(src.transform) if src.transform else None
                metadata["nodata_val"] = src.nodatavals[0] if src.nodatavals else None

                # Extract acquisition date from metadata if present
                tags = src.tags()
                if "ACQUISITION_DATE" in tags:
                    metadata["acquisition_date"] = tags["ACQUISITION_DATE"]
                elif "TIFFTAG_DATETIME" in tags:
                    metadata["acquisition_date"] = tags["TIFFTAG_DATETIME"]

                # Estimate Ground Sample Distance (GSD in meters)
                if src.res and src.res[0] > 0:
                    res_val = round(abs(src.res[0]), 3)
                    metadata["gsd_m"] = f"{res_val}m"
                else:
                    metadata["gsd_m"] = "10.0m (Default RS Resolution)"

                if src.bounds:
                    metadata["bounds"] = {
                        "left": round(src.bounds.left, 5),
                        "bottom": round(src.bounds.bottom, 5),
                        "right": round(src.bounds.right, 5),
                        "top": round(src.bounds.top, 5)
                    }

                # Support windowed read if dimensions exceed max_dim to avoid memory overflow
                if src.width > max_dim or src.height > max_dim:
                    data = src.read(out_shape=(src.count, max_dim, max_dim), resampling=rasterio.enums.Resampling.bilinear)
                else:
                    data = src.read()  # Shape: (count, H, W)

                # NaN / NoData handling
                data = data.astype(np.float32)
                if metadata["nodata_val"] is not None:
                    data[data == metadata["nodata_val"]] = np.nan
                data = np.nan_to_num(data, nan=0.0)

                # Sensor & Modality classification
                is_sar_filename = any(k in filename.lower() for k in ["sar", "radar", "risat", "sentinel1", "s1"])
                if src.count in [1, 2] or is_sar_filename:
                    metadata["sensor_type"] = "SAR"
                    metadata["modality"] = "SAR"
                    # Render SAR grayscale / 3-channel with log scaling if dynamic range is high
                    vv = data[0]
                    # Optional log transform for high dynamic range radar backscatter
                    if np.max(vv) > 255:
                        vv = np.log1p(np.maximum(0, vv))
                    p2, p98 = np.percentile(vv, 2), np.percentile(vv, 98)
                    denom = p98 - p2 if (p98 - p2) > 1e-6 else 1.0
                    vv_norm = np.clip((vv - p2) / denom, 0, 1)
                    rgb_img = (np.stack([vv_norm] * 3, axis=-1) * 255).astype(np.uint8)
                else:
                    metadata["sensor_type"] = "Optical"
                    metadata["modality"] = "Optical"
                    if src.count >= 3:
                        r, g, b = data[0], data[1], data[2]
                    else:
                        r = g = b = data[0]

                    def norm_band(b_arr):
                        p2, p98 = np.percentile(b_arr, 2), np.percentile(b_arr, 98)
                        denom = p98 - p2 if (p98 - p2) > 1e-6 else 1.0
                        return np.clip((b_arr - p2) / denom, 0, 1)

                    rgb_img = (np.stack([norm_band(r), norm_band(g), norm_band(b)], axis=-1) * 255).astype(np.uint8)

                return rgb_img, metadata
        except Exception as e:
            print(f"Rasterio error reading {file_path}: {e}. Falling back to PIL.")

    # Fallback with PIL for standard PNG/JPEG/TIFF
    pil_img = Image.open(file_path).convert("RGB")
    rgb_img = np.array(pil_img)
    metadata["width"] = pil_img.width
    metadata["height"] = pil_img.height
    metadata["channels"] = 3
    metadata["gsd_m"] = "10.0m (Benchmark Data Resolution)"
    
    if "sar" in filename.lower() or "radar" in filename.lower():
        metadata["sensor_type"] = "SAR"
        metadata["modality"] = "SAR"
    else:
        metadata["sensor_type"] = "Optical"
        metadata["modality"] = "Optical"

    return rgb_img, metadata
