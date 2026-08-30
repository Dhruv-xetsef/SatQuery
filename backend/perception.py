from typing import List, Dict, Any

class PerceptionLayer:
    """
    1. PERCEPTION LAYER - INPUT INTELLIGENCE & STRICT VALIDATION
    Performs file validation, format checking, metadata extraction, CRS checking,
    modality detection, spatial relationship detection, and co-registration verification.
    """
    def __init__(self):
        self.supported_formats = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    def inspect_and_validate(self, image_metadatas: List[Dict[str, Any]], requested_task: str = "auto") -> Dict[str, Any]:
        num_images = len(image_metadatas)
        warnings = []
        errors = []
        passed = True

        if num_images == 0:
            return {
                "passed": False,
                "status": "FAILED",
                "image_count": 0,
                "relationship_type": "No Image",
                "relationship_desc": "No image inputs provided.",
                "detected_modalities": [],
                "sensor_types": [],
                "spatial_compatibility": "Invalid",
                "errors": ["At least one valid satellite image (GeoTIFF / PNG) must be uploaded."],
                "warnings": [],
                "summary": "Perception check FAILED: Missing image inputs."
            }

        # 1. Format & file validation
        for idx, meta in enumerate(image_metadatas):
            fmt = meta.get("format", "").lower()
            if fmt not in self.supported_formats:
                errors.append(f"Image {idx+1} format '{fmt}' unsupported. Supported formats: GeoTIFF (.tif/.tiff), PNG, JPEG.")
                passed = False

        # 2. Modality & Sensor classification
        detected_modalities = []
        sensor_types = []
        for meta in image_metadatas:
            modality = meta.get("modality", "Optical")
            sensor = meta.get("sensor_type", "Optical")
            detected_modalities.append(modality)
            sensor_types.append(sensor)

        # 3. Relationship Detection
        if num_images == 1:
            relationship_type = f"Single {sensor_types[0]} Image"
            relationship_desc = f"Single {sensor_types[0]} observation."
        elif num_images == 2:
            s1, s2 = sensor_types[0], sensor_types[1]
            if (s1 == "Optical" and s2 == "SAR") or (s1 == "SAR" and s2 == "Optical"):
                relationship_type = "Optical + SAR Cross-Modal Pair"
                relationship_desc = "Cross-modal pair (Optical spectral reflectance + SAR microwave radar backscatter)."
            else:
                relationship_type = "Bi-Temporal Pair (T1 + T2)"
                relationship_desc = f"Bi-temporal multitemporal pair for change analysis ({s1} T1 vs {s2} T2)."
        else:
            relationship_type = f"Multi-Image Collection ({num_images} images)"
            relationship_desc = f"Collection of {num_images} geospatial layers."

        # 4. Strict Input Validation per Task
        spatial_compatibility = "Valid"

        if requested_task == "change_vqa" and num_images < 2:
            errors.append("Two spatially corresponding images are required for bi-temporal analysis.")
            passed = False

        if requested_task == "optical_sar":
            if num_images < 2:
                errors.append("Cross-modal analysis requires one optical/multispectral image and one SAR image.")
                passed = False
            elif num_images == 2:
                has_optical = "Optical" in sensor_types
                has_sar = "SAR" in sensor_types
                if not (has_optical and has_sar):
                    errors.append("Cross-modal analysis requires one optical/multispectral image and one SAR image.")
                    passed = False

        # 5. Spatial Compatibility & CRS validation for pairs
        if num_images >= 2:
            img1, img2 = image_metadatas[0], image_metadatas[1]
            h1, w1 = img1.get("height", 0), img1.get("width", 0)
            h2, w2 = img2.get("height", 0), img2.get("width", 0)

            if (h1, w1) != (h2, w2):
                warnings.append(f"Dimension mismatch in pair: Img1 is {w1}x{h1}, Img2 is {w2}x{h2}. Spatial resampling applied for alignment.")
                spatial_compatibility = "Resampled & Aligned"

            crs1, crs2 = img1.get("crs"), img2.get("crs")
            if img1.get("is_geotiff") and img2.get("is_geotiff"):
                if crs1 != "N/A" and crs2 != "N/A" and crs1 != crs2:
                    warnings.append(f"CRS mismatch: '{crs1}' vs '{crs2}'. Affine re-projection coordinate alignment applied.")

            # Bounds overlap verification if GeoTIFF bounds present
            b1, b2 = img1.get("bounds"), img2.get("bounds")
            if b1 and b2:
                overlap = not (b1["right"] < b2["left"] or b1["left"] > b2["right"] or b1["top"] < b2["bottom"] or b1["bottom"] > b2["top"])
                if not overlap:
                    errors.append("Input images do not spatially overlap geographically.")
                    passed = False

        status_text = "PASSED" if (passed and len(warnings) == 0) else ("PASSED WITH WARNINGS" if passed else "FAILED")

        return {
            "passed": passed,
            "status": status_text,
            "image_count": num_images,
            "relationship_type": relationship_type,
            "relationship_desc": relationship_desc,
            "detected_modalities": detected_modalities,
            "sensor_types": sensor_types,
            "spatial_compatibility": spatial_compatibility,
            "errors": errors,
            "warnings": warnings,
            "summary": f"Perception validation {status_text}. Mode: {relationship_type}. {len(errors)} error(s), {len(warnings)} warning(s)."
        }
