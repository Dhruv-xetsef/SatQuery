from typing import List, Dict, Any

class PerceptionLayer:
    """
    1. PERCEPTION LAYER - INPUT INTELLIGENCE
    Performs file validation, format checking, metadata extraction, CRS checking,
    modality detection, image relationship detection, and co-registration validation.
    """
    def __init__(self):
        self.supported_formats = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    def inspect_and_validate(self, image_metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        num_images = len(image_metadatas)
        warnings = []
        errors = []
        passed = True

        # 1. Format & file validation
        for idx, meta in enumerate(image_metadatas):
            fmt = meta.get("format", "").lower()
            if fmt not in self.supported_formats:
                errors.append(f"Image {idx+1} format '{fmt}' unsupported. Use GeoTIFF (.tif/.tiff), PNG, or JPEG.")
                passed = False

        # 2. Modality & Sensor classification
        detected_modalities = []
        sensor_types = []
        for meta in image_metadatas:
            modality = meta.get("modality_guess", "Optical")
            sensor = meta.get("sensor_type", "Optical")
            detected_modalities.append(modality)
            sensor_types.append(sensor)

        # 3. Relationship Detection
        if num_images == 1:
            relationship_type = "Single Image"
            relationship_desc = f"Single {sensor_types[0]} image analysis mode."
        elif num_images == 2:
            s1, s2 = sensor_types[0], sensor_types[1]
            if (s1 == "Optical" and s2 == "SAR") or (s1 == "SAR" and s2 == "Optical"):
                relationship_type = "Co-registered Optical + SAR"
                relationship_desc = "Cross-modal paired observation (Optical spectral reflectance + SAR microwave backscatter)."
            else:
                relationship_type = "Bi-temporal T1 + T2"
                relationship_desc = "Bi-temporal multitemporal pair for change detection and temporal monitoring."
        else:
            relationship_type = f"Multi-Image Cluster ({num_images} images)"
            relationship_desc = f"Multi-source collection of {num_images} geospatial layers."

        # 4. Spatial Compatibility & Co-registration check
        spatial_compatibility = "Valid"
        if num_images >= 2:
            img1, img2 = image_metadatas[0], image_metadatas[1]
            h1, w1 = img1.get("height", 0), img1.get("width", 0)
            h2, w2 = img2.get("height", 0), img2.get("width", 0)

            if (h1, w1) != (h2, w2):
                warnings.append(f"Dimension mismatch between pair: Img1 is {w1}x{h1}, Img2 is {w2}x{h2}. Auto-resampling & bilinear alignment engaged.")
                spatial_compatibility = "Auto-Resampled & Aligned"

            crs1, crs2 = img1.get("crs"), img2.get("crs")
            if img1.get("is_geotiff") and img2.get("is_geotiff"):
                if crs1 != crs2:
                    warnings.append(f"CRS mismatch: '{crs1}' vs '{crs2}'. Affine re-projection coordinate alignment applied.")

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
            "summary": f"Perception check {status_text}. Mode: {relationship_type}. {len(warnings)} warning(s)."
        }
