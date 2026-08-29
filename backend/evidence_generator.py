import os
import time
import cv2
import numpy as np
from PIL import Image

class EvidenceGenerator:
    """
    10. EVIDENCE GENERATOR
    Generates georeferenced visual evidence images (Bounding boxes, Segmentation masks,
    Change maps, Heatmaps, Optical/SAR overlays, Uncertainty maps) and saves export artifacts.
    """
    def __init__(self, exports_dir: str = "exports"):
        self.exports_dir = exports_dir
        os.makedirs(self.exports_dir, exist_ok=True)

    def generate_and_save_artifacts(self, tool_output: dict, trust_output: dict) -> dict:
        timestamp = int(time.time() * 1000)
        
        # 1. Primary Visual Overlay
        overlay_rgb = tool_output.get("visual_overlay_rgb")
        if overlay_rgb is None:
            overlay_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

        overlay_filename = f"evidence_{timestamp}.png"
        overlay_path = os.path.join(self.exports_dir, overlay_filename)
        Image.fromarray(overlay_rgb).save(overlay_path)

        # 2. Uncertainty Map Overlay
        uncertainty_rgb = trust_output.get("uncertainty_heatmap_rgb")
        if uncertainty_rgb is None:
            uncertainty_rgb = np.zeros_like(overlay_rgb)
        
        uncertainty_filename = f"uncertainty_{timestamp}.png"
        uncertainty_path = os.path.join(self.exports_dir, uncertainty_filename)
        Image.fromarray(uncertainty_rgb).save(uncertainty_path)

        return {
            "primary_evidence_url": f"/exports/{overlay_filename}",
            "uncertainty_map_url": f"/exports/{uncertainty_filename}",
            "primary_evidence_path": overlay_path,
            "uncertainty_map_path": uncertainty_path
        }
