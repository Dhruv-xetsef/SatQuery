import cv2
import numpy as np
from typing import Dict, Any

class ClassicalOpticalSARBaseline:
    """
    Classical Computer Vision Baseline for Optical-SAR Cross-Modal Analysis.
    Uses thresholding on optical channels and SAR backscatter intensities.
    Explicitly marked as Classical Baseline.
    """
    def execute(self, optical_rgb: np.ndarray, sar_rgb: np.ndarray, query: str, meta_opt: Dict[str, Any], meta_sar: Dict[str, Any]) -> Dict[str, Any]:
        ho, wo, _ = optical_rgb.shape
        hs, ws, _ = sar_rgb.shape

        if (hs, ws) != (ho, wo):
            sar_rgb = cv2.resize(sar_rgb, (wo, ho), interpolation=cv2.INTER_LINEAR)
        h, w = ho, wo

        opt_gray = cv2.cvtColor(optical_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sar_gray = cv2.cvtColor(sar_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        cloud_mask = opt_gray > 220
        cloud_pct = round(float(np.mean(cloud_mask) * 100), 2)
        sar_bright = sar_gray > 160
        water_mask = (sar_gray < 40) & (opt_gray < 80)

        fused_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        fused_rgb[:, :, 0] = cv2.normalize(sar_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fused_rgb[:, :, 1] = optical_rgb[:, :, 1]
        fused_rgb[:, :, 2] = optical_rgb[:, :, 2]
        fused_rgb[sar_bright] = [255, 235, 59]
        fused_rgb[water_mask] = [0, 229, 255]

        builtup_pct = round(float(np.mean(sar_bright) * 100), 2)
        water_pct = round(float(np.mean(water_mask) * 100), 2)

        text_response = (
            f"[Classical Baseline Result — Optical/SAR Thresholding]\n"
            f"Optical Cloud Cover: {cloud_pct}%\n"
            f"SAR High-Backscatter (Built-up candidate): {builtup_pct}%\n"
            f"Low Backscatter Water Candidate: {water_pct}%\n"
            f"Note: This is a classical rule-based threshold baseline, not a deep learning model."
        )

        return {
            "text_response": text_response,
            "confidence": 0.65,
            "is_baseline": True,
            "specialist_tool": "Classical Optical-SAR Threshold Baseline",
            "cloud_percentage": cloud_pct,
            "builtup_percentage": builtup_pct,
            "water_percentage": water_pct,
            "visual_overlay_rgb": fused_rgb,
            "cloud_mask": cloud_mask,
            "water_mask": water_mask,
            "sar_bright_mask": sar_bright
        }
