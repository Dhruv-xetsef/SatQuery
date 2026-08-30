import cv2
import numpy as np
from typing import Dict, Any

class ClassicalChangeBaseline:
    """
    Classical Computer Vision Baseline for Bi-Temporal Change Detection.
    Uses pixel-level intensity differencing and adaptive thresholding.
    Explicitly marked as Classical Baseline (non-learned).
    """
    def execute(self, image1_rgb: np.ndarray, image2_rgb: np.ndarray, query: str, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> Dict[str, Any]:
        h1, w1, _ = image1_rgb.shape
        h2, w2, _ = image2_rgb.shape

        if (h1, w1) != (h2, w2):
            image2_rgb = cv2.resize(image2_rgb, (w1, h1), interpolation=cv2.INTER_LINEAR)
        h, w = h1, w1

        g1 = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        diff = cv2.absdiff(g2, g1)
        blur_diff = cv2.GaussianBlur(diff, (5, 5), 0)

        max_diff = np.max(blur_diff) if np.max(blur_diff) > 0 else 1.0
        norm_diff = (blur_diff / max_diff * 255).astype(np.uint8)
        _, change_mask = cv2.threshold(norm_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        changed_pixels = int(np.count_nonzero(change_mask))
        total_pixels = h * w
        change_pct = round((changed_pixels / total_pixels) * 100, 2)

        # Baseline change map visualization
        change_vis = image2_rgb.copy()
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(change_vis, 0.60, heatmap, 0.40, 0)
        contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 0, 255), 2)

        confidence = round(float(np.mean(norm_diff[change_mask > 0])) / 255.0, 4) if changed_pixels > 0 else 0.50

        text_response = (
            f"[Classical Baseline Result — Image Differencing]\n"
            f"Detected change across {change_pct}% of the scene ({changed_pixels:,} pixels).\n"
            f"Note: This is a classical CV thresholding baseline, not a deep learning model."
        )

        return {
            "text_response": text_response,
            "confidence": max(0.40, min(0.85, confidence)),
            "is_baseline": True,
            "specialist_tool": "Classical Image Differencing Baseline",
            "change_percentage": change_pct,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "visual_overlay_rgb": blended,
            "change_mask_binary": change_mask,
            "diff_map": norm_diff
        }
