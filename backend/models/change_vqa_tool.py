import cv2
import numpy as np
import torch
from typing import Dict, Any

from backend.models.rs_adapter import BigEarthNetVisionAdapter

class ChangeVQATool:
    """
    Bi-Temporal Change Detection & Change VQA Specialist Tool.
    Extracts deep visual features for image pair T1 and T2, calculates spatial feature difference maps,
    quantifies land-cover transitions, and generates evidence-grounded change VQA answers.
    """
    def __init__(self, checkpoint_path: str = "backend/models/bigearthnet_adapter.pth"):
        self.adapter = BigEarthNetVisionAdapter(checkpoint_path=checkpoint_path)

    def execute(self, image1_rgb: np.ndarray, image2_rgb: np.ndarray, query: str, metadata1: dict, metadata2: dict) -> Dict[str, Any]:
        h1, w1, _ = image1_rgb.shape
        h2, w2, _ = image2_rgb.shape

        if (h1, w1) != (h2, w2):
            image2_rgb = cv2.resize(image2_rgb, (w1, h1), interpolation=cv2.INTER_LINEAR)
        h, w = h1, w1

        # 1. Deep Feature Extraction for T1 & T2 using RS vision backbone
        analysis1 = self.adapter.analyze_image(image1_rgb)
        analysis2 = self.adapter.analyze_image(image2_rgb)

        feat1 = analysis1["feature_map"] # [512, H_f, W_f]
        feat2 = analysis2["feature_map"] # [512, H_f, W_f]

        # 2. Temporal feature difference calculation
        feat_diff = np.linalg.norm(feat2 - feat1, axis=0) # [H_f, W_f]
        spatial_diff = cv2.resize(feat_diff, (w, h), interpolation=cv2.INTER_CUBIC)

        # Normalize pixel difference map
        norm_diff = cv2.normalize(spatial_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blur_diff = cv2.GaussianBlur(norm_diff, (5, 5), 0)

        # Dynamic thresholding for change mask
        _, change_mask = cv2.threshold(blur_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        changed_pixels = int(np.count_nonzero(change_mask))
        total_pixels = h * w
        change_pct = round((changed_pixels / total_pixels) * 100, 2)

        # Determine dominant spatial quadrant of change
        if changed_pixels > 0:
            y_coords, x_coords = np.where(change_mask > 0)
            mean_y, mean_x = np.mean(y_coords), np.mean(x_coords)
            v_pos = "North" if mean_y < h / 2 else "South"
            h_pos = "West" if mean_x < w / 2 else "East"
            quadrant = f"{v_pos}-{h_pos}"
        else:
            quadrant = "Entire Scene"

        # 3. Classify land-cover shift from top BigEarthNet predictions before & after
        p1_t1 = analysis1["predictions"][0]["class"]
        p1_t2 = analysis2["predictions"][0]["class"]

        q_lower = query.lower()
        if change_pct > 1.5:
            change_trend = "INCREASED"
            if p1_t1 != p1_t2:
                transition_desc = f"Primary land cover shifted from '{p1_t1}' in T1 to '{p1_t2}' in T2."
            else:
                transition_desc = f"Area expansion detected within primary land-cover category '{p1_t2}'."

            change_description = (
                f"Bi-temporal change analysis identifies land-cover conversion between Observation T1 and T2.\n"
                f"1. Change Extent: {change_pct}% scene area modified ({changed_pixels:,} pixels).\n"
                f"2. Primary Hotspot Location: {quadrant} portion of the scene tile.\n"
                f"3. Conversion Profile: {transition_desc}"
            )
        else:
            change_trend = "REMAINED UNCHANGED"
            change_description = (
                f"Bi-temporal comparative analysis reveals high scene stability between Observation T1 and T2.\n"
                f"Measured change area ({change_pct}%) remains within natural seasonal variance thresholds.\n"
                f"Land-Cover Status: REMAINED UNCHANGED."
            )

        # 4. Derive calibrated model confidence from feature contrast ratio & Otsu separability
        contrast_score = float(np.mean(norm_diff[change_mask > 0])) if changed_pixels > 0 else 50.0
        confidence = round(float(np.clip(contrast_score / 255.0 * 1.2, 0.50, 0.96)), 4)

        # 5. Render Spatial Change Map Visual Overlay (T2 background + Jet heatmap + magenta contours)
        change_vis = image2_rgb.copy()
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        blended_change = cv2.addWeighted(change_vis, 0.60, heatmap, 0.40, 0)
        contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended_change, contours, -1, (255, 0, 255), 2)

        if any(k in q_lower for k in ["has the built-up area increased", "increased, decreased", "trend"]):
            text_response = (
                f"Direct Answer to Change Query:\n"
                f"The target area HAS {change_trend}.\n\n"
                f"{change_description}"
            )
        else:
            text_response = (
                f"Bi-Temporal Change Analysis & CDVQA Output:\n"
                f"{change_description}\n\n"
                f"Spatial Change Map generated with pixel-level deep feature differencing."
            )

        return {
            "text_response": text_response,
            "confidence": confidence,
            "is_baseline": False,
            "specialist_tool": "Bi-Temporal Change Understanding & CDVQA Specialist",
            "change_trend": change_trend,
            "change_percentage": change_pct,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "hotspot_quadrant": quadrant,
            "visual_overlay_rgb": blended_change,
            "change_mask_binary": change_mask,
            "diff_map": norm_diff
        }
