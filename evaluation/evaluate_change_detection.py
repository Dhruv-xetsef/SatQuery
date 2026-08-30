import numpy as np
import cv2
from typing import Dict, Any
from backend.agent import SatQueryAgent
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_change_detection(agent: SatQueryAgent = None) -> Dict[str, Any]:
    if agent is None:
        agent = SatQueryAgent()

    # Ground truth change mask for bitemporal sample pair (Urban expansion in South-East box [300:480, 300:480])
    gt_mask = np.zeros((512, 512), dtype=np.uint8)
    gt_mask[300:480, 300:480] = 1

    res = agent.process_query(
        "What changed between these two dates, and where did the change occur?",
        ["dataset/sample_data/bitemporal_t1.tif", "dataset/sample_data/bitemporal_t2.tif"],
        force_task="change_vqa"
    )

    # Re-run tool to evaluate binary mask
    tool_res = agent.registry.change_tool.execute(
        cv2.imread("dataset/sample_data/bitemporal_t1.png"),
        cv2.imread("dataset/sample_data/bitemporal_t2.png"),
        "What changed?", {}, {}
    )
    pred_mask = (tool_res["change_mask_binary"] > 0).astype(int)

    f1 = float(f1_score(gt_mask.ravel(), pred_mask.ravel(), zero_division=0))
    prec = float(precision_score(gt_mask.ravel(), pred_mask.ravel(), zero_division=0))
    rec = float(recall_score(gt_mask.ravel(), pred_mask.ravel(), zero_division=0))
    
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = float(inter / union) if union > 0 else 0.0

    return {
        "change_f1": round(f1, 4),
        "change_iou": round(iou, 4),
        "change_precision": round(prec, 4),
        "change_recall": round(rec, 4)
    }
