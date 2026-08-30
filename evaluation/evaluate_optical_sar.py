import cv2
import numpy as np
from typing import Dict, Any
from backend.agent import SatQueryAgent
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_optical_sar(agent: SatQueryAgent = None) -> Dict[str, Any]:
    if agent is None:
        agent = SatQueryAgent()

    # Ground truth optical cloud region (circle at 120, 120, r=100)
    gt_cloud = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(gt_cloud, (120, 120), 100, 1, -1)

    opt_img = cv2.imread("dataset/sample_data/crossmodal_optical.png")
    sar_img = cv2.imread("dataset/sample_data/crossmodal_sar.png")

    tool_res = agent.registry.optical_sar_tool.execute(
        opt_img, sar_img, "Use optical and SAR together", {}, {}
    )
    pred_cloud = (tool_res["cloud_mask"] > 0).astype(int)

    f1 = float(f1_score(gt_cloud.ravel(), pred_cloud.ravel(), zero_division=0))
    prec = float(precision_score(gt_cloud.ravel(), pred_cloud.ravel(), zero_division=0))
    rec = float(recall_score(gt_cloud.ravel(), pred_cloud.ravel(), zero_division=0))

    return {
        "optical_sar_f1": round(f1, 4),
        "optical_sar_precision": round(prec, 4),
        "optical_sar_recall": round(rec, 4)
    }
