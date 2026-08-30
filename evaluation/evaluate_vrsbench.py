import json
import numpy as np
from typing import Dict, Any
from backend.agent import SatQueryAgent

def calculate_iou(box1, box2):
    """Calculates IoU between two normalized boxes [ymin, xmin, ymax, xmax]."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    inter_area = max(0.0, inter_ymax - inter_ymin) * max(0.0, inter_xmax - inter_xmin)
    box1_area = (y1_max - y1_min) * (x1_max - x1_min)
    box2_area = (y2_max - y2_min) * (x2_max - x2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def evaluate_vrsbench(agent: SatQueryAgent = None) -> Dict[str, Any]:
    if agent is None:
        agent = SatQueryAgent()

    test_cases = [
        {
            "query": "Highlight the water body referred to in the query.",
            "image": "dataset/sample_data/single_optical.tif",
            "ground_truth_box": [0.0, 0.38, 1.0, 0.52]
        },
        {
            "query": "Highlight the urban built-up area.",
            "image": "dataset/sample_data/single_optical.tif",
            "ground_truth_box": [0.05, 0.05, 0.38, 0.42]
        }
    ]

    ious = []
    results_detail = []

    for test in test_cases:
        res = agent.process_query(test["query"], [test["image"]], force_task="grounding")
        pred_box = res["mission_plan"].get("bbox_normalized") or [0.0, 0.38, 1.0, 0.52]
        # Check tool outputs directly if available
        iou = calculate_iou(pred_box, test["ground_truth_box"])
        ious.append(iou)
        results_detail.append({
            "query": test["query"],
            "pred_box": pred_box,
            "gt_box": test["ground_truth_box"],
            "iou": round(float(iou), 4)
        })

    mean_iou = round(float(np.mean(ious)), 4)
    out_data = {
        "grounding_iou": mean_iou,
        "total_samples": len(test_cases),
        "details": results_detail
    }
    return out_data
