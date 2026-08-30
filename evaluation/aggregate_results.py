import json
import os
from typing import Dict, Any

def aggregate_benchmark_results(
    rsvqa_res: Dict[str, Any],
    vrsbench_res: Dict[str, Any],
    cdvqa_res: Dict[str, Any],
    change_res: Dict[str, Any],
    optical_sar_res: Dict[str, Any],
    output_dir: str = "results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "rsvqa.json"), "w") as f:
        json.dump(rsvqa_res, f, indent=2)

    with open(os.path.join(output_dir, "vrsbench.json"), "w") as f:
        json.dump(vrsbench_res, f, indent=2)

    with open(os.path.join(output_dir, "cdvqa.json"), "w") as f:
        json.dump(cdvqa_res, f, indent=2)

    with open(os.path.join(output_dir, "change_detection.json"), "w") as f:
        json.dump(change_res, f, indent=2)

    with open(os.path.join(output_dir, "optical_sar.json"), "w") as f:
        json.dump(optical_sar_res, f, indent=2)

    overall_metrics = {
        "rsvqa_accuracy": rsvqa_res.get("rsvqa_accuracy", 0.0),
        "grounding_iou": vrsbench_res.get("grounding_iou", 0.0),
        "change_f1": change_res.get("change_f1", 0.0),
        "change_vqa_accuracy": cdvqa_res.get("change_vqa_accuracy", 0.0),
        "optical_sar_f1": optical_sar_res.get("optical_sar_f1", 0.0)
    }

    with open(os.path.join(output_dir, "overall.json"), "w") as f:
        json.dump(overall_metrics, f, indent=2)

    return overall_metrics
