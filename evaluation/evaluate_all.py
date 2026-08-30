import argparse
import json
import os
import sys

from backend.agent import SatQueryAgent
from evaluation.evaluate_rsvqa import evaluate_rsvqa
from evaluation.evaluate_vrsbench import evaluate_vrsbench
from evaluation.evaluate_cdvqa import evaluate_cdvqa
from evaluation.evaluate_change_detection import evaluate_change_detection
from evaluation.evaluate_optical_sar import evaluate_optical_sar
from evaluation.aggregate_results import aggregate_benchmark_results

def run_all_evaluations(output_dir: str = "results") -> dict:
    print("=======================================================================")
    print("   SatQuery AI — Ground Truth Remote Sensing Evaluation Suite          ")
    print("=======================================================================\n")

    agent = SatQueryAgent()

    print("[1/5] Evaluating RSVQA / BigEarthNet VQA...")
    rsvqa_res = evaluate_rsvqa(agent)
    print(f"      -> RSVQA Accuracy: {rsvqa_res['rsvqa_accuracy']*100:.1f}%\n")

    print("[2/5] Evaluating VRSBench Region Grounding...")
    vrsbench_res = evaluate_vrsbench(agent)
    print(f"      -> Grounding Mean IoU: {vrsbench_res['grounding_iou']:.4f}\n")

    print("[3/5] Evaluating CDVQA Bi-Temporal Change VQA...")
    cdvqa_res = evaluate_cdvqa(agent)
    print(f"      -> CDVQA Accuracy: {cdvqa_res['change_vqa_accuracy']*100:.1f}%\n")

    print("[4/5] Evaluating Bi-Temporal Change Detection Maps...")
    change_res = evaluate_change_detection(agent)
    print(f"      -> Change Detection F1: {change_res['change_f1']:.4f} | IoU: {change_res['change_iou']:.4f}\n")

    print("[5/5] Evaluating Optical-SAR Cross-Modal Fusion...")
    optical_sar_res = evaluate_optical_sar(agent)
    print(f"      -> Optical-SAR F1: {optical_sar_res['optical_sar_f1']:.4f}\n")

    overall = aggregate_benchmark_results(
        rsvqa_res, vrsbench_res, cdvqa_res, change_res, optical_sar_res, output_dir=output_dir
    )

    print("=======================================================================")
    print(f"   EVALUATION COMPLETE — Saved metrics to '{output_dir}/overall.json' ")
    print("=======================================================================")
    print(json.dumps(overall, indent=2))
    print("=======================================================================\n")

    return overall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SatQuery AI Benchmark Evaluation Suite")
    parser.add_argument("--dataset_root", type=str, default="dataset/sample_data", help="Dataset directory")
    parser.add_argument("--output", type=str, default="results", help="Output directory for JSON results")
    args = parser.parse_args()

    run_all_evaluations(output_dir=args.output)
