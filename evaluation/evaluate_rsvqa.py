import json
import os
from typing import Dict, Any
from backend.agent import SatQueryAgent

def evaluate_rsvqa(agent: SatQueryAgent = None) -> Dict[str, Any]:
    if agent is None:
        agent = SatQueryAgent()

    test_cases = [
        {
            "query": "Describe the land-cover and major objects visible in this image.",
            "image": "dataset/sample_data/single_optical.tif",
            "expected_keywords": ["land cover", "visual", "urban", "forest", "water", "features"]
        },
        {
            "query": "What type of land cover dominates the scene?",
            "image": "dataset/sample_data/single_optical.tif",
            "expected_keywords": ["land cover", "visual", "urban", "forest", "vegetation"]
        }
    ]

    correct = 0
    total = len(test_cases)
    results_detail = []

    for test in test_cases:
        res = agent.process_query(test["query"], [test["image"]], force_task="vqa_caption")
        text = res["text_response"].lower()
        match = any(kw in text for kw in test["expected_keywords"])
        if match:
            correct += 1
        results_detail.append({
            "query": test["query"],
            "matched": match,
            "confidence": res["confidence"]
        })

    accuracy = round(float(correct / total), 4)
    out_data = {
        "rsvqa_accuracy": accuracy,
        "total_samples": total,
        "correct_samples": correct,
        "details": results_detail
    }
    return out_data
