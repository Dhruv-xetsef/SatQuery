from typing import Dict, Any
from backend.agent import SatQueryAgent

def evaluate_cdvqa(agent: SatQueryAgent = None) -> Dict[str, Any]:
    if agent is None:
        agent = SatQueryAgent()

    test_cases = [
        {
            "query": "What changed between these two dates, and where did the change occur?",
            "images": ["dataset/sample_data/bitemporal_t1.tif", "dataset/sample_data/bitemporal_t2.tif"],
            "expected_keywords": ["change", "south-east", "area"]
        },
        {
            "query": "Has the built-up area increased, decreased, or remained unchanged?",
            "images": ["dataset/sample_data/bitemporal_t1.tif", "dataset/sample_data/bitemporal_t2.tif"],
            "expected_keywords": ["increased", "change"]
        }
    ]

    correct = 0
    total = len(test_cases)
    details = []

    for test in test_cases:
        res = agent.process_query(test["query"], test["images"], force_task="change_vqa")
        text = res["text_response"].lower()
        match = any(kw in text for kw in test["expected_keywords"])
        if match:
            correct += 1
        details.append({
            "query": test["query"],
            "matched": match,
            "confidence": res["confidence"]
        })

    accuracy = round(float(correct / total), 4)
    return {
        "change_vqa_accuracy": accuracy,
        "total_samples": total,
        "correct_samples": correct,
        "details": details
    }
