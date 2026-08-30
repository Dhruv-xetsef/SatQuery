import os
import json
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from backend.models.rs_adapter import BigEarthNetVisionAdapter, BIGEARTHNET_CLASSES
from training.dataset_loader import get_dataloaders

def evaluate_model():
    config_path = "training/configs/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "num_classes": len(BIGEARTHNET_CLASSES),
            "batch_size": 16,
            "checkpoint_save_dir": "backend/models",
            "checkpoint_name": "bigearthnet_adapter.pth"
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(config["checkpoint_save_dir"], config["checkpoint_name"])

    print(f"[Evaluation] Loading model from checkpoint: {checkpoint_path}")
    model = BigEarthNetVisionAdapter(checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None).to(device)
    model.eval()

    _, val_loader = get_dataloaders(config)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            _, _, probs = model(images)
            preds = (probs.cpu().numpy() >= 0.5).astype(int)
            all_preds.append(preds)
            all_targets.append(targets.numpy().astype(int))

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    print("=========================================================")
    print("   BigEarthNet RS Vision Adapter Evaluation Metrics     ")
    print("=========================================================")
    print(f" Macro F1 Score:  {macro_f1:.4f}")
    print(f" Micro F1 Score:  {micro_f1:.4f}")
    print(f" Precision:       {precision:.4f}")
    print(f" Recall:          {recall:.4f}")
    print("=========================================================")

    return {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4)
    }

if __name__ == "__main__":
    evaluate_model()
