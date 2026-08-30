import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from backend.models.rs_adapter import BigEarthNetVisionAdapter, BIGEARTHNET_CLASSES
from training.dataset_loader import get_dataloaders

def train_adapter():
    config_path = "training/configs/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "num_classes": len(BIGEARTHNET_CLASSES),
            "batch_size": 16,
            "learning_rate": 0.0003,
            "epochs": 5,
            "train_val_split": 0.8,
            "checkpoint_save_dir": "backend/models",
            "checkpoint_name": "bigearthnet_adapter.pth"
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Starting BigEarthNet RS Model Adaptation on device: {device}")

    train_loader, val_loader = get_dataloaders(config)

    model = BigEarthNetVisionAdapter(num_classes=len(BIGEARTHNET_CLASSES)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

    os.makedirs(config["checkpoint_save_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_save_dir"], config["checkpoint_name"])

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            _, _, probs = model(images)
            loss = criterion(probs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                _, _, probs = model(images)
                loss = criterion(probs, targets)
                val_loss += loss.item() * images.size(0)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch}/{config['epochs']}] - Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

    # Save fine-tuned checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[Training Complete] Checkpoint saved successfully to: {checkpoint_path}")

if __name__ == "__main__":
    train_adapter()
