import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

BIGEARTHNET_CLASSES = [
    "Urban fabric (Continuous/Discontinuous)",
    "Industrial, commercial and transport units",
    "Mine, dump and construction sites",
    "Artificial non-agricultural vegetated areas",
    "Arable land (Permanently irrigated/Non-irrigated)",
    "Permanent crops (Vineyards, Orchards)",
    "Pastures and natural grasslands",
    "Heterogeneous agricultural areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Scrub and/or herbaceous vegetation",
    "Open spaces with little/no vegetation",
    "Inland wetlands (Marshes, Peat bogs)",
    "Coastal wetlands (Salt marshes)",
    "Inland waters (Rivers, Lakes, Reservoirs)",
    "Marine waters (Estuaries, Oceans)",
    "SAR Backscatter Structural Features",
    "Multispectral NIR Vegetative Canopy"
]

class BigEarthNetVisionAdapter(nn.Module):
    """
    Vision-Language Domain Adapter Fine-Tuned / Adapted on BigEarthNet Remote Sensing Dataset.
    Extracts deep visual features and predicts multi-label land-cover taxonomy distribution.
    """
    def __init__(self, num_classes=len(BIGEARTHNET_CLASSES), checkpoint_path=None):
        super(BigEarthNetVisionAdapter, self).__init__()
        base_resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                print(f"[BigEarthNet Adapter] Successfully loaded checkpoint weights from: {checkpoint_path}")
            except Exception as e:
                print(f"[BigEarthNet Adapter] Error loading checkpoint: {e}. Using initialized model.")
        else:
            print("[BigEarthNet Adapter] Checkpoint path not found. Running initialized adapter.")

        self.eval()

    def forward(self, x):
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        probs = self.projection_head(feat)
        return feat, probs

    def analyze_image(self, image_rgb: np.ndarray):
        """
        Runs forward pass on numpy RGB image [H, W, 3] and returns predictions + embeddings.
        """
        tensor_img = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            feat, probs = self.forward(tensor_img)
            probs = probs.squeeze(0).numpy()

        results = []
        for idx, score in enumerate(probs):
            results.append({
                "class": BIGEARTHNET_CLASSES[idx],
                "score": float(score)
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "predictions": results,
            "embedding": feat.squeeze(0).numpy().tolist()
        }
