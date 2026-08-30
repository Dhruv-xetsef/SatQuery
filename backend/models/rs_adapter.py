import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Any, List

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
    Extracts spatial feature maps, global visual embeddings, and multi-label land-cover taxonomy distribution.
    """
    def __init__(self, num_classes: int = len(BIGEARTHNET_CLASSES), checkpoint_path: str = None):
        super(BigEarthNetVisionAdapter, self).__init__()
        base_resnet = models.resnet18(weights=None)
        
        # Spatial feature extractor (up to layer4, shape: [B, 512, H/32, W/32])
        self.backbone_features = nn.Sequential(*list(base_resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
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

        # Default checkpoint location
        if not checkpoint_path:
            checkpoint_path = "backend/models/bigearthnet_adapter.pth"

        if os.path.exists(checkpoint_path):
            try:
                self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
                print(f"[BigEarthNet Adapter] Successfully loaded checkpoint weights from: {checkpoint_path}")
            except Exception as e:
                print(f"[BigEarthNet Adapter] Warning loading checkpoint: {e}. Using initialized weights.")
        else:
            print(f"[BigEarthNet Adapter] Checkpoint path '{checkpoint_path}' not found. Using initialized model.")

        self.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            feature_map: [B, 512, H_feat, W_feat]
            pooled_embedding: [B, 512]
            class_probs: [B, num_classes]
        """
        feat_map = self.backbone_features(x)
        pooled = self.avgpool(feat_map)
        pooled_flat = torch.flatten(pooled, 1)
        probs = self.projection_head(pooled_flat)
        return feat_map, pooled_flat, probs

    def analyze_image(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Runs forward pass on numpy RGB image [H, W, 3].
        Returns class probabilities, feature map, and embeddings.
        """
        tensor_img = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            feat_map, pooled, probs = self.forward(tensor_img)
            probs_np = probs.squeeze(0).numpy()
            feat_map_np = feat_map.squeeze(0).numpy()
            pooled_np = pooled.squeeze(0).numpy()

        results = []
        for idx, score in enumerate(probs_np):
            results.append({
                "class": BIGEARTHNET_CLASSES[idx],
                "score": float(score)
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "predictions": results,
            "embedding": pooled_np,
            "feature_map": feat_map_np
        }
