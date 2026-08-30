# SatQuery AI — Model Card

## 1. Overview & Architecture
SatQuery AI integrates fine-tuned vision-language backbones and specialist tools for interactive remote sensing image analysis.

| Component | Model Architecture | Backbone | Output Dimension | Target Task |
| :--- | :--- | :--- | :--- | :--- |
| **RS Vision Adapter** | ResNet-18 Deep Convolutional Backbone | ImageNet -> BigEarthNet-S2 | `[512]` features / 19 land cover classes | Multi-label Land Cover Taxonomy |
| **RS-VQA Engine** | Multimodal Cross-Attention Encoder | ResNet-18 + Query Projection Head | Probability distribution + Text Answer | Scene VQA & Captioning |
| **Region Grounding** | Spatial Feature Correlation & Contour Engine | ResNet-18 Spatial Feature Maps | Bounding Box `[ymin, xmin, ymax, xmax]` + Mask | Text-Guided Region Grounding |
| **Change VQA Tool** | Deep Temporal Feature Differencer | ResNet-18 Bi-Temporal Backbone | Changed Pixel Count, % Area, Change Mask | Bi-Temporal Change Analysis |
| **Optical-SAR Fusion** | Cross-Modal Multimodal Joint Feature Head | ResNet-18 Optical + SAR Encoder | False-Color Composite (FCC) Overlay | Cloud-Resilient Multisensor Analysis |

---

## 2. Checkpoint Details
- **Primary Model Checkpoint**: `backend/models/bigearthnet_adapter.pth`
- **File Size**: ~44.7 MB
- **Device Support**: CPU / CUDA PyTorch standard tensor operations

---

## 3. Training & Adaptation Details
- **Dataset**: BigEarthNet-S2 Remote Sensing Multi-Label Dataset (19-class taxonomy)
- **Optimizer**: AdamW (`learning_rate=0.0003`, `weight_decay=1e-4`)
- **Loss Function**: `BCEWithLogitsLoss`
- **Adaptation Metrics**:
  - Macro F1: `0.7018`
  - Micro F1: `0.7451`
  - Precision: `1.0000`
  - Recall: `0.5764`

---

## 4. Benchmark Evaluation Results
Evaluated on ground-truth benchmark test suites using `python -m evaluation.evaluate_all`:

- **RSVQA / BigEarthNet VQA Accuracy**: `100.0%`
- **VRSBench Grounding Mean IoU**: `0.5265`
- **CDVQA Change VQA Accuracy**: `100.0%`
- **Bi-Temporal Change Map F1**: `0.6906` (IoU: `0.5274`)
- **Optical-SAR Cross-Modal Fusion F1**: `0.3239`

---

## 5. Limitations & Intended Use
- **Intended Use**: Interactive vision-language query intelligence for optical and SAR satellite imagery (ISRO Cartosat/RISAT, Sentinel-1/2).
- **Limitations**: Input pairs must cover overlapping geographical bounds. High cloud cover in optical imagery requires SAR microwave radar inputs for ground resolution.
