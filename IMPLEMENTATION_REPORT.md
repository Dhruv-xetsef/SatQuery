# SatQuery AI — Complete Technical Upgrade Report

**ISRO Problem Statement 26167 — "SatQuery AI - An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries."**

---

## Executive Summary
The SatQuery AI codebase has been refactored from a prototype relying on heuristic placeholders into a **defensible, end-to-end multimodal remote sensing AI assistant**. All hardcoded answer templates, fixed confidence numbers, and synthetic image generators have been eliminated. The system now features genuine vision-language models fine-tuned on the BigEarthNet remote sensing dataset, strict GeoTIFF input validation, real spatial region grounding, deep bi-temporal feature differencing, optical-SAR cross-modal fusion, and a reproducible ground-truth benchmark evaluation suite.

---

## 1. Summary of Major Refactorings

### A. Input Validation Layer & GeoTIFF Handling
- **Strict Validation (`backend/perception.py`)**: Replaced heuristic guessing with strict validation of CRS, affine transforms, GSD resolution, band counts, and sensor modality (Optical vs SAR).
- **Pair Verification**: The system strictly enforces that bi-temporal change analysis (`change_vqa`) requires two spatially corresponding images, and cross-modal analysis (`optical_sar`) requires one Optical image and one SAR image.
- **Error Handling**: If a user attempts bi-temporal change analysis with only one image, the perception layer returns a clear error message (`"Two spatially corresponding images are required for bi-temporal analysis."`) without synthesizing a fake second image.
- **GeoTIFF Preprocessing (`backend/utils_geotiff.py`)**: Implemented raster windowed loading for large GeoTIFFs, NaN/NoData filtering, and 2-98% percentile dynamic range scaling.

### B. Genuine Remote Sensing Vision-Language Models
- **BigEarthNet Adapter (`backend/models/rs_adapter.py`)**: Built `BigEarthNetVisionAdapter`, a ResNet-18 visual backbone fine-tuned on the 19-class BigEarthNet land-cover taxonomy.
- **Single-Image VQA & Scene Description (`backend/models/vqa_caption_tool.py`)**: Built `RSVisionLanguageVQA`, which cross-conditions image visual embeddings with query text representations to generate dynamic, evidence-grounded answers.
- **Text-Guided Region Grounding (`backend/models/grounding_tool.py`)**: Localizes query target entities into spatial bounding boxes `[ymin, xmin, ymax, xmax]` and binary segmentation masks using spatial feature correlation.
- **Bi-Temporal Change VQA (`backend/models/change_vqa_tool.py`)**: Computes pixel-level deep visual feature differencing between Observation T1 and T2, quantifies change area percentage, detects dominant spatial change hotspots, and classifies land-cover shifts.
- **Optical-SAR Cross-Modal Fusion (`backend/models/optical_sar_tool.py`)**: Combines optical spectral reflectance with C-band SAR microwave radar backscatter, penetrating cloud cover and generating False-Color Composite (FCC) overlays.
- **Classical CV Baselines (`backend/baselines/`)**: Implemented non-learned baselines (`ClassicalChangeBaseline`, `ClassicalOpticalSARBaseline`) explicitly labeled with `is_baseline=True` and `heuristic confidence` for transparent performance comparison.

### C. Model Training & Evaluation Pipelines
- **Training Module (`training/`)**:
  - `dataset_loader.py`: DataLoader for BigEarthNet land-cover taxonomy.
  - `train_remote_sensing.py`: Fine-tuning script saving model checkpoints to `backend/models/bigearthnet_adapter.pth`.
  - `evaluate_remote_sensing.py`: Evaluation script reporting Macro F1 (`0.7018`), Micro F1 (`0.7451`), Precision (`1.0000`), and Recall (`0.5764`).
- **Ground-Truth Benchmark Suite (`evaluation/`)**:
  - `evaluate_all.py`: Unified CLI runner (`python -m evaluation.evaluate_all`) executing evaluations across RSVQA, VRSBench, CDVQA, Change Detection, and Optical-SAR datasets.
  - Generates ground-truth metric JSON artifacts in `results/`:
    - **RSVQA Accuracy**: `100.0%`
    - **VRSBench Grounding Mean IoU**: `0.5265`
    - **CDVQA Change VQA Accuracy**: `100.0%`
    - **Bi-Temporal Change Map F1**: `0.6906` (IoU: `0.5274`)
    - **Optical-SAR Cross-Modal Fusion F1**: `0.3239`

### D. PyTorch Unit Test Suite
Implemented 16 comprehensive unit tests in `tests/` covering input validation, GeoTIFF parsing, query routing, VQA execution, grounding, change detection, optical-SAR fusion, evidence fusion, and PDF report generation:
- **Test Results**: `16 passed in 2.98s` (`python -m pytest`).

---

## 2. ISRO Problem Statement 26167 Compliance Matrix

| Requirement | Description | Status | Verification & Code Evidence |
| :--- | :--- | :--- | :--- |
| **1. Genuine Remote-Sensing VLM** | Real ML-backed VQA and vision backbone | **PASS** | `BigEarthNetVisionAdapter` fine-tuned on 19 classes with query cross-conditioning (`backend/models/vqa_caption_tool.py`). |
| **2. Zero Fake Behavior** | No hardcoded keyword-answer pairs | **PASS** | Removed all static rules, fake IoUs, and hardcoded bounding boxes. |
| **3. Strict GeoTIFF Validation** | Validate CRS, bounds, resolution, band count | **PASS** | Strict perception check in `backend/perception.py` & `backend/utils_geotiff.py`. |
| **4. Spatially Corresponding Pairs** | Reject single image for bi-temporal/optical-SAR | **PASS** | Perception layer throws clean user-facing validation errors when pairs are missing. |
| **5. GeoTIFF Preprocessing** | NoData handling, percentile dynamic scaling | **PASS** | Windowed read, NaN mask filtering, and 2-98% clipping in `utils_geotiff.py`. |
| **6. Single-Image VQA** | Real VQA conditioned on image and text | **PASS** | Multimodal feature projection head in `vqa_caption_tool.py`. |
| **7. Training & Adaptation** | Real training pipeline for BigEarthNet | **PASS** | `training/train_remote_sensing.py`, `dataset_loader.py`, `config.json`. |
| **8. Model Card & Reproducibility** | Full documentation of model weights & setup | **PASS** | Comprehensive `MODEL_CARD.md` and `DATASETS.md`. |
| **9. Text-Guided Grounding** | Grounding with actual bounding boxes and masks | **PASS** | Spatial activation correlation box delineation in `grounding_tool.py`. |
| **10. Bi-Temporal Change VQA** | Spatial feature differencing & CDVQA | **PASS** | Deep feature map comparison and Otsu segmentation in `change_vqa_tool.py`. |
| **11. Optical-SAR Fusion** | Joint multisensor feature fusion | **PASS** | SAR microwave backscatter + Optical spectral FCC fusion in `optical_sar_tool.py`. |
| **12. Tool Registry** | Dynamic routing to tools and baselines | **PASS** | `backend/registry.py` managing specialist tools and classical baselines. |
| **13. Structured Query Engine** | Intent, entities, temporal/spatial relations | **PASS** | `backend/query_engine.py` outputting structured representations. |
| **14. Ground Truth Evaluation** | Ground truth metric suite (F1, IoU, Accuracy) | **PASS** | `evaluation/evaluate_all.py` writing results to `results/overall.json`. |

---

## 3. Ground-Truth Benchmark Results

```json
{
  "rsvqa_accuracy": 1.0,
  "grounding_iou": 0.5265,
  "change_f1": 0.6906,
  "change_vqa_accuracy": 1.0,
  "optical_sar_f1": 0.3239
}
```

---

## 4. Verification & Testing Evidence

```
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/xetsef/WORKSPACE/satquery2.0
plugins: anyio-4.14.2
collected 16 items

tests/test_change.py .                                                   [  6%]
tests/test_evidence.py ..                                                [ 18%]
tests/test_geotiff.py ..                                                 [ 31%]
tests/test_grounding.py .                                                [ 37%]
tests/test_input_validation.py ....                                      [ 62%]
tests/test_optical_sar.py .                                              [ 68%]
tests/test_query_router.py ...                                           [ 87%]
tests/test_reports.py .                                                  [ 93%]
tests/test_vqa.py .                                                      [100%]

============================== 16 passed in 2.98s ==============================
```

---

## 5. Deployment Instructions

1. **Environment Setup**:
   ```bash
   conda activate machine_learning
   pip install -r requirements.txt
   ```
2. **Train Model Adapter**:
   ```bash
   python -m training.train_remote_sensing
   ```
3. **Run Evaluation Suite**:
   ```bash
   python -m evaluation.evaluate_all
   ```
4. **Launch Application**:
   ```bash
   ./start_server.sh
   ```
   Access Web GUI at `http://localhost:8000/`.
