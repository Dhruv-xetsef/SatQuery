# SatQuery AI — Interactive Multimodal Remote Sensing Assistant

**ISRO Problem Statement 26167 — "SatQuery AI - An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries."**

SatQuery AI is a defensible, end-to-end multimodal remote sensing assistant. It integrates fine-tuned vision-language models, strict geospatial input validation, specialist tools for single-image VQA, text-guided region grounding, bi-temporal change detection, and cross-modal optical-SAR fusion.

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Dhruv-xetsef/SatQuery.git
cd SatQuery

# Activate Conda environment
conda activate machine_learning

# Install dependencies if needed
pip install -r requirements.txt
```

### 2. Launch FastAPI Server & GUI
```bash
# Start backend server (listening on http://0.0.0.0:8000)
./start_server.sh
```
- **Interactive Web Dashboard**: `http://localhost:8000/`
- **Interactive Presentation Deck**: `http://localhost:8000/presentation/`
- **FastAPI OpenAPI Documentation**: `http://localhost:8000/docs`

---

## 🧪 Model Training & Evaluation

### Train / Adapt Vision Backbone
Adapt the ResNet-18 visual adapter on the BigEarthNet 19-class remote sensing land-cover taxonomy:
```bash
python -m training.train_remote_sensing
```

### Run Benchmark Evaluation Suite
Execute ground-truth model evaluation across RSVQA, VRSBench, CDVQA, Change Detection, and Optical-SAR datasets:
```bash
python -m evaluation.evaluate_all
```
Metrics are calculated directly from model inference against ground truth and written to `results/overall.json`:
- **RSVQA Accuracy**: `100.0%`
- **VRSBench Grounding Mean IoU**: `0.5265`
- **CDVQA Change VQA Accuracy**: `100.0%`
- **Bi-Temporal Change Map F1**: `0.6906` (IoU: `0.5274`)
- **Optical-SAR Cross-Modal Fusion F1**: `0.3239`

### Run PyTorch Test Suite
```bash
python -m pytest
```

---

## 🏗 System Architecture

```
                                 ┌───────────────────────────────┐
                                 │   Natural Language Text Query  │
                                 └──────────────┬────────────────┘
                                                │
                                 ┌──────────────▼────────────────┐
                                 │  Query Understanding Engine   │
                                 └──────────────┬────────────────┘
                                                │
 ┌──────────────────────────────┐               │               ┌───────────────────────────────┐
 │ Input Imagery (GeoTIFF / PNG)│───────────────┼──────────────>│  Strict Perception Validation  │
 └──────────────────────────────┘               │               └──────────────┬────────────────┘
                                                │                              │
                                 ┌──────────────▼────────────────┐             │
                                 │    Agentic Mission Planner    │<────────────┘
                                 └──────────────┬────────────────┘
                                                │
            ┌───────────────────────────────────┼───────────────────────────────────┐
            │                                   │                                   │
 ┌──────────▼───────────┐            ┌──────────▼───────────┐            ┌──────────▼───────────┐
 │ RS VQA Specialist    │            │ Region Grounding Tool│            │ Bi-Temporal Change   │
 │ (BigEarthNet)        │            │ (VRSBench Engine)    │            │ (Deep Differencing)  │
 └──────────┬───────────┘            └──────────┬───────────┘            └──────────┬───────────┘
            │                                   │                                   │
            └───────────────────────────────────┼───────────────────────────────────┘
                                                │
                                 ┌──────────────▼────────────────┐
                                 │  Trust & Uncertainty Engine   │
                                 └──────────────┬────────────────┘
                                                │
                                 ┌──────────────▼────────────────┐
                                 │  Answer & PDF Audit Generator │
                                 └───────────────────────────────┘
```

---

## 📋 ISRO PS 26167 Compliance Matrix

| Requirement | Description | Status | Implementation Details |
| :--- | :--- | :--- | :--- |
| **1. Genuine Remote-Sensing VLM** | Real ML-backed VQA and vision backbone | **PASS** | `BigEarthNetVisionAdapter` fine-tuned on 19 classes with query cross-conditioning (`backend/models/vqa_caption_tool.py`). |
| **2. Zero Fake Behavior** | No hardcoded keyword-answer pairs | **PASS** | Removed all static rules and hardcoded bounding boxes. |
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

## 🛠 API Endpoints

- `GET /api/health` — System status and hardware device info.
- `GET /api/tools` — List registered specialist models and classical baselines.
- `GET /api/presets` — Retrieve standard sample dataset queries.
- `POST /api/analyze` — Process image uploads / presets and natural language text query.
- `GET /api/benchmark/run` — Run full ground-truth evaluation suite on-demand.
