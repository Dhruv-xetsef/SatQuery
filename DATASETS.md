# SatQuery AI — Remote Sensing Benchmark Datasets

This document details the ground-truth remote sensing benchmark datasets integrated and supported by SatQuery AI.

## 1. BigEarthNet-S2 (Land Cover Taxonomy)
- **Description**: Large-scale Sentinel-2 multispectral dataset featuring 19-class CORINE land-cover taxonomy.
- **Role in SatQuery**: Fine-tuning vision backbones (`backend/models/rs_adapter.py`) to learn remote sensing visual representations.
- **Classes**: Urban fabric, Industrial/Commercial, Mines, Vegetated areas, Arable land, Permanent crops, Pastures, Forests, Wetlands, Water bodies, SAR structural features.

## 2. RSVQA (Remote Sensing Vision-Language Question Answering)
- **Description**: Benchmark dataset for visual question answering over satellite imagery.
- **Role in SatQuery**: Evaluation suite for single-image natural language scene VQA (`evaluation/evaluate_rsvqa.py`).

## 3. VRSBench (Visual Region Grounding Benchmark)
- **Description**: Remote sensing dataset for text-guided target region localization and bounding box delineation.
- **Role in SatQuery**: Grounding evaluation suite measuring Intersection over Union (IoU) (`evaluation/evaluate_vrsbench.py`).

## 4. CDVQA (Change Detection Vision Question Answering)
- **Description**: Bi-temporal satellite image pair dataset with natural language question-answer pairs regarding land-cover conversion.
- **Role in SatQuery**: Bi-temporal change VQA benchmark evaluation (`evaluation/evaluate_cdvqa.py`).

## 5. ISRO Cartosat / RISAT Multimodal Sample Collection
- **Description**: High-resolution optical (Cartosat) and C-band SAR (RISAT) satellite image pairs.
- **Location**: `dataset/sample_data/`
  - `single_optical.tif`: High-resolution optical tile (512x512)
  - `bitemporal_t1.tif`: Pre-expansion observation (Observation T1)
  - `bitemporal_t2.tif`: Post-expansion observation (Observation T2)
  - `crossmodal_optical.tif`: Optical tile with cloud cover
  - `crossmodal_sar.tif`: C-band SAR backscatter tile
