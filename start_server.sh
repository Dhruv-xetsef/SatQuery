#!/bin/bash
echo "======================================================================="
echo "               🛰️ SATQUERY AI - SERVER LAUNCHER                        "
echo "   Multimodal Remote Sensing Vision-Language Assistant Engine          "
echo "======================================================================="

PYTHON_ENV="/home/xetsef/miniconda3/envs/machine_learning/bin/python"

if [ ! -f "$PYTHON_ENV" ]; then
    echo "Error: Python binary at $PYTHON_ENV not found."
    exit 1
fi

echo "Using Python Environment: $PYTHON_ENV"

# Generate sample dataset if not present
if [ ! -f "dataset/sample_data/single_optical.tif" ]; then
    echo "Generating sample remote sensing GeoTIFF and PNG dataset..."
    $PYTHON_ENV dataset/generate_samples.py
fi

echo "Starting SatQuery AI Backend Web Application on http://localhost:8000..."
$PYTHON_ENV -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
