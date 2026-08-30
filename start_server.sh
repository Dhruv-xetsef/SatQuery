#!/bin/bash
# SatQuery AI Startup Script
# Portable launcher using active environment python

set -e

PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}

echo "========================================================="
echo "        Starting SatQuery AI Backend Server             "
echo "========================================================="
echo " Listening on http://${HOST}:${PORT}"
echo " Interactive Web GUI: http://${HOST}:${PORT}/"
echo " Presentation Deck: http://${HOST}:${PORT}/presentation/"
echo "========================================================="

# Ensure directories exist
mkdir -p exports dataset/sample_data results

# Launch server
python -m uvicorn backend.app:app --host "${HOST}" --port "${PORT}" --reload
