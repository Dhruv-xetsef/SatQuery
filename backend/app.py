import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.agent import SatQueryAgent
from backend.registry import ToolRegistry
from eval_benchmarks import run_benchmark_evaluation

app = FastAPI(
    title="SatQuery AI Backend",
    description="Interactive Vision-Language Assistant for Multimodal Remote Sensing Analysis",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPORTS_DIR = "exports"
SAMPLE_DIR = "dataset/sample_data"
FRONTEND_DIR = "frontend"
PRESENTATION_DIR = "presentation"

os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(PRESENTATION_DIR, exist_ok=True)

agent = SatQueryAgent(exports_dir=EXPORTS_DIR)
registry = ToolRegistry()

# Mount Static Files
app.mount("/exports", StaticFiles(directory=EXPORTS_DIR), name="exports")
app.mount("/sample_data", StaticFiles(directory=SAMPLE_DIR), name="sample_data")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/presentation", StaticFiles(directory=PRESENTATION_DIR, html=True), name="presentation")

@app.get("/")
def read_root():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "SatQuery AI Backend Server Online"}

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "SatQuery AI Multimodal Agent",
        "version": "2.0.0",
        "device": "CPU / Multithreaded PyTorch"
    }

@app.get("/api/tools")
def list_tools():
    return {"tools": registry.get_available_tools()}

@app.get("/api/presets")
def get_presets():
    return {
        "queries": [
            {
                "id": "q1",
                "title": "Single-Image Scene Description",
                "query": "Describe the land-cover and major objects visible in this image.",
                "type": "single_optical",
                "sample_images": ["/sample_data/single_optical.tif"]
            },
            {
                "id": "q2",
                "title": "Text-Guided Region Grounding",
                "query": "Highlight the water body referred to in the query.",
                "type": "grounding",
                "sample_images": ["/sample_data/single_optical.tif"]
            },
            {
                "id": "q3",
                "title": "Bi-Temporal Change Analysis",
                "query": "What changed between these two dates, and where did the change occur?",
                "type": "bitemporal",
                "sample_images": ["/sample_data/bitemporal_t1.tif", "/sample_data/bitemporal_t2.tif"]
            },
            {
                "id": "q4",
                "title": "Built-up Expansion Trend Query",
                "query": "Has the built-up area increased, decreased, or remained unchanged?",
                "type": "bitemporal",
                "sample_images": ["/sample_data/bitemporal_t1.tif", "/sample_data/bitemporal_t2.tif"]
            },
            {
                "id": "q5",
                "title": "Cross-Modal Optical + SAR Analysis",
                "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
                "type": "crossmodal",
                "sample_images": ["/sample_data/crossmodal_optical.tif", "/sample_data/crossmodal_sar.tif"]
            }
        ]
    }

from typing import List, Optional

@app.post("/api/analyze")
async def analyze(
    query: str = Form(...),
    force_task: str = Form("auto"),
    sample_preset: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[])
):
    image_paths = []
    
    # 1. Process uploaded files if provided
    if files and len(files) > 0 and files[0].filename != "":
        for idx, file in enumerate(files):
            temp_path = os.path.join(EXPORTS_DIR, f"upload_{int(os.times().elapsed)}_{idx}_{file.filename}")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_paths.append(temp_path)

    # 2. Fallback to sample presets if sample_preset specified or no files uploaded
    if len(image_paths) == 0:
        if sample_preset == "crossmodal":
            image_paths = [os.path.join(SAMPLE_DIR, "crossmodal_optical.tif"), os.path.join(SAMPLE_DIR, "crossmodal_sar.tif")]
        elif sample_preset == "bitemporal":
            image_paths = [os.path.join(SAMPLE_DIR, "bitemporal_t1.tif"), os.path.join(SAMPLE_DIR, "bitemporal_t2.tif")]
        elif sample_preset == "single_sar":
            image_paths = [os.path.join(SAMPLE_DIR, "single_sar.tif")]
        else: # Default single optical
            image_paths = [os.path.join(SAMPLE_DIR, "single_optical.tif")]

    try:
        results = agent.process_query(query, image_paths, force_task=force_task)
        return JSONResponse(content=results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmark/run")
def run_benchmark():
    res = run_benchmark_evaluation()
    return res
