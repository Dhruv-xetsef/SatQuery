// ==========================================================================
// SatQuery AI Presentation Script - Interactive 14-Step Flowchart Explorer
// ==========================================================================

const STEPS_DATA = {
    1: {
        title: "Step 01: Perception Layer (Input Intelligence)",
        file: "backend/perception.py & backend/utils_geotiff.py",
        desc: "Inspects satellite image inputs, extracts geospatial CRS metadata (EPSG projection) and GSD resolution, normalizes optical/SAR bands, and auto-detects relationship modes (Single Image, Bi-temporal Pair, or Optical+SAR Co-registered Pair).",
        inputs: ["Raw GeoTIFF / TIFF / PNG / JPEG Satellite Images", "Multispectral & SAR Band Arrays"],
        outputs: ["Normalized Image Tensor Array [H, W, C]", "GSD Resolution & EPSG CRS Metadata", "Relationship Type Classification"]
    },
    2: {
        title: "Step 02: Query Understanding Engine",
        file: "backend/query_engine.py",
        desc: "Parses natural language user queries into structured intent specifications. Performs entity/object extraction (water, built-up, forest), spatial relationship analysis ('near', 'bounding box'), and temporal relationship parsing ('between T1 and T2').",
        inputs: ["User Natural Language Query String", "Number of Provided Satellite Images"],
        outputs: ["Parsed Query Intent (VQA, Grounding, Change, Optical-SAR)", "Extracted Target Entities & Spatial/Temporal Relations", "Required Evidence Types Identification"]
    },
    3: {
        title: "Step 03: Agentic Mission Planner",
        file: "backend/agent_planner.py",
        desc: "Generates dynamic task graphs with sequential action nodes, checks input-task compatibility, selects appropriate specialist models from the registry, and configures execution parameters.",
        inputs: ["Parsed Intent from Query Engine", "Perception Layer Compatibility Metadata"],
        outputs: ["Selected Specialist Tool ID & Parameters", "Dynamic Task-Graph Nodes & Edges Representation", "Execution Plan Strategy Summary"]
    },
    4: {
        title: "Step 04: Specialist Model Registry",
        file: "backend/registry.py & backend/models/*.py",
        desc: "Routes task execution to domain-specific specialist tools: RS-VQA & Scene Description (BigEarthNet VLM), VRSBench Region Grounding, CDVQA Bi-Temporal Change, and ISRO Cross-Modal Optical-SAR Fusion.",
        inputs: ["Target Task Type & Parameter Configuration", "Pre-processed Image Tensors"],
        outputs: ["Text Response & Prediction Distributions", "Confidence Score Metrics", "Raw Overlay Tensors & Masks"]
    },
    5: {
        title: "Step 05: Multi-Modal Evidence Fusion",
        file: "backend/evidence_fusion.py",
        desc: "Aligns text, spatial, temporal, optical, and SAR evidence streams. Resolves cross-modal conflicts (such as optical cloud cover vs SAR microwave penetration) and generates hypotheses.",
        inputs: ["Specialist Model Output", "Query Intent & Perception Profile"],
        outputs: ["Aligned Multi-Modal Evidence List", "Conflict Resolution Report", "Unified Evidence Hypothesis"]
    },
    6: {
        title: "Step 06: Evidence Graph Builder",
        file: "backend/evidence_graph.py",
        desc: "Answers 'Why do we believe this?' by building a Directed Acyclic Graph (DAG): Query → Task → Target Entity → Region → Modality → Hypothesis.",
        inputs: ["Query Plan & Fusion Output", "Specialist Predictions"],
        outputs: ["DAG Nodes & Support Edges Structure", "Provenance Lineage Visualization Data"]
    },
    7: {
        title: "Step 07: Trust & Uncertainty Engine",
        file: "backend/trust_uncertainty.py",
        desc: "Calculates an aggregated Reliability Score (0-100%) across 4 weighted pillars (Model Confidence, Cross-Model Agreement, Spatial Consistency, Temporal Alignment) and renders a spatial uncertainty heatmap.",
        inputs: ["Tool Predictions & Perception Metadata", "Visual Overlay Tensors"],
        outputs: ["0-100% Reliability Score & Rating", "Magma Colormap Spatial Uncertainty Map", "Conflict Flags & Warning Alerts"]
    },
    8: {
        title: "Step 08: Discovery Engine ('Beyond Your Query')",
        file: "backend/discovery_engine.py",
        desc: "Runs an autonomous secondary background scan across imagery to spot unqueried anomalies, land-cover shifts, or sub-surface SAR water features that the user did not explicitly ask about.",
        inputs: ["Primary Specialist Outputs", "Multi-spectral & SAR Feature Layers"],
        outputs: ["'Beyond Your Query' Discovery Items", "Risk Level Classifications (INFO, MODERATE, HIGH)"]
    },
    9: {
        title: "Step 09: Investigation Copilot",
        file: "backend/copilot.py",
        desc: "Formulates 4-5 contextual 'Suggested Next Questions' allowing analysts to perform interactive multi-turn deep-dives and spatial drills.",
        inputs: ["Final Answer & Task Type", "Discovery Findings"],
        outputs: ["List of Actionable Next Question Pills for Interactive UI"]
    },
    10: {
        title: "Step 10: Visual Evidence Generator",
        file: "backend/evidence_generator.py",
        desc: "Generates georeferenced visual evidence images (Bounding boxes, Segmentation masks, JET change heatmaps, False-Color Optical/SAR overlays) and saves export artifacts.",
        inputs: ["Tool Overlay Tensors & Uncertainty Heatmaps"],
        outputs: ["PNG Evidence Artifact Files in /exports", "Static Web Access URLs (/exports/*.png)"]
    },
    11: {
        title: "Step 11: Answer Generator",
        file: "backend/answer_generator.py",
        desc: "Synthesizes final natural language answer, evidence-linked explanation, trust metrics summary, spatial stats, and discovery highlights into a unified response payload.",
        inputs: ["Tool Text Output & Evidence Fusion Report", "Trust & Discovery Data"],
        outputs: ["Evidence-Grounded Final Response Payload"]
    },
    12: {
        title: "Step 12: Observable Audit Trace & PDF Report",
        file: "backend/audit_trace.py",
        desc: "Constructs an observable execution trace log and compiles downloadable ReportLab PDF audit reports containing query details, execution steps, taxonomy scores, and confidence metrics.",
        inputs: ["Full Agent Pipeline Execution History"],
        outputs: ["Auditable Execution Trace Log", "Downloadable PDF Audit Report (/exports/*.pdf)"]
    },
    13: {
        title: "Step 13: SatQuery Web GUI",
        file: "frontend/index.html, css/styles.css, js/main.js",
        desc: "Delivers an interactive, dark glassmorphic web interface featuring Earth Viewer, layer toggles (Primary vs Uncertainty), evidence graph visualizer, trust score gauge, and copilot buttons.",
        inputs: ["User Interactions & File Uploads"],
        outputs: ["Interactive Web Dashboard & Real-Time Visualization"]
    },
    14: {
        title: "Step 14: Benchmark Evaluation Suite",
        file: "eval_benchmarks.py",
        desc: "Evaluates model accuracy, task alignment, and latency across BigEarthNet, VRSBench, RSVQA, CDVQA, and ISRO/SAC test splits, yielding an overall 93.16/100 benchmark score.",
        inputs: ["Test Dataset Sample Queries"],
        outputs: ["Normalized Accuracy & Latency Metrics", "Benchmark Evaluation Results Matrix"]
    }
};

document.addEventListener("DOMContentLoaded", () => {
    const flowCards = document.querySelectorAll(".flow-card");
    const stepTitle = document.getElementById("step-title");
    const stepFile = document.getElementById("step-file");
    const stepDesc = document.getElementById("step-desc");
    const stepInputs = document.getElementById("step-inputs");
    const stepOutputs = document.getElementById("step-outputs");

    flowCards.forEach(card => {
        card.addEventListener("click", () => {
            flowCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");

            const stepNum = parseInt(card.dataset.step);
            const data = STEPS_DATA[stepNum];

            if (data) {
                stepTitle.textContent = data.title;
                stepFile.textContent = data.file;
                stepDesc.textContent = data.desc;

                stepInputs.innerHTML = data.inputs.map(i => `<li>${i}</li>`).join("");
                stepOutputs.innerHTML = data.outputs.map(o => `<li>${o}</li>`).join("");
            }
        });
    });
});
