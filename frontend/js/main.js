// ==========================================================================
// SatQuery AI Frontend Script - Interactive Agentic Earth Observation UI
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    let currentInputMode = "single_optical";
    let selectedFiles = [];
    let currentAnalysisData = null;

    // Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const selectedFilesList = document.getElementById("selected-files-list");
    const presetPillsContainer = document.getElementById("preset-pills");
    const queryInput = document.getElementById("query-input");
    const forceTaskSelect = document.getElementById("force-task-select");
    const btnSubmit = document.getElementById("btn-submit");

    const mainEvidenceImg = document.getElementById("main-evidence-img");
    const scannerOverlay = document.getElementById("scanner-overlay");
    const viewerTitle = document.getElementById("viewer-title");

    const answerText = document.getElementById("answer-text");
    const evidenceJustification = document.getElementById("evidence-justification");
    const trustNumber = document.getElementById("trust-number");
    const trustRating = document.getElementById("trust-rating");
    const trustBarFill = document.getElementById("trust-bar-fill");
    const mConf = document.getElementById("m-conf");
    const mAgree = document.getElementById("m-agree");
    const mSpatial = document.getElementById("m-spatial");
    
    const discoveryList = document.getElementById("discovery-list");
    const copilotSuggestions = document.getElementById("copilot-suggestions");
    const downloadPdfBtn = document.getElementById("download-pdf-btn");

    const graphNodesList = document.getElementById("graph-nodes-list");
    const traceTimeline = document.getElementById("trace-timeline");

    // Initialize Presets & Tools
    fetchPresets();
    fetchTools();

    // Mode Selector
    document.querySelectorAll(".mode-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".mode-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            currentInputMode = tab.dataset.mode;
        });
    });

    // View Tabs (Viewer / Graph / Trace)
    document.querySelectorAll(".view-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".view-tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(`view-${tab.dataset.view}`).classList.add("active");
        });
    });

    // Layer Buttons (Primary / Uncertainty)
    document.querySelectorAll(".layer-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".layer-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            if (currentAnalysisData && currentAnalysisData.visual_artifacts) {
                if (btn.dataset.layer === "uncertainty") {
                    mainEvidenceImg.src = currentAnalysisData.visual_artifacts.uncertainty_map_url;
                } else {
                    mainEvidenceImg.src = currentAnalysisData.visual_artifacts.primary_evidence_url;
                }
            }
        });
    });

    // Drag & Drop
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

    function handleFiles(files) {
        selectedFiles = Array.from(files);
        selectedFilesList.innerHTML = "";
        selectedFiles.forEach((file, idx) => {
            const pill = document.createElement("div");
            pill.className = "file-pill";
            pill.innerHTML = `<span><i class="ph-bold ph-file-tif"></i> ${file.name}</span> <span>${(file.size/1024).toFixed(1)} KB</span>`;
            selectedFilesList.appendChild(pill);
        });
    }

    // Fetch Presets
    async function fetchPresets() {
        try {
            const res = await fetch("/api/presets");
            const data = await res.json();
            presetPillsContainer.innerHTML = "";
            data.queries.forEach(p => {
                const btn = document.createElement("button");
                btn.className = "preset-pill";
                btn.innerHTML = `<strong>${p.title}</strong><br><small>${p.query}</small>`;
                btn.addEventListener("click", () => {
                    queryInput.value = p.query;
                    currentInputMode = p.type;
                    document.querySelectorAll(".mode-tab").forEach(t => {
                        t.classList.toggle("active", t.dataset.mode === p.type);
                    });
                    executeAnalysis(p.query, p.type);
                });
                presetPillsContainer.appendChild(btn);
            });
        } catch (e) {
            console.error("Error fetching presets:", e);
        }
    }

    // Fetch Tools Modal
    async function fetchTools() {
        try {
            const res = await fetch("/api/tools");
            const data = await res.json();
            const body = document.getElementById("tools-modal-body");
            body.innerHTML = "";
            data.tools.forEach(t => {
                const card = document.createElement("div");
                card.className = "card";
                card.style.marginBottom = "10px";
                card.innerHTML = `
                    <h4 style="color: var(--primary);">${t.name}</h4>
                    <p style="font-size: 11px; color: var(--text-dim);">Benchmark: ${t.benchmark} | Scope: ${t.input_scope}</p>
                    <p style="font-size: 12px; margin-top: 5px;">${t.description}</p>
                `;
                body.appendChild(card);
            });
        } catch (e) {
            console.error("Error fetching tools:", e);
        }
    }

    // Submit Query Button
    btnSubmit.addEventListener("click", () => {
        const query = queryInput.value.trim();
        if (!query) {
            alert("Please enter a natural language query.");
            return;
        }
        executeAnalysis(query, currentInputMode);
    });

    // Execute Analysis Function
    async function executeAnalysis(query, samplePresetMode) {
        scannerOverlay.classList.remove("hidden");
        btnSubmit.disabled = true;

        const formData = new FormData();
        formData.append("query", query);
        formData.append("force_task", forceTaskSelect.value);
        if (samplePresetMode) {
            formData.append("sample_preset", samplePresetMode);
        }

        if (selectedFiles.length > 0) {
            selectedFiles.forEach(file => formData.append("files", file));
        }

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error(await res.text());
            }

            const data = await res.json();
            currentAnalysisData = data;
            renderResults(data);
        } catch (e) {
            alert("Analysis Error: " + e.message);
        } finally {
            scannerOverlay.classList.add("hidden");
            btnSubmit.disabled = false;
        }
    }

    // Render Results
    function renderResults(data) {
        // Image Viewer
        if (data.visual_artifacts && data.visual_artifacts.primary_evidence_url) {
            mainEvidenceImg.src = data.visual_artifacts.primary_evidence_url;
        }
        viewerTitle.textContent = `Visual Evidence: ${data.specialist_tool}`;

        // Answer Card
        answerText.textContent = data.text_response;
        evidenceJustification.textContent = data.evidence_explanation;

        // Trust Score
        const trust = data.trust_and_uncertainty;
        trustNumber.textContent = `${trust.reliability_score}%`;
        trustRating.textContent = trust.reliability_rating;
        trustBarFill.style.width = `${trust.reliability_score}%`;
        mConf.textContent = `${trust.model_confidence_pct}%`;
        mAgree.textContent = `${trust.cross_model_agreement_pct}%`;
        mSpatial.textContent = `${trust.spatial_consistency_pct}%`;

        // Discoveries
        discoveryList.innerHTML = "";
        if (data.discoveries && data.discoveries.length > 0) {
            data.discoveries.forEach(d => {
                const item = document.createElement("div");
                item.className = "discovery-item";
                item.innerHTML = `<strong>[${d.category}] ${d.title}</strong><p>${d.description}</p>`;
                discoveryList.appendChild(item);
            });
        }

        // Copilot Suggestions
        copilotSuggestions.innerHTML = "";
        if (data.copilot_suggestions) {
            data.copilot_suggestions.forEach(s => {
                const btn = document.createElement("button");
                btn.className = "copilot-btn";
                btn.textContent = s.query;
                btn.addEventListener("click", () => {
                    queryInput.value = s.query;
                    executeAnalysis(s.query, currentInputMode);
                });
                copilotSuggestions.appendChild(btn);
            });
        }

        // PDF Download Link
        if (data.pdf_report_url) {
            downloadPdfBtn.href = data.pdf_report_url;
            downloadPdfBtn.classList.remove("disabled");
        }

        // Evidence Graph View
        graphNodesList.innerHTML = "";
        if (data.evidence_graph && data.evidence_graph.nodes) {
            data.evidence_graph.nodes.forEach(n => {
                const card = document.createElement("div");
                card.className = "graph-node-card";
                card.innerHTML = `<strong>Node [${n.type.toUpperCase()}]:</strong> ${n.label}`;
                graphNodesList.appendChild(card);
            });
        }

        // Trace Timeline View
        traceTimeline.innerHTML = "";
        if (data.execution_trace) {
            data.execution_trace.forEach(t => {
                const div = document.createElement("div");
                div.className = "trace-step";
                div.innerHTML = `<span class="step-stage">[${t.stage}]</span> ${t.detail}`;
                traceTimeline.appendChild(div);
            });
        }
    }

    // Modal Listeners
    const toolsModal = document.getElementById("tools-modal");
    document.getElementById("btn-tools").addEventListener("click", () => toolsModal.classList.remove("hidden"));
    document.getElementById("close-tools-modal").addEventListener("click", () => toolsModal.classList.add("hidden"));

    const benchmarkModal = document.getElementById("benchmark-modal");
    document.getElementById("btn-benchmark").addEventListener("click", () => benchmarkModal.classList.remove("hidden"));
    document.getElementById("close-benchmark-modal").addEventListener("click", () => benchmarkModal.classList.add("hidden"));

    // Run Benchmark Button
    document.getElementById("run-benchmarks-now").addEventListener("click", async () => {
        const scoresList = document.getElementById("benchmark-scores-list");
        scoresList.innerHTML = "<p style='color: var(--primary);'>Executing Benchmark Evaluation Suite across test sets...</p>";

        try {
            const res = await fetch("/api/benchmark/run");
            const data = await res.json();
            scoresList.innerHTML = `<h4 style="margin-bottom: 10px; color: #34d399;">Overall Benchmark Score: ${data.average_score} / 100.0</h4>`;

            data.results.forEach(r => {
                const div = document.createElement("div");
                div.className = "card";
                div.style.marginBottom = "8px";
                div.innerHTML = `
                    <strong>${r.benchmark}</strong><br>
                    <small>Tool: ${r.specialist_tool} | Score: ${r.score}/100.0 | Latency: ${r.latency_ms} ms | Status: ${r.status}</small>
                `;
                scoresList.appendChild(div);
            });
        } catch (e) {
            scoresList.innerHTML = "<p style='color: #f87171;'>Error running benchmark: " + e.message + "</p>";
        }
    });
});
