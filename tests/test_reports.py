import os
from backend.audit_trace import AuditTraceGenerator

def test_pdf_report_generation(tmp_path):
    tracer = AuditTraceGenerator()
    trace_data = tracer.build_trace(
        query="Describe scene",
        perception_plan={"status": "PASSED", "image_count": 1, "relationship_type": "Single Optical"},
        query_plan={"task_type": "vqa_caption", "intent": "SCENE_VQA_CAPTIONING", "extracted_entities": ["forest"]},
        mission_plan={"selected_tool_id": "vqa_caption_tool", "tool_name": "RS-VQA Specialist", "parameters": {}, "task_graph_nodes": []},
        tool_output={"specialist_tool": "RS-VQA Specialist", "confidence": 0.85, "text_response": "Forest region"},
        fusion_output={"status": "FUSED", "evidences": [], "conflict_resolution": "No conflicts"},
        trust_output={"reliability_score": 88.5, "reliability_rating": "HIGH TRUST", "confidence_label": "calibrated model confidence", "cross_model_agreement_pct": 90.0, "conflict_flags": []},
        discovery_output={"discoveries": [], "summary": "No additional discoveries"},
        answer_output={"final_answer": "Forest region"},
        execution_time_ms=150.0
    )
    pdf_out = os.path.join(tmp_path, "test_report.pdf")
    tracer.generate_pdf_report(trace_data, pdf_out)
    assert os.path.exists(pdf_out)
    assert os.path.getsize(pdf_out) > 0
