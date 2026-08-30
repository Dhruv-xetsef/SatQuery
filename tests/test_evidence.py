from backend.evidence_fusion import MultiModalEvidenceFusion
from backend.evidence_graph import EvidenceGraphBuilder

def test_evidence_fusion():
    fusion = MultiModalEvidenceFusion()
    q_plan = {"task_type": "vqa_caption", "extracted_entities": ["forest"]}
    t_out = {"confidence": 0.85, "text_response": "Forest region", "specialist_tool": "VQA Specialist"}
    p_plan = {"status": "PASSED", "image_count": 1, "relationship_type": "Single Optical", "sensor_types": ["Optical"]}
    res = fusion.fuse_evidence(q_plan, t_out, p_plan)
    assert "evidences" in res
    assert "conflict_resolution" in res

def test_evidence_graph():
    graph_builder = EvidenceGraphBuilder()
    q_plan = {"query": "Describe scene", "task_type": "vqa_caption", "extracted_entities": ["forest"]}
    t_out = {"confidence": 0.85, "specialist_tool": "VQA Specialist"}
    f_out = {
        "hypothesis": "Evidence synthesis supports response",
        "evidences": [{"modality": "Text", "source": "VQA Specialist", "claim": "Forest"}]
    }
    res = graph_builder.build_graph(q_plan, t_out, f_out)
    assert "nodes" in res
    assert "edges" in res
